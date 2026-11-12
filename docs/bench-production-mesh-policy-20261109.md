# CoupledRIS-Bench Production Mesh-Repair and Geometry-Preflight Policy (2026-11-09)

This document declares the unified LoD2 mesh-repair policy and fail-closed geometry preflight contract applied to all production and pilot maps (`munich`, `rotterdam`, and `maastricht`) before any GPU ray-tracing solve in `scripts/run_bench_production_trace.py`.

## 1. Unified Coordinate and Orientation Contract

1. **Single origin subtraction (`one_origin_shift_only`)**:
   - Every map defines a single projected-CRS scene origin $\mathbf{o} = (o_x, o_y, o_z)$ in its native metric coordinate reference system (`EPSG:25832` for Munich; `EPSG:7415` for Rotterdam and Maastricht).
   - During face triangulation, vertices of each semantic face are temporarily shifted by the face's first outer-ring vertex $\mathbf{v}_0$ ($\tilde{\mathbf{v}} = \mathbf{v}_{\text{world}} - \mathbf{v}_0$) to prevent floating-point cancellation in 2D projection and constrained Delaunay triangulation, then restored to world CRS ($\mathbf{v}_{\text{world}} = \tilde{\mathbf{v}} + \mathbf{v}_0$).
   - The shared scene writer (`write_scene` / `write_ply`) subtracts the scene origin $\mathbf{o}$ **exactly once** ($\mathbf{v}_{\text{local}} = \mathbf{v}_{\text{world}} - \mathbf{o}$) when emitting PLY meshes and transforming BS, RIS, and UE coordinates.

2. **Outward Newell normal orientation (`outward_newell_normal_orientation`)**:
   - For every semantic face with outer ring $\mathbf{r}_0 = (\mathbf{p}_0, \dots, \mathbf{p}_{m-1})$, the outward 3D Newell normal vector is computed as:
     $$\mathbf{n}_{\text{face}} = \sum_{i=0}^{m-1} \begin{bmatrix} (y_i - y_{i+1})(z_i + z_{i+1}) \\ (z_i - z_{i+1})(x_i + x_{i+1}) \\ (x_i - x_{i+1})(y_i + y_{i+1}) \end{bmatrix}, \qquad \hat{\mathbf{n}}_{\text{face}} = \frac{\mathbf{n}_{\text{face}}}{\|\mathbf{n}_{\text{face}}\|_2}.$$
   - Every emitted triangle $(\mathbf{t}_0, \mathbf{t}_1, \mathbf{t}_2)$ is oriented so that its right-hand cross-product normal $\mathbf{n}_\triangle = (\mathbf{t}_1 - \mathbf{t}_0) \times (\mathbf{t}_2 - \mathbf{t}_0)$ satisfies $\mathbf{n}_\triangle \cdot \hat{\mathbf{n}}_{\text{face}} > 0$. If $\mathbf{n}_\triangle \cdot \hat{\mathbf{n}}_{\text{face}} < 0$, vertices $\mathbf{t}_1$ and $\mathbf{t}_2$ are swapped before emission.

3. **Declared RF scene crop**:
   - Only buildings whose `GroundSurface` footprint intersects the declared RF scene bounding box (for Munich, the $50\text{ m}$ anchor box buffered by $35\text{ m}$; for Rotterdam and Maastricht, the $180\text{ m}$ anchor box clipped to the frozen v1.1 $440\text{ m} \times 440\text{ m}$ map bounds) are included in the RF scene.
   - Any individual semantic face whose 2D axis-aligned bounding box is wholly disjoint from the declared RF scene box is cropped prior to triangulation and counted per semantic class in `mesh_audit`.

## 2. Deterministic Mesh-Repair and Face-Drop Policy

1. **Zero-area face and triangle removal**:
   - Consecutive duplicate ring vertices (within $10^{-9}\text{ m}$) and closing duplicate vertices are stripped.
   - Any semantic face whose outer ring has fewer than 3 distinct vertices, whose vertices are collinear, or whose net 3D Newell area
     $$A_{\text{src}} = \frac{1}{2}\|\mathbf{n}_{\text{outer}}\|_2 - \sum_{h \in \text{holes}} \frac{1}{2}\|\mathbf{n}_h\|_2$$
     is $\le 10^{-8}\text{ m}^2$ is dropped as a zero-area degenerate face (`dropped_zero_area_faces`).
   - Any zero-area triangle produced during triangulation with 3D area $\le 10^{-10}\text{ m}^2$ is omitted.

2. **Deterministic repair of self-intersecting or non-planar LoD2 faces**:
   - Each face is projected onto the 2D coordinate plane orthogonal to the dominant component of $\hat{\mathbf{n}}_{\text{face}}$ ($\arg\max_{k \in \{0,1,2\}} |\hat{n}_{\text{face},k}|$).
   - If the projected 2D polygon (with interior hole rings) is topologically invalid (`not polygon.is_valid` or `polygon.is_empty`) or if the face vertices deviate from the Newell plane by more than $0.01\text{ m}$ ($\max_{\mathbf{p}} |(\mathbf{p} - \mathbf{p}_0) \cdot \hat{\mathbf{n}}_{\text{face}}| > 0.01\text{ m}$), the face is marked as repaired (`repaired_faces`) and processed deterministically:
     - Invalid 2D projections are repaired using `shapely.make_valid` and decomposed into positive-area `Polygon` components.
     - Each 2D polygon component is triangulated with `shapely.constrained_delaunay_triangles`, retaining triangles whose centroids lie inside the polygon domain.
     - Each 2D triangle vertex is lifted back to 3D by snapping/interpolating along the nearest source 3D ring edge (within $10^{-3}\text{ m}$ in projection) or, for strictly interior Steiner points, projecting onto the mean Newell plane.
   - **1.0% 3D area acceptance gate**: A repaired or non-planar face is **accepted only if** its total emitted 3D triangle area $A_{\text{emit}} = \sum_{\triangle} \frac{1}{2}\|\mathbf{n}_\triangle\|_2$ agrees with the source face net 3D Newell area $A_{\text{src}}$ within **$1.0\%$**:
     $$\frac{|A_{\text{emit}} - A_{\text{src}}|}{A_{\text{src}}} \le 0.01.$$
     If this relative area discrepancy exceeds $1.0\%$ (for example, a severely warped non-planar roof polygon whose Newell vector cancels across opposing slopes), the face is rejected and dropped (`dropped_positive_area_faces`).

3. **Per-building, per-class, and per-map provenance and 0.5% stop rule**:
   - Every clean face, repaired face, zero-area dropped face, positive-area dropped face, and cropped outside-scene face is counted per building (`per_building`), per semantic surface class (`per_class`), and per map (`map_totals`), and recorded under `scene.mesh_audit` in `geometry_preflight.json`.
   - **Hard stop rule (`max_dropped_area_share_per_class <= 0.005`)**: If the total source 3D area of dropped positive-area faces in any semantic class exceeds **$0.5\%$** ($0.005$ fraction) of that class's total source 3D area on any map, geometry preflight fails closed before any GPU ray tracing is launched.

## 3. Fail-Closed Map Geometry Preflight Contract

Before `--reuse-scenes` ray tracing is permitted, `--build-only` must pass all four hard preflight checks on every map in the registered manifest (`status == "pass"`):

1. **World $z$-bounds check (`z_bounds_within_terrain_plus_height`)**:
   - Let $[z_{\text{terrain,min}}, z_{\text{terrain,max}}]$ be the finite elevation range of the frozen v1.1 terrain raster (`outputs/previews/bench-users-v1.1/geometry/{map_id}-terrain.npz`) and $H_{\max}$ be the maximum building height (`max(v1.1 building height_m, in-scene LoD2 building height)`).
   - The emitted RF mesh world $z$ range $[z_{\text{mesh,min}}, z_{\text{mesh,max}}]$ must lie within $[z_{\text{terrain,min}}, z_{\text{terrain,max}} + H_{\max}]$ (with $0.5\text{ m}$ numerical tolerance).

2. **Building base $z$ check (`building_base_matches_reference`)**:
   - Every in-scene building's `GroundSurface` base elevation $z_{\text{base}}$ must lie within the frozen v1.1 terrain elevation band $[z_{\text{terrain,min}} - 0.5\text{ m}, z_{\text{terrain,max}} + 0.5\text{ m}]$ and match its frozen v1.1 building `ground_m` (or the native `b3_h_maaiveld` / footprint-ground-fill elevation for boundary parts) within the declared v1.1 ground-fill guard tolerance of **$2.5\text{ m}$** (observed maximum discrepancy against matched v1.1 `ground_m` across Munich, Rotterdam, and Maastricht is $0.000000\text{ m}$).

3. **Outward-normal share check (`outward_normal_share == 1.0`)**:
   - Across all emitted triangles on every map and semantic class, the fraction of triangles satisfying $\mathbf{n}_\triangle \cdot \hat{\mathbf{n}}_{\text{face}} > 0$ must be exactly **$100.0\%$** (`1.0`).

4. **Per-class emitted vs. source area conservation (`emitted_vs_kept_source_area_within_0_1_pct`)**:
   - After declared face drops, the total emitted 3D triangle area per semantic class must match the kept source 3D Newell area within **$0.1\%$** ($\le 0.001$ relative difference) on every map.
