# Round 2 Track 2: bounded native-scene RT smoke

## Declared design and acceptance checks (before rerun)

This is a development-only, two-case pipeline smoke on the existing v1
development cases `01-munich-natural` and `39-maastricht-natural`. It does not
generate benchmark data and cannot support optimizer, rate, or generalization
claims. Use only the already hash-verified Phase 0b Munich and Maastricht
native LoD2 inputs; do not open or fetch any held-out scene.

For non-planar native faces, preserve the original 3D surface by emitting
triangles through the source 3D vertices. Use a local face coordinate origin
and constrained-triangulate the dominant-plane projection. If the projected
native polygon is invalid, repair its 2D topology with `make_valid`, constrained
triangulate the repaired polygon, and lift new intersection vertices by linear
interpolation along the closest projected source boundary segment. This is an
explicit RF geometry approximation: warped faces become piecewise-planar
triangles; a self-intersecting projected face uses the planar valid geometry
returned by GEOS. Do not silently drop a nondegenerate face. Report the number
of approximated faces and maximum source-vertex distance from its fitted plane.
The Phase0b builder expects projected footprints relative to the scene origin
(it adds the origin before clipping) while it expects 3D triangles in source
coordinates (it subtracts the origin); the RT wrapper preserves this mixed
interface explicitly.

Geometry preflight passes only if (a) all emitted vertices are finite, (b) no
emitted triangle has 3D area at or below `1e-12 m^2`, (c) each face's
triangulation references every source ring vertex, and (d) constrained
triangulation covers its valid projected polygon within relative area and
symmetric-difference tolerances `1e-8`. Report invalid-topology repair counts
separately. Any source face that cannot meet these checks blocks tracing.

The GPU smoke uses a vertical 8x8, 8x16, and 16x16 probe lattice, 28 mm / 30
mm pitch, 5 cm offset toward the BS, 5.53 GHz, 10,000 samples per source,
maximum depth 2, LOS/specular/diffraction enabled (isolated interaction checks
at 8x8), refraction and synthetic array disabled, seed 7. Include the actual
frame-0 served users. Use 3060-3 GPU 1, with one solver call at a time. Bind
the runner away from logical CPUs 5 and 11 (`taskset -c 0-4,6-10`). The full
runner lifetime must hold `/var/tmp/atomwise-bench-round2-highmem.lock`
with nonblocking `flock`. Before every PathSolver call, require at least 2 GiB
free on one GPU. Raw scenes and path arrays stay on node3.

Acceptance requires a complete marker written after clean process exit, both
development cases and all three lattice sizes, finite coefficients and
delays, receiver-specific delays and all four angle arrays, verified receiver
IDs, source/mesh/path hashes, path classes, elapsed time, peak RSS and GPU
memory. The output is a compact summary only; no raw arrays are copied back.

## Execution record

Pending. Prior diagnostics remain preserved under
`/var/tmp/bench-tracks-rt-20261108/diagnostic_attempt1` and `attempt2`;
neither is accepted as round-2 evidence.
