# Production RT trace pilot: implementation record

## Scope and guardrails

This module is limited to the lead-approved development preview entries
`01-munich-natural` and `15-rotterdam-natural`, frame 0, and 64/128/256-element
arrays. It verifies the v1.1 freeze and the exact case-manifest SHA-256 before
building scenes or reading cases. It rejects any other case, frame, or map.
The pilot preregistration is in `bench-production-pilot-20261109.md`; no full
production run or held-out map is included here.

The Sionna run is fixed at 5.53 GHz, Sionna RT 2.0.1, explicit `VH` antennas,
maximum interaction depth 2, 10,000 samples per source, seed 7, LOS/specular/
diffraction enabled, and diffuse reflection/refraction disabled. GPU0 is
exclusive under a task-local flock; process CPU affinity excludes cores 5 and
11. A 100 ms `nvidia-smi` sampler records GPU0 peak allocated/used memory
observed during each solve, with before/after snapshots as context.

## Geometry and link conventions

The runner imports the audited Munich mesh builder and shared mesh writer from
`run_bench_rt_preview_smoke.py`. Rotterdam uses the same native PLY/material
writer and the pinned 2025-09-03 3DBAG CityJSON source tiles listed in the
approved recovered source manifest. It invokes the audited CityJSON vertex
decoder and semantic-face triangulator from `build_bench_phase0b_scene.py`;
tiles are never substituted. Rotterdam coordinates are EPSG:7415 (RD New +
NAP); Munich is EPSG:25832. Within each case, all ray coordinates use one local
metric frame formed by subtracting the recorded projected XYZ origin.

Array ordering is explicit. Transmitters are four BS ports followed by the
served UEs. Receivers are all RIS centres followed by the same external ports.
The BS ports are vertically polarized short-wire locations spaced by half a
wavelength in the horizontal direction perpendicular to BS–RIS. RIS points
use an 8×8, 8×16, or 16×16 lattice aligned to the declared panel normal and
include 5 cm face clearance. Self endpoint links are omitted, not represented
as zero-power channels. UE→RIS paths preserve that direction in the index arrays
and supply the reciprocal RIS→UE channel under the passive reciprocal-medium
assumption.

### Rotterdam geometry amendment, declared 2026-11-09 08:40 WIB before retry

The failed Rotterdam preflight found `NL.IMBAG.Pand.0599100000046867-0`,
geometry 2, `WallSurface`, face 7, in pinned tile `9-312-516.city.json.gz`.
Its ring points are `(92387.848875, 436128.833, 11.190000885)`, the same point
again, and `(92386.167875, 436126.904, 11.229000885)` metres. It has two unique
3-D vertices; after translating to a ring-local origin, its Newell norm and
area are exactly zero. The 1.681 m by 1.929 m XY extent is a line segment, not
a polygon area.

Before retry, the Rotterdam-only adapter will normalize repeated closing and
consecutive duplicate vertices for the area check, translate each face to its
first vertex before invoking the pinned semantic-face triangulator, and restore
the exact translation after triangulation. It may omit a face only when the
outer ring has fewer than three unique 3-D points or every local cross product
is exactly zero in float64 (zero area; no positive-area epsilon is introduced).
Every omitted face is counted and identified by object, geometry, face index,
and semantic type. Any face with positive local area must pass the existing
triangulation checks or the build fails; there is no fallback and no silently
discarded nonzero face. This changes no source geometry, terrain, bounds,
placement, material, array, or channel threshold. Munich's already-built scene
will be reused only after checking the pinned GML and all scene/mesh hashes.

### Rotterdam scene-crop and origin amendment, declared 2026-11-09 08:53 WIB before retry

The next preflight found `NL.IMBAG.Pand.0599100000423762-4`, geometry 2,
`GroundSurface`, face 0, in the same pinned tile. Its 15-point outer ring has
14 unique coordinates and is planar at z `-1.443999114990234`; its local XY
projection self-intersects at `(92051.699875, 436516.031)`. GEOS `make_valid`
returns two polygons with total area `70.91971349993425 m²`, equal to the
original projected area `70.91971349993423 m²` to floating-point precision.
However, the trace scene is bounded to
`[92211.4889167, 436715.3847278, 92670.675875, 437112.7044582]` EPSG:28992;
this face's box is `[92047.964875, 436498.69, 92057.798875, 436516.237]`,
fully outside the declared RF scene. Its parent building box overlaps the
scene, so filtering whole buildings alone does not exclude the irrelevant
face.

Before retry, the Rotterdam adapter will crop a face only when its complete
native XY bounding box is disjoint from that declared scene box, and will
count/report each cropped face by object and semantic type. Every face that
intersects the RF scene remains subject to the pinned triangulation checks;
invalid positive-area in-scene surfaces still fail closed. The crop has no
epsilon and changes no native vertices or scene bounds. Also, the audited
surface builder returns triangles with the scene origin already subtracted;
the writer must receive those triangles with the origin restored exactly once,
and footprint polygons must remain in scene-local XY for clipping. This is a
coordinate-frame correction only.

The approved preview geometry pipeline uses GEOS `make_valid` for footprint
polygons, while its RF-face helper fails on invalid face projections; its
Netherlands model builder catches those failures and skips the affected faces.
That lossy behavior is not copied into the RF scene. The inherited audited
`polygon_surfaces` routine first selects building parts with a retained,
scene-intersecting `GroundSurface`, then emits all faces retained for those
parts. Consequently, it may also omit a roof overhang from a part whose ground
footprint does not intersect the scene bounds. This inherited footprint filter
will be reported as a geometry completeness limitation; the production trace
does not claim a watertight/full-city mesh.

## Export format

Each raw file is written at:

```text
<map>/n<64|128|256>/<case-id>/frame-0000.npz
```

The archive contains local `tx_xyz`, `rx_xyz`, RIS centre coordinates,
`link_tx`, `link_rx`, path index, `path_delay_s`, complex `path_jones[P,2,2]`,
the four departure/arrival angle arrays, `path_type`, exact interaction enum
sequences and names, and explicit LOS/reflection/diffraction flags. `Paths.a`
and `Paths.tau` are read in the native Sionna rank-five ordering
`[rx, rx-pol, tx, tx-pol, path]`; the V/H Jones matrix is retained without CIR
reordering. Path type is 1=LOS, 2=reflection only, 3=diffraction only,
4=mixed reflection/diffraction. Unsupported enum values fail the run rather
than being silently labeled as mixed. The installed Sionna 2.0.1
`InteractionType` is a plain class: `NONE=0` is padding, `SPECULAR=1`,
`DIFFUSE=2`, `REFRACTION=4`, and `DIFFRACTION=8`. With synthetic arrays off,
native angles are indexed at `[rx,0,tx,0,path]`; no guessed axis layout is
accepted. Each `Paths` array is materialized once per solve before export.

For LOS, geometric path length is recomputed from the double-precision
endpoints. For reflected/diffracted paths, the current exporter uses `c*tau`
and records the source precision limitation; it does not claim those lengths
were reconstructed from vertices. NPZ stores all source metadata, case,
frequency, port order, coordinate origin, scene hashes, solver settings, path
type mapping, and reciprocity note. `trace_manifest.json` stores file hashes,
bytes, elapsed solve times, and sampled GPU peaks.

## Validation and launch status

Focused local checks currently cover: RIS lattice placement and normal,
5 cm clearance, array sizes, four BS port spacing, Jones shape/order, Sionna
path-axis indexing, and equality of the pinned Rotterdam tile hashes with the
approved source manifest. They pass. The pilot manifest SHA-256 is
`998083842c02870e45724313151c15c77910754372baba6669776433206eb865` and the
case/freeze verification succeeds.

The first remote start stopped in geometry preflight before GPU work because
the Munich shared builder expects its pinned GML under the staging `raw/`
directory. That attempt is preserved as
`trace/FAILED.munich-input-missing-preflight` with log
`trace/trace-job.preflight.log`. The runner now downloads the Munich GML through
the shared builder's size/SHA-256 checked downloader before parsing it.
Rotterdam then exposed the exactly zero-area face documented above; this
second failure is preserved as `trace/FAILED.rotterdam-degenerate-face` with
log `trace/trace-job.rotterdam-degenerate-face.log`. The next preflight then
stopped on the self-intersecting ground ring outside the scene; this is
preserved as `trace/FAILED.rotterdam-out-of-scene-ground-face` with log
`trace/trace-job.rotterdam-out-of-scene-ground-face.log`. None of these
attempts reached GPU tracing.

The amended run had been active in tmux session `bench-prod-trace-20261109`
on 3060-3, started 2026-11-09 08:55 WIB. The complete source lock bundle and the
two-case manifest were staged under
`<lab-home>/projects/bench-production-pilot-20261109/trace-src`; source
freeze verification and the pinned `.venv` runtime check passed before launch.
Munich geometry was reused without rewrite after validating the pinned GML
SHA-256 and recording scene hash
`39ef1aeaa9b00cd8d0b601b65ec220b5ddde02021755959bee05d3f98d051c02` plus all
three mesh hashes in `trace-work/munich_reuse_attestation.json`.
The separate geometry workspace is
`<lab-home>/projects/bench-production-pilot-20261109/trace-work`, and the
approved channel handoff path is
`<lab-home>/projects/bench-production-pilot-20261109/trace`. At last
check, `trace/RUNNING` existed and the job was executing `--build-only`; the
partial `trace-work/scene_manifest.json` contains the hash-checked Munich
scene. GPU0 was idle and no trace NPZ files had yet been produced. The exporter
was also corrected to use actual plain-class Sionna enum values, ignore
`NONE=0` padding when determining path type, index native angles directly at
`[rx,0,tx,0,path]`, materialize path arrays once per solve, and record solver
and export time separately.

Use this one-shot state check; do not relaunch while `RUNNING` or tmux session
exists:

```sh
ssh <3060-3> \
  'base=<lab-home>/projects/bench-production-pilot-20261109; ls -l "$base/trace"/{RUNNING,DONE,FAILED,geometry_preflight.json,trace_manifest.json} 2>/dev/null; tail -40 "$base/trace/trace-job.log" 2>/dev/null; tmux list-sessions 2>/dev/null'
```

The wrapper executes the native scene geometry preflight first, then reuses
the hash-checked scene and performs six sequential array traces on GPU0. It
writes `DONE` only after all six NPZ files and `trace_manifest.json` are
finalized; failures write `FAILED`. The measured pilot summary is pending.

### Current terminal state and Munich-only diagnostic, 2026-11-09

The two-map preflight subsequently failed closed on the pinned Rotterdam tile
`9-312-520.city.json.gz`, object
`NL.IMBAG.Pand.0599100000640552-0`, LoD2.2 geometry 2, shell 0, `RoofSurface`
face 56. The exact outer ring has 16 indices and 15 unique XYZ coordinates
(the repeated coordinate is not the closing duplicate). Its EPSG:7415 XYZ
bounds are `[92359.00325, 436961.6475, 19.5720042]` to
`[92364.51825, 436968.6395, 20.0810042]`, inside the preregistered scene
footprint `[92211.4889167, 436715.3847278, 92670.675875, 437112.7044582]`.
The dominant XY projection has positive area `17.98155849996 m²`; Newell norm
is `36.12401263`, and local plane error is `0.00072218 m`. GEOS identifies a
self-intersection at `(92360.71725, 436962.9745)`. `buffer(0)` happens to
produce one valid polygon with area equal to floating-point precision, but
this is a positive-area in-scene roof face. Repairing or omitting it would
change the declared fail-closed rule, so no such repair is authorized and no
Rotterdam retry was launched.

The official two-map output remains failed, not complete. On 3060-3 its
`trace/FAILED` marker records `2026-11-09T01:55:25Z`; the original
`trace/trace-job.log` remains intact. The geometry preflight produced no GPU
traces. No process or tmux session for this trace runner remained when
checked. This failure cannot be treated as a two-city channel-ready pilot.

For the useful portion of the bounded pilot that can still be diagnosed
without claiming the two-map gate, the runner now has an explicit
`--munich-only-diagnostic` mode. It accepts only the hash-bound registered
manifest, selects only `01-munich-natural`, requires separate directory names
`munich-diagnostic-work` and `munich-diagnostic` under the same dedicated
parent, and requires an existing Munich scene manifest. The standard path
still requires both approved scenes. A separate
`run_bench_production_munich_diag.sh` wrapper writes markers and the trace
manifest only under that diagnostic output directory; it cannot create the
channel consumer's `trace/DONE` marker. The Munich scene hashes are rechecked
against its reusable preflight manifest. This diagnostic is limited to three
registered array sizes and is not a replacement for the Rotterdam trace or
two-map gate.

The Munich diagnostic was completed on 3060-3 in a detached tmux job. The
first isolated attempt at `munich-diagnostic/` failed before the solver because
the exporter passed a metadata-only `note` field to Sionna's `PathSolver`.
That `FAILED` marker and log remain preserved. The runner now filters this
metadata field out of solver kwargs. A fresh, separate attempt completed at:

```text
<lab-home>/projects/bench-production-pilot-20261109/munich-diagnostic-retry1/
```

It wrote an atomic `DONE` marker at 2026-11-09 09:04:32 WIB with manifest
SHA-256 `69b2449e4d5cba945cdda9a8115298fa8935ad9efd8681d3baf2636546f8c7b1`.
Independent validation re-opened all three NPZ archives, checked their byte
counts and hashes against the manifest, confirmed the required channel/path
fields and finite delay/length values, and verified the JSON manifest's
`isolated_munich_diagnostic` scope. Results:

| N | Paths | Nonzero links | Stored bytes | Solver (s) | Export (s) | Total (s) | Sampled GPU0 used / free (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5,693 | 318 | 264,737 | 1.433 | 0.307 | 1.739 | 775 / 11,136 |
| 128 | 9,210 | 510 | 422,376 | 1.012 | 0.430 | 1.442 | 807 / 11,104 |
| 256 | 17,217 | 990 | 759,114 | 1.481 | 0.758 | 2.239 | 841 / 11,070 |

The GPU sampler ran every 100 ms and captured 9, 9, and 14 samples for the
three solves. These are sampled device-wide usage peaks, not isolated process
allocation. Total raw NPZ size is 1,446,227 bytes. Hashes are recorded in the
remote manifest. The outputs remain on the lab host and are not copied into
the repository. The diagnostic confirms the pinned Munich scene/path export
works; it cannot satisfy the two-map pilot, Rotterdam geometry, or production
channel gate.

## Known limitations before production

- The response is an idealized Sionna `VH` point antenna at each endpoint, not
  measured BS/UE antenna patterns.
- Rotating the panel lattice follows the supplied outward normal; orientation
  correctness still needs the case-geometric panel-normal audit.
- The current remote pilot will validate path extraction and conversion
  compatibility only. Channel operators, NEC coupling, labels, packaging, and
  rate round-trip are separate production pipeline gates.
- Reflected/diffracted path lengths use Sionna delay precision until vertices
  can be exported reliably.
