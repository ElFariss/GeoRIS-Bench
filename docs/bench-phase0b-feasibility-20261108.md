# Phase 0b Maastricht scene feasibility protocol (declared 2026-11-08)

## Scope and pass criteria

One **development feasibility** tile, fixed by the lead before any tracing,
with EPSG:7415 bounds [175967, 316952, 176407, 317392] m (440 m square;
center approximately 5.688 E, 50.844 N). It does not select any of
the five final environments and is not a training/validation/test assignment.
No optimizer, channel dataset, labels, or production RF results are generated.
No held-out map is touched. The pilot tests source downloads, terrain-height
sampling, LoD2 mesh conversion, Sionna scene load and one link smoke only.

Use the newest openly accessible 3DBAG LoD2.2 source, subject to verifying the
page's version and source vintage. Use a PDOK/AHN bare-earth DTM in NAP for
ground height only, resampled to at most 1 m. Preserve source geometry,
projection, source checksums, metadata and license attribution locally on
3060-1. No LiDAR terrain mesh is added to the RF scene: sample DTM elevation
at the building/user/BS coordinates, but do not trace rays against terrain.
Assign building surfaces to simple ITU material classes from their CityJSON
semantics: roof surface, exterior wall surface, ground surface. Do not claim
measured RF materials.

Before the smoke trace, the downloaded feature count and checksum manifest
must be complete; the DTM must have valid coverage over the tile with at most
1% nodata and at least 5 m relief; LoD2 surfaces must have valid topology and
material labels; placed audit points must be at least 0.5 m above the sampled
terrain at their XY; and Sionna 2.0.1 must load the built Mitsuba scene without
errors. A single LOS smoke from one outdoor transmitter to one outdoor receiver
must return a finite coefficient and valid delay. Report all failures; do not
change thresholds after the run.

If the full tile fails only its DTM nodata criterion, report that failure
unchanged, then evaluate one separately labelled diagnostic patch: enumerate
100 m square windows on a 10 m aligned grid inside the same bounds, order by
easting then northing, and choose the first with at most 1% nodata and at least
5 m relief. This selection uses only DTM validity/elevation and spatial order.
If none exists, stop with no scene trace. If one exists, use DTM samples only
inside that patch for two geometry-selected outdoor link endpoints and height
checks; load all LoD2 buildings in the original 440 m square as the RF geometry
halo. This conditional patch does not replace the failed full-tile verdict and
is not a production map or experiment. On 3060-1, record wall time, host peak
RSS, free memory and GPU memory before/after (no other GPU jobs are stopped).
If the host is too busy to safely run the scene, keep downloaded geometry
summary and stop before loading the solver.

## Execution result: Maastricht feasibility stop

The pinned remote acquisition completed on `labkc-comm1` (3060-1)
from tmux session `bench-phase0b-attempt5`. It downloaded all four official
3DBAG tiles and the PDOK AHN4 crop, checked byte counts and checksums, validated
v2025.09.03 / CityJSON / EPSG:7415 tile metadata, and found 4,215 distinct
CityObjects across the tiles (2,100 Building, 2,115 BuildingPart; zero ID
duplicates). Counts are source-tile totals before clipping to the proposed
440 m square. The CityJSON tile extents are retained in the result JSON. The
combined decompressed source digest is recorded there too.

The AHN4 crop is 880 × 880 at 0.5 m with EPSG:28992 horizontal coordinates
and NAP vertical heights. It has 33.247% nodata. Its 5th–95th percentile
relief is 6.616 m (full valid-cell range 16.049 m), so the fixed full-square
gate fails on nodata. The preregistered 100 m windows were evaluated on the
1 m grid in easting-then-northing order; none met both the ≤1% nodata and ≥5 m
relief gates. The run therefore stopped before geometry placement, Mitsuba
scene loading, or Sionna tracing, as required. No channel, path, or RF smoke
output exists; the LoD2 mesh conversion and material scene load remain
unverified. Terrain stayed height-only.

The completed acquisition and audit took 11.77 s wall time and peak RSS was
180,848 KiB. The remote wrapper recorded available host memory and GPU memory
before and after; it did not stop any existing workload. The report is
`outputs/audits/bench-phase0b-20261108/feasibility.json` and its raw log is
`outputs/audits/bench-phase0b-20261108/feasibility.log`. Raw source tiles and
DTM remain under `/var/tmp/bench-phase0b-20261108/attempt5/raw` on the lab
host. The run is complete with status `stopped_pretrace_no_qualifying_dtm_patch`,
not a scene-smoke pass.

## Geometry-only scene candidate

Use the single preselected 440 m × 440 m square centred near
5.688 E, 50.844 N in the 3DBAG projected CRS. Use only the DTM and building geometry, not rate or channel
information, for selection. Report local building count, LoD2.2 coverage,
building-height provenance, slope range, and source vintages before picking
one demonstrator building. The selected building is only a parser/scene smoke
fixture. It does not fix any of the five benchmark areas.

## Fidelity comparison plan (future; no channel runs now)

On each approved map, preserve one geometry manifest and render the same
buildings in (a) LoD1 extrusions from verified height attributes / OSM where
explicitly available and otherwise a flagged default, and (b) LoD2.2 roof and
wall surfaces. Match the footprint, terrain samples, BS/RIS/user placements,
frequency, solver options, material proxies, ray budget, random seeds and
path-to-channel mapping. Record per-building measured/OSM/default height
provenance and both models' surface/edge counts and hashes. Compare building
silhouette/roof-angle error first, then LOS/NLOS class changes and path
coefficient changes on the same geometry-selected link set. Keep LoD1 vs LoD2
as development fidelity analysis; it cannot be used to pick the held-out maps.

## Reproduce

The live 3DBAG `pand` API is not used: its pagination counts did not yield a
consistent, auditable complete building set. The builder instead downloads
four publisher-pinned v2025.09.03 CityJSON tiles covering the proposed extent,
verifies each compressed-byte SHA-256 and byte count before decoding, then
checks tile version and EPSG:7415 reference system. Duplicate CityObject IDs
across tiles are a hard failure. Tile URLs and hashes are embedded in
`scripts/build_bench_phase0b_scene.py` and repeated in the result manifest.
The PDOK WCS AHN4 DTM crop is also pinned by byte count and SHA-256; its native
0.5 m cells are sampled on a deterministic 1 m grid for the predeclared patch
search. NAP is the vertical datum.

An initial API probe stopped on inconsistent pagination before tracing. Two
subsequent preflight retries also stopped before parsing: one exposed
requests’ automatic HTTP gzip decoding, and the next exposed a copied tile-hash
transcription error against the lead-verified manifest. Both were corrected;
all were non-tracing implementation failures. The final pinned tile run is the
sole completed Phase 0b feasibility evaluation. If the strict 440 m DTM
gate fails and no predeclared 100 m patch qualifies, the wrapper records a
completed pre-trace stop. No scene load or link trace is allowed in that case.

The bounded builder lives in `scripts/build_bench_phase0b_scene.py`; the remote
runner is `scripts/run_bench_phase0b_scene.sh`. Source versions, input and
output hashes, run log, status markers, and quality metrics are recorded in
`outputs/audits/bench-phase0b-20261108/feasibility.json`. Raw downloaded tiles,
terrain rasters, geometry and any future path files remain on the tracing
host. Terrain is used for height only and is never added to the RF scene.
