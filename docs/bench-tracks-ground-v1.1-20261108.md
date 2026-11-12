# Building-only terrain reference and preview v1.1 — declared 2026-11-08

Use only checksum-verified Munich, Rotterdam and Maastricht inputs from Phase 0b.
Do not open Berlin or Utrecht. Retain the 50-case v1 design, seeds, 5,000 anonymous
draws, slope rule, site and scheduler thresholds. Run no RF simulation here.
Leave v1 outputs untouched. Street heights use the native DTM unchanged;
building references are a separate derived layer.

## Interpolation and acceptance

Rasterize native LoD2 footprints at the original DTM cell centres. Fill only
original no-data under a footprint inside the approved area, outside the
protected mask. Raster ownership is deterministic by building ID in overlaps.
Protect mapped water (including loader water areas), OSM bridge, covered and
tunnel corridors, underground structures, and the cells within 1 m of water.
Protect line ways by half the mapped width plus 2 m (8 m half width plus 2 m if
width is absent). The inputs cannot identify unmapped infrastructure.

For each building up to 2,500 m² and maximum bounding width 80 m, seed a
Delaunay linear interpolator with original finite DTM cell centres within 10 m
outside the footprint, excluding other footprints and protected cells. Require
at least 8 seeds, a containing triangle and nearest source cell within 12 m.
Do not extrapolate, chain fills, smooth, or use averages. In the use layer only,
reject predicted cells more than 2.5 m from the building's native model ground
value (agreement guard, not a fitted correction). Compare raw predictions before
this guard. Never filter measured values for agreement with LoD2.

Validate on Munich by masking every original-valid cell underneath eligible
footprints and applying the same method without seed leakage from any footprint.
This checks reconstruction of the provider's surface under buildings, not
independent LiDAR measurements. Gate on raw predictions: at least 2,000 predicted
masked cells, at least 40% coverage of eligible masked cells, median absolute
error ≤0.75 m and p95 ≤2.0 m. If it fails, retain the audit but do not use
estimated cells in v1.1. Report bias, RMSE and absolute-error percentiles
separately for large (≥500 m²), courtyard, steep (seed height p90–p10 /
footprint diameter ≥0.0875), canal-side (within 15 m of mapped water), and
underpass footprints (intersect a protected transport corridor). Report
native-cell, raw predicted and accepted filled agreement with native LoD2
ground heights for each map, including protected and capped footprint counts.
A passing aggregate gate does not certify ground at a specific terrace,
excavated plot or underpass.

For a rooftop user/BS or facade RIS, choose the nearest valid same-footprint
cell centre within 1.5 m by Euclidean distance with deterministic ties.
Record the cell and its source. If none is eligible, use the native building
ground level with a distinct fallback flag (no raster cell is created);
reject supports that lie in a protected cell. The native base is a fallback
reference, not a local survey elevation. Raster flags are 0=no_data,
1=measured (unmodified provider DTM, including any publisher reconstruction),
and 2=interpolated_under_footprint. Pedestrians, drones, walking segments and
masts always sample the original valid 4×4 DTM stencil.

## Durable run

**Recovery amendment, before the first v1.1 run (19:13 WIB):** The host reboot
cleared its private Phase 0b source cache. All three development LoD2 and DTM
sources must still match the original hashes. If the original OSM XML cannot
be restored byte for byte, use a fresh download of the same development
bounding box in a separate manifest and record the changed hashes; do not
alter the Phase 0b or v1 manifests. Require the exported building records,
original terrain arrays, street counts, slope summaries and rejection counts
to match v1 before drawing cases. This checks the street/terrain gate, not
byte equality or identity of all OSM tags. Report v1.1 as a development
preview using the recovered source snapshot; do not attribute all changes
from v1 solely to the footprint overlay. No acceptance threshold changes.

**Site-search repair, declared before retry (19:18 WIB):** The initial
1,000-pair search selected three valid Munich sites and stopped before case
sampling. If a city's first pass returns fewer than six sites, examine one
additional seeded batch of at most 3,000 unused BS–RIS pairs from the same
candidate pool. Keep the original sites, probes and every coverage, distance
and diversity criterion; accept at most the sites needed to reach six. The
repair changes only site-search effort on a failed development map. If the
extra batch cannot reach six, the gate remains failed. No preview outcome
was available to select this change.

Local CPU only; tmux bench-users-v11-20261108, wrapper:
BENCH_USERS_OUT=outputs/previews/bench-users-v1.1 BENCH_USERS_GROUND_FILL=1
bash scripts/run_bench_users_preview.sh (with BENCH_USERS_SOURCE_MANIFEST and
BENCH_USERS_BASELINE_GEOMETRY set for a recovered snapshot). The root contains run.log, DONE or
FAILED, the original geometry export, separate ground_fill layer and audit,
and preview validation. Staged cases can resume only under the same input
hashes. No production trace is authorized here.
