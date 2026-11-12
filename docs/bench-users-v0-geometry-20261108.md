# User preview geometry gate — 2026-11-08

## Declared geometry criteria (before user sampling)

This stage uses only the checksum-verified Phase 0b development map bundles for
Munich, Rotterdam and Maastricht. The loader rejects Berlin and Utrecht. No RF
channels, Sionna, GPU, or user samples are involved in this gate.

- Ground elevations remain exactly as supplied by DGM1/AHN4. No-data is never
  filled, extrapolated, or interpolated across. Elevation and the bilinear local
  slope derivative require all four source cells.
- A mapped OSM street segment is walkable only when all samples at no more than
  1 m spacing have valid ground, lie outside LoD2 footprints and mapped water,
  and its OSM way is not marked bridge, tunnel, covered, or underground.
- Every candidate user point and every BS/RIS support point must have a valid
  local ground sample. A terrain-invalid candidate is rejected, never relocated
  by the loader.
- Building occlusion uses publisher LoD2 surfaces where available. Incomplete
  shells are preserved and receive an explicitly tagged conservative extruded
  slab. `los_batch` uses native triangle ray hits through Embree when installed;
  the CPU triangle path is the small-preview fallback.
- Slope has no arbitrary acceptance cutoff. Report the valid street-sample
  distribution and invalid/rejected route counts; steep terrain remains
  eligible if it has valid ground and a connected mapped path.
- BS–RIS candidate site pairs require valid ground at both XY locations and a
  clear 3-D geometry LOS. BS supports are rooftop or 12 m mast; RIS candidates
  are real facade points only. Roof-as-RIS fallback is forbidden.

## Scope and provenance

Inputs and hashes are read from `outputs/audits/bench-phase0b-20261108/map_downloads.json`;
the local cache is `/tmp/atomwise-phase0b-input`. Projected XY uses EPSG:25832
for Munich and EPSG:28992 for Rotterdam/Maastricht. The terrain vertical datum
is DHHN2016 (EPSG:7837) for Munich and NAP (EPSG:5709) for the Dutch areas.

The preparation command is:

```sh
PYTHONPATH=src /tmp/atomwise-phase0b-audit-env/bin/python scripts/prepare_bench_users_geometry.py
```

It writes one map JSON and one terrain NPZ per development area under
`outputs/previews/bench-users-v0/geometry/`, plus this report's results below.

## Results

## Slope and site geometry gate results

No cells were filled. The street sample slope is length-weighted at approximately 1 m spacing; the slope cutoff remains intentionally unset.

### Munich

- Source-verified LoD2 building geometries: 335; exact/partial height provenance is preserved.
- Valid walk-edge segments: 782; valid hotspot points: 197.
- Street ground samples: 11935/11935 valid; 0 rejected.
- Slope (degrees; min / p50 / p90 / p95 / max): 0.00561233432337723 / 1.67182018892837 / 3.7664304034340534 / 5.381478348777814 / 74.22469246144496; samples >10° / >20° / >30°: 215 / 135 / 63.
- BS–RIS geometric LOS site candidates: 24 across 13 distinct BS anchors ({"mast": 13, "rooftop": 11}); Embree enabled: True.
- Rejected road counts: `{"bridge_or_tunnel": 160, "building_interior": 129, "invalid_terrain": 70, "water": 0}`.

### Rotterdam

- Source-verified LoD2 building geometries: 193; exact/partial height provenance is preserved.
- Valid walk-edge segments: 532; valid hotspot points: 116.
- Street ground samples: 6984/6984 valid; 0 rejected.
- Slope (degrees; min / p50 / p90 / p95 / max): 0.058596537788491054 / 2.368882999415031 / 5.0959879418860945 / 6.5301665016778125 / 83.54764280063726; samples >10° / >20° / >30°: 140 / 48 / 25.
- BS–RIS geometric LOS site candidates: 24 across 13 distinct BS anchors ({"mast": 24}); Embree enabled: True.
- Rejected road counts: `{"bridge_or_tunnel": 68, "building_interior": 56, "invalid_terrain": 285, "water": 0}`.

### Maastricht

- Source-verified LoD2 building geometries: 248; exact/partial height provenance is preserved.
- Valid walk-edge segments: 522; valid hotspot points: 59.
- Street ground samples: 5412/5412 valid; 0 rejected.
- Slope (degrees; min / p50 / p90 / p95 / max): 0.012489308340416918 / 2.240522614147962 / 4.902516140782289 / 6.406445363120641 / 81.64637419126018; samples >10° / >20° / >30°: 120 / 34 / 12.
- BS–RIS geometric LOS site candidates: 24 across 14 distinct BS anchors ({"mast": 22, "rooftop": 2}); Embree enabled: True.
- Rejected road counts: `{"bridge_or_tunnel": 4, "building_interior": 14, "invalid_terrain": 360, "water": 1}`.

### Maastricht planned-site validity

The cached Maastricht development bounds have 100.0% valid 1 m street-ground samples across the accepted walking graph. There are 24 terrain-valid, positive geometric BS–RIS LOS candidate pairs. This is a geometry availability result only; it does not imply radio coverage or a user-distribution pass.

The slope tails include a small set of very steep samples (above 30°; counts are
listed per map). Those points remain in the geometry audit because this gate
does not impose a slope cap; they may be mapped steps or a street/DTM mismatch
and should be visible in the preview review rather than silently discarded.

Full structured evidence: `outputs/previews/bench-users-v0/geometry/slope_gate.json`.
