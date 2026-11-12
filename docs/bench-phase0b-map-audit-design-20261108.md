# Phase 0b geometry audit design (2026-11-08)

The five 440 m squares and proposed roles are in
[`configs/bench/phase0b-area-proposals-20261108.json`](../configs/bench/phase0b-area-proposals-20261108.json).
Use only geometry, coverage, survey provenance, release terms and urban
form to propose maps. No channel calculation in any proposed held-out map
or any production map. The only channel run allowed is one bounded,
development-only scene-builder smoke, documented separately.

## Quantities recorded before selection is accepted

- For each area, take provider buildings whose native footprint representative
  point is inside the square. For CityGML, use LoD2 GroundSurface XY; for
  CityJSON use each parent building's LoD0 footprint and check its children's
  LoD2.2 geometry. A usable model needs finite coordinates, a positive roof
  height above its ground/base, and actual roof and wall surface categories.
  Report rejected and provider-reported invalid/quality flags separately
  where available; this audit is not solid topology validation.
- For a second, independently comparable denominator, use the OSM building
  footprint ways/multipolygon relations with representative point inside the
  square (not `building:part` or unclosed ways). Fetch the enclosing WGS84
  bounding rectangle; de-duplicate any way member consumed by a building
  multipolygon. Match each footprint by at least 50% area overlap with the
  union of usable provider LoD2 footprints. Report matched, unmatched but
  with valid `height`/`building:height`/levels, and unmatched default.
  Also report OSM-tagged versus default among **all** OSM buildings as the
  counterfactual LoD1 baseline. OSM mismatches may be segmentation or
  vintage differences; do not label them all missing physical buildings.
- LoD1 uses explicit height first, levels × 3 m second, 9 m default third.
  These are baseline assumptions, not inferred labels for the main LoD2
  dataset. Reject nonpositive, unparsable and implausible (>300 m) values.
- Read DTM in its native projected grid. Report valid cell count, no-data
  share, min/max and 5th–95th percentile relief. A sloped candidate should
  show at least 3 m of robust relief; a bridge or wall top in one cell
  cannot establish the slope. Missing elevation at actual future probe
  locations is a Phase 1 rejection, never silent fill across water.
- Report observed download bytes and SHA-256 per asset/page; record public
  feed and actual tile filenames. Distinguish source acquisition date,
  model generation/update dates, and catalog timestamps. Preserve datum
  and horizontal CRS, translate to local metric ENU only after matching
  the vertical references, and retain transform provenance.
- Approval remains manual. Material class assignment is fixed by surface
  type, not fitted. Terrain heights can place nodes/building bases; the
  ground mesh is used for geometry checks only and is absent from the RF
  scene, as requested.
