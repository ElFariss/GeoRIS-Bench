# Bench users v1: declared rule changes (2026-11-08, before any v1 case was generated)

v1 fixes four defects found in the v0 preview (which failed its own validation
with `label used a no-data stencil`). Everything not listed here is **unchanged
from [v0](bench-users-v0-design-20261108.md)**: seed `202611081501`, clustered
hotspot crowd, Zipf group sizes (exponent 1.9, cap 24), speeds, clip lengths, the
94 / 5.5 / 0.5 % pedestrian / elevated / drone split, leader-follower jitter,
14 natural cases per development map plus 8 challenge cases, and the
5,000-case bulk draw. Development maps only (Munich, Rotterdam, Maastricht);
Berlin and Utrecht are never opened. Geometry and clear-segment (LOS) tests only;
no rate, channel or RF solver. v0 output (`outputs/previews/bench-users-v0`) is left
untouched; v1 writes to `outputs/previews/bench-users-v1`.

## 1. One ground-validity rule (4x4 stencil)

* v0 sampled height from a 2x2 bilinear patch but the validator (and the intended
  slope estimator) used a 4x4 neighbourhood, so 30 cases touched no-data.
* v1 has **one rule**: a position is valid for placement, a height label or a
  slope label only if all 16 DTM cells of its 4x4 stencil are valid. Elevation is
  the bilinear interpolation of the inner 2x2 cells. Slope is the bilinear
  interpolation of the four central-difference gradients at those cell centres
  (the extra ring supplies the central differences), so the slope estimate is
  smoother than a raw 2x2 difference.
* A position failing the rule is **rejected, never filled, interpolated across,
  or moved by the loader**. This applies to street segments (checked at 1 m
  spacing, as before), street-graph nodes, every follower and leader sample at
  every 20 ms frame, drones, and mast/ground supports. Rejections are counted per
  step (geometry, graph, per case).
* The validator recomputes the 4x4 validity, the elevation and the slope from the
  exported native terrain independently of the generator.

### Building-mounted supports (new, needed for the Dutch maps)

AHN4 (Rotterdam, Maastricht) has **no ground data under buildings**: only 0.4 % of
facade and 0.1 % of roof centroids have a valid 4x4 stencil at their own XY, so
v0's "valid ground at the support's own XY" rule left essentially no RIS facade or
rooftop-BS candidate on those maps. For the three building-mounted supports
(facade RIS, rooftop BS, rooftop "elevated" users), the **ground reference** is
the *building-foot DTM ring*: valid, 4x4-stencil, outdoor DTM samples on the
building footprint buffered by 3 m (else 4.5 m, else 6 m), spaced 2 m; the
reference is the median of the 3 ring samples nearest to the support. At least 3
valid samples are needed or the support is rejected. This reads real DTM at the
street/garden level next to the building and is **not** a fill of the no-data
cells. Masts and users on the street keep the DTM at their own XY. The method is
stored per site (`*_ground_method`, `*_ground_samples`) and per user (`ground_ref`
= `dtm_at_position` | `building_foot_dtm`), and the validator checks each stored
sample against the native terrain. Munich uses the same rule for consistency.

## 2. Walkable slope cap and steps

* **Cap: 15 deg** DTM slope for pedestrians and walking paths (the 1 m DTM turns
  kerbs, walls and steps into 30-80 deg spikes; v0 had 1.1 % of pedestrian samples
  above 20 deg).
* OSM `highway=steps` is **excluded** from the walkable graph (counted as
  `steps` in the geometry rejections).
* Street-graph pruning (per map, before any sampling): remove graph nodes with
  invalid ground or slope above the cap, remove edges whose midpoint is invalid or
  above the cap, then drop connected fragments smaller than 8 nodes (about 14 m).
  Nothing is relocated.
* Per user and per frame: a pedestrian follower whose lateral track has any
  invalid or above-cap sample is reverted to the validated centreline (counted);
  a centreline sample above the cap rejects the whole route and it is resampled
  (counted: `centreline_slope_cap_routes`). Drones need valid ground but no slope
  cap.
* **Slope label**: pedestrians get the DTM slope. **Elevated (rooftop) users and
  drones get `slope_deg = null`** (they do not stand on that ground); their
  `ground_z_m` still gives the height reference. Pedestrian slope <= 15 deg and
  null slope for the other classes are validation checks.

## 3. BS / RIS site selection

v0 took the first clear BS-RIS pair found per anchor and never asked whether the
RIS helps anyone (7-15 % of users had RIS LOS; case 02 had the BS and RIS 7 m
apart). v1 plans sites per map from geometry only (own seeded stream
`SeedSequence([seed, 5, map])`).

* **Candidates.** RIS: wall-face centroids on real LoD2 facades with RIS height
  3-30 m above the building-foot ground, thinned to one per 10 m x 10 m bin. BS:
  rooftop points on real roof triangles at least 6 m above the foot ground (one
  per 15 m bin, antenna evaluated at roof + 1.5 m) and 80 street masts (ground +
  12 m).
* **Hard constraints.** Clear BS-RIS segment (native LoD2 Embree test) at a 3-D
  distance of **20-150 m**; the same range must also hold after the antenna-height
  offset below.
* **RIS coverage (geometry-only, street probes).** About 1,200 street-graph nodes
  at 1.5 m height are the probes. A pair is eligible if the BS has LOS to at least
  10 % of probes, the RIS has LOS to at least 10 %, and the RIS reaches probes the
  BS cannot see ("BS shadow") amounting to at least 5 % of all probes (and at
  least 25 probes). Up to 1,000 seeded-random clear pairs are evaluated per map.
* **Pool.** Up to 16 sites per map, randomly ordered, with RIS points at least
  25 m apart and at most 2 sites per BS anchor. At least 6 sites are required.
  The thresholds were fixed after looking at pool statistics (geometry only),
  before any user was drawn; no user labels were used.
* **Per case.** A site is drawn uniformly from the pool; a BS antenna offset is
  drawn lognormal (median 4 m, log-SD 0.75, capped at 24 m; for rooftop BS the
  1.5 m minimum antenna height is already included), and the clear segment and the
  20-150 m range are re-checked (rejections counted, redraw).
* User distances to the BS or RIS are **never** used to accept, reject or
  resample natural cases, and no distance bands or class quotas apply. The natural
  crowd, heights and speeds are exactly the v0 distributions.

## 4. Served group (2-4 users) prefers RIS-reachable users

* RIS-reachable user: RIS LOS in at least 50 % of the clip frames.
* Number served: uniform on {2, 3, 4} (all, if fewer users). Each served slot is
  drawn from users **without** RIS reach with probability **0.15** (declared
  negative cases) and otherwise from RIS-reachable users, uniformly within the
  pool and without any distance filter. If a pool is empty the slot falls back to
  the other pool (counted as `served_slot_pool_fallback`). Membership is fixed
  for the clip. Unserved users stay in the crowd with all labels and are not
  blockers.
* Challenge `shadow_edge` cases force the tracked shadow-edge user into the
  served group (the other slots follow the rule). The `all_no_los` challenge uses
  the same rule.
* The bulk draw applies the same scheduler (to its single snapshot) so the served
  RIS-LOS share is reportable for it as well.
* The validator reports the served RIS-LOS share and requires, as a sanity floor,
  at least 50 % across the 42 natural preview cases.

## 5. Link labels (rename)

| v0 label | v1 label | Meaning |
| --- | --- | --- |
| `los` | `los` | clear direct segment (LoD2 native triangles) |
| `blocked` | `no_los` | no clear direct segment. Reflections are **not traced** in the preview; this is not a statement that no path exists |
| `reflection_only_candidate` | `no_los_reflection_candidate` | no direct LOS, but a bounded one-bounce geometry candidate exists (up to 32 wall triangles; image-method point inside the triangle, two clear segments). A candidate flag only, not a traced or field-valued path |
| `no_los_unresolved` | `no_los_unresolved` | unchanged (wall surfaces unavailable) |

The challenge scenario `all_blocked` becomes `all_no_los`. `ris_los` stays a
boolean (clear RIS-to-user segment).

## Validation additions (all must pass)

* no sample (user, BS or RIS ground reference) has a stencil that touches no-data;
* pedestrian slope <= 15 deg, matching an independent recomputation (< 1e-3 deg);
* elevated and drone slope is null;
* per-case BS-RIS distance within 20-150 m and a recorded clear BS-RIS segment;
* the BS/RIS ground references rest on valid native DTM samples;
* served and all-user RIS-LOS shares reported per kind, plus the bulk draw.
