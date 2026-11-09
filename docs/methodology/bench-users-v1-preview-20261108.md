# Bench users v1: development previews (2026-11-08)

Geometry and clear-segment (LOS) labels only: no channel, rate or RF solver.
Munich, Rotterdam and Maastricht only; Berlin and Utrecht were never opened.
The declared rules are in [bench-users-v1-design-20261108.md](bench-users-v1-design-20261108.md).
Outputs: `outputs/previews/bench-users-v1/` (index: `index.md`; 50 cases, 5 movement
strips, distribution and link-class figures; `validation.json`; `DONE` marker). v0
(`outputs/previews/bench-users-v0`) is untouched. Entry point:
`bash scripts/run_bench_users_preview.sh` (prepare -> generate -> render -> validate);
the validation **passes**.

## Rule changes (see the design doc for parameters)

1. **One ground rule.** A position is valid only if its full 4x4 DTM stencil is valid
   (elevation = bilinear 2x2, slope = bilinear central-difference gradients). Never
   filled. Applies to street segments, graph nodes, every user frame, drones, masts.
2. **Walkable slope cap 15 deg**, `highway=steps` excluded, street graph pruned
   (nodes/edges above the cap or invalid, fragments under 8 nodes), followers
   reverted to the centreline or the route rejected. **Elevated and drone users have
   `slope_deg = null`.**
3. **Site planning.** BS-RIS clear segment at 20-150 m (3-D, after the antenna draw);
   BS sees at least 10 % of street probes, RIS sees at least 10 %, and the RIS reaches
   BS-shadow probes amounting to at least 5 % of all probes. 12 / 16 / 16 sites per
   map (Munich / Rotterdam / Maastricht). **Served group** (2-4) draws each slot from
   RIS-reachable users, except 15 % deliberate negatives from users without RIS reach.
4. **Names.** `blocked` -> `no_los` (no clear direct segment; reflections are not
   traced), `reflection_only_candidate` -> `no_los_reflection_candidate` (bounded
   one-bounce geometry candidate flag only), `all_blocked` -> `all_no_los`.
5. **Building-mounted supports (new, needs your sign-off).** AHN4 has no ground under
   buildings (0.4 % of facade, 0.1 % of roof centroids valid), so the facade RIS,
   rooftop BS and rooftop users take their height reference from valid 4x4-stencil
   DTM samples on a ring 3-6 m outside the footprint (median of the 3 nearest). Not a
   fill. Stored and validated per site and per user (`ground_ref`). Also: Munich roof
   faces are now triangulated (v0 only used 3-vertex faces: 31 usable roofs, now 2,952).

## Rejection counts, v0 vs v1

| Step | v0 | v1 |
| --- | --- | --- |
| Validation | **failed**: 30 cases touched a no-data stencil | passes: 0 stencil violations; max height and slope readback residual 0 |
| Geometry: rejected OSM road pieces, Munich / Rotterdam / Maastricht | bridge-or-tunnel 160 / 68 / 4; inside building 129 / 56 / 14; invalid terrain 70 / 285 / 360 | bridge-or-tunnel 122 / 60 / 4; **steps 64 / 50 / 18** (new); inside building 120 / 54 / 14; invalid terrain **70 / 325 / 410** |
| Graph pruning (nodes total; invalid; over 15 deg; fragment) | n/a | Munich 4,123; 0; 15; 41. Rotterdam 3,039; 2; 8; 51. Maastricht 2,358; 1; 3; 68. Edges failing at the midpoint: 0 / 3 / 3 |
| Roofs usable for rooftop users, Munich / Rotterdam / Maastricht | 31 / 44 / 27 | 2,952 / 6,773 / 1,980 (foot reference) |
| 50 preview cases: lateral samples rejected | 922 | 1,967 (4x4 stencil is stricter) |
| 50 preview cases: slope-cap rejections | no cap | 0 samples, 0 centreline routes (cap bites at graph level) |
| 50 preview cases: invalid centreline routes | 7 | 4 |
| 50 preview cases: site redraws after the antenna draw (BS-RIS blocked or out of range) | 61 | 0 |
| 50 preview cases: speed-guard reverted tracks | 74 | 82 |
| Served slots that fell back to the other pool (no RIS-reachable user) | n/a | 36 (preview), 3,179 of 5,000 bulk cases have a fallback slot |
| 5,000 bulk: lateral rejected / slope-cap samples / invalid centreline | 359 / n/a / 45 | 277 / 47 / 9 |
| 5,000 bulk: site redraws | 713 | 0 |
| Site planning, pairs (in range -> clear BS-RIS -> evaluated -> eligible), Munich / Rotterdam / Maastricht | n/a | 263,037 -> 10,347 -> 1,000 -> 28; 176,664 -> 16,715 -> 1,000 -> 122; 102,614 -> 8,555 -> 1,000 -> 182 |

## RIS LOS share (first frame, geometry only)

| Set | v0 all users | v1 all users | v0 served | v1 served |
| --- | ---: | ---: | ---: | ---: |
| 42 natural preview cases | 6.8 % (85 / 1,259) | **27.5 %** (317 / 1,152) | 7.6 % (9 / 119) | **50.8 %** (62 / 122) |
| 8 challenge cases | 15.2 % | 25.2 % | 14.8 % | 63.6 % (14 / 22) |
| 5,000-case bulk draw | 10.3 % | 23.0 % | not scheduled in v0 | **63.6 %** (9,551 / 15,017) |

32 of 42 natural cases have at least one RIS-reachable user; 10 crowds have none, so
their served users cannot have RIS LOS (this is the natural crowd, not a quota). In 11
of 42 natural cases every served user has RIS LOS. BS link classes (natural, first
frame): `los` 379, `no_los` 739, `no_los_reflection_candidate` 34. Bulk: 44,143 / 96,398 / 1,867.

## BS-RIS distance (m, per case, after the antenna draw)

All 50 cases: min 34.0, p5 39.2, median 94.8, p90 140.3, max 144.7 (the 50 cases span
34-145 m; v0 spanned 13.7-300 m, median 65 m, with the 7 m case). Bulk draw (5,000
cases): 31.7 / 38.8 / 98.8 / 140.0 / 147.1 (min / p5 / median / p90 / max). Declared
range 20-150 m; every case and bulk draw is inside it and has a clear BS-RIS segment.
Pool distances (base, before the antenna offset): Munich 32-143, Rotterdam 54-140,
Maastricht 34-146. Rooftop BS dominates (Munich 12 of 12, Rotterdam 13 of 16,
Maastricht 12 of 16 sites; the rest are 12 m masts).

## DTM slope by height class

| Class | Rows (50 cases) | Slope p0 / p50 / p95 / p99 / max (deg) | Bulk users: null / max |
| --- | ---: | --- | --- |
| pedestrian | 690,988 | 0.001 / 1.43 / 3.85 / 6.10 / **11.8** | 0 null; max 14.999 (cap 15) |
| elevated (rooftop) | 38,438 | all **null** | 7,863 of 7,863 null |
| drone | 4,666 | all **null** | 752 of 752 null |

v0 pedestrian bulk samples: 1.9 % above 15 deg, 1.4 % above 20 deg, max 83.6 deg;
elevated users carried slopes up to 85 deg.

## Validation result

PASS. It recomputes, independently of the generator and from the exported native
terrain: full 4x4 stencil validity for every user row on the DTM, every BS/RIS ground
sample and the elevation; pedestrian slope at most 15 deg and equal to a recomputed
value; elevated/drone slope null; BS-RIS distance 20-150 m with the recorded clear
segment (`bs_ris_los`); RIS AGL 3-30 m; the shadow-edge user is served; served and
all-user RIS-LOS shares (floor of 50 % on the natural served share, which passes with
50.8 %). Site ground methods: RIS 50 of 50 `building_foot_dtm`; BS 43 foot, 7 mast.

## Known limitations and open items

* **Served share is crowd-limited.** The natural served RIS-LOS share is only about
  half because 10 of 42 crowds have no RIS-reachable user and the share of RIS-LOS
  users is about a quarter. The floor (50 %) is met by 0.8 points, so it is a thin
  margin. If more is wanted: raise `RIS_MIN_LOS_SHARE`, or choose the site after the
  crowd (changes the site distribution; not done).
* **Distances pile up near the 150 m cap** (p90 about 140 m): long clear pairs are more
  numerous than short ones in a 440 m square. The cap is a hard declared range; no
  distance weighting was added. A longer tail would need a wider range.
* BS sees only about 10-45 % of street probes (rooftop centroids are set back from the
  roof edge); BS and RIS sit on facades/roofs with a foot-ring reference, which rests on
  my choice (item 5 above) and was not in the brief.
* Rooftop supports are not a claim of public access; no terraces or bridges exist in
  the source. The 15 deg cap and the 4x4 rule remove steps and kerbs but cannot
  distinguish a real ramp from a DTM artefact. Slope labels are the local DTM gradient,
  not the gradient along the walking direction.
* `no_los` does not mean no propagation path; reflections are not traced and the
  candidate flag searches up to 32 wall triangles.
* Site thresholds (10 %, 10 %, 5 %, 25 m separation, 16 sites) were fixed after
  inspecting geometry-only pool statistics; no user draw or rate informed them.
* v0 outputs are no longer reproducible from the current sources (rules changed in
  place); the v0 directory is kept only as the failed reference.
* Maastricht passes the site and street checks with the 4x4 rule (410 street pieces
  rejected for invalid terrain, up from 360). No city swap was needed.
