# Bench users v0: geometry-only preview protocol (declared 2026-11-08)

This protocol precedes all crowd sampling. Use only Munich, Rotterdam and
Maastricht, from the approved Phase 0b sources. First prepare and verify
native street/terrain geometry; no missing ground is interpolated or filled.
Do not open Berlin/Utrecht assets, generate held-out cases, calculate channels
or rates, or run an RF solver. Local CPU geometry tests suffice. If Sionna
were needed later, it is restricted to 3060-3 GPUs (no CPU 5/11 and no
overlap with a CPU 5 timing run). This preview does not close P4.

## Slope, sites and supported heights

Use projected metre coordinates and the native map vertical datum. Each
moving user samples valid bare-earth DTM at its own XY every 20 ms. Sample
real valid street segments only; reject invalid ground, roof/footprint
encroachment, non-walkable ways, bridges, tunnels and water. No automatic
fill. Urban LoD2 wall/roof elevations remain their native values and are
not reconstructed from the terrain. All retained positions and endpoints
stay within their approved map square. Store absolute Z and local height
above ground separately. Report local DTM slope and along-street grade,
and explicitly check Maastricht streets and the actual candidate sites.
Do not substitute another city without recording and auditing it.

Use the geometry-prepared site pool; each facade RIS and mast/roof BS has
valid ground at its own XY, and its BS–RIS segment must be visible. Draw
from eligible sites uniformly; add a roof/mast height offset drawn from
a lognormal (median 4 m, log-SD 0.75, capped at 24 m), and recheck the
BS–RIS segment. User distances to BS or RIS are never used to accept,
reject or resample natural cases. Terrain is for height checks only,
not a reflecting or obstructing RF object. A clear geometric ray is
not a link-budget prediction.

## Natural crowd and scheduler

- RNG root seed 202611081501. Preview draws and bulk draws use separate
  deterministic RNG streams; each map and challenge has its own stream.
- Draw a map-local crowd intensity by lognormal (median 9, log-SD 0.45,
  clipped to [2,22]), then Poisson hotspot cluster count. Draw centres
  from eligible OSM hotspots with OSM-type weights (entrance 1.3,
  crossing 2.2, transit stop 3.5, square 3; existing place/junction
  weights are retained), with 15% street
  background anchors. A centre is displaced by isotropic Gaussian
  jitter (SD 7 m), then snapped to valid street geometry (up to 25 m).
  Mapped entrances can sit on a footprint edge; their users must snap to valid street support. Each cluster size is Zipf exponent 1.9, rejecting draws above 24;
  there is no cap at three people. Add Poisson(3) independently anchored
  singletons; they can occur anywhere on the valid street graph. Empty
  crowds are resampled. No link-class, occupancy or BS/RIS distance
  quotas apply to natural cases.
- Per-user class probabilities before geometry checks: pedestrian 94%,
  elevated outdoor 5.5%, drone 0.5%. Pedestrian phone height
  `1.0 + 0.7 * Beta(2.5,2)` m. A rooftop user is on a real near-level (at most 10°)
  native roof with valid underlying DTM and at least 3 m of roof-to-DTM
  clearance, plus the same phone offset.
  Roofs are geometry supports, not a claim of public access; no bridge or
  terrace is fabricated when the source doesn't represent it. If that
  class has no valid support locally, record the rejection and resample
  as pedestrian. Drone height is a lognormal (median 34 m, log-SD 0.5)
  truncated to 20–80 m above valid ground. Rooftop users are static; rare
  drones can follow the same route XY at their own AGL.
- Keep four BS antennas. A seeded uniform scheduler serves 2, 3 or 4
  eligible crowd members, chosen without replacement (or all if fewer).
  Fix membership during a clip. It uses no link class or rate and has no
  spatial distance filter. Everybody else remains visible as present and
  is not an RF blocker. The preview does not assume that four users can
  all sustain the same service quality.

## Motion and geometric labels

A group shares a walk along the street graph. At a junction, use
forward-biased random turns (weights exponential in 2 × heading cosine),
avoiding immediate U-turns except at dead ends. Followers start with
longitudinal displacement up to ±2.5 m for groups of up to four,
increasing by 0.45 m per extra member up to ±10 m, and up to 0.8 m
lateral jitter. This gives larger groups more room. Smooth the heading
over 2 m and preserve each member's AGL. Revert a complete lateral
track to the validated centreline if any sample is invalid or the
horizontal speed would exceed 1.45 times its scheduled speed; count
these rejections.
Invalid centreline routes are resampled and counted. 24% of group
tracks are static. For moving tracks: 87.5% lognormal typical speed
(median 1.3 m/s, log-SD 0.27, capped at 3.5), 10% slow lognormal
(median 0.45 m/s, log-SD 0.30, capped at 0.9), 2.5% cyclist
(uniform 4–7 m/s). Moving groups have Poisson start events
(0.12 per second) for stops, with exponential dwell (mean 0.9 s,
capped at 3 s). Clip duration lognormal median 10 s, log-SD 0.22,
clipped to 6–16 s and aligned to 0.02 s frames.

Use exact native LoD2 segment occlusion for BS and RIS LOS, with first
blocking building ID and native height provenance. Classify clear BS
as LOS. For absent direct LOS, optionally check up to 32 representative
native wall triangles (at most two per plane), image-method reflection
point inside the triangle, endpoints on the same side, and two clear
geometry segments. A surviving candidate is `reflection_only_candidate`;
otherwise `blocked` means no direct path and no candidate in the tested
set. If native reflection surfaces cannot be checked, use
`no_los_unresolved` for that map. None of these labels establishes the
absence of other propagation paths. Angles are receiver-to-BS/RIS
azimuth and elevation in the local projected frame. Speeds in full
clips are finite differences of the recorded 3D positions.

## Preview and bulk allocation

Use 14 unfiltered natural cases per development map (42 total).
Add a separately labelled challenge group of eight: four moving
shadow-edge cases (including one with at least 0.25 m sampled ground
rise/fall), two all-BS-blocked static clips, one dense crowd (at least 60)
and one with at least five genuine rooftop users. A challenge may search
other development sites/maps if its first map has no qualifying case.
Challenge rejection or route searches may use only geometry; never
feed challenge candidates into the natural draw or fit the natural
intensities to satisfy a quota. Stop and report if a required challenge
cannot be generated under its fixed definition.

The bulk comparison is 5,000 independent natural case snapshots, balanced
across maps before drawing people (1,667 / 1,667 / 1,666). It uses the same
crowd, site, speed and height distributions; initial speed is the drawn
cruise/stop state. Save only shuffled anonymous distribution columns, with no case IDs,
coordinates or trajectories. The 50 retained previews store exact
per-frame, per-user labels in Parquet and figures/cards for human review.


## Durable preview run record

Run after `geometry/slope_gate.json` is present and the three-map slope
and site recheck passes. Local tmux session `bench-users-v0-20261108`,
entry point `bash scripts/run_bench_users_preview.sh`. The preview root
is `outputs/previews/bench-users-v0`; stages write `run.log` and a final
`DONE` or stage-specific `FAILED` marker. A lock prevents duplicate runs.
Hashes bind the source files and slope-gate evidence. On restart, already
written cases and the bulk distribution are accepted only after byte
and input-identity checks. No stage starts a production experiment.


A first diagnostic run was interrupted before bulk completion after
integration review found that the geometry loader did not ingest OSM
node POI tags. The final draw uses the explicit four node categories above.
The source snapshot, seed, geometry gates and other distribution rules
remain fixed. The diagnostic is not pooled with the final cases or used
to tune parameters.


Before the retained rerun, a readback of the first completed candidate
distribution found roof-class points below ordinary phone height. Native
LoD2 roof semantics alone allowed ground-adjacent or partly buried supports.
The rooftop support gate now requires at least 3 m clearance above valid
local DTM, independent of BS/RIS distances and link outcomes. The prior
candidate set is archived separately. Same seeds and all other settings.
