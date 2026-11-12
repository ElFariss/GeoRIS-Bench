# Bench users v0 — development previews, 2026-11-08

The [declared protocol](bench-users-v0-design-20261108.md) keeps natural
crowds separate from intentionally hard cases. The preview is for
reviewing geometry and plausible placement, not for estimating rates,
training a controller, or choosing a held-out split. Munich, Rotterdam
and Maastricht are the only maps in the preview. Berlin and Utrecht
were not loaded for this work.

## Slope gate before sampling

The checksum-verified [geometry gate](../outputs/previews/bench-users-v0/geometry/slope_gate.json)
and [geometry implementation notes](bench-users-v0-geometry-20261108.md)
record the native LoD2, OSM and bare-earth inputs. DTM sampling is bilinear
only when the full gradient stencil is valid. It never fills a no-data
cell. The graph uses OSM walkable segments, excludes mapped
bridge/tunnel/covered ways and building interiors, and checks ground at
one-metre or finer spacing. Placement and walking then recheck every
20 ms sample. Native roof heights are retained rather than derived from
the terrain.

| Map | Native LoD2 footprints intersecting the area | Retained street edges | Valid 1 m street samples | Local DTM slope p50 / p90 / p95 | BS/RIS pairs |
| --- | ---: | ---: | ---: | --- | ---: |
| Munich | 335 | 782 | 11,935 | 1.67° / 3.77° / 5.38° | 24 |
| Rotterdam | 193 | 532 | 6,984 | 2.37° / 5.10° / 6.53° | 24 |
| Maastricht | 248 | 522 | 5,412 | 2.24° / 4.90° / 6.41° | 24 |

The [generator's independent site and slope recheck](../outputs/previews/bench-users-v0/generator_slope_gate.json)
adds along-segment grade, ground validity at actual BS and RIS coordinates,
and OSM node eligibility for entrances, crossings, transit stops and
squares. The per-map geometry rejection counts were:

| Map | Bridge / tunnel / covered ways | Segment crossed building | Segment had no valid DTM stencil | Water |
| --- | ---: | ---: | ---: | ---: |
| Munich | 160 | 129 | 70 | 0 |
| Rotterdam | 68 | 56 | 285 | 0 |
| Maastricht | 4 | 14 | 360 | 1 |

The first column counts OSM ways, while the others count split segments;
they are not one denominator. **Maastricht passes the planned-site
placement check and stays in the preview.** Its Phase 0b full-square
33.25% no-data result remains true: this work uses only validated street
segments and placements and does not claim complete coverage. No map
swap was needed. The large tails of *local* DTM gradient include sharp
changes at retaining walls and edges; they are not the gradient along
every person's direction of motion or measurements of human behavior.

## Population, service and label interpretation

The natural draw uses weighted OSM hotspot clusters and background
singletons, a truncated Zipf group-size tail, and a separate uniform
2–4 person scheduler for the four BS antennas. It does not accept or
reject a case based on link classes or BS/RIS distance. The scheduler
keeps membership fixed within a clip, with everyone else still present.
Unserved people are not blockers. Clip lengths and walking speeds vary;
groups share a route with pauses and validated follower offsets. The
ordinary phone height is 1–1.7 m above local ground. The small elevated
class uses valid near-level native roof supports at least 3 m above the DTM, and the rare drone
class is 20–80 m AGL. Rooftop public access is not established by this
geometry, and bridge or terrace supports were not invented.

The case ID identifies the crowd; the frame table's `group_id` identifies
a cluster within that crowd. Each card links its complete Parquet table.
Absolute elevations are in the source datum (DHHN2016 for Munich, NAP for
the Dutch maps). Angle columns point from the user toward the BS or RIS:
azimuth is zero east and positive toward north, with elevation above the
horizontal. The figures show native footprints extruded to model heights
for readability; visibility uses the geometry rather than those drawn
prisms.

`los` is a clear finite segment, with the first blocker and source height
recorded where direct LOS fails. `reflection_only_candidate` means direct
LOS fails but a tested native wall supports a clear geometric single
bounce. The shortlist checks up to 32 wall triangles, so `blocked`
means no direct LOS and no candidate found in that bounded search. It
does not rule out other reflections, diffraction, or penetration. No
electromagnetic field, channel, rate, or human-body loss was calculated.
P4 remains open.

