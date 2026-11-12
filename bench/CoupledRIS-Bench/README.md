# CoupledRIS-Bench

This directory holds the [Phase 0 source audit](LICENSES/README.md), its
[provenance record](provenance.json), and the
[Phase 0b proposed map licenses](LICENSES/phase0b-map-sources.md).
The [five map areas](../../configs/bench/phase0b-area-proposals-20261108.json)
were approved for the benchmark design. This directory does not contain
released channels, optimizer labels, scene meshes, or frozen splits.
Release terms remain for review.

The planned format is HDF5 for complex channels and gain arrays, Parquet for
per-frame trajectory records, and JSON for source provenance, geometry,
validation and splits. Install the optional format dependencies with
`uv sync --locked --extra dataset` from the repository root.

See the [Phase 0 audit](../../docs/bench-phase0-audit-20261108.md) for the
Munich finding and validation. See the
[Phase 0b feasibility record](../../docs/bench-phase0b-feasibility-20261108.md)
for the proposed areas, terrain coverage and scene-builder feasibility.

## Ground model and gap filling

The v1.1 development preview uses the native Bavaria DGM1 raster in Munich
and the AHN 0.5 m terrain raster in Rotterdam and Maastricht. Bavarian LoD2
and 3DBAG supply the native building footprints and base heights. OSM water,
bridge, covered, tunnel and underground features define conservative masks.
The exact source hashes are recorded in the preview source manifest. The
terrain is used to place users and equipment; the footprint overlay is not
an RF scattering surface.

The overlay leaves every valid provider DTM sample unchanged. Within each
eligible building, a linear interpolator triangulates valid exterior samples
and estimates missing cells only inside that footprint. It never extrapolates
or uses a previous fill as a source. Areas over 2,500 m² or 80 m across, cells
more than 12 m from a valid source, protected corridors, and estimates more
than 2.5 m from the native building base are excluded from the used fills.
The full [protocol](../../docs/bench-tracks-ground-v1.1-20261108.md) specifies
the masks, held-out reconstruction checks and support rules.

Raster codes are `0` for no data, `1` for `measured` (unmodified provider
DTM, which can include provider interpolation), and `2` for
`interpolated_under_footprint`. Building-mounted objects use a same-building
cell within 1.5 m, or record `lod2_native_ground_fallback` with no raster
cell. Pedestrians, drones, walking segments and ground masts use only the
original DTM and its full valid 4×4 stencil. The stored support provenance
distinguishes interpolation from a native model fallback.

Munich validation masks and reconstructs original valid cells under eligible
buildings. That measures agreement with the published terrain raster, which
may already estimate heights under buildings. The separate native-base
comparison reports errors on all three development maps, with large
footprints, courtyards, steep sites, canal-side buildings and protected
transport corridors shown separately. Neither source can resolve hidden
terraces, excavations, underground infrastructure, or local foundation
steps reliably. Consult the four track records (`docs/bench-tracks-*-20261108.md`)
for the measured status of the preview and the separate propagation checks.

The development ground audit passed the aggregate Munich gate: it recovered
37,552 of 56,659 eligible masked cells (66.28%), with 0.017 m median absolute
error, 0.210 m at the 95th percentile, and 5.942 m maximum. The steep subset
had 3.786 m error at the 95th percentile. Munich had no actual missing cells
inside its modeled footprints. Raw estimates for missing Rotterdam cells
had 0.214 m median / 1.658 m 95th-percentile absolute disagreement with the
native building base; 127,366 cells passed the guard. Maastricht had
0.398 m / 3.419 m disagreement and retained 124,428 cells. Its steep subset
reached 4.304 m at the 95th percentile before the guard. These are
development results; the accepted-cell agreement is conditioned on the
2.5 m guard and is not independent validation.

The [development preview](../../outputs/previews/bench-users-v1.1/index.md)
passed its separate readback: 50 cases and 5,000 anonymous draws. The source
manifest records the recovered OSM snapshots; LoD2 and DTM hashes were unchanged
from v1. The subsequent production freeze is documented below.

## Ray propagation and wire coupling

The user approved the v1.1 generator and froze its sources and validation
artifacts. The [freeze](../../docs/bench-users-v1.1-freeze-20261109.md) is
verified before production stages. The geometric preview `bs_link` remains
a placement diagnostic; production link classes are derived from ray paths.

All three development maps (`munich`, `rotterdam`, `maastricht`) follow a
unified [LoD2 mesh-repair and preflight policy](../../docs/bench-production-mesh-policy-20261109.md)
before ray tracing: coordinates are triangulated in a per-face local frame
and shifted by the scene origin once; every triangle is oriented to its
parent face's outward 3D Newell normal (`100%` outward-normal share);
zero-area faces and triangles are dropped; and self-intersecting or
non-planar LoD2 faces are deterministically repaired via `shapely.make_valid`
in the dominant projection plane with 3D edge/plane lifting and accepted
only if their emitted 3D area is within `1.0%` of the source face 3D Newell
area (dropped otherwise, with a fail-closed `0.5%` per-class dropped-area
stop rule and `0.1%` kept-area conservation check). Preflight also verifies
that mesh world $z$ bounds lie within the frozen terrain range plus maximum
building height and that building `GroundSurface` bases match the terrain or
ground-fill reference within tolerance.

[P4 is accepted for conversion](../../docs/p4-closure-20261109.md): registered
Munich LOS and controlled PEC-plane reflection met the unchanged no-fit
polarized field and port criteria. Sionna supplies urban propagation, while
NEC supplies coupled wire currents and the independent open-gap correction
carried into the direct channel. This validates the conversion in those
reference cases; native urban materials, city-scale full-wave accuracy and
measured hardware are outside the claim. The 256-element model uses only the
81-segment mesh; the 64- and 128-element mesh comparisons do not certify
convergence for 256.
