# Phase 0b Munich fallback smoke — declared 2026-11-08 before tracing

Maastricht has 33.25% no-data in the audited AHN4 tile, above the fixed
1% limit. Its proposed 100 m / 10 m-lattice diagnostic returned no
qualifying patch. Keep both verdicts on record. The brief allows one small
development scene smoke. Use a separately declared Munich patch instead,
with the same maximum 1% no-data and minimum 5 m range; neither limit
changes. No channel or score was used to choose it.

- Use the 100 m square `[691383, 5334880, 691483, 5334980]` m (UTM32,
  EPSG:25832; DHHN2016/NHN, EPSG:7837), the first qualifying 100 m patch
  on a 10 m grid in the audited Munich development square, sorted by
  increasing easting then northing. Select LoD2 building geometry from a
  35 m halo around that patch. Use the provider-hashed `690_5334.gml`
  and `691_5334.tif` audited already. Build a native 1 m height mesh for
  the patch; do not include it as an RF shape.
- Keep native roof, wall, and ground semantic faces as distinct ITU
  proxy classes (concrete roof, brick wall, concrete building underside).
  Preserve concavity and holes by constrained triangulation in the
  dominant face projection, reconstruct 3D vertices from the originals,
  and compare triangle areas with the source polygon in that projection.
  Zero-area rings or faces may be skipped and counted; other invalid
  inputs fail closed. Do not fit material properties.
- Select the first two 5 m grid cell centres in deterministic easting /
  northing order whose distance is 15–35 m, whose entire horizontal
  segment is at least 1 m outside the union of native ground and
  projected roof footprints, and whose straight segment at the
  sampled DTM + 1.5 m clears the DTM by at least 0.2 m. Do not use RF
  outcomes to choose another pair. Test a valid in-footprint control and
  the two outdoor points using vertical intersections. The two probes
  must reproduce the sampled ground in the height-only mesh to 2 cm,
  with no overhead building hit in the RF scene. The in-footprint
  control must hit a building roof above terrain.
- Run exactly one 5.53 GHz Sionna RT 2.0.1 explicit LOS-only check on
  3060-1, two polarization ports per end and no terrain RF interaction.
  Pass requires one positive finite-delay path and nonzero finite
  co-polar coefficients. This is a load and plumbing test only.
- Record geometry build/load/solve wall time and process peak RSS, and
  GPU capacity before/after with concurrent use noted. Stop before the
  trace if no device has at least 2 GiB free. Do not interfere with other
  jobs. Keep the raw assets, height mesh, RF meshes and any path data on
  3060-1. Copy a compact summary only. No held-out scene or production
  channels, placements, results or calibration may be produced here.

## Reporting recovery, declared before the replay

Two initial invocations stopped before a solve because Sionna normalizes
both the mesh and material ID prefixes into one namespace. The corrected
invocation reached and passed the fixed one-link numerical assertions,
but its summary write stopped on an unavailable Scene metadata property.
Preserve all failures. Replay that same single link once only to recover
the summary and timing, after replacing the metadata accessor with the
public Mitsuba shape inventory. No input, endpoint, threshold, solver
setting, split, or selection rule changes.
