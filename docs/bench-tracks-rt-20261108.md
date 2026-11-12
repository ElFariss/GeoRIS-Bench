# Track 4 — bounded native-scene RT smoke (declared 2026-11-08, WIB)

## Frozen smoke design

This is a development-only pipeline check. It is not a channel-generation run
for the benchmark and cannot support rate, optimizer, or generalization claims.
Before opening or tracing inputs, use exactly the v1 preview cases
`01-munich-natural` and `39-maastricht-natural`, one from each development
map. Use their recorded BS and RIS centers, with a vertical 8x8, 8x16 and
16x16 NEC-lattice probe plane centered at the RIS: 28 mm horizontal pitch,
30 mm vertical pitch, oriented toward the BS. Record this local orientation
convention as a smoke approximation because the v1 case file has no wall-normal
field. Probe both polarizations at 5.53 GHz.

Build RF-only native LoD2 meshes from the pinned Phase 0b Munich CityGML tile
and the pinned Phase 0b Maastricht 3DBAG v2025.09.03 tiles. Keep terrain out
of the RF scene. Use material classes already registered by the corresponding
Phase 0b builders. Do not fit material parameters. Check all RF mesh extents
and probe positions before tracing. Do not download or open held-out maps.

For each scene and lattice size, run one bounded PathSolver call with LOS,
specular reflection, and diffraction enabled; transmission/refraction disabled;
maximum interaction depth 2; 10,000 samples per source; seed 7; no synthetic
array approximation. Add the actual frame-0 served UE coordinates from each
case Parquet to each receiver batch. Save nonzero path coefficients, delays,
departure and arrival angles, retaining the receiver and path dimensions, to
the isolated host workspace. Also record receiver masks from LOS-only,
reflection-only, and diffraction-only runs at 64 elements; derive disjoint
LOS-present, reflection-only, diffraction-only, mixed, and fully blocked
counts from those masks. These categories are diagnostics; a smoke case need
not contain every class.

Stop before any solve if neither 3060-3 GPU has at least 2 GiB free. Use only
3060-3, pin CPU work away from logical CPUs 5 and 11, and use a separate
`/var/tmp/bench-tracks-rt-20261108` workspace. The only artifact copied back
is a compact JSON summary with source hashes, versions, counts, timings, peak
RSS, GPU memory and pass/fail checks. Raw assets, meshes and path arrays stay
on the host. The production ray-tracing run remains disabled.

## Results

The Munich builder succeeded for case `01-munich-natural`: 204 LoD2 buildings
were intersected with the local scene region and 23,421 semantic triangles
were produced (4,158 ground, 6,947 roof, 12,316 wall). The first Sionna run
reached its Munich combined RIS-probe traces, but the earlier script reused a
single path-output filename across lattice sizes and did not include the served
users. Preserve it as a diagnostic only under
`/var/tmp/bench-tracks-rt-20261108/diagnostic_attempt1/`; it is **not accepted
as Track 4 evidence**, and its FAILED marker records the subsequent Maastricht
geometry failure. Its partial path file stays on 3060-3.

The corrected script includes the two frame-0 served UEs for Munich (IDs 3 and
10) and three for Maastricht (IDs 7, 40 and 50), unique path filenames per
lattice, and receiver-specific indexing based on the installed Sionna Paths
tensor layouts. It was syntax-checked locally. CPU-only geometry preflight on
the cached, checksum-verified pinned inputs then failed closed on the
Maastricht native roof `NL.IMBAG.Pand.0935100000016380-0/RoofSurface`. After
face-local coordinate recentering fixed the large-CRS closure-tolerance issue,
that roof still had a measured 1.90543 m plane deviation and an invalid
dominant-plane polygon. The face contains more than three unique points, so
the declared zero-area cleanup does not apply. No triangulation that preserves
the source geometry was verified. Do not omit or replace the roof: Maastricht
has no smoke result.

The first wrapper attempt also stopped before Sionna because `/bin/sh` did not
expand a brace expression while creating log directories. That attempt did
not invoke the solver. After that, the Munich-only portion of
`diagnostic_attempt1` did invoke the solver. The corrected Maastricht
preflights were CPU-only; no second GPU run was launched. Both 3060-3 GPUs were
idle at the final status check and there was no active Track 4 process. No OSM
data is required by this RF-scene smoke.

Current status check:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  <3060-3> \
  'cat /var/tmp/bench-tracks-rt-20261108/diagnostic_attempt1/FAILED; test ! -e /var/tmp/bench-tracks-rt-20261108/attempt2/RUNNING; nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader'
```

The updated runtime code is `scripts/run_bench_rt_preview_smoke.py`, staged
with SHA-256 `141376715f7569139f88075697006e9d72c551e8a5fbeca27cf8e58387c1c255`.
The case manifest is selected from the v1 preview Parquets and has SHA-256
`c26e67a4198ef860710e5ed4ec2d7eef0fa979b15f1af855e53c8b66e844ed01`.
The first run used earlier staged runtime code
`aaf94430d2f1c28e5c9fb6bb1aff4423430c4f71c55061fa57d38a7a47c63dcc`; its
path output is preserved only for diagnosis. The corrected smoke status is
**blocked at native Maastricht geometry validation**. A safe resume requires a
geometry-preserving triangulation for the reported native roof, followed by a
successful CPU build-only check before launching a new isolated GPU attempt.
Production tracing remains disabled.
