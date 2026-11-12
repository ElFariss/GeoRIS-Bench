# Track 3 — RIS-only `Z_RIS` extraction

**Criteria declared:** 2026-11-08 18:16 WIB, before any new extraction in this track.

## Scope and geometry

Measure the PEC thin-wire RIS-only port impedance at 5.53 GHz for 64 and 128
elements, with 81 and 161 segments per wire. Each matrix is an independently
computed `Z_RIS` with no BS/UE ports and no optimizer. Arrays lie in the yz
plane, are centered at x=0 and z=1 m, use z-directed 24 mm wires with 50 µm
radius, 28 mm horizontal pitch and 30 mm vertical pitch. Layouts are 8×8,
8×16 and (for the prepared 256-element case) 16×16. This is a PEC wire-array
baseline, not an OpenSourceRIS PCB model and not an ON/OFF meta-atom response.

Write each run's `Z_RIS_ohm` complex128 matrix into `impedance.h5`, plus
`metrics.json`, runtime, peak RSS, code hash and file hash. Store the two-mesh
comparison as `mesh_validation.json`. Compact geometry and run metadata remain
in this repository; matrix artifacts may remain on rx6600 if transfer or local
storage becomes undesirable. Do not use held-out maps or channel/test data.

## Predeclared numerical criteria

- **Reciprocity:** relative Frobenius residual `||Z-Z^T||F / ||Z||F <= 1e-6`.
- **Passivity:** minimum eigenvalue of `(Z+Z^H)/2` must be at least
  `-1e-6 * max(1 ohm, ||(Z+Z^H)/2||2)`. Also report the largest singular value
  of the 50-ohm scattering matrix; it must be `<= 1+1e-6`.
- **Conditioning:** report the 2-norm condition number; it is diagnostic, with
  no pass/fail threshold because this is not itself a solver-accuracy bound.
- **Mesh sensitivity:** for each array, compare 81 vs 161 using relative
  Frobenius difference `||Z161-Z81||F/||Z161||F` and maximum entry error
  normalized by `||Z161||F`. The provisional engineering criterion is at most
  5% and 10%, respectively. If either is exceeded, label the matrix mesh
  sensitive; do not loosen the thresholds or call 161-segment NEC truth.

These are numerical checks for this model and discretization, not validation
against a measured surface. Report the values even on failure. All arrays use
identical wire dimensions, pitch and coordinate conventions across meshes.

## Run status and evidence

No new extraction had started when these criteria were written. Historical
128×81 evidence is recorded in `docs/hybrid-track2-20261108.md`; it measured
reciprocity only and did not save the matrix, so it cannot replace the new
HDF5-producing run or mesh comparison.

## Results

### N=64, completed 2026-11-08

The rx6600 durable run produced the 8x8 matrices at both meshes. I checked the
HDF5 readback (complex matrix, finite values, 64x64, 5.53 GHz, stated segment
count and ohm convention), metric JSON, and SHA-256s. The arrays satisfy the
declared reciprocity, passivity, and scattering checks:

| Segments/wire | Wall time | Peak RSS | Reciprocity residual | Min Hermitian eigenvalue (ohm) | max singular value of S | cond2 | HDF5 SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 81 | 28.76 s | 0.93 GB | 4.55e-8 | 0.5421 | 0.98663 | 13.62 | `abdedebd18869f8fb6c79a4466c492e71012a5ea8481a31ff9e866621d9829f6` |
| 161 | 189.56 s | 3.50 GB | 8.00e-9 | 0.5401 | 0.98664 | 13.70 | `b101605e7a9161bb7b0ce69466ad8af6a9f1b0d0825ec937473458801f6cd91b` |

The 81-to-161 relative Frobenius difference is 0.6279%; max-entry difference
normalized by the fine-mesh Frobenius norm is 0.0306%. Both satisfy the
declared 5% and 10% mesh-sensitivity limits. This is only discretization
sensitivity for an ideal PEC wire array, not measured hardware accuracy.
The assembled matrices and evidence are in
`bench/CoupledRIS-Bench/physics/array_64/`; combined HDF5 SHA-256 is
`bcb059ba507a2a6a579a81bef95fa8aacd90ac19180570da402125ba750b38a3`.
The comparison implementation initially had a tuple comparison bug that
rejected every legitimate 81/161 pair. Fixed it, copied the corrected script
to the rx6600 task workspace, and reran the comparison. A small helper now
allows the compatibility check to be regression-tested without h5py; the
final script SHA-256 locally and remotely is
`5aa64bf87d5665eac26c79908ba459665e7d07c6d7810b930074a4d1a9fa45f4`.
Metric JSON SHA-256s are `36307aaa937044ddaeea5d96a0c756966f709085a00c5b4269fdfd8a5ef08677`
(81) and `5446e26a9333e04dc6548a82f13504682b2513a06e6f9547bc26c671f9428617`
(161).

### N=128 in progress; N=256 not launched

After verifying no existing extraction or tmux session, launched only the
missing 128-element series on rx6600 in tmux session `zris128`. It reuses the
same pinned source/runtime directory and writes resumable, per-array output to
`/var/tmp/bench-tracks-zris-20261108/physics/array_128/`. The 81-segment run
completed at 19:13:44 WIB (runtime 233 s; see remote metrics). The 161-segment
run started immediately afterward and was active at 19:14 WIB. The wrapper skips already complete matrices, writes
`128_DONE` only after both matrices and mesh comparison finish, and writes
`128_FAILED` on failure. Check with:

```sh
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes <rx6600> \
  'tmux ls; cd /var/tmp/bench-tracks-zris-20261108; cat 128_RUNNING 128_DONE 128_FAILED 2>/dev/null; tail -20 logs/128_resume.log'
```

The completed 128×81 matrix passes the declared reciprocity, passivity and
scattering checks (6.95e-8, minimum Hermitian eigenvalue 0.0273 ohm, and
maximum S singular value 0.99933). Its 128×128 matrix SHA-256 is
`458f195cea6a3eb09ce9f8f9d692f75d54bdafd85bd33f724d82d9ba6679a564`;
its metrics JSON is preserved locally at
`bench/CoupledRIS-Bench/physics/array_128/81/metrics.json` with SHA-256
`ad680094fc145e56eae81ece3846390f431c7144e1ff68a09f504ec1816825fd`.

The authorized compute-status probe reports rx6600 available, but the A100
helper could not create its temporary config under the read-only home sandbox.
Thus no A100 job was started and the 256-element case remains unrun. It needs
an available permitted A100 launch path and an initial resource preflight; do
not infer an extraction from the geometry metadata.

The resumed lead session could check Faris using scratch state and the
existing credential references (no credentials copied). The server reported
an A100, but its CPU solver cgroup had a **16 GiB** hard RAM limit and was
already using **17,163,595,776 bytes** of 17,179,869,184. The approximately
27 GB fine-mesh extraction does not fit; neither mesh was launched under
that load. A100 VRAM and host-wide free memory do not increase the NEC
process's cgroup limit. No existing workload was modified.
