# Round 2 — RIS-only impedance, 2026-11-08 WIB

## Declared run and acceptance (before the 256-element launch)

Reuse the lattice, frequency and numerical checks in
[bench-tracks-zris-20261108.md](bench-tracks-zris-20261108.md). Both matrices
must be finite complex128 arrays with the expected geometry metadata. Check
reciprocity residual ≤1e-6, minimum Hermitian resistance eigenvalue ≥
−1e-6 × max(1 Ω, Hermitian spectral norm), and 50 Ω scattering maximum
singular value ≤1+1e-6. Report conditioning without a cutoff. Compare 81 vs
161 segments with the same relative Frobenius (≤5%) and normalized maximum
entry (≤10%) thresholds. Preserve measured failures and do not replace the
mesh comparison with a hardware calibration claim.

No new 128-element solve: collect the completed 81/161 artifacts on rx6600,
recompute metrics and mesh differences from their bytes, and package locally.
For 256 elements use only the project account on 3060-3, the existing PyNEC venv and a
separate workspace, sequential meshes and BLAS threads set to one. Pin the
process to CPUs `0-4,6-10`; CPUs 5 and 11 are excluded. No timing job may
overlap. The high-memory 161 mesh and corrected RT smoke share an advisory
lock on this host. The 81 mesh may overlap bounded RT, subject to its own
available memory check.

The measured 128×161 peak RSS was 13,746,847,744 bytes. A quadratic estimate
for 256×161 is 54,987,390,976 bytes (51.21 GiB). This is a planning estimate,
not a measurement on 3060-3. Require at least 58 GiB usable RAM headroom before
161 and cap its virtual address space at 53.5 GiB. For 81 require 18 GiB
headroom and cap at 16 GiB. Check the minimum of host MemAvailable and any
finite cgroup memory headroom. If a stage cannot fit or fails under its cap,
preserve the successful stage and stop. Do not adjust host resources or other
jobs. Timeouts are 2 h for 81 and 8 h for 161, not timing benchmarks.

The durable entry point is `scripts/run_bench_zris_256_round2.sh`; workspace
`/var/tmp/atomwise-bench-round2-zris256`, tmux `bench-round2-zris256`. It
records resource checks and per-stage logs, writes DONE only after HDF5,
mesh comparison and package completion, and records FAILED on nonzero exit.
The same runner skips an existing extraction only after verifying its HDF5
hash against completed metrics. It never reuses mismatched bytes. Run
manifests and source hashes live in the task workspace.

## 128 elements: completed, collected and independently verified

rx6600's existing `128_DONE` was written at 19:38:52 WIB. No mesh was
rerun. Both matrices were read back and packaged into
`bench/CoupledRIS-Bench/physics/array_128/impedance.h5`, SHA-256
`c89280f3487ac48a9de95ce040f7442f88429529dcc39dcc2c1309ee8d812afb`.
`scripts/validate_bench_zris_bundle.py` independently verified both numerical
checks and that each packed matrix equals its own extraction. The 81/161
relative Frobenius difference is **0.87764%** and maximum normalized entry
difference is **0.02295%**. Both meshes passed reciprocity, passivity and
scattering contraction. Fine-mesh elapsed time was 1,508 s; peak RSS
13.75 GB decimal. The per-mesh metrics and comparison hashes are recorded
in `numerical_validation.json`.

## 256 elements: durable run receipt and collection

The 3060-3 workspace was staged with checked source and dependency hashes:
`STAGED_SHA256SUMS` SHA-256
`d4e835728a5401a0449b0890e25d3e080f85e9cca93d9cc48b2c0b7f3f824fa7`.
The pinned `~/projects/atom-wise-ris/.venv-em` imported PyNEC and NumPy
2.2.6. An isolated `deps/` directory supplies the checked h5py 3.16.0 wheel
for CPython 3.12. No existing environment or Paper 1 workspace was changed.

tmux `bench-round2-zris256` started at 19:50:36 WIB. A new SSH connection
verified the `RUNNING` marker, logged affinity `0-4,6-10`, 63.97 GB
available, and the 81-segment PyNEC process active (PID 1931686). The
161-segment extraction is queued inside the same wrapper, subject to the
advisory lock and new headroom check. `DONE` means both HDF5 artifacts,
packaging, and a numerical verdict were written. It does not imply the
verdict is `pass`; read `numerical_validation.json` for that.

**Bounded retry declared after the initial `mesh_81` failure:** at 19:51:24 WIB
PyNEC 2.3.4's first `xq_card` returned `RuntimeError: Unknown exception`, with
no HDF5 or metrics written. The prior tmux is gone and its evidence stays in
the original workspace. The `_PyNEC` binary SHA-256 is identical to rx6600
(`c0fa8ee0e4b847ad28cda8be18fc4e025545db8415c964221ee2ba79e16dbdd1`).
Allocation failure is a hypothesis, not a diagnosis. One 81-segment retry in
the `-attempt2` workspace uses 24 GiB address-space cap, at least 32 GiB
headroom, and records failed allocation syscalls and GNU time's RSS.
If 81 fails again, stop and preserve the error. If it completes, require its
observed RSS × (161/81)² to fit the unchanged 53.5 GiB fine-stage address-space
cap before even trying the 161 solve. The physical parameters, numerical
thresholds and affinity are unchanged.

The retry launched at **20:03:58 WIB** in `bench-round2-zris256-attempt2`, with
64,016,977,920 bytes of available memory. The checked staging manifest SHA-256
is `b047e9728538d561696ba797ca6d4683e841685c21088662bb328655285091c5`.
It uses `env ZRIS_DIAGNOSTIC_RETRY=1 taskset -c 0-4,6-10 bash run.sh
/var/tmp/atomwise-bench-round2-zris256-attempt2` from that workspace. New output
and state checks below refer to the retry, not the archived first failure.

A process readback after the original failure point found PID 1933886 active
with `VmPeak=17035436 kB` and `VmHWM=13592112 kB`, under the 24 GiB cap.
The observed peak address space already exceeds the original 16 GiB cap
(`16777216 kB`), supporting the resource diagnosis. This is an interim
readback, not the final matrix or a guarantee that the fine solver fits.
The projected virtual footprint may exceed the fine-stage cap even if the
resident estimate passes; preserve a complete coarse matrix on that failure.
A trivial `systemd-run --user --scope` probe on 3060-3 verified that the
kernel applies `MemoryMax` and `MemorySwapMax` (readback 1,073,741,824 and 0
bytes). If later evidence isolates an address-space-only failure, this is a
possible resident-memory guard for a deliberately revised fine run. It was
not used to start a second fine solver or alter the active wrapper.

```sh
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  <3060-3> \
  'cd /var/tmp/atomwise-bench-round2-zris256-attempt2; tmux ls; cat RUNNING DONE FAILED 2>/dev/null; tail -12 logs/runner.log'
```

After `DONE`, use the same connection to run `sha256sum -c
output_sha256sums` in the workspace and transfer only the completed
`physics/array_256/` bundle and `output_sha256sums`. Run the bundle validator
against a staging copy before copying verified files into the local
`bench/CoupledRIS-Bench/physics/array_256/` directory. Keep the local geometry
if its hash agrees. If `FAILED`, inspect the named stage and matching log.
The runner will not use partial output; preserve it separately before any
intentional retry. No automatic agent reentry is configured.

Example collection, from the repository with `h5py` and NumPy available:

```bash
set -euo pipefail
mkdir -p .cache
stage=$(mktemp -d .cache/round2-zris256-collect.XXXXXX)
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  <3060-3> \
  'cd /var/tmp/atomwise-bench-round2-zris256-attempt2 && test -s DONE && sha256sum -c --quiet output_sha256sums && tar -cf - physics/array_256 output_sha256sums DONE' \
  > "$stage/evidence.tar"
tar -xf "$stage/evidence.tar" -C "$stage"
sed 's@  /var/tmp/atomwise-bench-round2-zris256-attempt2/@  @' \
  "$stage/output_sha256sums" > "$stage/local_sha256sums"
(cd "$stage" && sha256sum -c local_sha256sums)
OPENBLAS_NUM_THREADS=1 python3 scripts/validate_bench_zris_bundle.py \
  --array-dir "$stage/physics/array_256"
diff -q "$stage/physics/array_256/geometry.json" \
  bench/CoupledRIS-Bench/physics/array_256/geometry.json
cp -an "$stage/physics/array_256/." bench/CoupledRIS-Bench/physics/array_256/
diff -qr "$stage/physics/array_256" bench/CoupledRIS-Bench/physics/array_256
```
