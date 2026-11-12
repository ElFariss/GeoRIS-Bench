# Production pilot packaging and labels — 2026-11-09

## Scope and input contract

This package stage consumes the lead-owned pilot manifest, channel HDF5 and
JSON metadata; it does not trace rays or read raw per-path archives. The pilot
output is isolated under `outputs/runs/bench_production_pilot_20261109/package/`.
The CLI refuses to overwrite a non-empty output directory and takes an
exclusive sibling `.stage.lock` while writing. It writes `DONE` only after the
label HDF5, trajectories, split manifests and provenance hashes are complete.

Expected channel groups are
`/<city>/<array numeric>/<case>/<frame numeric>` and contain
`H_direct`, `C_in`, `C_out`, `Z_effective`, `loads`, `precoder`; `noise` is a
positive scalar group attribute. Here `Z_effective` is the terminated Schur
block **A**, not bare `Z_RIS`. Approved bare `Z_RIS` remains separately in the
physics archive. `loads` contains the two shared complex OFF/ON impedances;
`precoder` is the fixed channel precoder. Complex arrays may be native HDF5
complex, compound `r/i`, or a trailing real/imaginary pair.

The metadata JSON is flat and keyed by the channel group path (with or without
the leading slash). It provides `users` as a list of records with `user_id`,
`bs_user` and `ris_user`, plus `link_classes.bs_ris`. Missing link
classes remain `unknown`; preview labels are not used to infer ray-tracing
classes. The pilot manifest contains `cases[]` with `id`, `map_id` and
`frames[]`; each frame has `frame`, optional `time_s`, and `users[]` with
`user_id`, `xyz`, and optional `velocity_xyz`.

## Labels and splits

For each frame and array, the stage makes two deterministic parent states:
all-off and a maximum-four-step positive single-flip greedy state (stop when
no individual flip improves the minimum-user rate). It computes every single
flip gain using `HybridCoupledNetwork.moves`, then exact pair gains using
`moves_pairs`. The stored identity is

`single_gain[i] = score(toggle(i)) - score(parent)`

`pair_gain[i,j] = score(toggle(i,j)) - score(parent)`

`synergy[i,j] = pair_gain[i,j] - single_gain[i] - single_gain[j]`.

The case manifest normally omits `array`, causing each case/frame to use all
three array sizes. `--partial-arrays` is only for diagnostics; it marks the
package `partial_diagnostic_only` and cannot satisfy the pilot gate. All
unordered pairs are stored for 64 and 128 elements. At 256, 4096 pairs are
sampled without replacement with a stable hash-derived seed; the all-off
parent is a full-pair audit and the greedy parent is sampled. Pair index and
seed are recorded per parent. Labels live in separate
`optimization/single_flip_gains.h5`, `optimization/pair_flip_gains.h5` and
`optimization/synergy.h5` files; `trajectories.parquet` has one row per
frame, served user and array, with positions, velocity, channel-derived link
classes and the development split. Label groups are written once per
array-frame, independent of the number of users. Splits are expressly development
splits: deterministic trajectory partitioning within each map and leave-map-
out cross-validation among the development maps. A one-trajectory map is
marked as insufficient for a within-map train/validation split. No test map is
generated or designated.

Example invocation once the lead-owned pilot channel files are ready:

```sh
PYTHONPATH=src:scripts python3 scripts/package_bench_production.py \
  --pilot-manifest outputs/runs/bench_production_pilot_20261109/pilot_cases.json \
  --channels outputs/runs/bench_production_pilot_20261109/channels/channel_realizations.h5 \
  --metadata outputs/runs/bench_production_pilot_20261109/channels/metadata.json \
  --output outputs/runs/bench_production_pilot_20261109/package
```

The CLI calls `atomwise_ris.bench.production_freeze.verify_freeze()` before
staging. The input metadata sidecar must be named `metadata.json` (if omitted,
the CLI looks for that filename beside the channel HDF5). A checked copy of the
derived channel file is included at
`package/channels/channel_realizations.h5` alongside `metadata.json`; raw
per-path archives are excluded.
The example loader at
`bench/CoupledRIS-Bench/examples/load_dataset.py` is copied to the package and reconstructs the exact
`HybridCoupledNetwork` from that packaged channel HDF5 and checks stored
base-rate readback within 1e-10. The package role is `pilot_development` only
when all three array sizes are present.

## Synthetic validation

`tests/test_bench_production_package.py` verifies deterministic unique pair
indices, complete 64-element coverage, synergy arithmetic, a synthetic
partial-diagnostic package with a 64-element operator, Parquet and split artifacts, and loader
rate round-trip. Command:

```sh
PYTHONPATH=src:scripts uv run --python 3.12 \
  --with 'h5py>=3.14,<4' --with 'pandas>=2.2,<3.1' --with 'pyarrow>=20,<25' \
  -- env PYTHONPATH=src:scripts python -m unittest discover -s tests -p 'test_bench_production_package.py'
```

Result: **2 focused tests passed**. Synthetic round-trip absolute error was
within 1e-10. No pilot channels were available at implementation time, so no
real label production, runtime/storage projection, or pilot gate result is
claimed here. The stage deliberately fails closed on missing channel keys,
unsupported load shapes, non-finite arrays, forbidden maps, and output reuse.

## Files

- `scripts/package_bench_production.py`
- `bench/CoupledRIS-Bench/examples/load_dataset.py`
- `tests/test_bench_production_package.py`
- `docs/bench-production-package-20261109.md`
