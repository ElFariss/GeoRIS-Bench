# Full-production run layout and contract (lead, 2026-11-09)

This is the contract shared by the T6b sub-tasks (trace, channels, package/gate) and T7 (launch). Each task implements its own part and must read and write exactly these paths and fields. Shared helpers live in `src/atomwise_ris/bench/production_run.py`, which the lead owns; import them, do not copy them:
- `PRODUCTION_SCOPES`, `DEVELOPMENT_MAPS`, `ARRAY_SIZES`;
- `stop_rule_exceeded`, `atomic_write_json`, `cases_by_map`, `check_cpu_set`.

## Scopes
- **Registered production scopes**, both validated by `registered_manifest()` in `src/atomwise_ris/bench/production_pilot.py`:
  - `development_full`: the natural manifest `outputs/runs/bench_production_full_20261109/cases.json`, 360 cases.
  - `development_challenge`: `outputs/runs/bench_production_full_20261109/challenge_cases.json`, 69 cases. Its cases carry `kind == "challenge"` and `scenario`.
- **Each scope is a separate run,** with its own trace output, channel output, package and gate. The two are never merged.
- **Pilot scopes** (`two_map_pilot`, `three_map_pilot2`) keep their current behaviour and code paths unchanged.
- **Every case has 16 frames** (`case["frames"]`, each with `frame`, `time_s` and `users[]`).

## Trace (stage 1): `scripts/run_bench_production_trace.py --production`
- **Arguments:**
  - `--root TRACE_ROOT`: raw sources and scenes;
  - `--out TRACE_OUT`: one per scope;
  - `--maps a,b`: optional filter, default all three;
  - `--gpu-id {0,1}`.
- **Scenes:** one scene per case, at `TRACE_ROOT/scenes/<map>/<case_id>/scene.xml`, with `scene_info.json` beside it. The crop anchors and the endpoint preflight use the BS, the RIS and **all 16 frames'** users.
- **Per-case outputs:**
  - `TRACE_OUT/<map>/n<N>/<case_id>/frame-<FFFF>.npz`: one per frame and array. `FFFF` is the original frame index, zero-padded to 4.
  - `TRACE_OUT/<map>/cases/<case_id>.json`: the **case record**, written atomically and last. Its existence plus verified hashes means the case is traced. Fields:
    `{"status":"pass","case_id","map_id","run_scope","case_manifest_sha256","scene_sha256","geometry":{...scene info incl. mesh_audit/preflight_checks and endpoint_preflight...},"records":[{"case_id","map_id","frame","array_elements","sha256","bytes","elapsed_seconds","solver_elapsed_seconds","export_elapsed_seconds",...}]}`.
    There are 48 records per case (16 frames × 3 arrays).
  - `TRACE_OUT/<map>/cases/<case_id>.FAILED.json`: `{"case_id","map_id","stage","error","time_utc"}`.
- **Per-map markers:**
  - `TRACE_OUT/<map>/trace_manifest.json` plus `TRACE_OUT/<map>/DONE`: written once every case of the map has a record or a FAILED file, and the stop rule is not exceeded. The manifest is `{"status":"complete","run_scope","case_manifest_sha256","map_id","cases":[all records of passed cases],"failed_cases":[ids]}`.
  - `TRACE_OUT/<map>/FAILED`: written as soon as `stop_rule_exceeded(failed, map_case_count)`. The map then stops.
- **Resume:**
  - A case with a valid record (hashes verified) is skipped.
  - A case with a FAILED file is skipped, unless `--retry-failed` is given (a deliberate lead decision; the old FAILED file is renamed `.FAILED.prev.json`).
  - A map with `DONE` or `FAILED` is skipped.

## Channels (stage 2): `scripts/build_bench_production_channels.py --production`
- **Arguments:** `--manifest`, `--trace-out TRACE_OUT`, `--out CHANNEL_OUT`, `--physics`, `--worker-id K`, `--arrays 128,256,64` (this worker's ordered array preference), `--cpus 0,6`.
- **Work unit:** one (map, array, case).
  - It is eligible once the trace case record exists and verifies.
  - It is skipped if done (valid shard + sidecar) or failed.
  - The worker builds the NEC basis once per array, and only when it has an eligible unit for that array.
- **Claiming:** a non-blocking `fcntl.flock` on `CHANNEL_OUT/claims/<map>/<N>/<case_id>.lock`, held while the unit is processed. A crash releases the lock, so there are no stale claims.
- **Per-unit outputs:**
  - `CHANNEL_OUT/<map>/cases/<case_id>/array_<N>.h5`: the groups are keyed `<map>/<N>/<case_id>/<frame>`, with the same datasets and attrs as the pilot.
  - `CHANNEL_OUT/<map>/cases/<case_id>/array_<N>.json`: the sidecar, written last. It is `{"records":{key: <pilot metadata record>}, "h5_sha256", "case_manifest_sha256", "trace_record_sha256", "ris_archive_sha256", "setup_s", "peak_rss_bytes", "worker_id"}`.
  - `CHANNEL_OUT/<map>/cases/<case_id>/array_<N>.FAILED.json`: on any exception for that unit, including the dense round-trip > 1e-10.
- **A case counts as failed for its map** if its trace FAILED or any of its array units FAILED.
  - If `stop_rule_exceeded(failed, map_case_count)`, write `CHANNEL_OUT/<map>/FAILED`; all workers then skip that map.
  - If every non-failed case has all 3 verified shards, the trace map is `DONE`, and the stop rule holds, write `CHANNEL_OUT/<map>/DONE` (with counts).
- **Worker exit:** a worker exits 0 when its arrays have no remaining eligible or pending unit and every in-scope map's trace is `DONE` or `FAILED`. While trace is still running and it has nothing to do, it sleeps 60 s inside the process (no LLM polling).
- **Worker markers:** `CHANNEL_OUT/workers/w<K>.{RUNNING,DONE,FAILED}` and `w<K>.log`.
- **Assemble:** `--assemble` runs only when every map is `DONE`. It writes `CHANNEL_OUT/channel_realizations.h5`, `metadata.json` and `channel_run.json`:
  `{"run_scope","case_manifest_sha256","hdf5_sha256","metadata_sha256","maps":{map:{"cases_total","cases_included","failed_cases":[...]}},"arrays":{"64":{"setup_s":[...],"peak_rss_bytes":max},...}}`.

## Package and gate (stage 3)
- **Package:** `scripts/package_bench_production.py --production --manifest M --channels CHANNEL_OUT/channel_realizations.h5 --metadata CHANNEL_OUT/metadata.json --channel-run CHANNEL_OUT/channel_run.json --output CHANNEL_OUT/package`.
  - Cases listed in `failed_cases` are excluded and recorded in the provenance.
  - **Labels** follow the declaration §3:
    - labelled frames are positions `[0, 5, 10, 15]` of each case's 16-frame list;
    - parents are all-OFF and single greedy;
    - 64 and 128 get all pairs;
    - 256 gets 4,096 sampled pairs per parent, except the full 32,640 for the all-OFF parent on the first labelled frame of trajectories whose index (the id suffix) satisfies `i % 10 == 0`;
    - unlabelled frames keep channels only.
  - **Splits:**
    - `development_full`: indices 0–83 train, 84–101 validation, 102–119 test within each map, plus three leave-environment-out folds.
    - `development_challenge`: every row has split `challenge`, there are no folds, and it is flagged `evaluation_only_never_mixed`.
- **Gate:** `scripts/verify_bench_production_gate.py --production ... --output CHANNEL_OUT/production_gate.json`. It applies the declaration §6.1 checks per case and per map, plus the §6.2 stop rule.
- **Progress check:** `scripts/check_bench_production_progress.py --manifest M --trace-out TRACE_OUT --channel-out CHANNEL_OUT [--verify-finished]`. It prints a per-map JSON summary: traced, trace-failed, channel shards per array, channel-failed, stop-rule state and markers. With `--verify-finished`, it re-runs the per-case §6.1 checks on finished case shards. T7 and the user use it.

## Hosts and CPUs (3060-3)
- The physical cores are CPU k and k+6. CPUs 5 and 11 stay free.
- **Trace:** CPUs `3,9`, GPU 0 (natural) and GPU 1 (challenge).
- **Channel workers:** `w0` → `0,6` with arrays `128,256,64`; `w1` → `1,7` with arrays `256,128,64`; `w2` → `2,8` with arrays `64,128,256`.
- **Each worker:** `ulimit -v` 24 GiB, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, and `MemAvailable` ≥ 16 GiB checked before each basis build.
- **Package and gate:** CPUs `4,10`.
