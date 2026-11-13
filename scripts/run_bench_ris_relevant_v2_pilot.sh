#!/usr/bin/env bash
# RIS-relevant v2 pilot chain (docs/bench-ris-relevant-scope-v2-20261113.md), 3060-3.
# 1. trace (GPU 0, CPUs 3,9) in the background; 2. four channel workers (0,6 1,7 2,8 4,10) that
# consume traced cases as they appear; 3. assemble; 4. package (wrapper around the frozen packager).
# Resume: rerun this script. Every stage skips verified work. CPUs 5 and 11 are never used.
set -Eeuo pipefail
V=${V:-/var/tmp/atomwise-bench-ris-relevant-v2-20261113}
M=$V/outputs/runs/bench_ris_relevant_v2_pilot_20261113/cases.json
R=$V/trace-work; T=$V/trace; N=$V/channels; P=$V/bench/CoupledRIS-Bench/physics
PYT=/mnt/home/faris/projects/atom-wise-ris/.venv/bin/python      # Sionna RT 2.0.1
PYE=/mnt/home/faris/projects/atom-wise-ris/.venv-em/bin/python   # PyNEC/NumPy (+ $V/deps)
mkdir -p "$T" "$N/workers" "$N/logs"
exec >>"$V/pilot_chain.log" 2>&1
cd "$V"
say() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { say "FAILED: $1"; printf 'stage=%s\ntime_utc=%s\n' "$1" "$(date -u +%FT%TZ)" > "$V/PILOT_FAILED"; exit 1; }
exec 9>"$V/pilot.lock"; flock -n 9 || { say "another chain holds pilot.lock"; exit 4; }
[[ -e "$V/PILOT_DONE" ]] && { say "PILOT_DONE exists"; exit 0; }
rm -f "$V/PILOT_FAILED"
sha256sum -c --quiet STAGED_SHA256SUMS || fail staged_sha256
avail=$(df --output=avail -B1G "$V" | tail -1 | tr -d ' ')
(( avail >= 5 )) || fail "disk_below_5GB"
say "chain start (free ${avail} GB)"

# 1. trace
tpid=""
if [[ ! -f "$T/JOB_DONE" ]]; then
  rm -f "$T/JOB_FAILED"
  ( set +e
    taskset -c 3,9 env PYTHONPATH="$V/src:$V/scripts" PYTHONDONTWRITEBYTECODE=1 "$PYT" scripts/run_bench_ris_relevant_v2_trace.py \
      --production --root "$R" --manifest "$M" --out "$T" --gpu-id 0 --maps rotterdam >>"$T/trace-job.log" 2>&1
    rc=$?
    if (( rc == 0 )); then printf 'exit_status=0\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" > "$T/JOB_DONE"
    else printf 'exit_status=%s\nfailed_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$T/JOB_FAILED"; fi
    exit $rc ) &
  tpid=$!
  say "trace started pid=$tpid"
fi

# 2. channel workers
export PYTHONPATH="$V/deps:$V/src:$V/scripts" PYTHONDONTWRITEBYTECODE=1
"$PYE" -c 'from atomwise_ris.bench.production_freeze import verify_freeze; print("freeze", verify_freeze()); import h5py' || fail freeze
pids=()
launch() {
  ( ulimit -v 25165824
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
    exec taskset -c "$2" "$PYE" -u scripts/run_bench_ris_relevant_v2_channels.py --production --manifest "$M" \
      --trace-out "$T" --out "$N" --physics "$P" --worker-id "$1" --arrays "$3" --cpus "$2"
  ) >>"$N/workers/w$1.log" 2>&1 &
  pids+=("$!")
}
if [[ ! -f "$N/JOB_DONE" ]]; then
  launch 0 "0,6"  "128,256,64"
  launch 1 "1,7"  "256,128,64"
  launch 2 "2,8"  "64,128,256"
  launch 3 "4,10" "256,64,128"
  say "channel workers started: ${pids[*]}"
  trap 'kill -TERM ${pids[*]} $tpid 2>/dev/null || true' INT TERM
  rcs=()
  for p in "${pids[@]}"; do rc=0; wait "$p" || rc=$?; rcs+=("$rc"); done
  say "workers finished: ${rcs[*]}"
fi
if [[ -n "$tpid" ]]; then rc=0; wait "$tpid" || rc=$?; say "trace finished rc=$rc"; fi
[[ -f "$T/JOB_DONE" ]] || fail trace
[[ -f "$T/rotterdam/DONE" ]] || fail trace_map_done
[[ -f "$N/rotterdam/DONE" ]] || fail channels_map_done

# 3. assemble
if [[ ! -f "$N/JOB_DONE" ]]; then
  "$PYE" scripts/run_bench_ris_relevant_v2_channels.py --production --manifest "$M" --trace-out "$T" --out "$N" \
    --physics "$P" --assemble || fail assemble
  printf 'status=pass\ntime=%s\n' "$(date -Is)" > "$N/JOB_DONE"
  say "assemble done"
fi

# 4. package
if [[ ! -f "$N/ris_relevant_package_sidecar.json" ]]; then
  [[ -d "$N/package" ]] && { mv "$N/package" "$N/package.partial.$(date +%s)"; say "moved a partial package aside"; }
  taskset -c 4,10 "$PYE" scripts/run_bench_ris_relevant_v2_package.py --production --manifest "$M" \
    --channels "$N/channel_realizations.h5" --metadata "$N/metadata.json" --channel-run "$N/channel_run.json" \
    --output "$N/package" --seed 20261113 > "$V/package.log" 2>&1 || fail package
  ( cd "$N" && sha256sum channel_realizations.h5 metadata.json channel_run.json ris_relevant_split.json \
      ris_relevant_package_sidecar.json $(find package -type f | LC_ALL=C sort) > output_sha256sums )
  say "package done"
fi
printf 'status=pass\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" > "$V/.PILOT_DONE.tmp" && mv "$V/.PILOT_DONE.tmp" "$V/PILOT_DONE"
say "PILOT_DONE"
