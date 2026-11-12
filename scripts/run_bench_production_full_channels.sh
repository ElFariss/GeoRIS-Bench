#!/usr/bin/env bash
# Production channel stage launcher: 3 parallel workers + assemble.
set -Eeuo pipefail

[[ $# == 5 ]] || { echo "usage: $0 REPO MANIFEST TRACE_OUT CHANNEL_OUT PHYSICS" >&2; exit 2; }

repo=$(realpath "$1")
manifest=$(realpath "$2")
trace_out=$(realpath "$3")
channel_out=$(realpath "$4")
physics=$(realpath "$5")

cd "$repo"

[[ -f "$manifest" && -d "$trace_out" && -d "$physics" ]] || {
  echo "manifest, trace_out, or physics directory missing" >&2
  exit 2
}

mkdir -p "$channel_out" "$channel_out/workers" "$channel_out/logs"

# Prevent concurrent launcher runs on same CHANNEL_OUT
exec 9>"$channel_out/job.lock"
flock -n 9 || { echo "another channel job is holding job.lock in $channel_out" >&2; exit 4; }

# Resume check: if JOB_DONE already exists, nothing to do
if [[ -e "$channel_out/JOB_DONE" ]]; then
  echo "job already completed: $channel_out/JOB_DONE exists" >&2
  exit 0
fi

# Conflicting timing process guard (modelled on scripts/run_bench_production_channel_job.sh)
if pgrep -u "$(id -u)" -af 'python[^ ]* .*([/]?measure_selector_cost|[/]?sample_cpu_siblings|[/]?benchmark_pair_plus|[/]?benchmark_coupled)\.py' >"$channel_out/logs/conflicting-timing.txt" 2>/dev/null; then
  echo 'detected conflicting timing process' >&2
  exit 5
fi

# STAGED_SHA256SUMS check if exists
if [[ -f STAGED_SHA256SUMS ]]; then
  sha256sum -c --quiet STAGED_SHA256SUMS
fi

# Select python interpreter
if [[ -n "${PYTHON:-}" ]]; then
  py="$PYTHON"
elif [[ -x "$repo/.venv/bin/python" ]]; then
  py="$repo/.venv/bin/python"
else
  py="python3"
fi

# The pinned .venv-em lacks h5py/pyarrow/pandas; the staged workspace carries them in deps/ (as in the pilot).
deps_path=""
[[ -d "$repo/deps" ]] && deps_path="$repo/deps:"
export PYTHONPATH="${deps_path}$repo/src:$repo/scripts:${PYTHONPATH:-}" PYTHONDONTWRITEBYTECODE=1

# Preflight verification: verify freeze
"$py" -c 'from atomwise_ris.bench.production_freeze import verify_freeze; print("Freeze verified:", verify_freeze()); import h5py'

# Check MemAvailable >= 16 GiB
available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo "99999999")
if (( available_kb < 16 * 1024 * 1024 )); then
  echo "insufficient MemAvailable: $available_kb KiB < 16 GiB" >&2
  exit 6
fi

pids=()

# Launch worker helper
launch_worker() {
  local wid="$1"
  local cpus="$2"
  local arrays="$3"
  (
    ulimit -v 25165824
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
    exec taskset -c "$cpus" "$py" -u scripts/build_bench_production_channels.py \
      --production \
      --manifest "$manifest" \
      --trace-out "$trace_out" \
      --out "$channel_out" \
      --physics "$physics" \
      --worker-id "$wid" \
      --arrays "$arrays" \
      --cpus "$cpus"
  ) >"$channel_out/workers/w${wid}.log" 2>&1 &
  pids+=("$!")
}

# Contract specifications:
# w0: 0,6  128,256,64
# w1: 1,7  256,128,64
# w2: 2,8  64,128,256
launch_worker 0 "0,6" "128,256,64"
launch_worker 1 "1,7" "256,128,64"
launch_worker 2 "2,8" "64,128,256"

# Forward termination signals to background workers
cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

# Wait for all workers and capture exit codes
rc0=0; wait "${pids[0]}" || rc0=$?
rc1=0; wait "${pids[1]}" || rc1=$?
rc2=0; wait "${pids[2]}" || rc2=$?

trap - INT TERM

echo "Workers finished: w0=$rc0 w1=$rc1 w2=$rc2"

# Check if every map is DONE
all_maps_done=0
if "$py" -c '
import json, sys
from pathlib import Path
from atomwise_ris.bench.production_run import cases_by_map
manifest = json.loads(Path(sys.argv[1]).read_text())
ch_out = Path(sys.argv[2])
for m in cases_by_map(manifest.get("cases", [])):
    if not (ch_out / m / "DONE").is_file():
        sys.exit(1)
sys.exit(0)
' "$manifest" "$channel_out"; then
  all_maps_done=1
fi

assemble_rc=0
if (( all_maps_done == 1 )); then
  echo "All maps DONE. Running assemble..."
  "$py" scripts/build_bench_production_channels.py \
    --production \
    --manifest "$manifest" \
    --trace-out "$trace_out" \
    --out "$channel_out" \
    --physics "$physics" \
    --assemble || assemble_rc=$?
else
  echo "Not all maps are DONE; skipping assemble."
  assemble_rc=99
fi

if (( rc0 == 0 && rc1 == 0 && rc2 == 0 && all_maps_done == 1 && assemble_rc == 0 )); then
  cat <<EOF > "$channel_out/JOB_DONE.tmp"
status=pass
worker_rcs=w0:$rc0,w1:$rc1,w2:$rc2
assemble_rc=$assemble_rc
time=$(date -Is)
EOF
  mv "$channel_out/JOB_DONE.tmp" "$channel_out/JOB_DONE"
  echo "Job completed successfully."
  exit 0
else
  cat <<EOF > "$channel_out/JOB_FAILED.tmp"
status=failed
worker_rcs=w0:$rc0,w1:$rc1,w2:$rc2
all_maps_done=$all_maps_done
assemble_rc=$assemble_rc
time=$(date -Is)
EOF
  mv "$channel_out/JOB_FAILED.tmp" "$channel_out/JOB_FAILED"
  echo "Job failed." >&2
  exit 1
fi
