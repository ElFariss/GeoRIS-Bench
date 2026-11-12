#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 REPO TRACE_ROOT MANIFEST TRACE_OUT GPU_ID MAPS [EXTRA_ARGS...]" >&2
  exit 64
fi

repo=$(realpath "$1")
root=$(realpath -m "$2")
cases=$(realpath "$3")
out=$(realpath -m "$4")
gpu_id="$5"
maps="$6"
shift 6 || true
extra_args=("$@")

python_bin=${PYTHON:-python3}
# CPU set for this trace process (default 3,9). CPUs 5 and 11 are reserved for timing and refused.
trace_cpus=${TRACE_CPUS:-3,9}
if [[ ",$trace_cpus," == *",5,"* || ",$trace_cpus," == *",11,"* ]]; then
  echo "TRACE_CPUS must not include 5 or 11: $trace_cpus" >&2
  exit 64
fi
mkdir -p "$root" "$out"
if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 69
fi

log="$out/trace-job.log"
running="$out/RUNNING"
job_done="$out/JOB_DONE"
job_failed="$out/JOB_FAILED"

if [[ -e "$running" ]]; then
  echo "trace job already has a RUNNING marker: $running" >&2
  exit 73
fi

printf 'pid=%s\nstarted_utc=%s\ncase_manifest=%s\ngpu_id=%s\nmaps=%s\n' \
  "$$" "$(date -u +%FT%TZ)" "$cases" "$gpu_id" "$maps" > "$running"

set +e
cd "$repo"
taskset -c "$trace_cpus" env PYTHONPATH=src:scripts "$python_bin" scripts/run_bench_production_trace.py \
  --production \
  --root "$root" \
  --manifest "$cases" \
  --out "$out" \
  --gpu-id "$gpu_id" \
  --maps "$maps" \
  "${extra_args[@]}" >>"$log" 2>&1
status=$?
set -e

rm -f "$running"
if [[ $status -eq 0 ]]; then
  done_tmp="$job_done.tmp.$$"
  printf 'exit_status=0\nfinished_utc=%s\n' "$(date -u +%FT%TZ)" > "$done_tmp"
  mv "$done_tmp" "$job_done"
  rm -f "$job_failed"
  exit 0
else
  failed_tmp="$job_failed.tmp.$$"
  printf 'exit_status=%s\nfailed_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$failed_tmp"
  mv "$failed_tmp" "$job_failed"
  exit "$status"
fi
