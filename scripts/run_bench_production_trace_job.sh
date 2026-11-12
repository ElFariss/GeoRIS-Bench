#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 REPO ROOT CASE_MANIFEST OUTPUT_DIR" >&2
  exit 64
fi
repo=$(realpath "$1")
root=$(realpath -m "$2")
cases=$(realpath "$3")
out=$(realpath -m "$4")
python_bin=${PYTHON:-python3}
mkdir -p "$root" "$out"
if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 69
fi
log="$out/trace-job.log"
running="$out/RUNNING"
done_marker="$out/DONE"
failed="$out/FAILED"

if [[ -e "$done_marker" ]]; then
  echo "pilot already completed: $done_marker" >&2
  exit 73
fi
if [[ -e "$running" ]]; then
  echo "pilot already has a RUNNING marker: $running" >&2
  exit 73
fi
if [[ -e "$failed" ]]; then
  echo "pilot failed previously; inspect $failed before any deliberate resume" >&2
  exit 73
fi

printf 'pid=%s\nstarted_utc=%s\ncase_manifest=%s\n' "$$" "$(date -u +%FT%TZ)" "$cases" > "$running"
set +e
cd "$repo"
PYTHONPATH=src:scripts "$python_bin" scripts/run_bench_production_trace.py \
  --root "$root" --case-manifest "$cases" --out "$out" --build-only >"$log" 2>&1
status=$?
if [[ $status -eq 0 ]]; then
  PYTHONPATH=src:scripts "$python_bin" scripts/run_bench_production_trace.py \
    --root "$root" --case-manifest "$cases" --out "$out" --reuse-scenes --gpu-id 0 >>"$log" 2>&1
  status=$?
fi
set -e
rm -f "$running"
if [[ $status -eq 0 && -f "$out/trace_manifest.json" ]]; then
  done_tmp="$done_marker.tmp.$$"
  printf 'exit_status=0\nfinished_utc=%s\nmanifest_sha256=%s\n' \
    "$(date -u +%FT%TZ)" "$(sha256sum "$out/trace_manifest.json" | awk '{print $1}')" > "$done_tmp"
  mv "$done_tmp" "$done_marker"
  exit 0
fi
failed_tmp="$failed.tmp.$$"
printf 'exit_status=%s\nfailed_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$failed_tmp"
mv "$failed_tmp" "$failed"
exit "${status:-1}"
