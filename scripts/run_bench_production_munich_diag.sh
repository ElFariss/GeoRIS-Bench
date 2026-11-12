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
root_name=$(basename "$root")
out_name=$(basename "$out")
[[ $root_name =~ ^munich-diagnostic-work(-retry[1-9][0-9]*)?$ &&
   $out_name =~ ^munich-diagnostic(-retry[1-9][0-9]*)?$ &&
   ${root_name#munich-diagnostic-work} == ${out_name#munich-diagnostic} ]] || {
  echo "refusing non-isolated or mismatched diagnostic paths" >&2; exit 64;
}
[[ $(dirname "$root") == $(dirname "$out") ]] || {
  echo "diagnostic work and output must share their isolated parent" >&2; exit 64;
}
[[ -x "$python_bin" ]] || { echo "pinned Python is unavailable: $python_bin" >&2; exit 69; }
[[ -f "$root/scene_manifest.json" && -f "$root/scenes/munich/scene.xml" ]] || {
  echo "hash-attested Munich geometry is not staged in the diagnostic workspace" >&2; exit 66;
}
mkdir -p "$out"
for marker in DONE FAILED RUNNING; do
  [[ ! -e "$out/$marker" ]] || { echo "diagnostic marker already exists: $out/$marker" >&2; exit 73; }
done
log="$out/trace-job.log"
running_tmp="$out/RUNNING.tmp.$$"
printf 'pid=%s\nstarted_utc=%s\ncase_manifest=%s\n' "$$" "$(date -u +%FT%TZ)" "$cases" > "$running_tmp"
mv "$running_tmp" "$out/RUNNING"
set +e
cd "$repo"
PYTHONPATH=src:scripts "$python_bin" scripts/run_bench_production_trace.py \
  --root "$root" --case-manifest "$cases" --out "$out" \
  --reuse-scenes --munich-only-diagnostic --gpu-id 0 >"$log" 2>&1
status=$?
set -e
rm -f "$out/RUNNING"
if [[ $status -eq 0 && -f "$out/trace_manifest.json" ]]; then
  done_tmp="$out/DONE.tmp.$$"
  printf 'exit_status=0\nfinished_utc=%s\nmanifest_sha256=%s\n' \
    "$(date -u +%FT%TZ)" "$(sha256sum "$out/trace_manifest.json" | awk '{print $1}')" > "$done_tmp"
  mv "$done_tmp" "$out/DONE"
  exit 0
fi
failed_tmp="$out/FAILED.tmp.$$"
printf 'exit_status=%s\nfailed_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$failed_tmp"
mv "$failed_tmp" "$out/FAILED"
exit "${status:-1}"
