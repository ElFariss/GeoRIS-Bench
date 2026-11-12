#!/usr/bin/env bash
# Pilot only. Trace source stays on labkc-comm3; no remote GPU use here.
set -Eeuo pipefail
[[ $# == 3 || $# == 4 ]] || { echo "usage: $0 WORKSPACE TRACE_ROOT TRACE_DONE_MARKER [CASE_MANIFEST]" >&2; exit 2; }
cd "$1"
trace_root=$(realpath "$2")
trace_done=$(realpath "$3")
# Optional host guard used for the original lab runs: set BENCH_REQUIRE_HOST to enable it.
[[ -z ${BENCH_REQUIRE_HOST:-} || $(hostname -s) == "$BENCH_REQUIRE_HOST" ]]
[[ -s "$trace_done" && -d "$trace_root" ]]
exec 9>run.lock
flock -n 9
[[ ! -e DONE ]] || exit 0
if [[ -e RUNNING || -e FAILED ]]; then
  echo 'inspect prior marker before deliberately resuming' >&2; exit 3
fi
mkdir -p channels logs
stage=preflight
date -Is > RUNNING
finish() {
  local rc=$?
  if ((rc)); then printf 'exit=%s stage=%s time=%s\n' "$rc" "$stage" "$(date -Is)" > FAILED.tmp; mv FAILED.tmp FAILED; fi
  unlink RUNNING
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec >logs/run.log 2>&1
export PYTHONPATH="$PWD/deps:$PWD/src:$PWD/scripts" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
py="$HOME/projects/atom-wise-ris/.venv-em/bin/python"
manifest=${4:-outputs/runs/bench_production_pilot_20261109/pilot_cases.json}
sha256sum -c --quiet STAGED_SHA256SUMS
trace_manifest_sha=$(sed -n 's/^manifest_sha256=//p' "$trace_done")
[[ $trace_manifest_sha =~ ^[a-f0-9]{64}$ ]]
printf '%s  %s\n' "$trace_manifest_sha" "$trace_root/trace_manifest.json" | sha256sum -c --quiet
"$py" -c 'from atomwise_ris.bench.production_freeze import verify_freeze; print(verify_freeze()); import h5py, PyNEC'
command_args=(--manifest "$manifest" --trace-root "$trace_root" --out channels)
for n in 64 128 256; do
  stage=array_$n
  if pgrep -u "$(id -u)" -af 'python[^ ]* .*([/]?measure_selector_cost|[/]?sample_cpu_siblings|[/]?benchmark_pair_plus|[/]?benchmark_coupled)\.py' >logs/conflicting-timing.txt; then
    echo 'detected conflicting timing process'; exit 5
  fi
  available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  ((available>=32*1024*1024)) || { echo "insufficient host headroom: $available KiB"; exit 6; }
  echo "$(date -Is) array=$n mem_available_kib=$available"
  (ulimit -v 25165824; timeout --signal=TERM --kill-after=45s 5h taskset -c 2,8 "$py" -u \
      scripts/build_bench_production_channels.py "${command_args[@]}" --array "$n")
done
stage=assemble
"$py" scripts/build_bench_production_channels.py "${command_args[@]}" --assemble
stage=package
"$py" scripts/package_bench_production.py --pilot-manifest "$manifest" \
  --channels channels/channel_realizations.h5 --metadata channels/metadata.json --output package
stage=pilot_gate
"$py" scripts/verify_bench_production_gate.py --manifest "$manifest" \
  --trace-manifest "$trace_root/trace_manifest.json" --channel-run channels/channel_run.json \
  --metadata channels/metadata.json --package package --output pilot_gate.json
sha256sum channels/{channel_realizations.h5,metadata.json,channel_run.json} \
  package/provenance.json package/trajectories.parquet package/optimization/*.h5 \
  pilot_gate.json > output_sha256sums
sha256sum -c output_sha256sums
date -Is > DONE.tmp
mv DONE.tmp DONE
