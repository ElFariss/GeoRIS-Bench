#!/usr/bin/env bash
set -Eeuo pipefail

# Run from /var/tmp/bench-round2-rt-20261108/attempt3 after staging the two
# Python sources. Geometry mode is CPU-only; smoke mode holds the shared
# high-memory lock for its full lifetime.
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mode=${1:?usage: run_bench_rt_round2.sh geometry|smoke}
py=${PYTHON:-python3}
export PYTHONPATH="$ROOT/code" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
command -v taskset >/dev/null
mkdir -p logs results scenes

if [[ $mode == smoke ]]; then
  exec 9>"${BENCH_HIGHMEM_LOCK:-/var/tmp/atomwise-bench-round2-highmem.lock}"
  flock -n 9 || { echo 'shared high-memory lock is held; refusing RT smoke' >&2; exit 4; }
  [[ -e GEOMETRY_DONE ]] || { echo 'geometry preflight is incomplete' >&2; exit 5; }
  [[ ! -e DONE && ! -e RUNNING ]] || { echo 'duplicate or completed smoke marker exists' >&2; exit 6; }
  gpu_log=logs/gpu_before.csv
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tee "$gpu_log"
  awk -F, '$1==1 && $4+0>=2048 {ok=1} END {exit !ok}' "$gpu_log"
  echo "$(date -Iseconds)" > RUNNING.tmp; mv RUNNING.tmp RUNNING
  stage=smoke
  trap 'rc=$?; if ((rc)); then printf "exit=%d stage=%s time=%s\n" "$rc" "$stage" "$(date -Iseconds)" > FAILED.tmp; mv FAILED.tmp FAILED; fi; rm -f RUNNING' EXIT
  export CUDA_VISIBLE_DEVICES=1
  "$py" -c 'import importlib.metadata as m, mitsuba as mi; print("sionna-rt",m.version("sionna-rt"),"mitsuba",m.version("mitsuba"),"variant",mi.variant())'
  timeout --signal=TERM --kill-after=45s 4h taskset -c 0-4,6-10 \
    "$py" "$ROOT/code/run_bench_rt_preview_smoke.py" \
    --root "$ROOT" --case-manifest inputs/cases.json --out results --reuse-scenes \
    2>&1 | tee logs/smoke.log
  stage=verify
  sha256sum results/summary.json | tee results/summary.sha256
  test -s results/summary.json
  echo "$(date -Iseconds)" > DONE.tmp; mv DONE.tmp DONE
  rm -f FAILED
elif [[ $mode == geometry ]]; then
  [[ ! -e GEOMETRY_DONE && ! -e GEOMETRY_RUNNING ]] || { echo 'duplicate geometry marker exists' >&2; exit 6; }
  echo "$(date -Iseconds)" > GEOMETRY_RUNNING.tmp; mv GEOMETRY_RUNNING.tmp GEOMETRY_RUNNING
  stage=geometry_preflight
  trap 'rc=$?; if ((rc)); then printf "exit=%d stage=%s time=%s\n" "$rc" "$stage" "$(date -Iseconds)" > GEOMETRY_FAILED.tmp; mv GEOMETRY_FAILED.tmp GEOMETRY_FAILED; fi; rm -f GEOMETRY_RUNNING' EXIT
  timeout --signal=TERM --kill-after=30s 30m taskset -c 0-4,6-10 \
    "$py" "$ROOT/code/run_bench_rt_preview_smoke.py" \
    --root "$ROOT" --case-manifest inputs/cases.json --out results --build-only \
    2>&1 | tee logs/geometry_preflight.log
  sha256sum results/geometry_preflight.json | tee results/geometry_preflight.sha256
  test -s results/geometry_preflight.json
  echo "$(date -Iseconds)" > GEOMETRY_DONE.tmp; mv GEOMETRY_DONE.tmp GEOMETRY_DONE
  rm -f GEOMETRY_FAILED
else
  echo "unknown mode: $mode" >&2; exit 2
fi
