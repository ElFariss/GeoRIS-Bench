#!/usr/bin/env bash
# v1: local CPU geometry and rendering; never opens either held-out map.
# Stages: prepare (geometry + slope gate) -> generate -> render -> validate.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
out=${BENCH_USERS_OUT:-outputs/previews/bench-users-v1}
cache=${BENCH_USERS_MAP_CACHE:-/tmp/atomwise-phase0b-input}
py=${BENCH_USERS_PYTHON:-/tmp/atomwise-phase0b-audit-env/bin/python}
gate=${BENCH_USERS_SLOPE_GATE:-$out/geometry/slope_gate.json}
mkdir -p "$out"
exec 9>"$out/.pipeline.lock"
flock -n 9
if [[ -e "$out/DONE" ]]; then
    printf 'Completed run already present: %s/DONE\n' "$out"
    exit 0
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts
export MPLCONFIGDIR=/tmp/atomwise-bench-users-mpl CUDA_VISIBLE_DEVICES=""
stamp=$(date +%Y%m%dT%H%M%S)
if [[ -e "$out/run.log" ]]; then mv "$out/run.log" "$out/run.$stamp.log"; fi
if [[ -e "$out/FAILED" ]]; then mv "$out/FAILED" "$out/FAILED.$stamp"; fi
exec >"$out/run.log" 2>&1
stage=preflight
trap 'r=$?; if ((r)); then printf "exit=%s stage=%s time=%s\n" "$r" "$stage" "$(date -Iseconds)" > "$out/FAILED.tmp"; mv "$out/FAILED.tmp" "$out/FAILED"; fi; rm -f "$out/RUNNING"' EXIT
date -Iseconds > "$out/RUNNING"
"$py" -c 'from importlib.metadata import version; print({k:version(k) for k in ("numpy","scipy","pandas","pyarrow","matplotlib","shapely","rasterio","trimesh","embreex")})'
stage=prepare
sources=()
if [[ -n "${BENCH_USERS_SOURCE_MANIFEST:-}" ]]; then
    [[ "${BENCH_USERS_GROUND_FILL:-0}" == 1 ]]
    sources+=(--source-manifest "$BENCH_USERS_SOURCE_MANIFEST")
    : "${BENCH_USERS_BASELINE_GEOMETRY:?Recovered snapshot needs the original v1 geometry}"
fi
if [[ ! -s "$gate" ]]; then
    baseline=()
    if [[ -n "${BENCH_USERS_BASELINE_GEOMETRY:-}" ]]; then baseline+=(--baseline-geometry "$BENCH_USERS_BASELINE_GEOMETRY"); fi
    "$py" scripts/prepare_bench_users_geometry.py --out "$out/geometry" --cache "$cache" "${sources[@]}" "${baseline[@]}"
fi
[[ -s "$gate" ]]
if [[ -n "${BENCH_USERS_SOURCE_MANIFEST:-}" ]]; then
    "$py" -c 'import json,sys; assert len(json.load(open(sys.argv[1]))["v1_source_recovery_checks"])==3' "$gate"
fi
sha256sum docs/bench-users-v1-design-20261108.md "$gate" scripts/generate_bench_users_preview.py src/atomwise_ris/bench/users{,_geometry}.py
stage=generate
ground_args=()
if [[ "${BENCH_USERS_GROUND_FILL:-0}" == 1 ]]; then ground_args+=(--ground-fill); fi
"$py" scripts/generate_bench_users_preview.py --out "$out" --cache "$cache" --geometry-gate "$gate" "${ground_args[@]}" "${sources[@]}"
stage=render
"$py" scripts/render_bench_users_preview.py --root "$out" --draws "$out/draws.parquet"
stage=validate
"$py" scripts/validate_bench_users_preview.py --root "$out"
date -Iseconds > "$out/DONE.tmp"
mv "$out/DONE.tmp" "$out/DONE"
