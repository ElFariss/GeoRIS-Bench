#!/usr/bin/env bash
# Durable single-tile feasibility run on 3060-1 under faris.
set -u
ROOT="${BENCH_PHASE0B_ROOT:-/var/tmp/bench-phase0b-20261108}"
cd "$ROOT" || exit 2
PY=/mnt/home/faris/projects/atom-wise-ris/.venv/bin/python
SUPP=/var/tmp/bench-phase0b-20261108/.venv/lib/python3.12/site-packages
rm -rf "$SUPP/numpy" "$SUPP"/numpy-*.dist-info
mkdir -p "$ROOT/logs" "$ROOT/results"
exec >> "$ROOT/logs/run.log" 2>&1
date --iso-8601=seconds
hostname
df -h /var/tmp
free -h
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
printf 'source sha256: '
sha256sum build_bench_phase0b_scene.py
printf 'running import and dependency preflight\n'
PYTHONPATH="$SUPP" "$PY" - <<'PY'
import importlib.metadata as md
import numpy, pyproj, rasterio, requests, shapely, sionna.rt
print({"python":__import__("sys").version.split()[0], "numpy":numpy.__version__,
       "pyproj":pyproj.__version__, "rasterio":rasterio.__version__,
       "requests":requests.__version__, "shapely":shapely.__version__,
       "sionna_rt":md.version("sionna-rt")})
assert md.version("sionna-rt") == "2.0.1"
assert tuple(map(int, numpy.__version__.split(".")[:2])) < (2, 5)
PY
preflight=$?
if (( preflight != 0 )); then printf 'preflight_exit=%s\n' "$preflight"; date --iso-8601=seconds > "$ROOT/FAILED"; exit "$preflight"; fi
printf 'running one bounded geometry and Sionna smoke; timeout 15m\n'
set +e
CUDA_VISIBLE_DEVICES=0 BENCH_PHASE0B_ROOT="$ROOT" PYTHONPATH="$SUPP" timeout 900 "$PY" build_bench_phase0b_scene.py
rc=$?
set -e
date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
printf 'run_exit=%s\n' "$rc"
if (( rc == 0 )) && test -s "$ROOT/results/feasibility.json"; then
    date --iso-8601=seconds > "$ROOT/DONE.tmp"
    mv "$ROOT/DONE.tmp" "$ROOT/DONE"
else
    printf 'exit=%s time=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$ROOT/FAILED.tmp"
    mv "$ROOT/FAILED.tmp" "$ROOT/FAILED"
fi
exit "$rc"
