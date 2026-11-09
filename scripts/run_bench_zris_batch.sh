#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=${TASK_ROOT:-./work/zris}
PY=${PY:-python3}
SRC="$TASK_ROOT/code"
DEPS="$TASK_ROOT/deps"
OUT="$TASK_ROOT/physics"
mkdir -p "$OUT" "$TASK_ROOT/logs"
export PYTHONPATH="$DEPS:$SRC/src:$SRC"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -v 32000000
date -Is > "$TASK_ROOT/started"

run_one() {
  local n="$1" s="$2"
  local d="$OUT/array_$n/$s"
  mkdir -p "$d"
  echo "$(date -Is) start N=$n segments=$s" | tee -a "$TASK_ROOT/logs/batch.log"
  if timeout --signal=TERM --kill-after=30s 2700s "$PY" "$SRC/scripts/run_bench_zris_extract.py" \
      --elements "$n" --segments "$s" --out-dir "$d" \
      >"$TASK_ROOT/logs/array_${n}_${s}.log" 2>&1; then
    test -s "$d/impedance.h5" && test -s "$d/metrics.json"
    echo "$(date -Is) complete N=$n segments=$s" | tee -a "$TASK_ROOT/logs/batch.log"
  else
    status=$?
    echo "$(date -Is) failed exit=$status N=$n segments=$s" | tee -a "$TASK_ROOT/logs/batch.log"
    exit "$status"
  fi
}

for N in 64 128; do
  run_one "$N" 81
  run_one "$N" 161
  "$PY" "$SRC/scripts/compare_bench_zris_mesh.py" \
    --coarse "$OUT/array_$N/81/impedance.h5" \
    --fine "$OUT/array_$N/161/impedance.h5" \
    --out "$OUT/array_$N/mesh_validation.json" \
    > "$TASK_ROOT/logs/mesh_${N}.log" 2>&1
done

date -Is > "$TASK_ROOT/DONE"
rm -f "$TASK_ROOT/FAILED"
echo "complete $(cat "$TASK_ROOT/DONE")" | tee -a "$TASK_ROOT/logs/batch.log"
