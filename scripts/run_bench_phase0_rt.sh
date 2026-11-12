#!/usr/bin/env bash
# Launch in tmux from the staged Phase 0 directory on 3060-1; see audit doc.
set -u
cd /var/tmp/atomwise-bench-phase0-20261108 || exit 2
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/mplconfig"
py=/mnt/home/faris/projects/atom-wise-ris/.venv/bin/python
rc=0
for mode in geometry capabilities; do
    date -Iseconds
    timeout 180 "$py" audit_bench_phase0_rt.py "$mode" --geometry-input atomwise-bench-geometry-input.json --out results
    status=$?
    if (( status != 0 )); then
        printf '%s exit %d\n' "$mode" "$status"
        rc=1
    fi
done
marker=DONE
if (( rc != 0 )); then marker=FAILED; fi
date -Iseconds > "$marker.tmp"
mv "$marker.tmp" "$marker"
exit "$rc"
