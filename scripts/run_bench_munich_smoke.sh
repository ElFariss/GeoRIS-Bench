#!/usr/bin/env bash
set -Eeuo pipefail
cd /var/tmp/bench-phase0b-munich-20261108
trap 'r=$?; if ((r)); then printf "exit=%d time=%s\n" "$r" "$(date -Iseconds)" > FAILED.tmp; mv FAILED.tmp FAILED; fi' EXIT
mkdir -p raw results
date -Iseconds
hostname
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for remote in   "https://download1.bayernwolke.de/a/lod2/citygml/690_5334.gml"   "https://download1.bayernwolke.de/a/dgm/dgm1/691_5334.tif"; do
    dest="raw/${remote##*/}"
    if [[ ! -f "$dest" ]]; then
        curl -fsSL --max-time 150 "$remote" -o "$dest.part"
        mv "$dest.part" "$dest"
    fi
done
sha256sum audit_bench_munich_smoke.py raw/*
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | tee gpu_before.csv
gpu=$(sort -t, -k2,2nr gpu_before.csv | awk -F, 'NR==1 && $2>=2048 {print $1}')
[[ -n "$gpu" ]]
export CUDA_VISIBLE_DEVICES="$gpu"
py=/mnt/home/faris/projects/atom-wise-ris/.venv/bin/python
gis=/var/tmp/bench-phase0b-20261108/.venv/lib/python3.12/site-packages
printf 'visible_device=%s\n' "$gpu"
timeout 120 "$py" -c 'import numpy,runpy,sys; sys.path.append(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name="__main__")' "$gis" audit_bench_munich_smoke.py --raw raw --out results
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | tee gpu_after.csv
test -s results/summary.json
date -Iseconds > DONE.tmp
mv DONE.tmp DONE
