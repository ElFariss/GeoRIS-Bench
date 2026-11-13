#!/usr/bin/env bash
# ris_relevant_v3 per-node stage (docs/bench-ris-relevant-scope-v3-20261113.md section 8).
# Usage: run_bench_ris_relevant_v3_node.sh "SHARD_ON_GPU0 SHARD_ON_GPU1"   e.g. "0 1" on comm3, "2 3" on comm2.
# Two trace shards (GPU0 on CPU 3, GPU1 on CPU 9), a node-local trace merge, 4 channel workers on CPU pairs
# 0,6 1,7 2,8 4,10 and a fifth on 3,9 once both traces finish. CPUs 5 and 11 are never used. Resumable: rerun.
set -Eeuo pipefail
V=${V:-$HOME/atomwise-bench-ris-relevant-v3-20261113}
read -r S0 S1 <<< "$1"
M=$V/outputs/runs/bench_ris_relevant_v3_20261113/cases.json
MSHA=$(sha256sum "$M" | cut -d' ' -f1)
T=$V/trace; N=$V/channels; P=$V/bench/CoupledRIS-Bench/physics
PYT=$HOME/projects/atom-wise-ris/.venv/bin/python
PYE=$HOME/projects/atom-wise-ris/.venv-em/bin/python
mkdir -p "$T" "$N/workers" "$N/logs" "$V/logs"
exec >>"$V/logs/node_chain.log" 2>&1
cd "$V"
say() { echo "[$(date -u +%FT%TZ)] $*"; }
exec 9>"$V/node.lock"; flock -n 9 || { say "another node chain holds node.lock"; exit 4; }
[[ -e "$V/NODE_DONE" ]] && { say "NODE_DONE exists"; exit 0; }
sha256sum -c --quiet STAGED_SHA256SUMS || { say "staged checksum failure"; exit 3; }
say "node chain start shards=$S0,$S1 free=$(df --output=avail -B1G "$V" | tail -1)G"
tp=()
for g in 0 1; do
  s=$([[ $g == 0 ]] && echo $S0 || echo $S1); cpu=$([[ $g == 0 ]] && echo 3 || echo 9)
  mkdir -p "$V/trace-s$s"
  if [[ ! -f "$V/trace-s$s/rotterdam/DONE" ]]; then
    ( RR3_SHARD="$s/4" taskset -c $cpu env PYTHONPATH="$V/src:$V/scripts" PYTHONDONTWRITEBYTECODE=1 "$PYT" scripts/run_bench_ris_relevant_v3_trace.py \
        --production --root "$V/trace-work-g$g" --manifest "$M" --out "$V/trace-s$s" --gpu-id $g --maps rotterdam \
        >>"$V/logs/trace-s$s.log" 2>&1; echo "rc=$?" > "$V/trace-s$s/EXIT" ) &
    tp+=("$!"); say "trace shard $s on GPU$g CPU$cpu pid=$!"
  fi
done
( taskset -c 3 "$PYE" scripts/merge_bench_ris_relevant_v3_traces.py --watch --out "$T" --manifest-sha256 "$MSHA" \
    "$V/trace-s$S0" "$V/trace-s$S1" >>"$V/logs/merge.log" 2>&1; echo "rc=$?" > "$V/logs/merge.EXIT" ) &
mp=$!
export PYTHONPATH="$V/deps:$V/src:$V/scripts" PYTHONDONTWRITEBYTECODE=1
wp=()
launch() {
  ( ulimit -v 25165824
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
    exec taskset -c "$2" "$PYE" -u scripts/run_bench_ris_relevant_v3_channels.py --production --manifest "$M" \
      --trace-out "$T" --out "$N" --physics "$P" --worker-id "$1" --arrays "$3" --cpus "$2"
  ) >>"$N/workers/w$1.log" 2>&1 &
  wp+=("$!")
}
launch 0 "0,6"  "128,64,256"
launch 1 "1,7"  "128,64,256"
launch 2 "2,8"  "256,64,128"
launch 3 "4,10" "256,64,128"
say "workers 0-3 started"
for p in "${tp[@]}"; do wait "$p" || true; done
say "traces finished: $(cat $V/trace-s$S0/EXIT 2>/dev/null) $(cat $V/trace-s$S1/EXIT 2>/dev/null)"
launch 4 "3,9" "64,256,128"
say "worker 4 started"
wait "$mp" || true; say "merge: $(cat $V/logs/merge.EXIT)"
for p in "${wp[@]}"; do wait "$p" || true; done
say "workers finished"
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" > "$V/NODE_DONE"
