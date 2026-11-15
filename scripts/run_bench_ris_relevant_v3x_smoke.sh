#!/usr/bin/env bash
# ris_relevant_v3x preflight C (smoke) for one city: trace check-split trajectories 0000 and 0001 (both GPUs),
# merge them into the run's trace dir WITHOUT a DONE marker, start channel workers 0-2 (they build their NEC bases
# and then wait for the full run), and run rr3x_smoke_check.py once the 6 units exist. Writes $R/SMOKE.json.
# The trace shards' smoke DONE markers are archived so the full run resumes them. CPUs 4,5,10,11 never used.
set -Eeuo pipefail
CITY=$1
V=${V:-$HOME/atomwise-bench-ris-relevant-v3x-20261114}
R=$V/$CITY
M=$V/outputs/runs/bench_ris_relevant_v3x_20261114/$CITY/cases.json
MSHA=$(sha256sum "$M" | cut -d' ' -f1)
T=$R/trace; N=$R/channels; P=$V/bench/CoupledRIS-Bench/physics; W=$V/leverage-code
PYT=$HOME/projects/atom-wise-ris/.venv/bin/python; PYE=$HOME/projects/atom-wise-ris/.venv-em/bin/python
mkdir -p "$T" "$N/workers" "$R/logs"
exec >>"$R/logs/smoke.log" 2>&1
cd "$V"; export RR3X_CITY=$CITY PYTHONDONTWRITEBYTECODE=1
GEO="$V/deps:$V/src:$V/scripts"; LEV="$V/deps:$W/src:$W/scripts"
say() { echo "[$(date -u +%FT%TZ)] $*"; }
sha256sum -c --quiet STAGED_SHA256SUMS || { say "staged checksum failure"; exit 3; }
say "smoke start $CITY"
tp=()
for g in 0 1; do
  cpu=$([[ $g == 0 ]] && echo 3 || echo 9)
  ( env RR3_SHARD="$g/2" RR3X_ONLY=0,1 PYTHONPATH="$V/src:$V/scripts" taskset -c $cpu "$PYT" scripts/run_bench_ris_relevant_v3x_trace.py \
      --production --root "$V/trace-work-g$g" --manifest "$M" --out "$R/trace-s$g" --gpu-id $g --maps "$CITY" \
      >>"$R/logs/trace-s$g.log" 2>&1; echo "rc=$?" > "$R/logs/smoke-trace-s$g.EXIT" ) &
  tp+=("$!")
done
for p in "${tp[@]}"; do wait "$p" || true; done
say "smoke traces: $(cat "$R"/logs/smoke-trace-s*.EXIT | tr '\n' ' ')"
for g in 0 1; do
  d=$R/trace-s$g/$CITY
  [[ -f $d/FAILED ]] && { say "smoke trace FAILED: $(head -c 400 $d/FAILED)"; echo '{"pass": false, "stage": "trace"}' > "$R/SMOKE.json"; exit 1; }
  [[ -f $d/DONE ]] && mv "$d/DONE" "$d/DONE.smoke"
  [[ -f $d/trace_manifest.json ]] && mv "$d/trace_manifest.json" "$d/trace_manifest.smoke.json"
done
if [[ "${SMOKE_TRACE_ONLY:-0}" == 1 ]]; then   # early G5 check while the CPUs are busy with another city
  python3 -c "
import json,glob
for f in sorted(glob.glob('$R/trace-s*/$CITY/cases/*.json')):
    d=json.load(open(f)); pc=d['geometry']['mesh_audit']['preflight_checks']
    print(f.split('/')[-1], d['status'], 'G5', pc['all_passed'], d['geometry']['endpoint_preflight']['points_checked'])
" > "$R/SMOKE_TRACE_G5.txt"; say "trace-only smoke: $(tr '\n' ' ' < $R/SMOKE_TRACE_G5.txt)"; exit 0
fi
env PYTHONPATH="$GEO" "$PYE" scripts/merge_bench_ris_relevant_v3x_traces.py --out "$T" --manifest-sha256 "$MSHA" "$R/trace-s0" "$R/trace-s1"
say "merged: $(ls $T/$CITY/cases | wc -l) records"
launch() {
  ( ulimit -v 25165824
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
    exec env PYTHONPATH="$GEO" taskset -c "$2" "$PYE" -u scripts/run_bench_ris_relevant_v3x_channels.py --production \
      --manifest "$M" --trace-out "$T" --out "$N" --physics "$P" --worker-id "$1" --arrays "$3" --cpus "$2"
  ) >>"$N/workers/w$1.log" 2>&1 < /dev/null &
}
if ! pgrep -f "run_bench_ris_relevant_v3x_channels.py .*--out $N " > /dev/null; then
  setsid bash -c "$(declare -f launch); V=$V M=$M T=$T N=$N P=$P GEO=$GEO PYE=$PYE RR3X_CITY=$CITY; export RR3X_CITY; launch 0 0,6 128,64,256; launch 1 1,7 256,64,128; launch 2 2,8 64,256,128; wait" > /dev/null 2>&1 < /dev/null &
  say "channel workers 0-2 started (persistent, shared with the full run)"
fi
cid0=$CITY-rx3-0000; cid1=$CITY-rx3-0001
while true; do
  n=$(ls $N/$CITY/cases/$cid0/array_*.json $N/$CITY/cases/$cid1/array_*.json 2>/dev/null | grep -vc FAILED || true)
  f=$(ls $N/$CITY/cases/$cid0/ $N/$CITY/cases/$cid1/ 2>/dev/null | grep -c FAILED || true)
  (( f > 0 )) && { say "smoke channel unit FAILED"; echo '{"pass": false, "stage": "channels"}' > "$R/SMOKE.json"; exit 1; }
  (( n >= 6 )) && break
  sleep 30
done
say "6 smoke units done"
env PYTHONPATH="$LEV" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 3 "$PYE" scripts/rr3x_smoke_check.py "$CITY" "$R" 0,1 "$R/SMOKE.json" \
  || { say "smoke check script error"; exit 1; }
say "SMOKE: $(python3 -c "import json;print(json.load(open('$R/SMOKE.json'))['pass'])")"
