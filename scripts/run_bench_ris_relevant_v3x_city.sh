#!/usr/bin/env bash
# ris_relevant_v3x: one city on one node (docs/bench-ris-relevant-v3-crosscity-20261114.md section 8).
# Usage: run_bench_ris_relevant_v3x_city.sh CITY
# Trace: 2 shards on GPU0 (CPU 3) and GPU1 (CPU 9); channel workers on 0,6 1,7 2,8, plus 3,9 after the trace;
# then merge, assemble, package, production gate, data check (CHECK split only), READY.json.
# CPUs 4, 5, 10 and 11 are never used (benchmark timing cores). Resumable: rerun the same command.
set -Eeuo pipefail
CITY=$1
V=${V:-$HOME/atomwise-bench-ris-relevant-v3x-20261114}
R=$V/$CITY
M=$V/outputs/runs/bench_ris_relevant_v3x_20261114/$CITY/cases.json
MSHA=$(sha256sum "$M" | cut -d' ' -f1)
T=$R/trace; N=$R/channels; P=$V/bench/CoupledRIS-Bench/physics; W=$V/leverage-code
PYT=$HOME/projects/atom-wise-ris/.venv/bin/python
PYE=$HOME/projects/atom-wise-ris/.venv-em/bin/python
SEED=$(python3 -c "import json;print(json.load(open('$M'))['seed_word'])")
mkdir -p "$T" "$N/workers" "$N/logs" "$R/logs"
exec >>"$R/logs/city_chain.log" 2>&1
cd "$V"
export RR3X_CITY=$CITY PYTHONDONTWRITEBYTECODE=1
GEO="$V/deps:$V/src:$V/scripts"; LEV="$V/deps:$W/src:$W/scripts"
say() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { say "FAILED: $1"; printf 'stage=%s\ntime_utc=%s\n' "$1" "$(date -u +%FT%TZ)" > "$R/CITY_FAILED"; exit 1; }
exec 9>"$R/city.lock"; flock -n 9 || { say "another chain holds $R/city.lock"; exit 4; }
[[ -e "$R/READY.json" ]] && { say "READY.json exists"; exit 0; }
rm -f "$R/CITY_FAILED"
sha256sum -c --quiet STAGED_SHA256SUMS || fail staged_sha256
say "city chain start $CITY seed=$SEED free=$(df --output=avail -B1G "$V" | tail -1)G"
tp=()
for g in 0 1; do
  cpu=$([[ $g == 0 ]] && echo 3 || echo 9)
  if [[ ! -f "$R/trace-s$g/$CITY/DONE" ]]; then
    retry=""; [[ -f "$R/trace-s$g/$CITY/FAILED" ]] && retry="--retry-failed"
    ( env RR3_SHARD="$g/2" PYTHONPATH="$V/src:$V/scripts" taskset -c $cpu "$PYT" scripts/run_bench_ris_relevant_v3x_trace.py \
        --production --root "$V/trace-work-g$g" --manifest "$M" --out "$R/trace-s$g" --gpu-id $g --maps "$CITY" $retry \
        >>"$R/logs/trace-s$g.log" 2>&1; echo "rc=$?" > "$R/logs/trace-s$g.EXIT" ) &
    tp+=("$!"); say "trace shard $g/2 GPU$g CPU$cpu"
  fi
done
( env PYTHONPATH="$GEO" taskset -c 3 "$PYE" scripts/merge_bench_ris_relevant_v3x_traces.py --watch --out "$T" \
    --manifest-sha256 "$MSHA" "$R/trace-s0" "$R/trace-s1" >>"$R/logs/merge.log" 2>&1; echo "rc=$?" > "$R/logs/merge.EXIT" ) &
mp=$!
wp=()
launch() {
  ( ulimit -v 25165824
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
    exec env PYTHONPATH="$GEO" taskset -c "$2" "$PYE" -u scripts/run_bench_ris_relevant_v3x_channels.py --production \
      --manifest "$M" --trace-out "$T" --out "$N" --physics "$P" --worker-id "$1" --arrays "$3" --cpus "$2"
  ) >>"$N/workers/w$1.log" 2>&1 &
  wp+=("$!")
}
[[ -f "$R/SMOKE.json" ]] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$R/SMOKE.json'))['pass'] else 1)" \
  || fail smoke_preflight_not_passed
if [[ ! -f "$N/JOB_DONE" ]] && ! pgrep -f "run_bench_ris_relevant_v3x_channels.py .*--out $N " > /dev/null; then
  launch 0 "0,6" "128,64,256"
  launch 1 "1,7" "256,64,128"
  launch 2 "2,8" "64,256,128"
  say "workers 0-2 started"
fi
for p in "${tp[@]}"; do wait "$p" || true; done
say "traces finished: $(cat "$R"/logs/trace-s*.EXIT 2>/dev/null | tr '\n' ' ')"
[[ -f "$N/JOB_DONE" ]] || { launch 3 "3,9" "128,256,64"; say "worker 3 started"; }
wait "$mp" || true; say "merge: $(cat "$R/logs/merge.EXIT" 2>/dev/null)"
for p in "${wp[@]}"; do wait "$p" || true; done
while pgrep -f "run_bench_ris_relevant_v3x_channels.py .*--out $N " > /dev/null; do sleep 30; done   # smoke-started workers
say "channel workers finished"
[[ -f "$R/trace-s0/$CITY/DONE" && -f "$R/trace-s1/$CITY/DONE" ]] || fail trace
env PYTHONPATH="$GEO" "$PYE" scripts/merge_bench_ris_relevant_v3x_traces.py --final --out "$T" --manifest-sha256 "$MSHA" \
  "$R/trace-s0" "$R/trace-s1" || fail merge_final
env PYTHONPATH="$GEO" "$PYE" scripts/rr3x_channel_map_done.py "$M" "$T" "$N" || fail channel_map_done
if [[ ! -f "$N/JOB_DONE" ]]; then
  env PYTHONPATH="$GEO" taskset -c 0,6 "$PYE" scripts/run_bench_ris_relevant_v3x_channels.py --production --manifest "$M" \
    --trace-out "$T" --out "$N" --physics "$P" --assemble || fail assemble
  printf 'status=pass\ntime=%s\n' "$(date -Is)" > "$N/JOB_DONE"; say "assemble done"
fi
# Size budget (user constraint 2026-11-14): package <= 1.6 GB per city. Projection = assembled HDF5 x 1.12
# (v3 package/HDF5 ratio 3,007,789,105 / 2,701,177,395 = 1.1135). Stop BEFORE packaging if over budget.
H5B=$(stat -c %s "$N/channel_realizations.h5")
PROJ=$(( H5B * 112 / 100 ))
say "size check: channel HDF5 ${H5B} B, projected package ${PROJ} B, budget 1600000000 B"
if (( PROJ > 1600000000 )); then
  printf 'h5_bytes=%s\nprojected_package_bytes=%s\nbudget_bytes=1600000000\n' "$H5B" "$PROJ" > "$R/SIZE_BUDGET_EXCEEDED"
  fail size_budget_before_packaging
fi
if [[ ! -f "$N/ris_relevant_package_sidecar.json" ]]; then
  if [[ -d "$N/package" ]]; then mv "$N/package" "$N/package.partial.$(date +%s)"; fi
  env PYTHONPATH="$GEO" taskset -c 0,6 "$PYE" scripts/run_bench_ris_relevant_v3x_package.py --production --manifest "$M" \
    --channels "$N/channel_realizations.h5" --metadata "$N/metadata.json" --channel-run "$N/channel_run.json" \
    --output "$N/package" --seed "$SEED" > "$R/logs/package.log" 2>&1 || fail package
  ( cd "$N" && sha256sum channel_realizations.h5 metadata.json channel_run.json ris_relevant_split.json \
      ris_relevant_package_sidecar.json $(find package -type f | LC_ALL=C sort) > output_sha256sums )
  say "package done"
fi
cp "$N/ris_relevant_split.json" "$R/ris_relevant_split.json"
printf 'status=pass\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" > "$R/.PILOT_DONE.tmp" && mv "$R/.PILOT_DONE.tmp" "$R/PILOT_DONE"
say "PILOT_DONE"
O=$R/check; mkdir -p "$O"
( env PYTHONPATH="$GEO" taskset -c 9 "$PYE" scripts/run_bench_ris_relevant_v3x_gate.py --production --manifest "$M" \
    --trace-out "$T" --channel-run "$N/channel_run.json" --metadata "$N/metadata.json" --package "$N/package" \
    --output "$O/production_gate.json" > "$O/production_gate.log" 2>&1; echo "rc=$?" > "$O/production_gate.rc" ) &
gp=$!
pids=(); i=0; cpus=(0 1 2 3 6 7 8)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for a in 64 128 256; do for wk in 0 1; do
  env PYTHONPATH="$LEV" taskset -c "${cpus[$i]}" "$PYE" "$W/scripts/ris_leverage_rx3_20261114.py" "$N" "$a" "$wk" 2 \
    "$O/a${a}_w${wk}.jsonl" > "$O/a${a}_w${wk}.log" 2>&1 &
  pids+=("$!"); i=$((i+1))
done; done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || fail check_workers
env PYTHONPATH="$LEV" "$PYE" "$W/scripts/ris_leverage_rx3_summary_20261114.py" "$O" "$M" > "$O/summary.log" 2>&1 || fail check_summary
wait "$gp" || true
PKGB=$(du -sb "$N/package" | cut -f1)
say "package size ${PKGB} B"
python3 - "$R" "$O" "$N" "$CITY" "$PKGB" <<'PY'
import json, sys, time
R, O, N, city, pkgb = sys.argv[1:]
g = json.load(open(f"{O}/leverage_gate.json")); pg = json.load(open(f"{O}/production_gate.json"))
per = {a: {"verdict": "PASS" if all(v["pass"] for k, v in c.items() if not k.startswith("L3")) else "FAIL",
           **{k.split("_")[0]: {"value": v["value"], "pass": v["pass"]} for k, v in c.items() if not k.startswith("L3")}}
       for a, c in g["cells"].items()}
ok = pg.get("status") == "pass"
ready = {"city": city, "scope": "ris_relevant_v3x", "declaration": "GeoRIS-Bench/docs/bench-ris-relevant-v3-crosscity-20261114.md",
         "package_path": f"{N}/package", "split_file": f"{R}/ris_relevant_split.json", "check_split": "0000-0011", "test_split": "0012-0047 SEALED",
         "production_gate": pg.get("status"), "check_per_array": per,
         "package_size_bytes": int(pkgb), "package_budget_bytes": 1600000000,
         "ready_arrays": [int(a) for a, v in per.items() if v["verdict"] == "PASS" and ok],
         "city_status": "ready" if ok and any(v["verdict"] == "PASS" for v in per.values()) else "failed",
         "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(ready, open(f"{R}/READY.json.tmp", "w"), indent=2)
import os; os.replace(f"{R}/READY.json.tmp", f"{R}/READY.json")
print(json.dumps({"city": city, "ready_arrays": ready["ready_arrays"], "status": ready["city_status"]}))
PY
say "READY.json written"
