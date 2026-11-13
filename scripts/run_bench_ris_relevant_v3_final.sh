#!/usr/bin/env bash
# ris_relevant_v3 final stage on labkc-comm3, after both nodes' shards are present under $V:
# full trace merge -> channel map DONE -> assemble -> package -> PILOT_DONE -> G6 -> leverage gate (validation only).
set -Eeuo pipefail
V=${V:-$HOME/atomwise-bench-ris-relevant-v3-20261113}
M=$V/outputs/runs/bench_ris_relevant_v3_20261113/cases.json
MSHA=$(sha256sum "$M" | cut -d' ' -f1)
T=$V/trace; N=$V/channels; P=$V/bench/CoupledRIS-Bench/physics; W=$V/leverage-code
PYE=$HOME/projects/atom-wise-ris/.venv-em/bin/python
mkdir -p "$V/logs"
exec >>"$V/logs/final.log" 2>&1
cd "$V"; say() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { say "FAILED: $1"; printf 'stage=%s\ntime_utc=%s\n' "$1" "$(date -u +%FT%TZ)" > "$V/PILOT_FAILED"; exit 1; }
GEO_PATH="$V/deps:$V/src:$V/scripts"
LEV_PATH="$V/deps:$W/src:$W/scripts"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
say "final start"
env PYTHONPATH="$GEO_PATH" taskset -c 4,10 "$PYE" scripts/merge_bench_ris_relevant_v3_traces.py --final --out "$T" \
  --manifest-sha256 "$MSHA" "$V/trace-s0" "$V/trace-s1" "$V/trace-s2" "$V/trace-s3" || fail merge_final
grep -q '^rr3_final=1' "$T/rotterdam/DONE" || fail merge_final_marker
env PYTHONPATH="$GEO_PATH" taskset -c 4,10 "$PYE" scripts/rr3_channel_map_done.py "$M" "$T" "$N" || fail channel_map_done
if [[ ! -f "$N/JOB_DONE" ]]; then
  env PYTHONPATH="$GEO_PATH" taskset -c 4,10 "$PYE" scripts/run_bench_ris_relevant_v3_channels.py --production \
    --manifest "$M" --trace-out "$T" --out "$N" --physics "$P" --assemble || fail assemble
  printf 'status=pass\ntime=%s\n' "$(date -Is)" > "$N/JOB_DONE"; say "assemble done"
fi
if [[ ! -f "$N/ris_relevant_package_sidecar.json" ]]; then
  if [[ -d "$N/package" ]]; then mv "$N/package" "$N/package.partial.$(date +%s)"; fi
  env PYTHONPATH="$GEO_PATH" taskset -c 4,10 "$PYE" scripts/run_bench_ris_relevant_v3_package.py --production \
    --manifest "$M" --channels "$N/channel_realizations.h5" --metadata "$N/metadata.json" \
    --channel-run "$N/channel_run.json" --output "$N/package" --seed 202611133 > "$V/logs/package.log" 2>&1 || fail package
  ( cd "$N" && sha256sum channel_realizations.h5 metadata.json channel_run.json ris_relevant_split.json \
      ris_relevant_package_sidecar.json $(find package -type f | LC_ALL=C sort) > output_sha256sums )
  say "package done"
fi
cp "$N/ris_relevant_split.json" "$V/ris_relevant_v3_split.json"
printf 'status=pass\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" > "$V/.PILOT_DONE.tmp" && mv "$V/.PILOT_DONE.tmp" "$V/PILOT_DONE"
say "PILOT_DONE"
# G6: shared H_direct identical across sizes (train+validation only; the sealed test is not read)
if env PYTHONPATH="$LEV_PATH" taskset -c 4 "$PYE" "$W/scripts/rr3_g6_direct_identity.py" "$N/package" > "$V/logs/g6.log" 2>&1; then
  say "G6 $(tail -1 "$V/logs/g6.log")"
else
  say "G6 FAILED $(tail -2 "$V/logs/g6.log")"
fi
O=$W/runs/leverage; mkdir -p "$O"
( env PYTHONPATH="$GEO_PATH" taskset -c 6 "$PYE" scripts/run_bench_ris_relevant_v3_gate.py --production --manifest "$M" \
    --trace-out "$T" --channel-run "$N/channel_run.json" --metadata "$N/metadata.json" --package "$N/package" \
    --output "$O/production_gate.json" > "$O/production_gate.log" 2>&1; echo "rc=$?" > "$O/production_gate.rc" ) &
gp=$!
pids=(); i=0; cpus=(0 1 2 3 4 7 8 9 10)
for a in 64 128 256; do for wk in 0 1 2; do
  env PYTHONPATH="$LEV_PATH" taskset -c "${cpus[$i]}" "$PYE" "$W/scripts/ris_leverage_rr3_20261113.py" "$N" "$a" "$wk" 3 \
    "$O/a${a}_w${wk}.jsonl" > "$O/a${a}_w${wk}.log" 2>&1 &
  pids+=("$!"); i=$((i+1))
done; done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || fail leverage_workers
env PYTHONPATH="$LEV_PATH" "$PYE" "$W/scripts/ris_leverage_rr3_summary_20261113.py" "$O" "$M" > "$O/summary.log" 2>&1 || fail leverage_summary
wait "$gp" || true
cp "$O/leverage_gate.json" "$V/LEVERAGE_GATE.json"
say "LEVERAGE_GATE written; production gate $(cat "$O/production_gate.rc")"
