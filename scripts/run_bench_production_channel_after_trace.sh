#!/usr/bin/env bash
# A single bounded file-event wait, then the pilot-only channel runner.
set -Eeuo pipefail
[[ $# == 2 || $# == 3 ]]
root=$(realpath "$1")
trace=$(realpath -m "$2")
cd "$root"
# Optional host guard used for the original lab runs: set BENCH_REQUIRE_HOST to enable it.
[[ -z ${BENCH_REQUIRE_HOST:-} || $(hostname -s) == "$BENCH_REQUIRE_HOST" ]]
exec 7>dependency.lock; flock -n 7
[[ ! -e WAITING && ! -e DEPENDENCY_FAILED && ! -e DONE && ! -e FAILED && ! -e RUNNING ]]
mkdir -p logs
exec >logs/wait_trace.log 2>&1
date -Is > WAITING
finish() {
  local rc=$?
  if ((rc)); then printf 'exit=%s time=%s\n' "$rc" "$(date -Is)" > DEPENDENCY_FAILED.tmp; mv DEPENDENCY_FAILED.tmp DEPENDENCY_FAILED; fi
  if [[ -e WAITING ]]; then unlink WAITING; fi
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 scripts/wait_bench_production_trace.py "$trace" --seconds 3600
unlink WAITING
trap - EXIT TERM INT
if [[ $# == 3 ]]; then
  exec bash scripts/run_bench_production_channel_job.sh "$root" "$trace" "$trace/DONE" "$3"
fi
exec bash scripts/run_bench_production_channel_job.sh "$root" "$trace" "$trace/DONE"
