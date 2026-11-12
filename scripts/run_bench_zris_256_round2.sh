#!/usr/bin/env bash
# Run inside the dedicated 3060-3 task workspace; no protected checkout writes.
set -Eeuo pipefail
root=${1:?usage: run_bench_zris_256_round2.sh TASK_WORKSPACE}
cd "$root"
# Optional host guard used for the original lab runs: set BENCH_REQUIRE_HOST to enable it.
[[ -z ${BENCH_REQUIRE_HOST:-} || $(hostname -s) == "$BENCH_REQUIRE_HOST" ]]
mkdir -p logs physics/array_256
exec 9>run.lock
flock -n 9
[[ ! -e DONE ]] || exit 0
if [[ -e RUNNING || -e FAILED ]]; then
    echo 'Prior RUNNING/FAILED exists; inspect and archive before intentional resume.' >&2
    exit 3
fi
exec > >(tee -a logs/runner.log) 2>&1
stage=preflight
finalize() {
    local rc=$?
    if (( rc )); then
        printf 'exit=%s stage=%s time=%s\n' "$rc" "$stage" "$(date -Is)" > FAILED.tmp
        mv FAILED.tmp FAILED
    fi
    unlink RUNNING
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -Is > RUNNING.tmp
mv RUNNING.tmp RUNNING

py="$HOME/projects/atom-wise-ris/.venv-em/bin/python"
export PYTHONPATH="$PWD/deps:$PWD/code/src"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
out="$PWD/physics/array_256"
sha256sum -c STAGED_SHA256SUMS
"$py" -c 'import sys, PyNEC, numpy, h5py; print(sys.version.split()[0], numpy.__version__, h5py.__version__, PyNEC.__file__)'
taskset -cp $$

memory_guard() {
    "$py" - "$1" <<'PY'
import json
from pathlib import Path
import sys

floor = int(sys.argv[1]) * (1 << 30)
host = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
            if s.startswith('MemAvailable:'))
path = next(s.split('::', 1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if '::' in s)
root = Path('/sys/fs/cgroup')
cur = root / path.lstrip('/')
groups = {}
for d in (cur, *cur.parents):
    if d == root.parent:
        break
    used = d / 'memory.current'
    if used.is_file():
        for key in ('memory.max', 'memory.high'):
            p = d / key
            if p.is_file() and (value := p.read_text().strip()) != 'max':
                groups[str(p)] = int(value) - int(used.read_text())
headroom = min([host, *groups.values()])
print(json.dumps({'host_available_bytes': host, 'cgroup_headroom_bytes': groups,
                  'usable_bytes': headroom, 'minimum_bytes': floor}), flush=True)
if headroom < floor:
    raise SystemExit('Insufficient memory headroom; preserve the completed mesh and stop.')
PY
}

timing_guard() {
    # Current project timing tools; the team also reserves the node by agreement.
    if pgrep -u "$(id -u)" -af 'python[^ ]* .*([/]?measure_selector_cost|[/]?sample_cpu_siblings|[/]?benchmark_pair_plus|[/]?benchmark_coupled)\.py' > logs/conflicting-timing.txt; then
        echo 'Conflicting project timing process; stopping.' >&2
        exit 4
    fi
}

verify_one() {
    "$py" - "$1" "$2" <<'PY'
from hashlib import file_digest
import json
from pathlib import Path
import sys
import h5py
import numpy as np
d = Path(sys.argv[1]); s = int(sys.argv[2]); f = d / 'impedance.h5'
m = json.loads((d / 'metrics.json').read_text())
with f.open('rb') as fp:
    assert file_digest(fp, 'sha256').hexdigest() == m['hdf5_sha256']
with h5py.File(f, 'r') as h:
    z = h['Z_RIS_ohm']; assert z.shape == (256, 256) and z.dtype == np.complex128
    assert np.isfinite(z[:]).all() and int(z.attrs['segments_per_wire']) == s
    assert int(z.attrs['elements']) == 256 and z.attrs['frequency_hz'] == 5.53e9
assert m['status'] == 'complete' and m['segments_per_wire'] == s and m['elements'] == 256
print('Verified', f, m['hdf5_sha256'])
PY
}

extract_one() {
    local s=$1 gib=$2 vmem_kib=$3 wall=$4
    local d="$out/$s"
    if [[ -e "$d/impedance.h5" || -e "$d/metrics.json" ]]; then
        verify_one "$d" "$s"
        return
    fi
    timing_guard
    memory_guard "$gib" | tee -a "logs/memory_${s}.jsonl"
    printf '%s start segments=%s vmem_kib=%s\n' "$(date -Is)" "$s" "$vmem_kib"
    local -a diagnostics=(/usr/bin/time -v -o "logs/time_${s}.txt")
    if [[ ${ZRIS_DIAGNOSTIC_RETRY:-0} == 1 && $s == 81 ]]; then
        diagnostics+=(strace -f -qq -e trace=mmap,mremap,brk -e status=failed -o "logs/alloc_${s}.log")
    fi
    (ulimit -v "$vmem_kib"; timeout --signal=TERM --kill-after=60s "$wall" \
        "${diagnostics[@]}" "$py" -u code/scripts/run_bench_zris_extract.py --elements 256 --segments "$s" --out-dir "$d") \
        > "logs/array_256_${s}.log" 2>&1
    verify_one "$d" "$s"
    printf '%s finished segments=%s\n' "$(date -Is)" "$s"
}

stage=mesh_81
if [[ ${ZRIS_DIAGNOSTIC_RETRY:-0} == 1 ]]; then
    extract_one 81 32 25165824 2h
else
    extract_one 81 18 16777216 2h
fi
stage=estimate_mesh_161
"$py" - "$out/81/metrics.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    coarse = json.load(f)['peak_rss_bytes']
estimate = coarse * (161 / 81) ** 2
cap = int(53.5 * (1 << 30))
print(json.dumps({'fine_rss_estimate_from_coarse_bytes': round(estimate),
                  'coarse_rss_bytes': coarse, 'fine_address_cap_bytes': cap}), flush=True)
if estimate >= cap:
    raise SystemExit('Coarse memory implies insufficient fine-mesh headroom; preserve the coarse result.')
PY
stage=reserve_mesh_161
exec 8>"${BENCH_HIGHMEM_LOCK:-/var/tmp/atomwise-bench-round2-highmem.lock}"
flock -w 1800 8
stage=mesh_161
extract_one 161 58 56098816 8h
flock -u 8
stage=assemble
"$py" code/scripts/compare_bench_zris_mesh.py --coarse "$out/81/impedance.h5" --fine "$out/161/impedance.h5" --out "$out/mesh_validation.json"
"$py" code/scripts/assemble_bench_zris_hdf5.py --array-dir "$out"
"$py" code/scripts/validate_bench_zris_bundle.py --array-dir "$out"
sha256sum "$out"/{geometry.json,mesh_validation.json,numerical_validation.json,impedance.h5} "$out"/{81,161}/{impedance.h5,metrics.json} > output_sha256sums
sha256sum -c output_sha256sums
date -Is > DONE.tmp
mv DONE.tmp DONE
