#!/usr/bin/env python3
"""Smoke preflight C for ris_relevant_v3x (user requirement 2026-11-14; declaration section 'Preflight').

Usage: rr3x_smoke_check.py CITY RUN_ROOT IDX[,IDX] OUT.json   (check-split indices only, e.g. 0,1)
Reads the per-unit channel HDF5 (RUN_ROOT/channels/<city>/cases/<id>/array_<n>.h5) and trace npz of the listed cases:
- every dataset finite; loader rate (examples/load_dataset.py) vs stored rate <= 1e-10;
- H_direct_uncorrected bit-identical 64 vs 128 and relative difference <= 1% at 256 (per frame);
- no served user pathless; max path index per source well below the 2e6 cap (no truncation);
- projected package bytes per trajectory (unit HDF5 x 1.12) <= 1.6e9/48;
- DESCRIPTIVE ONLY: best (b35, 300/500/800 ms, seed 7) - all OFF per frame.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, h5py
from atomwise_ris.bench import search_adapter_ris_relevant_v3x as rr
from atomwise_ris.optimization import pair_plus

city, R, idxs, out = sys.argv[1], Path(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
V = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("load_dataset", V / "bench/CoupledRIS-Bench/examples/load_dataset.py")
loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
DL = {64: 300.0, 128: 500.0, 256: 800.0}; CAP = 2_000_000
res = {"city": city, "cases": {}, "checks": {}}
fin = rate_ok = pathless_ok = True; max_rate_err = 0.0; hu = {}; gains = {64: [], 128: [], 256: []}; bytes_per = []
max_pidx = 0
for i in idxs:
    cid = f"{city}-rx3-{i:04d}"
    assert rr.split_of(cid) == "check", cid
    tot = 0
    for n in (64, 128, 256):
        f5 = R / "channels" / city / "cases" / cid / f"array_{n}.h5"
        tot += f5.stat().st_size
        keys = []
        with h5py.File(f5, "r") as f:
            f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Group) and "H_direct" in obj else None)
            for k in sorted(keys):
                g = f[k]
                for nm in g:
                    if isinstance(g[nm], h5py.Dataset) and g[nm].dtype.kind in "fc" and not np.isfinite(g[nm][()]).all():
                        fin = False
                hu[(cid, k.split("/")[-1], n)] = g["H_direct_uncorrected"][()]
                lr = loader.rate_for_bits(f5, k, g["stored_bits"][()])
                e = abs(lr - float(g.attrs["stored_rate_bps_hz"])); max_rate_err = max(max_rate_err, e)
                ev = rr.network_from_group(g, 8 if n == 64 else 16)
                if rr.pathless_users(ev).any():
                    pathless_ok = False
                r0 = float(ev.score(ev.state(np.zeros(ev.n, int)).field[None])[0])
                bits, _, _, _ = pair_plus.search(ev, "pair_plus_b35", time.perf_counter_ns() + int(DL[n] * 1e6), 7)
                rb = max(float(ev.score(ev.state(np.asarray(bits, int)).field[None])[0]), r0)
                gains[n].append(rb - r0)
        for npz in sorted((R / "trace").glob(f"{city}/n{n}/{cid}/frame-*.npz")) + sorted((R / "trace").glob(f"{city}/direct/{cid}/frame-*.npz")):
            with np.load(npz) as z:
                if len(z["path_index"]):
                    max_pidx = max(max_pidx, int(z["path_index"].max()))
    bytes_per.append(tot * 1.12)
    res["cases"][cid] = {"unit_hdf5_bytes": tot}
eq128 = []; rel256 = []
for (cid, fk, n), v in hu.items():
    if n == 64:
        eq128.append(bool(np.array_equal(v, hu[(cid, fk, 128)])))
        rel256.append(float(np.linalg.norm(hu[(cid, fk, 256)] - v) / max(np.linalg.norm(v), 1e-300)))
c = res["checks"]
c["finite"] = fin
c["loader_rate_error_max"] = max_rate_err; c["loader_rate_ok"] = max_rate_err <= 1e-10
c["hdu_64_eq_128_frames"] = [int(sum(eq128)), len(eq128)]; c["hdu_64_eq_128"] = all(eq128) and len(eq128) > 0
c["hdu_256_rel_max"] = max(rel256) if rel256 else None; c["hdu_256_within_1pct"] = bool(rel256) and max(rel256) <= 0.01
c["no_pathless_served_user"] = pathless_ok
c["max_path_index"] = max_pidx; c["no_path_cap_truncation"] = max_pidx < CAP - 1
c["projected_package_bytes_per_traj_max"] = max(bytes_per); c["size_within_budget"] = max(bytes_per) <= 1.6e9 / 48
c["frames_per_array"] = {n: len(v) for n, v in gains.items()}
res["pass"] = all(c[k] for k in ("finite", "loader_rate_ok", "hdu_64_eq_128", "hdu_256_within_1pct",
                                  "no_pathless_served_user", "no_path_cap_truncation", "size_within_budget"))
res["descriptive_L2_median_best_minus_alloff"] = {n: float(np.median(v)) for n, v in gains.items()}
res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(res, open(out, "w"), indent=2)
print(json.dumps({"pass": res["pass"], **{k: v for k, v in c.items()}, "L2_desc": res["descriptive_L2_median_best_minus_alloff"]}))
