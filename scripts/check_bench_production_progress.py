#!/usr/bin/env python3
"""Check bench production progress across trace and channel stages.

Works on partial outputs at any time while production runs. Read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from atomwise_ris.bench.production_run import stop_rule_exceeded
from atomwise_ris.hybrid.operator import HybridCoupledNetwork


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _as_complex(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    if arr.dtype.fields and {"r", "i"}.issubset(arr.dtype.fields):
        return arr["r"] + 1j * arr["i"]
    if arr.ndim and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    raise ValueError("unsupported complex array encoding")


def check_progress(manifest_path: Path, trace_out: Path, channel_out: Path,
                   verify_finished: bool = False, maps_filter: list[str] | None = None,
                   max_cases: int | None = None) -> dict:
    manifest_path, trace_out, channel_out = map(Path, (manifest_path, trace_out, channel_out))
    manifest = json.loads(manifest_path.read_text())

    cases_by_map: dict[str, list[dict]] = defaultdict(list)
    for c in manifest.get("cases", []):
        m = c["map_id"]
        if maps_filter is None or m in maps_filter:
            cases_by_map[m].append(c)

    if max_cases is not None:
        for m in cases_by_map:
            cases_by_map[m] = cases_by_map[m][:max_cases]

    workers_dir = channel_out / "workers"
    worker_markers = {"running": [], "done": [], "failed": []}
    if workers_dir.is_dir():
        for f in sorted(workers_dir.iterdir()):
            if f.name.endswith(".RUNNING"):
                worker_markers["running"].append(f.name[:-8])
            elif f.name.endswith(".DONE"):
                worker_markers["done"].append(f.name[:-5])
            elif f.name.endswith(".FAILED"):
                worker_markers["failed"].append(f.name[:-7])

    map_reports = {}
    verified_cases = {}

    for map_id in sorted(cases_by_map.keys()):
        map_cases = cases_by_map[map_id]
        total_cases = len(map_cases)

        trace_map_dir = trace_out / map_id
        trace_cases_dir = trace_map_dir / "cases"
        trace_done_cases = []
        trace_failed_cases = []
        if trace_cases_dir.is_dir():
            for c in map_cases:
                cid = c["id"]
                if (trace_cases_dir / f"{cid}.json").is_file():
                    trace_done_cases.append(cid)
                if (trace_cases_dir / f"{cid}.FAILED.json").is_file():
                    trace_failed_cases.append(cid)

        trace_markers = {
            "DONE": (trace_map_dir / "DONE").is_file(),
            "FAILED": (trace_map_dir / "FAILED").is_file(),
        }

        channel_map_dir = channel_out / map_id
        channel_cases_dir = channel_map_dir / "cases"
        channel_shards = {}
        shards_failed_cases = set()

        for n in (64, 128, 256):
            n_done = []
            n_failed = []
            if channel_cases_dir.is_dir():
                for c in map_cases:
                    cid = c["id"]
                    cdir = channel_cases_dir / cid
                    h5_file = cdir / f"array_{n}.h5"
                    sidecar = cdir / f"array_{n}.json"
                    failed_file = cdir / f"array_{n}.FAILED.json"
                    if h5_file.is_file() and sidecar.is_file():
                        n_done.append(cid)
                    if failed_file.is_file():
                        n_failed.append(cid)
                        shards_failed_cases.add(cid)
            channel_shards[str(n)] = {
                "done_count": len(n_done),
                "failed_count": len(n_failed),
                "done_cases": n_done,
                "failed_cases": n_failed,
            }

        channel_markers = {
            "DONE": (channel_map_dir / "DONE").is_file(),
            "FAILED": (channel_map_dir / "FAILED").is_file(),
        }

        distinct_failed = sorted(set(trace_failed_cases) | shards_failed_cases)
        is_stop_exceeded = False
        if total_cases > 0:
            try:
                is_stop_exceeded = stop_rule_exceeded(len(distinct_failed), total_cases)
            except Exception:
                is_stop_exceeded = (len(distinct_failed) * 100 > 2 * total_cases)

        map_reports[map_id] = {
            "total_cases": total_cases,
            "trace_records_done": len(trace_done_cases),
            "trace_records_failed": len(trace_failed_cases),
            "trace_done_cases": trace_done_cases,
            "trace_failed_cases": trace_failed_cases,
            "trace_markers": trace_markers,
            "channel_shards": channel_shards,
            "channel_markers": channel_markers,
            "distinct_failed_cases": distinct_failed,
            "distinct_failed_count": len(distinct_failed),
            "stop_rule_exceeded": is_stop_exceeded,
        }

        if verify_finished and channel_cases_dir.is_dir():
            for c in map_cases:
                cid = c["id"]
                cdir = channel_cases_dir / cid
                has_all_shards = all(
                    (cdir / f"array_{n}.h5").is_file() and (cdir / f"array_{n}.json").is_file()
                    for n in (64, 128, 256)
                )
                if not has_all_shards:
                    continue

                reasons = []

                # Trace mesh preflight
                trace_file = trace_cases_dir / f"{cid}.json"
                if not trace_file.is_file():
                    reasons.append("missing trace record")
                else:
                    try:
                        tr = json.loads(trace_file.read_text())
                        geom = tr.get("geometry", {})
                        preflight = geom.get("preflight_checks") or geom.get("mesh_audit", {}).get("preflight_checks") or tr.get("preflight_checks", {})
                        if not isinstance(preflight, dict) or preflight.get("all_passed") is not True:
                            reasons.append("mesh preflight checks failed")
                    except Exception as exc:
                        reasons.append(f"error reading trace record: {exc}")

                for n in (64, 128, 256):
                    h5_path = cdir / f"array_{n}.h5"
                    json_path = cdir / f"array_{n}.json"

                    # 1. Sidecar sha256
                    try:
                        sc = json.loads(json_path.read_text())
                        if sc.get("h5_sha256") != sha256(h5_path):
                            reasons.append(f"array_{n} sidecar h5_sha256 mismatch")
                    except Exception as exc:
                        reasons.append(f"array_{n} sidecar read error: {exc}")
                        sc = {}

                    # 2. BS-RIS LOS
                    records = sc.get("records", {})
                    for rkey, rval in records.items():
                        bs_ris = rval.get("link_classes", {}).get("bs_ris")
                        if bs_ris != "los":
                            reasons.append(f"array_{n} BS-RIS link is '{bs_ris}', expected 'los' at {rkey}")

                    # 3. Finite datasets & dense rate within 1e-10
                    try:
                        with h5py.File(h5_path, "r") as hf:
                            leaf_groups = []
                            def _visit(name, obj):
                                if isinstance(obj, h5py.Group) and "Z_effective" in obj:
                                    leaf_groups.append(obj)
                            hf.visititems(_visit)
                            if not leaf_groups:
                                reasons.append(f"array_{n} contains no channel groups")

                            for g in leaf_groups:
                                ds_names = ("H_direct", "C_in", "C_out", "Z_effective", "loads", "precoder", "stored_bits")
                                for ds in ds_names:
                                    if ds not in g:
                                        reasons.append(f"missing dataset {ds} at {g.name}")
                                        continue
                                    arr = g[ds][()]
                                    if not np.isfinite(arr).all():
                                        reasons.append(f"non-finite values in {ds} at {g.name}")

                                noise = float(g.attrs.get("noise", 0.0))
                                stored_rate = float(g.attrs.get("stored_rate_bps_hz", np.nan))
                                if not (np.isfinite(noise) and noise > 0 and np.isfinite(stored_rate)):
                                    reasons.append(f"invalid noise/stored_rate at {g.name}")

                                # Dense rate via HybridCoupledNetwork
                                try:
                                    z = _as_complex(g["Z_effective"])
                                    ci, co, hd = (_as_complex(g[k]) for k in ("C_in", "C_out", "H_direct"))
                                    loads = _as_complex(g["loads"])
                                    if loads.shape == (z.shape[0], 2): loads = loads[0]
                                    w = _as_complex(g["precoder"])
                                    ev = HybridCoupledNetwork(z, co, ci, hd, loads, w=w, noise=noise)
                                    rate = float(ev.score(ev.channel(g["stored_bits"][()]) @ ev.w))
                                    if abs(rate - stored_rate) > 1e-10:
                                        reasons.append(f"dense rate error {abs(rate - stored_rate):.3e} > 1e-10 at {g.name}")
                                except Exception as exc:
                                    reasons.append(f"dense rate evaluation failed at {g.name}: {exc}")
                    except Exception as exc:
                        reasons.append(f"failed opening {h5_path}: {exc}")

                verified_cases[cid] = {
                    "map_id": map_id,
                    "status": "pass" if not reasons else "fail",
                    "reasons": reasons,
                }

    report = {
        "manifest": str(manifest_path),
        "trace_out": str(trace_out),
        "channel_out": str(channel_out),
        "workers": worker_markers,
        "maps": map_reports,
    }
    if verify_finished:
        report["verified_cases"] = verified_cases

    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--trace-out", required=True, type=Path)
    p.add_argument("--channel-out", required=True, type=Path)
    p.add_argument("--verify-finished", action="store_true", help="verify finished case shards")
    p.add_argument("--maps", type=str, help="comma-separated maps to include")
    p.add_argument("--max-cases", type=int, help="max cases to check per map")
    a = p.parse_args(argv)

    maps_filter = [m.strip() for m in a.maps.split(",") if m.strip()] if a.maps else None
    report = check_progress(a.manifest, a.trace_out, a.channel_out,
                            verify_finished=a.verify_finished,
                            maps_filter=maps_filter, max_cases=a.max_cases)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
