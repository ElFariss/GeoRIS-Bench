#!/usr/bin/env python3
"""Fail-closed CHECK-only integrity gate for the v3x cross-city smoke.

Usage: rr3x_strict_smoke_check.py CITY CITY_ROOT IDX[,IDX] OUT.json

This complements the frozen rr3x_smoke_check.py. It checks completeness and
provenance that the original smoke summary did not assert. It never opens TEST
channel metrics; indices must be in the declared CHECK interval 0..11.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np


CITY_META = {
    "maastricht": {"map_index": 2, "seed_word": 2026111402},
    "utrecht": {"map_index": 4, "seed_word": 2026111403},
    "berlin": {"map_index": 3, "seed_word": 2026111404},
}
ARRAYS = (64, 128, 256)
FRAMES_PER_CASE = 16
PATH_CAP = 2_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(checks: dict, name: str, detail: str) -> None:
    checks[name] = {"pass": False, "detail": detail}


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    city, city_root, index_text, output = sys.argv[1:]
    root = Path(city_root).resolve()
    idxs = [int(x) for x in index_text.split(",") if x != ""]
    out_path = Path(output)
    checks: dict = {}
    result = {"city": city, "indices": idxs, "checks": checks}
    try:
        if city not in CITY_META:
            raise ValueError(f"unsupported city {city!r}")
        if not idxs or len(set(idxs)) != len(idxs) or any(i < 0 or i > 11 for i in idxs):
            raise ValueError("indices must be unique declared CHECK indices in 0..11")

        repo = root.parent
        manifest_path = repo / "outputs/runs/bench_ris_relevant_v3x_20261114" / city / "cases.json"
        manifest = json.loads(manifest_path.read_text())
        meta = CITY_META[city]
        checks["city_manifest_identity"] = {
            "pass": manifest.get("city") == city and manifest.get("map_index") == meta["map_index"]
                    and manifest.get("seed_word") == meta["seed_word"]
                    and manifest.get("subset") == "ris_relevant_v3x",
            "manifest": str(manifest_path), "city": manifest.get("city"),
            "map_index": manifest.get("map_index"), "seed_word": manifest.get("seed_word"),
        }
        if not checks["city_manifest_identity"]["pass"]:
            raise ValueError("manifest city/map-index/seed identity mismatch")
        by_id = {c["id"]: c for c in manifest.get("cases", [])}
        expected_ids = [f"{city}-rx3-{i:04d}" for i in idxs]
        manifest_ok = True
        for i, cid in zip(idxs, expected_ids):
            c = by_id.get(cid)
            if c is None or c.get("map_id") != city or c.get("split") != "check" or c.get("trajectory_index") != i:
                manifest_ok = False
                break
        checks["check_split_identity"] = {"pass": manifest_ok, "case_ids": expected_ids}
        if not manifest_ok:
            raise ValueError("requested case missing or not marked CHECK for this city")

        # Verify held-out source selection from the exact manifest actually consumed
        # by the builder, including local file hashes. Maastricht uses the frozen
        # development-map source tree and is recorded by manifest identity above.
        source_info = {"applicable": city in ("berlin", "utrecht")}
        if city in ("berlin", "utrecht"):
            heldout = repo / "outputs/runs/bench_ris_relevant_v3x_20261114/heldout_inputs"
            source_path = heldout / "source-manifest.json"
            sources = json.loads(source_path.read_text())
            entry = next((x for x in sources if x.get("id") == city), None)
            if entry is None:
                raise ValueError(f"{city}: source absent from held-out source manifest")
            # The builder reads derived city files from heldout_inputs/cache.
            # Upstream archive hashes stay in derived_from for provenance but
            # those original archives are not all staged on the compute node.
            file_rows = [(x.get("file"), x.get("sha256")) for x in entry.get("buildings", [])]
            file_rows += [(entry.get(k, {}).get("file"), entry.get(k, {}).get("sha256")) for k in ("terrain", "osm")]
            unique = {}
            for name, digest in file_rows:
                if not name or not digest or str(name).startswith(("derived:", "http://", "https://")):
                    raise ValueError(f"{city}: malformed consumed-source entry")
                matches = list((heldout / "cache").rglob(Path(name).name))
                p = matches[0] if len(matches) == 1 else heldout / "cache" / Path(name).name
                unique[p.resolve()] = digest
            source_ok = bool(unique) and all(p.is_file() and sha256(p) == d for p, d in unique.items())
            source_info.update({"manifest": str(source_path), "city_entry": city,
                                "verified_consumed_files": len(unique),
                                "upstream_provenance_entries": sum(len(x.get("derived_from", [])) for x in entry.get("buildings", [])) + int(bool(entry.get("terrain", {}).get("derived_from"))),
                                "pass": source_ok})
            if not source_ok:
                raise ValueError(f"{city}: selected held-out source files missing or hash mismatch")
        else:
            source_info["pass"] = True
        checks["source_provenance"] = source_info

        # Exact frame coverage is derived from declared CHECK case frame ids.
        expected_frames = {cid: [int(f["frame"]) for f in by_id[cid]["frames"]] for cid in expected_ids}
        frame_ok = all(len(fs) == FRAMES_PER_CASE and len(set(fs)) == FRAMES_PER_CASE for fs in expected_frames.values())
        # The mutable aggregate `trace/` is not authoritative; consume shard
        # roots only, and collapse any symlink aliases by resolved file path.
        trace_roots = sorted(root.glob("trace-s*"))
        trace_ok = bool(trace_roots)
        cap_records = 0
        cap_max_paths = 0
        trace_evidence = []
        for cid, frames in expected_frames.items():
            direct_seen = set()
            for tr in trace_roots:
                for p in tr.glob(f"**/{city}/direct/{cid}/frame-*.json"):
                    if p.resolve() != p.absolute():
                        continue
                    side = json.loads(p.read_text())
                    fidx = int(p.stem.split("-")[-1])
                    if fidx in direct_seen:
                        raise ValueError(f"duplicate shared-direct trace for {cid} frame {fidx}")
                    direct_seen.add(fidx)
                    npz = p.with_suffix(".npz")
                    if not npz.is_file() or side.get("sha256") != sha256(npz):
                        raise ValueError(f"shared-direct artifact/hash missing or inconsistent: {npz}")
                    if int(side.get("max_num_paths_per_src", -1)) != PATH_CAP:
                        raise ValueError(f"shared-direct path-cap setting mismatch: {p}")
                    if not isinstance(side.get("paths"), int) or side["paths"] < 0 or side["paths"] >= PATH_CAP:
                        raise ValueError(f"cannot establish shared-direct cap safety: {p}")
                    with np.load(npz) as z:
                        if "path_index" not in z or len(z["path_index"]) != side["paths"]:
                            raise ValueError(f"shared-direct sidecar path count disagrees with NPZ: {p}")
                    cap_records += 1
                    cap_max_paths = max(cap_max_paths, side["paths"])
            if direct_seen != set(frames):
                trace_ok = False
                trace_evidence.append({"case": cid, "array": "direct", "expected": len(frames), "found": len(direct_seen)})
            for n in ARRAYS:
                seen = set()
                for tr in trace_roots:
                    for p in tr.glob(f"**/{city}/n{n}/{cid}/frame-*.npz"):
                        if p.resolve() != p.absolute():
                            continue
                        fidx = int(p.stem.split("-")[-1])
                        if fidx in seen:
                            raise ValueError(f"duplicate trace record for {cid} array {n} frame {fidx}")
                        seen.add(fidx)
                        with np.load(p) as z:
                            if "path_index" not in z or "metadata_json" not in z:
                                raise ValueError(f"trace NPZ lacks path/cap metadata: {p}")
                            md = json.loads(str(z["metadata_json"].item()))
                            pi = z["path_index"]
                            solver = md.get("solver", {})
                            if md.get("case_id") != cid or md.get("map_id") != city or int(md.get("frame", -1)) != fidx:
                                raise ValueError(f"trace NPZ identity mismatch: {p}")
                            if int(md.get("array_elements", -1)) != n or int(solver.get("max_num_paths_per_src", -1)) != PATH_CAP:
                                raise ValueError(f"trace NPZ array/path-cap setting mismatch: {p}")
                            if int(md.get("ray_count", -1)) != len(pi) or len(pi) == 0 or int(pi.max()) >= PATH_CAP - 1:
                                raise ValueError(f"cannot establish cap safety from per-source path indices: {p}")
                            cap_records += 1
                            cap_max_paths = max(cap_max_paths, int(pi.max()) + 1)
                if seen != set(frames):
                    trace_ok = False
                    trace_evidence.append({"case": cid, "array": n, "expected": len(frames), "found": len(seen)})
        checks["trace_completeness_and_cap"] = {
            "pass": frame_ok and trace_ok and cap_records == len(idxs) * FRAMES_PER_CASE * (len(ARRAYS) + 1),
            "trace_records": cap_records, "expected_records": len(idxs) * FRAMES_PER_CASE * (len(ARRAYS) + 1),
            "max_paths_in_any_record": cap_max_paths,
            "cap": PATH_CAP, "coverage_issues": trace_evidence[:20],
            "basis": "direct sidecars and array NPZ metadata declare the cap; path_index maxima and lengths are below it",
        }
        if not checks["trace_completeness_and_cap"]["pass"]:
            raise ValueError("trace frame coverage incomplete or cap evidence insufficient")

        # All three channel files must carry every expected frame, all numeric
        # datasets must be finite, the stored rate must match the shared loader,
        # and served-user network rows must not be pathless.
        loader_path = repo / "bench/CoupledRIS-Bench/examples/load_dataset.py"
        spec = importlib.util.spec_from_file_location("rr3x_load_dataset", loader_path)
        loader = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(loader)
        leverage_src = repo / "leverage-code/src"
        if leverage_src.is_dir() and str(leverage_src) not in sys.path:
            sys.path.insert(0, str(leverage_src))
        import atomwise_ris.bench as bench_pkg
        leverage_bench = leverage_src / "atomwise_ris/bench"
        if leverage_bench.is_dir() and str(leverage_bench) not in bench_pkg.__path__:
            bench_pkg.__path__.append(str(leverage_bench))
        import atomwise_ris.bench.search_adapter_ris_relevant_v3x as rr

        channel_records = 0
        max_rate_error = 0.0
        no_pathless = True
        finite = True
        channel_issues = []
        for cid, frames in expected_frames.items():
            for n in ARRAYS:
                p = root / "channels" / city / "cases" / cid / f"array_{n}.h5"
                if not p.is_file():
                    channel_issues.append(f"missing {p}")
                    continue
                with h5py.File(p, "r") as h:
                    groups = []
                    h.visititems(lambda name, obj: groups.append((name, obj))
                                 if isinstance(obj, h5py.Group) and "H_direct" in obj else None)
                    if len(groups) != FRAMES_PER_CASE:
                        channel_issues.append(f"{cid}/a{n}: {len(groups)} channel frames")
                    actual_frames = {int(name.rsplit("/", 1)[-1]) for name, _ in groups}
                    if actual_frames != set(frames):
                        channel_issues.append(f"{cid}/a{n}: frame IDs differ from manifest")
                    if actual_frames != set(frames):
                        continue
                    for name, g in groups:
                        for node in g.values():
                            if isinstance(node, h5py.Dataset) and node.dtype.kind in "fc":
                                if not np.isfinite(node[()]).all():
                                    finite = False
                        rate = loader.rate_for_bits(p, name, g["stored_bits"][()])
                        err = abs(float(rate) - float(g.attrs["stored_rate_bps_hz"]))
                        max_rate_error = max(max_rate_error, err)
                        ev = rr.network_from_group(g, 8 if n == 64 else 16)
                        no_pathless = no_pathless and not bool(rr.pathless_users(ev).any())
                        channel_records += 1
        channel_ok = not channel_issues and finite and no_pathless and max_rate_error <= 1e-10
        checks["channels_and_loader"] = {
            "pass": channel_ok, "channel_groups": channel_records,
            "expected_groups": len(idxs) * FRAMES_PER_CASE * len(ARRAYS),
            "finite": finite, "no_pathless_served_user": no_pathless,
            "loader_rate_error_max": max_rate_error, "loader_rate_ok": max_rate_error <= 1e-10,
            "issues": channel_issues[:20],
        }
        if not channel_ok:
            raise ValueError("channel frame coverage or numerical integrity failure")

        result["pass"] = all(v.get("pass", False) for v in checks.values())
        result["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception as exc:
        result["pass"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"city": city, "pass": result["pass"], "checks": checks, **({"error": result["error"]} if "error" in result else {})}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
