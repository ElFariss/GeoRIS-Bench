#!/usr/bin/env python3
"""Package channel-realization pilot cases and exact HybridCoupledNetwork labels.

The input is a compact case manifest plus the lead-owned channel HDF5. Outputs
are written only below an explicit, isolated output directory. No ray tracing
or GPU work is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from importlib.metadata import version
import json
import os
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from atomwise_ris.hybrid.operator import HybridCoupledNetwork


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (bytes, np.bytes_)): return v.decode("utf-8", errors="replace")
    return v


def deterministic_pairs(n: int, count: int, seed: int) -> np.ndarray:
    """Uniform deterministic sample without replacement from unordered pairs."""
    total = n * (n - 1) // 2
    if count >= total:
        return np.column_stack(np.triu_indices(n, 1)).astype(np.int32)
    rng = np.random.default_rng(seed)
    # Sampling integer triangular ranks avoids allocating O(n^2) for large N.
    ranks = np.sort(rng.choice(total, size=count, replace=False))
    # Map lexicographic (i,j), i<j, ranks to indices via row offsets.
    out = np.empty((count, 2), dtype=np.int32)
    offsets = np.arange(n - 1, 0, -1, dtype=np.int64)
    csum = np.cumsum(offsets)
    for q, r in enumerate(ranks):
        i = int(np.searchsorted(csum, r, side="right"))
        before = 0 if i == 0 else int(csum[i - 1])
        out[q] = (i, i + 1 + int(r - before))
    return out


def greedy_parent(ev: HybridCoupledNetwork, depth: int = 4) -> np.ndarray:
    """Bounded deterministic positive single-flip greedy parent from all-off."""
    state = ev.state(np.zeros(ev.n, dtype=np.int8))
    for _ in range(depth):
        gains = ev.moves(state) - ev.score(state.field)
        if not len(gains) or not np.isfinite(gains).all():
            raise ValueError("non-finite single-flip score during parent construction")
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        state = ev.flip(state, i)
    return state.bits.copy()


def _get_loads(g, metadata, n):
    loads = g["loads"][()]
    loads = np.asarray(loads, dtype=np.complex128)
    if loads.shape == (2,):
        return loads
    if loads.shape == (n, 2):
        if not np.allclose(loads, loads[0:1], rtol=0, atol=0):
            raise ValueError("per-element load pairs differ; expected common binary states")
        return loads[0]
    if loads.shape == (n,):
        # A vector of two states is the standard physical convention; an N-vector
        # is ambiguous and intentionally rejected rather than guessed.
        raise ValueError("loads must contain the two binary complex impedances")
    raise ValueError(f"unsupported loads shape {loads.shape}")


def _complex_data(ds):
    a = np.asarray(ds[()])
    if np.iscomplexobj(a):
        return a.astype(np.complex128, copy=False)
    if a.dtype.fields and {"r", "i"}.issubset(a.dtype.fields):
        return a["r"] + 1j * a["i"]
    if a.ndim and a.shape[-1] == 2 and np.issubdtype(a.dtype, np.number):
        return a[..., 0] + 1j * a[..., 1]
    raise ValueError(f"{ds.name} must be complex or (...,2) real/imag")


def _network(g, metadata):
    a = _complex_data(g["Z_effective"])
    c_in, c_out, h_direct = (_complex_data(g[k]) for k in ("C_in", "C_out", "H_direct"))
    n = a.shape[0]
    if a.shape != (n, n): raise ValueError(f"Z_effective must be square, got {a.shape}")
    w = _complex_data(g["precoder"])
    loads = _get_loads(g, metadata, n)
    noise = float(g.attrs["noise"])
    if not (np.isfinite(noise) and noise > 0): raise ValueError("noise must be positive finite scalar")
    if not all(np.isfinite(x).all() for x in (a, c_in, c_out, h_direct, w, loads)):
        raise ValueError("non-finite channel/operator/load values")
    ev = HybridCoupledNetwork(a, c_out, c_in, h_direct, loads, w=w, noise=noise)
    if ev.n not in (64, 128, 256): raise ValueError(f"unregistered array size {ev.n}")
    return ev


def _user_links(metadata, user):
    """Accept the sidecar's user list or ID-keyed map without guessing classes."""
    users = metadata.get("users", {}) if isinstance(metadata, dict) else {}
    if isinstance(users, list):
        row = next((x for x in users if str(x.get("user_id")) == str(user["user_id"])), {})
    elif isinstance(users, dict):
        row = users.get(str(user["user_id"]), {})
    else:
        row = {}
    shared = metadata.get("link_classes", {}) if isinstance(metadata, dict) else {}
    return {
        # Never promote the v1.1 geometry preview labels into RT-derived classes.
        "bs_link": row.get("bs_user", row.get("bs_link", "unknown")),
        "ris_link": row.get("ris_user", row.get("ris_los", "unknown")),
        "bs_ris_link": shared.get("bs_ris", metadata.get("bs_ris_link", "unknown")),
    }


def _pairs_for(n, count, seed, full):
    if n in (64, 128) or full:
        return deterministic_pairs(n, n * (n - 1) // 2, seed)
    return deterministic_pairs(n, count, seed)


def label_group(ev, parent_bits, pairs, parent_id):
    state = ev.state(parent_bits)
    base = float(ev.score(state.field))
    single = np.asarray(ev.moves(state), dtype=np.float64) - base
    pair = np.asarray(ev.moves_pairs(state, pairs), dtype=np.float64) - base
    syn = pair - single[pairs[:, 0]] - single[pairs[:, 1]]
    if not all(np.isfinite(x).all() for x in (single, pair, syn, state.field)):
        raise ValueError("label calculation produced non-finite values")
    return {"parent_id": parent_id, "bits": state.bits, "base_rate": base,
            "single_gain": single, "pairs": pairs, "pair_gain": pair, "synergy": syn}


def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, f"pid={os.getpid()} started={time.time():.6f}\n".encode())
    os.close(fd)


def package(pilot_manifest: Path, channels: Path, metadata_path: Path, output: Path,
            seed: int = 20261109, pairs_256: int = 4096, greedy_depth: int = 4,
            verify_freeze: bool = True, partial_arrays=None,
            production: bool = False, channel_run: Path | None = None):
    pilot_manifest, channels, metadata_path = map(Path, (pilot_manifest, channels, metadata_path))
    output = Path(output)
    if metadata_path.name != "metadata.json":
        raise ValueError("channel metadata sidecar must be named metadata.json")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is non-empty; refusing overwrite: {output}")
    lock = output.parent / (output.name + ".stage.lock")
    _exclusive_lock(lock)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        channel_run_data = None
        if production:
            if channel_run is None:
                raise ValueError("channel_run is required in --production mode")
            channel_run = Path(channel_run)
            channel_run_data = json.loads(channel_run.read_text())
            if verify_freeze:
                from atomwise_ris.bench.production_freeze import verify_freeze as _verify
                from atomwise_ris.bench.production_pilot import registered_manifest
                manifest, scope, digest = registered_manifest(pilot_manifest, _verify())
            else:
                manifest = json.loads(pilot_manifest.read_text())
                digest = sha256(pilot_manifest)
                scope = manifest.get("run_scope", manifest.get("scope", "development_full"))
                if scope == "development":
                    scope = "development_challenge" if manifest.get("subset") == "challenge" else "development_full"
            from atomwise_ris.bench.production_run import PRODUCTION_SCOPES
            if scope not in PRODUCTION_SCOPES:
                raise ValueError(f"production mode requires scope in {PRODUCTION_SCOPES}, got '{scope}'")
            if channel_run_data.get("case_manifest_sha256") != digest:
                raise ValueError(f"channel_run case_manifest_sha256 mismatch: {channel_run_data.get('case_manifest_sha256')} != {digest}")
            failed_cases_set = set()
            for m_data in channel_run_data.get("maps", {}).values():
                failed_cases_set.update(m_data.get("failed_cases", []))
            failed_cases = sorted(failed_cases_set)
        else:
            if verify_freeze:
                from atomwise_ris.bench.production_freeze import verify_freeze as _verify
                from atomwise_ris.bench.production_pilot import registered_manifest
                manifest, scope, digest = registered_manifest(pilot_manifest, _verify())
            else:
                manifest, scope = json.loads(pilot_manifest.read_text()), "synthetic_diagnostic"
                digest = sha256(pilot_manifest)
            failed_cases_set = set()
            failed_cases = []

        metadata = json.loads(metadata_path.read_text())
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("manifest requires non-empty cases[]")
        if any(c.get("map_id") not in {"munich", "rotterdam", "maastricht"} for c in cases):
            raise ValueError("forbidden map in manifest")
        label_path = output / "optimization"
        label_path.mkdir()
        trajectory_rows = []
        label_index = []
        frame_keys = []
        all_channel_keys = []
        arrays = tuple(int(x) for x in partial_arrays) if partial_arrays else (64, 128, 256)
        if not arrays or any(n not in (64, 128, 256) for n in arrays) or len(set(arrays)) != len(arrays):
            raise ValueError("partial arrays must be a unique subset of 64,128,256")
        is_partial = set(arrays) != {64, 128, 256}
        label_frame_seconds = {}
        with ExitStack() as stack:
            cf = stack.enter_context(h5py.File(channels, "r"))
            single_file = stack.enter_context(h5py.File(label_path / "single_flip_gains.h5", "w"))
            pair_file = stack.enter_context(h5py.File(label_path / "pair_flip_gains.h5", "w"))
            synergy_file = stack.enter_context(h5py.File(label_path / "synergy.h5", "w"))
            for case in cases:
                city, case_id = str(case["map_id"]), str(case["id"])
                if city.lower() in {"berlin", "utrecht"}: raise ValueError("forbidden map")
                if production and case_id in failed_cases_set:
                    continue

                traj_idx = case.get("trajectory_index")
                if traj_idx is None:
                    try:
                        traj_idx = int(case_id.split("-")[-1])
                    except Exception:
                        traj_idx = 0

                if production:
                    if scope == "development_challenge":
                        row_split = "challenge"
                        kind = "challenge"
                        scenario = case.get("scenario")
                    else:
                        if traj_idx <= 83:
                            row_split = "train"
                        elif traj_idx <= 101:
                            row_split = "validation"
                        else:
                            row_split = "test"
                        kind = case.get("kind", "natural")
                        scenario = case.get("scenario", None)
                else:
                    row_split = "development"
                    kind = None
                    scenario = None

                case_frames = case.get("frames", [])
                for fr_idx, fr in enumerate(case_frames):
                    frame = str(int(fr["frame"]))
                    is_labelled_frame = (fr_idx in (0, 5, 10, 15)) if production else True

                    if "array" in case and not partial_arrays:
                        raise ValueError("manifest array restriction requires explicit --partial-arrays diagnostic mode")
                    case_arrays = (int(case["array"]),) if "array" in case else arrays
                    if set(case_arrays) != {64, 128, 256}: is_partial = True
                    if partial_arrays and not set(case_arrays).issubset(set(arrays)):
                        raise ValueError("case array field conflicts with --partial-arrays")

                    for n in case_arrays:
                        key = f"{city}/{n}/{case_id}/{frame}"
                        all_channel_keys.append(key)
                        if key not in cf: raise KeyError(f"channel HDF5 missing group /{key}")
                        g = cf[key]
                        md = metadata.get(key, metadata.get("/" + key, {}))

                        if is_labelled_frame:
                            label_started = time.perf_counter()
                            ev = _network(g, md)
                            if ev.n != n: raise ValueError(f"channel array mismatch at /{key}")
                            channel_attrs = {str(k): _jsonable(v) for k, v in g.attrs.items()}
                            greedy = greedy_parent(ev, depth=greedy_depth)
                            parents = [("all_off", np.zeros(n, dtype=np.int8)), ("single_greedy", greedy)]
                            frame_seed = int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "little")
                            for pi, (category, bits) in enumerate(parents):
                                if production:
                                    if n in (64, 128):
                                        full = True
                                    elif n == 256:
                                        full = (pi == 0) and (fr_idx == 0) and (traj_idx % 10 == 0)
                                    else:
                                        full = False
                                else:
                                    full = n == 256 and pi == 0
                                pairs = _pairs_for(n, pairs_256, frame_seed + pi, full)
                                out = label_group(ev, bits, pairs, category)
                                groups = [single_file.require_group(f"{key}/{category}"),
                                          pair_file.require_group(f"{key}/{category}"),
                                          synergy_file.require_group(f"{key}/{category}")]
                                sg, pg, yg = groups
                                for target in groups:
                                    target.attrs["base_rate_bps_hz"] = out["base_rate"]
                                    target.attrs["pair_coverage"] = "full" if len(pairs) == n * (n - 1) // 2 else "sampled"
                                    target.attrs["pair_seed"] = frame_seed + pi
                                    target.attrs["channel_attrs_json"] = json.dumps(channel_attrs, sort_keys=True, default=_jsonable)
                                    target.create_dataset("bits", data=out["bits"], compression="gzip")
                                sg.create_dataset("single_flip_gains", data=out["single_gain"], compression="gzip")
                                pg.create_dataset("pair_indices", data=out["pairs"], compression="gzip")
                                pg.create_dataset("pair_flip_gains", data=out["pair_gain"], compression="gzip")
                                yg.create_dataset("pair_indices", data=out["pairs"], compression="gzip")
                                yg.create_dataset("synergy", data=out["synergy"], compression="gzip")
                                label_index.append({"channel_key": key, "parent": category, "N": n,
                                    "pair_count": len(pairs), "pair_coverage": str(pg.attrs["pair_coverage"]),
                                    "base_rate_bps_hz": out["base_rate"], "channel_attrs": channel_attrs})
                            label_frame_seconds[key] = time.perf_counter() - label_started
                            frame_keys.append(key)

                        for user in fr.get("users", []):
                            link = _user_links(md, user)
                            row = {"case_id": case_id, "map_id": city,
                                "trajectory_id": str(case.get("trajectory_id", case_id)), "frame": int(fr["frame"]),
                                "time_s": float(fr.get("time_s", 0.0)),
                                "user_id": str(user["user_id"]), "user_x": float(user["xyz"][0]),
                                "user_y": float(user["xyz"][1]), "user_z": float(user["xyz"][2]),
                                "velocity_xyz": json.dumps(user.get("velocity_xyz", [0, 0, 0])),
                                "bs_link": str(link["bs_link"]), "ris_link": str(link["ris_link"]),
                                "bs_ris_link": str(link["bs_ris_link"]), "array_n": n,
                                "split": row_split}
                            if production:
                                row["kind"] = kind
                                row["scenario"] = scenario
                                row["trajectory_index"] = traj_idx
                                row["labelled"] = bool(is_labelled_frame)
                            trajectory_rows.append(row)
            if not trajectory_rows: raise ValueError("no user trajectory rows in manifest")
            for h5f in (single_file, pair_file, synergy_file):
                h5f.attrs["definition"] = "single_gain=score(toggle_i)-score(parent); pair_gain=score(toggle_i,j)-score(parent); synergy=pair_gain-single_i-single_j"
                h5f.attrs["channel_realizations_sha256"] = sha256(channels)
        import pandas as pd
        pd.DataFrame(trajectory_rows).to_parquet(output / "trajectories.parquet", index=False)
        maps = sorted({r["map_id"] for r in trajectory_rows})
        traj_ids = sorted({r["trajectory_id"] for r in trajectory_rows})
        split_dir = output / "splits"; split_dir.mkdir()
        if production:
            if scope == "development_full":
                within = {
                    "role": "production_development",
                    "method": "within-map trajectory split: 0-83 train, 84-101 validation, 102-119 test",
                    "per_map": {},
                    "train": [],
                    "validation": [],
                    "test": []
                }
                for map_id in maps:
                    map_trajs = sorted({(r["trajectory_index"], r["trajectory_id"])
                                        for r in trajectory_rows if r["map_id"] == map_id})
                    m_train = [t for idx, t in map_trajs if idx <= 83]
                    m_val = [t for idx, t in map_trajs if 84 <= idx <= 101]
                    m_test = [t for idx, t in map_trajs if 102 <= idx <= 119]
                    within["per_map"][map_id] = {
                        "train": m_train,
                        "validation": m_val,
                        "test": m_test
                    }
                    within["train"].extend(m_train)
                    within["validation"].extend(m_val)
                    within["test"].extend(m_test)
                (split_dir / "within_map_trajectory.json").write_text(json.dumps(within, indent=2) + "\n")

                folds = []
                dev_maps = [m for m in ("munich", "rotterdam", "maastricht") if m in maps]
                for test_map in dev_maps:
                    train_maps = [m for m in dev_maps if m != test_map]
                    test_trajs = sorted({r["trajectory_id"] for r in trajectory_rows if r["map_id"] == test_map})
                    train_trajs = sorted({r["trajectory_id"] for r in trajectory_rows if r["map_id"] in train_maps})
                    folds.append({
                        "fold_id": f"{test_map}_test",
                        "test_map": test_map,
                        "train_maps": train_maps,
                        "test_trajectories": test_trajs,
                        "train_trajectories": train_trajs
                    })
                leo = {
                    "role": "production_development",
                    "method": "leave-environment-out rotation (geographical cross-validation)",
                    "folds": folds
                }
                (split_dir / "leave_environment_out.json").write_text(json.dumps(leo, indent=2) + "\n")
            elif scope == "development_challenge":
                all_ch = sorted({r["trajectory_id"] for r in trajectory_rows})
                ch = {
                    "role": "production_challenge",
                    "evaluation_only_never_mixed": True,
                    "trajectories": all_ch,
                    "per_map": {
                        m: sorted({r["trajectory_id"] for r in trajectory_rows if r["map_id"] == m})
                        for m in maps
                    }
                }
                (split_dir / "challenge.json").write_text(json.dumps(ch, indent=2) + "\n")
        else:
            within = {"method": "deterministic trajectory hash partition within each map", "role": "development_only",
                      "trajectory_ids": traj_ids, "per_map": {}}
            for map_id in maps:
                mt = sorted({r["trajectory_id"] for r in trajectory_rows if r["map_id"] == map_id})
                nf = min(5, max(2, len(mt)))
                folds = []
                for fold in range(nf):
                    val = [t for t in mt if int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % nf == fold]
                    folds.append({"fold": fold, "validation_trajectories": val,
                                  "train_trajectories": [t for t in mt if t not in val],
                                  "usable_for_model_fit": bool(val and len(mt) > 1 and len(mt) > len(val))})
                within["per_map"][map_id] = {"trajectory_count": len(mt), "folds": folds,
                                              "pilot_split_warning": "single-trajectory map cannot provide a within-map train/validation split" if len(mt) < 2 else None}
            loo = {"method": "leave-map-out cross-validation among development maps", "role": "development_only",
                   "held_out_map_is_not_test": True,
                   "folds": [{"map_validation": m, "train_maps": [x for x in maps if x != m]} for m in maps]}
            (split_dir / "within_map_trajectory.json").write_text(json.dumps(within, indent=2) + "\n")
            (split_dir / "leave_map_out_development.json").write_text(json.dumps(loo, indent=2) + "\n")

        package_channels = output / "channels"
        package_channels.mkdir()
        packaged_channels = package_channels / "channel_realizations.h5"
        shutil.copy2(channels, packaged_channels)
        packaged_channel_sha = sha256(packaged_channels)
        source_channel_sha = sha256(channels)
        if packaged_channel_sha != source_channel_sha:
            raise IOError("copied channel HDF5 hash differs from checked source")
        shutil.copy2(metadata_path, package_channels / "metadata.json")
        examples_dir = output / "examples"
        examples_dir.mkdir()
        packaged_loader = examples_dir / "load_dataset.py"
        shutil.copy2(Path(__file__).resolve().parents[1] /
                     "bench/CoupledRIS-Bench/examples/load_dataset.py", packaged_loader)

        if production:
            package_role = "production_challenge" if scope == "development_challenge" else "production_development"
        else:
            package_role = "partial_diagnostic_only" if is_partial else "pilot_development"

        provenance = {"format": "coupledris-bench-production-package-v1" if production else "coupledris-bench-pilot-package-v1",
            "development_only": True,
            "package_role": package_role,
            "run_scope": scope,
            "package_path": str(output.resolve()),
            "generator_freeze_sha256": manifest.get("freeze_sha256"),
            "code_sha256": {"package_bench_production.py": sha256(Path(__file__)),
                            "hybrid/operator.py": sha256(Path(inspect.getfile(HybridCoupledNetwork)))},
            "packaging_runtime_versions": {"python": sys.version.split()[0],
                **{name: version(name) for name in ("numpy", "h5py", "pandas", "pyarrow")}},
            "pilot_manifest_sha256": digest,
            "manifest_sha256": digest,
            "channel_realizations_sha256": packaged_channel_sha,
            "metadata_sha256": sha256(metadata_path),
            "label_files_sha256": {name: sha256(label_path / name) for name in
                ("single_flip_gains.h5", "pair_flip_gains.h5", "synergy.h5")},
            "trajectories_sha256": sha256(output / "trajectories.parquet"),
            "loader_sha256": sha256(packaged_loader),
            "split_files_sha256": {path.name: sha256(path) for path in sorted(split_dir.glob("*.json"))},
            "seed": seed,
            "arrays": list(arrays), "pairs_256": pairs_256, "greedy_depth": greedy_depth, "label_records": label_index,
            "label_frame_seconds": label_frame_seconds,
            "channel_keys": all_channel_keys if production else frame_keys,
            "labelled_keys": frame_keys,
            "maps": maps, "source_paths": {"manifest": str(pilot_manifest),
            "channels": str(channels), "metadata": str(metadata_path)},
            "packaged_channel_path": "channels/channel_realizations.h5",
            "runtime_s": time.time() - started,
            "raw_per_path_archives_included": False}
        if production:
            provenance["failed_cases"] = failed_cases
            provenance["excluded_failed_cases"] = failed_cases
            if channel_run is not None:
                provenance["source_paths"]["channel_run"] = str(channel_run)
                provenance["channel_run_sha256"] = sha256(Path(channel_run))
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=_jsonable) + "\n")
        (output / "DONE").write_text("complete\n")
        return provenance
    except Exception:
        (output / "FAILED").write_text("failed; inspect caller traceback\n")
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production", action="store_true", help="run in full production packaging mode")
    p.add_argument("--manifest", type=Path, help="manifest path in production mode")
    p.add_argument("--pilot-manifest", type=Path, help="pilot manifest path (non-production mode)")
    p.add_argument("--channels", required=True, type=Path)
    p.add_argument("--metadata", type=Path, help="defaults to metadata.json beside --channels")
    p.add_argument("--channel-run", type=Path, help="required in --production mode")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--pairs-256", type=int, default=4096)
    p.add_argument("--greedy-depth", type=int, default=4)
    p.add_argument("--partial-arrays", help="comma-separated diagnostic subset; package is marked partial and cannot pass pilot gate")
    a = p.parse_args(argv)
    manifest = a.manifest if a.production else (a.pilot_manifest or a.manifest)
    if not manifest:
        if a.production:
            p.error("--manifest is required when running with --production")
        else:
            p.error("--pilot-manifest is required")
    if a.production and not a.channel_run:
        p.error("--channel-run is required when running with --production")
    metadata = a.metadata or a.channels.parent / "metadata.json"
    partial = [int(x) for x in a.partial_arrays.split(",")] if a.partial_arrays else None
    result = package(manifest, a.channels, metadata, a.output, a.seed,
                     a.pairs_256, a.greedy_depth, True, partial,
                     production=a.production, channel_run=a.channel_run)
    print(json.dumps(result, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
