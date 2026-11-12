#!/usr/bin/env python3
"""Rescore saved pair-plus DEV actions on coarse coupled-wire frames.

This is intentionally not a mesh-only comparison unless verified fine matrices
are supplied through a future explicit extension. Saved fine-run rates use a
fine-derived precoder; this script uses a single coarse all-OFF precoder per
frame for every method and action and keeps the two protocols separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atomwise_ris.physics.network import CoupledNetwork

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pair", "pair_plus_b0", "pair_plus_b3", "pair_plus_b5", "pair_plus_b35")
SEEDS = (3, 7, 11, 19, 23, 31, 43, 59, 67, 83)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap(values: np.ndarray, seed: int = 8251, draws: int = 10_000):
    """Paired trajectory bootstrap of a vector of per-trajectory contrasts."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[indices].mean(axis=1)
    return [float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def rate_on_fixed_precoder(evaluator, bits: np.ndarray, w: np.ndarray) -> float:
    """Score an action with the caller's shared W; never request an action-specific precoder."""
    w = np.asarray(w, dtype=np.complex128)
    if w.ndim != 2 or w.shape[0] != 4:
        raise ValueError("common precoder must have shape (4, users)")
    return float(evaluator.score(evaluator.channel(np.asarray(bits, dtype=np.int8)) @ w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="outputs/runs/pair_plus_ladder_v2_combined_n3/combined_dev161.jsonl")
    ap.add_argument("--data", default="data/processed/coupled_wire_dev_v1")
    ap.add_argument("--fine-data", default="outputs/runs/hybrid_track2_20261108/input_fine161")
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget-ms", type=float, default=50.0)
    args = ap.parse_args()

    rows_path, data_dir, fine_dir, out = ((ROOT / args.rows).resolve(), (ROOT / args.data).resolve(),
                                           (ROOT / args.fine_data).resolve(), (ROOT / args.output).resolve())
    if "test" in data_dir.parts or data_dir.name != "coupled_wire_dev_v1":
        raise ValueError("only the registered 81-segment coupled_wire_dev_v1 dataset is allowed")
    if not rows_path.is_file() or not (data_dir / "scenario.json").is_file():
        raise FileNotFoundError("missing registered DEV inputs")
    if out.exists():
        raise FileExistsError(f"preserving existing output: {out}")

    scenario = json.loads((data_dir / "scenario.json").read_text())
    if scenario.get("split_label") != "dev" or not scenario.get("development_set"):
        raise ValueError("scenario is not registered DEV data")
    row_lookup = {int(x["index"]): x for x in scenario["rows"] if x.get("split") == "dev"}
    if len(row_lookup) != 480:
        raise ValueError("expected the complete 480-frame DEV scenario")
    fine_scenario_path, fine_manifest_path = fine_dir / "scenario.json", fine_dir / "generation_status.json"
    fine_scenario = json.loads(fine_scenario_path.read_text())
    fine_manifest = json.loads(fine_manifest_path.read_text())
    if (Path(str(fine_scenario.get("fine_mesh_of", ""))).name != "coupled_wire_dev_v1"
            or fine_scenario.get("fine_mesh_split") != "dev"
            or fine_scenario.get("segments") != 161
            or fine_scenario.get("source_scenario_sha256") != sha256(data_dir / "scenario.json")):
        raise ValueError("fine mesh scenario does not identify this exact coarse DEV scenario")
    if (fine_manifest.get("status") != "complete" or not fine_manifest.get("ready_for_development_evaluation")
            or len(fine_manifest.get("verified_frame_indices", [])) != 480
            or fine_manifest.get("missing_frame_indices") or fine_manifest.get("invalid_frames")):
        raise ValueError("fine DEV generation manifest is incomplete or invalid")
    fine_rows = {int(x["index"]): x for x in fine_scenario["rows"] if x.get("split") == "dev"}
    if len(fine_rows) != 480:
        raise ValueError("fine scenario does not contain all 480 DEV rows")
    for idx in range(480):
        if row_lookup[idx] != fine_rows[idx]:
            raise ValueError(f"coarse/fine scenario row mismatch at {idx}")

    selected = []
    with rows_path.open() as f:
        for line in f:
            r = json.loads(line)
            if (r.get("split") == "development" and r.get("budget_ms") == args.budget_ms
                    and r.get("method") in METHODS):
                selected.append(r)
    expected_rows = 480 * len(METHODS) * len(SEEDS)
    if len(selected) != expected_rows:
        raise ValueError(f"expected {expected_rows} method/seed/frame rows; found {len(selected)}")
    keyed = {}
    for r in selected:
        key = (int(r["index"]), str(r["method"]), int(r["seed"]))
        if key in keyed or int(r["seed"]) not in SEEDS:
            raise ValueError(f"duplicate or unexpected source row: {key}")
        idx = key[0]
        meta = row_lookup[idx]
        if (int(r["trajectory"]) != int(meta["trajectory"])
                or int(r["step"]) != int(meta["step"])):
            raise ValueError(f"row identity mismatch for frame {idx}")
        bits = np.asarray(r["bits"], dtype=np.int8)
        if bits.shape != (64,) or not np.isin(bits, (0, 1)).all():
            raise ValueError(f"invalid returned state for {key}")
        keyed[key] = r

    frame_records = []
    gate_path = ROOT / "outputs/runs/matched_coupled_v1/physics_gate.json"
    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        raise ValueError("registered coupled physics gate is not passed")
    loads = np.asarray(gate["loads_real_ohm"], dtype=float) + 1j * np.asarray(gate["loads_imag_ohm"], dtype=float)
    for idx in range(480):
        p = data_dir / f"frame_{idx:04d}.npz"
        if not p.is_file():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as zdata:
            z = np.asarray(zdata["z"], dtype=np.complex128)
        if z.shape != (70, 70):
            raise ValueError(f"unexpected 81-segment frame matrix shape {z.shape} at {idx}")
        # Construct once with its all-OFF reference precoder, then retain that W
        # explicitly for every source action from every method and seed.
        reference = CoupledNetwork(z, loads)
        w = reference.w.copy()
        ev = CoupledNetwork(z, loads, w=w)
        method_seed_rates, method_seed_fine_rates = {}, {}
        source_rates = {}
        fine_path = fine_dir / f"frame_{idx:04d}.npz"
        expected_hash = fine_manifest["frame_npz_sha256"].get(str(idx))
        if not fine_path.is_file() or sha256(fine_path) != expected_hash:
            raise ValueError(f"fine-frame provenance mismatch for index {idx}")
        with np.load(fine_path, allow_pickle=False) as fzdata:
            zf = np.asarray(fzdata["z"], dtype=np.complex128)
        if zf.shape != (70, 70):
            raise ValueError(f"unexpected 161-segment frame matrix shape {zf.shape} at {idx}")
        fine_ev = CoupledNetwork(zf, loads, w=w)
        for method in METHODS:
            vals, fine_vals, source_vals = [], [], []
            for seed in SEEDS:
                row = keyed[idx, method, seed]
                bits = np.asarray(row["bits"], dtype=np.int8)
                vals.append(rate_on_fixed_precoder(ev, bits, w))
                fine_vals.append(rate_on_fixed_precoder(fine_ev, bits, w))
                source_vals.append(float(row["rate"]))
            method_seed_rates[method] = float(np.mean(vals))
            method_seed_fine_rates[method] = float(np.mean(fine_vals))
            source_rates[method] = float(np.mean(source_vals))
        frame_records.append({"index": idx, "trajectory": int(row_lookup[idx]["trajectory"]),
                              "step": int(row_lookup[idx]["step"]), "coarse_81_fixedW": method_seed_rates,
                              "fine_161_fixed_coarseW": method_seed_fine_rates,
                              "source_161_ownW": source_rates})

    contrasts = {}
    mesh_contrasts = {}
    for candidate in METHODS[1:]:
        per_mesh = {}
        for mesh_key in ("coarse_81_fixedW", "fine_161_fixed_coarseW"):
            diffs = defaultdict(list)
            for f in frame_records:
                diffs[f["trajectory"]].append(f[mesh_key][candidate] - f[mesh_key]["pair"])
            traj = np.asarray([np.mean(diffs[t]) for t in sorted(diffs)])
            per_mesh[mesh_key] = {
                "paired_trajectory_mean_ci95": bootstrap(traj),
                "trajectory_means": {str(t): float(np.mean(diffs[t])) for t in sorted(diffs)},
                "trajectories_won": int(np.sum(traj > 0)),
                "frame_mean": float(np.mean([f[mesh_key][candidate] - f[mesh_key]["pair"] for f in frame_records])),
            }
        contrasts[f"{candidate}-pair"] = per_mesh["coarse_81_fixedW"]
        mesh_contrasts[f"{candidate}-pair"] = per_mesh

    mesh_shifts = {}
    for method in METHODS:
        by_traj = defaultdict(list)
        per_frame = []
        for f in frame_records:
            delta = f["fine_161_fixed_coarseW"][method] - f["coarse_81_fixedW"][method]
            by_traj[f["trajectory"]].append(delta)
            per_frame.append(delta)
        trajectory_means = np.asarray([np.mean(by_traj[t]) for t in sorted(by_traj)])
        mesh_shifts[method] = {
            "fine_minus_coarse_per_trajectory_mean_ci95": bootstrap(trajectory_means),
            "trajectory_means": {str(t): float(np.mean(by_traj[t])) for t in sorted(by_traj)},
            "mean_shift": float(np.mean(per_frame)),
            "median_shift": float(np.median(per_frame)),
            "median_absolute_shift": float(np.median(np.abs(per_frame))),
            "p95_absolute_shift": float(np.quantile(np.abs(per_frame), 0.95)),
            "maximum_absolute_shift": float(np.max(np.abs(per_frame))),
        }

    # Exact action ordering reversals across the two meshes under common W.
    reversals = {}
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            key = f"{a}__vs__{b}"
            flip_count = 0
            for f in frame_records:
                coarse_d = f["coarse_81_fixedW"][a] - f["coarse_81_fixedW"][b]
                fine_d = f["fine_161_fixed_coarseW"][a] - f["fine_161_fixed_coarseW"][b]
                if coarse_d * fine_d < 0:
                    flip_count += 1
            reversals[key] = {"reversed_frames": flip_count, "frames": len(frame_records)}

    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete_dev_mesh_rescore",
        "scope": "DEV only; exact saved 161-run actions re-scored on matching 81 and 161 DEV frames",
        "primary_mesh_only_comparison": "both meshes scored with one 81-segment all-OFF RZF precoder per frame, shared across methods, seeds, and actions",
        "secondary_source_rate_protocol": "raw saved 161 rate uses fine-derived per-frame W and is descriptive only",
        "budget_ms": args.budget_ms, "methods": list(METHODS), "seeds": list(SEEDS),
        "frames": len(frame_records), "trajectories": 30,
        "input_sha256": {"source_jsonl": sha256(rows_path), "coarse_scenario": sha256(data_dir / "scenario.json"),
                         "fine_scenario": sha256(fine_scenario_path), "fine_generation_manifest": sha256(fine_manifest_path),
                         "physics_gate": sha256(gate_path),
                         "coarse_frames": {str(i): sha256(data_dir / f"frame_{i:04d}.npz") for i in range(480)},
                         "fine_frames": {str(i): sha256(fine_dir / f"frame_{i:04d}.npz") for i in range(480)}},
        "input_file_count": 1 + 4 + 2 * 480 + 1,
        "script_sha256": sha256(Path(__file__).resolve()),
        "contrasts": contrasts, "mesh_contrasts": mesh_contrasts,
        "mesh_shifts": mesh_shifts, "ranking_reversals": reversals,
        "frame_records": frame_records,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(out), "frames": len(frame_records),
                      "contrasts": contrasts, "ranking_reversals": reversals}, sort_keys=True))


if __name__ == "__main__":
    main()
