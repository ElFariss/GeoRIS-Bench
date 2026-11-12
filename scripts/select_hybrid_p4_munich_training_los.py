"""Deterministically select the first training UE with an actual Munich LOS path."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layout_path = Path(args.layout)
    layout = json.loads(layout_path.read_text())
    frame_indices = [i for i, split in enumerate(layout["split"]) if split == "train"]
    candidates = [(i, u) for i in frame_indices for u in range(len(layout["users"][i]))]
    if not candidates:
        raise RuntimeError("No registered training users")
    bs_idx = 0
    tx = np.asarray(layout["bs"][bs_idx], dtype=float)
    rx = np.asarray([layout["users"][i][u] for i, u in candidates], dtype=float)

    import mitsuba as mi
    import sionna.rt as rt

    scene = rt.load_scene(rt.scene.munich)
    scene.frequency = 5.53e9
    ant = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                         horizontal_spacing=.5, pattern="iso", polarization="V")
    scene.tx_array = scene.rx_array = ant
    scene.add(rt.Transmitter("selection_bs0", position=tx.tolist()))
    for j, p in enumerate(rx):
        scene.add(rt.Receiver(f"training_candidate_{j:03d}", position=p.tolist()))
    paths = rt.PathSolver()(scene=scene, max_depth=0, los=True,
                            specular_reflection=False, diffuse_reflection=False,
                            diffraction=False, refraction=False,
                            samples_per_src=1, max_num_paths_per_src=1,
                            synthetic_array=False, seed=7)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    if a.ndim != 6 or a.shape[0] != len(candidates):
        raise RuntimeError(f"Unexpected direct LOS CIR shape: {a.shape}")
    coeff = a[:, 0, 0, 0, 0, 0]
    valid = np.isfinite(coeff.real) & np.isfinite(coeff.imag) & (np.abs(coeff) > 0.0)
    chosen = next((j for j, ok in enumerate(valid) if ok), None)
    if chosen is None:
        raise RuntimeError("No finite nonzero direct LOS link in registered training rows")
    frame, user = candidates[chosen]
    result = {
        "status": "selected", "selection_rule": "first training row index, then user index; BS index 0; first finite nonzero LOS-only CIR",
        "layout_sha256": sha256(layout_path), "frame_index": frame,
        "trajectory_id": int(layout["trajectory"][frame]), "split": layout["split"][frame],
        "user_index": user, "bs_index": bs_idx,
        "tx_xyz_m": tx.tolist(), "ue_xyz_m": rx[chosen].tolist(),
        "los_channel_real": float(coeff[chosen].real), "los_channel_imag": float(coeff[chosen].imag),
        "delay_s": np.asarray(tau).reshape(len(candidates), -1)[chosen].tolist(),
        "training_candidates_checked": int(chosen + 1),
        "total_training_candidates": len(candidates),
        "scene": "Sionna built-in Munich", "frequency_hz": 5.53e9,
        "ray_settings": {"max_depth": 0, "los": True, "specular_reflection": False,
                         "diffuse_reflection": False, "diffraction": False,
                         "refraction": False, "seed": 7},
        "versions": {n: importlib.metadata.version(n)
                     for n in ("numpy", "sionna-rt", "mitsuba", "drjit")},
        "mitsuba_variant": mi.variant(), "raw_paths_exported": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
