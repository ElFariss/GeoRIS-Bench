"""Independent monolithic NEC port and open-structure reference for Munich P4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from atomwise_ris.physics.nec import WireGeometry


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def direct_transfer(z_external: np.ndarray, n_bs: int = 1) -> np.ndarray:
    n = len(z_external)
    if z_external.shape != (n, n) or not 0 < n_bs < n:
        raise ValueError("bad external-port matrix or BS partition")
    return -100.0 * np.linalg.inv(z_external + 50.0 * np.eye(n))[n_bs:, :n_bs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--segments", type=int, default=81)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.segments != 81:
        raise ValueError("P4 monolithic reference is fixed at 81 segments per wire")
    layout_path, selection_path = Path(args.layout), Path(args.selection)
    layout = json.loads(layout_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "selected" or selection.get("split") != "train":
        raise ValueError("selection must be a completed training-only LOS selection")
    if selection.get("layout_sha256") != sha256(layout_path):
        raise ValueError("selection and NEC geometry do not share the same layout hash")
    if selection.get("scene") != "Sionna built-in Munich":
        raise ValueError("wrong scene selection")
    frequency = float(selection["frequency_hz"])
    if abs(frequency - 5.53e9) > 1:
        raise ValueError("unexpected carrier frequency")

    center = np.asarray(layout["ris_center"], dtype=float)
    normal = np.asarray(layout["panel_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0], 0.0])
    tangent /= np.linalg.norm(tangent)
    ris = np.asarray([
        center + (j - 3.5) * .028 * tangent + (i - 3.5) * np.array([0., 0., 1.]) * .030
        for i in range(8) for j in range(8)
    ])
    bs = np.asarray([selection["tx_xyz_m"]], dtype=float)
    ue = np.asarray([selection["ue_xyz_m"]], dtype=float)
    full_centers = np.vstack([ris, bs, ue])
    ext_centers = np.vstack([bs, ue])
    lengths = np.full(len(full_centers), .024)
    start = time.perf_counter()
    z_full, timing_full = WireGeometry(full_centers, lengths, radius=.00005,
                                       frequency=frequency).impedance(args.segments)
    z_free, timing_free = WireGeometry(ext_centers, np.full(len(ext_centers), .024), radius=.00005,
                                       frequency=frequency).impedance(args.segments)
    elapsed = time.perf_counter() - start

    # This is the exact external block of the full monolithic NEC solution with
    # the physical continuous RIS wires present. The difference from the same
    # BS/UE NEC solve without RIS is the independent open-structure delta-Z.
    n_ris = len(ris)
    z_ee_mono = z_full[n_ris:, n_ris:].copy()
    z_ee_free = z_free.copy()
    dz_open = z_ee_mono - z_ee_free
    h_open = direct_transfer(z_ee_mono, n_bs=1)
    h_free = direct_transfer(z_ee_free, n_bs=1)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "munich_los_nec_reference.npz"
    np.savez_compressed(
        data_path, z_full=z_full, z_ee_monolithic_open=z_ee_mono,
        z_ee_without_ris=z_ee_free, delta_z_open_structure=dz_open,
        h_direct_monolithic_open=h_open, h_direct_without_ris=h_free,
        ris_centers_xyz_m=ris, bs_centers_xyz_m=bs, ue_centers_xyz_m=ue,
    )
    summary = {
        "status": "complete", "scope": "single selected Munich training LOS, independent monolithic NEC port reference",
        "split": selection["split"], "frame_index": selection["frame_index"],
        "trajectory_id": selection["trajectory_id"], "user_index": selection["user_index"],
        "selection_sha256": sha256(selection_path), "layout_sha256": sha256(layout_path),
        "frequency_hz": frequency, "segments_per_wire": args.segments,
        "wire_counts": {"RIS": len(ris), "BS": len(bs), "UE": len(ue), "total": len(full_centers)},
        "unknowns_monolithic": len(full_centers) * args.segments,
        "coordinates": {"ris_m": ris.tolist(), "bs_m": bs.tolist(), "ue_m": ue.tolist()},
        "port_order": [f"RIS{i:02d}" for i in range(n_ris)] + ["BS0", "UE0"],
        "direct_channel_convention": "-100 * inv(Z_external + 50*I)[UE, BS], identical declared hybrid port convention",
        "open_structure_delta_definition": "Z_EE from monolithic RIS+BS+UE NEC minus Z_EE from BS+UE-alone NEC; no RT-derived values or fitting",
        "timing_full_s": timing_full["seconds"], "timing_free_s": timing_free["seconds"],
        "elapsed_s_wall": elapsed,
        "matrix_diagnostics": {
            "z_full_shape": list(z_full.shape),
            "z_full_reciprocity_relative": float(np.linalg.norm(z_full - z_full.T) / np.linalg.norm(z_full)),
            "z_ee_open_shape": list(z_ee_mono.shape),
            "delta_z_open_norm_ohm": float(np.linalg.norm(dz_open)),
            "h_direct_open_real": h_open.real.tolist(), "h_direct_open_imag": h_open.imag.tolist(),
            "h_direct_free_real": h_free.real.tolist(), "h_direct_free_imag": h_free.imag.tolist(),
        },
        "calibration": "none; RT data are not used to tune NEC or vice versa",
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy")},
        "artifact": data_path.name, "artifact_sha256": sha256(data_path),
    }
    summary_path = output / "munich_los_nec_reference.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
