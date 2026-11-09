"""Compare 81- and 161-segment RIS-only Z_RIS matrices and emit mesh evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_mesh_pair(n_coarse: int, s_coarse: int, n_fine: int, s_fine: int) -> None:
    if n_coarse != n_fine or s_coarse != 81 or s_fine != 161:
        raise ValueError(
            f"incompatible meshes: coarse={(n_coarse, s_coarse)}, "
            f"fine={(n_fine, s_fine)}"
        )


def main() -> None:
    import h5py

    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", required=True)
    ap.add_argument("--fine", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with h5py.File(a.coarse, "r") as f:
        z81 = f["Z_RIS_ohm"][:]
        n81, s81 = int(f["Z_RIS_ohm"].attrs["elements"]), int(f["Z_RIS_ohm"].attrs["segments_per_wire"])
    with h5py.File(a.fine, "r") as f:
        z161 = f["Z_RIS_ohm"][:]
        n161, s161 = int(f["Z_RIS_ohm"].attrs["elements"]), int(f["Z_RIS_ohm"].attrs["segments_per_wire"])
    validate_mesh_pair(n81, s81, n161, s161)
    delta = z161 - z81
    scale = float(np.linalg.norm(z161))
    result = {
        "status": "complete", "elements": n81,
        "coarse_segments": s81, "fine_segments": s161,
        "relative_frobenius_difference": float(np.linalg.norm(delta) / scale),
        "maximum_absolute_entry_difference_ohm": float(np.max(np.abs(delta))),
        "maximum_entry_difference_over_fine_frobenius": float(np.max(np.abs(delta)) / scale),
        "median_diagonal_relative_difference": float(np.median(np.abs(np.diag(delta)) / np.maximum(np.abs(np.diag(z161)), 1e-15))),
        "coarse_reciprocity_relative_fro": float(np.linalg.norm(z81-z81.T)/np.linalg.norm(z81)),
        "fine_reciprocity_relative_fro": float(np.linalg.norm(z161-z161.T)/np.linalg.norm(z161)),
        "interpretation": "mesh sensitivity between these two NEC discretizations; not physical calibration",
    }
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
