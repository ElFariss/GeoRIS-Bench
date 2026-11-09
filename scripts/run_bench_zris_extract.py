"""Extract one RIS-only NEC port-impedance matrix into a compact HDF5 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from atomwise_ris.physics.nec import WireGeometry


def lattice(elements: int) -> tuple[np.ndarray, int, int]:
    shape = {64: (8, 8), 128: (8, 16), 256: (16, 16)}.get(elements)
    if shape is None:
        raise ValueError("elements must be one of 64, 128, or 256")
    nr, nc = shape
    centers = np.asarray(
        [[0.0, (j - (nc - 1) / 2) * 0.028,
          1.0 + (i - (nr - 1) / 2) * 0.030]
         for i in range(nr) for j in range(nc)], dtype=np.float64)
    return centers, nr, nc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", type=int, required=True)
    ap.add_argument("--segments", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.segments not in (81, 161):
        raise ValueError("segments must be 81 or 161")
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError("h5py is required (install into the task-local environment)") from e

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers, nr, nc = lattice(args.elements)
    geometry = WireGeometry(centers, np.full(args.elements, 0.024),
                            radius=0.00005, frequency=5.53e9)
    start = time.perf_counter()
    z, timing = geometry.impedance(segments=args.segments)
    elapsed = time.perf_counter() - start
    z = np.asarray(z, dtype=np.complex128)
    if z.shape != (args.elements, args.elements) or not np.isfinite(z).all():
        raise RuntimeError(f"invalid Z_RIS result: shape={z.shape}")

    h5_path = out / "impedance.h5"
    with h5py.File(h5_path, "w") as f:
        d = f.create_dataset("Z_RIS_ohm", data=z, compression="gzip", compression_opts=4,
                             shuffle=True, fletcher32=True)
        d.attrs["frequency_hz"] = 5.53e9
        d.attrs["elements"] = args.elements
        d.attrs["segments_per_wire"] = args.segments
        d.attrs["units"] = "ohm"
        d.attrs["matrix_convention"] = "port impedance Z = inverse(port admittance Y); V=Z I"
        d.attrs["wire_states"] = "PEC thin-wire baseline; no ON/OFF loads"
        f.attrs["scope"] = "RIS-only NEC-2; no BS/UE wires, no optimizer"

    # Stable diagnostics for a passive reciprocal multiport.
    herm = (z + z.conj().T) / 2
    eig = np.linalg.eigvalsh(herm)
    z0 = 50.0
    scattering = np.linalg.solve((z + z0 * np.eye(args.elements)).T,
                                 (z - z0 * np.eye(args.elements)).T).T
    smax = float(np.linalg.svd(scattering, compute_uv=False)[0])
    metrics = {
        "status": "complete", "scope": "RIS-only NEC-2 port impedance; no BS/UE, no optimizer",
        "frequency_hz": 5.53e9, "elements": args.elements,
        "array_rows": nr, "array_columns": nc,
        "segments_per_wire": args.segments,
        "nec_unknowns": args.elements * args.segments,
        "elapsed_wall_s": elapsed, "nec_reported_s": timing["seconds"],
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "matrix_shape": list(z.shape),
        "reciprocity_relative_fro": float(np.linalg.norm(z - z.T) / np.linalg.norm(z)),
        "minimum_hermitian_resistance_eigenvalue_ohm": float(eig.min()),
        "maximum_scattering_singular_value": smax,
        "condition_number_2": float(np.linalg.cond(z)),
        "z_frobenius_norm_ohm": float(np.linalg.norm(z)),
        "z_max_abs_ohm": float(np.max(np.abs(z))),
        "hdf5_path": str(h5_path), "hdf5_sha256": sha256(h5_path),
    }
    metric_path = out / "metrics.json"
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
