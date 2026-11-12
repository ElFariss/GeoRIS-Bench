"""Bounded RIS-alone NEC impedance extraction scale probe (no optimizer use)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from atomwise_ris.physics.nec import WireGeometry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", type=int, default=128)
    ap.add_argument("--segments", type=int, default=81)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.elements != 128 or args.segments != 81:
        raise ValueError("This registered bounded probe is only 128 elements × 81 segments")
    nrow, ncol = 8, 16
    if nrow * ncol != args.elements:
        raise ValueError("Bad probe lattice")
    centers = np.asarray([[0.0, (j - (ncol - 1) / 2) * .028,
                           1.0 + (i - (nrow - 1) / 2) * .030]
                          for i in range(nrow) for j in range(ncol)], dtype=float)
    geometry = WireGeometry(centers, np.full(args.elements, .024),
                            radius=.00005, frequency=5.53e9)
    started = time.perf_counter()
    z, timing = geometry.impedance(segments=args.segments)
    elapsed = time.perf_counter() - started
    if z.shape != (args.elements, args.elements) or not np.isfinite(z).all():
        raise RuntimeError(f"Invalid impedance matrix: shape={z.shape}")
    result = {
        "status": "complete", "scope": "RIS-alone NEC impedance extraction; no BS/UE, no optimizer",
        "frequency_hz": 5.53e9, "elements": args.elements,
        "segments_per_wire": args.segments, "unknowns": args.elements * args.segments,
        "lattice": {"rows": nrow, "columns": ncol, "pitch_y_m": .028,
                    "pitch_z_m": .030, "wire_length_m": .024, "radius_m": .00005},
        "elapsed_s_wall": elapsed, "nec_reported_seconds": timing["seconds"],
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "matrix_shape": list(z.shape),
        "reciprocity_relative": float(np.linalg.norm(z - z.T) / np.linalg.norm(z)),
        "matrix_frobenius_norm": float(np.linalg.norm(z)),
        "matrix_max_abs_ohm": float(np.max(np.abs(z))),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
