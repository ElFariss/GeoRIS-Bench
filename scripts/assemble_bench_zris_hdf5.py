"""Assemble segment-specific extracts into physics/array_N/impedance.h5."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-dir", required=True)
    a = ap.parse_args()
    root = Path(a.array_dir)
    dest = root / "impedance.h5"
    tmp = root / "impedance.h5.tmp"
    with h5py.File(tmp, "w") as out:
        out.attrs["scope"] = "RIS-only NEC-2 port impedance; no BS/UE, no optimizer"
        segments = (81,) if (root.name == "array_256" and not (root / "161" / "impedance.h5").exists()) else (81, 161)
        out.attrs["assembled_segments_per_wire"] = list(segments)
        for seg in segments:
            source = root / str(seg) / "impedance.h5"
            with h5py.File(source, "r") as inp:
                inp.copy("Z_RIS_ohm", out, name=f"segments_{seg}/Z_RIS_ohm")
        out.flush()
    tmp.replace(dest)
    print(dest)


if __name__ == "__main__":
    main()
