"""Write deterministic geometry metadata for the benchmark NEC array sizes."""
from __future__ import annotations

import json
from pathlib import Path

from run_bench_zris_extract import lattice


def main() -> None:
    root = Path("bench/CoupledRIS-Bench/physics")
    for n in (64, 128, 256):
        centers, rows, cols = lattice(n)
        metadata = {
            "status": "geometry_prepared_extraction_not_implied",
            "elements": n,
            "array_rows": rows,
            "array_columns": cols,
            "frequency_hz": 5.53e9,
            "coordinate_frame": {
                "origin": "panel center projection at x=0,z=1 m",
                "x": "panel normal; all element centers x=0 m",
                "y": "horizontal lattice axis, 28 mm pitch",
                "z": "vertical lattice axis, 30 mm pitch; center z=1 m",
                "units": "m",
            },
            "wire": {
                "axis": "+z", "length_m": 0.024, "radius_m": 0.00005,
                "material_model": "PEC thin wire in free-space NEC-2",
                "electrical_state": "unloaded; no meta-atom ON/OFF response",
            },
            "nec": {
                "unknown_meshes_segments_per_wire": [81, 161],
                "extended_thin_wire_kernel": True,
                "port": "center segment of each wire",
                "matrix": "Z_RIS = inverse(port admittance), V=Z I, ohm",
                "external_ports": 0,
            },
            "centers_m": centers.tolist(),
            "limitations": [
                "isolated PEC wire array, not a PCB or calibrated meta-atom surface",
                "narrowband single-frequency model",
                "no dielectric substrate, bias network, enclosure, or measured loading states",
            ],
        }
        dest = root / f"array_{n}" / "geometry.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(metadata, indent=2) + "\n")
        print(dest)


if __name__ == "__main__":
    main()
