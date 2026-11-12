"""One-link bounded Sionna RT scene smoke; never exports meshes or paths."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


SCENES = {
    "simple_street_canyon": {
        "tx": [-20.0, 0.0, 1.5], "rx": [20.0, 0.0, 1.5],
        "basis": "open central roadway between bundled canyon buildings; direct LOS candidate",
    },
    "san_francisco": {
        "tx": [-30.0, 0.0, 10.0], "rx": [30.0, 0.0, 10.0],
        "basis": "bounded coordinate smoke only; scene LOS must be checked in output",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=sorted(SCENES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = SCENES[args.scene]
    import mitsuba as mi
    import sionna.rt as rt

    scene_path = getattr(rt.scene, args.scene)
    scene = rt.load_scene(scene_path)
    scene.frequency = 5.53e9
    ant = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                         horizontal_spacing=.5, pattern="iso", polarization="V")
    scene.tx_array = scene.rx_array = ant
    scene.add(rt.Transmitter("track2_tx", position=spec["tx"]))
    scene.add(rt.Receiver("track2_rx", position=spec["rx"]))
    started = time.perf_counter()
    paths = rt.PathSolver()(scene=scene, max_depth=1, los=True,
                            specular_reflection=True, diffuse_reflection=False,
                            diffraction=False, refraction=False,
                            samples_per_src=16, max_num_paths_per_src=128,
                            synthetic_array=False, seed=7)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    if a.ndim != 6:
        raise RuntimeError(f"Unexpected CIR shape: {a.shape}")
    h = a[:, 0, :, 0, :, 0].sum(-1).astype(np.complex128).reshape(-1)
    if h.size != 1:
        raise RuntimeError(f"Expected one link, got {h.size}")
    result = {
        "status": "smoke_complete", "gate": "NOT_P4_EVIDENCE",
        "scene_name": args.scene, "scene_path": str(scene_path),
        "coordinates_basis": spec["basis"], "tx_xyz_m": spec["tx"], "rx_xyz_m": spec["rx"],
        "frequency_hz": 5.53e9, "channel_real": float(h[0].real),
        "channel_imag": float(h[0].imag), "delay_s": np.asarray(tau).reshape(-1).tolist(),
        "cir_shape": list(a.shape), "elapsed_s": time.perf_counter() - started,
        "versions": {n: importlib.metadata.version(n)
                     for n in ("numpy", "torch", "sionna-rt", "mitsuba", "drjit")},
        "mitsuba_variant": mi.variant(),
        "raw_paths_exported": False, "raw_meshes_exported": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
