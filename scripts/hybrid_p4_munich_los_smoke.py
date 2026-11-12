"""Tiny, non-exporting Munich BS-to-NEC-centre LOS trace for P4 plumbing.

This is a bounded path-tracer smoke, not a P4 gate result: it records one CIR
coefficient at the first registered RIS cell. It does not emit raw paths/meshes,
fit a calibration, or compare against NEC.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="data/raw/acquisition/munich_specular_v3/layout.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frequency-hz", type=float, default=5.53e9)
    args = ap.parse_args()
    layout = json.loads(Path(args.layout).read_text())
    tx = np.asarray(layout["bs"][0], dtype=float)
    # Map the registered panel center/orientation to the 8x8 NEC lattice.
    # `layout.cells` is the legacy scalar 20x13-mm probe and is not the NEC array.
    center = np.asarray(layout["ris_center"], dtype=float)
    normal = np.asarray(layout["panel_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0], 0.0])
    tangent /= np.linalg.norm(tangent)
    nec_centers = np.asarray([
        center + (j - 3.5) * 0.028 * tangent + (i - 3.5) * 0.030 * np.array([0.0, 0.0, 1.0])
        for i in range(8) for j in range(8)
    ])
    rx = nec_centers[0]

    import mitsuba as mi
    import sionna.rt as rt

    scene = rt.load_scene(rt.scene.munich)
    scene.frequency = float(args.frequency_hz)
    ant = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                         horizontal_spacing=.5, pattern="iso", polarization="V")
    scene.tx_array = scene.rx_array = ant
    scene.add(rt.Transmitter("p4_bs", position=tx.tolist()))
    scene.add(rt.Receiver("p4_nec_center", position=rx.tolist()))
    started = time.perf_counter()
    paths = rt.PathSolver()(scene=scene, max_depth=0, los=True,
                            specular_reflection=False, diffuse_reflection=False,
                            diffraction=False, refraction=False,
                            samples_per_src=1, max_num_paths_per_src=1,
                            synthetic_array=False, seed=7)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    if a.ndim != 6:
        raise RuntimeError(f"Unexpected CIR shape: {a.shape}")
    h = a[:, 0, :, 0, :, 0].sum(-1).astype(np.complex128).reshape(-1)
    if h.size != 1:
        raise RuntimeError(f"Expected exactly one transmitter-receiver link, got {h.size}")
    result = {
        "status": "smoke_complete",
        "gate": "P4_NOT_EVALUATED",
        "scope": "one Munich BS to NEC-centre coordinate 0, LOS-only, no path export",
        "frequency_hz": float(args.frequency_hz),
        "tx_xyz_m": tx.tolist(), "rx_xyz_m": rx.tolist(),
        "ris_center_m": center.tolist(), "panel_normal": normal.tolist(),
        "nec_lattice": {"shape": [8, 8], "pitch_tangent_m": 0.028,
                        "pitch_vertical_m": 0.030, "centers_m": nec_centers.tolist(),
                        "mapping": "ris_center + (column-3.5)*28mm*tangent + (row-3.5)*30mm*z"},
        "channel_real": float(h[0].real), "channel_imag": float(h[0].imag),
        "delay_s": np.asarray(tau).reshape(-1).tolist(),
        "cir_shape": list(a.shape),
        "elapsed_s": time.perf_counter() - started,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "sionna-rt", "mitsuba", "drjit")},
        "mitsuba_variant": mi.variant(),
        "scene": "Sionna RT built-in Munich",
        "calibration": "none; no scale/phase fitting",
        "limitation": "No independent NEC field/port response or open-gap correction was measured.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
