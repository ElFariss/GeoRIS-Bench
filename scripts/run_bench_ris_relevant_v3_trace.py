#!/usr/bin/env python3
"""Trace entry point for the ris_relevant_v3 scope (docs/bench-ris-relevant-scope-v3-20261113.md section 3).

Imports the unchanged scripts/run_bench_production_trace.py and, in this process only:
- extends the manifest registry / scope tuple (ris_relevant_scope_v3.install);
- restricts the case list to shard RR3_SHARD="k/K" (trajectory index mod K == k); records keep the full-manifest hash;
- replaces trace_one by trace_one_v3: the original body with samples_per_src=1e6 and max_num_paths_per_src=2e6 (D1), plus
  one shared direct trace per frame (receivers = external endpoints only), written once to
  <out>/<map>/direct/<case>/frame-XXXX.npz (+ .json sidecar with sha256). The v3 channel wrapper merges its
  BS<->user / user<->user rows into every array size.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench_production_trace as T  # noqa: E402
from run_bench_production_trace import *  # noqa: F401,F403,E402  (names used by the copied trace_one body)
from run_bench_production_trace import (array_layout, bs_ports, extract_npz, gpu_snapshot, GpuSampler,  # noqa: E402
                                        FREQ_HZ)
from atomwise_ris.bench import ris_relevant_scope_v3  # noqa: E402
from atomwise_ris.bench.production_freeze import sha256  # noqa: E402

SAMPLES_PER_SRC = 1_000_000
MAX_PATHS_PER_SRC = 2_000_000


def trace_direct(rt, dr, scene_path, case, frame, tx_xyz, origin, solver_cfg, out_dir):
    frame_idx = int(frame.get("frame", 0))
    out_path = out_dir / case["map_id"] / "direct" / case["id"] / f"frame-{frame_idx:04d}.npz"
    side = out_path.with_suffix(".json")
    if out_path.is_file() and side.is_file():
        info = json.loads(side.read_text())
        if info.get("sha256") == sha256(out_path):
            return {**info, "reused": True}
    scene = rt.load_scene(str(scene_path), merge_shapes=False)
    scene.frequency = FREQ_HZ
    scene.tx_array = scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                                                     horizontal_spacing=.5, pattern="iso", polarization="VH")
    for i, point in enumerate(tx_xyz - origin):
        scene.add(rt.Transmitter(f"tx_{i:03d}", position=point.tolist()))
    for i, point in enumerate(tx_xyz - origin):
        scene.add(rt.Receiver(f"rx_{i:04d}", position=point.tolist()))
    cfg = {**solver_cfg, "note": "v3 shared direct trace: receivers = external endpoints only"}
    t0 = time.perf_counter()
    paths = rt.PathSolver()(scene=scene, **{k: v for k, v in cfg.items() if k != "note"})
    dr.sync_thread()
    el = time.perf_counter() - t0
    local_case = {**case, "_origin": origin}
    # extract_npz looks up ARRAY_SHAPES[count]; add the count-0 entry only for this call (the scene builder iterates
    # ARRAY_SHAPES, so it must not persist; fix of 2026-11-13 07:15Z, see the declaration's run log).
    T.ARRAY_SHAPES[0] = (0, 0)
    try:
        res = extract_npz(local_case, 0, paths, tx_xyz, tx_xyz.copy(), origin, case["_scene_info"],
                          {**cfg, "solver_elapsed_seconds": el}, out_path, frame=frame)
    finally:
        T.ARRAY_SHAPES.pop(0, None)
    info = {"file": str(out_path), "sha256": sha256(out_path), "solver_elapsed_seconds": el,
            "samples_per_src": cfg["samples_per_src"], "max_num_paths_per_src": cfg["max_num_paths_per_src"],
            "paths": res.get("paths"), "nonzero_links": res.get("nonzero_links")}
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=1))
    tmp.replace(side)
    return info


def trace_one_v3(rt, dr, root: Path, case: dict, count: int, out_dir: Path,
              frame: dict | None = None, gpu_id: int = 0,
              scene_path: Path | None = None) -> dict:
    if frame is None:
        frame = case["frames"][0]
    frame_idx = int(frame.get("frame", 0))
    origin = np.asarray(case["_origin"], dtype=np.float64)
    ris = array_layout(case["ris_xyz"], case["ris_normal_xyz"], count)
    bs = bs_ports(case["bs_xyz"], case["ris_xyz"])
    users = np.asarray([u["xyz"] for u in frame["users"]], dtype=np.float64)
    external = np.vstack([bs, users])
    tx_xyz = external.copy()
    rx_xyz = np.vstack([ris, external])
    prep_t0 = time.perf_counter()
    if scene_path is None:
        per_case_scene = root / "scenes" / case["map_id"] / case["id"] / "scene.xml"
        if per_case_scene.is_file():
            scene_path = per_case_scene
        else:
            scene_path = root / "scenes" / case["map_id"] / "scene.xml"
    scene = rt.load_scene(str(scene_path), merge_shapes=False)
    scene.frequency = FREQ_HZ
    scene.tx_array = scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
        pattern="iso", polarization="VH")
    for i, point in enumerate(tx_xyz-origin):
        scene.add(rt.Transmitter(f"tx_{i:03d}", position=point.tolist()))
    for i, point in enumerate(rx_xyz-origin):
        scene.add(rt.Receiver(f"rx_{i:04d}", position=point.tolist()))
    scene_load_and_ports_s = time.perf_counter() - prep_t0
    snapshot = gpu_snapshot()
    gpu = next((g for g in snapshot if g["index"] == gpu_id), None)
    if gpu is None or gpu["free_mib"] < 4096:
        raise RuntimeError(f"GPU{gpu_id} stop rule: need at least 4096 MiB free")
    solver_cfg = {"max_depth": 2, "los": True, "specular_reflection": True,
                  "diffraction": True, "diffuse_reflection": False, "refraction": False,
                  "synthetic_array": False, "samples_per_src": SAMPLES_PER_SRC,
                  "max_num_paths_per_src": MAX_PATHS_PER_SRC, "seed": 7,
                  "note": "v3 (D1): 1e6 rays/source, cap 2e6; depth 2; transmission/refraction off"}
    t0 = time.perf_counter()
    with GpuSampler() as sampler:
        paths = rt.PathSolver()(scene=scene,
                                **{key: value for key, value in solver_cfg.items() if key != "note"})
        dr.sync_thread()
        solver_elapsed = time.perf_counter()-t0
        export_start = time.perf_counter()
        local_case = {**case, "_origin": origin}
        out_path = out_dir / case["map_id"] / f"n{count}" / case["id"] / f"frame-{frame_idx:04d}.npz"
        result = extract_npz(local_case, count, paths, tx_xyz, rx_xyz, origin,
                             case["_scene_info"], {**solver_cfg,
                             "solver_elapsed_seconds": solver_elapsed}, out_path, frame=frame)
        export_elapsed = time.perf_counter()-export_start
        dr.sync_thread()
    elapsed = time.perf_counter()-t0
    direct_info = trace_direct(rt, dr, scene_path, case, frame, tx_xyz, origin, solver_cfg, out_dir)
    gpu_after = gpu_snapshot()
    peak_gpu = sampler.peak_for(gpu_id)
    return {"case_id": case["id"], "map_id": case["map_id"], "frame": frame_idx,
            "array_elements": count, "elapsed_seconds": elapsed,
            "solver_elapsed_seconds": solver_elapsed,
            "export_elapsed_seconds": export_elapsed,
            "scene_load_and_ports_s": scene_load_and_ports_s,
            "gpu_before": snapshot, "gpu_after": gpu_after,
            "gpu_peak_sampled": peak_gpu, "rr3_direct": direct_info, **result}




def _shard_filter():
    spec = os.environ.get("RR3_SHARD")
    if not spec:
        raise SystemExit("RR3_SHARD=k/K is required")
    k, K = map(int, spec.split("/"))
    orig = T.verify_case_manifest

    def verify(path, production=False, smoke_unregistered=False):
        obj, cases = orig(path, production=production, smoke_unregistered=smoke_unregistered)
        mine = [c for c in cases if int(c["id"].rsplit("-", 1)[1]) % K == k]
        print(f"RR3 shard {k}/{K}: {len(mine)} of {len(cases)} cases", flush=True)
        return obj, mine
    T.verify_case_manifest = verify


ris_relevant_scope_v3.install(T)
T.trace_one = trace_one_v3
_shard_filter()

if __name__ == "__main__":
    T.main()
