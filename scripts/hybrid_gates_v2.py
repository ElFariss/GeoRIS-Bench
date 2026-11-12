#!/usr/bin/env python3
"""Full-size hybrid physics gates P1 and P2 (declared in docs/hybrid-gates-v2-20261107.md).

Runs independent PyNEC solves (EX/LD cards) on the 64 RIS + 4 BS + 2 UE structure and compares them with the
network reductions and the factored spherical hybrid. Raw NEC outputs are stored in ``nec_raw.npz`` so the
evaluation can be rerun with ``--reuse-raw`` without repeating the solves.

Never run this with 64 wires beyond 21 segments on the laptop; use a large-memory CPU node (see docs).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from atomwise_ris.hybrid import full_validation as fv
from atomwise_ris.hybrid.embedded import EmbeddedRISResponse
from atomwise_ris.hybrid.evidence import sha256_file
from atomwise_ris.physics.network import CoupledNetwork
from atomwise_ris.physics.nec import geometry as nec_geometry


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(json.dumps(fv.jc(value), indent=2, sort_keys=True) + "\n")


def resp_to_arrays(r: EmbeddedRISResponse, prefix: str) -> dict:
    return {f"{prefix}_{k}": getattr(r, k) for k in
            ("z_ris", "y_ris", "segment_centers", "segment_lengths", "iy", "iz", "wire_centers")}


def resp_from_arrays(d, prefix: str, frequency: float) -> EmbeddedRISResponse:
    return EmbeddedRISResponse(**{k: d[f"{prefix}_{k}"] for k in
                                  ("z_ris", "y_ris", "segment_centers", "segment_lengths", "iy", "iz", "wire_centers")},
                               frequency=frequency)


def run_nec(args, geom, n, centers, lengths, states, ports, log_fn) -> dict:
    """Execute all NEC solves (parallel processes); return the raw outputs."""
    seg, rad, freq = args.segments, fv.RADIUS, fv.FREQUENCY
    cells, cell_len = centers[:n], lengths[:n]
    exc = ports["bs"] + ports["ue"] + ports["ris"]
    angles_all = list(fv.PLANE_WAVE_ANGLES_DEG.values())
    angles_loaded = [fv.PLANE_WAVE_ANGLES_DEG[a] for a in fv.LOADED_PLANE_WAVE_ANGLES]
    ctx = mp.get_context("spawn")
    raw: dict = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
        futs = {
            "z_full": pool.submit(fv.task_impedance, centers, lengths, rad, freq, seg),
            "resp": pool.submit(fv.task_ris_response, cells, fv.WIRE_LENGTH, rad, freq, seg),
            "pw_short": pool.submit(fv.task_plane_wave, cells, cell_len, rad, freq, seg, None, angles_all),
        }
        for name in ("all_off", "all_on", "alternating", "R1"):
            term = fv.LOADS[states[name].astype(int)]
            futs[f"pw_loaded:{name}"] = pool.submit(fv.task_plane_wave, cells, cell_len, rad, freq, seg, term, angles_loaded)
        for name in ("all_off", "all_on", "alternating", "R1", "R2", "R3", "R4", "R5"):
            futs[f"loaded:{name}"] = pool.submit(
                fv.task_loaded_solve, centers, lengths, rad, freq, seg, fv.terminations_for(states[name]), exc)
        z_full, timing = futs["z_full"].result()
        log_fn(f"Z(70x70) done, NEC timing {timing}")
        raw["z_full"], raw["z_timing"] = z_full, timing
        # Phase B: states that depend on Woodbury flips and exhaustive search on the truth network
        net = CoupledNetwork(z_full, fv.LOADS)
        parent = states["R1"]
        st = net.state(parent)
        single_flip = args.flip_single
        pair_flip = list(args.flip_pair)
        f1 = net.flip(st, [single_flip])
        f2 = net.flip(st, pair_flip)
        i_pair = int(np.flatnonzero((net.pairs == np.sort(pair_flip)).all(axis=1))[0])
        woodbury = {
            "F1": {"bits": f1.bits, "parent": "R1", "flip": [single_flip], "field_woodbury": f1.field,
                   "se_moves": float(net.moves(st)[single_flip])},
            "F2": {"bits": f2.bits, "parent": "R1", "flip": pair_flip, "field_woodbury": f2.field,
                   "se_moves": float(net.moves(st, pairs=True)[i_pair])},
        }
        best = fv.select_best_states(net, n)
        log_fn(f"truth best single {best['single_index']} pair {best['pair_indices']}")
        for name in ("F1", "F2"):
            states[name] = woodbury[name]["bits"]
            futs[f"loaded:{name}"] = pool.submit(
                fv.task_loaded_solve, centers, lengths, rad, freq, seg, fv.terminations_for(states[name]), exc)
        for name, bits in (("best_single", best["bits_single"]), ("best_pair", best["bits_pair"])):
            futs[f"best:{name}"] = pool.submit(
                fv.task_loaded_solve, centers, lengths, rad, freq, seg, fv.terminations_for(bits), ports["bs"])
        for key, fut in futs.items():
            if key == "z_full":
                continue
            raw[key] = fut.result()
            log_fn(f"done {key} (t={time.perf_counter() - t0:.0f}s)")
    raw["woodbury"], raw["best"] = woodbury, best
    raw["wall_seconds"] = time.perf_counter() - t0
    return raw


def save_raw(path: Path, raw: dict, states: dict) -> None:
    arrays = {"z_full": raw["z_full"], "pw_short": raw["pw_short"]}
    arrays.update(resp_to_arrays(raw["resp"], "resp"))
    for key, val in raw.items():
        if key.split(":")[0] in ("pw_loaded", "loaded", "best") and ":" in key:
            kind, name = key.split(":", 1)
            arrays[f"solve__{kind}__{name}"] = val
    for name, info in raw["woodbury"].items():
        arrays[f"wood__{name}__bits"] = info["bits"]
        arrays[f"wood__{name}__field"] = info["field_woodbury"]
        arrays[f"wood__{name}__flip"] = np.asarray(info["flip"])
        arrays[f"wood__{name}__se_moves"] = np.asarray(info["se_moves"])
    for name, bits in states.items():
        arrays[f"state__{name}"] = bits
    b = raw["best"]
    arrays.update({"bestinfo__bits_single": b["bits_single"], "bestinfo__bits_pair": b["bits_pair"],
                   "bestinfo__single_index": np.asarray(b["single_index"]),
                   "bestinfo__pair_indices": np.asarray(b["pair_indices"]),
                   "bestinfo__singles": b["scores"]["singles"], "bestinfo__pairs": b["scores"]["pairs"],
                   "bestinfo__pair_index": b["scores"]["pair_index"]})
    np.savez_compressed(path, **arrays)


def load_raw(path: Path, frequency: float):
    d = np.load(path)
    raw = {"z_full": d["z_full"], "pw_short": d["pw_short"], "resp": resp_from_arrays(d, "resp", frequency)}
    states = {}
    for key in d.files:
        if key.startswith("state__"):
            states[key[len("state__"):]] = d[key]
        elif key.startswith("solve__"):
            _, kind, name = key.split("__", 2)
            raw[f"{kind}:{name}"] = d[key]
    raw["woodbury"] = {n: {"bits": d[f"wood__{n}__bits"], "parent": "R1", "flip": d[f"wood__{n}__flip"].tolist(),
                           "field_woodbury": d[f"wood__{n}__field"], "se_moves": float(d[f"wood__{n}__se_moves"])}
                       for n in ("F1", "F2")}
    raw["best"] = {"bits_single": d["bestinfo__bits_single"], "bits_pair": d["bestinfo__bits_pair"],
                   "single_index": int(d["bestinfo__single_index"]), "pair_indices": d["bestinfo__pair_indices"].tolist(),
                   "scores": {"singles": d["bestinfo__singles"], "pairs": d["bestinfo__pairs"],
                              "pair_index": d["bestinfo__pair_index"]}}
    return raw, states


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", type=int, default=81)
    ap.add_argument("--side", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--flip-single", type=int, default=17)
    ap.add_argument("--flip-pair", type=int, nargs=2, default=[5, 42])
    ap.add_argument("--reuse-raw", action="store_true", help="re-evaluate from out/nec_raw.npz without NEC solves")
    ap.add_argument("--reeval-tag", default="", help="suffix for output names when using --reuse-raw")
    ap.add_argument("--smoke", action="store_true", help="mark artifacts as non-gate smoke output")
    ap.add_argument("--builder", choices=["legacy", "fixed"], default="legacy",
                    help="fixed = NEC external block + open-gap scattering (P2 rerun after model fix); needs --reuse-raw")
    ap.add_argument("--only-diag", action="store_true",
                    help="only write diag_external_only.json (6-wire NEC attribution, non-gating) from stored raw outputs")
    args = ap.parse_args()

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geom = nec_geometry(args.side)
    centers, lengths = np.asarray(geom.centers), np.asarray(geom.lengths)
    n = args.side * args.side
    bs, ue = centers[n:n + fv.N_BS], centers[n + fv.N_BS:]
    ports = fv.excitation_ports(n)
    states = fv.declared_states(n)
    started = datetime.now(timezone.utc).isoformat()
    tag = args.reeval_tag if args.reuse_raw else ""

    if args.only_diag:
        raw, states = load_raw(out / "nec_raw.npz", fv.FREQUENCY)
        diag = fv.external_only_diagnostic(z_full=raw["z_full"], resp=raw["resp"], bs=bs, ue=ue,
                                           segments=args.segments, n_ris=n)
        diag.update({"segments_per_wire": args.segments, "created_utc": datetime.now(timezone.utc).isoformat(),
                     "non_gating": True})
        write_new(out / "diag_external_only.json", diag)
        log(f"wrote {out / 'diag_external_only.json'}")
        return 0

    if args.reuse_raw:
        raw, states = load_raw(out / "nec_raw.npz", fv.FREQUENCY)
        log("re-evaluating from stored raw NEC outputs")
    else:
        for fname in ("p1.json", "p2.json", "summary.json", "nec_raw.npz"):
            if (out / fname).exists():
                raise FileExistsError(f"{out / fname} exists; use a fresh --out or --reuse-raw")
        raw = run_nec(args, geom, n, centers, lengths, states, ports, log)
        np.save(out / "z_full.npy", raw["z_full"])
        np.save(out / "z_ris_alone.npy", raw["resp"].z_ris)
        save_raw(out / "nec_raw.npz", raw, states)

    loaded_names = ["all_off", "all_on", "alternating", "R1", "R2", "R3", "R4", "R5", "F1", "F2"]
    angle_cols = {a: i for i, a in enumerate(fv.LOADED_PLANE_WAVE_ANGLES)}
    p1 = fv.evaluate_p1(
        z_full=raw["z_full"], n_ris=n, states=states, loaded={k: raw[f"loaded:{k}"] for k in loaded_names},
        ports=ports, woodbury=raw["woodbury"], resp=raw["resp"], pw_short=raw["pw_short"],
        pw_loaded={(k, a): raw[f"pw_loaded:{k}"][:, c] for k in ("all_off", "all_on", "alternating", "R1")
                   for a, c in angle_cols.items()})
    p2_states = {k: states[k] for k in ("all_off", "all_on", "R1", "R2", "R3", "R4", "R5")}
    if args.builder == "fixed":
        if not args.reuse_raw:
            raise SystemExit("--builder fixed re-uses the stored truth NEC outputs; pass --reuse-raw")
        fix_path = out / "nec_raw_fix1.npz"
        if fix_path.exists():
            d = np.load(fix_path)
            ext_resp = resp_from_arrays(d, "ext", fv.FREQUENCY)
            i_sc = d["i_sc"]
        else:
            from atomwise_ris.hybrid.ray_coupling import external_field_vectors
            t1 = time.perf_counter()
            ext_resp = fv.task_external_response(centers[n:], fv.WIRE_LENGTH, fv.RADIUS, fv.FREQUENCY, args.segments)
            log(f"external-alone NEC done ({time.perf_counter() - t1:.0f}s)")
            e_vec = external_field_vectors(raw["resp"], ext_resp, fv.FREQUENCY)
            i_sc = fv.task_open_gap_currents(centers[:n], e_vec, raw["resp"].segment_lengths, fv.WIRE_LENGTH,
                                             fv.RADIUS, fv.FREQUENCY, args.segments)
            log(f"RIS-alone impressed-field NEC done ({time.perf_counter() - t1:.0f}s)")
            np.savez_compressed(fix_path, i_sc=i_sc, **resp_to_arrays(ext_resp, "ext"))
        p2 = fv.evaluate_p2_fixed(
            z_full=raw["z_full"], resp=raw["resp"], ext_resp=ext_resp, continuous_currents=i_sc, bs=bs, ue=ue,
            states=p2_states, best=raw["best"], direct_best={k: raw[f"best:{k}"] for k in ("best_single", "best_pair")},
            n_ris=n)
    else:
      p2 = fv.evaluate_p2(
          z_full=raw["z_full"], resp=raw["resp"], bs=bs, ue=ue, states=p2_states, best=raw["best"],
          direct_best={k: raw[f"best:{k}"] for k in ("best_single", "best_pair")}, n_ris=n)
    finished = datetime.now(timezone.utc).isoformat()

    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve()] + [root / "src/atomwise_ris" / p for p in (
        "hybrid/full_validation.py", "hybrid/embedded.py", "hybrid/ray_coupling.py", "hybrid/operator.py",
        "physics/nec.py", "physics/network.py")] + [root / "docs/hybrid-gates-v2-20261107.md"]
    try:
        import PyNEC  # noqa: F401
        pynec = getattr(PyNEC, "__version__", "unknown")
    except Exception:
        pynec = None
    meta = {
        "created_utc": finished, "run_started_utc": started, "host": platform.node(), "python": platform.python_version(),
        "numpy": np.__version__, "pynec": pynec, "git_commit": git_commit(),
        "argv": sys.argv, "segments_per_wire": args.segments, "side": args.side, "n_ris": n,
        "ports": {"bs": ports["bs"], "ue": ports["ue"], "ris_excitations": ports["ris"]},
        "smoke": bool(args.smoke), "reuse_raw": bool(args.reuse_raw), "builder_flag": args.builder,
        "nec_wall_seconds": raw.get("wall_seconds"), "z_timing": raw.get("z_timing"),
        "source_sha256": {str(p.relative_to(root)): sha256_file(p) for p in sources if p.exists()},
        "geometry": {"wire_length_m": fv.WIRE_LENGTH, "radius_m": fv.RADIUS, "frequency_hz": fv.FREQUENCY,
                     "loads_ohm": fv.LOADS, "reference_ohm": fv.REFERENCE_OHM,
                     "centers_m": centers},
        "excluded": ["data/processed/*", "urban ray paths", "optimizer", "training"],
    }
    p1.update(meta)
    p2.update(meta)
    write_new(out / f"p1{tag}.json", p1)
    write_new(out / f"p2{tag}.json", p2)
    summary = {
        **{k: meta[k] for k in ("created_utc", "host", "segments_per_wire", "side", "smoke", "git_commit")},
        "doc": "docs/hybrid-gates-v2-20261107.md",
        "doc_sha256": meta["source_sha256"].get("docs/hybrid-gates-v2-20261107.md"),
        "P1": {"passed": p1["passed"], "criteria": p1["criteria"]},
        "P2": {"passed": p2["passed"], "criteria": p2["criteria"]},
        "P1_P2_passed_at_this_mesh": bool(p1["passed"] and p2["passed"]),
        "optimizer_authorized": False, "training_authorized": False,
        "note": "P3 and P4 are not evaluated here; no gate result authorizes hybrid optimization.",
    }
    write_new(out / f"summary{tag}.json", summary)
    for g, r in (("P1", p1), ("P2", p2)):
        log(f"{g}: {'PASS' if r['passed'] else 'FAIL'}")
        for c in r["criteria"]:
            log(f"  {c['id']:6s} value={c['value']:.6g} {c['op']} {c['tolerance']:g} -> {'ok' if c['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
