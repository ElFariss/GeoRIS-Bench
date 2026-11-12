#!/usr/bin/env python3
"""Build bounded pilot and production channel shards from explicit ray archives."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import h5py
import numpy as np

from atomwise_ris.bench.production_channel import (
    FREQ,
    LOADS,
    RisBasis,
    aggregate_classes,
    project_rays,
)
from atomwise_ris.bench.production_freeze import sha256, verify_freeze
from atomwise_ris.bench.production_pilot import registered_manifest
from atomwise_ris.bench.production_run import (
    ARRAY_SIZES,
    DEVELOPMENT_MAPS,
    PRODUCTION_SCOPES,
    atomic_write_json,
    cases_by_map,
    check_cpu_set,
    stop_rule_exceeded,
)
from atomwise_ris.hybrid.operator import HybridCoupledNetwork

ARRAYS = {64: 161, 128: 161, 256: 81}


def atomic_json(p: Path, value) -> None:
    atomic_write_json(p, value)


def check_mem_available(min_bytes: int = 16 * 1024 * 1024 * 1024) -> None:
    """Fail cleanly if host MemAvailable is under threshold."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            kb = int(parts[1])
            if kb * 1024 < min_bytes:
                raise RuntimeError(
                    f"insufficient MemAvailable: {kb * 1024 / (1024**3):.2f} GiB < {min_bytes / (1024**3):.2f} GiB"
                )
            return


def input_physics(root: Path, n: int) -> tuple[np.ndarray, np.ndarray, str]:
    mesh = ARRAYS[n]
    d = root / f"array_{n}"
    meta = json.loads((d / "geometry.json").read_text())
    zfile = d / "impedance.h5"
    if not zfile.is_file() and n == 256:
        zfile = d / "81/impedance.h5"
    with h5py.File(zfile, "r") as h:
        names = (f"segments_{mesh}/Z_RIS_ohm", "Z_RIS_ohm")
        a = next((h[k] for k in names if k in h), None)
        if a is None or a.shape != (n, n) or int(a.attrs["segments_per_wire"]) != mesh:
            raise ValueError(f"unapproved array mesh or shape: {zfile}")
        z = a[:]
    if not np.isfinite(z).all() or float(meta["frequency_hz"]) != FREQ:
        raise ValueError("nonfinite or wrong-frequency impedance")
    return np.asarray(meta["centers_m"], float), z, sha256(zfile)


def frame_inputs(p: Path, case: dict, frame: dict, n: int, basis_centers: np.ndarray):
    with np.load(p, allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    md = json.loads(t["metadata_json"].item())
    if (
        md["map_id"],
        md["case_id"],
        int(md["frame"]),
        int(md["array_elements"]),
        float(md["frequency_hz"]),
        md["pol_order"],
    ) != (case["map_id"], case["id"], frame["frame"], n, FREQ, ["V", "H"]):
        raise ValueError(f"wrong trace provenance: {p}")
    origin = np.asarray(md["projected_origin_xyz_m"])
    tx = np.asarray(t["tx_xyz"], float)
    ext = tx + origin
    k = len(frame["users"])
    if len(tx) != 4 + k or not np.allclose(ext[:4].mean(0), case["bs_xyz"], rtol=0, atol=1e-6):
        raise ValueError("wrong BS coordinates")
    if not np.allclose(ext[4:], [u["xyz"] for u in frame["users"]], rtol=0, atol=1e-6):
        raise ValueError("wrong UE coordinates/order")
    ris = np.asarray(t["rx_xyz"][:n], float)
    normal = np.asarray(case["ris_normal_xyz"])
    horizontal = np.cross([0.0, 0.0, 1.0], normal)
    base = basis_centers - basis_centers.mean(0)
    expected = base[:, 1, None] * horizontal + base[:, 2, None] * np.array([0.0, 0.0, 1.0])
    if not np.allclose(ris - ris.mean(0), expected, rtol=0, atol=1e-6) or not np.allclose(
        ris.mean(0) + origin, np.asarray(case["ris_xyz"]) + 0.05 * normal, rtol=0, atol=1e-6
    ):
        raise ValueError("wire order or panel placement does not match archived RIS")
    if "has_other_interaction" in t and np.any(t["has_other_interaction"]):
        raise ValueError("path archive has unknown or out-of-scope interactions")
    for angle in ("path_theta_t", "path_theta_r", "path_phi_t", "path_phi_r"):
        if not np.isfinite(t[angle]).all():
            raise ValueError("missing angle for a nonzero path")
    return t, md, tx


def channel_frame(
    h5,
    key: str,
    case: dict,
    frame: dict,
    n: int,
    basis,
    center: np.ndarray,
    zsha: str,
    npz_path: Path,
    rec: dict | None = None,
    project_rays_fn=project_rays,
    network_cls=HybridCoupledNetwork,
    inputs=None,
):
    """Process a single frame with verified physics, writing datasets and returning metadata record."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing trace npz: {npz_path}")
    digest = sha256(npz_path)
    file_bytes = npz_path.stat().st_size
    if rec is not None:
        if digest != rec["sha256"] or file_bytes != rec["bytes"]:
            raise ValueError(f"trace artifact hash mismatch: {npz_path}")

    if inputs is not None:
        t, md, tx = inputs
    else:
        t, md, tx = frame_inputs(npz_path, case, frame, n, center)

    tick = time.perf_counter()
    mats, (s, d, k) = project_rays_fn(t, basis, tx, 4)
    ev = network_cls(mats["Z_effective"], mats["C_out"], mats["C_in"], mats["H_direct"], LOADS)
    bit = np.zeros(n, dtype=np.int8)
    state = ev.state(bit)
    rate = float(ev.score(state.field))
    dense = float(ev.score(ev.channel(bit) @ ev.w))
    if not np.isfinite(rate) or abs(dense - rate) > 1e-10:
        raise ValueError(f"dense round-trip failed at {key}")

    g = h5.create_group(key)
    for nm, x in mats.items():
        g.create_dataset(nm, data=x, compression="gzip")
    g.create_dataset("loads", data=LOADS)
    g.create_dataset("precoder", data=ev.w)
    g.create_dataset("stored_bits", data=bit)
    g.attrs["stored_rate_bps_hz"] = rate
    g.attrs["noise"] = ev.noise
    g.attrs["mesh_segments"] = ARRAYS[n]
    g.attrs["ris_archive_sha256"] = zsha
    g.attrs["frequency_hz"] = FREQ

    u = frame["users"]
    classes = aggregate_classes(s, d, k, n, 4, len(u))
    elapsed = time.perf_counter() - tick

    record = {
        "users": [
            {"user_id": usr["user_id"], "bs_user": bl, "ris_user": rl}
            for usr, bl, rl in zip(u, classes["bs_to_user"], classes["ris_to_user"], strict=True)
        ],
        "link_classes": {"bs_ris": classes["bs_to_ris"]},
        "trace_sha256": digest,
        "trace_path_host_only": str(npz_path),
        "source_geometry_sha256": md["scene_sha256"],
        "paths": len(s),
        "elapsed_channel_s": elapsed,
        "raw_path_bytes": file_bytes,
        "geometry_agreement": {
            "bs": [
                (usr["bs_link"] == "los") == (bl == "los")
                for usr, bl in zip(u, classes["bs_to_user"], strict=True)
            ],
            "ris": [
                bool(usr["ris_los"]) == (rl == "los")
                for usr, rl in zip(u, classes["ris_to_user"], strict=True)
            ],
        },
    }
    return key, record


def write_array(a, cases, manifest_hash, freeze):
    """Pilot single-array builder."""
    n = a.array
    final = a.out / f"array_{n}.h5"
    side = a.out / f"array_{n}.json"
    if final.exists() or side.exists():
        if final.is_file() and side.is_file():
            rec = json.loads(side.read_text())
            if rec["h5_sha256"] == sha256(final) and rec["case_manifest_sha256"] == manifest_hash:
                print("verified existing channel shard", n, flush=True)
                return
        raise ValueError(f"unverified or partially present channel shard: {n}")
    center, z, zsha = input_physics(a.physics, n)
    trace_run = json.loads((a.trace_root / "trace_manifest.json").read_text())
    if (
        trace_run["status"] != "complete"
        or trace_run["case_manifest_sha256"] != manifest_hash
        or trace_run.get("run_scope") != a.scope
    ):
        raise ValueError("the completed trace run is not for the frozen pilot manifest")
    trace_records = {
        (x["map_id"], x["array_elements"], x["case_id"], x["frame"]): x
        for x in trace_run["cases"]
    }
    expected = {
        (c["map_id"], size, c["id"], f["frame"])
        for c in cases
        for size in ARRAYS
        for f in c["frames"]
    }
    if set(trace_records) != expected or len(trace_run["cases"]) != len(expected):
        raise ValueError("incomplete or duplicate trace records")
    checked = {}
    for case in cases:
        city, case_id = case["map_id"], case["id"]
        for frame in case["frames"]:
            i = int(frame["frame"])
            p = a.trace_root / city / f"n{n}" / case_id / f"frame-{i:04d}.npz"
            digest = sha256(p)
            rec = trace_records[(city, n, case_id, i)]
            if digest != rec["sha256"] or p.stat().st_size != rec["bytes"]:
                raise ValueError(f"trace artifact hash mismatch: {p}")
            checked[(city, case_id, i)] = (p, digest, frame_inputs(p, case, frame, n, center))
    started = time.perf_counter()
    print("build and verify reusable NEC basis", n, ARRAYS[n], flush=True)
    basis = RisBasis(center, ARRAYS[n], z)
    setup = time.perf_counter() - started
    print("NEC basis ready", n, setup, basis.archive_error, flush=True)
    records = {}
    tmp = final.with_suffix(".h5.tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    with h5py.File(tmp, "w") as h:
        h.attrs["case_manifest_sha256"] = manifest_hash
        h.attrs["freeze_sha256"] = freeze
        for case in cases:
            city, case_id = case["map_id"], case["id"]
            for frame in case["frames"]:
                i = int(frame["frame"])
                key = f"{city}/{n}/{case_id}/{i}"
                p, digest, inp = checked[(city, case_id, i)]
                rec = trace_records[(city, n, case_id, i)]
                k, meta_rec = channel_frame(
                    h,
                    key,
                    case,
                    frame,
                    n,
                    basis,
                    center,
                    zsha,
                    p,
                    rec=rec,
                    inputs=inp,
                )
                records[k] = meta_rec
                print(key, "paths", meta_rec["paths"], "channel_seconds", meta_rec["elapsed_channel_s"], flush=True)
    tmp.replace(final)
    atomic_json(
        side,
        {
            "records": records,
            "h5_sha256": sha256(final),
            "case_manifest_sha256": manifest_hash,
            "ris_archive_sha256": zsha,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "setup_s": setup,
            "basis_archive_relative_error": basis.archive_error,
        },
    )


def assemble(a, manifest_hash, freeze):
    """Pilot assemble."""
    records = {}
    shards = {}
    target = a.out / "channel_realizations.h5"
    if target.exists() or (a.out / "metadata.json").exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".h5.tmp")
    with h5py.File(tmp, "x") as dst:
        dst.attrs["case_manifest_sha256"] = manifest_hash
        dst.attrs["freeze_sha256"] = freeze
        for n in ARRAYS:
            h = a.out / f"array_{n}.h5"
            m = json.loads((a.out / f"array_{n}.json").read_text())
            if m["h5_sha256"] != sha256(h) or m["case_manifest_sha256"] != manifest_hash:
                raise ValueError("channel shard provenance mismatch")
            with h5py.File(h, "r") as src:
                for key in m["records"]:
                    parent, name = key.rsplit("/", 1)
                    src.copy(key, dst.require_group(parent), name=name)
            records.update(m.pop("records"))
            shards[str(n)] = m
    tmp.replace(target)
    atomic_json(a.out / "metadata.json", records)
    atomic_json(
        a.out / "channel_run.json",
        {
            "arrays": shards,
            "hdf5_sha256": sha256(target),
            "metadata_sha256": sha256(a.out / "metadata.json"),
            "case_manifest_sha256": manifest_hash,
        },
    )


# ---------------------------------------------------------------------------
# Production mode implementation
# ---------------------------------------------------------------------------

def is_unit_done(unit_h5: Path, unit_json: Path, manifest_hash: str) -> bool:
    """True if shard and sidecar exist, sidecar manifest hash matches and h5_sha256 matches."""
    if not unit_h5.is_file() or not unit_json.is_file():
        return False
    try:
        side = json.loads(unit_json.read_text())
        if side.get("case_manifest_sha256") != manifest_hash:
            return False
        if side.get("h5_sha256") != sha256(unit_h5):
            return False
        return True
    except Exception:
        return False


def get_map_failed_cases(
    map_id: str,
    map_cases: list[dict],
    trace_out: Path,
    channel_out: Path,
) -> set[str]:
    """Collect distinct failed case IDs for this map from trace and channel stages."""
    manifest_case_ids = {c["id"] for c in map_cases}
    failed = set()
    # 1. Trace failures: trace_out/<map>/cases/<case_id>.FAILED.json
    trace_cases_dir = trace_out / map_id / "cases"
    if trace_cases_dir.is_dir():
        for p in trace_cases_dir.glob("*.FAILED.json"):
            cid = p.name[:-12]
            if cid in manifest_case_ids:
                failed.add(cid)
    # 2. Channel failures: channel_out/<map>/cases/<case_id>/array_<N>.FAILED.json
    ch_cases_dir = channel_out / map_id / "cases"
    if ch_cases_dir.is_dir():
        for p in ch_cases_dir.glob("*/*.FAILED.json"):
            cid = p.parent.name
            if cid in manifest_case_ids:
                failed.add(cid)
    return failed


def update_map_failure_status(
    map_id: str,
    map_cases: list[dict],
    trace_out: Path,
    channel_out: Path,
) -> bool:
    """If stop rule is exceeded for map, write channel_out/<map>/FAILED atomically."""
    failed_cases = get_map_failed_cases(map_id, map_cases, trace_out, channel_out)
    total = len(map_cases)
    if stop_rule_exceeded(len(failed_cases), total):
        failed_path = channel_out / map_id / "FAILED"
        if not failed_path.exists():
            atomic_write_json(
                failed_path,
                {
                    "map_id": map_id,
                    "status": "failed",
                    "reason": "stop rule exceeded",
                    "failed_cases_count": len(failed_cases),
                    "total_cases": total,
                    "failed_cases": sorted(failed_cases),
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        return True
    return False


def update_map_completion_status(
    map_id: str,
    map_cases: list[dict],
    trace_out: Path,
    channel_out: Path,
    manifest_hash: str,
) -> bool:
    """If all non-failed cases have all 3 verified shards and trace is DONE, write C/<map>/DONE."""
    done_path = channel_out / map_id / "DONE"
    if done_path.exists():
        return True
    if (channel_out / map_id / "FAILED").exists() or (trace_out / map_id / "FAILED").exists():
        return False
    if not (trace_out / map_id / "DONE").exists():
        return False

    total = len(map_cases)
    failed_cases = get_map_failed_cases(map_id, map_cases, trace_out, channel_out)
    if stop_rule_exceeded(len(failed_cases), total):
        update_map_failure_status(map_id, map_cases, trace_out, channel_out)
        return False

    # Every non-failed case must have verified shards for 64, 128, and 256
    for case in map_cases:
        cid = case["id"]
        if cid in failed_cases:
            continue
        for n in ARRAY_SIZES:
            h5_p = channel_out / map_id / "cases" / cid / f"array_{n}.h5"
            json_p = channel_out / map_id / "cases" / cid / f"array_{n}.json"
            if not is_unit_done(h5_p, json_p, manifest_hash):
                return False

    # Under map.lock, write DONE
    map_lock = channel_out / map_id / "map.lock"
    map_lock.parent.mkdir(parents=True, exist_ok=True)
    with open(map_lock, "w") as ml:
        fcntl.flock(ml.fileno(), fcntl.LOCK_EX)
        try:
            if not done_path.exists():
                atomic_write_json(
                    done_path,
                    {
                        "status": "done",
                        "map_id": map_id,
                        "cases_total": total,
                        "cases_included": total - len(failed_cases),
                        "failed_cases": sorted(failed_cases),
                        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
        finally:
            fcntl.flock(ml.fileno(), fcntl.LOCK_UN)
    return True


def process_unit(
    a,
    map_id: str,
    case: dict,
    n: int,
    basis,
    center: np.ndarray,
    zsha: str,
    setup_s: float,
    manifest_hash: str,
    freeze: str,
    trace_out: Path,
    trace_record_path: Path,
    trace_record: dict,
    frame_fn=channel_frame,
) -> None:
    """Process all frames for one case and array, writing array_<N>.h5 and sidecar atomically."""
    case_id = case["id"]
    unit_dir = a.out / map_id / "cases" / case_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_h5 = unit_dir / f"array_{n}.h5"
    unit_json = unit_dir / f"array_{n}.json"
    tmp_h5 = unit_dir / f"array_{n}.h5.tmp"

    if tmp_h5.exists():
        tmp_h5.unlink()

    records = {}
    tr_records = {
        (r["array_elements"], r["frame"]): r for r in trace_record.get("records", [])
    }

    with h5py.File(tmp_h5, "w") as h:
        h.attrs["case_manifest_sha256"] = manifest_hash
        h.attrs["freeze_sha256"] = freeze
        for frame in case["frames"]:
            i = int(frame["frame"])
            key = f"{map_id}/{n}/{case_id}/{i}"
            tr_rec = tr_records.get((n, i))
            if tr_rec is None:
                raise ValueError(f"trace record missing entry for frame {i} array {n}")
            npz_path = trace_out / map_id / f"n{n}" / case_id / f"frame-{i:04d}.npz"
            f_key, f_rec = frame_fn(
                h5=h,
                key=key,
                case=case,
                frame=frame,
                n=n,
                basis=basis,
                center=center,
                zsha=zsha,
                npz_path=npz_path,
                rec=tr_rec,
            )
            records[f_key] = f_rec

    tmp_h5.replace(unit_h5)

    h5_sha = sha256(unit_h5)
    tr_sha = sha256(trace_record_path)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    sidecar_data = {
        "records": records,
        "h5_sha256": h5_sha,
        "case_manifest_sha256": manifest_hash,
        "trace_record_sha256": tr_sha,
        "ris_archive_sha256": zsha,
        "setup_s": setup_s,
        "peak_rss_bytes": peak_rss,
        "worker_id": a.worker_id,
    }
    atomic_write_json(unit_json, sidecar_data)


def production_assemble(a, manifest: dict, scope: str, manifest_hash: str, freeze: str) -> None:
    """Assemble production shards into channel_realizations.h5, metadata.json and channel_run.json."""
    target = a.out / "channel_realizations.h5"
    meta_target = a.out / "metadata.json"
    run_target = a.out / "channel_run.json"
    if target.exists() or meta_target.exists() or run_target.exists():
        raise FileExistsError(f"assembly output already exists in {a.out}")

    map_groups = cases_by_map(manifest["cases"])
    for map_id in map_groups:
        done_file = a.out / map_id / "DONE"
        if not done_file.is_file():
            raise RuntimeError(f"map {map_id} is not DONE in {a.out}")

    maps_summary = {}
    for map_id, map_cases in map_groups.items():
        done_data = json.loads((a.out / map_id / "DONE").read_text())
        failed_cases = sorted(done_data.get("failed_cases", []))
        total = len(map_cases)
        maps_summary[map_id] = {
            "cases_total": total,
            "cases_included": total - len(failed_cases),
            "failed_cases": failed_cases,
        }

    records = {}
    arrays_summary = {}
    tmp = target.with_suffix(".h5.tmp")
    if tmp.exists():
        tmp.unlink()

    with h5py.File(tmp, "x") as dst:
        dst.attrs["case_manifest_sha256"] = manifest_hash
        dst.attrs["freeze_sha256"] = freeze
        for n in ARRAY_SIZES:
            setups = []
            peak_rss = 0
            for map_id, map_cases in map_groups.items():
                failed_set = set(maps_summary[map_id]["failed_cases"])
                for case in map_cases:
                    case_id = case["id"]
                    if case_id in failed_set:
                        continue
                    shard_h5 = a.out / map_id / "cases" / case_id / f"array_{n}.h5"
                    shard_json = a.out / map_id / "cases" / case_id / f"array_{n}.json"
                    if not shard_h5.is_file() or not shard_json.is_file():
                        raise ValueError(f"missing shard for included case {case_id} array {n}")
                    m = json.loads(shard_json.read_text())
                    if m["h5_sha256"] != sha256(shard_h5) or m["case_manifest_sha256"] != manifest_hash:
                        raise ValueError(f"channel shard provenance mismatch: {shard_h5}")
                    with h5py.File(shard_h5, "r") as src:
                        for key in m["records"]:
                            parent, name = key.rsplit("/", 1)
                            src.copy(key, dst.require_group(parent), name=name)
                    records.update(m.get("records", {}))
                    if "setup_s" in m:
                        setups.append(m["setup_s"])
                    if "peak_rss_bytes" in m:
                        peak_rss = max(peak_rss, m["peak_rss_bytes"])
            arrays_summary[str(n)] = {
                "setup_s": setups,
                "peak_rss_bytes": peak_rss,
            }

    tmp.replace(target)
    atomic_write_json(meta_target, records)
    channel_run = {
        "run_scope": scope,
        "case_manifest_sha256": manifest_hash,
        "hdf5_sha256": sha256(target),
        "metadata_sha256": sha256(meta_target),
        "maps": maps_summary,
        "arrays": arrays_summary,
    }
    atomic_write_json(run_target, channel_run)


# --- Unit isolation and failure classification (lead fix, 2026-11-10) -------------------------
# Root cause of the 2026-11-09 failure: every xq_card solve on the resident PyNEC basis context
# appends a result set that PyNEC can never free (~0.2-0.4 GB per case), so long-lived workers
# grew until they hit ulimit -v (C++ std::bad_alloc surfaced by SWIG as "Unknown exception") and
# then exhausted host RAM (numpy "Unable to allocate"). Each unit now runs in a forked child that
# inherits the already-factorized basis copy-on-write; the leak dies with the child, and the
# physics is byte-identical because the same basis object and code path are used.
MAX_TRANSIENT_ATTEMPTS = 3
TRANSIENT_BACKOFF_S = (300, 900)
UNIT_ADMISSION_BYTES = 4 * 1024 ** 3


def mem_available_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def classify_exception(exc: BaseException) -> str:
    """'transient' for resource exhaustion, else 'case'. Transient errors are retried (bounded)
    and never counted toward the stop rule unless every retry is exhausted."""
    if isinstance(exc, MemoryError):  # includes numpy._core._exceptions._ArrayMemoryError
        return "transient"
    if isinstance(exc, (RuntimeError, SystemError)) and str(exc).strip() == "Unknown exception":
        # SWIG's catch-all for a C++ exception from PyNEC; on 2026-11-09 this was std::bad_alloc
        # at the worker's ulimit -v. Bounded retries keep a genuine NEC error from looping.
        return "transient"
    return "case"


def describe_exception(exc: BaseException) -> dict:
    import traceback
    return {"error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "error": str(exc) or repr(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
            "classification": classify_exception(exc)}


def reexec_worker():
    """Replace this worker with a fresh process (same argv, env, affinity and ulimit)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)


def run_unit_inprocess(fn) -> dict:
    """Run a unit in this process (tests and pilot-sized runs)."""
    try:
        fn()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, **describe_exception(exc)}


def run_unit_in_child(fn) -> dict:
    """Run a unit in a forked child so PyNEC result storage cannot accumulate in the worker."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: never return into the parent's control flow
        code = 0
        try:
            os.close(r)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                fn()
                payload = {"ok": True}
            except BaseException as exc:
                payload = {"ok": False, **describe_exception(exc)}
                code = 3
            os.write(w, json.dumps(payload).encode())
            os.close(w)
        except BaseException:
            code = 4
        finally:
            os._exit(code)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 1 << 16)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        return {"ok": False, "error_type": "ChildKilledBySignal", "error": f"unit child killed by signal {sig}"
                " (SIGKILL usually means the kernel OOM killer)", "traceback": "",
                "classification": "transient" if sig == signal.SIGKILL else "case"}
    try:
        payload = json.loads(b"".join(chunks).decode() or "{}")
    except ValueError:
        payload = {}
    if not payload:
        payload = {"ok": False, "error_type": "ChildNoReport", "traceback": "",
                   "error": f"unit child exited {os.WEXITSTATUS(status)} without a report",
                   "classification": "case"}
    return payload


def _transient_path(out: Path, map_id: str, case_id: str, n: int) -> Path:
    return out / map_id / "cases" / case_id / f"array_{n}.transient.json"


def production_worker(
    a,
    manifest: dict,
    scope: str,
    manifest_hash: str,
    freeze: str,
    trace_out: Path,
    basis_builder=RisBasis,
    frame_fn=channel_frame,
    input_physics_fn=input_physics,
    sleep_fn=time.sleep,
    unit_runner=run_unit_inprocess,
    mem_fn=mem_available_bytes,
    reexec_fn=None,
) -> None:
    """Production channel worker: process units in preference array order, resume, stop-rule."""
    w_tag = f"w{str(a.worker_id).lstrip('w')}"
    workers_dir = a.out / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    running_marker = workers_dir / f"{w_tag}.RUNNING"
    done_marker = workers_dir / f"{w_tag}.DONE"
    failed_marker = workers_dir / f"{w_tag}.FAILED"

    if done_marker.exists():
        done_marker.unlink()
    if failed_marker.exists():
        failed_marker.unlink()

    running_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")

    def mark_failed(err_msg: str):
        if running_marker.exists():
            try:
                running_marker.unlink()
            except OSError:
                pass
        atomic_write_json(
            failed_marker,
            {
                "worker_id": a.worker_id,
                "status": "failed",
                "error": err_msg,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    def handle_sig(sig, frame):
        mark_failed(f"interrupted by signal {sig}")
        sys.exit(128 + sig)

    old_sigint = signal.signal(signal.SIGINT, handle_sig)
    old_sigterm = signal.signal(signal.SIGTERM, handle_sig)

    try:
        # Validate cpus and affinity
        cpus = [int(c.strip()) for c in a.cpus.split(",") if c.strip()]
        valid_cpus = check_cpu_set(cpus)
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, valid_cpus)
            affinity = os.sched_getaffinity(0)
            if 5 in affinity or 11 in affinity:
                raise ValueError(f"CPU affinity contains forbidden core 5 or 11: {affinity}")

        # Ordered array preference
        arrays = [int(x.strip()) for x in a.arrays.split(",") if x.strip()]
        for n in arrays:
            if n not in ARRAYS:
                raise ValueError(f"unsupported array size: {n}")

        map_groups = cases_by_map(manifest["cases"])

        built_basis_in_process = False
        for n in arrays:
            basis = None
            center, z, zsha = None, None, None
            setup_s = 0.0

            while True:
                claimed_any = False
                all_units_done_or_failed = True
                waiting_retry = False

                for map_id, map_cases in map_groups.items():
                    if (a.out / map_id / "FAILED").exists() or (trace_out / map_id / "FAILED").exists():
                        continue

                    for case in map_cases:
                        if (a.out / map_id / "FAILED").exists():
                            break  # a map that stopped mid-scan must not keep consuming units
                        case_id = case["id"]
                        unit_h5 = a.out / map_id / "cases" / case_id / f"array_{n}.h5"
                        unit_json = a.out / map_id / "cases" / case_id / f"array_{n}.json"
                        unit_failed = a.out / map_id / "cases" / case_id / f"array_{n}.FAILED.json"

                        if unit_failed.exists():
                            continue
                        if (trace_out / map_id / "cases" / f"{case_id}.FAILED.json").exists():
                            # Trace already failed this case; it counts once toward the map stop rule.
                            continue

                        if is_unit_done(unit_h5, unit_json, manifest_hash):
                            continue

                        all_units_done_or_failed = False

                        tpath = _transient_path(a.out, map_id, case_id, n)
                        tstate = None
                        if tpath.is_file():
                            try:
                                tstate = json.loads(tpath.read_text())
                            except ValueError:
                                tstate = None
                            if tstate and time.time() < float(tstate.get("next_retry_unix", 0)):
                                waiting_retry = True
                                continue

                        trace_record_path = trace_out / map_id / "cases" / f"{case_id}.json"
                        if not trace_record_path.is_file():
                            continue

                        try:
                            tr = json.loads(trace_record_path.read_text())
                            if tr.get("status") != "pass" or tr.get("case_manifest_sha256") != manifest_hash:
                                continue
                            tr_arrays = {r.get("array_elements") for r in tr.get("records", [])}
                            if n not in tr_arrays:
                                continue
                        except Exception:
                            continue

                        # Unit is eligible; attempt claim with non-blocking flock
                        lock_path = a.out / "claims" / map_id / str(n) / f"{case_id}.lock"
                        lock_path.parent.mkdir(parents=True, exist_ok=True)
                        lock_f = open(lock_path, "w")
                        try:
                            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except (BlockingIOError, OSError):
                            lock_f.close()
                            continue

                        claimed_any = True
                        # Lazy basis build. Host-memory shortage or a basis error is a worker-level
                        # problem: it must never be recorded as a case failure (stop rule).
                        if basis is None:
                            try:
                                check_mem_available(16 * 1024 * 1024 * 1024)
                            except RuntimeError as mem_err:
                                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                                lock_f.close()
                                print(f"{w_tag}: deferring array {n}: {mem_err}", flush=True)
                                sleep_fn(300)
                                break
                            if built_basis_in_process and reexec_fn is not None:
                                # PyNEC never returns the memory of an earlier basis (del + gc does not
                                # help), so a second basis in the same process hits ulimit -v. Restart
                                # this worker in a fresh process; resume skips everything already done.
                                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                                lock_f.close()
                                print(f"{w_tag}: re-exec before basis {n} to release memory", flush=True)
                                reexec_fn()
                                raise RuntimeError("re-exec returned")
                            try:
                                center, z, zsha = input_physics_fn(a.physics, n)
                                b_start = time.perf_counter()
                                basis = basis_builder(center, ARRAYS[n], z)
                                built_basis_in_process = True
                                setup_s = time.perf_counter() - b_start
                                print(f"{w_tag}: basis {n} ready in {setup_s:.1f} s", flush=True)
                            except BaseException:
                                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                                lock_f.close()
                                raise
                        try:
                            avail = mem_fn()
                            if avail is not None and avail < UNIT_ADMISSION_BYTES:
                                # Admission control: wait for memory instead of failing a case.
                                print(f"{w_tag}: MemAvailable {avail/2**30:.1f} GiB < 4 GiB; waiting", flush=True)
                                waiting_retry = True
                                sleep_fn(120)
                                continue
                            unit_kwargs = dict(
                                a=a, map_id=map_id, case=case, n=n, basis=basis, center=center,
                                zsha=zsha, setup_s=setup_s, manifest_hash=manifest_hash,
                                freeze=freeze, trace_out=trace_out,
                                trace_record_path=trace_record_path, trace_record=tr,
                                frame_fn=frame_fn)
                            outcome = unit_runner(lambda: process_unit(**unit_kwargs))
                            if outcome.get("ok") and not is_unit_done(unit_h5, unit_json, manifest_hash):
                                outcome = {"ok": False, "error_type": "UnverifiedShard", "traceback": "",
                                           "error": "unit reported success but shard/sidecar do not verify",
                                           "classification": "case"}
                            if outcome.get("ok"):
                                tpath.unlink(missing_ok=True)
                            else:
                                tmp_h5 = a.out / map_id / "cases" / case_id / f"array_{n}.h5.tmp"
                                tmp_h5.unlink(missing_ok=True)
                                now = time.time()
                                attempt = {k: outcome.get(k) for k in ("error_type", "error", "traceback", "classification")}
                                attempt["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
                                attempt["mem_available_bytes"] = mem_fn()
                                history = (tstate or {}).get("attempts", []) + [attempt]
                                n_transient = sum(1 for x in history if x["classification"] == "transient")
                                if outcome.get("classification") == "transient" and n_transient < MAX_TRANSIENT_ATTEMPTS:
                                    backoff = TRANSIENT_BACKOFF_S[min(n_transient, len(TRANSIENT_BACKOFF_S)) - 1]
                                    atomic_write_json(tpath, {"case_id": case_id, "map_id": map_id, "array": n,
                                                              "attempts": history, "next_retry_unix": now + backoff})
                                    print(f"{w_tag}: transient {attempt['error_type']} on {map_id}/{n}/{case_id};"
                                          f" retry in {backoff} s", flush=True)
                                    waiting_retry = True
                                else:
                                    reason = ("transient_retries_exhausted"
                                              if outcome.get("classification") == "transient" else "case_failure")
                                    atomic_write_json(unit_failed, {
                                        "case_id": case_id, "map_id": map_id, "array": n, "stage": "channel",
                                        "failure_kind": reason, "error_type": attempt["error_type"],
                                        "error": attempt["error"], "traceback": attempt["traceback"],
                                        "attempts": history, "time_utc": attempt["time_utc"]})
                                    tpath.unlink(missing_ok=True)
                                    update_map_failure_status(map_id, map_cases, trace_out, a.out)
                        finally:
                            try:
                                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                            except OSError:
                                pass
                            lock_f.close()

                        update_map_completion_status(map_id, map_cases, trace_out, a.out, manifest_hash)

                if claimed_any:
                    continue

                if all_units_done_or_failed:
                    break

                if waiting_retry:
                    sleep_fn(60)
                    continue

                # When nothing is eligible, check if trace is still running on any in-scope map
                trace_pending = False
                for map_id in map_groups:
                    t_done = (trace_out / map_id / "DONE").exists()
                    t_failed = (trace_out / map_id / "FAILED").exists()
                    if not t_done and not t_failed:
                        trace_pending = True
                        break

                if trace_pending:
                    sleep_fn(60)
                    continue
                else:
                    break

            # Free basis before moving to the next array
            if basis is not None:
                del basis
                basis = None
                gc.collect()

        # Final sweep: a map whose last event was a trace failure still gets its DONE/FAILED marker.
        for map_id, map_cases in map_groups.items():
            if not update_map_failure_status(map_id, map_cases, trace_out, a.out):
                update_map_completion_status(map_id, map_cases, trace_out, a.out, manifest_hash)

        if running_marker.exists():
            running_marker.unlink()
        atomic_write_json(
            done_marker,
            {
                "worker_id": a.worker_id,
                "status": "done",
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    except Exception as exc:
        mark_failed(str(exc))
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production", action="store_true", default=False)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trace-root", type=Path)
    p.add_argument("--trace-out", type=Path)
    p.add_argument("--physics", type=Path, default=Path("bench/CoupledRIS-Bench/physics"))
    p.add_argument("--array", type=int, choices=ARRAYS)
    p.add_argument("--arrays", type=str)
    p.add_argument("--worker-id", type=str)
    p.add_argument("--cpus", type=str)
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--smoke-unregistered", action="store_true", default=False)
    a = p.parse_args(argv)

    if not a.production:
        if bool(a.array is not None) == bool(a.assemble):
            p.error("pilot mode requires exactly one of --array or --assemble")
        if not a.trace_root and a.trace_out:
            a.trace_root = a.trace_out
        if not a.trace_root and not a.assemble:
            p.error("--trace-root is required for --array")
        freeze = verify_freeze()
        v, a.scope, manifest_hash = registered_manifest(a.manifest, freeze)
        cases = v["cases"]
        a.out.mkdir(parents=True, exist_ok=True)
        if a.assemble:
            assemble(a, manifest_hash, freeze)
        else:
            write_array(a, cases, manifest_hash, freeze)
        return

    # Production mode
    trace_out = a.trace_out or a.trace_root
    a.out.mkdir(parents=True, exist_ok=True)
    freeze = verify_freeze()

    if a.smoke_unregistered:
        if "t6b2-smoke" not in str(a.out):
            raise ValueError("--smoke-unregistered requires output path containing 't6b2-smoke'")
        manifest_bytes = a.manifest.read_bytes()
        manifest_hash = sha256(a.manifest)
        v = json.loads(manifest_bytes.decode("utf-8"))
        scope = v.get("run_scope") or v.get("scope") or "development_full"
    else:
        v, scope, manifest_hash = registered_manifest(a.manifest, freeze)
        if scope not in PRODUCTION_SCOPES:
            raise ValueError(f"production mode requires scope in {PRODUCTION_SCOPES}, got {scope}")

    a.scope = scope
    if a.assemble:
        production_assemble(a, v, scope, manifest_hash, freeze)
    else:
        if not trace_out:
            p.error("--trace-out is required for production worker mode")
        if not a.worker_id:
            p.error("--worker-id is required for production worker mode")
        if not a.arrays:
            p.error("--arrays is required for production worker mode")
        if not a.cpus:
            p.error("--cpus is required for production worker mode")
        production_worker(a, v, scope, manifest_hash, freeze, trace_out,
                          unit_runner=run_unit_in_child, reexec_fn=reexec_worker)


if __name__ == "__main__":
    main()
