"""Tests for production channel worker mode, assemble, resume, and stop-rules."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import build_bench_production_channels as prod_ch
from atomwise_ris.bench.production_freeze import sha256
from atomwise_ris.bench.production_run import check_cpu_set, stop_rule_exceeded


def make_dummy_case(map_id: str, case_idx: int, frame_indices: tuple[int, ...] = (0,)) -> dict:
    return {
        "id": f"{map_id}-v11-{case_idx:04d}",
        "map_id": map_id,
        "bs_xyz": [10.0, 20.0, 15.0],
        "ris_xyz": [30.0, 40.0, 5.0],
        "ris_normal_xyz": [0.0, 1.0, 0.0],
        "frames": [
            {
                "frame": f_idx,
                "time_s": f_idx * 0.1,
                "users": [
                    {"user_id": f"u_{f_idx}_0", "xyz": [15.0, 25.0, 1.5], "bs_link": "los", "ris_los": True}
                ],
            }
            for f_idx in frame_indices
        ],
    }


def write_fake_trace_record(
    trace_out: Path,
    map_id: str,
    case_id: str,
    manifest_hash: str,
    arrays: tuple[int, ...] = (64, 128, 256),
    frames: tuple[int, ...] = (0,),
    status: str = "pass",
) -> Path:
    records = []
    for n in arrays:
        for f in frames:
            npz_dir = trace_out / map_id / f"n{n}" / case_id
            npz_dir.mkdir(parents=True, exist_ok=True)
            npz_path = npz_dir / f"frame-{f:04d}.npz"
            # Write a small file
            npz_path.write_bytes(b"FAKE_NPZ_DATA_" + f"{n}_{f}".encode())
            records.append(
                {
                    "case_id": case_id,
                    "map_id": map_id,
                    "frame": f,
                    "array_elements": n,
                    "sha256": sha256(npz_path),
                    "bytes": npz_path.stat().st_size,
                }
            )

    rec_data = {
        "status": status,
        "case_id": case_id,
        "map_id": map_id,
        "run_scope": "development_full",
        "case_manifest_sha256": manifest_hash,
        "scene_sha256": "fake_scene_sha",
        "geometry": {},
        "records": records,
    }
    cases_dir = trace_out / map_id / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cases_dir / f"{case_id}.json"
    rec_path.write_text(json.dumps(rec_data, indent=2) + "\n")
    return rec_path


class FakeBasis:
    def __init__(self, centers, mesh, z):
        self.n = len(centers)
        self.mesh = mesh
        self.z = z
        self.archive_error = 1e-6


def fake_physics(root: Path, n: int) -> tuple[np.ndarray, np.ndarray, str]:
    return np.zeros((n, 3)), np.eye(n, dtype=complex), "fake_zsha_123"


def make_fake_frame_fn(counter_dict: dict | None = None, failing_cases: set[str] | None = None):
    def fake_frame_fn(h5, key, case, frame, n, basis, center, zsha, npz_path, rec=None):
        if counter_dict is not None:
            counter_dict[key] = counter_dict.get(key, 0) + 1
        if failing_cases and case["id"] in failing_cases:
            raise ValueError(f"simulated dense round-trip failed at {key}")
        if rec is not None:
            digest = sha256(npz_path)
            file_bytes = npz_path.stat().st_size
            if digest != rec["sha256"] or file_bytes != rec["bytes"]:
                raise ValueError(f"trace artifact hash mismatch: {npz_path}")

        g = h5.create_group(key)
        g.create_dataset("test_data", data=np.array([1, 2, 3]))
        g.attrs["stored_rate_bps_hz"] = 2.5
        g.attrs["ris_archive_sha256"] = zsha

        return key, {
            "users": [{"user_id": usr["user_id"], "bs_user": "los", "ris_user": "los"} for usr in frame["users"]],
            "link_classes": {"bs_ris": "los"},
            "trace_sha256": rec["sha256"] if rec else sha256(npz_path),
            "trace_path_host_only": str(npz_path),
            "source_geometry_sha256": "fake_scene_sha",
            "paths": 10,
            "elapsed_channel_s": 0.001,
            "raw_path_bytes": 100,
            "geometry_agreement": {"bs": [True], "ris": [True]},
        }

    return fake_frame_fn


class ProductionChannelsFullTests(unittest.TestCase):
    def setUp(self):
        self.orig_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.trace_out = self.root / "trace"
        self.ch_out = self.root / "channel"
        self.physics = self.root / "physics"
        self.trace_out.mkdir()
        self.ch_out.mkdir()
        self.physics.mkdir()

    def tearDown(self):
        if self.orig_affinity is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, self.orig_affinity)
            except OSError:
                pass
        self.temp_dir.cleanup()

    def test_cpus_5_and_11_are_refused(self):
        """Refuse CPU sets that are empty or include core 5 or 11."""
        for bad in ([], [5], [11], [0, 5], [1, 11], [5, 11]):
            with self.assertRaises(ValueError):
                check_cpu_set(bad)

        # Valid cpu sets pass
        self.assertEqual(check_cpu_set([0, 6]), (0, 6))
        self.assertEqual(check_cpu_set([1, 7]), (1, 7))
        self.assertEqual(check_cpu_set([2, 8]), (2, 8))

        # Check worker CLI validation
        manifest_path = self.root / "cases.json"
        manifest_path.write_text(json.dumps({"cases": []}))
        with self.assertRaises(ValueError):
            prod_ch.main(["--production", "--manifest", str(manifest_path), "--out", str(self.ch_out),
                          "--trace-out", str(self.trace_out), "--worker-id", "0",
                          "--arrays", "64", "--cpus", "0,5"])

    def test_eligibility_waits_for_the_trace_record(self):
        """Units are not processed until trace case record exists with status pass and valid hash."""
        case = make_dummy_case("munich", 0)
        manifest = {"cases": [case], "scope": "development_full"}
        m_hash = "dummy_manifest_hash"
        freeze = "dummy_freeze"

        counts = {}
        frame_fn = make_fake_frame_fn(counts)
        args = argparse.Namespace(
            worker_id="0",
            arrays="64",
            cpus="0,6",
            out=self.ch_out,
            physics=self.physics,
        )

        sleep_calls = []

        def mock_sleep(s):
            sleep_calls.append(s)

        # 1. No trace record exists, and trace is marked DONE -> worker exits without processing unit
        (self.trace_out / "munich").mkdir(parents=True, exist_ok=True)
        (self.trace_out / "munich" / "DONE").write_text("done\n")

        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
            sleep_fn=mock_sleep,
        )
        self.assertEqual(len(counts), 0)
        self.assertFalse((self.ch_out / "munich" / "cases" / case["id"] / "array_64.h5").exists())

        # 2. Trace not done yet, worker would sleep waiting
        (self.trace_out / "munich" / "DONE").unlink()
        iter_count = [0]

        def sleep_and_create_trace(s):
            sleep_calls.append(s)
            iter_count[0] += 1
            if iter_count[0] == 1:
                # Write trace record during wait
                write_fake_trace_record(self.trace_out, "munich", case["id"], m_hash, arrays=(64,))
                (self.trace_out / "munich" / "DONE").write_text("done\n")

        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
            sleep_fn=sleep_and_create_trace,
        )

        # Now unit was processed after trace record became available!
        self.assertEqual(len(counts), 1)
        self.assertTrue((self.ch_out / "munich" / "cases" / case["id"] / "array_64.h5").exists())
        self.assertTrue((self.ch_out / "munich" / "cases" / case["id"] / "array_64.json").exists())

    def test_done_units_skipped_and_corrupted_sidecar_is_not(self):
        """Verified done units are skipped; if sidecar hash is corrupted, unit is reprocessed."""
        case = make_dummy_case("munich", 0)
        manifest = {"cases": [case], "scope": "development_full"}
        m_hash = "test_manifest_hash"
        freeze = "test_freeze"

        (self.trace_out / "munich").mkdir(parents=True, exist_ok=True)
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        write_fake_trace_record(self.trace_out, "munich", case["id"], m_hash, arrays=(64,))

        counts = {}
        frame_fn = make_fake_frame_fn(counts)
        args = argparse.Namespace(
            worker_id="0",
            arrays="64",
            cpus="0,6",
            out=self.ch_out,
            physics=self.physics,
        )

        # Pass 1: processes the unit
        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
        )
        key = f"munich/64/{case['id']}/0"
        self.assertEqual(counts[key], 1)

        # Pass 2: unit is done -> skipped!
        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
        )
        self.assertEqual(counts[key], 1)

        # Corrupt sidecar hash
        side_p = self.ch_out / "munich" / "cases" / case["id"] / "array_64.json"
        side_data = json.loads(side_p.read_text())
        side_data["h5_sha256"] = "corrupted_hash"
        side_p.write_text(json.dumps(side_data))

        # Pass 3: corrupted sidecar is NOT skipped -> reprocessed!
        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
        )
        self.assertEqual(counts[key], 2)
        # Sidecar is valid again
        fresh_side = json.loads(side_p.read_text())
        h5_p = self.ch_out / "munich" / "cases" / case["id"] / "array_64.h5"
        self.assertEqual(fresh_side["h5_sha256"], sha256(h5_p))

    def test_two_workers_never_process_the_same_unit_flock(self):
        """A unit locked by one worker is skipped by another worker."""
        case = make_dummy_case("munich", 0)
        manifest = {"cases": [case], "scope": "development_full"}
        m_hash = "test_manifest_hash"
        freeze = "test_freeze"

        (self.trace_out / "munich").mkdir(parents=True, exist_ok=True)
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        write_fake_trace_record(self.trace_out, "munich", case["id"], m_hash, arrays=(64,))

        # Lock the claim file for this unit as if worker 0 holds it
        lock_p = self.ch_out / "claims" / "munich" / "64" / f"{case['id']}.lock"
        lock_p.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_p, "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        counts = {}
        frame_fn = make_fake_frame_fn(counts)
        args_w1 = argparse.Namespace(
            worker_id="1",
            arrays="64",
            cpus="1,7",
            out=self.ch_out,
            physics=self.physics,
        )

        try:
            # Worker 1 runs while unit is locked
            prod_ch.production_worker(
                args_w1, manifest, "development_full", m_hash, freeze,
                trace_out=self.trace_out,
                basis_builder=FakeBasis,
                frame_fn=frame_fn,
                input_physics_fn=fake_physics,
            )
            # Worker 1 did not process the locked unit
            self.assertEqual(len(counts), 0)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

        # Once lock is released, worker 1 can process it
        prod_ch.production_worker(
            args_w1, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
        )
        self.assertEqual(len(counts), 1)

    def test_the_basis_is_built_lazily_once_per_array(self):
        """RisBasis is built lazily on first claimable unit, reused for all units of that array, and freed."""
        c0 = make_dummy_case("munich", 0)
        c1 = make_dummy_case("munich", 1)
        # 50 cases so one failure (2%) does not exceed the stop rule; the other 48 have no trace yet.
        manifest = {"cases": [c0, c1] + [make_dummy_case("munich", i) for i in range(2, 50)],
                    "scope": "development_full"}
        m_hash = "test_manifest_hash"
        freeze = "test_freeze"

        (self.trace_out / "munich").mkdir(parents=True, exist_ok=True)
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        # Trace records exist for array 64, but not 128
        write_fake_trace_record(self.trace_out, "munich", c0["id"], m_hash, arrays=(64,))
        write_fake_trace_record(self.trace_out, "munich", c1["id"], m_hash, arrays=(64,))

        basis_calls = {}

        def tracking_basis_builder(centers, mesh, z):
            n = len(centers)
            basis_calls[n] = basis_calls.get(n, 0) + 1
            return FakeBasis(centers, mesh, z)

        counts = {}
        frame_fn = make_fake_frame_fn(counts)
        args = argparse.Namespace(
            worker_id="0",
            arrays="64,128",
            cpus="0,6",
            out=self.ch_out,
            physics=self.physics,
        )

        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=tracking_basis_builder,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
        )

        # Basis for 64 was built exactly ONCE and shared across both units
        self.assertEqual(basis_calls.get(64), 1)
        # Array 128 had no eligible units, so basis for 128 was NEVER built (lazy!)
        self.assertNotIn(128, basis_calls)
        self.assertEqual(len(counts), 2)

    def test_unit_failure_writes_failed_and_continues(self):
        """An exception during unit processing writes array_<N>.FAILED.json, removes tmp, and continues."""
        c0 = make_dummy_case("munich", 0)
        c1 = make_dummy_case("munich", 1)
        # 50 cases so one failure (2%) does not exceed the stop rule; the other 48 have no trace yet.
        manifest = {"cases": [c0, c1] + [make_dummy_case("munich", i) for i in range(2, 50)],
                    "scope": "development_full"}
        m_hash = "test_manifest_hash"
        freeze = "test_freeze"

        (self.trace_out / "munich").mkdir(parents=True, exist_ok=True)
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        write_fake_trace_record(self.trace_out, "munich", c0["id"], m_hash, arrays=(64,))
        write_fake_trace_record(self.trace_out, "munich", c1["id"], m_hash, arrays=(64,))

        # c0 will fail in frame_fn, c1 will pass
        frame_fn = make_fake_frame_fn(failing_cases={c0["id"]})
        args = argparse.Namespace(
            worker_id="0",
            arrays="64",
            cpus="0,6",
            out=self.ch_out,
            physics=self.physics,
        )

        # Worker should finish without crashing
        prod_ch.production_worker(
            args, manifest, "development_full", m_hash, freeze,
            trace_out=self.trace_out,
            basis_builder=FakeBasis,
            frame_fn=frame_fn,
            input_physics_fn=fake_physics,
        )

        # c0 has array_64.FAILED.json, no tmp, no final h5
        c0_dir = self.ch_out / "munich" / "cases" / c0["id"]
        self.assertTrue((c0_dir / "array_64.FAILED.json").exists())
        self.assertFalse((c0_dir / "array_64.h5.tmp").exists())
        self.assertFalse((c0_dir / "array_64.h5").exists())
        failed_rec = json.loads((c0_dir / "array_64.FAILED.json").read_text())
        self.assertIn("simulated dense round-trip failed", failed_rec["error"])
        self.assertIn("time_utc", failed_rec)

        # c1 succeeded!
        c1_dir = self.ch_out / "munich" / "cases" / c1["id"]
        self.assertTrue((c1_dir / "array_64.h5").exists())
        self.assertTrue((c1_dir / "array_64.json").exists())

    def test_stop_rule_writes_map_failed_and_trace_failures_count(self):
        """Stop rule: 3/120 fails map, 2/120 does not; 1/23 fails challenge map; trace failures count."""
        # 1. 120-case map
        cases_120 = [make_dummy_case("munich", i) for i in range(120)]
        m_120 = {"cases": cases_120, "scope": "development_full"}
        m_hash = "h120"
        freeze = "freeze"

        (self.trace_out / "munich" / "cases").mkdir(parents=True, exist_ok=True)
        (self.ch_out / "munich" / "cases").mkdir(parents=True, exist_ok=True)

        # 2 failures: 1 trace failure, 1 channel failure -> total 2/120 (<= 2%)
        (self.trace_out / "munich" / "cases" / f"{cases_120[0]['id']}.FAILED.json").write_text("{}")
        c1_dir = self.ch_out / "munich" / "cases" / cases_120[1]["id"]
        c1_dir.mkdir(parents=True, exist_ok=True)
        (c1_dir / "array_64.FAILED.json").write_text("{}")

        self.assertFalse(prod_ch.update_map_failure_status("munich", cases_120, self.trace_out, self.ch_out))
        self.assertFalse((self.ch_out / "munich" / "FAILED").exists())

        # 3rd failure: another trace failure -> total 3/120 (> 2%)
        (self.trace_out / "munich" / "cases" / f"{cases_120[2]['id']}.FAILED.json").write_text("{}")
        self.assertTrue(prod_ch.update_map_failure_status("munich", cases_120, self.trace_out, self.ch_out))
        self.assertTrue((self.ch_out / "munich" / "FAILED").exists())
        failed_info = json.loads((self.ch_out / "munich" / "FAILED").read_text())
        self.assertEqual(failed_info["failed_cases_count"], 3)
        self.assertEqual(failed_info["total_cases"], 120)

        # 2. 23-case map (challenge)
        cases_23 = [make_dummy_case("rotterdam", i) for i in range(23)]
        (self.trace_out / "rotterdam" / "cases").mkdir(parents=True, exist_ok=True)
        (self.ch_out / "rotterdam" / "cases").mkdir(parents=True, exist_ok=True)

        # 0 failures: no stop rule
        self.assertFalse(prod_ch.update_map_failure_status("rotterdam", cases_23, self.trace_out, self.ch_out))
        self.assertFalse((self.ch_out / "rotterdam" / "FAILED").exists())

        # 1 failure -> 1/23 (> 2%) triggers stop rule!
        r0_dir = self.ch_out / "rotterdam" / "cases" / cases_23[0]["id"]
        r0_dir.mkdir(parents=True, exist_ok=True)
        (r0_dir / "array_128.FAILED.json").write_text("{}")

        self.assertTrue(prod_ch.update_map_failure_status("rotterdam", cases_23, self.trace_out, self.ch_out))
        self.assertTrue((self.ch_out / "rotterdam" / "FAILED").exists())

    def test_map_done_only_when_complete(self):
        """C/<map>/DONE is written only when trace is DONE and all non-failed cases have all 3 array shards."""
        # 100 cases: 1 failure (c0) -> 1% failure rate <= 2% stop rule
        cases = [make_dummy_case("munich", i) for i in range(100)]
        c0 = cases[0]
        m_hash = "h_done"

        # Case 0 failed in trace
        (self.trace_out / "munich" / "cases").mkdir(parents=True, exist_ok=True)
        (self.trace_out / "munich" / "cases" / f"{c0['id']}.FAILED.json").write_text("{}")

        # Trace map not DONE yet
        self.assertFalse(prod_ch.update_map_completion_status("munich", cases, self.trace_out, self.ch_out, m_hash))
        self.assertFalse((self.ch_out / "munich" / "DONE").exists())

        # Trace marked DONE
        (self.trace_out / "munich" / "DONE").write_text("done\n")

        # Setup 99 non-failed cases: all have 64 and 128, but case 1 is missing 256
        for c in cases[1:]:
            c_dir = self.ch_out / "munich" / "cases" / c["id"]
            c_dir.mkdir(parents=True, exist_ok=True)
            for n in (64, 128):
                h5_p = c_dir / f"array_{n}.h5"
                h5_p.write_bytes(b"DATA")
                side_p = c_dir / f"array_{n}.json"
                side_p.write_text(json.dumps({"h5_sha256": sha256(h5_p), "case_manifest_sha256": m_hash}))
            if c["id"] != cases[1]["id"]:
                h5_256 = c_dir / "array_256.h5"
                h5_256.write_bytes(b"DATA256")
                side_256 = c_dir / "array_256.json"
                side_256.write_text(json.dumps({"h5_sha256": sha256(h5_256), "case_manifest_sha256": m_hash}))

        # Still missing 256 for case 1
        self.assertFalse(prod_ch.update_map_completion_status("munich", cases, self.trace_out, self.ch_out, m_hash))
        self.assertFalse((self.ch_out / "munich" / "DONE").exists())

        # Add shard for 256 for case 1
        c1_dir = self.ch_out / "munich" / "cases" / cases[1]["id"]
        h5_256 = c1_dir / "array_256.h5"
        h5_256.write_bytes(b"DATA256")
        side_256 = c1_dir / "array_256.json"
        side_256.write_text(json.dumps({"h5_sha256": sha256(h5_256), "case_manifest_sha256": m_hash}))

        # Now all 99 non-failed cases have all 3 verified shards!
        self.assertTrue(prod_ch.update_map_completion_status("munich", cases, self.trace_out, self.ch_out, m_hash))
        self.assertTrue((self.ch_out / "munich" / "DONE").exists())
        done_info = json.loads((self.ch_out / "munich" / "DONE").read_text())
        self.assertEqual(done_info["cases_total"], 100)
        self.assertEqual(done_info["cases_included"], 99)
        self.assertEqual(done_info["failed_cases"], [c0["id"]])

    def test_assemble_excludes_failed_cases_and_refuses_when_map_not_done(self):
        """Assemble requires all maps DONE, excludes failed cases, and refuses to overwrite."""
        cases = [make_dummy_case("munich", i) for i in range(50)]
        c0 = cases[0]
        c1 = cases[1]
        manifest = {"cases": cases, "scope": "development_full"}
        m_hash = "h_asmb"
        freeze = "f_asmb"

        args = argparse.Namespace(out=self.ch_out)

        # 1. Map not DONE -> raises RuntimeError
        with self.assertRaises(RuntimeError):
            prod_ch.production_assemble(args, manifest, "development_full", m_hash, freeze)

        # 2. Mark map DONE with c0 failed, c1..c49 included
        (self.ch_out / "munich").mkdir(parents=True, exist_ok=True)
        (self.ch_out / "munich" / "DONE").write_text(json.dumps({
            "status": "done",
            "map_id": "munich",
            "cases_total": 50,
            "cases_included": 49,
            "failed_cases": [c0["id"]],
        }))

        # Setup shards for all included cases (cases[1:])
        for c in cases[1:]:
            c_dir = self.ch_out / "munich" / "cases" / c["id"]
            c_dir.mkdir(parents=True, exist_ok=True)
            for n in (64, 128, 256):
                h5_p = c_dir / f"array_{n}.h5"
                with h5py.File(h5_p, "w") as h:
                    g = h.create_group(f"munich/{n}/{c['id']}/0")
                    g.create_dataset("weights", data=[1.0, 2.0])
                side_p = c_dir / f"array_{n}.json"
                side_p.write_text(json.dumps({
                    "records": {f"munich/{n}/{c['id']}/0": {"paths": 5}},
                    "h5_sha256": sha256(h5_p),
                    "case_manifest_sha256": m_hash,
                    "setup_s": 0.5,
                    "peak_rss_bytes": 10240,
                }))

        # Run assemble
        prod_ch.production_assemble(args, manifest, "development_full", m_hash, freeze)

        target_h5 = self.ch_out / "channel_realizations.h5"
        target_meta = self.ch_out / "metadata.json"
        target_run = self.ch_out / "channel_run.json"

        self.assertTrue(target_h5.is_file())
        self.assertTrue(target_meta.is_file())
        self.assertTrue(target_run.is_file())

        with h5py.File(target_h5, "r") as h:
            # Included case is present
            self.assertIn(f"munich/64/{c1['id']}/0", h)
            # Failed case is excluded
            self.assertNotIn(f"munich/64/{c0['id']}/0", h)

        run_info = json.loads(target_run.read_text())
        self.assertEqual(run_info["maps"]["munich"]["failed_cases"], [c0["id"]])
        self.assertEqual(run_info["maps"]["munich"]["cases_included"], 49)

        # Refuses to overwrite
        with self.assertRaises(FileExistsError):
            prod_ch.production_assemble(args, manifest, "development_full", m_hash, freeze)

    def _args(self, arrays="64"):
        return argparse.Namespace(worker_id="0", arrays=arrays, cpus="0,6", out=self.ch_out, physics=self.physics)

    def test_memory_shortage_never_marks_a_case_failed(self):
        """Lead fix: a MemAvailable shortfall defers the basis build; no unit FAILED file is written."""
        case = make_dummy_case("munich", 0)
        manifest = {"cases": [case]}
        write_fake_trace_record(self.trace_out, "munich", case["id"], "h", arrays=(64,))
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        calls = {"n": 0}

        def flaky_mem(_min):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("insufficient MemAvailable")

        sleeps = []
        with patch.object(prod_ch, "check_mem_available", flaky_mem):
            prod_ch.production_worker(self._args(), manifest, "development_full", "h", "f",
                                      trace_out=self.trace_out, basis_builder=FakeBasis,
                                      frame_fn=make_fake_frame_fn(), input_physics_fn=fake_physics,
                                      sleep_fn=sleeps.append)
        unit = self.ch_out / "munich" / "cases" / case["id"]
        self.assertEqual(sleeps, [300])
        self.assertFalse(list(unit.glob("*.FAILED.json")))
        self.assertTrue(prod_ch.is_unit_done(unit / "array_64.h5", unit / "array_64.json", "h"))

    def test_basis_error_fails_worker_not_case(self):
        case = make_dummy_case("munich", 0)
        write_fake_trace_record(self.trace_out, "munich", case["id"], "h", arrays=(64,))
        (self.trace_out / "munich" / "DONE").write_text("done\n")

        def broken_basis(*_a):
            raise OSError("basis archive unreadable")

        with patch.object(prod_ch, "check_mem_available", lambda _m: None):
            with self.assertRaises(OSError):
                prod_ch.production_worker(self._args(), {"cases": [case]}, "development_full", "h", "f",
                                          trace_out=self.trace_out, basis_builder=broken_basis,
                                          frame_fn=make_fake_frame_fn(), input_physics_fn=fake_physics,
                                          sleep_fn=lambda s: None)
        self.assertFalse(list((self.ch_out).rglob("array_*.FAILED.json")))
        self.assertTrue((self.ch_out / "workers" / "w0.FAILED").exists())

    def test_trace_failed_last_case_still_gets_map_done(self):
        """Lead fix: a trace-failed case is settled, and the final sweep writes the map DONE."""
        cases = [make_dummy_case("munich", i) for i in range(60)]
        for c in cases[:-1]:
            write_fake_trace_record(self.trace_out, "munich", c["id"], "h", arrays=(64, 128, 256))
        (self.trace_out / "munich" / "cases" / f"{cases[-1]['id']}.FAILED.json").write_text("{}")
        (self.trace_out / "munich" / "DONE").write_text("done\n")
        sleeps = []
        with patch.object(prod_ch, "check_mem_available", lambda _m: None):
            prod_ch.production_worker(self._args("64,128,256"), {"cases": cases}, "development_full", "h", "f",
                                      trace_out=self.trace_out, basis_builder=FakeBasis,
                                      frame_fn=make_fake_frame_fn(), input_physics_fn=fake_physics,
                                      sleep_fn=sleeps.append)
        self.assertEqual(sleeps, [])
        done = json.loads((self.ch_out / "munich" / "DONE").read_text())
        self.assertEqual(done["failed_cases"], [cases[-1]["id"]])
        self.assertEqual(done["cases_included"], 59)

    def test_pilot_path_is_unchanged(self):
        """Without --production, pilot CLI requires single trace_manifest.json and --array/--assemble."""
        # Mutually exclusive --array and --assemble without --production
        with self.assertRaises(SystemExit):
            prod_ch.main(["--manifest", "dummy.json", "--out", str(self.ch_out)])

        # --trace-root is required for --array
        with self.assertRaises(SystemExit):
            prod_ch.main(["--manifest", "dummy.json", "--out", str(self.ch_out), "--array", "64"])

    def test_launcher_resumes_when_job_done_exists(self):
        """Launcher exits 0 immediately when JOB_DONE already exists."""
        import subprocess
        job_done = self.ch_out / "JOB_DONE"
        job_done.write_text("status=pass\n")
        manifest = self.root / "cases.json"
        manifest.write_text("{}")
        proc = subprocess.run(
            ["bash", "scripts/run_bench_production_full_channels.sh",
             str(Path.cwd()), str(manifest), str(self.trace_out), str(self.ch_out), str(self.physics)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("job already completed", proc.stderr)


if __name__ == "__main__":
    unittest.main()
