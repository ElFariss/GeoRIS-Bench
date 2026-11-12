import copy
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

import run_bench_production_trace as trace
from atomwise_ris.bench.production_run import ARRAY_SIZES, stop_rule_exceeded

ROOT = Path(__file__).resolve().parents[1]


def _make_dummy_npz(path: Path, frame: int = 0, array_elements: int = 64, users: list | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if users is None:
        users = [{"user_id": 1, "xyz": [1.0, 2.0, 3.0]}]
    meta = {
        "case_id": "test-case",
        "map_id": "munich",
        "frame": frame,
        "array_elements": array_elements,
        "tx_order": ["bs_port_0", "bs_port_1", "bs_port_2", "bs_port_3"] + [f"ue_{u['user_id']}" for u in users],
        "rx_order": [f"ris_{i}" for i in range(array_elements)]
        + ["bs_port_0", "bs_port_1", "bs_port_2", "bs_port_3"]
        + [f"ue_{u['user_id']}" for u in users],
    }
    np.savez_compressed(
        path,
        tx_xyz=np.zeros((5, 3), dtype=np.float64),
        rx_xyz=np.zeros((array_elements + 5, 3), dtype=np.float64),
        link_tx=np.zeros(10, dtype=np.int32),
        link_rx=np.zeros(10, dtype=np.int32),
        path_delay_s=np.zeros(10, dtype=np.float64),
        path_length_m=np.zeros(10, dtype=np.float64),
        path_jones=np.zeros((10, 2, 2), dtype=np.complex128),
        path_theta_t=np.zeros(10, dtype=np.float64),
        path_phi_t=np.zeros(10, dtype=np.float64),
        path_theta_r=np.zeros(10, dtype=np.float64),
        path_phi_r=np.zeros(10, dtype=np.float64),
        path_type=np.ones(10, dtype=np.uint8),
        metadata_json=np.asarray(json.dumps(meta)),
    )
    return {
        "file": str(path),
        "sha256": trace.sha256(path),
        "bytes": path.stat().st_size,
        "paths": 10,
        "nonzero_links": 5,
        "frame": frame,
        "array_elements": array_elements,
        "case_id": "test-case",
        "map_id": "munich",
        "elapsed_seconds": 0.1,
        "solver_elapsed_seconds": 0.05,
        "export_elapsed_seconds": 0.05,
    }


def _make_dummy_case(case_id: str, map_id: str = "munich", num_frames: int = 16) -> dict:
    frames = []
    for f in range(num_frames):
        frames.append({
            "frame": f,
            "time_s": f * 0.1,
            "users": [
                {"user_id": u, "xyz": [10.0 + f + u, 20.0 + f, 1.5], "velocity_xyz": [0.0, 0.0, 0.0]}
                for u in range(4)
            ],
        })
    return {
        "id": case_id,
        "map_id": map_id,
        "bs_xyz": [0.0, 0.0, 10.0],
        "ris_xyz": [15.0, 15.0, 12.0],
        "ris_normal_xyz": [1.0, 0.0, 0.0],
        "frames": frames,
    }


class ProductionTraceFullContractTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/previews/bench-users-v1.1/case_manifest.json').exists(), 'needs the packaged data artifact')
    def test_production_vs_pilot_manifest_acceptance(self):
        full_path = ROOT / "outputs/runs/bench_production_full_20261109/cases.json"
        chal_path = ROOT / "outputs/runs/bench_production_full_20261109/challenge_cases.json"
        p1_path = ROOT / "outputs/runs/bench_production_pilot_20261109/pilot_cases.json"
        p2_path = ROOT / "outputs/runs/bench_production_pilot2_20261109/pilot_cases.json"

        # Production scopes without --production must raise ValueError
        with self.assertRaises(ValueError):
            trace.verify_case_manifest(full_path, production=False)
        with self.assertRaises(ValueError):
            trace.verify_case_manifest(chal_path, production=False)

        # Pilot scopes with --production must raise ValueError
        with self.assertRaises(ValueError):
            trace.verify_case_manifest(p1_path, production=True)
        with self.assertRaises(ValueError):
            trace.verify_case_manifest(p2_path, production=True)

        # Production scopes with --production must succeed
        m_full, c_full = trace.verify_case_manifest(full_path, production=True)
        self.assertEqual(m_full["_run_scope"], "development_full")
        self.assertEqual(len(c_full), 360)
        # Verify 16 frames per case
        self.assertTrue(all(len(c["frames"]) == 16 for c in c_full))

        m_chal, c_chal = trace.verify_case_manifest(chal_path, production=True)
        self.assertEqual(m_chal["_run_scope"], "development_challenge")
        self.assertEqual(len(c_chal), 69)
        self.assertTrue(all(len(c["frames"]) == 16 for c in c_chal))

        # Test validation failures in production mode
        with TemporaryDirectory() as tmpdir:
            bad_cases_path = Path(tmpdir) / "bad_cases.json"

            # Case with 15 frames instead of 16
            bad_data = copy.deepcopy(json.loads(full_path.read_text()))
            bad_data["cases"][0]["frames"] = bad_data["cases"][0]["frames"][:15]
            bad_cases_path.write_text(json.dumps(bad_data))
            with self.assertRaises(ValueError):
                trace.verify_case_manifest(bad_cases_path, production=True)

            # Case with non-finite user xyz coordinate
            bad_data = copy.deepcopy(json.loads(full_path.read_text()))
            bad_data["cases"][0]["frames"][0]["users"][0]["xyz"] = [float("nan"), 0.0, 0.0]
            bad_cases_path.write_text(json.dumps(bad_data))
            with self.assertRaises(ValueError):
                trace.verify_case_manifest(bad_cases_path, production=True)

    def test_production_anchors_include_every_frames_users(self):
        case = _make_dummy_case("munich-v11-0000", "munich", num_frames=16)
        total_users = sum(len(f["users"]) for f in case["frames"])
        self.assertEqual(total_users, 64)

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            mock_scene_info = {
                "mesh_bounds_local_m": [[-100.0, -100.0, 0.0], [100.0, 100.0, 50.0]],
                "origin_m": [0.0, 0.0, 0.0],
                "scene_sha256": "fake_scene_sha256",
            }

            with patch.object(trace, "_munich_scene", return_value=mock_scene_info) as mock_munich:
                # In production mode: anchor_users must contain all 64 users
                info, origin = trace.build_scene(case, root, out, production=True)
                mock_munich.assert_called_once()
                anchor_users = mock_munich.call_args[1]["anchor_users"]
                self.assertEqual(len(anchor_users), 64)
                # Verify scene_dir is per-case in production
                scene_dir_arg = mock_munich.call_args[0][1]
                self.assertEqual(scene_dir_arg, root / "scenes" / "munich" / "munich-v11-0000")
                # Verify scene_info.json written beside scene.xml
                self.assertTrue((scene_dir_arg / "scene_info.json").is_file())

            with patch.object(trace, "_munich_scene", return_value=mock_scene_info) as mock_munich:
                # In pilot mode: anchor_users must contain only frame-0 users (4 users)
                info, origin = trace.build_scene(case, root, out, production=False)
                mock_munich.assert_called_once()
                anchor_users = mock_munich.call_args[1]["anchor_users"]
                self.assertEqual(len(anchor_users), 4)
                # Verify scene_dir is map-level in pilot
                scene_dir_arg = mock_munich.call_args[0][1]
                self.assertEqual(scene_dir_arg, root / "scenes" / "munich")

    def test_npz_naming_and_frame_metadata(self):
        case = _make_dummy_case("munich-v11-0000", "munich", num_frames=16)
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            frame_idx = 7
            frame = case["frames"][frame_idx]
            out_path = out_dir / case["map_id"] / "n64" / case["id"] / f"frame-{frame_idx:04d}.npz"
            self.assertEqual(out_path.name, "frame-0007.npz")

            rec = _make_dummy_npz(out_path, frame=frame_idx, array_elements=64, users=frame["users"])
            self.assertEqual(rec["frame"], 7)
            self.assertTrue(out_path.is_file())

            with np.load(out_path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata_json"]))
                self.assertEqual(meta["frame"], 7)
                expected_ues = [f"ue_{u['user_id']}" for u in frame["users"]]
                self.assertTrue(all(ue in meta["tx_order"] for ue in expected_ues))
                self.assertTrue(all(ue in meta["rx_order"] for ue in expected_ues))

    def test_resume_skips_verified_case_and_retraces_on_hash_mismatch(self):
        case = _make_dummy_case("munich-v11-0000", "munich", num_frames=2)
        manifest = {"_run_scope": "development_full", "cases": [case]}

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            trace_call_count = [0]
            scene_build_count = [0]

            def fake_scene_builder(c, r, o):
                scene_build_count[0] += 1
                return {"mesh_bounds_local_m": [[-50, -50, 0], [50, 50, 30]], "origin_m": [0, 0, 0], "scene_sha256": "fake_scene_sha"}, np.zeros(3)

            def fake_trace_one(rt, dr, r, c, count, o, frame=None, gpu_id=0):
                trace_call_count[0] += 1
                f_idx = int(frame.get("frame", 0)) if frame else 0
                npz_path = o / c["map_id"] / f"n{count}" / c["id"] / f"frame-{f_idx:04d}.npz"
                return _make_dummy_npz(npz_path, frame=f_idx, array_elements=count)

            # First run: case should be traced (2 frames * 3 arrays = 6 trace calls)
            res1 = trace.run_production_trace(
                manifest=manifest,
                cases=[case],
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_one,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(res1["status"], "complete")
            self.assertEqual(scene_build_count[0], 1)
            self.assertEqual(trace_call_count[0], 6)

            # Second run: verified case should be skipped (0 trace calls)
            # Remove DONE marker so the map is re-checked
            (out / "munich" / "DONE").unlink()
            res2 = trace.run_production_trace(
                manifest=manifest,
                cases=[case],
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_one,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(res2["status"], "complete")
            self.assertEqual(scene_build_count[0], 1)  # unchanged
            self.assertEqual(trace_call_count[0], 6)   # unchanged

            # Corrupt one npz file
            corrupt_path = out / "munich" / "n64" / "munich-v11-0000" / "frame-0000.npz"
            with corrupt_path.open("ab") as f:
                f.write(b"corrupt_extra_bytes")

            # Third run: hash mismatch triggers re-trace!
            (out / "munich" / "DONE").unlink()
            res3 = trace.run_production_trace(
                manifest=manifest,
                cases=[case],
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_one,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(res3["status"], "complete")
            self.assertEqual(scene_build_count[0], 2)  # re-built
            self.assertEqual(trace_call_count[0], 12)  # re-traced (+6 calls)

    def test_failed_skip_and_retry_failed(self):
        case = _make_dummy_case("munich-v11-0000", "munich", num_frames=2)
        manifest = {"_run_scope": "development_full", "cases": [case]}

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            trace_calls = [0]

            def fake_scene_builder(c, r, o):
                return {"mesh_bounds_local_m": [[-50, -50, 0], [50, 50, 30]], "origin_m": [0, 0, 0], "scene_sha256": "fake_scene_sha"}, np.zeros(3)

            def fake_trace_one(rt, dr, r, c, count, o, frame=None, gpu_id=0):
                trace_calls[0] += 1
                f_idx = int(frame.get("frame", 0)) if frame else 0
                npz_path = o / c["map_id"] / f"n{count}" / c["id"] / f"frame-{f_idx:04d}.npz"
                return _make_dummy_npz(npz_path, frame=f_idx, array_elements=count)

            # Pre-seed a FAILED file
            cases_dir = out / "munich" / "cases"
            cases_dir.mkdir(parents=True)
            failed_file = cases_dir / f"{case['id']}.FAILED.json"
            trace.atomic_write_json(failed_file, {
                "case_id": case["id"],
                "map_id": "munich",
                "stage": "trace",
                "error": "simulated previous failure",
                "time_utc": "2026-11-09T00:00:00Z",
            })

            # Without --retry-failed, the failed case is skipped
            res = trace.run_production_trace(
                manifest=manifest,
                cases=[case],
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                retry_failed=False,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_one,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(trace_calls[0], 0)
            self.assertTrue(failed_file.is_file())

            # With --retry-failed, .FAILED.json is renamed to .FAILED.prev.json and case is traced
            res2 = trace.run_production_trace(
                manifest=manifest,
                cases=[case],
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                retry_failed=True,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_one,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(trace_calls[0], 6)
            self.assertFalse(failed_file.is_file())
            prev_file = cases_dir / f"{case['id']}.FAILED.prev.json"
            self.assertTrue(prev_file.is_file())
            case_rec = cases_dir / f"{case['id']}.json"
            self.assertTrue(case_rec.is_file())

    def test_stop_rule_thresholds_120_cases(self):
        # 120 cases: 2 failures pass stop rule (2/120 <= 2%); 3 failures exceed stop rule (3/120 > 2%)
        cases = [_make_dummy_case(f"munich-v11-{i:04d}", "munich", num_frames=1) for i in range(120)]
        manifest = {"_run_scope": "development_full", "cases": cases}

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            def fake_scene_builder(c, r, o):
                return {"mesh_bounds_local_m": [[-50, -50, 0], [50, 50, 30]], "origin_m": [0, 0, 0], "scene_sha256": "fake_scene_sha"}, np.zeros(3)

            # Test 2 failures out of 120: should write DONE and trace_manifest.json with 2 failed_cases
            def fake_trace_2fail(rt, dr, r, c, count, o, frame=None, gpu_id=0):
                if c["id"] in ("munich-v11-0005", "munich-v11-0010"):
                    raise RuntimeError(f"simulated failure on {c['id']}")
                npz_path = o / c["map_id"] / f"n{count}" / c["id"] / "frame-0000.npz"
                return _make_dummy_npz(npz_path, frame=0, array_elements=count)

            res = trace.run_production_trace(
                manifest=manifest,
                cases=cases,
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_2fail,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(res["status"], "complete")
            self.assertTrue((out / "munich" / "DONE").is_file())
            self.assertFalse((out / "munich" / "FAILED").is_file())
            manifest_doc = json.loads((out / "munich" / "trace_manifest.json").read_text())
            self.assertEqual(manifest_doc["status"], "complete")
            self.assertEqual(manifest_doc["failed_cases"], ["munich-v11-0005", "munich-v11-0010"])
            self.assertEqual(len(manifest_doc["cases"]), 118)

        # Test 3 failures out of 120: should write FAILED and stop map
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            def fake_trace_3fail(rt, dr, r, c, count, o, frame=None, gpu_id=0):
                if c["id"] in ("munich-v11-0001", "munich-v11-0002", "munich-v11-0003"):
                    raise RuntimeError(f"simulated failure on {c['id']}")
                npz_path = o / c["map_id"] / f"n{count}" / c["id"] / "frame-0000.npz"
                return _make_dummy_npz(npz_path, frame=0, array_elements=count)

            res = trace.run_production_trace(
                manifest=manifest,
                cases=cases,
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_3fail,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(res["status"], "failed")
            self.assertTrue((out / "munich" / "FAILED").is_file())
            self.assertFalse((out / "munich" / "DONE").is_file())
            self.assertFalse((out / "munich" / "trace_manifest.json").is_file())
            failed_doc = json.loads((out / "munich" / "FAILED").read_text())
            self.assertEqual(failed_doc["failed_count"], 3)
            self.assertTrue(failed_doc["stop_rule_exceeded"])

    def test_stop_rule_thresholds_23_cases(self):
        # 23 cases (challenge scope): 1 failure exceeds 2% (1/23 = 4.35% > 2%)
        cases = [_make_dummy_case(f"munich-v11c-{i:04d}", "munich", num_frames=1) for i in range(23)]
        manifest = {"_run_scope": "development_challenge", "cases": cases}

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            def fake_scene_builder(c, r, o):
                return {"mesh_bounds_local_m": [[-50, -50, 0], [50, 50, 30]], "origin_m": [0, 0, 0], "scene_sha256": "fake_scene_sha"}, np.zeros(3)

            def fake_trace_1fail(rt, dr, r, c, count, o, frame=None, gpu_id=0):
                if c["id"] == "munich-v11c-0000":
                    raise RuntimeError("simulated failure on challenge case 0")
                npz_path = o / c["map_id"] / f"n{count}" / c["id"] / "frame-0000.npz"
                return _make_dummy_npz(npz_path, frame=0, array_elements=count)

            res = trace.run_production_trace(
                manifest=manifest,
                cases=cases,
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_1fail,
                manifest_sha256="fake_manifest_sha",
            )
            self.assertEqual(res["status"], "failed")
            self.assertTrue((out / "munich" / "FAILED").is_file())
            self.assertFalse((out / "munich" / "DONE").is_file())
            failed_doc = json.loads((out / "munich" / "FAILED").read_text())
            self.assertEqual(failed_doc["failed_count"], 1)
            self.assertTrue(failed_doc["stop_rule_exceeded"])

    def test_per_map_trace_manifest_fields(self):
        case = _make_dummy_case("munich-v11-0000", "munich", num_frames=1)
        manifest = {"_run_scope": "development_full", "cases": [case]}

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            root.mkdir()
            out.mkdir()

            def fake_scene_builder(c, r, o):
                return {
                    "mesh_bounds_local_m": [[-50, -50, 0], [50, 50, 30]],
                    "origin_m": [0, 0, 0],
                    "scene_sha256": "fake_scene_sha",
                    "mesh_audit": {"preflight_checks": {"all_passed": True}},
                }, np.zeros(3)

            def fake_trace_one(rt, dr, r, c, count, o, frame=None, gpu_id=0):
                npz_path = o / c["map_id"] / f"n{count}" / c["id"] / "frame-0000.npz"
                return _make_dummy_npz(npz_path, frame=0, array_elements=count)

            res = trace.run_production_trace(
                manifest=manifest,
                cases=[case],
                root=root,
                out=out,
                maps=["munich"],
                gpu_id=0,
                scene_builder_fn=fake_scene_builder,
                trace_one_fn=fake_trace_one,
                manifest_sha256="expected_manifest_sha256",
            )
            self.assertEqual(res["status"], "complete")
            manifest_path = out / "munich" / "trace_manifest.json"
            self.assertTrue(manifest_path.is_file())
            m = json.loads(manifest_path.read_text())
            self.assertEqual(m["status"], "complete")
            self.assertEqual(m["run_scope"], "development_full")
            self.assertEqual(m["case_manifest_sha256"], "expected_manifest_sha256")
            self.assertEqual(m["map_id"], "munich")
            self.assertEqual(m["failed_cases"], [])
            self.assertEqual(len(m["cases"]), 1)
            c_rec = m["cases"][0]
            self.assertEqual(c_rec["status"], "pass")
            self.assertEqual(c_rec["case_id"], "munich-v11-0000")
            self.assertEqual(c_rec["map_id"], "munich")
            self.assertEqual(c_rec["run_scope"], "development_full")
            self.assertEqual(c_rec["case_manifest_sha256"], "expected_manifest_sha256")
            self.assertEqual(c_rec["scene_sha256"], "fake_scene_sha")
            self.assertIn("geometry", c_rec)
            self.assertEqual(len(c_rec["records"]), 3)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/previews/bench-users-v1.1/case_manifest.json').exists(), 'needs the packaged data artifact')
    def test_pilot_path_is_unchanged(self):
        p1 = ROOT / "outputs/runs/bench_production_pilot_20261109/pilot_cases.json"
        m1, c1 = trace.verify_case_manifest(p1, production=False)
        self.assertEqual(m1["_run_scope"], "two_map_pilot")
        self.assertEqual([c["id"] for c in c1], ["01-munich-natural", "15-rotterdam-natural"])
        self.assertEqual([len(c["frames"]) for c in c1], [1, 1])

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            out = Path(tmpdir) / "out"
            case0 = c1[0]
            origin = [case0["bs_xyz"][0] - 10.0, case0["bs_xyz"][1] - 10.0, 0.0]
            mock_info = {
                "mesh_bounds_local_m": [[-500.0, -500.0, 0.1], [500.0, 500.0, 600.0]],
                "origin_m": origin,
                "scene_sha256": "fake_scene_sha",
            }
            with patch.object(trace, "_munich_scene", return_value=mock_info) as mock_m:
                info, origin = trace.build_scene(c1[0], root, out, production=False)
                # pilot directory is scenes/<map>/, NOT per-case
                self.assertEqual(mock_m.call_args[0][1], root / "scenes" / "munich")
                # pilot anchor users is frame 0 only
                self.assertEqual(mock_m.call_args[1]["anchor_users"], c1[0]["frames"][0]["users"])
