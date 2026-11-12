import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import scripts.check_bench_production_progress as progress_script


class TestBenchProductionProgress(unittest.TestCase):
    def test_partial_layout_progress_reporting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = [
                {"id": "munich-v11-0000", "map_id": "munich", "trajectory_id": "munich-v11-0000",
                 "frames": [{"frame": 0, "time_s": 0.0, "users": []}]},
                {"id": "munich-v11-0001", "map_id": "munich", "trajectory_id": "munich-v11-0001",
                 "frames": [{"frame": 0, "time_s": 0.0, "users": []}]},
            ]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"cases": cases}))

            trace_out = root / "trace_out"
            (trace_out / "munich/cases").mkdir(parents=True)
            # munich-0000 trace done, munich-0001 trace failed
            (trace_out / "munich/cases/munich-v11-0000.json").write_text(json.dumps({"status": "pass"}))
            (trace_out / "munich/cases/munich-v11-0001.FAILED.json").write_text(json.dumps({"error": "test"}))
            (trace_out / "munich/DONE").write_text("done\n")

            channel_out = root / "channel_out"
            (channel_out / "munich/cases").mkdir(parents=True)
            (channel_out / "workers").mkdir(parents=True)
            # worker markers
            (channel_out / "workers/w0.DONE").touch()
            (channel_out / "workers/w1.RUNNING").touch()

            # Array shards for munich-0000: array 64 and 128 done, 256 not done
            c0_dir = channel_out / "munich/cases/munich-v11-0000"
            c0_dir.mkdir()
            (c0_dir / "array_64.h5").touch()
            (c0_dir / "array_64.json").touch()
            (c0_dir / "array_128.h5").touch()
            (c0_dir / "array_128.json").touch()

            # munich-0001: array 64 FAILED
            c1_dir = channel_out / "munich/cases/munich-v11-0001"
            c1_dir.mkdir()
            (c1_dir / "array_64.FAILED.json").touch()

            rep = progress_script.check_progress(manifest_path, trace_out, channel_out)

            self.assertEqual(rep["workers"]["done"], ["w0"])
            self.assertEqual(rep["workers"]["running"], ["w1"])

            m_rep = rep["maps"]["munich"]
            self.assertEqual(m_rep["total_cases"], 2)
            self.assertEqual(m_rep["trace_records_done"], 1)
            self.assertEqual(m_rep["trace_records_failed"], 1)
            self.assertTrue(m_rep["trace_markers"]["DONE"])
            self.assertFalse(m_rep["trace_markers"]["FAILED"])

            self.assertEqual(m_rep["channel_shards"]["64"]["done_count"], 1)
            self.assertEqual(m_rep["channel_shards"]["64"]["failed_count"], 1)
            self.assertEqual(m_rep["channel_shards"]["128"]["done_count"], 1)
            self.assertEqual(m_rep["channel_shards"]["256"]["done_count"], 0)

            self.assertEqual(m_rep["distinct_failed_cases"], ["munich-v11-0001"])
            self.assertTrue(m_rep["stop_rule_exceeded"])  # 1/2 > 2%

    def test_verify_finished_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case_id = "munich-v11-0000"
            cases = [{"id": case_id, "map_id": "munich", "trajectory_id": case_id,
                      "frames": [{"frame": 0, "time_s": 0.0, "users": []}]}]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"cases": cases}))

            trace_out = root / "trace_out"
            (trace_out / "munich/cases").mkdir(parents=True)
            (trace_out / f"munich/cases/{case_id}.json").write_text(json.dumps({
                "status": "pass",
                "geometry": {"preflight_checks": {"all_passed": True}}
            }))

            channel_out = root / "channel_out"
            c_dir = channel_out / f"munich/cases/{case_id}"
            c_dir.mkdir(parents=True)

            rng = np.random.default_rng(7)
            # Create all 3 shards with valid channels, sidecar, rate
            for n in (64, 128, 256):
                h5_path = c_dir / f"array_{n}.h5"
                m, k = 4, 1
                with h5py.File(h5_path, "w") as f:
                    g = f.require_group(f"munich/{n}/{case_id}/0")
                    g["Z_effective"] = (100 + 5j) * np.eye(n) + 0.01 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
                    g["C_in"] = (rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))) * 1e-4
                    g["C_out"] = (rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))) * 1e-4
                    g["H_direct"] = (rng.normal(size=(k, m)) + 1j * rng.normal(size=(k, m))) * 1e-3
                    g["loads"] = np.asarray([1 - 10.8j, 1 + 129.2j])
                    g["precoder"] = np.eye(m, k, dtype=np.complex128) * 0.5
                    g["stored_bits"] = np.zeros(n, dtype=np.int8)
                    g.attrs["noise"] = 1e-10
                    # Compute actual rate
                    ev = progress_script.HybridCoupledNetwork(
                        g["Z_effective"][()], g["C_out"][()], g["C_in"][()],
                        g["H_direct"][()], g["loads"][()], w=g["precoder"][()], noise=1e-10
                    )
                    rate = float(ev.score(ev.channel(g["stored_bits"][()]) @ ev.w))
                    g.attrs["stored_rate_bps_hz"] = rate

                json_path = c_dir / f"array_{n}.json"
                sidecar_data = {
                    "h5_sha256": progress_script.sha256(h5_path),
                    "records": {
                        f"munich/{n}/{case_id}/0": {
                            "link_classes": {"bs_ris": "los"}
                        }
                    }
                }
                json_path.write_text(json.dumps(sidecar_data))

            # Verify finished succeeds
            rep = progress_script.check_progress(manifest_path, trace_out, channel_out, verify_finished=True)
            self.assertIn("verified_cases", rep)
            self.assertEqual(rep["verified_cases"][case_id]["status"], "pass")
            self.assertEqual(rep["verified_cases"][case_id]["reasons"], [])

            # Corrupt sidecar sha256 and confirm verify_finished reports failure
            (c_dir / "array_64.json").write_text(json.dumps({
                "h5_sha256": "wrong_sha256",
                "records": {f"munich/64/{case_id}/0": {"link_classes": {"bs_ris": "los"}}}
            }))
            rep_corrupt = progress_script.check_progress(manifest_path, trace_out, channel_out, verify_finished=True)
            self.assertEqual(rep_corrupt["verified_cases"][case_id]["status"], "fail")
            self.assertTrue(any("mismatch" in r for r in rep_corrupt["verified_cases"][case_id]["reasons"]))

            # Test CLI stdout output
            buf = io.StringIO()
            with redirect_stdout(buf):
                progress_script.main([
                    "--manifest", str(manifest_path),
                    "--trace-out", str(trace_out),
                    "--channel-out", str(channel_out),
                    "--verify-finished"
                ])
            cli_json = json.loads(buf.getvalue())
            self.assertIn("verified_cases", cli_json)


if __name__ == "__main__":
    unittest.main()
