import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_bench_production", ROOT / "scripts/package_bench_production.py")
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)


def make_dummy_channel(h5_path, keys, n=64, m=4, k=2):
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "w") as f:
        for key in keys:
            g = f.require_group(key)
            g["Z_effective"] = (100 + 5j) * np.eye(n) + 0.01 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
            g["C_in"] = (rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))) * 1e-4
            g["C_out"] = (rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))) * 1e-4
            g["H_direct"] = (rng.normal(size=(k, m)) + 1j * rng.normal(size=(k, m))) * 1e-3
            g["loads"] = np.asarray([1 - 10.8j, 1 + 129.2j])
            g["precoder"] = np.eye(m, k, dtype=np.complex128) * 0.5
            g["stored_bits"] = np.zeros(n, dtype=np.int8)
            g.attrs["noise"] = 1e-10
            ev = pkg._network(g, {})
            rate = float(ev.score(ev.channel(g["stored_bits"][()]) @ ev.w))
            g.attrs["stored_rate_bps_hz"] = rate
            g.attrs["mesh_segments"] = 81 if n == 256 else 161


def make_metadata(keys):
    md = {}
    for key in keys:
        md[key] = {
            "users": [
                {"user_id": "u0", "bs_user": "los", "ris_user": "los"},
                {"user_id": "u1", "bs_user": "los", "ris_user": "los"}
            ],
            "link_classes": {"bs_ris": "los"},
            "geometry_agreement": {"bs_user": [1, 1], "ris_user": [1, 1]}
        }
    return md


class TestBenchProductionPackageFull(unittest.TestCase):
    def test_labelled_frame_positions_and_unlabelled(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            case_id = "munich-v11-0000"
            frames = [{"frame": i * 10, "time_s": i * 0.2, "users": [{"user_id": "u0", "xyz": [1, 2, 3]}]}
                      for i in range(16)]
            manifest_dict = {
                "scope": "development",
                "run_scope": "development_full",
                "is_pilot": False,
                "freeze_sha256": "fake",
                "cases": [{
                    "id": case_id, "map_id": "munich", "trajectory_id": case_id,
                    "frames": frames, "kind": "natural", "scenario": None
                }]
            }
            manifest_path = td / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_dict))
            m_hash = pkg.sha256(manifest_path)

            channel_keys = [f"munich/64/{case_id}/{i*10}" for i in range(16)]
            channel_h5 = td / "channel_realizations.h5"
            make_dummy_channel(channel_h5, channel_keys, n=64)

            metadata_path = td / "metadata.json"
            metadata_path.write_text(json.dumps(make_metadata(channel_keys)))

            channel_run_path = td / "channel_run.json"
            channel_run_path.write_text(json.dumps({
                "run_scope": "development_full",
                "case_manifest_sha256": m_hash,
                "maps": {"munich": {"cases_total": 1, "cases_included": 1, "failed_cases": []}},
                "arrays": {"64": {"setup_s": 0.1, "peak_rss_bytes": 1000}}
            }))

            output_dir = td / "package"
            pkg.package(manifest_path, channel_h5, metadata_path, output_dir,
                        production=True, channel_run=channel_run_path,
                        verify_freeze=False, partial_arrays=[64])

            # Verify labels exist only for positions [0, 5, 10, 15]
            labelled_frame_numbers = [frames[p]["frame"] for p in (0, 5, 10, 15)]
            with h5py.File(output_dir / "optimization/single_flip_gains.h5", "r") as sf:
                for p, fr in enumerate(frames):
                    f_key = f"munich/64/{case_id}/{fr['frame']}"
                    if p in (0, 5, 10, 15):
                        self.assertIn(f_key, sf)
                    else:
                        self.assertNotIn(f_key, sf)

            # Verify trajectories.parquet has labelled column correctly flagged
            df = pd.read_parquet(output_dir / "trajectories.parquet")
            self.assertIn("labelled", df.columns)
            self.assertIn("kind", df.columns)
            self.assertIn("scenario", df.columns)
            self.assertIn("trajectory_index", df.columns)
            self.assertIn("split", df.columns)

            for p, fr in enumerate(frames):
                frame_rows = df[df["frame"] == fr["frame"]]
                expected_labelled = (p in (0, 5, 10, 15))
                self.assertTrue((frame_rows["labelled"] == expected_labelled).all())

    def test_audit_rule_256_elements(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Case 0 has i = 0 (i % 10 == 0)
            # Case 1 has i = 1 (i % 10 != 0)
            cases = []
            frames = [{"frame": i * 10, "time_s": i * 0.2, "users": [{"user_id": "u0", "xyz": [1, 2, 3]}]}
                      for i in range(16)]
            for i in (0, 1):
                cid = f"munich-v11-{i:04d}"
                cases.append({
                    "id": cid, "map_id": "munich", "trajectory_id": cid,
                    "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": i
                })
            manifest_dict = {
                "scope": "development", "run_scope": "development_full",
                "is_pilot": False, "freeze_sha256": "fake", "cases": cases
            }
            manifest_path = td / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_dict))
            m_hash = pkg.sha256(manifest_path)

            channel_keys = [f"munich/256/{c['id']}/{fr['frame']}" for c in cases for fr in frames]
            channel_h5 = td / "channel_realizations.h5"
            make_dummy_channel(channel_h5, channel_keys, n=256)

            metadata_path = td / "metadata.json"
            metadata_path.write_text(json.dumps(make_metadata(channel_keys)))

            channel_run_path = td / "channel_run.json"
            channel_run_path.write_text(json.dumps({
                "run_scope": "development_full",
                "case_manifest_sha256": m_hash,
                "maps": {"munich": {"cases_total": 2, "cases_included": 2, "failed_cases": []}},
                "arrays": {"256": {"setup_s": 0.1, "peak_rss_bytes": 1000}}
            }))

            output_dir = td / "package"
            pkg.package(manifest_path, channel_h5, metadata_path, output_dir,
                        production=True, channel_run=channel_run_path,
                        verify_freeze=False, partial_arrays=[256], pairs_256=4096)

            with h5py.File(output_dir / "optimization/pair_flip_gains.h5", "r") as pf:
                # Case 0, first labelled frame (pos 0, frame 0), all_off parent -> full 32640
                c0_f0_alloff = pf[f"munich/256/munich-v11-0000/0/all_off"]
                self.assertEqual(len(c0_f0_alloff["pair_indices"]), 32640)
                self.assertEqual(c0_f0_alloff.attrs["pair_coverage"], "full")

                # Case 0, first labelled frame (pos 0, frame 0), single_greedy parent -> 4096 sampled
                c0_f0_greedy = pf[f"munich/256/munich-v11-0000/0/single_greedy"]
                self.assertEqual(len(c0_f0_greedy["pair_indices"]), 4096)
                self.assertEqual(c0_f0_greedy.attrs["pair_coverage"], "sampled")

                # Case 0, second labelled frame (pos 5, frame 50), all_off parent -> 4096 sampled
                c0_f5_alloff = pf[f"munich/256/munich-v11-0000/50/all_off"]
                self.assertEqual(len(c0_f5_alloff["pair_indices"]), 4096)
                self.assertEqual(c0_f5_alloff.attrs["pair_coverage"], "sampled")

                # Case 1 (i=1), first labelled frame (pos 0, frame 0), all_off parent -> 4096 sampled
                c1_f0_alloff = pf[f"munich/256/munich-v11-0001/0/all_off"]
                self.assertEqual(len(c1_f0_alloff["pair_indices"]), 4096)
                self.assertEqual(c1_f0_alloff.attrs["pair_coverage"], "sampled")

    def test_failed_cases_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = [{"frame": 0, "time_s": 0.0, "users": [{"user_id": "u0", "xyz": [1, 2, 3]}]}]
            cases = [
                {"id": "munich-v11-0000", "map_id": "munich", "trajectory_id": "munich-v11-0000",
                 "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": 0},
                {"id": "munich-v11-0001", "map_id": "munich", "trajectory_id": "munich-v11-0001",
                 "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": 1}
            ]
            manifest_path = td / "manifest.json"
            manifest_path.write_text(json.dumps({
                "scope": "development", "run_scope": "development_full",
                "is_pilot": False, "freeze_sha256": "fake", "cases": cases
            }))
            m_hash = pkg.sha256(manifest_path)

            channel_keys = ["munich/64/munich-v11-0000/0"]
            channel_h5 = td / "channel_realizations.h5"
            make_dummy_channel(channel_h5, channel_keys, n=64)

            metadata_path = td / "metadata.json"
            metadata_path.write_text(json.dumps(make_metadata(channel_keys)))

            channel_run_path = td / "channel_run.json"
            channel_run_path.write_text(json.dumps({
                "run_scope": "development_full",
                "case_manifest_sha256": m_hash,
                "maps": {"munich": {"cases_total": 2, "cases_included": 1, "failed_cases": ["munich-v11-0001"]}},
                "arrays": {"64": {"setup_s": 0.1, "peak_rss_bytes": 1000}}
            }))

            output_dir = td / "package"
            prov = pkg.package(manifest_path, channel_h5, metadata_path, output_dir,
                               production=True, channel_run=channel_run_path,
                               verify_freeze=False, partial_arrays=[64])

            # munich-v11-0001 excluded
            df = pd.read_parquet(output_dir / "trajectories.parquet")
            self.assertIn("munich-v11-0000", set(df["case_id"]))
            self.assertNotIn("munich-v11-0001", set(df["case_id"]))
            self.assertIn("munich-v11-0001", prov["failed_cases"])

    def test_natural_vs_challenge_splits(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = [{"frame": 0, "time_s": 0.0, "users": [{"user_id": "u0", "xyz": [1, 2, 3]}]}]

            # 1. Natural manifest
            nat_cases = [
                {"id": "munich-v11-0000", "map_id": "munich", "trajectory_id": "munich-v11-0000",
                 "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": 0},
                {"id": "munich-v11-0085", "map_id": "munich", "trajectory_id": "munich-v11-0085",
                 "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": 85},
                {"id": "munich-v11-0105", "map_id": "munich", "trajectory_id": "munich-v11-0105",
                 "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": 105},
            ]
            nat_manifest = td / "nat_manifest.json"
            nat_manifest.write_text(json.dumps({
                "scope": "development", "run_scope": "development_full",
                "is_pilot": False, "freeze_sha256": "fake", "cases": nat_cases
            }))
            nat_keys = [f"munich/64/{c['id']}/0" for c in nat_cases]
            nat_channel = td / "nat_channel.h5"
            make_dummy_channel(nat_channel, nat_keys, n=64)
            nat_meta_dir = td / "nat"; nat_meta_dir.mkdir()
            nat_meta = nat_meta_dir / "metadata.json"
            nat_meta.write_text(json.dumps(make_metadata(nat_keys)))
            nat_crun = td / "nat_crun.json"
            nat_crun.write_text(json.dumps({
                "run_scope": "development_full", "case_manifest_sha256": pkg.sha256(nat_manifest),
                "maps": {"munich": {"cases_total": 3, "cases_included": 3, "failed_cases": []}},
                "arrays": {"64": {"setup_s": 0.1, "peak_rss_bytes": 1000}}
            }))

            nat_out = td / "nat_package"
            pkg.package(nat_manifest, nat_channel, nat_meta, nat_out,
                        production=True, channel_run=nat_crun,
                        verify_freeze=False, partial_arrays=[64])

            # Check natural splits
            self.assertTrue((nat_out / "splits/within_map_trajectory.json").is_file())
            self.assertTrue((nat_out / "splits/leave_environment_out.json").is_file())
            self.assertFalse((nat_out / "splits/challenge.json").exists())

            df_nat = pd.read_parquet(nat_out / "trajectories.parquet")
            self.assertEqual(set(df_nat[df_nat["trajectory_index"] == 0]["split"]), {"train"})
            self.assertEqual(set(df_nat[df_nat["trajectory_index"] == 85]["split"]), {"validation"})
            self.assertEqual(set(df_nat[df_nat["trajectory_index"] == 105]["split"]), {"test"})
            self.assertTrue((df_nat["kind"] == "natural").all())

            # 2. Challenge manifest
            ch_cases = [
                {"id": "munich-v11c-0000", "map_id": "munich", "trajectory_id": "munich-v11c-0000",
                 "frames": frames, "kind": "challenge", "scenario": "shadow_edge", "trajectory_index": 0}
            ]
            ch_manifest = td / "ch_manifest.json"
            ch_manifest.write_text(json.dumps({
                "scope": "development", "run_scope": "development_challenge", "subset": "challenge",
                "is_pilot": False, "freeze_sha256": "fake", "cases": ch_cases
            }))
            ch_keys = [f"munich/64/{c['id']}/0" for c in ch_cases]
            ch_channel = td / "ch_channel.h5"
            make_dummy_channel(ch_channel, ch_keys, n=64)
            ch_meta_dir = td / "ch"; ch_meta_dir.mkdir()
            ch_meta = ch_meta_dir / "metadata.json"
            ch_meta.write_text(json.dumps(make_metadata(ch_keys)))
            ch_crun = td / "ch_crun.json"
            ch_crun.write_text(json.dumps({
                "run_scope": "development_challenge", "case_manifest_sha256": pkg.sha256(ch_manifest),
                "maps": {"munich": {"cases_total": 1, "cases_included": 1, "failed_cases": []}},
                "arrays": {"64": {"setup_s": 0.1, "peak_rss_bytes": 1000}}
            }))

            ch_out = td / "ch_package"
            pkg.package(ch_manifest, ch_channel, ch_meta, ch_out,
                        production=True, channel_run=ch_crun,
                        verify_freeze=False, partial_arrays=[64])

            # Check challenge splits
            self.assertTrue((ch_out / "splits/challenge.json").is_file())
            self.assertFalse((ch_out / "splits/within_map_trajectory.json").exists())
            self.assertFalse((ch_out / "splits/leave_environment_out.json").exists())

            ch_split_data = json.loads((ch_out / "splits/challenge.json").read_text())
            self.assertTrue(ch_split_data.get("evaluation_only_never_mixed"))

            df_ch = pd.read_parquet(ch_out / "trajectories.parquet")
            self.assertEqual(set(df_ch["split"]), {"challenge"})
            self.assertTrue((df_ch["kind"] == "challenge").all())
            self.assertEqual(set(df_ch["scenario"]), {"shadow_edge"})


if __name__ == "__main__":
    unittest.main()
