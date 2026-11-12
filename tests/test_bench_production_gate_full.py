import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec_pkg = importlib.util.spec_from_file_location("package_bench_production", ROOT / "scripts/package_bench_production.py")
pkg = importlib.util.module_from_spec(spec_pkg)
spec_pkg.loader.exec_module(pkg)

spec_gate = importlib.util.spec_from_file_location("verify_bench_production_gate", ROOT / "scripts/verify_bench_production_gate.py")
gate = importlib.util.module_from_spec(spec_gate)
spec_gate.loader.exec_module(gate)


def create_valid_production_fixture(root: Path, map_id="munich", case_id="munich-v11-0000",
                                     traj_idx=0, n_frames=16, arrays=(64, 128, 256),
                                     agreement_matches=100, agreement_total=100,
                                     bs_ris_class="los", preflight_passed=True):
    frames = [{"frame": i * 10, "time_s": i * 0.2, "users": [{"user_id": "u0", "xyz": [1, 2, 3]}]}
              for i in range(n_frames)]
    manifest_dict = {
        "scope": "development", "run_scope": "development_full",
        "is_pilot": False, "freeze_sha256": "fake",
        "cases": [{
            "id": case_id, "map_id": map_id, "trajectory_id": case_id,
            "frames": frames, "kind": "natural", "scenario": None, "trajectory_index": traj_idx
        }]
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict))
    m_hash = pkg.sha256(manifest_path)

    # Trace output
    trace_out = root / "trace_out"
    case_trace_dir = trace_out / map_id / "cases"
    case_trace_dir.mkdir(parents=True, exist_ok=True)
    trace_record = {
        "status": "pass",
        "case_id": case_id,
        "map_id": map_id,
        "run_scope": "development_full",
        "case_manifest_sha256": m_hash,
        "scene_sha256": "fake",
        "geometry": {
            "preflight_checks": {
                "all_passed": preflight_passed,
                "checks": {"terrain_bounds": preflight_passed}
            }
        },
        "records": []
    }
    (case_trace_dir / f"{case_id}.json").write_text(json.dumps(trace_record))

    # Channels HDF5
    channel_dir = root / "channels"
    channel_dir.mkdir(parents=True, exist_ok=True)
    channel_h5 = channel_dir / "channel_realizations.h5"
    channel_keys = [f"{map_id}/{n}/{case_id}/{i*10}" for n in arrays for i in range(n_frames)]

    rng = np.random.default_rng(123)
    m, k = 4, 1
    with h5py.File(channel_h5, "w") as f:
        for key in channel_keys:
            n = int(key.split("/")[1])
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

    # Metadata
    # Distribute matches/total across keys
    md = {}
    matches_per_key = [1] * agreement_matches + [0] * (agreement_total - agreement_matches)
    for idx, key in enumerate(channel_keys):
        md[key] = {
            "users": [{"user_id": "u0", "bs_user": "los", "ris_user": "los"}],
            "link_classes": {"bs_ris": bs_ris_class},
            "geometry_agreement": {
                "bs_user": [matches_per_key[idx % len(matches_per_key)]],
                "ris_user": [1]
            }
        }
    metadata_path = channel_dir / "metadata.json"
    metadata_path.write_text(json.dumps(md))

    # Channel run
    channel_run_path = root / "channel_run.json"
    channel_run_dict = {
        "run_scope": "development_full",
        "case_manifest_sha256": m_hash,
        "hdf5_sha256": pkg.sha256(channel_h5),
        "metadata_sha256": pkg.sha256(metadata_path),
        "maps": {map_id: {"cases_total": 1, "cases_included": 1, "failed_cases": []}},
        "arrays": {str(n): {"setup_s": 0.1, "peak_rss_bytes": 1000} for n in arrays}
    }
    channel_run_path.write_text(json.dumps(channel_run_dict))

    # Package
    package_dir = root / "package"
    pkg.package(manifest_path, channel_h5, metadata_path, package_dir,
                production=True, channel_run=channel_run_path,
                verify_freeze=False, partial_arrays=list(arrays), pairs_256=4096)

    return {
        "manifest": manifest_path,
        "trace_out": trace_out,
        "channel_run": channel_run_path,
        "metadata": metadata_path,
        "package": package_dir,
        "channel_h5": channel_h5,
    }


class TestBenchProductionGateFull(unittest.TestCase):
    def test_gate_passes_on_valid_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td))
            out = Path(td) / "gate.json"
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "pass")
            self.assertTrue(all(res["checks"].values()))
            self.assertEqual(len(res["failing_cases"]), 0)

    def test_gate_fails_on_missing_frame(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td))
            out = Path(td) / "gate.json"
            # Delete one group in the packaged channel realizations
            pkg_cf = fixture["package"] / "channels/channel_realizations.h5"
            with h5py.File(pkg_cf, "a") as f:
                del f["munich/64/munich-v11-0000/0"]
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["complete_hdf5_schema"])

    def test_gate_fails_on_nan(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td))
            out = Path(td) / "gate.json"
            pkg_cf = fixture["package"] / "channels/channel_realizations.h5"
            with h5py.File(pkg_cf, "a") as f:
                # Insert NaN in H_direct
                arr = f["munich/64/munich-v11-0000/0/H_direct"][()]
                arr[0, 0] = np.nan
                del f["munich/64/munich-v11-0000/0/H_direct"]
                f["munich/64/munich-v11-0000/0/H_direct"] = arr
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["finite"])

    def test_gate_fails_on_rate_error_1e_9(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td))
            out = Path(td) / "gate.json"
            pkg_cf = fixture["package"] / "channels/channel_realizations.h5"
            with h5py.File(pkg_cf, "a") as f:
                # Corrupt stored_rate by 1e-9
                f["munich/64/munich-v11-0000/0"].attrs["stored_rate_bps_hz"] += 1e-9
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["loader_dense_1e_10"])

    def test_gate_fails_on_preflight_false(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td), preflight_passed=False)
            out = Path(td) / "gate.json"
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["mesh_preflight_all_passed"])

    def test_gate_fails_on_bs_ris_nlos(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td), bs_ris_class="blocked")
            out = Path(td) / "gate.json"
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["bs_ris_link_los"])

    def test_gate_fails_on_los_agreement_74_percent(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = create_valid_production_fixture(Path(td))
            md = json.loads(fixture["metadata"].read_text())
            # Set exact 74% agreement: 74 ones and 26 zeros per key
            agreement_74 = [1] * 74 + [0] * 26
            for k, v in md.items():
                v["geometry_agreement"]["bs_user"] = agreement_74
                v["geometry_agreement"]["ris_user"] = agreement_74
            fixture["metadata"].write_text(json.dumps(md))
            (fixture["package"] / "channels/metadata.json").write_text(json.dumps(md))
            crun = json.loads(fixture["channel_run"].read_text())
            crun["metadata_sha256"] = pkg.sha256(fixture["metadata"])
            fixture["channel_run"].write_text(json.dumps(crun))

            out = Path(td) / "gate.json"
            res = gate.verify_production_gate(
                fixture["manifest"], fixture["trace_out"], fixture["channel_run"],
                fixture["metadata"], fixture["package"], out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["los_geometry_agreement"])

    def test_gate_fails_on_stop_rule_breach(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create a 23-case challenge fixture where 1 case failed -> 1/23 > 2% stop rule breached
            cases = [{"id": f"munich-v11c-{i:04d}", "map_id": "munich",
                      "frames": [{"frame": 0, "time_s": 0.0, "users": []}],
                      "kind": "challenge", "scenario": "shadow_edge", "trajectory_index": i}
                     for i in range(23)]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "scope": "development", "run_scope": "development_challenge", "subset": "challenge",
                "is_pilot": False, "freeze_sha256": "fake", "cases": cases
            }))
            m_hash = pkg.sha256(manifest_path)

            trace_dir = root / "trace_out"; trace_dir.mkdir()
            meta_path = root / "metadata.json"; meta_path.write_text("{}")
            pkg_dir = root / "package"; pkg_dir.mkdir()
            (pkg_dir / "DONE").write_text("complete\n")
            (pkg_dir / "provenance.json").write_text(json.dumps({
                "package_role": "production_challenge", "run_scope": "development_challenge",
                "manifest_sha256": m_hash
            }))
            (pkg_dir / "channels").mkdir()
            h5_file = pkg_dir / "channels/channel_realizations.h5"
            with h5py.File(h5_file, "w") as f: pass
            (pkg_dir / "channels/metadata.json").write_text("{}")
            (pkg_dir / "optimization").mkdir()
            for nm in ("single_flip_gains.h5", "pair_flip_gains.h5", "synergy.h5"):
                with h5py.File(pkg_dir / "optimization" / nm, "w") as f: pass
            (pkg_dir / "examples").mkdir()
            (pkg_dir / "examples/load_dataset.py").touch()
            (pkg_dir / "splits").mkdir()
            (pkg_dir / "splits/challenge.json").write_text(json.dumps({"evaluation_only_never_mixed": True}))

            crun_path = root / "channel_run.json"
            crun_path.write_text(json.dumps({
                "run_scope": "development_challenge",
                "case_manifest_sha256": m_hash,
                "metadata_sha256": pkg.sha256(meta_path),
                "hdf5_sha256": pkg.sha256(pkg_dir / "channels/channel_realizations.h5"),
                "maps": {"munich": {"cases_total": 23, "cases_included": 22, "failed_cases": ["munich-v11c-0000"]}}
            }))

            out = root / "gate.json"
            res = gate.verify_production_gate(
                manifest_path, trace_dir, crun_path,
                meta_path, pkg_dir, out, verify_freeze_check=False
            )
            self.assertEqual(res["status"], "fail")
            self.assertFalse(res["checks"]["stop_rule_not_exceeded"])


if __name__ == "__main__":
    unittest.main()
