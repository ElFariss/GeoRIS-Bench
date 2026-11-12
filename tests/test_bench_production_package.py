import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(p) is None for p in ("h5py", "pandas", "pyarrow")):
    raise unittest.SkipTest("requires the optional production HDF5/Parquet environment")

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_bench_production", ROOT / "scripts/package_bench_production.py")
pkg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pkg)
loader_spec = importlib.util.spec_from_file_location("load_dataset", ROOT / "bench/CoupledRIS-Bench/examples/load_dataset.py")
loader = importlib.util.module_from_spec(loader_spec); loader_spec.loader.exec_module(loader)


class ProductionPackageTests(unittest.TestCase):
    def test_sampled_pair_mapping_unique_and_in_range(self):
        p = pkg.deterministic_pairs(256, 4096, 1729)
        self.assertEqual(p.shape, (4096, 2))
        self.assertTrue(np.all(p[:, 0] < p[:, 1]))
        self.assertEqual(len({tuple(x) for x in p}), len(p))
        self.assertGreaterEqual(int(p.min()), 0); self.assertLess(int(p.max()), 256)
        full = pkg.deterministic_pairs(64, 999999, 2)
        self.assertEqual(full.shape, (2016, 2))
        self.assertEqual(len({tuple(x) for x in full}), 2016)

    def test_label_identity_and_package_loader_roundtrip(self):
        rng = np.random.default_rng(8)
        n, m, k = 64, 4, 2
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            channels = root / "channel_realizations.h5"
            key = "munich/64/case-1/0"
            with h5py.File(channels, "w") as f:
                g = f.require_group(key)
                g["Z_effective"] = (100 + 5j) * np.eye(n) + 0.01 * (rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))
                g["C_in"] = (rng.normal(size=(n,m)) + 1j*rng.normal(size=(n,m))) * 1e-4
                g["C_out"] = (rng.normal(size=(k,n)) + 1j*rng.normal(size=(k,n))) * 1e-4
                g["H_direct"] = (rng.normal(size=(k,m)) + 1j*rng.normal(size=(k,m))) * 1e-3
                g["loads"] = np.asarray([1-10.8j, 1+129.2j])
                g["precoder"] = np.eye(m, k, dtype=np.complex128) * 0.5
                g.attrs["noise"] = 1e-10
            manifest = root / "pilot_cases.json"
            manifest.write_text(json.dumps({"cases":[{"id":"case-1","map_id":"munich","array":64,
                "frames":[{"frame":0,"time_s":0,"users":[{"user_id":"u0","xyz":[1,2,1.5]},
                    {"user_id":"u1","xyz":[2,3,1.5]}]}]}]}))
            metadata = root / "metadata.json"; metadata.write_text(json.dumps({key:{
                "users":[{"user_id":"u0","bs_user":"los","ris_user":"reflection_only"},
                    {"user_id":"u1","bs_user":"blocked","ris_user":"diffraction_only"}],
                "link_classes":{"bs_ris":"los"}}}))
            out = root / "package"
            pkg.package(manifest, channels, metadata, out, verify_freeze=False, partial_arrays=[64])
            verdict = loader.verify_label_rate(channels, key, out / "optimization")
            self.assertLessEqual(verdict["absolute_error"], 1e-10)
            with h5py.File(out / "optimization/pair_flip_gains.h5", "r") as f:
                pg = f[f"{key}/all_off"]
                pair = pg["pair_flip_gains"][()]
                pairs = pg["pair_indices"][()]
                self.assertEqual(pg.attrs["pair_coverage"], "full")
            with h5py.File(out / "optimization/single_flip_gains.h5", "r") as f:
                single = f[f"{key}/all_off/single_flip_gains"][()]
            with h5py.File(out / "optimization/synergy.h5", "r") as f:
                synergy = f[f"{key}/all_off/synergy"][()]
            np.testing.assert_allclose(pair-single[pairs[:,0]]-single[pairs[:,1]], synergy, rtol=0, atol=1e-12)
            self.assertTrue((out / "DONE").is_file())
            self.assertTrue((out / "trajectories.parquet").is_file())
            self.assertEqual(pkg.sha256(root / "metadata.json"), pkg.sha256(out / "channels/metadata.json"))
            self.assertEqual(pkg.sha256(ROOT / "bench/CoupledRIS-Bench/examples/load_dataset.py"),
                             pkg.sha256(out / "examples/load_dataset.py"))
            self.assertEqual(pkg.sha256(channels), pkg.sha256(out / "channels/channel_realizations.h5"))
            self.assertTrue((out / "splits/leave_map_out_development.json").is_file())
            import pandas as pd
            rows = pd.read_parquet(out / "trajectories.parquet")
            self.assertEqual(list(rows.user_id), ["u0", "u1"])
            self.assertEqual(list(rows.array_n), [64, 64])
            self.assertEqual(set(rows.split), {"development"})
            self.assertEqual(rows.iloc[0].ris_link, "reflection_only")
            self.assertEqual(rows.iloc[1].ris_link, "diffraction_only")
            self.assertEqual(rows.iloc[0].bs_ris_link, "los")
            self.assertEqual(json.loads((out / "provenance.json").read_text())["package_role"], "partial_diagnostic_only")


if __name__ == "__main__": unittest.main()
