import unittest

import numpy as np

from scripts.run_bench_zris_extract import lattice
from scripts.compare_bench_zris_mesh import validate_mesh_pair


class BenchZrisTests(unittest.TestCase):
    def test_registered_lattices(self):
        for n, shape in ((64, (8, 8)), (128, (8, 16)), (256, (16, 16))):
            centers, rows, cols = lattice(n)
            self.assertEqual((rows, cols), shape)
            self.assertEqual(centers.shape, (n, 3))
            self.assertAlmostEqual(centers[:, 1].mean(), 0.0)
            self.assertAlmostEqual(centers[:, 2].mean(), 1.0)
            self.assertTrue(np.all(np.diff(np.unique(centers[:, 1])) > 0))

    def test_invalid_array_size_rejected(self):
        with self.assertRaises(ValueError):
            lattice(65)

    def test_mesh_pair_accepts_same_array_at_81_and_161_segments(self):
        validate_mesh_pair(64, 81, 64, 161)
        with self.assertRaises(ValueError):
            validate_mesh_pair(64, 81, 128, 161)

    @unittest.skipUnless(__import__('pathlib').Path('bench/CoupledRIS-Bench/physics/array_256/mesh_validation.json').exists(), 'needs the packaged data artifact')
    def test_packaged_array_256_coarse_only_validation_metadata(self):
        import json
        from pathlib import Path
        from scripts.validate_bench_zris_bundle import digest

        root = Path("bench/CoupledRIS-Bench/physics")
        mv256 = json.loads((root / "array_256" / "mesh_validation.json").read_text())
        nv256 = json.loads((root / "array_256" / "numerical_validation.json").read_text())
        self.assertEqual(mv256["status"], "coarse_only_161_not_extracted_memory_cap")
        self.assertEqual(mv256["elements"], 256)
        self.assertEqual(mv256["coarse_segments"], 81)
        self.assertIsNone(mv256["fine_segments"])
        self.assertFalse(mv256["fine_segments_extracted"])
        for ref_n in ("64", "128"):
            ref_mv = json.loads((root / f"array_{ref_n}" / "mesh_validation.json").read_text())
            self.assertAlmostEqual(
                mv256["supporting_mesh_delta_trend"][ref_n]["relative_frobenius_difference"],
                ref_mv["relative_frobenius_difference"],
            )
        self.assertEqual(nv256["status"], "pass")
        self.assertEqual(nv256["elements"], 256)
        self.assertEqual(nv256["matrices"]["81"]["hdf5_sha256"], digest(root / "array_256" / "81" / "impedance.h5"))
        self.assertEqual(nv256["combined_sha256"], digest(root / "array_256" / "impedance.h5"))
        self.assertEqual(nv256["mesh_validation_sha256"], digest(root / "array_256" / "mesh_validation.json"))


if __name__ == "__main__":
    unittest.main()
