"""RIS-relevant scope v2 (docs/bench-ris-relevant-scope-v2-20261113.md)."""
import types
import unittest

import numpy as np

from atomwise_ris.bench import production_run, ris_relevant_v2 as rr, ris_relevant_scope_v2 as scope
from atomwise_ris.bench.production_freeze import ROOT, verify_freeze

MANIFEST = ROOT / "outputs/runs/bench_ris_relevant_v2_pilot_20261113/cases.json"


class V2(unittest.TestCase):
    def test_seed_and_identity(self):
        self.assertNotIn(rr.SEED_WORD, {1, 2, 3, 4, 5, 20261109, 202611092, 20261112})
        self.assertEqual(rr.case_id("rotterdam", 7), "rotterdam-rr2-0007")
        self.assertEqual(rr.RADIUS_LADDER_M, (40., 50., 60.))

    def test_scopes(self):
        self.assertEqual(production_run.PRODUCTION_SCOPES, ("development_full", "development_challenge"))
        self.assertEqual(scope.pipeline_scopes()[-2:], ("ris_relevant_pilot", "ris_relevant_v2_pilot"))
        mod = types.SimpleNamespace(registered_manifest=None, PRODUCTION_SCOPES=())
        scope.install(mod)
        self.assertIs(mod.registered_manifest, scope.registered_manifest)

    def test_radius_term(self):
        class G:  # minimal stand-in: everything LOS-eligible, distance decides
            radius_m = 40.
            class city:
                @staticmethod
                def los_batch(src, pts):
                    src = np.asarray(src)
                    return (np.zeros(len(pts), bool) if src[2] > 50 else np.ones(len(pts), bool)), None
        site = {"bs_xyz": np.array([0., 0., 60.]), "ris_xyz": np.array([0., 0., 10.])}
        xyz = np.zeros((3, 2, 3)); xyz[:, 0] = [30., 0., 1.5]; xyz[:, 1] = [45., 0., 1.5]
        crowd = types.SimpleNamespace(xyz=xyz)
        self.assertEqual(rr.quick_eligibility(G, site, crowd, np.arange(3)).tolist(), [True, False])

    @unittest.skipUnless(MANIFEST.is_file(), "v2 manifest not generated here")
    def test_registered_manifest(self):
        m, s, d = scope.registered_manifest(MANIFEST, verify_freeze())
        self.assertEqual((s, d, m["radius_m"]), ("ris_relevant_v2_pilot", scope.RIS_RELEVANT_V2_HASH, 40.0))
        for c in m["cases"]:
            ris = np.asarray(c["ris_xyz"])
            self.assertTrue(all(np.linalg.norm(np.asarray(u["xyz"]) - ris) <= 40. for f in c["frames"] for u in f["users"]))


if __name__ == "__main__":
    unittest.main()
