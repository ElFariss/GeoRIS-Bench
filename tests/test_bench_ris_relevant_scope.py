"""RIS-relevant pilot scope (docs/bench-ris-relevant-scope-v1-20261112.md)."""
import json
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from atomwise_ris.bench import production_pilot, production_run, ris_relevant as rr, ris_relevant_scope as scope
from atomwise_ris.bench import users as u
from atomwise_ris.bench.production_freeze import ROOT, sha256, verify_freeze

MANIFEST = ROOT / "outputs/runs/bench_ris_relevant_pilot_20261112/cases.json"


class SeedsAndSplit(unittest.TestCase):
    def test_seed_word_disjoint(self):
        self.assertNotIn(rr.SEED_WORD, {1, 2, 3, 4, 5, 20261109, 202611092})
        a = rr.trajectory_rng(1, 0).integers(1 << 62)
        b = np.random.default_rng(np.random.SeedSequence([u.SEED, 20261109, 1, 0])).integers(1 << 62)
        self.assertNotEqual(a, b)

    def test_split_by_index(self):
        parts = Counter(rr.split_of(i) for i in range(rr.CASE_COUNT))
        self.assertEqual(parts, {"train": 34, "validation": 7, "test": 7})
        self.assertEqual(rr.split_of(40), "validation")
        self.assertEqual(rr.split_of(41), "test")
        with self.assertRaises(ValueError):
            rr.split_of(48)

    def test_selected_frames_match_production(self):
        f = rr.selected_frames(501)
        self.assertEqual(len(f), 16)
        self.assertEqual((f[0], f[-1]), (0, 500))


class ServedRule(unittest.TestCase):
    def test_eligibility_requires_every_frame(self):
        bs = np.array([["no_los", "los", "no_los_reflection_candidate"],
                       ["no_los", "no_los", "no_los"]])
        ris = np.array([[True, True, True], [True, True, False]])
        self.assertEqual(rr.eligible_users(bs, ris).tolist(), [True, False, False])

    def test_schedule_only_eligible_and_size(self):
        elig = np.array([True, False, True, True, False, True, True])
        for s in range(200):
            out = rr.schedule_ris_relevant(elig, np.random.default_rng(s))
            self.assertTrue(2 <= out.sum() <= 4)
            self.assertFalse((out & ~elig).any())

    def test_schedule_refuses_fewer_than_two(self):
        counts = Counter()
        self.assertIsNone(rr.schedule_ris_relevant(np.array([True, False, False]), np.random.default_rng(0), counts))
        self.assertEqual(counts["rr_crowd_too_few_eligible"], 1)


class Registry(unittest.TestCase):
    def test_existing_scopes_untouched(self):
        self.assertEqual(production_run.PRODUCTION_SCOPES, ("development_full", "development_challenge"))
        self.assertEqual(production_pilot.PRODUCTION_SCOPES, ("development_full", "development_challenge"))
        self.assertEqual(len(production_pilot.PILOTS), 4)
        self.assertEqual(scope.pipeline_scopes(), ("development_full", "development_challenge", "ris_relevant_pilot"))

    def test_install_is_module_local(self):
        mod = types.SimpleNamespace(registered_manifest=production_pilot.registered_manifest,
                                    PRODUCTION_SCOPES=production_run.PRODUCTION_SCOPES)
        scope.install(mod)
        self.assertIs(mod.registered_manifest, scope.registered_manifest)
        self.assertIn("ris_relevant_pilot", mod.PRODUCTION_SCOPES)
        self.assertEqual(production_run.PRODUCTION_SCOPES, ("development_full", "development_challenge"))

    def test_unknown_hash_delegates_to_production_registry(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps({"scope": "development", "cases": []}))
            with self.assertRaisesRegex(ValueError, "approved registration"):
                scope.registered_manifest(p, "x")

    def test_identity_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            cases = [{"id": rr.case_id("rotterdam", i), "map_id": "rotterdam", "kind": rr.SCOPE_NAME,
                      "split": rr.split_of(i), "trajectory_index": i, "frames": [{}] * 16} for i in range(48)]
            m = {"scope": "development", "subset": rr.SCOPE_NAME, "is_pilot": False, "freeze_sha256": "f",
                 "seed_word": rr.SEED_WORD, "cases": cases}
            p.write_text(json.dumps(m))
            got, s, _ = scope.registered_manifest(p, "f", _hash=sha256(p), _failed=())
            self.assertEqual(s, "ris_relevant_pilot")
            cases[3]["split"] = "test"
            p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                scope.registered_manifest(p, "f", _hash=sha256(p), _failed=())

    @unittest.skipUnless(MANIFEST.is_file(), "pilot manifest not generated here")
    def test_registered_pilot_manifest(self):
        m, s, digest = scope.registered_manifest(MANIFEST, verify_freeze())
        self.assertEqual((s, digest), ("ris_relevant_pilot", scope.RIS_RELEVANT_PILOT_HASH))
        for c in m["cases"]:
            d = float(np.linalg.norm(np.subtract(c["bs_xyz"], c["ris_xyz"])))
            self.assertTrue(5. <= d <= 20.)
            self.assertTrue(all(2 <= len(f["users"]) <= 4 for f in c["frames"]))
            self.assertTrue(all(us["bs_link"] != "los" and us["ris_los"] for f in c["frames"] for us in f["users"]))


class GeneratorWrapper(unittest.TestCase):
    def test_only_declared_overrides(self):
        own = {k for k, v in vars(rr.RISRelevantGenerator).items() if callable(v)}
        self.assertEqual(own, {"__init__", "plan_sites", "choose_site", "anchor"})
        self.assertTrue(issubclass(rr.RISRelevantGenerator, u.Generator))

    def test_freeze_lock_intact(self):
        self.assertEqual(verify_freeze(), "e152e78cb7b66204ea4080339c29a80a53c94685f8c3743b4eddd0cbab6e0ecd")


if __name__ == "__main__":
    unittest.main()
