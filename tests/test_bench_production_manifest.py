"""Unit tests for production pilot and full manifest registrations and validation."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from atomwise_ris.bench.production_freeze import ROOT, sha256, verify_freeze
from atomwise_ris.bench.production_pilot import (
    CHALLENGE_CASES,
    CHALLENGE_HASH,
    FULL_CASES,
    FULL_HASH,
    PILOT1_CASES,
    PILOT1_HASH,
    PILOT2_CASES,
    PILOT2_HASH,
    PILOTS,
    PRODUCTION_SCOPES,
    registered_manifest,
    registered_scopes,
)
from build_bench_production_cases import CHALLENGE_ORDER, challenge_on_map, choose_challenge


class FakeCrowd:
    def __init__(self, size=10):
        self.size = size
        self.ground = np.zeros((10, size))


class FakeGenerator:
    def __init__(self, map_id="munich", fail_categories=()):
        self.map_id = map_id
        self.fail_categories = set(fail_categories)
        self.calls = []

    def shadow(self, t, rng, counts, slope_required=False):
        self.calls.append(("shadow", slope_required))
        cat = "shadow_edge_sloped" if slope_required else "shadow_edge"
        if cat in self.fail_categories:
            return None
        return ({"bs_xyz": [0, 0, 0], "ris_xyz": [1, 1, 1]}, FakeCrowd(10), 100)

    def no_los_group(self, t, rng, counts):
        self.calls.append("no_los_group")
        if "all_no_los" in self.fail_categories:
            return None
        return ({"bs_xyz": [0, 0, 0], "ris_xyz": [1, 1, 1]}, FakeCrowd(10))

    def choose_site(self, rng, counts):
        self.calls.append("choose_site")
        return {"bs_xyz": [0, 0, 0], "ris_xyz": [1, 1, 1]}

    def elevated(self, t, rng, counts):
        self.calls.append("elevated")
        if "elevated_crowd" in self.fail_categories:
            return None
        return FakeCrowd(10)

    def natural(self, t, rng, counts):
        self.calls.append("natural")
        if "dense_crowd" in self.fail_categories:
            return FakeCrowd(20)
        return FakeCrowd(65)


@unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/previews/bench-users-v1.1/case_manifest.json').exists(), 'needs the packaged data artifact')
class BenchProductionManifestRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.freeze = verify_freeze()
        self.full_path = ROOT / "outputs/runs/bench_production_full_20261109/cases.json"
        self.challenge_path = ROOT / "outputs/runs/bench_production_full_20261109/challenge_cases.json"
        self.pilot2_path = ROOT / "outputs/runs/bench_production_pilot2_20261109/pilot_cases.json"
        self.pilot1_path = ROOT / "outputs/runs/bench_production_pilot_20261109/pilot_cases.json"

    def test_registered_constants(self):
        self.assertIn(PILOT1_HASH, PILOTS)
        self.assertIn(PILOT2_HASH, PILOTS)
        self.assertIn(FULL_HASH, PILOTS)
        self.assertIn(CHALLENGE_HASH, PILOTS)
        self.assertEqual(PILOTS[PILOT1_HASH], ("two_map_pilot", PILOT1_CASES))
        self.assertEqual(PILOTS[PILOT2_HASH], ("three_map_pilot2", PILOT2_CASES))
        self.assertEqual(PILOTS[FULL_HASH], ("development_full", FULL_CASES))
        self.assertEqual(PILOTS[CHALLENGE_HASH], ("development_challenge", CHALLENGE_CASES))
        self.assertEqual(len(FULL_CASES), 360)
        self.assertEqual(len(CHALLENGE_CASES), 69)
        self.assertEqual(registered_scopes(), ("development_full", "development_challenge"))
        self.assertEqual(PRODUCTION_SCOPES, ("development_full", "development_challenge"))

    def test_full_production_manifest_validates(self):
        self.assertTrue(self.full_path.is_file(), "full production manifest must exist")
        self.assertEqual(sha256(self.full_path), FULL_HASH)
        manifest, scope, digest = registered_manifest(self.full_path, self.freeze)
        self.assertEqual(scope, "development_full")
        self.assertEqual(digest, FULL_HASH)
        self.assertFalse(manifest.get("is_pilot"))
        self.assertEqual(manifest.get("scope"), "development")
        self.assertEqual(manifest.get("freeze_sha256"), self.freeze)
        cases = manifest["cases"]
        self.assertEqual(len(cases), 360)
        counts = {"munich": 0, "rotterdam": 0, "maastricht": 0}
        for case in cases:
            counts[case["map_id"]] += 1
            self.assertEqual(len(case["frames"]), 16)
            self.assertEqual(len(case["bs_xyz"]), 3)
            self.assertEqual(len(case["ris_xyz"]), 3)
            self.assertEqual(len(case["ris_normal_xyz"]), 3)
            self.assertIn("bs_ris_geometry_los", case)
            for frame in case["frames"]:
                self.assertGreaterEqual(len(frame["users"]), 2)
                self.assertLessEqual(len(frame["users"]), 4)
        self.assertEqual(counts, {"munich": 120, "rotterdam": 120, "maastricht": 120})

    def test_challenge_production_manifest_validates(self):
        self.assertTrue(self.challenge_path.is_file(), "challenge production manifest must exist")
        self.assertEqual(sha256(self.challenge_path), CHALLENGE_HASH)
        manifest, scope, digest = registered_manifest(self.challenge_path, self.freeze)
        self.assertEqual(scope, "development_challenge")
        self.assertEqual(digest, CHALLENGE_HASH)
        self.assertFalse(manifest.get("is_pilot"))
        self.assertEqual(manifest.get("scope"), "development")
        self.assertEqual(manifest.get("subset"), "challenge")
        self.assertEqual(manifest.get("freeze_sha256"), self.freeze)
        self.assertEqual(manifest.get("seed_word"), 202611092)
        self.assertEqual(manifest.get("challenge_order"), list(CHALLENGE_ORDER))
        cases = manifest["cases"]
        self.assertEqual(len(cases), 69)
        counts = {"munich": 0, "rotterdam": 0, "maastricht": 0}
        for case in cases:
            counts[case["map_id"]] += 1
            self.assertEqual(case["kind"], "challenge")
            self.assertIn(case["scenario"], CHALLENGE_ORDER)
            self.assertIn(case["requested_scenario"], CHALLENGE_ORDER)
            self.assertEqual(len(case["frames"]), 16)
            self.assertEqual(len(case["bs_xyz"]), 3)
            self.assertEqual(len(case["ris_xyz"]), 3)
            self.assertEqual(len(case["ris_normal_xyz"]), 3)
            self.assertIn("bs_ris_geometry_los", case)
            for frame in case["frames"]:
                self.assertGreaterEqual(len(frame["users"]), 2)
                self.assertLessEqual(len(frame["users"]), 4)
        self.assertEqual(counts, {"munich": 23, "rotterdam": 23, "maastricht": 23})

    def test_pilot2_manifest_validation_unchanged(self):
        self.assertTrue(self.pilot2_path.is_file(), "pilot2 manifest must exist")
        manifest, scope, digest = registered_manifest(self.pilot2_path, self.freeze)
        self.assertEqual(scope, "three_map_pilot2")
        self.assertEqual(digest, PILOT2_HASH)
        self.assertTrue(manifest.get("is_pilot"))
        self.assertEqual(len(manifest["cases"]), 3)
        self.assertTrue(all([f["frame"] for f in c["frames"]] == [0] for c in manifest["cases"]))

    def test_pilot1_manifest_validation_unchanged(self):
        if not self.pilot1_path.is_file():
            self.skipTest("pilot1 manifest not present")
        manifest, scope, digest = registered_manifest(self.pilot1_path, self.freeze)
        self.assertEqual(scope, "two_map_pilot")
        self.assertEqual(digest, PILOT1_HASH)
        self.assertTrue(manifest.get("is_pilot"))
        self.assertEqual(len(manifest["cases"]), 2)
        self.assertTrue(all([f["frame"] for f in c["frames"]] == [0] for c in manifest["cases"]))

    def test_rejects_unapproved_hash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write(json.dumps({"scope": "development", "cases": []}))
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                registered_manifest(Path(f.name), self.freeze)
            self.assertIn("case manifest hash differs", str(ctx.exception))

    def test_rejects_freeze_mismatch(self):
        with self.assertRaises(ValueError):
            registered_manifest(self.full_path, "0" * 64)
        with self.assertRaises(ValueError):
            registered_manifest(self.challenge_path, "0" * 64)

    def test_tampered_development_full_rejected(self):
        data = json.loads(self.full_path.read_text())
        data["is_pilot"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write(json.dumps(data))
            f.flush()
            with self.assertRaises(ValueError):
                registered_manifest(Path(f.name), self.freeze)

    def test_tampered_challenge_rejected(self):
        base_data = json.loads(self.challenge_path.read_text())

        # 1. Wrong subset (or missing subset)
        tampered_subset = dict(base_data, subset="natural")
        self._assert_manifest_rejected(tampered_subset, scope="development_challenge")

        # 2. Case with kind == "natural"
        tampered_kind = json.loads(json.dumps(base_data))
        tampered_kind["cases"][0]["kind"] = "natural"
        self._assert_manifest_rejected(tampered_kind, scope="development_challenge")

        # 3. Missing case (68 cases instead of 69)
        tampered_missing = json.loads(json.dumps(base_data))
        tampered_missing["cases"].pop()
        self._assert_manifest_rejected(tampered_missing, scope="development_challenge")

        # 4. is_pilot is True
        tampered_pilot = dict(base_data, is_pilot=True)
        self._assert_manifest_rejected(tampered_pilot, scope="development_challenge")

        # 5. Case with frame count != 16
        tampered_frames = json.loads(json.dumps(base_data))
        tampered_frames["cases"][0]["frames"].pop()
        self._assert_manifest_rejected(tampered_frames, scope="development_challenge")

    def _assert_manifest_rejected(self, data, scope):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(data, f)
            f.flush()
            tmp_path = Path(f.name)
            digest = sha256(tmp_path)
            expected_cases = CHALLENGE_CASES if scope == "development_challenge" else FULL_CASES
            PILOTS[digest] = (scope, expected_cases)
            try:
                with self.assertRaises(ValueError):
                    registered_manifest(tmp_path, self.freeze)
            finally:
                PILOTS.pop(digest, None)

    def test_scopes_never_cross_validate(self):
        # A natural manifest registered as development_challenge fails validation
        natural_data = json.loads(self.full_path.read_text())
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(natural_data, f)
            f.flush()
            tmp_path = Path(f.name)
            digest = sha256(tmp_path)
            PILOTS[digest] = ("development_challenge", CHALLENGE_CASES)
            try:
                with self.assertRaises(ValueError):
                    registered_manifest(tmp_path, self.freeze)
            finally:
                PILOTS.pop(digest, None)

        # A challenge manifest registered as development_full fails validation
        challenge_data = json.loads(self.challenge_path.read_text())
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(challenge_data, f)
            f.flush()
            tmp_path = Path(f.name)
            digest = sha256(tmp_path)
            PILOTS[digest] = ("development_full", FULL_CASES)
            try:
                with self.assertRaises(ValueError):
                    registered_manifest(tmp_path, self.freeze)
            finally:
                PILOTS.pop(digest, None)

    def test_challenge_on_map_never_touches_other_map(self):
        fake_gen = FakeGenerator("munich")
        t = np.linspace(0, 1, 10)
        rng = np.random.default_rng(42)
        counts = Counter()

        for cat in CHALLENGE_ORDER:
            res = challenge_on_map(fake_gen, cat, t, rng, counts)
            self.assertIsNotNone(res, f"Expected success for {cat}")
            site, c, extra = res
            self.assertEqual(len(site["bs_xyz"]), 3)
            self.assertEqual(len(site["ris_xyz"]), 3)
            self.assertIsInstance(extra, dict)

        # Confirm all calls went to fake_gen
        method_names = [call if isinstance(call, str) else call[0] for call in fake_gen.calls]
        self.assertIn("shadow", method_names)
        self.assertIn("no_los_group", method_names)
        self.assertIn("natural", method_names)
        self.assertIn("elevated", method_names)

        # Returns None on failure without touching anything else
        failing_gen = FakeGenerator("munich", fail_categories=CHALLENGE_ORDER)
        for cat in CHALLENGE_ORDER:
            res = challenge_on_map(failing_gen, cat, t, rng, counts)
            self.assertIsNone(res, f"Expected None for {cat}")

    def test_category_order_and_substitution_deterministic(self):
        t = np.linspace(0, 1, 10)
        counts = Counter()

        # Check requested category follows CHALLENGE_ORDER cyclically
        for i in range(10):
            rng = np.random.default_rng(100 + i)
            gen = FakeGenerator("munich")
            site, c, extra, scenario, requested = choose_challenge(gen, i, t, rng, counts)
            self.assertEqual(requested, CHALLENGE_ORDER[i % len(CHALLENGE_ORDER)])
            self.assertEqual(scenario, requested)

        # Check substitution when requested fails
        for i in range(5):
            rng1 = np.random.default_rng(200 + i)
            rng2 = np.random.default_rng(200 + i)
            req = CHALLENGE_ORDER[i]
            expected_scenario = CHALLENGE_ORDER[(i + 1) % len(CHALLENGE_ORDER)]
            gen1 = FakeGenerator("munich", fail_categories=[req])
            gen2 = FakeGenerator("munich", fail_categories=[req])

            res1 = choose_challenge(gen1, i, t, rng1, counts)
            res2 = choose_challenge(gen2, i, t, rng2, counts)

            # Deterministic across runs
            self.assertEqual(res1[3], expected_scenario)
            self.assertEqual(res1[4], req)
            self.assertEqual(res1[3], res2[3])
            self.assertEqual(res1[4], res2[4])

        # Check error raised when all categories fail
        all_fail_gen = FakeGenerator("munich", fail_categories=CHALLENGE_ORDER)
        rng = np.random.default_rng(999)
        with self.assertRaises(RuntimeError) as ctx:
            choose_challenge(all_fail_gen, 0, t, rng, counts)
        self.assertIn("All 5 challenge categories exhausted", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
