"""Regression checks against over-classifying algebraic smoke tests as physics gates."""
import unittest
from pathlib import Path

from atomwise_ris.hybrid.evidence import audit_legacy_gate_evidence


class TestHybridGateEvidence(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/runs/hybrid_gates/gate_P1.json').exists(), 'needs the packaged data artifact')
    def test_legacy_self_comparisons_do_not_authorize_optimizer(self):
        audit = audit_legacy_gate_evidence()
        self.assertTrue(audit["legacy_summary_claimed_all_passed"])
        self.assertEqual(audit["gates"]["P1"]["status"], "not_verified_by_legacy_artifact")
        self.assertEqual(audit["gates"]["P2"]["status"], "not_verified_by_legacy_artifact")
        self.assertEqual(audit["gates"]["P4"]["status"], "blocked_uncalibrated_operator")
        self.assertFalse(audit["optimizer_authorized"])
        self.assertFalse(audit["training_authorized"])

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/runs/hybrid_gates/gate_P1.json').exists(), 'needs the packaged data artifact')
    def test_returned_action_mesh_audit_is_reported_separately(self):
        audit = audit_legacy_gate_evidence()
        p3 = audit["gates"]["P3"]
        self.assertEqual(p3["status"], "partial_mesh_evidence_action_error_observed")
        self.assertEqual(p3["action_audit_actions"], 54)
        self.assertAlmostEqual(p3["action_audit_max_rate_absolute_error"], 0.643806341438097)


if __name__ == "__main__":
    unittest.main()
