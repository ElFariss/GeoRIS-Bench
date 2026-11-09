import unittest

import numpy as np

from atomwise_ris.hybrid.p4_calibration import complex_metrics, cross_polar_leakage
from evaluate_hybrid_p4 import evaluate, expected_builder_sha256


class P4CalibrationTests(unittest.TestCase):
    def test_exact_match_passes_without_alignment(self):
        x = np.array([1 + 2j, 3 - 1j])
        m = complex_metrics(x, x.copy())
        self.assertTrue(m["all_scalar_pass"])
        self.assertEqual(m["complex_relative_error"], 0.0)

    def test_phase_or_gain_mismatch_is_not_fitted_away(self):
        ref = np.array([1 + 0j])
        self.assertFalse(complex_metrics(ref * np.exp(0.02j), ref)["phase_pass"])
        self.assertFalse(complex_metrics(ref * 1.02, ref)["amplitude_pass"])

    def test_cross_polar_projection(self):
        self.assertEqual(cross_polar_leakage(np.array([1 + 0j, 0j]), np.array([1, 0])), 0.0)
        self.assertAlmostEqual(cross_polar_leakage(np.array([1 + 0j, 0.1j]), np.array([1, 0])),
                               0.09950371902099892)

    def test_invalid_comparison_is_rejected(self):
        with self.assertRaises(ValueError):
            complex_metrics(np.ones(2), np.ones(3))
        with self.assertRaises(ValueError):
            complex_metrics(np.ones(2), np.zeros(2))

    def test_p4_evaluator_fails_closed_without_physical_inputs(self):
        result = evaluate({"frequency_hz": 5.53e9})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["gate"], "P4_NOT_EVALUATED")
        self.assertIn("nec_incident_complex", result["missing"])

    def test_p4_evaluator_never_accepts_fitted_calibration(self):
        result = evaluate({"calibration_fit_applied": True})
        self.assertEqual(result["status"], "blocked")

    def test_p4_direct_open_gap_is_recomputed_without_fitting(self):
        z = np.array([[50, 1], [1, 50]], dtype=np.complex128)
        dz = np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.complex128)
        direct = lambda x: -100 * np.linalg.inv(x + 50 * np.eye(2))[1:, :1]
        ddelta = direct(z + dz) - direct(z)
        h_nec = direct(z + dz)
        enc = lambda x: np.stack([np.asarray(x).real, np.asarray(x).imag], axis=-1).tolist()
        inp = {
            "absolute_mapping": "analytical common convention, no fitted alignment",
            "calibration_fit_applied": False, "frequency_hz": 5.53e9,
            "port_order": ["BS0", "UE0"], "polarization_basis": "V/H/right-handed",
            "rt_incident_complex": enc([1 + 2j]), "nec_incident_complex": enc([1 + 2j]),
            "rt_incident_field_xyz": enc([[0, 0, 1 + 0j]]),
            "nec_incident_field_xyz": enc([[0, 0, 1 + 0j]]),
            "rt_direct_h": enc(h_nec - ddelta), "nec_direct_h": enc(h_nec),
            "z_ee_free": enc(z), "delta_z_open_gap_pred": enc(dz), "n_bs": 1,
            "delta_z_provenance": {
                "origin": "ris_alone_fixed_builder",
                "builder": "build_spherical_nec_external_operators",
                "builder_source_sha256": expected_builder_sha256(),
                "ris_alone_response_sha256": "1" * 64,
                "incident_field_sha256": "2" * 64,
                "external_nec_sha256": "3" * 64,
                "used_monolithic_truth": False,
                "calibration_fit_applied": False,
            },
        }
        result = evaluate(inp)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["passed"])
        self.assertLess(result["direct_h_ray_plus_predicted_open_gap_vs_nec"]["complex_relative_error"], 1e-14)

    def test_monolithic_truth_delta_cannot_become_predictor(self):
        inp = {
            "absolute_mapping": "test", "calibration_fit_applied": False,
            "frequency_hz": 5.53e9, "port_order": ["BS0", "UE0"],
            "polarization_basis": "V/H", "rt_incident_complex": [[1, 0]],
            "nec_incident_complex": [[1, 0]], "rt_incident_field_xyz": [[[0, 0, 1, 0]]],
            "nec_incident_field_xyz": [[[0, 0], [0, 0], [1, 0]]], "rt_direct_h": [[1, 0]],
            "rt_incident_field_xyz": [[[0, 0], [0, 0], [1, 0]]],
            "nec_direct_h": [[1, 0]], "z_ee_free": [[[50, 0], [1, 0]], [[1, 0], [50, 0]]],
            "delta_z_open_gap_pred": [[[0, 0], [0, 0]], [[0, 0], [0, 0]]], "n_bs": 1,
            "delta_z_provenance": {
                "origin": "monolithic_nec_truth", "builder": "monolithic reference",
                "builder_source_sha256": expected_builder_sha256(),
                "ris_alone_response_sha256": "1" * 64,
                "incident_field_sha256": "2" * 64,
                "external_nec_sha256": "3" * 64,
                "used_monolithic_truth": True,
                "calibration_fit_applied": False,
            },
        }
        result = evaluate(inp)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["gate"], "P4_NOT_EVALUATED")
        self.assertIn("monolithic truth", result["reason"])

    def test_legacy_truth_delta_field_is_not_a_prediction_input(self):
        result = evaluate({"delta_z_open_gap": [[0, 0]]})
        self.assertEqual(result["status"], "blocked")
        self.assertIn("not an allowed prediction input", result["reason"])


if __name__ == "__main__":
    unittest.main()
