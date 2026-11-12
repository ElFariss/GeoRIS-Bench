"""Independent PyNEC comparisons for small physical convention probes."""
import importlib.util
import unittest
import numpy as np

from atomwise_ris.hybrid.direct_validation import run_direct_nec_microproof


@unittest.skipUnless(importlib.util.find_spec("PyNEC"), "PyNEC is required for direct NEC probes")
class TestDirectNecMicroproof(unittest.TestCase):
    def test_incident_port_and_load_currents_match_direct_nec(self):
        result = run_direct_nec_microproof()
        p1 = result["p1_direct_nec_port_consistency"]
        self.assertLess(p1["short_circuit_current_relative_error"], 1.0e-3)
        # NEC's discretized source/field extraction leaves a small numerical
        # reciprocity residual; keep this as a regression check, not a gate.
        self.assertLess(p1["open_circuit_voltage_helper_vs_minus_Z_i_sc_relative_error"], 1.0e-5)
        self.assertLess(max(p1["loaded_current_relative_errors"].values()), 1.0e-3)
        self.assertEqual(result["geometry"]["ris_elements"], 9)
        self.assertTrue(result["geometry"]["extended_thin_wire_kernel_enabled"])
        self.assertAlmostEqual(result["geometry"]["segment_length_to_radius"], 480.0 / 21)
        self.assertIn("not a 64-port gate pass", p1["conclusion"])

    def test_spherical_hybrid_error_is_recorded_against_full_nec(self):
        result = run_direct_nec_microproof()
        p2 = result["p2_direct_nec_spherical_comparison"]
        errors = [record["relative_error"] for record in p2["states"].values()]
        np.testing.assert_allclose(errors, [0.02498233156592434, 0.03342797413593567, 0.026015340422427733], rtol=0.1, atol=1e-6)
        self.assertIn("no pass threshold applied", p2["conclusion"])


if __name__ == "__main__":
    unittest.main()
