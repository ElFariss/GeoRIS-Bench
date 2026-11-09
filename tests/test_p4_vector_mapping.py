import unittest

import numpy as np

from atomwise_ris.hybrid.embedded import zdip_E
from atomwise_ris.hybrid.p4_vector_mapping import (
    current_to_free_space_field,
    direct_channel_with_open_gap,
    field_to_open_circuit_voltage,
)


class P4VectorMappingTests(unittest.TestCase):
    def test_incoming_map_resolves_cartesian_polarization(self):
        centers = np.zeros((2, 3))
        tangents = np.array([[0, 0, 2], [1, 0, 0]], dtype=float)
        dl = np.array([0.5, 0.25])
        currents = np.array([[1 + 0j], [2 + 0j]])
        e = np.array([[0, 0, 3 + 0j], [4 + 0j, 0, 0]])
        v = field_to_open_circuit_voltage(currents, centers, tangents, dl, e)
        self.assertEqual(v.shape, (1,))
        self.assertEqual(v[0], -3.5 + 0j)

    def test_reciprocity_product_is_unconjugated(self):
        currents = np.array([[1j]])
        e = np.array([[2 + 3j, 0, 0]])
        v = field_to_open_circuit_voltage(
            currents, np.zeros((1, 3)), [[1, 0, 0]], [1], e)
        self.assertEqual(v[0], 3 - 2j)

    def test_z_axis_matches_pinned_hertzian_field_convention(self):
        f = 5.53e9
        obs = np.array([[0.7, -0.3, 2.4]])
        src = np.zeros((1, 3))
        tangent = np.array([[0, 0, 1]])
        current = np.array([1.2 - 0.4j])
        dl = np.array([0.007])
        actual = current_to_free_space_field(current, src, tangent, dl, obs, f)
        expected = zdip_E(obs, src[0], p=current[0] * dl[0], frequency=f)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)
        # A missing factor of j is observable in the absolute phasor.
        wrong = zdip_E(obs, src[0], p=current[0] * dl[0] / 1j, frequency=f)
        self.assertGreater(abs(np.angle(actual[0, 0] * np.conjugate(wrong[0, 0]))), 1.5)

    def test_vector_map_rotates_with_geometry_and_polarization(self):
        f = 5.53e9
        obs = np.array([[0.1, 0.3, 1.1]])
        src = np.array([[0.0, 0.0, 0.0]])
        t = np.array([[0.0, 0.0, 1.0]])
        cur = np.array([1.0 + 0.2j])
        dl = np.array([0.01])
        e1 = current_to_free_space_field(cur, src, t, dl, obs, f)[0]
        # Proper +90 degree rotation around y: z -> x.
        r = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        e2 = current_to_free_space_field(cur, src @ r.T, t @ r.T, dl, obs @ r.T, f)[0]
        np.testing.assert_allclose(e2, r @ e1, rtol=2e-14, atol=1e-14)

    def test_bad_shapes_and_coincident_observation_fail_closed(self):
        with self.assertRaises(ValueError):
            field_to_open_circuit_voltage([1], [[0, 0, 0]], [[0, 0, 1]], [1], [[1, 2]])
        with self.assertRaises(ValueError):
            current_to_free_space_field([1], [[0, 0, 0]], [[0, 0, 1]], [1], [[0, 0, 0]], 5.53e9)

    def test_open_gap_correction_adds_predicted_impedance_once(self):
        z = np.array([[55 - 1j, 0.2 + 0.1j], [0.2 + 0.1j, 53 - 2j]])
        dz = np.array([[0.1 + 0.2j, 0.03 - 0.01j], [0.03 - 0.01j, 0.2 - 0.1j]])
        h0, h1 = direct_channel_with_open_gap(z, dz, n_bs=1)
        expected0 = -100 * np.linalg.inv(z + 50 * np.eye(2))[1:, :1]
        expected1 = -100 * np.linalg.inv(z + dz + 50 * np.eye(2))[1:, :1]
        np.testing.assert_allclose(h0, expected0, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(h1, expected1, rtol=1e-14, atol=1e-14)
        self.assertNotEqual(h0[0, 0], h1[0, 0])

    def test_open_gap_rejects_shape_or_port_partition_errors(self):
        with self.assertRaises(ValueError):
            direct_channel_with_open_gap(np.eye(2), np.eye(3), n_bs=1)
        with self.assertRaises(ValueError):
            direct_channel_with_open_gap(np.eye(2), np.eye(2), n_bs=2)


if __name__ == "__main__":
    unittest.main()
