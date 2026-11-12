"""moves_pairs: exact rank-2 scores for an explicit pair list, both network classes."""
import unittest
import numpy as np

from atomwise_ris.hybrid.operator import HybridCoupledNetwork
from atomwise_ris.physics.network import CoupledNetwork


def passive_z(n, seed):
    """Symmetric (n+6)x(n+6) impedance with positive-definite real part."""
    rng = np.random.default_rng(seed)
    m = n + 6
    b = rng.normal(size=(m, m))
    x = rng.normal(size=(m, m))
    return 4.0 * (b @ b.T) / m + 5.0 * np.eye(m) + 1j * 3.0 * (x + x.T) / 2


class TestMovesPairs(unittest.TestCase):
    n = 14

    def networks(self):
        z = passive_z(self.n, 5)
        loads = np.array([1.0 + 20.0j, 3.0 - 40.0j])
        return {"coupled": CoupledNetwork(z, loads),
                "hybrid": HybridCoupledNetwork.from_monolithic_z(z, loads, self.n)}

    def test_matches_full_pair_scan_and_flip_score(self):
        rng = np.random.default_rng(11)
        for name, ev in self.networks().items():
            state = ev.state(rng.integers(0, 2, ev.n))
            full = np.asarray(ev.moves(state, True))
            pick = rng.choice(len(ev.pairs), size=40, replace=False)
            scores = ev.moves_pairs(state, ev.pairs[pick])
            self.assertEqual(scores.shape, (40,))
            np.testing.assert_allclose(scores, full[pick], rtol=0, atol=1e-12, err_msg=name)
            for pair, s in zip(ev.pairs[pick], scores):
                direct = float(ev.score(ev.flip(state, pair).field))
                self.assertAlmostEqual(float(s), direct, delta=1e-12, msg=name)
                rebuilt = ev.state(ev.flip(state, pair).bits)
                self.assertAlmostEqual(float(s), float(ev.score(rebuilt.field)), delta=1e-9, msg=name)

    def test_order_duplicates_and_empty(self):
        rng = np.random.default_rng(3)
        for name, ev in self.networks().items():
            state = ev.state(rng.integers(0, 2, ev.n))
            pairs = np.array([[1, 7], [7, 1], [1, 7], [0, ev.n - 1]])
            s = ev.moves_pairs(state, pairs)
            self.assertAlmostEqual(float(s[0]), float(s[1]), delta=1e-12)
            self.assertEqual(float(s[0]), float(s[2]))
            self.assertEqual(ev.moves_pairs(state, np.empty((0, 2), dtype=int)).shape, (0,))
            self.assertEqual(ev.moves_pairs(state, []).shape, (0,))

    def test_rejects_invalid_pairs(self):
        for ev in self.networks().values():
            state = ev.state(np.zeros(ev.n, dtype=np.int8))
            for bad in ([[2, 2]], [[0, ev.n]], [[-1, 3]], [0, 1, 2]):
                with self.assertRaises(ValueError):
                    ev.moves_pairs(state, bad)

    def test_subspace_and_anchor_scans_equal_explicit_pairs(self):
        ev = self.networks()["coupled"]
        state = ev.state(np.random.default_rng(8).integers(0, 2, ev.n))
        s, pairs = ev.moves_subspace(state, [2, 5, 9, 11])
        self.assertTrue(np.array_equal(s, ev.moves_pairs(state, pairs)))
        s, pairs = ev.moves_anchors(state, [3, 12])
        self.assertTrue(np.array_equal(s, ev.moves_pairs(state, pairs)))
        full = ev.moves(state, True)
        sub, sub_pairs = ev.moves_subspace(state, [2, 5, 9, 11])
        index = {tuple(p): k for k, p in enumerate(ev.pairs)}
        np.testing.assert_allclose(sub, full[[index[tuple(p)] for p in sub_pairs]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
