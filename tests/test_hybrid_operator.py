"""Unit tests for the factored hybrid channel operator."""
import unittest
import numpy as np

from atomwise_ris.hybrid.operator import HybridCoupledNetwork

class TestHybridOperator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.n = 8
        self.m = 2
        self.k = 2
        
        # Construct synthetic well-conditioned matrices
        v = self.rng.normal(size=(self.n, self.n))
        x = self.rng.normal(size=(self.n, self.n))
        self.z_ris = v @ v.T + 20.0 * np.eye(self.n) + 1j * (x + x.T)
        
        self.c_in = self.rng.normal(size=(self.n, self.m)) + 1j * self.rng.normal(size=(self.n, self.m))
        self.c_out = self.rng.normal(size=(self.k, self.n)) + 1j * self.rng.normal(size=(self.k, self.n))
        self.h_direct = self.rng.normal(size=(self.k, self.m)) + 1j * self.rng.normal(size=(self.k, self.m))
        self.loads = np.array([1.0 + 50.0j, 1.0 - 50.0j])
        
        self.net = HybridCoupledNetwork(
            self.z_ris, self.c_out, self.c_in, self.h_direct, self.loads
        )

    def test_channel_evaluation(self):
        """Channel evaluation matches direct linear solve."""
        for b_int in range(16):
            b = np.array([(b_int >> i) & 1 for i in range(self.n)], dtype=np.int8)
            h = self.net.channel(b)
            # Direct solve
            zl = np.diag(self.loads[b])
            h_ref = self.h_direct + self.c_out @ np.linalg.solve(self.z_ris + zl, self.c_in)
            np.testing.assert_allclose(h, h_ref, atol=1e-12)

    def test_woodbury_single_and_pair_moves(self):
        """Rank-1 and rank-2 Woodbury moves match full matrix inversions."""
        s = self.net.state(np.zeros(self.n, dtype=np.int8))
        
        # Single moves
        single_scores = self.net.moves(s, pairs=False)
        self.assertEqual(len(single_scores), self.n)
        for i in range(self.n):
            new_s = self.net.flip(s, [i])
            direct_s = self.net.state(new_s.bits)
            np.testing.assert_allclose(new_s.field, direct_s.field, atol=1e-12)
            self.assertAlmostEqual(float(single_scores[i]), float(self.net.score(direct_s.field)), places=10)

        # Pair moves
        pair_scores = self.net.moves(s, pairs=True)
        self.assertEqual(len(pair_scores), len(self.net.pairs))
        for idx, (i, j) in enumerate(self.net.pairs[:10]): # check first 10 pairs
            new_s = self.net.flip(s, [i, j])
            direct_s = self.net.state(new_s.bits)
            np.testing.assert_allclose(new_s.field, direct_s.field, atol=1e-12)
            self.assertAlmostEqual(float(pair_scores[idx]), float(self.net.score(direct_s.field)), places=10)

    def test_monolithic_reduction(self):
        """Monolithic full-network Z reduces identically via Schur complement."""
        n_tot = self.n + 6
        v = self.rng.normal(size=(n_tot, n_tot))
        x = self.rng.normal(size=(n_tot, n_tot))
        z_full = v @ v.T + 20.0 * np.eye(n_tot) + 1j * (x + x.T)
        
        net_mono = HybridCoupledNetwork.from_monolithic_z(
            z_full, self.loads, n_ris=self.n, n_bs=4, n_ue=2
        )
        
        b = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        h_factored = net_mono.channel(b)
        
        zl_full = np.r_[self.loads[b], np.full(6, 50.0)]
        rhs = np.eye(n_tot)[:, self.n:self.n+4]
        sol = np.linalg.solve(z_full + np.diag(zl_full), rhs)
        h_direct = -100.0 * sol[self.n+4:]
        np.testing.assert_allclose(h_factored, h_direct, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
