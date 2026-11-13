"""NSGA-II reference solver (atomwise_ris.optimization.nsga2): operators, evaluator exactness, deadline, determinism."""
import itertools
import unittest
from time import perf_counter_ns

import numpy as np

from atomwise_ris.hybrid.operator import HybridCoupledNetwork
from atomwise_ris.optimization import nsga2
from atomwise_ris.physics.network import CoupledNetwork


def passive_z(n, seed):
    rng = np.random.default_rng(seed)
    m = n + 6
    b = rng.normal(size=(m, m)); x = rng.normal(size=(m, m))
    return 4.0 * (b @ b.T) / m + 5.0 * np.eye(m) + 1j * 3.0 * (x + x.T) / 2


def networks(n=24, seed=5):
    z = passive_z(n, seed); loads = np.array([1.0 + 20.0j, 3.0 - 40.0j])
    return {"coupled": CoupledNetwork(z, loads), "hybrid": HybridCoupledNetwork.from_monolithic_z(z, loads, n)}


def brute_fronts(f):
    """Reference: repeatedly peel the set of non-dominated points (definition, O(n^3))."""
    left = list(range(len(f))); rank = np.full(len(f), -1); r = 0
    while left:
        nd = [i for i in left if not any(np.all(f[j] >= f[i]) and np.any(f[j] > f[i]) for j in left)]
        rank[nd] = r; left = [i for i in left if i not in nd]; r += 1
    return rank


class TestOperators(unittest.TestCase):
    def test_non_dominated_sort_matches_definition(self):
        rng = np.random.default_rng(3)
        for k in (2, 3, 4):
            for n in (1, 2, 7, 40):
                f = rng.integers(0, 5, (n, k)).astype(float)      # many ties and duplicates
                fronts, rank = nsga2.fast_non_dominated_sort(f)
                np.testing.assert_array_equal(rank, brute_fronts(f))
                self.assertEqual(sorted(np.concatenate(fronts).tolist()), list(range(n)))
                for i, j in itertools.product(range(n), repeat=2):  # no member dominated within its own front
                    if rank[i] == rank[j]:
                        self.assertFalse(np.all(f[i] >= f[j]) and np.any(f[i] > f[j]))

    def test_known_fronts(self):
        f = np.array([[3, 1], [1, 3], [2, 2], [1, 1], [0, 0], [2, 1]], float)
        fronts, rank = nsga2.fast_non_dominated_sort(f)
        np.testing.assert_array_equal(rank, [0, 0, 0, 2, 3, 1])

    def test_crowding_distance_hand_computed(self):
        f = np.array([[0., 4.], [1., 3.], [3., 1.], [4., 0.]])
        d = nsga2.crowding_distance(f)
        self.assertTrue(np.isinf(d[0]) and np.isinf(d[3]))
        # point 1: (3-0)/4 + (4-1)/4 = 1.5 ; point 2: (4-1)/4 + (3-0)/4 = 1.5
        np.testing.assert_allclose(d[1:3], [1.5, 1.5])
        f = np.array([[0., 4.], [0.5, 3.5], [3., 1.], [4., 0.]])
        d = nsga2.crowding_distance(f)
        np.testing.assert_allclose(d[1:3], [(3 - 0) / 4 + (4 - 1) / 4, (4 - 0.5) / 4 + (3.5 - 0) / 4])
        np.testing.assert_array_equal(nsga2.crowding_distance(np.ones((2, 3))), [np.inf, np.inf])
        d = nsga2.crowding_distance(np.ones((5, 2)))           # zero span: interior gets 0
        self.assertEqual(int(np.isinf(d).sum()), 2); self.assertEqual(float(d[np.isfinite(d)].sum()), 0.0)

    def test_environmental_selection(self):
        f = np.array([[3, 1], [1, 3], [2, 2], [1, 1], [0, 0], [2, 1]], float)
        sel, rank, cd = nsga2.environmental_selection(f, 4)
        self.assertEqual(sorted(sel.tolist()), [0, 1, 2, 5]); self.assertEqual(rank.tolist(), [0, 0, 0, 1])
        sel, rank, cd = nsga2.environmental_selection(f, 2)    # truncate front 0 by crowding: boundaries kept
        self.assertEqual(sorted(sel.tolist()), [0, 1])

    def test_offspring_shapes_and_mutation_rate(self):
        rng = np.random.default_rng(0); n = 64
        pop = rng.integers(0, 2, (50, n), dtype=np.int8)
        rank = np.zeros(50, int); cd = np.zeros(50)
        for kind in ("uniform", "two_point"):
            kids = nsga2.make_offspring(pop, rank, cd, np.random.default_rng(1), 0.9, kind)
            self.assertEqual(kids.shape, pop.shape); self.assertEqual(kids.dtype, np.int8)
            self.assertTrue(set(np.unique(kids)) <= {0, 1})
        same = np.zeros((2000, n), dtype=np.int8)               # pc=0, identical parents: only mutation acts
        kids = nsga2.make_offspring(same, np.zeros(2000, int), np.zeros(2000), np.random.default_rng(2), 0.0)
        self.assertAlmostEqual(float(kids.mean()), 1 / n, delta=0.15 / n)


class TestEvaluatorExactness(unittest.TestCase):
    def test_rates_match_direct_solves(self):
        rng = np.random.default_rng(9)
        for name, ev in networks().items():
            pe = nsga2.make_rate_population(ev)
            base = rng.integers(0, 2, ev.n, dtype=np.int8)
            near = np.where(rng.random((30, ev.n)) < 0.15, 1 - base, base).astype(np.int8)   # Woodbury path
            far = rng.integers(0, 2, (10, ev.n), dtype=np.int8)                               # solve path
            bits = np.concatenate([near, far])
            for method in ("auto", "solve", "inverse", "woodbury"):
                pe.method = method
                r = pe.rates(bits, base)
                for b, row in zip(bits, r):
                    st = ev.state(b)
                    direct = nsga2.user_rates(ev, st.field)
                    np.testing.assert_allclose(row, direct, rtol=1e-9, atol=0, err_msg=f"{name}/{method}")
                    full = float(ev.score(ev.channel(b) @ ev.w))
                    np.testing.assert_allclose(row.min(), full, rtol=1e-9, atol=0)
            self.assertGreater(pe.woodbury_children, 0, name)

    def test_min_of_user_rates_is_score_bit_for_bit(self):
        for name, ev in networks().items():
            fields = np.stack([ev.state(np.random.default_rng(i).integers(0, 2, ev.n)).field for i in range(5)])
            np.testing.assert_array_equal(nsga2.user_rates(ev, fields).min(-1), ev.score(fields), err_msg=name)


class TestController(unittest.TestCase):
    def test_returned_value_is_true_rate_and_at_least_all_off(self):
        for name, ev in networks().items():
            p = nsga2.NSGA2Params(population=20, max_generations=15)
            bits, value, evals, bd = nsga2.search_coupled(ev, "nsga2", None, 4, params=p)
            self.assertAlmostEqual(value, float(ev.score(ev.channel(bits) @ ev.w)), delta=1e-9)
            off = float(ev.score(ev.state(np.zeros(ev.n, np.int8)).field))
            self.assertGreaterEqual(value, off - 1e-12)
            self.assertEqual(bd["generations"], 15); self.assertEqual(bd["stop"], "max_generations")

    def test_determinism_per_seed(self):
        ev = networks()["hybrid"]; p = nsga2.NSGA2Params(population=20, max_generations=10)
        a = nsga2.search_coupled(ev, "nsga2", None, 11, params=p)
        b = nsga2.search_coupled(ev, "nsga2", None, 11, params=p)
        c = nsga2.search_coupled(ev, "nsga2", None, 12, params=p)
        np.testing.assert_array_equal(a[0], b[0]); self.assertEqual(a[1], b[1]); self.assertEqual(a[2], b[2])
        info1, info2 = {}, {}
        nsga2.search_coupled(ev, "nsga2", None, 11, params=p, info=info1)
        nsga2.search_coupled(ev, "nsga2", None, 11, params=p, info=info2)
        self.assertEqual(info1["front0_size"], info2["front0_size"])
        self.assertFalse(np.array_equal(a[0], c[0]) and a[2] == c[2] and a[1] == c[1])

    def test_deadline_respected(self):
        nets = networks(n=64, seed=2)
        for name, ev in nets.items():
            nsga2.search_coupled(ev, "nsga2", perf_counter_ns() + 20_000_000, 1)      # warm-up
            for budget_ms in (2, 5, 10, 25):
                over = []
                for seed in range(5):
                    t = perf_counter_ns()
                    bits, value, evals, bd = nsga2.search_coupled(ev, "nsga2", t + int(budget_ms * 1e6), seed)
                    over.append((perf_counter_ns() - t) / 1e6 - budget_ms)
                    self.assertEqual(bd["stop"], "deadline")
                    self.assertGreaterEqual(evals, 1)
                self.assertLessEqual(max(over), 0.0, f"{name} {budget_ms} ms overran: {max(over):.3f} ms")

    def test_deadline_respected_large_array_tight_budget(self):
        """A1: the uncalibrated first chunk and solve/Woodbury cost changes must not overrun at large N."""
        ev = networks(n=192, seed=4)["hybrid"]
        nsga2.search_coupled(ev, "nsga2", perf_counter_ns() + 20_000_000, 1)
        for budget_ms in (3, 5, 10, 30):
            for seed in range(4):
                t = perf_counter_ns()
                bits, value, evals, bd = nsga2.search_coupled(ev, "nsga2", t + int(budget_ms * 1e6), seed)
                self.assertLessEqual((perf_counter_ns() - t) / 1e6, budget_ms, f"{budget_ms} ms seed {seed}")
                self.assertAlmostEqual(value, float(ev.score(ev.state(bits).field)), delta=1e-9)

    def test_tiny_deadline_returns_all_off(self):
        ev = networks()["coupled"]
        bits, value, evals, bd = nsga2.search_coupled(ev, "nsga2", perf_counter_ns(), 3)
        self.assertEqual(evals, 1); self.assertEqual(int(bits.sum()), 0)

    def test_unknown_method_and_bad_population(self):
        ev = networks()["coupled"]
        with self.assertRaises(ValueError):
            nsga2.search_coupled(ev, "me_ga", None, 1)
        with self.assertRaises(ValueError):
            nsga2.search_coupled(ev, "nsga2", None, 1, params=nsga2.NSGA2Params(population=7, max_generations=1))

    def test_dedup_does_not_change_trajectory(self):
        ev = networks()["coupled"]
        a = nsga2.search_coupled(ev, "nsga2", None, 5, params=nsga2.NSGA2Params(population=20, max_generations=12))
        b = nsga2.search_coupled(ev, "nsga2", None, 5, params=nsga2.NSGA2Params(population=20, max_generations=12, dedup=False))
        np.testing.assert_array_equal(a[0], b[0]); self.assertAlmostEqual(a[1], b[1], delta=1e-12)
        self.assertLessEqual(a[2], b[2])


if __name__ == "__main__":
    unittest.main()
