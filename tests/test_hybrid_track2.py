import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_hybrid_p3_dev import bootstrap, rate_on_fixed_precoder  # noqa: E402


class _Eval:
    def __init__(self):
        self.w_seen = None

    def channel(self, bits):
        return np.asarray(bits, dtype=np.complex128).reshape(1, -1)

    def score(self, field):
        self.w_seen = np.asarray(field)
        return float(np.sum(np.abs(field) ** 2))


class HybridTrack2HelpersTest(unittest.TestCase):
    def test_rescore_uses_shared_precoder(self):
        ev = _Eval()
        w = np.ones((4, 1), dtype=np.complex128)
        bits = np.array([1, 0, 1, 0], dtype=np.int8)
        value = rate_on_fixed_precoder(ev, bits, w)
        np.testing.assert_array_equal(ev.w_seen, np.array([[2 + 0j]]))
        self.assertEqual(value, 4.0)

    def test_rejects_wrong_precoder_shape(self):
        with self.assertRaises(ValueError):
            rate_on_fixed_precoder(_Eval(), np.zeros(64, dtype=np.int8), np.zeros((2, 1)))

    def test_cluster_bootstrap_is_deterministic_and_uses_cluster_means(self):
        x = np.array([-1.0, 0.0, 1.0, 3.0])
        a = bootstrap(x, seed=17, draws=500)
        b = bootstrap(x, seed=17, draws=500)
        self.assertEqual(a, b)
        self.assertEqual(a[0], 0.75)
        self.assertLessEqual(a[1], a[0])
        self.assertGreaterEqual(a[2], a[0])


if __name__ == "__main__":
    unittest.main()
