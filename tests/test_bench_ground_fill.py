import unittest

import numpy as np

from atomwise_ris.bench.ground_fill import FALLBACK, FILLED, MEASURED, GroundLayer, interpolate_xy


class GroundFillTests(unittest.TestCase):
    def test_affine_terrain_without_extrapolation_or_distant_seeds(self):
        ring = np.array([(x, y) for y in [-15., 0., 15.] for x in [-15., 0., 15.]
                         if (x, y) != (0., 0.)])
        plane = lambda xy: 3. + .2 * xy[:, 0] - .1 * xy[:, 1]
        target = np.array([[9., 3.], [0., 0.], [16., 0.]])
        pred = interpolate_xy(ring, plane(ring), target)
        self.assertAlmostEqual(pred[0], plane(target)[0])
        self.assertTrue(np.isnan(pred[1:]).all())

    def test_mounted_reference_never_borrows_neighbor_or_protected_cell(self):
        x, y = np.arange(5.), np.arange(5.)[::-1]
        owner = np.zeros((5, 5), dtype=int)
        source = owner.copy()
        z = np.full((5, 5), np.nan)
        protected = np.zeros((5, 5), dtype=bool)
        for col, ident, height, code in [(0, 1, 10., MEASURED), (2, 1, 12., FILLED), (3, 2, 20., MEASURED)]:
            owner[2, col], z[2, col], source[2, col] = ident, height, code
        protected[2, 4] = True
        buildings = [{"id": id_, "native_ground_m": height, "window": [0, 5, 0, 5]}
                     for id_, height in [("a", 9.), ("b", 21.)]]
        l = GroundLayer(x, y, z, source, owner, protected, buildings)
        got, refs, rc = l.lookup(["a"] * 4, [[1., 2.], [2., 2.], [3.6, 2.6], [4., 2.]])
        np.testing.assert_allclose(got, [10., 12., 9., np.nan], equal_nan=True)
        self.assertEqual(refs.tolist(), ["measured", "interpolated_under_footprint", FALLBACK, "unavailable"])
        np.testing.assert_array_equal(rc, [[2, 0], [2, 2], [-1, -1], [-1, -1]])
