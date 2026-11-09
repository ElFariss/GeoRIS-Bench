import unittest
import json
from pathlib import Path

import numpy as np

from scripts.build_bench_phase0b_scene import (
    BAG_TILES,
    decode_vertices,
    face_surface_indices,
    select_height_window,
    triangulate_semantic_face,
)


class CityJSONTest(unittest.TestCase):
    @unittest.skipUnless(__import__('pathlib').Path('outputs/audits/bench-phase0b-20261108/area_coverage.json').exists(), 'needs the packaged data artifact')
    def test_pinned_building_tiles_match_lead_source_manifest(self):
        root = Path(__file__).resolve().parents[1]
        audit = json.loads((root / "outputs/audits/bench-phase0b-20261108/area_coverage.json").read_text())
        area = next(row for row in audit["areas"] if row["id"] == "maastricht")
        expected = {entry["file"]: (entry["bytes"], entry["sha256"])
                    for entry in area["sources"]["buildings"]}
        actual = {entry["file"]: (entry["bytes"], entry["sha256"]) for entry in BAG_TILES}
        self.assertEqual(actual, expected)

    def test_vertex_transform_and_ground_semantics(self):
        feature = {
            "vertices": [[0, 0, 0], [2, 0, 0], [2, 2, 2], [0, 2, 2]],
            "transform": {"scale": [0.5, 0.5, 0.1], "translate": [100, 200, 10]},
        }
        vertices = decode_vertices(feature)
        np.testing.assert_allclose(vertices[2], [101, 201, 10.2])

        geometry = {
            "type": "Solid",
            "boundaries": [[[0, 1, 2, 3]]],
            "semantics": {
                "surfaces": [{"type": "RoofSurface"}],
                "values": [[0]],
            },
        }
        self.assertEqual(face_surface_indices(geometry), [([0, 1, 2, 3], "RoofSurface")])

    def test_semantics_index_is_resolved_per_polygon(self):
        geometry = {
            "type": "Solid",
            "boundaries": [[[0, 1, 2], [0, 2, 3]]],
            "semantics": {
                "surfaces": [{"type": "GroundSurface"}, {"type": "WallSurface"}],
                "values": [[0, 1]],
            },
        }
        result = face_surface_indices(geometry)
        self.assertEqual([x[1] for x in result], ["GroundSurface", "WallSurface"])

    def test_non_solid_geometry_does_not_claim_semantic_surfaces(self):
        self.assertEqual(face_surface_indices({"type": "MultiSurface", "boundaries": []}), [])

    def test_diagnostic_patch_uses_fixed_quality_gate_and_easting_order(self):
        dtm = np.full((440, 440), np.nan, dtype=float)
        dtm[200:300, 10:110] = np.linspace(20, 27, 100)[:, None]
        dtm[100:200, 20:120] = np.linspace(30, 37, 100)[:, None]
        patch = select_height_window(dtm)
        self.assertIsNotNone(patch)
        self.assertEqual(patch["col_start_1m"], 10)
        self.assertEqual(patch["row_start_1m"], 200)
        self.assertLessEqual(patch["nodata_share"], .01)
        self.assertGreaterEqual(patch["relief_m"], 5.)

    def test_no_valid_patch_stays_failed(self):
        self.assertIsNone(select_height_window(np.full((440, 440), np.nan)))


class ConstrainedTriangulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import shapely  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Shapely is an optional geospatial dependency")

    def test_concave_face_area_is_preserved(self):
        verts = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0],
                          [2, 2, 0], [0, 4, 0]], dtype=float)
        triangles, polygon = triangulate_semantic_face([[0, 1, 2, 3, 4]], verts)
        self.assertAlmostEqual(polygon.area, 12.0)
        self.assertAlmostEqual(sum(abs(np.cross(t[1]-t[0], t[2]-t[0])[2])/2
                                   for t in triangles), 12.0)

    def test_hole_is_preserved(self):
        verts = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0],
                          [1, 1, 0], [1, 3, 0], [3, 3, 0], [3, 1, 0]], dtype=float)
        triangles, polygon = triangulate_semantic_face([[0, 1, 2, 3], [4, 5, 6, 7]], verts)
        self.assertAlmostEqual(polygon.area, 12.0)
        self.assertAlmostEqual(sum(abs(np.cross(t[1]-t[0], t[2]-t[0])[2])/2
                                   for t in triangles), 12.0)
        self.assertTrue(all(not (1 < t[:, 0].mean() < 3 and 1 < t[:, 1].mean() < 3)
                            for t in triangles))


if __name__ == "__main__":
    unittest.main()
