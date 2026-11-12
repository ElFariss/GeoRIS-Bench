import copy
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import run_bench_production_trace as trace


ROOT = Path(__file__).resolve().parents[1]


class _Paths:
    def __init__(self, a):
        self.a = (a.real, a.imag)


class _RotterdamDecoder:
    @staticmethod
    def decode_vertices(feature):
        return np.asarray(feature["vertices"], dtype=np.float64)


class ProductionTraceGeometryTests(unittest.TestCase):
    def test_ris_lattice_is_planar_centered_and_has_five_cm_clearance(self):
        center = np.array([10., 20., 5.])
        normal = np.array([1., 0., 0.])
        points = trace.array_layout(center, normal, 64)
        self.assertEqual(points.shape, (64, 3))
        np.testing.assert_allclose(points.mean(axis=0), center + 0.05*normal, atol=1e-12)
        np.testing.assert_allclose((points-center) @ normal, .05, atol=1e-12)
        self.assertEqual(trace.array_layout(center, normal, 128).shape, (128, 3))
        self.assertEqual(trace.array_layout(center, normal, 256).shape, (256, 3))

    def test_bs_ports_are_four_half_wavelength_spaced_colocated_ports(self):
        points = trace.bs_ports([0., 0., 4.], [10., 0., 5.])
        self.assertEqual(points.shape, (4, 3))
        spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
        np.testing.assert_allclose(spacing, trace.BS_SPACING_M, atol=1e-12)
        np.testing.assert_allclose(points.mean(axis=0), [0., 0., 4.], atol=1e-12)

    def test_path_coefficients_use_vh_jones_rank_five_without_cir_reordering(self):
        a = np.arange(2*2*3*2*4).reshape(3, 2, 2, 2, 4).astype(float) + 1j
        result = trace._jones(_Paths(a))
        self.assertEqual(result.shape, (3, 2, 2, 2, 4))
        np.testing.assert_array_equal(result, a)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/previews/bench-users-v1.1/recovered-source-manifest.json').exists(), 'needs the packaged data artifact')
    def test_rotterdam_pinned_tile_hashes_match_approved_source_manifest(self):
        source = json.loads((ROOT / "outputs/previews/bench-users-v1.1/recovered-source-manifest.json").read_text())
        expected = {x["file"]: x["sha256"] for x in next(m for m in source if m["id"] == "rotterdam")["buildings"]}
        actual = {item[0]: item[3] for item in trace.ROTTERDAM_TILES}
        self.assertEqual(actual, expected)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/previews/bench-users-v1.1/recovered-source-manifest.json').exists(), 'needs the packaged data artifact')
    def test_maastricht_pinned_tile_hashes_match_approved_source_manifest(self):
        source = json.loads((ROOT / "outputs/previews/bench-users-v1.1/recovered-source-manifest.json").read_text())
        expected = {x["file"]: x["sha256"] for x in next(m for m in source if m["id"] == "maastricht")["buildings"]}
        actual = {item[0]: item[3] for item in trace.MAASTRICHT_TILES}
        self.assertEqual(actual, expected)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/previews/bench-users-v1.1/case_manifest.json').exists(), 'needs the packaged data artifact')
    def test_verify_case_manifest_accepts_pilot1_and_pilot2(self):
        p1 = ROOT / "outputs/runs/bench_production_pilot_20261109/pilot_cases.json"
        p2 = ROOT / "outputs/runs/bench_production_pilot2_20261109/pilot_cases.json"
        m1, c1 = trace.verify_case_manifest(p1)
        self.assertEqual(m1["_run_scope"], "two_map_pilot")
        self.assertEqual([c["id"] for c in c1], ["01-munich-natural", "15-rotterdam-natural"])
        m2, c2 = trace.verify_case_manifest(p2)
        self.assertEqual(m2["_run_scope"], "three_map_pilot2")
        self.assertEqual(
            [c["id"] for c in c2],
            ["01-munich-natural", "15-rotterdam-natural", "39-maastricht-natural"],
        )

    def test_unified_triangulator_orients_to_outward_newell_and_enforces_area_gate(self):
        class FakePoint:
            def __init__(self, x, y):
                self.x, self.y = float(x), float(y)

        class FakeRing:
            def __init__(self, coords):
                self.coords = [tuple(c) for c in coords]

        class FakePolygon:
            def __init__(self, shell, holes=None, valid=True):
                pts = np.asarray(shell, dtype=float)
                if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
                    pts = pts[:-1]
                self.pts = pts
                self.geom_type = "Polygon"
                self.is_valid = valid
                self.is_empty = len(pts) < 3
                self.exterior = FakeRing([*pts.tolist(), pts[0].tolist()] if len(pts) else [])
                if len(pts) >= 3:
                    x, y = pts[:, 0], pts[:, 1]
                    self.area = 0.5 * float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
                    self.centroid = FakePoint(np.mean(x), np.mean(y))
                else:
                    self.area = 0.0
                    self.centroid = FakePoint(0.0, 0.0)

            def contains(self, _pt):
                return True

        class FakeCollection:
            def __init__(self, geoms):
                self.geoms = geoms

        fake_geom = types.ModuleType("shapely.geometry")
        fake_geom.Polygon = FakePolygon
        fake_shapely = types.ModuleType("shapely")
        fake_shapely.geometry = fake_geom
        fake_shapely.make_valid = lambda poly: FakePolygon(poly.pts, valid=True)
        fake_shapely.constrained_delaunay_triangles = lambda poly: FakeCollection([
            FakePolygon(poly.pts[:3]),
            FakePolygon(np.vstack([poly.pts[0], poly.pts[2], poly.pts[3]])),
        ])

        rect = np.array([[10., 20., 5.], [12., 20., 5.], [12., 23., 5.], [10., 23., 5.]])
        with patch.dict(sys.modules, {"shapely": fake_shapely, "shapely.geometry": fake_geom}):
            status, tris, _, meta = trace.triangulate_lod2_face([rect])
            self.assertEqual(status, "clean")
            self.assertEqual(len(tris), 2)
            self.assertEqual(meta["outward_triangles"], 2)
            self.assertAlmostEqual(meta["source_area_m2"], 6.0, places=6)
            self.assertAlmostEqual(meta["emitted_area_m2"], 6.0, places=6)
            for t in tris:
                n = np.cross(t[1] - t[0], t[2] - t[0])
                self.assertGreater(n[2], 0.0)

            # Degenerate collinear ring is dropped as zero-area
            collinear = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
            st_zero, tris_zero, _, _ = trace.triangulate_lod2_face([collinear])
            self.assertEqual(st_zero, "dropped_zero")
            self.assertEqual(tris_zero, [])

            # Severely warped face where triangulated 3D area differs by > 1% is dropped
            fake_shapely.constrained_delaunay_triangles = lambda poly: FakeCollection([
                FakePolygon(poly.pts[:3]),
            ])
            st_drop, tris_drop, _, meta_drop = trace.triangulate_lod2_face([rect])
            self.assertEqual(st_drop, "dropped_area")
            self.assertEqual(tris_drop, [])
            self.assertGreater(meta_drop["relative_area_error"], 0.01)

    def test_preflight_contract_passes_valid_and_fails_closed_on_violations(self):
        v11_ref = {
            "bounds": [0.0, 100.0, 0.0, 100.0],
            "terrain_z_min_m": 10.0,
            "terrain_z_max_m": 20.0,
            "v11_max_building_height_m": 30.0,
            "v11_ground_by_id": {"b1": 12.0},
        }
        per_class = {
            "GroundSurface": {
                **trace._empty_class_audit(),
                "source_faces": 1,
                "clean_faces": 1,
                "emitted_triangles": 2,
                "outward_oriented_triangles": 2,
                "source_area_m2": 100.0,
                "kept_source_area_m2": 100.0,
                "emitted_area_m2": 100.0,
            },
            "RoofSurface": {
                **trace._empty_class_audit(),
                "source_faces": 10,
                "clean_faces": 9,
                "dropped_positive_area_faces": 1,
                "emitted_triangles": 18,
                "outward_oriented_triangles": 18,
                "source_area_m2": 1000.0,
                "kept_source_area_m2": 998.0,
                "emitted_area_m2": 998.0,
                "dropped_positive_area_m2": 2.0,
            },
        }
        per_building = {
            "b1": {
                "clean_faces": 10,
                "repaired_faces": 0,
                "dropped_zero_area_faces": 0,
                "dropped_positive_area_faces": 1,
                "cropped_outside_scene_faces": 0,
                "ground_base_z_m": 12.0,
            }
        }
        audit = trace.finalize_mesh_audit_and_preflight(
            "rotterdam", copy.deepcopy(per_class), copy.deepcopy(per_building), (12.0, 40.0), 28.0, v11_ref=v11_ref
        )
        self.assertTrue(audit["preflight_checks"]["all_passed"])
        self.assertEqual(audit["map_totals"]["outward_normal_share"], 1.0)

        # Violate 0.5% dropped area limit
        bad_drop = copy.deepcopy(per_class)
        bad_drop["RoofSurface"]["dropped_positive_area_m2"] = 6.0
        with self.assertRaises(RuntimeError):
            trace.finalize_mesh_audit_and_preflight(
                "rotterdam", bad_drop, copy.deepcopy(per_building), (12.0, 40.0), 28.0, v11_ref=v11_ref
            )

        # Violate outward normal 100% check
        bad_norm = copy.deepcopy(per_class)
        bad_norm["GroundSurface"]["outward_oriented_triangles"] = 1
        with self.assertRaises(RuntimeError):
            trace.finalize_mesh_audit_and_preflight(
                "rotterdam", bad_norm, copy.deepcopy(per_building), (12.0, 40.0), 28.0, v11_ref=v11_ref
            )

        # Violate z bounds check
        with self.assertRaises(RuntimeError):
            trace.finalize_mesh_audit_and_preflight(
                "rotterdam", copy.deepcopy(per_class), copy.deepcopy(per_building), (12.0, 99.0), 28.0, v11_ref=v11_ref
            )

    def test_delay_angle_helper_indexes_sionna_paths_layout(self):
        a = np.arange(3*2*4*2*5).reshape(3, 2, 4, 2, 5)
        self.assertEqual(trace._receiver_path_value(a, 2, 3, 4, 3, 4, 5), a[2, 0, 3, 0, 4])
        self.assertEqual(trace._receiver_path_value(a, 1, 2, 1, 3, 4, 5), a[1, 0, 2, 0, 1])

    def test_rotterdam_adapter_drops_only_exactly_zero_area_face_and_counts_it(self):
        feature = {
            "vertices": [[0, 0, 0], [0, 0, 0], [2, 3, 4],
                         [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
            "CityObjects": {"buildingpart": {"type": "BuildingPart", "geometry": [{
                "lod": "2.2", "type": "Solid",
                "boundaries": [[[[0, 1, 2]], [[3, 4, 5, 6]]]],
                "semantics": {"surfaces": [{"type": "WallSurface"}], "values": [[0, 0]]},
            }]}}
        }
        prepared, audit = trace._prepare_rotterdam_features([feature], _RotterdamDecoder)
        geometry = prepared[0]["CityObjects"]["buildingpart"]["geometry"][0]
        self.assertEqual(audit["zero_area_faces_dropped"], 1)
        self.assertEqual(audit["zero_area_face_counts_by_surface"], {"WallSurface": 1})
        self.assertEqual(len(geometry["boundaries"][0]), 1)
        self.assertEqual(len(geometry["semantics"]["values"][0]), 1)
        self.assertEqual(audit["retained_semantic_faces_by_surface"], {"WallSurface": 1})

    def test_rotterdam_scene_crop_counts_only_complete_outside_face_boxes(self):
        feature = {
            "vertices": [[0, 0, 0], [0, 0, 0], [2, 3, 4],
                         [100, 100, 0], [102, 100, 0], [102, 102, 0], [100, 102, 0]],
            "CityObjects": {"buildingpart": {"type": "BuildingPart", "geometry": [{
                "lod": "2.2", "type": "Solid",
                "boundaries": [[[[0, 1, 2]], [[3, 4, 5, 6]]]],
                "semantics": {"surfaces": [{"type": "GroundSurface"}], "values": [[0, 0]]},
            }]}}
        }
        prepared, audit = trace._prepare_rotterdam_features(
            [feature], _RotterdamDecoder, np.array([-1., -1., 10., 10.]))
        geometry = prepared[0]["CityObjects"]["buildingpart"]["geometry"][0]
        self.assertEqual(audit["zero_area_faces_dropped"], 1)
        self.assertEqual(audit["outside_scene_faces_cropped"], 1)
        self.assertEqual(audit["retained_semantic_faces_by_surface"], {})
        self.assertEqual(geometry["boundaries"], [[]])

    def test_rotterdam_face_local_projection_and_shared_writer_subtract_origin_once(self):
        class Polygon:
            def __init__(self, points):
                self.points = np.asarray(points, dtype=float)

            @property
            def bounds(self):
                return (float(self.points[:, 0].min()), float(self.points[:, 1].min()),
                        float(self.points[:, 0].max()), float(self.points[:, 1].max()))

            def translated(self, dx, dy):
                return Polygon(self.points + [dx, dy])

        affinity = types.ModuleType("shapely.affinity")
        affinity.translate = lambda polygon, xoff, yoff: polygon.translated(xoff, yoff)
        shapely = types.ModuleType("shapely")
        shapely.affinity = affinity

        class Decoder:
            @staticmethod
            def triangulate_semantic_face(rings, vertices, **kwargs):
                pts = np.asarray(vertices[np.asarray(rings[0], dtype=int), :3])
                return [pts[:3]], Polygon(pts[:, :2])

        origin = np.array([92000., 436000., -5.])
        world = origin + np.array([[0., 0., 0.], [2., 0., 0.], [2., 3., 0.], [0., 3., 0.]])
        _, triangulate = trace._triangulate_rotterdam_face_locally(Decoder, origin)
        with patch.dict(sys.modules, {"shapely": shapely, "shapely.affinity": affinity}):
            triangles, projected = triangulate([[0, 1, 2, 3]], world)
        np.testing.assert_allclose(np.asarray(triangles[0]), world[:3])
        np.testing.assert_allclose(projected.bounds, [0., 0., 2., 3.])
        local_tri = np.asarray(triangles[0]) - origin
        with TemporaryDirectory() as tmp:
            trace.native.write_scene(Path(tmp), {"concrete": [local_tri + origin]}, origin)
            ply = Path(tmp) / "concrete.ply"
            header = []
            with ply.open() as stream:
                for line in stream:
                    header.append(line)
                    if line.strip() == "end_header":
                        break
            points = np.loadtxt(ply, skiprows=len(header), max_rows=3, usecols=(0, 1, 2))
            np.testing.assert_allclose(points, local_tri, atol=1e-6)

    def test_interaction_class_handles_los_padding_without_calling_it_mixed(self):
        class Interaction:
            NONE = 0
            SPECULAR = 1
            DIFFUSE = 2
            REFRACTION = 4
            DIFFRACTION = 8

        interactions = np.full((3, 1, 2, 1, 2, 3), -1, dtype=np.int32)
        interactions[:, 0, 0, 0, 0, 0] = [0, 0, 0]
        interactions[:, 0, 0, 0, 0, 1] = [1, 0, 0]
        interactions[:, 0, 0, 0, 0, 2] = [1, 8, 0]
        kind, _ = trace._interaction_meta(interactions, Interaction, 0, 0, 0)
        self.assertEqual(kind, 1)
        kind, _ = trace._interaction_meta(interactions, Interaction, 0, 0, 1)
        self.assertEqual(kind, 2)
        kind, _ = trace._interaction_meta(interactions, Interaction, 0, 0, 2)
        self.assertEqual(kind, 4)

    def test_munich_diagnostic_is_forced_into_distinct_paths(self):
        cases = [{"id": "01-munich-natural", "map_id": "munich"},
                 {"id": "15-rotterdam-natural", "map_id": "rotterdam"}]
        with TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root, out = parent / "munich-diagnostic-work", parent / "munich-diagnostic"
            selected = trace.select_diagnostic_cases(cases, root, out, enabled=True)
            self.assertEqual([case["id"] for case in selected], ["01-munich-natural"])
            retry = trace.select_diagnostic_cases(
                cases, parent / "munich-diagnostic-work-retry1", parent / "munich-diagnostic-retry1",
                enabled=True)
            self.assertEqual([case["id"] for case in retry], ["01-munich-natural"])
            with self.assertRaises(ValueError):
                trace.select_diagnostic_cases(cases, root, parent / "trace", enabled=True)
            with self.assertRaises(ValueError):
                trace.select_diagnostic_cases(cases, parent / "munich-diagnostic-work-retry1", out,
                                              enabled=True)
            with self.assertRaises(ValueError):
                trace.select_diagnostic_cases(cases[1:], root, out, enabled=True)


if __name__ == "__main__":
    unittest.main()
