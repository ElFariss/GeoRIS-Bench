import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from run_bench_rt_preview_smoke import (
    MAASTRICHT_ORIGIN,
    compute_exclusive_link_classes,
    orient_triangle_to_normal,
    reduce_cir_links,
    validate_case_geometry,
    write_scene,
)


class BenchRTPreviewSmokeMaskTests(unittest.TestCase):
    def test_reduce_cir_links_handles_all_mode_shapes_including_empty_paths(self):
        rx_count = 66
        rng = np.random.default_rng(7)
        for num_paths in (0, 1, 13, 14):
            coeff = np.zeros((rx_count, 2, 1, 2, num_paths, 1), dtype=np.complex64)
            if num_paths > 0:
                active_rx = rng.choice(rx_count, size=10, replace=False)
                coeff[active_rx, 0, 0, 1, num_paths - 1, 0] = 1e-4 + 2e-4j
            links, path_mask, nonzero_rx = reduce_cir_links(coeff, rx_count)
            self.assertEqual(links.shape, (rx_count, 2, 2, num_paths))
            self.assertEqual(path_mask.shape, (rx_count, num_paths))
            self.assertEqual(path_mask.dtype, np.dtype(bool))
            self.assertEqual(nonzero_rx.shape, (rx_count,))
            self.assertEqual(nonzero_rx.dtype, np.dtype(bool))
            if num_paths == 0:
                self.assertEqual(int(nonzero_rx.sum()), 0)
                self.assertEqual(int(path_mask.sum()), 0)
            else:
                self.assertEqual(int(nonzero_rx.sum()), 10)
                self.assertEqual(int(path_mask.sum()), 10)

    def test_reduce_cir_links_rejects_invalid_cir_rank_or_receiver_count(self):
        with self.assertRaises(RuntimeError):
            reduce_cir_links(np.zeros((66, 2, 2, 13), dtype=np.complex64), 66)
        with self.assertRaises(RuntimeError):
            reduce_cir_links(np.zeros((64, 2, 1, 2, 5, 1), dtype=np.complex64), 66)

    def test_exclusive_link_classes_partition_ris_and_users_across_modes(self):
        ris_count = 64
        n_users = 2
        total = ris_count + n_users
        # Construct synthetic 6-D CIR arrays for los_only (path=1), reflection_only (path=13),
        # and diffraction_only (path=0, matching the attempt7 failure shape).
        coeff_los = np.zeros((total, 2, 1, 2, 1, 1), dtype=np.complex64)
        coeff_ref = np.zeros((total, 2, 1, 2, 13, 1), dtype=np.complex64)
        coeff_dif = np.zeros((total, 2, 1, 2, 0, 1), dtype=np.complex64)

        # Indices 0..9: LOS only
        coeff_los[:10, 0, 0, 0, 0, 0] = 1.0
        # Indices 10..24: reflection only (multiple paths active)
        coeff_ref[10:25, 1, 0, 0, 3, 0] = 1.0
        coeff_ref[10:25, 0, 0, 1, 12, 0] = 0.5
        # Indices 25..39: LOS + reflection (mixed_interactions)
        coeff_los[25:40, 0, 0, 0, 0, 0] = 1.0
        coeff_ref[25:40, 0, 0, 0, 7, 0] = 1.0
        # User 0 (idx 64): reflection only; User 1 (idx 65): fully blocked
        coeff_ref[64, 0, 0, 0, 0, 0] = 1.0

        _, _, los_mask = reduce_cir_links(coeff_los, total)
        _, _, ref_mask = reduce_cir_links(coeff_ref, total)
        _, _, dif_mask = reduce_cir_links(coeff_dif, total)

        counts = compute_exclusive_link_classes(los_mask, ref_mask, dif_mask, ris_count, n_users)
        self.assertEqual(counts["LOS_present"], {"ris_elements": 10, "served_users": 0})
        self.assertEqual(counts["reflection_only"], {"ris_elements": 15, "served_users": 1})
        self.assertEqual(counts["diffraction_only"], {"ris_elements": 0, "served_users": 0})
        self.assertEqual(counts["mixed_interactions"], {"ris_elements": 15, "served_users": 0})
        self.assertEqual(counts["fully_blocked"], {"ris_elements": 24, "served_users": 1})
        self.assertEqual(sum(v["ris_elements"] for v in counts.values()), ris_count)
        self.assertEqual(sum(v["served_users"] for v in counts.values()), n_users)

    def test_exclusive_link_classes_rejects_2d_unreduced_mask(self):
        los = np.zeros(66, dtype=bool)
        bad_ref = np.zeros((66, 13), dtype=bool)
        dif = np.zeros(66, dtype=bool)
        with self.assertRaises(ValueError):
            compute_exclusive_link_classes(los, bad_ref, dif, 64, 2)

    def test_write_scene_avoids_sionna_material_shape_name_collision(self):
        tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_scene(out, {"brick": [tri], "concrete": [tri]}, np.zeros(3))
            root = ET.parse(out / "scene.xml").getroot()
            mat_names = {el.attrib["id"].removeprefix("mat-") for el in root.findall("bsdf")}
            shape_names = {el.attrib["id"].removeprefix("mesh-") for el in root.findall("shape")}
            self.assertTrue(mat_names.isdisjoint(shape_names))

    def test_orient_triangle_to_normal_preserves_or_flips_winding_to_match_outward_normal(self):
        tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        up = np.array([0.0, 0.0, 1.0])
        down = np.array([0.0, 0.0, -1.0])
        aligned = orient_triangle_to_normal(tri, up)
        flipped = orient_triangle_to_normal(tri, down)
        self.assertGreater(float(np.dot(np.cross(aligned[1] - aligned[0], aligned[2] - aligned[0]), up)), 0.0)
        self.assertGreater(float(np.dot(np.cross(flipped[1] - flipped[0], flipped[2] - flipped[0]), down)), 0.0)
        np.testing.assert_allclose(aligned, tri)
        np.testing.assert_allclose(flipped, tri[[0, 2, 1]])

    def test_validate_case_geometry_rejects_z_shifted_floating_mesh(self):
        case = {
            "bs_xyz": (MAASTRICHT_ORIGIN + np.array([302.98, 54.66, 66.17])).tolist(),
            "ris_xyz": (MAASTRICHT_ORIGIN + np.array([339.40, 30.50, 53.01])).tolist(),
            "ris_normal_xy": [-0.3055, 0.9522, 0.0],
            "served_users_xyz": [
                (MAASTRICHT_ORIGIN + np.array([326.86, 49.06, 51.90])).tolist(),
                (MAASTRICHT_ORIGIN + np.array([335.47, 134.88, 50.81])).tolist(),
            ],
        }
        good_scene = {
            "origin_m": MAASTRICHT_ORIGIN.tolist(),
            "mesh_bounds_local_m": [[-86.61, -6.95, 45.99], [459.06, 497.93, 79.85]],
        }
        checked = validate_case_geometry(good_scene, case)
        self.assertTrue(checked["all_xy_inside_mesh_bounds"])
        self.assertTrue(checked["all_z_inside_mesh_band"])

        # Double-offset / double-origin bug in attempt10 shifted z to [92.57, 159.71]
        floating_scene = {
            "origin_m": MAASTRICHT_ORIGIN.tolist(),
            "mesh_bounds_local_m": [[-173.22, -13.91, 92.57], [918.13, 995.86, 159.71]],
        }
        with self.assertRaises(RuntimeError):
            validate_case_geometry(floating_scene, case)

    def test_maastricht_scene_origin_subtracted_exactly_once(self):
        world_tri = MAASTRICHT_ORIGIN + np.array(
            [[339.0, 30.0, 50.0], [340.0, 30.0, 50.0], [339.0, 30.0, 60.0]]
        )
        # Simulate polygon_surfaces output (origin subtracted once) then restore before write_scene
        classes_from_polygon_surfaces = {"brick": [world_tri - MAASTRICHT_ORIGIN]}
        world_classes = {
            k: [t + MAASTRICHT_ORIGIN for t in tris]
            for k, tris in classes_from_polygon_surfaces.items()
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            info = write_scene(out, world_classes, MAASTRICHT_ORIGIN)
            np.testing.assert_allclose(info["mesh_bounds_local_m"][0], [339.0, 30.0, 50.0])
            np.testing.assert_allclose(info["mesh_bounds_local_m"][1], [340.0, 30.0, 60.0])


if __name__ == "__main__":
    unittest.main()
