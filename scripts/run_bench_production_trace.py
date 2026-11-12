#!/usr/bin/env python3
"""Bounded, manifest-driven Sionna RT trace export for production pilot cases.

This entry point intentionally accepts only the two approved development pilot
cases. It writes raw per-path arrays to NPZ on the compute host; those files are
not intended to be copied into the repository.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_bench_rt_preview_smoke as native

from atomwise_ris.bench.production_freeze import verify_freeze
from atomwise_ris.bench.production_pilot import PILOT1_HASH, PILOT2_HASH, registered_manifest
from atomwise_ris.bench.production_run import (
    ARRAY_SIZES,
    DEVELOPMENT_MAPS,
    PRODUCTION_SCOPES,
    atomic_write_json,
    cases_by_map,
    check_cpu_set,
    stop_rule_exceeded,
)

FREQ_HZ = 5.53e9
C0 = 299_792_458.0
PITCH_H_M = 0.028
PITCH_V_M = 0.030
BS_PORTS = 4
BS_SPACING_M = C0 / FREQ_HZ / 2
ARRAY_SHAPES = {64: (8, 8), 128: (8, 16), 256: (16, 16)}
APPROVED_CASES = {"01-munich-natural", "15-rotterdam-natural"}
APPROVED_CASES_PILOT2 = {"01-munich-natural", "15-rotterdam-natural", "39-maastricht-natural"}
PILOT_CASES_SHA256 = PILOT1_HASH
PILOT2_CASES_SHA256 = PILOT2_HASH
REPAIR_AREA_TOLERANCE_REL = 0.01
MAX_DROPPED_AREA_SHARE = 0.005
EMITTED_AREA_TOLERANCE_REL = 0.001
BASE_Z_TOLERANCE_M = 2.5

ROTTERDAM_TILES = [
    ("9-312-516.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/9/312/516/9-312-516.city.json.gz", 1904964, "5eb1ad405df4f28b663e9872c90f14a12193d3704493eb05b3cd009d37b6ca85"),
    ("9-312-520.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/9/312/520/9-312-520.city.json.gz", 2061214, "c9c84484af995fb9bcdc25b19a7cdc0cdb76d0d5822fad7d77d637a125ae07d6"),
    ("9-316-516.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/9/316/516/9-316-516.city.json.gz", 990411, "a069cab77b0e54c64dbf0b1867d9bc696336d75bf0fcb641956560a4a18802e5"),
    ("9-316-520.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/9/316/520/9-316-520.city.json.gz", 1861561, "a2ed8b21e3f9398388dd3b4ccc86c1e304d93de0fb83dc38d8ba713e22c9def6"),
]

MAASTRICHT_TILES = list(native.MAASTRICHT)
THREE_DBAG_TILES = {
    "rotterdam": ROTTERDAM_TILES,
    "maastricht": MAASTRICHT_TILES,
}
THREE_DBAG_MATERIAL_MAP = {
    "GroundSurface": "concrete",
    "WallSurface": "brick",
    "OuterCeilingSurface": "concrete",
    "RoofSurface": "concrete",
    "Window": "glass",
    "WindowSurface": "glass",
}
THREE_DBAG_FALLBACK_MATERIAL = "concrete"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_snapshot() -> list[dict]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in raw.strip().splitlines():
        idx, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "total_mib": total, "used_mib": used,
                     "free_mib": free, "util_pct": util})
    return rows


class GpuSampler:
    """Sample all devices during one solve so peak attribution is explicit."""
    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(gpu_snapshot())
            except Exception:
                pass
            self.stop_event.wait(self.interval_s)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def peak_for(self, gpu_id: int) -> dict:
        rows = [g for sample in self.samples for g in sample if g["index"] == gpu_id]
        if not rows:
            return {"samples": 0, "peak_used_mib": None, "min_free_mib": None,
                    "sampling_interval_s": self.interval_s}
        return {"samples": len(rows), "peak_used_mib": max(x["used_mib"] for x in rows),
                "min_free_mib": min(x["free_mib"] for x in rows),
                "sampling_interval_s": self.interval_s}


def verify_case_manifest(path: Path, production: bool = False,
                         smoke_unregistered: bool = False) -> tuple[dict, list[dict]]:
    lock_sha = verify_freeze()
    if smoke_unregistered:
        obj = json.loads(path.read_text())
        scope = obj.get("scope", "smoke")
        cases = obj.get("cases") if isinstance(obj, dict) else None
        if not cases:
            raise ValueError("smoke manifest must contain at least one case")
    else:
        obj, scope, _ = registered_manifest(path, lock_sha)
        cases = obj.get("cases") if isinstance(obj, dict) else None

    if production:
        if not smoke_unregistered and scope not in PRODUCTION_SCOPES:
            raise ValueError(f"pilot scope '{scope}' cannot be run with --production")
    else:
        if scope in PRODUCTION_SCOPES:
            raise ValueError(f"production scope '{scope}' requires --production")

    if not production:
        expected_map_by_id = {
            "01-munich-natural": "munich",
            "15-rotterdam-natural": "rotterdam",
            "39-maastricht-natural": "maastricht",
        }
        normalized = []
        for case in cases:
            expected_map = expected_map_by_id.get(case.get("id"))
            if case.get("map_id") != expected_map or len(case.get("frames", [])) != 1:
                raise ValueError(f"{case['id']}: only frame-0 registered pilot entries are accepted")
            frame = case["frames"][0]
            if int(frame.get("frame", -1)) != 0 or not frame.get("users"):
                raise ValueError(f"{case['id']}: frame 0 with served users is required")
            for field in ("bs_xyz", "ris_xyz", "ris_normal_xyz"):
                a = np.asarray(case.get(field), dtype=float)
                if a.shape != (3,) or not np.isfinite(a).all():
                    raise ValueError(f"{case['id']}: invalid {field}")
            normal = np.asarray(case["ris_normal_xyz"], dtype=float)
            if np.linalg.norm(normal) < 0.99 or np.linalg.norm(normal) > 1.01:
                raise ValueError(f"{case['id']}: RIS normal must be a unit vector")
            for user in frame["users"]:
                xyz = np.asarray(user.get("xyz"), dtype=float)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError(f"{case['id']}: invalid user coordinate")
            normalized.append(case)
        obj = {**obj, "_run_scope": scope}
        return obj, sorted(normalized, key=lambda c: c["id"])
    else:
        normalized = []
        for case in cases:
            if case.get("map_id") not in DEVELOPMENT_MAPS:
                raise ValueError(f"{case.get('id')}: unexpected map_id '{case.get('map_id')}'")
            frames = case.get("frames", [])
            if not smoke_unregistered and len(frames) != 16:
                raise ValueError(f"{case.get('id')}: production cases must contain exactly 16 frames, got {len(frames)}")
            if len(frames) < 1:
                raise ValueError(f"{case.get('id')}: must contain at least one frame")
            for frame in frames:
                if not frame.get("users"):
                    raise ValueError(f"{case.get('id')}: frame {frame.get('frame')} with served users is required")
                for user in frame["users"]:
                    xyz = np.asarray(user.get("xyz"), dtype=float)
                    if xyz.shape != (3,) or not np.isfinite(xyz).all():
                        raise ValueError(f"{case.get('id')}: invalid user coordinate")
            for field in ("bs_xyz", "ris_xyz", "ris_normal_xyz"):
                a = np.asarray(case.get(field), dtype=float)
                if a.shape != (3,) or not np.isfinite(a).all():
                    raise ValueError(f"{case.get('id')}: invalid {field}")
            normal = np.asarray(case["ris_normal_xyz"], dtype=float)
            if np.linalg.norm(normal) < 0.99 or np.linalg.norm(normal) > 1.01:
                raise ValueError(f"{case.get('id')}: RIS normal must be a unit vector")
            normalized.append(case)
        obj = {**obj, "_run_scope": scope}
        return obj, normalized


def select_diagnostic_cases(cases: list[dict], root: Path, out: Path,
                            enabled: bool) -> list[dict]:
    """Restrict an isolated diagnostic to Munich and non-gate output paths."""
    if not enabled:
        return cases
    root_match = re.fullmatch(r"munich-diagnostic-work(-retry[1-9][0-9]*)?", root.name)
    out_match = re.fullmatch(r"munich-diagnostic(-retry[1-9][0-9]*)?", out.name)
    if not root_match or not out_match or root_match.group(1) != out_match.group(1):
        raise ValueError("Munich diagnostic requires separate, matching diagnostic and optional retry paths")
    if root.resolve().parent != out.resolve().parent:
        raise ValueError("Munich diagnostic work and output must share their dedicated parent directory")
    selected = [case for case in cases if case["id"] == "01-munich-natural"]
    if len(selected) != 1 or selected[0].get("map_id") != "munich":
        raise ValueError("Munich diagnostic requires exactly the registered Munich case")
    return selected


def array_layout(center_xyz, normal_xyz, count: int) -> np.ndarray:
    rows, cols = ARRAY_SHAPES[count]
    center = np.asarray(center_xyz, dtype=np.float64)
    normal = np.asarray(normal_xyz, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    # Project world-up onto the plane, then obtain the horizontal panel tangent.
    up = np.array([0., 0., 1.])
    vertical = up - (up @ normal) * normal
    vertical /= np.linalg.norm(vertical)
    horizontal = np.cross(vertical, normal)
    horizontal /= np.linalg.norm(horizontal)
    # Center the lattice on the physical face; the case coordinate is the face origin.
    points = []
    for row in range(rows):
        for col in range(cols):
            points.append(center + 0.05*normal + (col-(cols-1)/2)*PITCH_H_M*horizontal
                          + (row-(rows-1)/2)*PITCH_V_M*vertical)
    return np.asarray(points, dtype=np.float64)


def bs_ports(bs_xyz, ris_xyz) -> np.ndarray:
    bs = np.asarray(bs_xyz, dtype=np.float64)
    delta = np.asarray(ris_xyz, dtype=np.float64)[:2] - bs[:2]
    delta /= np.linalg.norm(delta)
    horizontal = np.array([-delta[1], delta[0], 0.], dtype=np.float64)
    return np.stack([bs + (i-(BS_PORTS-1)/2)*BS_SPACING_M*horizontal
                     for i in range(BS_PORTS)])


def _prepare_rotterdam_features(features: list[dict], decoder,
                               scene_bounds_xy: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Remove only exactly zero-area LoD2 faces and count every removal."""
    counts: dict[str, int] = {}
    cropped_counts: dict[str, int] = {}
    details = []
    crop_details = []
    kept_count: dict[str, int] = {}
    prepared = []
    for source_feature in features:
        feature = copy.deepcopy(source_feature)
        vertices = decoder.decode_vertices(feature)
        for object_id, obj in feature.get("CityObjects", {}).items():
            if obj.get("type") != "BuildingPart":
                continue
            for geometry_index, geometry in enumerate(obj.get("geometry", [])):
                if geometry.get("lod") != "2.2" or geometry.get("type") != "Solid":
                    continue
                boundaries = geometry.get("boundaries", [])
                semantics = geometry.get("semantics", {})
                values = semantics.get("values", [])
                surfaces = semantics.get("surfaces", [])
                nested = bool(values and isinstance(values[0], list))
                new_boundaries, new_values = [], []
                for shell_index, shell in enumerate(boundaries):
                    semantic_shell = values[shell_index] if nested and shell_index < len(values) else values
                    retained_faces, retained_semantics = [], []
                    for face_index, face in enumerate(shell):
                        sid = semantic_shell[face_index] if face_index < len(semantic_shell) else None
                        surface = (surfaces[sid].get("type") if sid is not None and sid < len(surfaces)
                                   else "unknown")
                        clean_face = []
                        outer = None
                        for ring_index, ring in enumerate(face):
                            ids = [int(index) for index in ring]
                            clean_ids = []
                            for index in ids:
                                if not clean_ids or not np.array_equal(vertices[index, :3],
                                                                       vertices[clean_ids[-1], :3]):
                                    clean_ids.append(index)
                            if (len(clean_ids) > 1 and
                                    np.array_equal(vertices[clean_ids[0], :3], vertices[clean_ids[-1], :3])):
                                clean_ids.pop()
                            if ring_index == 0:
                                outer = vertices[np.asarray(clean_ids, dtype=int), :3]
                            clean_face.append(clean_ids)
                        all_face_points = np.concatenate(
                            [vertices[np.asarray(r, dtype=int), :3] for r in clean_face if r], axis=0)
                        if scene_bounds_xy is not None and len(all_face_points):
                            xmin, ymin, xmax, ymax = map(float, scene_bounds_xy)
                            fb = [float(all_face_points[:, 0].min()), float(all_face_points[:, 1].min()),
                                  float(all_face_points[:, 0].max()), float(all_face_points[:, 1].max())]
                            wholly_outside = (fb[2] < xmin or fb[0] > xmax or
                                              fb[3] < ymin or fb[1] > ymax)
                            if wholly_outside:
                                cropped_counts[surface] = cropped_counts.get(surface, 0) + 1
                                crop_details.append({"object_id": object_id,
                                                     "geometry_index": geometry_index,
                                                     "face_index": face_index,
                                                     "surface_type": surface,
                                                     "face_bounds_xy": fb,
                                                     "scene_bounds_xy": [xmin, ymin, xmax, ymax],
                                                     "reason": "complete_face_AABB_disjoint_from_declared_RF_scene"})
                                continue
                        unique = np.unique(outer, axis=0) if outer is not None and len(outer) else np.empty((0, 3))
                        zero_area = len(unique) < 3
                        area_m2 = 0.0
                        if not zero_area:
                            local = unique - unique[0]
                            baseline = local[1]
                            noncollinear = any(np.any(np.cross(baseline, point) != 0.0)
                                               for point in local[2:])
                            zero_area = not noncollinear
                            if noncollinear:
                                # Report a stable positive lower-bound witness: area of the
                                # largest triangle formed with the first unique vertex.
                                area_m2 = max(float(np.linalg.norm(np.cross(local[1], point)) / 2.0)
                                              for point in local[2:])
                        if zero_area:
                            counts[surface] = counts.get(surface, 0) + 1
                            details.append({"object_id": object_id, "geometry_index": geometry_index,
                                            "face_index": face_index, "surface_type": surface,
                                            "ring_lengths_after_duplicate_normalization":
                                                [len(r) for r in clean_face],
                                            "unique_outer_vertices": int(len(unique)),
                                            "area_m2": 0.0,
                                            "reason": "fewer_than_three_unique_vertices_or_exactly_collinear"})
                            continue
                        retained_faces.append(clean_face)
                        retained_semantics.append(sid)
                        kept_count[surface] = kept_count.get(surface, 0) + 1
                    new_boundaries.append(retained_faces)
                    new_values.append(retained_semantics)
                geometry["boundaries"] = new_boundaries
                semantics["values"] = new_values if nested else (new_values[0] if new_values else [])
        prepared.append(feature)
    return prepared, {"zero_area_faces_dropped": sum(counts.values()),
                      "zero_area_face_counts_by_surface": counts,
                      "retained_semantic_faces_by_surface": kept_count,
                      "zero_area_faces": details,
                      "outside_scene_faces_cropped": sum(cropped_counts.values()),
                      "outside_scene_face_counts_by_surface": cropped_counts,
                      "outside_scene_faces": crop_details}


def _triangulate_rotterdam_face_locally(decoder, scene_origin: np.ndarray):
    """Wrap the audited triangulator with a per-face coordinate origin."""
    original = decoder.triangulate_semantic_face

    class OffsetVertexView:
        def __init__(self, vertices, offset):
            self.vertices, self.offset = vertices, offset

        def __getitem__(self, index):
            return np.asarray(self.vertices[index], dtype=np.float64) - self.offset

    def triangulate(rings, vertices_world, **kwargs):
        vertices_world = np.asarray(vertices_world, dtype=np.float64)
        offset = vertices_world[int(rings[0][0]), :3].copy()
        view = OffsetVertexView(vertices_world, offset)
        triangles, polygon = original(rings, view, **kwargs)
        outer = view[np.asarray(rings[0], dtype=int), :3]
        normal = np.zeros(3, dtype=np.float64)
        for p, q in zip(outer, np.roll(outer, -1, axis=0)):
            normal += np.array([(p[1]-q[1])*(p[2]+q[2]),
                                (p[2]-q[2])*(p[0]+q[0]),
                                (p[0]-q[0])*(p[1]+q[1])])
        drop = int(np.argmax(np.abs(normal)))
        keep = [axis for axis in range(3) if axis != drop]
        from shapely.affinity import translate
        polygon = translate(polygon,
                            xoff=float(offset[keep[0]]-scene_origin[keep[0]]),
                            yoff=float(offset[keep[1]]-scene_origin[keep[1]]))
        return [np.asarray(triangle, dtype=np.float64) + offset for triangle in triangles], polygon

    return original, triangulate


def _newell_normal(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    nx = float(np.sum((y - np.roll(y, -1)) * (z + np.roll(z, -1))))
    ny = float(np.sum((z - np.roll(z, -1)) * (x + np.roll(x, -1))))
    nz = float(np.sum((x - np.roll(x, -1)) * (y + np.roll(y, -1))))
    return np.array([nx, ny, nz], dtype=np.float64)


def _clean_ring_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-9):
        pts = pts[:-1]
    if len(pts):
        pts = pts[np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)]]
    return pts


def triangulate_lod2_face(
    raw_rings: list[np.ndarray],
    *,
    area_tolerance_rel: float = REPAIR_AREA_TOLERANCE_REL,
    plane_tolerance_m: float = 0.01,
) -> tuple[str, list[np.ndarray], object | None, dict]:
    """Unified LoD2 semantic face repair and triangulation in per-face local coordinates.

    Returns (status, triangles_world, projected_polygon, meta) where status is one of:
    - 'clean': valid planar face triangulated without topology/planarity repair
    - 'repaired': self-intersecting or non-planar face deterministically repaired and within 1% 3D area
    - 'dropped_zero': zero-area or <3 unique vertices face dropped
    - 'dropped_area': repaired/non-planar face rejected because 3D area error > 1% or empty domain
    """
    import shapely
    from shapely.geometry import Polygon

    rings = [_clean_ring_points(r) for r in raw_rings if len(r) > 0]
    if not rings or len(rings[0]) < 3:
        return "dropped_zero", [], None, {
            "reason": "outer_ring_fewer_than_three_vertices",
            "source_area_m2": 0.0,
            "emitted_area_m2": 0.0,
            "zero_area_triangles_dropped": 0,
            "outward_triangles": 0,
        }
    outer = rings[0]
    if len(np.unique(np.round(outer, 9), axis=0)) < 3:
        return "dropped_zero", [], None, {
            "reason": "outer_ring_fewer_than_three_unique_vertices",
            "source_area_m2": 0.0,
            "emitted_area_m2": 0.0,
            "zero_area_triangles_dropped": 0,
            "outward_triangles": 0,
        }
    offset = outer[0].copy()
    local_rings = [r - offset for r in rings if len(r) >= 3]
    n_vec = _newell_normal(local_rings[0])
    n_norm = float(np.linalg.norm(n_vec))
    outer_area = 0.5 * n_norm
    if outer_area <= 1e-8 or not np.isfinite(outer_area):
        return "dropped_zero", [], None, {
            "reason": "collinear_or_zero_3d_newell_area",
            "source_area_m2": 0.0,
            "emitted_area_m2": 0.0,
            "zero_area_triangles_dropped": 0,
            "outward_triangles": 0,
        }
    n_unit = n_vec / n_norm
    hole_area = sum(0.5 * float(np.linalg.norm(_newell_normal(hr))) for hr in local_rings[1:])
    src_area = max(0.0, outer_area - hole_area)
    if src_area <= 1e-8:
        return "dropped_zero", [], None, {
            "reason": "zero_net_3d_area_after_holes",
            "source_area_m2": 0.0,
            "emitted_area_m2": 0.0,
            "zero_area_triangles_dropped": 0,
            "outward_triangles": 0,
        }
    drop = int(np.argmax(np.abs(n_unit)))
    if abs(n_unit[drop]) <= 1e-8:
        return "dropped_area", [], None, {
            "reason": "ill_conditioned_dominant_axis",
            "source_area_m2": src_area,
            "emitted_area_m2": 0.0,
            "zero_area_triangles_dropped": 0,
            "outward_triangles": 0,
        }
    keep = [i for i in range(3) if i != drop]
    plane_err = float(max(np.max(np.abs(r @ n_unit)) for r in local_rings))
    poly2 = Polygon(local_rings[0][:, keep], [hr[:, keep] for hr in local_rings[1:]])
    topology_repaired = bool(not poly2.is_valid or poly2.is_empty)
    if topology_repaired:
        poly2 = shapely.make_valid(poly2)
    plane_approximated = bool(plane_err > plane_tolerance_m)
    was_repaired = topology_repaired or plane_approximated
    polys = (
        [poly2]
        if getattr(poly2, "geom_type", None) == "Polygon" and poly2.area > 1e-10
        else [
            g
            for g in getattr(poly2, "geoms", [])
            if getattr(g, "geom_type", None) == "Polygon" and g.area > 1e-10
        ]
    )
    if not polys or sum(p.area for p in polys) <= 1e-10:
        return "dropped_area", [], None, {
            "reason": "empty_projected_domain",
            "source_area_m2": src_area,
            "emitted_area_m2": 0.0,
            "plane_deviation_m": plane_err,
            "topology_repaired": topology_repaired,
            "plane_approximated": plane_approximated,
            "zero_area_triangles_dropped": 0,
            "outward_triangles": 0,
        }

    def _lift(pt2: np.ndarray) -> np.ndarray:
        q = np.asarray(pt2, dtype=np.float64)
        best = None
        for r in local_rings:
            r2 = r[:, keep]
            for i in range(len(r)):
                a, b = r2[i], r2[(i + 1) % len(r)]
                v = b - a
                den = float(v @ v)
                t = 0.0 if den <= 1e-18 else float(np.clip((q - a) @ v / den, 0.0, 1.0))
                d = q - (a + t * v)
                d2 = float(d @ d)
                if best is None or d2 < best[0]:
                    best = (d2, (1.0 - t) * r[i] + t * r[(i + 1) % len(r)])
        if best is not None and best[0] <= 1e-6:
            return best[1]
        p0 = np.mean(local_rings[0], axis=0)
        out = np.zeros(3, dtype=np.float64)
        out[keep[0]] = q[0]
        out[keep[1]] = q[1]
        out[drop] = p0[drop] - (
            n_unit[keep[0]] * (q[0] - p0[keep[0]]) + n_unit[keep[1]] * (q[1] - p0[keep[1]])
        ) / n_unit[drop]
        return out

    tris_world: list[np.ndarray] = []
    emitted_area = 0.0
    zero_tris_dropped = 0
    outward_tris = 0
    for p in polys:
        for tri in shapely.constrained_delaunay_triangles(p).geoms:
            if getattr(tri, "geom_type", None) != "Polygon" or tri.area <= 1e-10 or not p.contains(tri.centroid):
                continue
            coords2 = np.asarray(tri.exterior.coords[:-1], dtype=np.float64)
            if len(coords2) != 3:
                continue
            t3 = np.stack([_lift(c) for c in coords2])
            tn = np.cross(t3[1] - t3[0], t3[2] - t3[0])
            ta = 0.5 * float(np.linalg.norm(tn))
            if ta <= 1e-10 or not np.isfinite(ta) or not np.isfinite(t3).all():
                zero_tris_dropped += 1
                continue
            if float(np.dot(tn, n_unit)) < 0.0:
                t3 = t3[[0, 2, 1]].copy()
                tn = -tn
            if float(np.dot(tn, n_unit)) > 0.0:
                outward_tris += 1
            tris_world.append(t3 + offset)
            emitted_area += ta

    rel_err = abs(emitted_area - src_area) / src_area
    meta = {
        "source_area_m2": src_area,
        "emitted_area_m2": emitted_area,
        "relative_area_error": rel_err,
        "plane_deviation_m": plane_err,
        "topology_repaired": topology_repaired,
        "plane_approximated": plane_approximated,
        "zero_area_triangles_dropped": zero_tris_dropped,
        "outward_triangles": outward_tris,
    }
    if not tris_world or rel_err > area_tolerance_rel:
        meta["reason"] = "repaired_3d_area_mismatch_gt_1pct"
        return "dropped_area", [], None, meta
    return ("repaired" if was_repaired else "clean"), tris_world, poly2, meta


def _empty_class_audit() -> dict:
    return {
        "source_faces": 0,
        "clean_faces": 0,
        "repaired_faces": 0,
        "topology_repaired_faces": 0,
        "plane_approximated_faces": 0,
        "dropped_zero_area_faces": 0,
        "dropped_positive_area_faces": 0,
        "cropped_outside_scene_faces": 0,
        "zero_area_triangles_dropped": 0,
        "emitted_triangles": 0,
        "outward_oriented_triangles": 0,
        "outward_normal_share": 1.0,
        "source_area_m2": 0.0,
        "kept_source_area_m2": 0.0,
        "emitted_area_m2": 0.0,
        "dropped_positive_area_m2": 0.0,
        "dropped_area_share": 0.0,
        "emitted_vs_kept_area_rel_diff": 0.0,
    }


def _load_v11_reference(map_id: str) -> dict:
    geom_path = REPO / "outputs" / "previews" / "bench-users-v1.1" / "geometry" / f"{map_id}.json"
    terrain_path = REPO / "outputs" / "previews" / "bench-users-v1.1" / "geometry" / f"{map_id}-terrain.npz"
    g = json.loads(geom_path.read_text())
    with np.load(terrain_path, allow_pickle=False) as t:
        tz = np.asarray(t["z"], dtype=np.float64)
        tz_min = float(np.nanmin(tz))
        tz_max = float(np.nanmax(tz))
    v11_by_id = {b["id"]: float(b["ground_m"]) for b in g["buildings"]}
    max_bh = float(max(b["height_m"] for b in g["buildings"]))
    return {
        "bounds": [float(x) for x in g["bounds"]],
        "terrain_z_min_m": tz_min,
        "terrain_z_max_m": tz_max,
        "v11_max_building_height_m": max_bh,
        "v11_ground_by_id": v11_by_id,
    }


def finalize_mesh_audit_and_preflight(
    map_id: str,
    per_class: dict[str, dict],
    per_building: dict[str, dict],
    mesh_z_world_m: tuple[float, float],
    max_scene_building_height_m: float,
    v11_ref: dict | None = None,
) -> dict:
    """Finalize per-class/per-building/per-map mesh audit and enforce fail-closed preflight checks."""
    if v11_ref is None:
        v11_ref = _load_v11_reference(map_id)
    active_classes = {
        k: v for k, v in per_class.items()
        if v["source_faces"] > 0 or v["cropped_outside_scene_faces"] > 0
    }
    max_dropped_share = 0.0
    max_emitted_rel_diff = 0.0
    total_emitted_tris = 0
    total_outward_tris = 0
    for cls_name, st in active_classes.items():
        src_a = float(st["source_area_m2"])
        kept_a = float(st["kept_source_area_m2"])
        emit_a = float(st["emitted_area_m2"])
        drop_a = float(st["dropped_positive_area_m2"])
        st["dropped_area_share"] = (drop_a / src_a) if src_a > 0.0 else 0.0
        st["emitted_vs_kept_area_rel_diff"] = (abs(emit_a - kept_a) / kept_a) if kept_a > 0.0 else 0.0
        st["outward_normal_share"] = (
            st["outward_oriented_triangles"] / st["emitted_triangles"]
            if st["emitted_triangles"] > 0 else 1.0
        )
        max_dropped_share = max(max_dropped_share, st["dropped_area_share"])
        max_emitted_rel_diff = max(max_emitted_rel_diff, st["emitted_vs_kept_area_rel_diff"])
        total_emitted_tris += int(st["emitted_triangles"])
        total_outward_tris += int(st["outward_oriented_triangles"])

    outward_share_map = (total_outward_tris / total_emitted_tris) if total_emitted_tris > 0 else 0.0
    tz_min = float(v11_ref["terrain_z_min_m"])
    tz_max = float(v11_ref["terrain_z_max_m"])
    max_bh = max(float(v11_ref["v11_max_building_height_m"]), float(max_scene_building_height_m))
    z_min_w, z_max_w = float(mesh_z_world_m[0]), float(mesh_z_world_m[1])
    z_bounds_ok = bool(
        total_emitted_tris > 0
        and z_min_w >= tz_min - 0.5
        and z_max_w <= tz_max + max_bh + 0.5
    )

    v11_ground = v11_ref["v11_ground_by_id"]
    matched_v11 = 0
    max_base_diff = 0.0
    all_base_in_band = True
    repaired_or_dropped_buildings = {}
    buildings_with_repairs = 0
    buildings_with_drops = 0
    for bid, brec in sorted(per_building.items()):
        gz = brec.get("ground_base_z_m")
        parent_id = bid.rsplit("-", 1)[0] if "-" in bid and bid not in v11_ground else bid
        ref_z = v11_ground.get(bid, v11_ground.get(parent_id))
        if ref_z is None and brec.get("native_ground_z_m") is not None:
            ref_z = float(brec["native_ground_z_m"])
        if gz is None or not (tz_min - 0.5 <= float(gz) <= tz_max + 0.5):
            all_base_in_band = False
        elif ref_z is not None:
            matched_v11 += 1
            diff = abs(float(gz) - float(ref_z))
            brec["reference_ground_diff_m"] = diff
            max_base_diff = max(max_base_diff, diff)
        if brec["repaired_faces"] > 0:
            buildings_with_repairs += 1
        if brec["dropped_zero_area_faces"] > 0 or brec["dropped_positive_area_faces"] > 0:
            buildings_with_drops += 1
        if (brec["repaired_faces"] > 0 or brec["dropped_zero_area_faces"] > 0
                or brec["dropped_positive_area_faces"] > 0):
            repaired_or_dropped_buildings[bid] = brec

    base_z_ok = bool(
        len(per_building) > 0
        and all_base_in_band
        and matched_v11 > 0
        and max_base_diff <= BASE_Z_TOLERANCE_M
    )
    outward_ok = bool(total_emitted_tris > 0 and total_outward_tris == total_emitted_tris)
    dropped_ok = bool(max_dropped_share <= MAX_DROPPED_AREA_SHARE)
    emitted_area_ok = bool(max_emitted_rel_diff <= EMITTED_AREA_TOLERANCE_REL)

    preflight_checks = {
        "z_bounds_within_terrain_plus_height": z_bounds_ok,
        "building_base_z_matches_reference": base_z_ok,
        "outward_normal_share_100pct": outward_ok,
        "dropped_area_within_0_5_pct_per_class": dropped_ok,
        "emitted_area_within_0_1_pct_per_class": emitted_area_ok,
        "mesh_z_world_m": [z_min_w, z_max_w],
        "terrain_z_range_m": [tz_min, tz_max],
        "max_building_height_m": max_bh,
        "allowed_z_world_m": [tz_min - 0.5, tz_max + max_bh + 0.5],
        "buildings_checked": len(per_building),
        "buildings_matched_reference": matched_v11,
        "max_building_base_diff_m": max_base_diff,
        "building_base_tolerance_m": BASE_Z_TOLERANCE_M,
        "outward_normal_share": outward_share_map,
        "max_class_dropped_area_share": max_dropped_share,
        "max_class_emitted_area_rel_diff": max_emitted_rel_diff,
    }
    preflight_checks["all_passed"] = all(
        preflight_checks[k]
        for k in (
            "z_bounds_within_terrain_plus_height",
            "building_base_z_matches_reference",
            "outward_normal_share_100pct",
            "dropped_area_within_0_5_pct_per_class",
            "emitted_area_within_0_1_pct_per_class",
        )
    )
    map_totals = {
        "buildings_in_scene": len(per_building),
        "buildings_with_repairs": buildings_with_repairs,
        "buildings_with_drops": buildings_with_drops,
        "source_faces": sum(s["source_faces"] for s in active_classes.values()),
        "clean_faces": sum(s["clean_faces"] for s in active_classes.values()),
        "repaired_faces": sum(s["repaired_faces"] for s in active_classes.values()),
        "topology_repaired_faces": sum(s["topology_repaired_faces"] for s in active_classes.values()),
        "plane_approximated_faces": sum(s["plane_approximated_faces"] for s in active_classes.values()),
        "dropped_zero_area_faces": sum(s["dropped_zero_area_faces"] for s in active_classes.values()),
        "dropped_positive_area_faces": sum(s["dropped_positive_area_faces"] for s in active_classes.values()),
        "cropped_outside_scene_faces": sum(s["cropped_outside_scene_faces"] for s in active_classes.values()),
        "zero_area_triangles_dropped": sum(s["zero_area_triangles_dropped"] for s in active_classes.values()),
        "emitted_triangles": total_emitted_tris,
        "outward_oriented_triangles": total_outward_tris,
        "outward_normal_share": outward_share_map,
        "source_area_m2": sum(s["source_area_m2"] for s in active_classes.values()),
        "kept_source_area_m2": sum(s["kept_source_area_m2"] for s in active_classes.values()),
        "emitted_area_m2": sum(s["emitted_area_m2"] for s in active_classes.values()),
        "dropped_positive_area_m2": sum(s["dropped_positive_area_m2"] for s in active_classes.values()),
        "max_class_dropped_area_share": max_dropped_share,
        "max_class_emitted_area_rel_diff": max_emitted_rel_diff,
    }
    audit = {
        "policy": "docs/bench-production-mesh-policy-20261109.md",
        "map_id": map_id,
        "one_origin_shift_only": True,
        "outward_newell_normal_orientation": True,
        "repair_area_tolerance_rel": REPAIR_AREA_TOLERANCE_REL,
        "max_dropped_area_share_limit": MAX_DROPPED_AREA_SHARE,
        "emitted_area_conservation_limit_rel": EMITTED_AREA_TOLERANCE_REL,
        "per_class": active_classes,
        "map_totals": map_totals,
        "repaired_or_dropped_buildings": repaired_or_dropped_buildings,
        "preflight_checks": preflight_checks,
    }
    if not preflight_checks["all_passed"]:
        raise RuntimeError(f"{map_id}: mesh preflight check failed: {json.dumps(preflight_checks)}")
    return audit


def _munich_scene(raw_dir: Path, scene_dir: Path, case: dict,
                  anchor_users: list[dict] | None = None) -> dict:
    """Build and audit Munich LoD2 surfaces under the unified mesh-repair policy."""
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon, box

    source = raw_dir / native.MUNICH["file"]
    native.download(source, native.MUNICH["url"], native.MUNICH["bytes"], native.MUNICH["sha256"])
    users = anchor_users if anchor_users is not None else case["frames"][0]["users"]
    anchors = np.asarray(
        [case["bs_xyz"], case["ris_xyz"], *[u["xyz"] for u in users]],
        dtype=np.float64,
    )
    bounds = [
        float(anchors[:, 0].min() - 50.0),
        float(anchors[:, 1].min() - 50.0),
        float(anchors[:, 0].max() + 50.0),
        float(anchors[:, 1].max() + 50.0),
    ]
    patch = box(*bounds).buffer(35.0)
    origin = np.array([bounds[0], bounds[1], float(anchors[:, 2].min() - 40.0)], dtype=np.float64)
    classes: dict[str, list[np.ndarray]] = {k: [] for k in native.MATERIAL}
    per_class = {k: _empty_class_audit() for k in native.MATERIAL}
    per_building: dict[str, dict] = {}
    z_min_w, z_max_w = 1e9, -1e9
    max_bldg_h = 0.0

    it = ET.iterparse(source, events=("start", "end"))
    _, root = next(it)
    for event, elem in it:
        if event != "end" or elem.tag.rsplit("}", 1)[-1] != "cityObjectMember":
            continue
        building = elem.find("./{*}Building")
        if building is None:
            elem.clear()
            continue
        bid = building.attrib.get("{http://www.opengis.net/gml}id", f"munich_bldg_{len(per_building)}")
        ground_rings = [
            rs
            for surf in building.findall(".//{*}GroundSurface")
            for poly in surf.findall(".//{*}Polygon")
            if (rs := native.munich_rings(poly)) is not None
        ]
        footprints = [
            Polygon(rs[0][:, :2], [r[:, :2] for r in rs[1:]])
            for rs in ground_rings
            if len(rs[0]) >= 3
        ]
        if not any(g.area > 0 and g.intersects(patch) for g in footprints):
            elem.clear()
            continue
        brec = {
            "clean_faces": 0,
            "repaired_faces": 0,
            "dropped_zero_area_faces": 0,
            "dropped_positive_area_faces": 0,
            "cropped_outside_scene_faces": 0,
            "ground_base_z_m": None,
        }
        b_zmin, b_zmax = 1e9, -1e9
        ground_zs = []
        for key in native.MATERIAL:
            st = per_class[key]
            for surface in building.findall(f".//{{*}}{key}"):
                for poly in surface.findall(".//{*}Polygon"):
                    rings = native.munich_rings(poly)
                    if not rings:
                        continue
                    st["source_faces"] += 1
                    status, tris, _, meta = triangulate_lod2_face(rings)
                    st["source_area_m2"] += meta["source_area_m2"]
                    st["zero_area_triangles_dropped"] += meta["zero_area_triangles_dropped"]
                    if status in ("clean", "repaired"):
                        if status == "clean":
                            st["clean_faces"] += 1
                            brec["clean_faces"] += 1
                        else:
                            st["repaired_faces"] += 1
                            brec["repaired_faces"] += 1
                            if meta.get("topology_repaired"):
                                st["topology_repaired_faces"] += 1
                            if meta.get("plane_approximated"):
                                st["plane_approximated_faces"] += 1
                        st["kept_source_area_m2"] += meta["source_area_m2"]
                        st["emitted_area_m2"] += meta["emitted_area_m2"]
                        st["emitted_triangles"] += len(tris)
                        st["outward_oriented_triangles"] += meta["outward_triangles"]
                        classes[key].extend(tris)
                        for t in tris:
                            tz0, tz1 = float(t[:, 2].min()), float(t[:, 2].max())
                            z_min_w = min(z_min_w, tz0)
                            z_max_w = max(z_max_w, tz1)
                            b_zmin = min(b_zmin, tz0)
                            b_zmax = max(b_zmax, tz1)
                            if key == "GroundSurface":
                                ground_zs.append(tz0)
                    elif status == "dropped_zero":
                        st["dropped_zero_area_faces"] += 1
                        brec["dropped_zero_area_faces"] += 1
                    else:
                        st["dropped_positive_area_faces"] += 1
                        st["dropped_positive_area_m2"] += meta["source_area_m2"]
                        brec["dropped_positive_area_faces"] += 1
        if ground_zs:
            brec["ground_base_z_m"] = float(min(ground_zs))
        if b_zmax > b_zmin:
            max_bldg_h = max(max_bldg_h, b_zmax - b_zmin)
        per_building[bid] = brec
        elem.clear()
        if len(root) > 500:
            root.clear()

    if not per_building or not any(classes.values()):
        raise RuntimeError("Munich scene build produced no LoD2 surfaces in case bounds")
    mesh_audit = finalize_mesh_audit_and_preflight(
        "munich", per_class, per_building, (z_min_w, z_max_w), max_bldg_h
    )
    scene_info = native.write_scene(scene_dir, classes, origin)
    case_for_builder = {**case, "served_users_xyz": [u["xyz"] for u in users]}
    return {
        "case_id": case["id"],
        "map": "munich",
        "source_crs": "EPSG:25832",
        "source": native.MUNICH,
        "scene_bounds_epsg25832_m": bounds,
        "building_count": len(per_building),
        "semantic_triangle_count": int(sum(scene_info["triangles_by_semantic"].values())),
        "mesh_audit": mesh_audit,
        **scene_info,
        "geometry_preflight": native.validate_case_geometry(
            {"origin_m": origin.tolist(), **scene_info}, case_for_builder
        ),
    }


def _three_dbag_scene(map_id: str, raw_dir: Path, scene_dir: Path, case: dict,
                      anchor_users: list[dict] | None = None) -> dict:
    """Build and audit Rotterdam or Maastricht 3DBAG LoD2.2 surfaces under the unified policy."""
    import gzip as gz
    import json as js
    import sys as system
    from shapely.geometry import box

    system.path.insert(0, str(HERE))
    import build_bench_phase0b_scene as decoder

    v11_ref = _load_v11_reference(map_id)
    mb = v11_ref["bounds"]  # [xmin, xmax, ymin, ymax]
    users = anchor_users if anchor_users is not None else case["frames"][0]["users"]
    anchors = np.vstack([
        case["bs_xyz"],
        case["ris_xyz"],
        *[u["xyz"] for u in users],
    ])
    scene_box_xy = np.array([
        max(float(anchors[:, 0].min() - 180.0), mb[0]),
        max(float(anchors[:, 1].min() - 180.0), mb[2]),
        min(float(anchors[:, 0].max() + 180.0), mb[1]),
        min(float(anchors[:, 1].max() + 180.0), mb[3]),
    ], dtype=np.float64)
    scene_box_geom = box(*scene_box_xy)
    origin = np.array(
        [scene_box_xy[0], scene_box_xy[1], float(anchors[:, 2].min() - 35.0)],
        dtype=np.float64,
    )
    tiles = THREE_DBAG_TILES[map_id]
    features = []
    used_sources = []
    for filename, url, size, digest in tiles:
        f = raw_dir / filename
        if not f.exists() or f.stat().st_size != size or sha256(f) != digest:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f.suffix + ".part")
            req = urllib.request.Request(
                url, headers={"User-Agent": "atomwise-ris-production-trace/1.0"}
            )
            with urllib.request.urlopen(req, timeout=180) as response, tmp.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            if tmp.stat().st_size != size or sha256(tmp) != digest:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"pinned {map_id} tile mismatch: {filename}")
            tmp.replace(f)
        with gz.open(f, "rt", encoding="utf-8") as stream:
            feature = js.load(stream)
        features.append(feature)
        used_sources.append({
            "file": filename,
            "bytes": size,
            "sha256": digest,
            "url": url,
            "tile_selected_whole": True,
        })

    by_material: dict[str, list[np.ndarray]] = {
        k: [] for k in set(THREE_DBAG_MATERIAL_MAP.values())
    }
    per_class: dict[str, dict] = {
        k: _empty_class_audit()
        for k in ("GroundSurface", "WallSurface", "RoofSurface", "OuterCeilingSurface", "Window", "WindowSurface")
    }
    per_building: dict[str, dict] = {}
    height_sources: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    parts_outside = 0
    z_min_w, z_max_w = 1e9, -1e9
    max_bldg_h = 0.0
    xmin, ymin, xmax, ymax = map(float, scene_box_xy)

    for feature in features:
        vertices = decoder.decode_vertices(feature)
        city_objects = feature.get("CityObjects", {})
        for cid, obj in city_objects.items():
            if obj.get("type") != "BuildingPart":
                continue
            geoms = [
                g for g in obj.get("geometry", [])
                if g.get("lod") == "2.2" and g.get("type") == "Solid"
            ]
            if not geoms:
                continue
            parent_id = cid.rsplit("-", 1)[0]
            parent_attr = city_objects.get(parent_id, {}).get("attributes", {})
            attr = obj.get("attributes") or parent_attr or {}
            native_ground_z = attr.get("b3_h_maaiveld", parent_attr.get("b3_h_maaiveld"))
            h_src = str(attr.get("b3_pw_bron", parent_attr.get("b3_pw_bron", "unknown")))

            candidate_faces = []
            cropped_by_class: dict[str, int] = {}
            has_in_scene_ground = False
            for geom in geoms:
                sem = geom.get("semantics", {})
                vals = sem.get("values", [])
                surfs = sem.get("surfaces", [])
                nested = bool(vals and isinstance(vals[0], list))
                for s_idx, shell in enumerate(geom.get("boundaries", [])):
                    sem_shell = vals[s_idx] if nested and s_idx < len(vals) else vals
                    for f_idx, face in enumerate(shell):
                        sid = sem_shell[f_idx] if f_idx < len(sem_shell) else None
                        stype = (
                            surfs[sid].get("type")
                            if sid is not None and sid < len(surfs) and surfs[sid].get("type")
                            else "RoofSurface"
                        )
                        rings = [
                            vertices[np.asarray(r, dtype=int), :3]
                            for r in face
                            if len(r) > 0
                        ]
                        if not rings:
                            continue
                        all_pts = np.vstack(rings)
                        if (all_pts[:, 0].max() < xmin or all_pts[:, 0].min() > xmax
                                or all_pts[:, 1].max() < ymin or all_pts[:, 1].min() > ymax):
                            cropped_by_class[stype] = cropped_by_class.get(stype, 0) + 1
                            continue
                        candidate_faces.append((stype, rings))
                        if stype == "GroundSurface":
                            from shapely.geometry import Polygon
                            c_outer = _clean_ring_points(rings[0])
                            if len(c_outer) >= 3:
                                g_poly = Polygon(c_outer[:, :2])
                                if not g_poly.is_valid:
                                    import shapely
                                    g_poly = shapely.make_valid(g_poly)
                                if not g_poly.is_empty and g_poly.intersects(scene_box_geom):
                                    has_in_scene_ground = True

            for stype, cnt in cropped_by_class.items():
                per_class.setdefault(stype, _empty_class_audit())["cropped_outside_scene_faces"] += cnt

            if not has_in_scene_ground:
                parts_outside += 1
                continue

            height_sources[h_src] = height_sources.get(h_src, 0) + 1
            brec = {
                "clean_faces": 0,
                "repaired_faces": 0,
                "dropped_zero_area_faces": 0,
                "dropped_positive_area_faces": 0,
                "cropped_outside_scene_faces": int(sum(cropped_by_class.values())),
                "ground_base_z_m": None,
                "native_ground_z_m": float(native_ground_z) if native_ground_z is not None else None,
            }
            b_zmin, b_zmax = 1e9, -1e9
            ground_zs = []
            for stype, rings in candidate_faces:
                st = per_class.setdefault(stype, _empty_class_audit())
                st["source_faces"] += 1
                status, tris, _, meta = triangulate_lod2_face(rings)
                st["source_area_m2"] += meta["source_area_m2"]
                st["zero_area_triangles_dropped"] += meta["zero_area_triangles_dropped"]
                if status in ("clean", "repaired"):
                    if status == "clean":
                        st["clean_faces"] += 1
                        brec["clean_faces"] += 1
                    else:
                        st["repaired_faces"] += 1
                        brec["repaired_faces"] += 1
                        if meta.get("topology_repaired"):
                            st["topology_repaired_faces"] += 1
                        if meta.get("plane_approximated"):
                            st["plane_approximated_faces"] += 1
                    st["kept_source_area_m2"] += meta["source_area_m2"]
                    st["emitted_area_m2"] += meta["emitted_area_m2"]
                    st["emitted_triangles"] += len(tris)
                    st["outward_oriented_triangles"] += meta["outward_triangles"]
                    mat = THREE_DBAG_MATERIAL_MAP.get(stype, THREE_DBAG_FALLBACK_MATERIAL)
                    by_material[mat].extend(tris)
                    surface_counts[stype] = surface_counts.get(stype, 0) + 1
                    for t in tris:
                        tz0, tz1 = float(t[:, 2].min()), float(t[:, 2].max())
                        z_min_w = min(z_min_w, tz0)
                        z_max_w = max(z_max_w, tz1)
                        b_zmin = min(b_zmin, tz0)
                        b_zmax = max(b_zmax, tz1)
                        if stype == "GroundSurface":
                            ground_zs.append(float(np.median(t[:, 2])))
                elif status == "dropped_zero":
                    st["dropped_zero_area_faces"] += 1
                    brec["dropped_zero_area_faces"] += 1
                else:
                    st["dropped_positive_area_faces"] += 1
                    st["dropped_positive_area_m2"] += meta["source_area_m2"]
                    brec["dropped_positive_area_faces"] += 1
            if ground_zs:
                brec["ground_base_z_m"] = float(np.median(ground_zs))
            if b_zmax > b_zmin:
                max_bldg_h = max(max_bldg_h, b_zmax - b_zmin)
            per_building[cid] = brec

    if not per_building or not any(by_material.values()):
        raise RuntimeError(f"pinned {map_id} tiles yielded no LoD2 surfaces in case bounds")
    mesh_audit = finalize_mesh_audit_and_preflight(
        map_id, per_class, per_building, (z_min_w, z_max_w), max_bldg_h, v11_ref=v11_ref
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_info = native.write_scene(scene_dir, by_material, origin)
    return {
        "case_id": case["id"],
        "map": map_id,
        "source_crs": "EPSG:7415",
        "origin_m": origin.tolist(),
        "origin_projected_m": origin.tolist(),
        "scene_bounds_projected_m": scene_box_xy.tolist(),
        "sources": used_sources,
        "whole_tiles_used": True,
        "footprints": len(per_building),
        "buildingpart_height_sources": height_sources,
        "semantic_surface_counts": surface_counts,
        "parts_outside_scene": parts_outside,
        "mesh_audit": mesh_audit,
        **scene_info,
    }


def _rotterdam_scene(raw_dir: Path, scene_dir: Path, case: dict,
                     anchor_users: list[dict] | None = None) -> dict:
    return _three_dbag_scene("rotterdam", raw_dir, scene_dir, case, anchor_users=anchor_users)


def _maastricht_scene(raw_dir: Path, scene_dir: Path, case: dict,
                      anchor_users: list[dict] | None = None) -> dict:
    return _three_dbag_scene("maastricht", raw_dir, scene_dir, case, anchor_users=anchor_users)


def build_scene(case: dict, root: Path, out: Path, production: bool = False,
                scene_dir: Path | None = None,
                anchor_users: list[dict] | None = None) -> tuple[dict, np.ndarray]:
    raw = root / "raw"
    if scene_dir is None:
        if production:
            scene_dir = root / "scenes" / case["map_id"] / case["id"]
        else:
            scene_dir = root / "scenes" / case["map_id"]
    if anchor_users is None:
        if production:
            anchor_users = [u for f in case.get("frames", []) for u in f.get("users", [])]
        else:
            anchor_users = case["frames"][0]["users"]
    if case["map_id"] == "munich":
        info = _munich_scene(raw, scene_dir, case, anchor_users=anchor_users)
        origin = np.asarray(info["origin_m"], dtype=np.float64)
    elif case["map_id"] == "rotterdam":
        info = _rotterdam_scene(raw, scene_dir, case, anchor_users=anchor_users)
        origin = np.asarray(info["origin_projected_m"], dtype=np.float64)
    elif case["map_id"] == "maastricht":
        info = _maastricht_scene(raw, scene_dir, case, anchor_users=anchor_users)
        origin = np.asarray(info["origin_projected_m"], dtype=np.float64)
    else:
        raise ValueError(f"unsupported map_id in pilot trace builder: {case['map_id']}")
    bounds = info["mesh_bounds_local_m"]
    lower, upper = (np.asarray(x, dtype=float) for x in bounds)
    external = np.vstack([bs_ports(case["bs_xyz"], case["ris_xyz"]),
                          *[u["xyz"] for u in anchor_users]])
    all_points = np.vstack([external, *[array_layout(case["ris_xyz"], case["ris_normal_xyz"], n)
                                      for n in ARRAY_SHAPES]]) - origin
    if (not np.isfinite(all_points).all() or
            not np.all((all_points[:, :2] >= lower[:2]-1e-5) &
                       (all_points[:, :2] <= upper[:2]+1e-5)) or
            np.any(all_points[:, 2] <= 0)):
        raise RuntimeError(f"{case['id']}: one or more endpoint/RIS probes outside native scene extent")
    info["endpoint_preflight"] = {"points_checked": len(all_points),
                                   "all_xy_within_mesh_bounds": True,
                                   "all_local_z_positive": True,
                                   "array_counts": list(ARRAY_SHAPES)}
    if production:
        scene_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(scene_dir / "scene_info.json", info)
    return info, origin


def _path_array(paths, name):
    value = getattr(paths, name, None)
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _jones(paths):
    value = getattr(paths, "a", None)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise RuntimeError("Sionna Paths.a must be the documented (real, imaginary) tuple")
    real, imag = (np.asarray(v.numpy() if hasattr(v, "numpy") else v) for v in value)
    if real.shape != imag.shape:
        raise RuntimeError("Paths.a real/imaginary component shapes differ")
    a = real.astype(np.float64) + 1j*imag.astype(np.float64)
    if a.ndim != 5:
        raise RuntimeError(f"expected Paths.a [rx,rx_pol,tx,tx_pol,path], got {a.shape}")
    return a


def _interaction_meta(raw: np.ndarray, interaction_type, ri: int, ti: int, pi: int) -> tuple[int, dict]:
    if raw is None or raw.ndim != 6:
        raise RuntimeError("Sionna 2.0.1 Paths.interactions must have [depth,rx,rx_pol,tx,tx_pol,path]")
    seq_raw = np.asarray(raw[:, ri, 0, ti, 0, pi]).reshape(-1)
    enum = {int(getattr(interaction_type, name)): name
            for name in ("NONE", "SPECULAR", "DIFFUSE", "REFRACTION", "DIFFRACTION")}
    none_id = int(interaction_type.NONE)
    seq = [int(x) for x in seq_raw if int(x) != none_id]
    spec_id = int(interaction_type.SPECULAR)
    diff_id = int(interaction_type.DIFFRACTION)
    diffuse_id = int(interaction_type.DIFFUSE)
    refr_id = int(interaction_type.REFRACTION)
    has_ref = spec_id in seq or diffuse_id in seq
    has_diff = diff_id in seq
    unsupported = [x for x in seq if x not in {spec_id, diff_id, diffuse_id}]
    if unsupported:
        raise RuntimeError(f"unsupported path interaction sequence: {seq}; expected no refraction/other interactions")
    if not seq:
        kind = 1
    elif has_ref and not has_diff:
        kind = 2
    elif has_diff and not has_ref:
        kind = 3
    elif has_ref and has_diff:
        kind = 4
    else:
        raise RuntimeError(f"unsupported path interaction sequence: {seq}")
    return kind, {"interaction_enum_sequence": seq_raw.astype(np.int16).tolist(),
                  "interaction_nonpadding_sequence": seq,
                  "interaction_enum_names": [enum.get(x, f"UNKNOWN_{x}") for x in seq],
                  "is_los": bool(kind == 1), "has_reflection": bool(has_ref),
                  "has_diffraction": bool(has_diff), "has_other_interaction": False}


def extract_npz(case: dict, count: int, paths, tx_xyz: np.ndarray,
                rx_xyz: np.ndarray, origin: np.ndarray, scene_info: dict,
                solver: dict, out_path: Path, frame: dict | None = None) -> dict:
    coeff = _jones(paths)
    tau = _path_array(paths, "tau")
    if tau is None or tau.shape != coeff.shape:
        raise RuntimeError(f"Paths.tau must align with Paths.a: {None if tau is None else tau.shape} vs {coeff.shape}")
    interaction_raw = _path_array(paths, "interactions")
    if interaction_raw is None:
        raise RuntimeError("Sionna Paths.interactions is missing")
    from sionna.rt.constants import InteractionType
    angle_arrays = {name: _path_array(paths, name)
                    for name in ("theta_t", "phi_t", "theta_r", "phi_r")}
    if any(value is None for value in angle_arrays.values()):
        raise RuntimeError("Sionna path angle arrays are incomplete")
    nrx, npr, ntx, npt, npath = coeff.shape
    if nrx != len(rx_xyz) or ntx != len(tx_xyz) or npr != 2 or npt != 2:
        raise RuntimeError(f"unexpected dual-pol Paths.a dimensions {coeff.shape}")
    if frame is None:
        frame = case["frames"][0]
    frame_idx = int(frame.get("frame", 0))
    frame_users = frame.get("users", [])
    rows = []
    path_indices = []
    jones = []
    delay_s = []
    theta_t, phi_t, theta_r, phi_r = [], [], [], []
    is_los, has_ref, has_dif, has_other, path_type = [], [], [], [], []
    interaction_sequences = []
    interaction_names = []
    array_xyz = np.asarray(rx_xyz[:count], dtype=np.float64)
    for ti in range(ntx):
        for ri in range(nrx):
            if ri >= count and np.linalg.norm(tx_xyz[ti]-rx_xyz[ri]) < 1e-7:
                continue  # no self channel; represented by absence and documented mask
            for pi in range(npath):
                matrix = coeff[ri, :, ti, :, pi]
                if not np.any(np.abs(matrix) > 0):
                    continue
                # Delays are normally [1,rx,tx,path,time]. Keep source values then
                # cast to float64 on export; numeric precision is not recovered.
                d = tau[ri, :, ti, :, pi]
                present = np.abs(matrix) > 0
                if not np.isfinite(d[present]).all() or np.ptp(d[present]) > 1e-12:
                    raise RuntimeError("nonfinite or polarization-inconsistent path delay")
                d = float(d[present][0])
                kind, meta = _interaction_meta(interaction_raw, InteractionType, ri, ti, pi)
                rows.append((ti, ri))
                path_indices.append(pi)
                jones.append(matrix.astype(np.complex128))
                delay_s.append(d)
                path_type.append(kind)
                is_los.append(meta.get("is_los", False))
                has_ref.append(meta.get("has_reflection", False))
                has_dif.append(meta.get("has_diffraction", False))
                has_other.append(meta.get("has_other_interaction", False))
                interaction_sequences.append(json.dumps(meta["interaction_enum_sequence"]))
                interaction_names.append(json.dumps(meta["interaction_enum_names"]))
                for name, target in (("theta_t", theta_t), ("phi_t", phi_t),
                                     ("theta_r", theta_r), ("phi_r", phi_r)):
                    val = _receiver_path_value(angle_arrays[name], ri, ti, pi, nrx, ntx, npath)
                    if val is None or not np.isfinite(val):
                        raise RuntimeError(f"missing/non-finite {name} for tx={ti}, rx={ri}, path={pi}")
                    target.append(val)
    links = np.asarray(rows, dtype=np.int32).reshape(-1, 2)
    meta_json = {
        "case_id": case["id"], "map_id": case["map_id"], "frame": frame_idx,
        "frame_source_sha256": frame.get("frame_source_sha256", case.get("frame_source_sha256")),
        "frequency_hz": FREQ_HZ, "array_elements": count,
        "array_shape": ARRAY_SHAPES[count], "array_pitch_h_m": PITCH_H_M,
        "array_pitch_v_m": PITCH_V_M, "bs_port_count": BS_PORTS,
        "tx_order": ["bs_port_0", "bs_port_1", "bs_port_2", "bs_port_3"]
                    + [f"ue_{u['user_id']}" for u in frame_users],
        "rx_order": [f"ris_{i}" for i in range(count)]
                    + ["bs_port_0", "bs_port_1", "bs_port_2", "bs_port_3"]
                    + [f"ue_{u['user_id']}" for u in frame_users],
        "rx_self_links_omitted": True, "self_link_rule": "tx/rx external endpoints at same position omitted",
        "pol_order": ["V", "H"], "solver": solver,
        "coordinate_frame": "shared local projected metric frame; xyz_local=xyz_projected-origin",
        "projected_crs": scene_info.get("source_crs", "EPSG:25832"),
        "projected_origin_xyz_m": origin.tolist(),
        "scene_sha256": scene_info.get("scene_sha256"),
        "scene_mesh_sha256": scene_info.get("mesh_sha256", {}),
        "path_type_encoding": {"1": "LOS", "2": "reflection only", "3": "diffraction only", "4": "mixed reflection/diffraction"},
        "interaction_enum_sequence_semantics": "Sionna RT 2.0.1 plain InteractionType constants: NONE=0 padding, SPECULAR=1, DIFFUSE=2, REFRACTION=4, DIFFRACTION=8. Empty non-padding sequence is LOS; full padded sequence is retained.",
        "path_vertices_exported": False,
        "reciprocity_note": "UE transmitter to RIS receiver rays encode the reciprocal RIS-to-UE link in this passive reciprocal environment; direction retained in tx/rx indices.",
        "length_precision_note": "Path length is c times Sionna tau (tau exported float64 after source conversion); only LOS lengths are replaced with exact Euclidean endpoint distances. Reflection/diffraction lengths retain Sionna delay precision.",
        "ray_count": len(rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            tx_xyz=(tx_xyz-origin).astype(np.float64),
            rx_xyz=(rx_xyz-origin).astype(np.float64),
            ris_xyz=(array_xyz-origin).astype(np.float64),
            link_tx=links[:, 0], link_rx=links[:, 1], path_index=np.asarray(path_indices, np.int32),
            path_delay_s=np.asarray(delay_s, np.float64),
            path_length_m=np.asarray([np.linalg.norm(tx_xyz[t]-rx_xyz[r]) if k == 1 else C0*d
                                      for (t,r), k, d in zip(rows, path_type, delay_s)], np.float64),
            path_length_is_exact=np.asarray([k == 1 for k in path_type], bool),
            path_jones=np.asarray(jones, np.complex128).reshape(-1, 2, 2),
            path_theta_t=np.asarray(theta_t, np.float64), path_phi_t=np.asarray(phi_t, np.float64),
            path_theta_r=np.asarray(theta_r, np.float64), path_phi_r=np.asarray(phi_r, np.float64),
            path_type=np.asarray(path_type, np.uint8),
            is_los=np.asarray(is_los, bool), has_reflection=np.asarray(has_ref, bool),
            has_diffraction=np.asarray(has_dif, bool), has_other_interaction=np.asarray(has_other, bool),
            interaction_enum_json=np.asarray(interaction_sequences, dtype="U"),
            interaction_enum_names_json=np.asarray(interaction_names, dtype="U"),
            tx_external_mask=np.ones(len(tx_xyz), bool),
            rx_external_mask=np.r_[np.zeros(count, bool), np.ones(len(rx_xyz)-count, bool)],
            metadata_json=np.asarray(json.dumps(meta_json, allow_nan=False)))
    tmp.replace(out_path)
    with np.load(out_path, allow_pickle=False) as archive:
        p = len(archive["link_tx"])
        required = ("tx_xyz", "rx_xyz", "link_tx", "link_rx", "path_delay_s", "path_jones",
            "path_theta_t", "path_phi_t", "path_theta_r", "path_phi_r", "path_type", "path_length_m")
        if (any(not np.isfinite(archive[k]).all() for k in required if k not in {
                "path_jones", "path_theta_t", "path_phi_t", "path_theta_r", "path_phi_r"})
                or not np.isfinite(archive["path_jones"].real).all()
                or not np.isfinite(archive["path_jones"].imag).all()):
            raise RuntimeError("trace export contains non-finite values")
        if not (p == len(archive["path_jones"]) == len(archive["path_delay_s"]) == len(archive["path_type"])):
            raise RuntimeError("trace export arrays are not path-aligned")
    return {"file": str(out_path), "sha256": sha256(out_path), "bytes": out_path.stat().st_size,
            "paths": len(rows), "nonzero_links": int(len({tuple(x) for x in rows}))}


def _receiver_path_value(value, rxi: int, txi: int, pi: int,
                         n_rx: int, n_tx: int, n_path: int):
    """Index native non-synthetic Sionna rank-five [rx,rx-pol,tx,tx-pol,path]."""
    if value is None:
        return None
    a = np.asarray(value)
    if a.ndim != 5 or a.shape[0] != n_rx or a.shape[2] != n_tx or a.shape[4] != n_path:
        return None
    return a[rxi, 0, txi, 0, pi]


def trace_one(rt, dr, root: Path, case: dict, count: int, out_dir: Path,
              frame: dict | None = None, gpu_id: int = 0,
              scene_path: Path | None = None) -> dict:
    if frame is None:
        frame = case["frames"][0]
    frame_idx = int(frame.get("frame", 0))
    origin = np.asarray(case["_origin"], dtype=np.float64)
    ris = array_layout(case["ris_xyz"], case["ris_normal_xyz"], count)
    bs = bs_ports(case["bs_xyz"], case["ris_xyz"])
    users = np.asarray([u["xyz"] for u in frame["users"]], dtype=np.float64)
    external = np.vstack([bs, users])
    tx_xyz = external.copy()
    rx_xyz = np.vstack([ris, external])
    prep_t0 = time.perf_counter()
    if scene_path is None:
        per_case_scene = root / "scenes" / case["map_id"] / case["id"] / "scene.xml"
        if per_case_scene.is_file():
            scene_path = per_case_scene
        else:
            scene_path = root / "scenes" / case["map_id"] / "scene.xml"
    scene = rt.load_scene(str(scene_path), merge_shapes=False)
    scene.frequency = FREQ_HZ
    scene.tx_array = scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
        pattern="iso", polarization="VH")
    for i, point in enumerate(tx_xyz-origin):
        scene.add(rt.Transmitter(f"tx_{i:03d}", position=point.tolist()))
    for i, point in enumerate(rx_xyz-origin):
        scene.add(rt.Receiver(f"rx_{i:04d}", position=point.tolist()))
    scene_load_and_ports_s = time.perf_counter() - prep_t0
    snapshot = gpu_snapshot()
    gpu = next((g for g in snapshot if g["index"] == gpu_id), None)
    if gpu is None or gpu["free_mib"] < 4096:
        raise RuntimeError(f"GPU{gpu_id} stop rule: need at least 4096 MiB free")
    solver_cfg = {"max_depth": 2, "los": True, "specular_reflection": True,
                  "diffraction": True, "diffuse_reflection": False, "refraction": False,
                  "synthetic_array": False, "samples_per_src": 10000,
                  "max_num_paths_per_src": 40000, "seed": 7,
                  "note": "fixed 10k rays/source; depth 2; transmission/refraction off"}
    t0 = time.perf_counter()
    with GpuSampler() as sampler:
        paths = rt.PathSolver()(scene=scene,
                                **{key: value for key, value in solver_cfg.items() if key != "note"})
        dr.sync_thread()
        solver_elapsed = time.perf_counter()-t0
        export_start = time.perf_counter()
        local_case = {**case, "_origin": origin}
        out_path = out_dir / case["map_id"] / f"n{count}" / case["id"] / f"frame-{frame_idx:04d}.npz"
        result = extract_npz(local_case, count, paths, tx_xyz, rx_xyz, origin,
                             case["_scene_info"], {**solver_cfg,
                             "solver_elapsed_seconds": solver_elapsed}, out_path, frame=frame)
        export_elapsed = time.perf_counter()-export_start
        dr.sync_thread()
    elapsed = time.perf_counter()-t0
    gpu_after = gpu_snapshot()
    peak_gpu = sampler.peak_for(gpu_id)
    return {"case_id": case["id"], "map_id": case["map_id"], "frame": frame_idx,
            "array_elements": count, "elapsed_seconds": elapsed,
            "solver_elapsed_seconds": solver_elapsed,
            "export_elapsed_seconds": export_elapsed,
            "scene_load_and_ports_s": scene_load_and_ports_s,
            "gpu_before": snapshot, "gpu_after": gpu_after,
            "gpu_peak_sampled": peak_gpu, **result}


def run_production_trace(
    manifest: dict,
    cases: list[dict],
    root: Path,
    out: Path,
    maps: list[str] | tuple[str, ...] | None = None,
    gpu_id: int = 0,
    retry_failed: bool = False,
    scene_builder_fn: callable = None,
    trace_one_fn: callable = None,
    smoke_unregistered: bool = False,
    manifest_sha256: str | None = None,
    frames_per_case: int | None = None,
) -> dict:
    if gpu_id not in (0, 1):
        raise ValueError(f"production GPU ID must be 0 or 1, got {gpu_id}")

    gpu_lock_path = root / f"gpu{gpu_id}.lock"
    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock_file = gpu_lock_path.open("w")
    try:
        fcntl.flock(gpu_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"GPU{gpu_id} task-local lock is already held") from exc

    try:
        allowed_cpus = set(os.sched_getaffinity(0)) - {5, 11}
        if allowed_cpus:
            os.sched_setaffinity(0, allowed_cpus)

        if trace_one_fn is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            import drjit as dr
            import mitsuba as mi
            mi.set_variant("cuda_ad_mono_polarized")
            import sionna.rt as rt
            if importlib.metadata.version("sionna-rt") != "2.0.1" or not mi.variant().startswith("cuda"):
                raise RuntimeError(f"expected Sionna RT 2.0.1/CUDA Mitsuba, got {mi.variant()}")
        else:
            rt, dr = None, None

        if maps is None:
            target_maps = list(DEVELOPMENT_MAPS)
        else:
            target_maps = list(maps)

        run_scope = manifest.get("_run_scope", "development_full")
        cases_grouped = cases_by_map(cases)
        overall_status = "complete"

        for map_id in target_maps:
            map_cases = cases_grouped.get(map_id, [])
            if not map_cases:
                continue

            map_dir = out / map_id
            map_dir.mkdir(parents=True, exist_ok=True)

            if (map_dir / "DONE").is_file():
                continue
            if (map_dir / "FAILED").is_file():
                if retry_failed:
                    (map_dir / "FAILED").rename(map_dir / "FAILED.prev")
                else:
                    overall_status = "failed"
                    continue

            map_lock_path = map_dir / "trace.lock"
            map_lock_file = map_lock_path.open("w")
            try:
                fcntl.flock(map_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Map {map_id} trace lock is already held") from exc

            try:
                cases_dir = map_dir / "cases"
                cases_dir.mkdir(parents=True, exist_ok=True)
                total_cases = len(map_cases)
                failed_cases: list[str] = []
                passed_cases: list[dict] = []

                for case in map_cases:
                    case_id = case["id"]
                    case_rec_path = cases_dir / f"{case_id}.json"
                    failed_path = cases_dir / f"{case_id}.FAILED.json"
                    prev_failed_path = cases_dir / f"{case_id}.FAILED.prev.json"

                    if failed_path.is_file():
                        if retry_failed:
                            failed_path.rename(prev_failed_path)
                        else:
                            failed_cases.append(case_id)
                            if stop_rule_exceeded(len(failed_cases), total_cases):
                                atomic_write_json(map_dir / "FAILED", {
                                    "status": "failed",
                                    "map_id": map_id,
                                    "failed_cases": failed_cases,
                                    "failed_count": len(failed_cases),
                                    "total_cases": total_cases,
                                    "stop_rule_exceeded": True,
                                    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                })
                                break
                            continue

                    case_verified = False
                    if case_rec_path.is_file() and not failed_path.is_file():
                        try:
                            rec_obj = json.loads(case_rec_path.read_text())
                            records = rec_obj.get("records", [])
                            expected_frame_count = (
                                frames_per_case if frames_per_case is not None else len(case["frames"])
                            )
                            expected_record_count = expected_frame_count * len(ARRAY_SIZES)
                            if rec_obj.get("status") == "pass" and len(records) == expected_record_count:
                                all_ok = True
                                for r in records:
                                    npz_file = Path(r["file"])
                                    if not npz_file.is_file():
                                        all_ok = False
                                        break
                                    if npz_file.stat().st_size != r["bytes"]:
                                        all_ok = False
                                        break
                                    if sha256(npz_file) != r["sha256"]:
                                        all_ok = False
                                        break
                                if all_ok:
                                    case_verified = True
                                    passed_cases.append(rec_obj)
                        except Exception:
                            case_verified = False

                    if case_verified:
                        continue

                    stage = "scene_build"
                    try:
                        if scene_builder_fn is not None:
                            info, origin = scene_builder_fn(case, root, out)
                        else:
                            info, origin = build_scene(case, root, out, production=True)

                        stage = "trace"
                        local_case = {**case, "_origin": origin, "_scene_info": info}
                        case_records = []
                        frames_to_trace = case["frames"]
                        if frames_per_case is not None:
                            frames_to_trace = frames_to_trace[:frames_per_case]

                        for frame in frames_to_trace:
                            for count in ARRAY_SIZES:
                                if trace_one_fn is not None:
                                    rec = trace_one_fn(rt, dr, root, local_case, count, out, frame=frame, gpu_id=gpu_id)
                                else:
                                    rec = trace_one(rt, dr, root, local_case, count, out, frame=frame, gpu_id=gpu_id)
                                case_records.append(rec)

                        case_record = {
                            "status": "pass",
                            "case_id": case["id"],
                            "map_id": case["map_id"],
                            "run_scope": run_scope,
                            "case_manifest_sha256": manifest_sha256,
                            "scene_sha256": info.get("scene_sha256"),
                            "geometry": info,
                            "records": case_records,
                        }
                        if smoke_unregistered:
                            case_record["smoke"] = True
                            for r in case_records:
                                r["smoke"] = True

                        atomic_write_json(case_rec_path, case_record)
                        passed_cases.append(case_record)

                    except Exception as exc:
                        failed_record = {
                            "case_id": case["id"],
                            "map_id": case["map_id"],
                            "stage": stage,
                            "error": str(exc),
                            "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                        atomic_write_json(failed_path, failed_record)
                        failed_cases.append(case["id"])

                        for count in ARRAY_SIZES:
                            case_npz_dir = out / map_id / f"n{count}" / case["id"]
                            if case_npz_dir.exists():
                                for f in case_npz_dir.glob("*.npz*"):
                                    try:
                                        f.unlink(missing_ok=True)
                                    except OSError:
                                        pass
                                try:
                                    case_npz_dir.rmdir()
                                except OSError:
                                    pass

                        if stop_rule_exceeded(len(failed_cases), total_cases):
                            atomic_write_json(map_dir / "FAILED", {
                                "status": "failed",
                                "map_id": map_id,
                                "failed_cases": failed_cases,
                                "failed_count": len(failed_cases),
                                "total_cases": total_cases,
                                "stop_rule_exceeded": True,
                                "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            })
                            break

                if (map_dir / "FAILED").is_file():
                    overall_status = "failed"
                elif len(passed_cases) + len(failed_cases) == total_cases:
                    if not stop_rule_exceeded(len(failed_cases), total_cases):
                        manifest_content = {
                            "status": "complete",
                            "run_scope": run_scope,
                            "case_manifest_sha256": manifest_sha256,
                            "map_id": map_id,
                            "cases": passed_cases,
                            "failed_cases": failed_cases,
                        }
                        trace_manifest_path = map_dir / "trace_manifest.json"
                        atomic_write_json(trace_manifest_path, manifest_content)
                        done_path = map_dir / "DONE"
                        done_text = (
                            f"status=complete\n"
                            f"finished_utc={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
                            f"manifest_sha256={sha256(trace_manifest_path)}\n"
                            f"cases_passed={len(passed_cases)}\n"
                            f"cases_failed={len(failed_cases)}\n"
                        )
                        done_tmp = done_path.with_suffix(f".tmp.{os.getpid()}")
                        done_tmp.write_text(done_text)
                        done_tmp.replace(done_path)
            finally:
                try:
                    fcntl.flock(map_lock_file, fcntl.LOCK_UN)
                    map_lock_file.close()
                except Exception:
                    pass

        return {"status": overall_status, "target_maps": target_maps}
    finally:
        try:
            fcntl.flock(gpu_lock_file, fcntl.LOCK_UN)
            gpu_lock_file.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case-manifest", "--manifest", type=Path, required=True, dest="case_manifest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--reuse-scenes", action="store_true")
    ap.add_argument("--munich-only-diagnostic", action="store_true",
                    help="trace the registered Munich case only to isolated diagnostic paths")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--production", action="store_true", help="run in production mode for full/challenge manifests")
    ap.add_argument("--maps", type=str, default=None, help="comma-separated maps to trace, default all")
    ap.add_argument("--retry-failed", action="store_true", help="retry previously failed cases")
    ap.add_argument("--smoke-unregistered", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.production:
        if args.build_only or args.reuse_scenes or args.munich_only_diagnostic:
            raise ValueError("pilot-specific flags (--build-only, --reuse-scenes, --munich-only-diagnostic) are not valid with --production")
        if args.gpu_id not in (0, 1):
            raise ValueError(f"production GPU ID must be 0 or 1, got {args.gpu_id}")
        if args.smoke_unregistered and "t6b1-smoke" not in str(args.out):
            raise ValueError("--smoke-unregistered is only allowed with output paths containing 't6b1-smoke'")

        manifest, cases = verify_case_manifest(args.case_manifest, production=True, smoke_unregistered=args.smoke_unregistered)
        if args.maps:
            maps_filter = [m.strip() for m in args.maps.split(",") if m.strip()]
            for m in maps_filter:
                if m not in DEVELOPMENT_MAPS:
                    raise ValueError(f"unknown map '{m}'; must be in {DEVELOPMENT_MAPS}")
        else:
            maps_filter = list(DEVELOPMENT_MAPS)

        args.root.mkdir(parents=True, exist_ok=True)
        args.out.mkdir(parents=True, exist_ok=True)

        m_sha = sha256(args.case_manifest)
        res = run_production_trace(
            manifest=manifest,
            cases=cases,
            root=args.root,
            out=args.out,
            maps=maps_filter,
            gpu_id=args.gpu_id,
            retry_failed=args.retry_failed,
            smoke_unregistered=args.smoke_unregistered,
            manifest_sha256=m_sha,
        )
        if res.get("status") != "complete":
            sys.exit(1)
        return

    if args.gpu_id != 0:
        raise ValueError("this pilot runner is locked to approved GPU0")
    lock_sha = verify_freeze()
    manifest, cases = verify_case_manifest(args.case_manifest)
    manifest_scope = manifest.get("_run_scope", "two_map_pilot")
    if args.munich_only_diagnostic and (args.build_only or not args.reuse_scenes):
        raise ValueError("Munich-only diagnostic is trace-only and requires --reuse-scenes")
    cases = select_diagnostic_cases(cases, args.root, args.out, args.munich_only_diagnostic)
    if not args.build_only and not args.reuse_scenes:
        raise ValueError("GPU traces require a completed --build-only geometry preflight and --reuse-scenes")
    args.root.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    run_scope = "isolated_munich_diagnostic" if args.munich_only_diagnostic else manifest_scope
    run = {"status": "running", "case_manifest_sha256": sha256(args.case_manifest),
           "manifest_id": manifest.get("manifest_id"), "started_unix": time.time(),
           "freeze_verified": True, "freeze_lock_sha256": lock_sha,
           "run_scope": run_scope,
           "cases": []}
    scene_manifest_path = args.root / "scene_manifest.json"
    geo_preflight_path = args.out / "geometry_preflight.json"
    if args.reuse_scenes:
        if not scene_manifest_path.is_file():
            raise RuntimeError("geometry preflight manifest is missing")
        scene_manifest = json.loads(scene_manifest_path.read_text())
        expected_scene_cases = (
            {"01-munich-natural"}
            if args.munich_only_diagnostic
            else {c["id"] for c in cases}
        )
        if set(scene_manifest) != expected_scene_cases:
            raise RuntimeError(f"geometry manifest must contain exactly {sorted(expected_scene_cases)}")
        if geo_preflight_path.is_file():
            run["geometry_preflight_sha256"] = sha256(geo_preflight_path)
        for case in cases:
            info = scene_manifest[case["id"]]
            scene_dir = args.root / "scenes" / case["map_id"]
            if not (scene_dir / "scene.xml").is_file() or sha256(scene_dir / "scene.xml") != info["scene_sha256"]:
                raise RuntimeError(f"{case['id']}: preflight scene hash mismatch")
            for filename, digest in info.get("mesh_sha256", {}).items():
                if sha256(scene_dir / filename) != digest:
                    raise RuntimeError(f"{case['id']}: preflight mesh hash mismatch: {filename}")
    else:
        scene_manifest = {}
        for case in cases:
            geo_t0 = time.perf_counter()
            info, origin = build_scene(case, args.root, args.out)
            geo_elapsed = time.perf_counter() - geo_t0
            case = {**case, "_origin": origin, "_scene_info": info}
            scene_manifest[case["id"]] = {**info, "origin_m": origin.tolist()}
            run["cases"].append({
                "case_id": case["id"],
                "map_id": case["map_id"],
                "status": "pass",
                "geometry_elapsed_s": geo_elapsed,
                "scene": info,
                "origin": origin.tolist(),
            })
            scene_manifest_tmp = scene_manifest_path.with_suffix(".json.tmp")
            scene_manifest_tmp.write_text(json.dumps(scene_manifest, indent=2, allow_nan=False)+"\n")
            scene_manifest_tmp.replace(scene_manifest_path)
    if not args.build_only:
        lock_path = args.root / "gpu0.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("w")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("GPU0 task-local lock is already held") from exc
        allowed_cpus = set(os.sched_getaffinity(0)) - {5, 11}
        if allowed_cpus:
            os.sched_setaffinity(0, allowed_cpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        import drjit as dr
        import mitsuba as mi
        mi.set_variant("cuda_ad_mono_polarized")
        import sionna.rt as rt
        if importlib.metadata.version("sionna-rt") != "2.0.1" or not mi.variant().startswith("cuda"):
            raise RuntimeError(f"expected Sionna RT 2.0.1/CUDA Mitsuba, got {mi.variant()}")
        for case in cases:
            info_path = args.root / "scenes" / case["map_id"] / "scene.xml"
            if not info_path.is_file():
                raise RuntimeError(f"scene build missing: {info_path}")
            scene_info = json.loads((args.root / "scene_manifest.json").read_text())[case["id"]]
            origin_key = "origin_projected_m" if "origin_projected_m" in scene_info else "origin_m"
            case = {**case, "_origin": np.asarray(scene_info[origin_key], dtype=np.float64),
                    "_scene_info": scene_info}
            for count in (64, 128, 256):
                run["cases"].append(trace_one(rt, dr, args.root, case, count, args.out))
    run["status"] = "complete"
    run["finished_unix"] = time.time()
    result = args.out / ("geometry_preflight.json" if args.build_only else "trace_manifest.json")
    tmp = result.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run, indent=2, allow_nan=False)+"\n")
    tmp.replace(result)
    print(json.dumps({"status": run["status"], "outputs": len(run["cases"]),
                      "manifest": str(result)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
