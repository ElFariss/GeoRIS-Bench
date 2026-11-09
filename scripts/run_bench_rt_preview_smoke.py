#!/usr/bin/env python3
"""Build two pinned development LoD2 scenes and run bounded Sionna RT probes.

Raw inputs and path arrays stay in the remote workspace. The script emits a
compact summary beside them; copy only that summary back to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import resource
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
try:
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover - optional when importing CIR mask helpers in unit tests
    constrained_delaunay_triangles = Polygon = box = unary_union = None

MUNICH = {
    "file": "690_5334.gml",
    "url": "https://download1.bayernwolke.de/a/lod2/citygml/690_5334.gml",
    "bytes": 161632238,
    "sha256": "232c75a398ec7ee81736aeb72f7eda374f88b9431f8c35cdbee18a51ff029e87",
}
MAASTRICHT = [
    ("10-648-40.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/648/40/10-648-40.city.json.gz", 456952, "738ee601f609ccfb77fe79d6607ed25d38c883855e83a902bc0b081cbe150e02"),
    ("10-648-42.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/648/42/10-648-42.city.json.gz", 1002425, "931ada9bf68dd7b20bcd1f11de6c00f1db25ea6a3b03163b63bb14dbd9093f57"),
    ("10-650-40.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/650/40/10-650-40.city.json.gz", 604852, "6c3a5ff84fb56d5e6a0caef68d2d2287995fdff736cc8a761fdc5b83cbb36cd3"),
    ("10-650-42.city.json.gz", "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/650/42/10-650-42.city.json.gz", 1758014, "eb79c0f71c859d3ebf6578824e644226d860c9ac08179a16bcb4a02b1e8a3348"),
]
MAASTRICHT_ORIGIN = np.array([175967., 316952., 0.])
MATERIAL = {"WallSurface": "brick", "RoofSurface": "concrete", "GroundSurface": "concrete"}
FALLBACK = "concrete"
FREQ = 5.53e9
PITCH_H = .028
PITCH_V = .030
SIZES = ((8, 8), (8, 16), (16, 16))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(path: Path, url: str, nbytes: int, digest: str) -> None:
    if path.exists() and path.stat().st_size == nbytes and sha(path) == digest:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "atom-wise-ris-bench-smoke/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    if tmp.stat().st_size != nbytes or sha(tmp) != digest:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"pinned input integrity mismatch: {path.name}")
    tmp.replace(path)


def munich_rings(poly: ET.Element) -> list[np.ndarray] | None:
    ring_elems = [poly.find("./{*}exterior/{*}LinearRing"), *poly.findall("./{*}interior/{*}LinearRing")]
    if ring_elems[0] is None:
        return None
    result = []
    for elem in ring_elems:
        text = elem.findtext("./{*}posList")
        if text is None:
            text = " ".join(v.text or "" for v in elem.findall("./{*}pos"))
        vals = np.fromstring(text, sep=" ")
        if vals.size % 3:
            raise ValueError("CityGML coordinate tuple is not 3D")
        xyz = vals.reshape(-1, 3)
        if len(xyz) > 1 and np.array_equal(xyz[0], xyz[-1]):
            xyz = xyz[:-1]
        if len(xyz):
            xyz = xyz[np.r_[True, np.any(np.diff(xyz, axis=0) != 0, axis=1)]]
        result.append(xyz)
    return result


def orient_triangle_to_normal(triangle: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Ensure 3D triangle winding matches the face's outward Newell normal."""
    tri = np.asarray(triangle, dtype=float)
    cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if float(np.dot(cross, normal)) < 0.0:
        return tri[[0, 2, 1]].copy()
    return tri


def munich_triangles(rings: list[np.ndarray]) -> list[np.ndarray]:
    if len(rings[0]) < 3:
        return []
    outer = rings[0]
    rel = outer - outer[0]
    normal = np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0)
    drop = int(np.argmax(np.abs(normal)))
    if abs(normal[drop]) < 1e-12:
        return []
    keep = [i for i in range(3) if i != drop]
    valid = [r for r in rings if len(r) >= 3 and Polygon(r[:, keep]).area > 0]
    if not valid:
        return []
    poly = Polygon(valid[0][:, keep], [r[:, keep] for r in valid[1:]])
    if not poly.is_valid:
        raise ValueError("invalid native Munich semantic face")
    point_map = {tuple(p[keep]): p for ring in valid for p in ring}
    pieces = constrained_delaunay_triangles(poly).geoms
    if abs(sum(t.area for t in pieces) - poly.area) > 1e-8 * max(1., poly.area):
        raise ValueError("Munich face triangulation did not preserve projected area")
    result = []
    for tri in pieces:
        xyz = np.stack([point_map[tuple(p)] for p in list(tri.exterior.coords)[:3]])
        result.append(orient_triangle_to_normal(xyz, normal))
    return result


def write_ply(path: Path, triangles: list[np.ndarray], origin: np.ndarray) -> None:
    verts = np.asarray(triangles, dtype=float).reshape(-1, 3, 3) - origin
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {verts.size // 3}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(verts)}\nproperty list uchar int vertex_indices\nend_header\n")
        np.savetxt(f, verts.reshape(-1, 3), fmt="%.6f")
        for i in range(len(verts)):
            f.write(f"3 {3*i} {3*i+1} {3*i+2}\n")


def write_scene(out: Path, classes: dict[str, list[np.ndarray]], origin: np.ndarray) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    material_for = lambda k: k if k in {"concrete", "brick", "glass"} else MATERIAL.get(k, FALLBACK)
    mats = sorted({material_for(k) for k, v in classes.items() if v})
    xml = ['<scene version="2.1.0">']
    counts = {}
    all_triangles = []
    for mat in mats:
        xml.append(f'<bsdf type="itu-radio-material" id="mat-{mat}"><string name="type" value="{mat}"/><float name="thickness" value="0.1"/></bsdf>')
    for klass, triangles in sorted(classes.items()):
        if not triangles:
            continue
        file = f"{klass}.ply"
        write_ply(out / file, triangles, origin)
        all_triangles.extend(triangles)
        xml.append(f'<shape type="ply" id="surface-{klass}"><string name="filename" value="{file}"/><ref id="mat-{material_for(klass)}" name="bsdf"/></shape>')
        counts[klass] = len(triangles)
    xml.append("</scene>")
    scene = out / "scene.xml"
    scene.write_text("\n".join(xml) + "\n")
    mesh_vertices = np.asarray(all_triangles, dtype=float).reshape(-1, 3, 3) - origin
    return {"origin_m": origin.tolist(), "triangles_by_semantic": counts,
            "mesh_bounds_local_m": [mesh_vertices.reshape(-1, 3).min(axis=0).tolist(),
                                     mesh_vertices.reshape(-1, 3).max(axis=0).tolist()],
            "scene_sha256": sha(scene), "mesh_sha256": {p.name: sha(p) for p in sorted(out.glob("*.ply"))}}


def validate_case_geometry(scene_info: dict, case: dict) -> dict:
    """Validate source scene extents and every planned probe/user coordinate."""
    lower, upper = [np.asarray(x, dtype=float) for x in scene_info["mesh_bounds_local_m"]]
    origin = np.asarray(scene_info["origin_m"], dtype=float)
    bs = np.asarray(case["bs_xyz"], dtype=float)
    ris = np.asarray(case["ris_xyz"], dtype=float)
    users = np.asarray(case["served_users_xyz"], dtype=float).reshape(-1, 3)
    probes_by_size = {f"{r}x{c}": layout(case, r, c) for r, c in SIZES}
    all_points = np.vstack([bs, ris, users, *probes_by_size.values()]) - origin
    if (not np.isfinite(np.concatenate([lower, upper, all_points.reshape(-1)])).all() or
            np.any(upper <= lower)):
        raise RuntimeError("non-finite/empty mesh extent or probe coordinate")
    xy_inside = ((all_points[:, :2] >= lower[:2]-1e-6) &
                 (all_points[:, :2] <= upper[:2]+1e-6)).all(axis=1)
    z_inside = ((all_points[:, 2] >= lower[2] - 5.0) &
                (all_points[:, 2] <= upper[2] + 15.0)).all()
    if not xy_inside.all() or np.any(all_points[:, 2] <= 0) or not z_inside:
        raise RuntimeError("BS/RIS/user/probe coordinate lies outside RF mesh XY/Z extent or local scene Z")
    return {"mesh_bounds_local_m": [lower.tolist(), upper.tolist()],
            "checked_point_count": int(len(all_points)), "all_xy_inside_mesh_bounds": True,
            "all_z_inside_mesh_band": True,
            "all_local_z_positive": True,
            "array_probe_counts": {k: len(v) for k, v in probes_by_size.items()},
            "array_probe_bounds_local_m": {k: [
                (v-origin).min(axis=0).tolist(), (v-origin).max(axis=0).tolist()]
                for k, v in probes_by_size.items()}}


def build_munich(raw: Path, out: Path, case: dict) -> dict:
    source = raw / MUNICH["file"]
    # Fixed margin around the selected BS, RIS and served v1 preview UEs.
    anchors = np.asarray([case["bs_xyz"], case["ris_xyz"], *case["served_users_xyz"]], dtype=float)
    bounds = (float(anchors[:, 0].min() - 50), float(anchors[:, 1].min() - 50),
              float(anchors[:, 0].max() + 50), float(anchors[:, 1].max() + 50))
    patch = box(*bounds).buffer(35)
    origin = np.array([bounds[0], bounds[1], float(anchors[:, 2].min() - 40.)])
    classes: dict[str, list[np.ndarray]] = {k: [] for k in MATERIAL}
    buildings = 0
    member_count = 0
    it = ET.iterparse(source, events=("start", "end"))
    _, root = next(it)
    for event, elem in it:
        if event != "end" or elem.tag.rsplit("}", 1)[-1] != "cityObjectMember":
            continue
        building = elem.find("./{*}Building")
        if building is None:
            elem.clear()
            continue
        ground = [rs for surf in building.findall(".//{*}GroundSurface")
                  for poly in surf.findall(".//{*}Polygon") if (rs := munich_rings(poly)) is not None]
        footprints = [Polygon(rs[0][:, :2], [r[:, :2] for r in rs[1:]])
                      for rs in ground if len(rs[0]) >= 3]
        if not any(g.area > 0 and g.intersects(patch) for g in footprints):
            elem.clear()
            continue
        buildings += 1
        for key in MATERIAL:
            for surface in building.findall(f".//{{*}}{key}"):
                for poly in surface.findall(".//{*}Polygon"):
                    rings = munich_rings(poly)
                    tris = munich_triangles(rings) if rings else []
                    classes[key].extend(tris)
                    member_count += len(tris)
        elem.clear()
        if len(root) > 500:
            root.clear()
    if not buildings or not any(classes.values()):
        raise RuntimeError("selected preview case has no native LoD2 in fixed smoke extent")
    result = write_scene(out, classes, origin)
    return {"case_id": case["id"], "map": "munich", "source": MUNICH,
            "scene_bounds_epsg25832_m": list(bounds), "building_count": buildings,
            "semantic_triangle_count": member_count, **result}


def build_maastricht(raw: Path, out: Path, case: dict) -> dict:
    # Use the already audited pinned 3DBAG decoder and constrained surface
    # triangulation, but only create the RF meshes; DTM remains outside RF.
    import sys
    repo_scripts = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_scripts))
    import build_bench_phase0b_scene as maas

    features = []
    for filename, _, _, _ in MAASTRICHT:
        import gzip
        with gzip.open(raw / filename, "rt", encoding="utf-8") as f:
            features.append(json.load(f))
    skipped_zero_area_faces = 0
    for feature in features:
        vertices = maas.decode_vertices(feature)
        for city_object in feature.get("CityObjects", {}).values():
            for geom in city_object.get("geometry", []):
                if geom.get("lod") != "2.2" or geom.get("type") != "Solid":
                    continue
                boundaries = geom.get("boundaries", [])
                semantic = geom.get("semantics", {})
                values = semantic.get("values", [])
                nested = bool(values and isinstance(values[0], list))
                kept_boundaries, kept_values = [], []
                for shell_idx, shell in enumerate(boundaries):
                    semantic_shell = values[shell_idx] if nested and shell_idx < len(values) else values
                    new_shell, new_semantic = [], []
                    for face_idx, face in enumerate(shell):
                        surface_id = semantic_shell[face_idx] if face_idx < len(semantic_shell) else None
                        degenerate = False
                        for ring in face:
                            points = vertices[np.asarray(ring, dtype=int), :3]
                            if len(np.unique(points, axis=0)) < 3:
                                degenerate = True
                                break
                        if degenerate:
                            skipped_zero_area_faces += 1
                            continue
                        new_shell.append(face)
                        new_semantic.append(surface_id)
                    kept_boundaries.append(new_shell)
                    kept_values.append(new_semantic)
                geom["boundaries"] = kept_boundaries
                if nested:
                    semantic["values"] = kept_values
                else:
                    semantic["values"] = kept_values[0] if kept_values else []
    bounds = maas.BOUNDS
    original_triangulator = maas.triangulate_semantic_face
    approximated_faces = []
    repaired_faces = []

    def local_origin_triangulator(rings, vertices_world, **kwargs):
        """Recenter CRS coordinates and preserve warped native boundaries."""
        vertices_world = np.asarray(vertices_world, dtype=float)
        offset = vertices_world[int(rings[0][0])].copy()
        local_vertices = vertices_world - offset

        def canonical(points):
            points = np.asarray(points, dtype=float)
            if len(points) > 1 and np.allclose(points[0], points[-1], rtol=0, atol=1e-9):
                points = points[:-1]
            if len(points):
                points = points[np.r_[True, np.any(np.abs(np.diff(points, axis=0)) > 1e-9, axis=1)]]
            return points

        xyz_rings = [canonical(local_vertices[np.asarray(r, dtype=int), :3]) for r in rings]
        if not xyz_rings or any(len(r) < 3 for r in xyz_rings):
            raise ValueError("native face has a ring with fewer than three distinct 3D points")
        outer = xyz_rings[0]
        normal = np.zeros(3, dtype=float)
        for p, q in zip(outer, np.roll(outer, -1, axis=0)):
            normal += np.array([(p[1]-q[1])*(p[2]+q[2]),
                                (p[2]-q[2])*(p[0]+q[0]),
                                (p[0]-q[0])*(p[1]+q[1])])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("native face has degenerate 3D boundary")
        normal /= norm
        plane_error = float(max(np.max(np.abs((r-outer[0]) @ normal)) for r in xyz_rings))
        try:
            local_triangles, local_polygon = original_triangulator(rings, local_vertices, **kwargs)
        except ValueError as exc:
            if ("non-planar/ill-conditioned face" not in str(exc) and
                    "invalid projected face polygon" not in str(exc)):
                raise
            # Declared RF approximation: warped polygons become piecewise-planar
            # triangles. Self-intersecting projections use GEOS make_valid;
            # newly introduced crossing points are lifted along source edges.
            import shapely
            drop = int(np.argmax(np.abs(normal)))
            if abs(normal[drop]) < 1e-8:
                raise ValueError("native face dominant-plane projection is ill-conditioned") from exc
            keep = [axis for axis in range(3) if axis != drop]
            projected_rings = [r[:, keep] for r in xyz_rings]
            polygon_local = Polygon(projected_rings[0], projected_rings[1:])
            repaired = not polygon_local.is_valid
            domain = shapely.make_valid(polygon_local) if repaired else polygon_local
            polygons = ([domain] if domain.geom_type == "Polygon" else
                        [g for g in getattr(domain, "geoms", []) if g.geom_type == "Polygon"])
            domain_area = float(sum(p.area for p in polygons))
            if not polygons or domain_area <= 0:
                raise ValueError("native face has no positive-area projected domain") from exc
            pieces = [tri for poly in polygons
                      for tri in shapely.constrained_delaunay_triangles(poly).geoms
                      if tri.geom_type == "Polygon" and tri.area > 0]
            covered = shapely.union_all(pieces) if pieces else Polygon()
            projected_domain = shapely.union_all(polygons)
            if (not pieces or abs(sum(t.area for t in pieces)-domain_area) > 1e-8*max(1., domain_area) or
                    projected_domain.symmetric_difference(covered).area > 1e-8*max(1., domain_area)):
                raise ValueError("native face triangulation failed projected coverage") from exc

            def lift(point):
                q = np.asarray(point, dtype=float)
                best = None
                for ring in xyz_rings:
                    ring2 = ring[:, keep]
                    for i, (a, b) in enumerate(zip(ring2, np.roll(ring2, -1, axis=0))):
                        v = b-a
                        den = float(v @ v)
                        t = 0. if den == 0 else float(np.clip((q-a) @ v/den, 0., 1.))
                        d = q-(a+t*v)
                        d2 = float(d @ d)
                        if best is None or d2 < best[0]:
                            best = (d2, (1-t)*ring[i] + t*ring[(i+1) % len(ring)])
                if best is None or best[0] > 1e-8:
                    raise ValueError("GEOS repair introduced an unliftable projected vertex")
                return best[1]

            local_triangles = [np.stack([lift(p) for p in np.asarray(tri.exterior.coords[:-1])])
                               for tri in pieces]
            source_xyz = np.concatenate(xyz_rings, axis=0)
            output_xyz = np.concatenate(local_triangles, axis=0)
            if not all(np.min(np.linalg.norm(output_xyz-p, axis=1)) <= 1e-8
                       for p in source_xyz):
                raise ValueError("triangulation failed to retain a native source vertex") from exc
            rec = {"plane_deviation_m": plane_error, "projected_area_m2": float(polygon_local.area),
                   "topology_repaired": repaired, "rings": len(xyz_rings),
                   "source_vertex_count": int(sum(map(len, xyz_rings))),
                   "triangle_count": len(local_triangles)}
            approximated_faces.append(rec)
            if repaired:
                repaired_faces.append(rec)
            local_polygon = polygon_local

        oriented_triangles = []
        for triangle in local_triangles:
            tri = orient_triangle_to_normal(triangle, normal)
            area3d = .5*float(np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])))
            if not np.isfinite(tri).all() or area3d <= 1e-12:
                raise ValueError("triangulation emitted non-finite or zero-area 3D triangle")
            oriented_triangles.append(tri + offset)
        outer_world = vertices_world[np.asarray(rings[0], dtype=int), :3]
        n = np.zeros(3, dtype=float)
        for p, q in zip(outer_world, np.roll(outer_world, -1, axis=0)):
            n += np.array([(p[1]-q[1])*(p[2]+q[2]),
                           (p[2]-q[2])*(p[0]+q[0]),
                           (p[0]-q[0])*(p[1]+q[1])])
        drop = int(np.argmax(np.abs(n)))
        keep = [axis for axis in range(3) if axis != drop]
        from shapely.affinity import translate
        # Phase0b polygon_surfaces consumes the footprint relative to the
        # scene origin, then adds that origin back for its world-space clip.
        polygon = translate(local_polygon,
                            xoff=float(offset[keep[0]]-MAASTRICHT_ORIGIN[keep[0]]),
                            yoff=float(offset[keep[1]]-MAASTRICHT_ORIGIN[keep[1]]))
        return oriented_triangles, polygon

    maas.triangulate_semantic_face = local_origin_triangulator
    try:
        classes, _, heights, surface_counts, outside, bad = maas.polygon_surfaces(
            features, MAASTRICHT_ORIGIN, bounds)
    finally:
        maas.triangulate_semantic_face = original_triangulator
    if not any(classes.values()):
        raise RuntimeError("pinned Maastricht tiles contained no native LoD2 surfaces")
    # maas.polygon_surfaces already subtracted MAASTRICHT_ORIGIN once; restore
    # world coordinates before write_scene subtracts origin so it is applied once.
    world_classes = {k: [tri + MAASTRICHT_ORIGIN for tri in tris] for k, tris in classes.items()}
    result = write_scene(out, world_classes, MAASTRICHT_ORIGIN)
    return {"case_id": case["id"], "map": "maastricht", "source_files":
            [{"file": n, "sha256": h} for n, _, _, h in MAASTRICHT],
            "scene_bounds_epsg7415_m": list(bounds), "buildingpart_height_source_counts": heights,
            "surface_counts": surface_counts, "parts_outside_scene": outside,
            "degenerate_zero_area_faces_skipped": skipped_zero_area_faces,
            "warped_or_invalid_faces_triangulated": len(approximated_faces),
            "approximated_faces_preserve_source_vertices": True,
            "invalid_projected_faces_repaired": len(repaired_faces),
            "approximated_face_max_plane_deviation_m": max(
                (f["plane_deviation_m"] for f in approximated_faces), default=0.),
            "invalid_ground_surfaces": bad, **result}


def gpu_state() -> list[dict]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "memory_total_mib": total, "memory_used_mib": used,
                     "memory_free_mib": free, "utilization_pct": util})
    return rows


def layout(case: dict, nrows: int, ncols: int) -> np.ndarray:
    bs = np.asarray(case["bs_xyz"], dtype=float)
    center = np.asarray(case["ris_xyz"], dtype=float)
    normal = bs[:2] - center[:2]
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    points = []
    for row in range(nrows):
        for col in range(ncols):
            xy = center[:2] + .05 * normal + (col - (ncols - 1) / 2) * PITCH_H * tangent
            z = center[2] + (row - (nrows - 1) / 2) * PITCH_V
            points.append([xy[0], xy[1], z])
    return np.asarray(points)


def arrays(obj) -> np.ndarray | None:
    if obj is None:
        return None
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    try:
        return np.asarray(obj)
    except Exception:
        return None


def reduce_cir_links(coeff: np.ndarray, receiver_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a 6-D Sionna CIR array [rx, rx_pol, tx, tx_pol, path, time] to per-receiver masks."""
    coeff = np.asarray(coeff)
    if (coeff.ndim != 6 or coeff.shape[0] != receiver_count
            or coeff.shape[2] < 1 or coeff.shape[5] < 1):
        raise RuntimeError(f"unexpected path coefficient shape: {coeff.shape}")
    links = coeff[:, :, 0, :, :, 0]
    num_paths = int(links.shape[-1])
    path_mask = (np.any(np.abs(links) > 0, axis=(1, 2))
                 if num_paths > 0 else np.zeros((receiver_count, 0), dtype=bool))
    nonzero_receiver_mask = np.any(path_mask, axis=-1)
    if path_mask.shape != (receiver_count, num_paths) or nonzero_receiver_mask.shape != (receiver_count,):
        raise RuntimeError(f"unexpected reduced mask shapes: {path_mask.shape}, {nonzero_receiver_mask.shape}")
    return links, path_mask, nonzero_receiver_mask


def compute_exclusive_link_classes(
    los_mask: np.ndarray,
    reflection_mask: np.ndarray,
    diffraction_mask: np.ndarray,
    ris_count: int,
    n_users: int,
) -> dict[str, dict[str, int]]:
    """Classify each receiver link into mutually exclusive propagation classes."""
    expected_shape = (ris_count + n_users,)
    los_b = np.asarray(los_mask, dtype=bool)
    ref_b = np.asarray(reflection_mask, dtype=bool)
    dif_b = np.asarray(diffraction_mask, dtype=bool)
    if los_b.shape != expected_shape or ref_b.shape != expected_shape or dif_b.shape != expected_shape:
        raise ValueError(
            f"link class masks must all have shape {expected_shape}; "
            f"got {los_b.shape}, {ref_b.shape}, {dif_b.shape}"
        )
    mixed_mask = (los_b & (ref_b | dif_b)) | (ref_b & dif_b)
    class_masks = {
        "LOS_present": los_b & ~mixed_mask,
        "reflection_only": ~los_b & ref_b & ~dif_b,
        "diffraction_only": ~los_b & ~ref_b & dif_b,
        "mixed_interactions": mixed_mask,
        "fully_blocked": ~los_b & ~ref_b & ~dif_b,
    }
    return {
        label: {
            "ris_elements": int(mask[:ris_count].sum()),
            "served_users": int(mask[ris_count:ris_count + n_users].sum()),
        }
        for label, mask in class_masks.items()
    }


def solve_case(rt, dr, scene_path: Path, case: dict, origin: np.ndarray,
               receivers: np.ndarray, ris_count: int, out_dir: Path, mode: str) -> dict:
    scene = rt.load_scene(str(scene_path), merge_shapes=False)
    scene.frequency = FREQ
    scene.tx_array = scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
        pattern="iso", polarization="VH")
    bs = (np.asarray(case["bs_xyz"], dtype=float) - origin).tolist()
    rx_local = receivers - origin
    scene.add(rt.Transmitter("tx", position=bs))
    for i, point in enumerate(rx_local):
        scene.add(rt.Receiver(f"probe_{i:03d}", position=point.tolist()))
    modes = {
        "combined": dict(los=True, specular_reflection=True, diffraction=True),
        "los_only": dict(los=True, specular_reflection=False, diffraction=False),
        "reflection_only": dict(los=False, specular_reflection=True, diffraction=False),
        "diffraction_only": dict(los=False, specular_reflection=False, diffraction=True),
    }
    params = dict(max_depth=2, diffuse_reflection=False, refraction=False,
                  synthetic_array=False, samples_per_src=10000,
                  max_num_paths_per_src=40000, seed=7, **modes[mode])
    t0 = time.perf_counter()
    paths = rt.PathSolver()(scene=scene, **params)
    coeff, delays = paths.cir(normalize_delays=False, out_type="numpy")
    dr.sync_thread()
    elapsed = time.perf_counter() - t0
    coeff = np.asarray(coeff)
    delays = np.asarray(delays)
    links, path_mask, nonzero_receiver_mask = reduce_cir_links(coeff, len(receivers))
    count = int(nonzero_receiver_mask.sum())
    max_paths_per_link = int(links.shape[-1])
    info = {"mode": mode, "receiver_count": len(receivers), "ris_probe_count": ris_count,
            "served_user_count": len(receivers) - ris_count, "cir_shape": list(coeff.shape),
            "delay_shape": list(delays.shape), "elapsed_seconds": elapsed,
            "links_with_nonzero_paths": count, "nonzero_path_count": int(path_mask.sum()),
            "ris_links_with_nonzero_paths": int(nonzero_receiver_mask[:ris_count].sum()),
            "served_user_links_with_nonzero_paths": int(nonzero_receiver_mask[ris_count:].sum()),
            "nonzero_receiver_mask": nonzero_receiver_mask.tolist(),
            "fully_blocked_link_count": int(len(receivers) - count),
            "max_paths_axis": max_paths_per_link,
            "finite_coefficients": bool(np.isfinite(coeff.real).all() and np.isfinite(coeff.imag).all()),
            "finite_delays": bool(np.isfinite(delays).all()),
            "max_abs_coefficient": float(np.max(np.abs(coeff))) if coeff.size else 0.,
            "solver_parameters": params}
    if mode == "combined":
        path_records = []
        phi_t, theta_t = arrays(getattr(paths, "phi_t", None)), arrays(getattr(paths, "theta_t", None))
        phi_r, theta_r = arrays(getattr(paths, "phi_r", None)), arrays(getattr(paths, "theta_r", None))
        angle_shapes = {name: list(a.shape) if a is not None else None for name, a in
                        (("phi_t", phi_t), ("theta_t", theta_t), ("phi_r", phi_r), ("theta_r", theta_r))}

        def receiver_path_value(arr: np.ndarray | None, rx_index: int, path_index: int):
            if arr is None or arr.ndim < 2 or arr.shape[0] != len(receivers):
                return None
            if arr.ndim == 2:
                indices = (rx_index, path_index)
            elif arr.ndim == 3:
                indices = (rx_index, 0, path_index)
            elif arr.ndim == 5:
                indices = (rx_index, 0, 0, 0, path_index)
            elif arr.ndim == 7:
                indices = (rx_index, 0, 0, 0, 0, 0, path_index)
            else:
                return None
            if any(i >= arr.shape[d] for d, i in enumerate(indices)):
                return None
            return float(arr[indices])

        # Keep only nonzero receiver/path pairs; path arrays remain on this host.
        for ri in range(len(receivers)):
            for pi in range(links.shape[-1]):
                h = links[ri, :, :, pi]
                if not np.any(np.abs(h) > 0):
                    continue
                record = {"rx_index": ri,
                          "rx_type": "ris_element" if ri < ris_count else "served_user",
                          "rx_local_id": ri if ri < ris_count else ri - ris_count,
                          "served_user_id": (case["served_users"][ri-ris_count]["user_id"]
                                             if ri >= ris_count else None),
                          "path_index": pi,
                          "coefficient_matrix_real": h.real.tolist(),
                          "coefficient_matrix_imag": h.imag.tolist(),
                          "delay_s": receiver_path_value(delays, ri, pi)}
                for name, arr in (("phi_t_rad", phi_t), ("theta_t_rad", theta_t),
                                  ("phi_r_rad", phi_r), ("theta_r_rad", theta_r)):
                    record[name] = receiver_path_value(arr, ri, pi)
                path_records.append(record)
        pathfile = out_dir / f"{case['id']}-{mode}-n{ris_count}-paths.jsonl"
        with pathfile.open("w") as f:
            for record in path_records:
                f.write(json.dumps(record, allow_nan=False) + "\n")
        info["path_record_file"] = pathfile.name
        info["path_record_count"] = len(path_records)
        info["path_record_sha256"] = sha(pathfile)
        info["angle_array_shapes"] = angle_shapes
        info["delay_array_shape_for_path_metadata"] = list(delays.shape)
        info["receiver_specific_path_metadata"] = bool(
            delays.ndim in (2, 3, 5, 7) and delays.shape[0] == len(receivers)
            and all(a is not None and a.ndim in (2, 3, 5, 7) and a.shape[0] == len(receivers)
                    for a in (phi_t, theta_t, phi_r, theta_r)))
    return info


def trace_scene(rt, dr, mi, scene_info: dict, case: dict, root: Path, out: Path) -> dict:
    origin = np.asarray(scene_info["origin_m"], dtype=float)
    per_size = {}
    users = np.asarray(case["served_users_xyz"], dtype=float)
    for nrows, ncols in SIZES:
        centers = layout(case, nrows, ncols)
        # A perimeter/case extent diagnostic guards against malformed geometry.
        if not np.isfinite(centers).all() or np.max(np.linalg.norm(centers[:, :2] - centers[0, :2], axis=1)) > 1:
            raise RuntimeError("invalid NEC-lattice probe coordinates")
        state_before = gpu_state()
        if max(g["memory_free_mib"] for g in state_before) < 2048:
            raise RuntimeError("GPU stop rule: no device has 2 GiB free")
        receivers = np.vstack([centers, users])
        combined = solve_case(rt, dr, root / "scene.xml", case, origin, receivers,
                              len(centers), out, "combined")
        state_after = gpu_state()
        if not combined["finite_coefficients"] or not combined["finite_delays"]:
            raise RuntimeError("non-finite Sionna path result")
        per_size[f"{nrows}x{ncols}"] = {
            "element_count": len(centers), "pitch_h_m": PITCH_H,
            "pitch_v_m": PITCH_V, "panel_center_offset_toward_bs_m": .05,
            "combined": combined,
            "gpu_before": state_before, "gpu_after": state_after,
        }
    # Interaction-isolated counts are measured on the 64-element lattice.
    centers = layout(case, *SIZES[0])
    receivers = np.vstack([centers, users])
    class_counts = {}
    for mode in ("los_only", "reflection_only", "diffraction_only"):
        before = gpu_state()
        if max(g["memory_free_mib"] for g in before) < 2048:
            raise RuntimeError("GPU stop rule: no device has 2 GiB free")
        class_counts[mode] = solve_case(rt, dr, root / "scene.xml", case, origin,
                                        receivers, len(centers), out, mode)
    for key, result in per_size.items():
        if key == "8x8":
            exclusive_class_counts = compute_exclusive_link_classes(
                class_counts["los_only"]["nonzero_receiver_mask"],
                class_counts["reflection_only"]["nonzero_receiver_mask"],
                class_counts["diffraction_only"]["nonzero_receiver_mask"],
                ris_count=64,
                n_users=len(users),
            )
        else:
            exclusive_class_counts = None
        result["link_class_counts"] = {
            "combined_nonblocked": result["combined"]["links_with_nonzero_paths"],
            "fully_blocked": result["combined"]["fully_blocked_link_count"],
            "los_only_nonzero": class_counts["los_only"]["links_with_nonzero_paths"] if key == "8x8" else None,
            "reflection_only_nonzero": class_counts["reflection_only"]["links_with_nonzero_paths"] if key == "8x8" else None,
            "diffraction_only_nonzero": class_counts["diffraction_only"]["links_with_nonzero_paths"] if key == "8x8" else None,
            "isolated_classes_measured_at_64_elements": key == "8x8",
            "exclusive_classes_from_64_element_isolated_receiver_masks": exclusive_class_counts,
        }
        result["link_class_counts"]["los_only_served_users"] = (
            class_counts["los_only"]["served_user_links_with_nonzero_paths"] if key == "8x8" else None)
        result["link_class_counts"]["reflection_only_served_users"] = (
            class_counts["reflection_only"]["served_user_links_with_nonzero_paths"] if key == "8x8" else None)
        result["link_class_counts"]["diffraction_only_served_users"] = (
            class_counts["diffraction_only"]["served_user_links_with_nonzero_paths"] if key == "8x8" else None)
    return {"case_id": case["id"], "map": case["map_id"], "native_scene": scene_info,
            "mitsuba_shape_ids": None, "array_smokes": per_size,
            "interaction_isolations_64": class_counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", "--workspace", dest="root", type=Path, required=True)
    ap.add_argument("--case-manifest", "--cases", dest="case_manifest", type=Path, required=True)
    ap.add_argument("--out", "--output-dir", dest="out", type=Path, required=True)
    ap.add_argument("--build-only", action="store_true",
                    help="build and validate native scenes without importing Sionna or solving paths")
    ap.add_argument("--reuse-scenes", action="store_true",
                    help="load meshes from a prior geometry preflight and verify their hashes")
    args = ap.parse_args()
    root, raw, out = args.root, args.root / "raw", args.out
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.case_manifest.read_text())
    for case in manifest:
        case["served_users_xyz"] = [u["xyz"] for u in case["served_users"]]
    expected = {"01-munich-natural", "39-maastricht-natural"}
    if {c["id"] for c in manifest} != expected or {c["map_id"] for c in manifest} != {"munich", "maastricht"}:
        raise ValueError("unexpected or non-development smoke case manifest")
    t0 = time.perf_counter()
    download(raw / MUNICH["file"], MUNICH["url"], MUNICH["bytes"], MUNICH["sha256"])
    for filename, url, nbytes, digest in MAASTRICHT:
        download(raw / filename, url, nbytes, digest)
    inputs = {p.name: sha(p) for p in sorted(raw.iterdir()) if p.is_file() and not p.name.endswith(".part")}
    if inputs.get(MUNICH["file"]) != MUNICH["sha256"]:
        raise RuntimeError("Munich pinned source hash mismatch")
    for name, _, _, digest in MAASTRICHT:
        if inputs.get(name) != digest:
            raise RuntimeError(f"3DBAG pinned source hash mismatch: {name}")
    if args.build_only:
        built = []
        for case in manifest:
            scene_dir = root / "scenes" / case["map_id"]
            scene_info = (build_munich(raw, scene_dir, case) if case["map_id"] == "munich"
                          else build_maastricht(raw, scene_dir, case))
            scene_info["geometry_preflight"] = validate_case_geometry(scene_info, case)
            built.append(scene_info)
        summary = {"status": "geometry_preflight_complete", "case_ids": [c["id"] for c in manifest],
                   "source_hashes": inputs, "terrain_in_rf_scene": False,
                   "build_seconds": time.perf_counter()-t0, "scenes": built}
        path = out / "geometry_preflight.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        tmp.replace(path)
        preflight_dir = root / "preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        preflight_path = preflight_dir / "geometry_preflight.json"
        if preflight_path.resolve() != path.resolve():
            ptmp = preflight_path.with_suffix(".json.tmp")
            ptmp.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
            ptmp.replace(preflight_path)
        print(json.dumps({"status": summary["status"], "case_ids": summary["case_ids"],
                          "build_seconds": summary["build_seconds"],
                          "maastricht_approximated_faces": next(
                              s.get("warped_or_invalid_faces_triangulated", 0) for s in built
                              if s["map"] == "maastricht"),
                          "maastricht_max_plane_deviation_m": next(
                              s.get("approximated_face_max_plane_deviation_m", 0.) for s in built
                              if s["map"] == "maastricht")}, indent=2), flush=True)
        return
    import drjit as dr
    import mitsuba as mi
    import sionna.rt as rt
    if importlib.metadata.version("sionna-rt") != "2.0.1" or not mi.variant().startswith("cuda"):
        raise RuntimeError(f"unexpected Sionna/Mitsuba variant: {mi.variant()}")
    gpu_start = gpu_state()
    if max(g["memory_free_mib"] for g in gpu_start) < 2048:
        raise RuntimeError("GPU stop rule: no device has 2 GiB free")
    prebuilt = {}
    if args.reuse_scenes:
        manifest_path = root / "preflight" / "geometry_preflight.json"
        if not manifest_path.is_file():
            manifest_path = out / "geometry_preflight.json"
        geometry = json.loads(manifest_path.read_text())
        if geometry.get("status") != "geometry_preflight_complete":
            raise RuntimeError("reused scene geometry manifest is not complete")
        prebuilt = {s["case_id"]: s for s in geometry["scenes"]}
        if set(prebuilt) != {c["id"] for c in manifest}:
            raise RuntimeError("reused scene set does not match development case manifest")
    results = []
    for case in manifest:
        scene_dir = root / "scenes" / case["map_id"]
        if args.reuse_scenes:
            scene_info = prebuilt[case["id"]]
            if scene_info["map"] != case["map_id"]:
                raise RuntimeError("reused scene map ID does not match case")
            if sha(scene_dir / "scene.xml") != scene_info["scene_sha256"]:
                raise RuntimeError("reused native scene XML hash mismatch")
            for filename, digest in scene_info["mesh_sha256"].items():
                if sha(scene_dir / filename) != digest:
                    raise RuntimeError(f"reused native mesh hash mismatch: {filename}")
        elif case["map_id"] == "munich":
            scene_info = build_munich(raw, scene_dir, case)
        else:
            scene_info = build_maastricht(raw, scene_dir, case)
        geometry_check = validate_case_geometry(scene_info, case)
        shape_scene = rt.load_scene(str(scene_dir / "scene.xml"), merge_shapes=False)
        shape_ids = [shape.id() for shape in shape_scene.mi_scene.shapes()]
        shape_scene_meta = {"shape_count": len(shape_ids), "ids": shape_ids}
        if not shape_ids:
            raise RuntimeError("empty RF mesh inventory")
        origin = np.asarray(scene_info["origin_m"], dtype=float)
        points = np.vstack([case["bs_xyz"], case["ris_xyz"]]) - origin
        if np.any(points[:, 2] <= 0) or not np.isfinite(points).all():
            raise RuntimeError("case points outside local scene coordinates")
        for point in points:
            hit = shape_scene.mi_scene.ray_intersect(
                mi.Ray3f(mi.Point3f(*map(float, point)), mi.Vector3f(0, 0, 1)))
            if bool(hit.is_valid()[0]) and float(hit.t[0]) < .02:
                raise RuntimeError("smoke endpoint lies on/inside native RF mesh")
        trace = trace_scene(rt, dr, mi, scene_info, case, scene_dir, out)
        trace["mitsuba_shape_ids"] = shape_scene_meta
        trace["geometry_preflight"] = geometry_check
        results.append(trace)
    summary = {
        "status": "complete", "scope": "two v1 development preview cases; bounded RT pipeline smoke only",
        "case_ids": [c["id"] for c in manifest], "frequency_hz": FREQ,
        "sionna_rt": importlib.metadata.version("sionna-rt"),
        "mitsuba": importlib.metadata.version("mitsuba"), "drjit": importlib.metadata.version("drjit"),
        "mitsuba_variant": mi.variant(), "source_hashes": inputs,
        "terrain_in_rf_scene": False,
        "solver": {"max_depth": 2, "samples_per_src": 10000, "max_num_paths_per_src": 40000,
                   "seed": 7, "specular_reflection": True, "diffraction": True,
                   "refraction_or_transmission": False, "synthetic_array": False},
        "array_layout": {"sizes": [list(x) for x in SIZES], "pitch_horizontal_m": PITCH_H,
                         "pitch_vertical_m": PITCH_V, "normal": "horizontal vector from RIS center to BS",
                         "offset_toward_bs_m": .05, "warning": "case schema lacks facade normal; plane orientation is a declared smoke approximation"},
        "gpu_before_mib": gpu_start, "gpu_final_mib": gpu_state(),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "build_and_trace_seconds": time.perf_counter() - t0,
        "cases": results,
    }
    path = out / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    print(json.dumps({k: summary[k] for k in ("status", "case_ids", "sionna_rt", "mitsuba_variant",
                                                   "peak_process_rss_kib", "build_and_trace_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
