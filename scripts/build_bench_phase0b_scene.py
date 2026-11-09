"""Build and smoke-test a tiny, development-only Maastricht LoD2/DTM Sionna tile."""
from __future__ import annotations
import hashlib
import gzip
import json
import os
import resource
import time
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("BENCH_PHASE0B_ROOT", "/var/tmp/bench-phase0b-20261108"))
LON, LAT = 5.688, 50.844
BOUNDS = (175967.0, 316952.0, 176407.0, 317392.0)
BAG_VERSION = "2025.09.03"
BAG_TILES = [
    {"file": "10-648-40.city.json.gz", "url": "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/648/40/10-648-40.city.json.gz", "bytes": 456952, "sha256": "738ee601f609ccfb77fe79d6607ed25d38c883855e83a902bc0b081cbe150e02"},
    {"file": "10-648-42.city.json.gz", "url": "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/648/42/10-648-42.city.json.gz", "bytes": 1002425, "sha256": "931ada9bf68dd7b20bcd1f11de6c00f1db25ea6a3b03163b63bb14dbd9093f57"},
    {"file": "10-650-40.city.json.gz", "url": "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/650/40/10-650-40.city.json.gz", "bytes": 604852, "sha256": "6c3a5ff84fb56d5e6a0caef68d2d2287995fdff736cc8a761fdc5b83cbb36cd3"},
    {"file": "10-650-42.city.json.gz", "url": "https://3dbag.fsn1.your-objectstorage.com/v20250903/tiles/10/650/42/10-650-42.city.json.gz", "bytes": 1758014, "sha256": "eb79c0f71c859d3ebf6578824e644226d860c9ac08179a16bcb4a02b1e8a3348"},
]
DTM_URL = "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/wcs/v1_0/?service=WCS&version=2.0.1&request=GetCoverage&CoverageID=dtm_05m&crs=http%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FEPSG%2F0%2F28992&format=image%2Ftiff&scalesize=x%28880%29%2Cy%28880%29&subset=x%28175967%2C176407%29&subset=y%28316952%2C317392%29"
DTM_SHA256 = "2c62b77b9787dc76a88a2f8ae7403e81e9783609d9883e965614b239c967c365"
MATERIAL_MAP = {
    "GroundSurface": "concrete",
    "WallSurface": "brick",
    "OuterCeilingSurface": "concrete",
    "RoofSurface": "concrete",
    "Window": "glass",
    "WindowSurface": "glass",
}
FALLBACK_MATERIAL = "concrete"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download():
    import requests
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tiles = []
    for source in BAG_TILES:
        t0 = time.perf_counter()
        r = requests.get(source["url"], timeout=90, stream=True)
        r.raise_for_status()
        # These are already-compressed publisher objects. requests may decode
        # HTTP Content-Encoding by default, which would invalidate the
        # publisher's hash over the stored gzip bytes. Disable that layer.
        r.raw.decode_content = False
        payload = r.raw.read()
        observed_sha = hashlib.sha256(payload).hexdigest()
        if len(payload) != source["bytes"] or observed_sha != source["sha256"]:
            raise RuntimeError(f"pinned 3DBAG tile integrity failure for {source['file']}: bytes={len(payload)} sha256={observed_sha}")
        path = raw / source["file"]
        path.write_bytes(payload)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if (data.get("type") != "CityJSON" or data.get("version") != "2.0"
                or data.get("metadata", {}).get("referenceSystem") != "https://www.opengis.net/def/crs/EPSG/0/7415"
                or data.get("metadata", {}).get("version") != "v2025.09.03"):
            raise RuntimeError(f"unexpected 3DBAG tile metadata in {source['file']}")
        tiles.append({"source": source, "path": path, "data": data,
                      "seconds": time.perf_counter()-t0,
                      "cityobjects": len(data.get("CityObjects", {})),
                      "vertices": len(data.get("vertices", [])),
                      "extent": data.get("metadata", {}).get("geographicalExtent")})
    unique_ids = set()
    duplicate_ids = set()
    buildings = 0
    parts = 0
    for tile in tiles:
        for cid, obj in tile["data"].get("CityObjects", {}).items():
            if cid in unique_ids:
                duplicate_ids.add(cid)
            else:
                unique_ids.add(cid)
            buildings += obj.get("type") == "Building"
            parts += obj.get("type") == "BuildingPart"
    if duplicate_ids:
        raise RuntimeError(f"duplicate CityObject IDs across pinned tiles: {len(duplicate_ids)}")
    cityjson = raw / "maastricht_lod2_cityjsonseq.jsonl"
    with cityjson.open("w") as f:
        for tile in tiles:
            f.write(json.dumps(tile["data"], separators=(",", ":")) + "\n")
    rt0 = time.perf_counter()
    tr = requests.get(DTM_URL, timeout=120)
    tr.raise_for_status()
    if tr.headers.get("content-type", "").split(";")[0] not in ("image/tiff", "application/octet-stream"):
        raise RuntimeError(f"WCS returned {tr.headers.get('content-type')}: {tr.text[:400]}")
    dem = raw / "ahn4_dtm_05m.tif"
    dem.write_bytes(tr.content)
    dtm_sha = sha(dem)
    if dtm_sha != DTM_SHA256:
        raise RuntimeError(f"pinned AHN DTM checksum mismatch: {dtm_sha}")
    return {"center_lonlat": [LON, LAT], "crs": "EPSG:7415 (RD New + NAP)",
            "bbox_epsg7415_m": list(BOUNDS),
            "tile_dimensions_m": [BOUNDS[2]-BOUNDS[0], BOUNDS[3]-BOUNDS[1]], "bag_version": BAG_VERSION,
            "bag_tiles": [{"file": t["source"]["file"], "url": t["source"]["url"],
                           "bytes": t["source"]["bytes"], "sha256": t["source"]["sha256"],
                           "seconds": t["seconds"], "cityobjects": t["cityobjects"],
                           "vertices": t["vertices"], "extent": t["extent"]} for t in tiles],
            "unique_cityobject_ids": len(unique_ids), "duplicate_cityobject_ids": 0,
            "building_cityobjects": buildings, "buildingpart_cityobjects": parts,
            "lod2_2_buildingparts": sum(obj.get("type") == "BuildingPart" and any(g.get("lod") == "2.2" for g in obj.get("geometry", []))
                                          for tile in tiles for obj in tile["data"]["CityObjects"].values()),
            "bag_file": {"path": str(cityjson), "bytes": cityjson.stat().st_size,
                         "sha256": sha(cityjson)},
            "dtm_url": tr.url, "dtm_seconds": time.perf_counter()-rt0,
            "dtm_file": {"path": str(dem), "bytes": dem.stat().st_size, "sha256": dtm_sha}}


def decode_vertices(feature):
    v = np.asarray(feature["vertices"], dtype=np.float64)
    transform = feature.get("transform")
    if transform:
        v = v * np.asarray(transform["scale"]) + np.asarray(transform["translate"])
    return v


def face_surface_indices(geometry):
    sem = geometry.get("semantics", {})
    values = sem.get("values")
    if geometry["type"] != "Solid" or not values:
        return []
    surfaces = sem.get("surfaces", [])
    # A CityJSON Solid has shells -> faces; semantics values mirror shells -> face indices.
    result = []
    for shell_index, shell in enumerate(geometry["boundaries"]):
        semantic_shell = values[shell_index] if isinstance(values[0], list) else values
        for face_index, face in enumerate(shell):
            sid = semantic_shell[face_index] if face_index < len(semantic_shell) else None
            typ = surfaces[sid].get("type") if sid is not None and sid < len(surfaces) else None
            result.append((face, typ))
    return result


def triangulate_semantic_face(rings, vertices_world, *, max_plane_error_m=0.01):
    """Triangulate a planar CityJSON face, preserving concavities and holes.

    Returns source-CRS 3-D triangles and the projected source-CRS polygon.
    Unsupported, invalid, non-planar or area-changing faces fail closed.
    """
    import shapely
    from shapely.geometry import Polygon

    xyz_rings = []
    for ring in rings:
        pts = np.asarray(vertices_world[np.asarray(ring, dtype=int), :3], dtype=float)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=1e-9):
            pts = pts[:-1]
        if len(pts) < 3:
            raise ValueError("ring has fewer than three unique vertices")
        xyz_rings.append(pts)
    outer = xyz_rings[0]
    normal = np.zeros(3, dtype=float)
    for p, q in zip(outer, np.roll(outer, -1, axis=0)):
        normal += np.array([(p[1]-q[1])*(p[2]+q[2]),
                            (p[2]-q[2])*(p[0]+q[0]),
                            (p[0]-q[0])*(p[1]+q[1])])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("degenerate face plane")
    normal /= norm
    drop = int(np.argmax(np.abs(normal)))
    keep = [axis for axis in range(3) if axis != drop]
    plane_origin = outer[0]
    all_points = np.concatenate(xyz_rings, axis=0)
    plane_error = float(np.max(np.abs((all_points-plane_origin) @ normal)))
    if plane_error > max_plane_error_m or abs(normal[drop]) < 1e-8:
        raise ValueError(f"non-planar/ill-conditioned face (plane error {plane_error:.6g} m)")
    polygon = Polygon(xyz_rings[0][:, keep], [r[:, keep] for r in xyz_rings[1:]])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("invalid projected face polygon")
    if not hasattr(shapely, "constrained_delaunay_triangles"):
        raise RuntimeError("Shapely constrained Delaunay triangulation is required")
    collection = shapely.constrained_delaunay_triangles(polygon)
    pieces = [g for g in collection.geoms if g.geom_type == "Polygon" and g.area > 0]
    if not pieces:
        raise ValueError("triangulator returned no faces")
    tri_area = float(sum(g.area for g in pieces))
    union = shapely.union_all(pieces)
    area_err = abs(tri_area-polygon.area)/polygon.area
    uncovered_err = polygon.symmetric_difference(union).area/polygon.area
    if area_err > 1e-8 or uncovered_err > 1e-8:
        raise ValueError(f"triangulation area mismatch: sum={area_err:.3g}, coverage={uncovered_err:.3g}")
    triangles = []
    for tri in pieces:
        coords2 = np.asarray(tri.exterior.coords[:-1], dtype=float)
        if len(coords2) != 3:
            raise ValueError("constrained triangulator returned a non-triangle")
        xyz = np.zeros((3, 3), dtype=float)
        xyz[:, keep] = coords2
        delta = (coords2-plane_origin[keep]) @ normal[keep]
        xyz[:, drop] = plane_origin[drop]-delta/normal[drop]
        triangles.append(xyz)
    return triangles, polygon


def polygon_surfaces(features, origin, bounds):
    by_class = {k: [] for k in set(MATERIAL_MAP.values())}
    footprints = []
    height_src = {}
    surface_counts = {}
    rejected_outside = 0
    invalid_ground = 0
    from shapely.geometry import Polygon, box
    scene_bounds = box(bounds[0], bounds[1], bounds[2], bounds[3])
    for feature in features:
        v = decode_vertices(feature)
        for cid, obj in feature.get("CityObjects", {}).items():
            if obj.get("type") != "BuildingPart":
                continue
            attr = obj.get("attributes", {})
            if not any(g.get("lod") == "2.2" for g in obj.get("geometry", [])):
                continue
            source = attr.get("b3_pw_bron", "unknown")
            part_surfaces = []
            part_footprints = []
            for g in obj.get("geometry", []):
                if g.get("lod") != "2.2":
                    continue
                for face, stype in face_surface_indices(g):
                    material = MATERIAL_MAP.get(stype, FALLBACK_MATERIAL)
                    try:
                        triangles, projected = triangulate_semantic_face(face, v)
                    except (ValueError, RuntimeError) as exc:
                        raise RuntimeError(f"unsupported LoD2.2 face {cid}/{stype}: {exc}") from exc
                    for triangle in triangles:
                        triangle[:, 0] -= origin[0]
                        triangle[:, 1] -= origin[1]
                        triangle[:, 2] -= origin[2]
                        part_surfaces.append((material, triangle, stype))
                    if stype == "GroundSurface":
                        # GroundSurface planes should be horizontal. Keep the
                        # projected polygon for outdoor tests; it retains holes.
                        footprint = Polygon(projected.exterior.coords,
                                            [ring.coords for ring in projected.interiors])
                        if not footprint.is_valid or footprint.is_empty:
                            invalid_ground += 1
                            raise RuntimeError(f"invalid GroundSurface footprint for {cid}")
                        world = Polygon([(x+origin[0], y+origin[1]) for x,y in footprint.exterior.coords],
                                        [[(x+origin[0], y+origin[1]) for x,y in ring.coords]
                                         for ring in footprint.interiors])
                        if world.intersects(scene_bounds):
                            part_footprints.append(footprint)
            if not part_footprints:
                rejected_outside += 1
                continue
            height_src[str(source)] = height_src.get(str(source), 0)+1
            footprints.extend(part_footprints)
            for material, points, stype in part_surfaces:
                by_class[material].append(points)
                surface_counts[stype or "missing"] = surface_counts.get(stype or "missing", 0)+1
    return by_class, footprints, height_src, surface_counts, rejected_outside, invalid_ground


def write_ply(path, polys):
    verts, faces = [], []
    for poly in polys:
        if np.asarray(poly).shape != (3, 3):
            raise ValueError("PLY input must contain validated triangles")
        start = len(verts)
        verts.extend(poly.tolist())
        faces.append((start, start+1, start+2))
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return len(verts), len(faces)



def select_height_window(dtm, window_m=100, step_m=1, scan_m=10, max_nodata=.01, min_relief=5.0):
    """Choose the first valid/elevated window by easting then northing only."""
    sampled = np.asarray(dtm)[::step_m, ::step_m]
    nrows, ncols = sampled.shape
    size = int(window_m/step_m)
    stride = int(scan_m/step_m)
    # For ascending northing order, scan image rows from south to north.
    row_starts = list(range(0, nrows-size+1, stride))[::-1]
    col_starts = range(0, ncols-size+1, stride)
    for c0 in col_starts:
        for r0 in row_starts:
            block = sampled[r0:r0+size, c0:c0+size]
            good = block[np.isfinite(block)]
            nodata = 1.0 - good.size/block.size
            if good.size and nodata <= max_nodata and float(good.max()-good.min()) >= min_relief:
                return {"col_start_1m": int(c0), "row_start_1m": int(r0),
                        "width_m": int(window_m), "height_m": int(window_m),
                        "nodata_share": float(nodata), "min_nap_m": float(good.min()),
                        "max_nap_m": float(good.max()),
                        "relief_m": float(good.max()-good.min()),
                        "selection": "first qualifying 100m window, 10m lattice, east then north"}
    return None


def make_scene(meta):
    import rasterio
    from shapely.geometry import Point, Polygon
    from shapely.prepared import prep
    raw = ROOT / "raw" / "maastricht_lod2_cityjsonseq.jsonl"
    features = [json.loads(line) for line in raw.read_text().splitlines()]
    bbox = meta["bbox_epsg7415_m"]
    half_x, half_y = (bbox[2]-bbox[0])/2.0, (bbox[3]-bbox[1])/2.0
    origin = (bbox[0], bbox[1], 0)
    byclass, footprints, height_src, surface_counts, rejected_outside, invalid_ground = polygon_surfaces(features, origin, bbox)
    if not footprints:
        raise RuntimeError("No LoD2.2 GroundSurface footprints parsed")
    # Preserve native terrain height data separately; it is for placing points only.
    dem_file = ROOT / "raw" / "ahn4_dtm_05m.tif"
    with rasterio.open(dem_file) as ds:
        z = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
        nodata_share = float(np.isnan(z).mean())
        valid = z[np.isfinite(z)]
        if valid.size == 0:
            raise RuntimeError("DTM tile contains no valid cells")
        dz = float(valid.max()-valid.min())
        transform = ds.transform
        crs = str(ds.crs)
        gy, gx = np.gradient(z, abs(ds.res[0]), abs(ds.res[1]))
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        slope_values = slope_deg[np.isfinite(slope_deg)]
        # DTM is 0.5 m natively; select the declared 100 m windows on a
        # deterministic 1 m grid (north-to-south raster row order).
        z_1m = z[::2, ::2]
        active_patch = select_height_window(z_1m)
        height = {"shape": list(z.shape), "pixel_size_m": list(ds.res),
                  "crs": crs, "vertical_datum": "NAP, from EPSG:7415 / PDOK AHN metadata",
                  "nodata_share": nodata_share, "min_nap_m": float(valid.min()),
                  "max_nap_m": float(valid.max()), "relief_m": dz,
                  "slope_deg_p50": float(np.nanpercentile(slope_values, 50)),
                  "slope_deg_p95": float(np.nanpercentile(slope_values, 95)),
                  "slope_deg_max": float(np.nanmax(slope_values)),
                  "elevation_source": "AHN4 2020–2022, PDOK dtm_05m",
                  "strict_full_tile_pass": nodata_share <= .01 and dz >= 5.0,
                  "diagnostic_patch": active_patch}
        if nodata_share <= .01:
            active_window = {"x_min_m": 0.0, "y_min_m": 0.0,
                             "x_max_m": float(BOUNDS[2]-BOUNDS[0]),
                             "y_max_m": float(BOUNDS[3]-BOUNDS[1]),
                             "source": "full_tile"}
        elif active_patch is not None:
            active_window = {"x_min_m": float(active_patch["col_start_1m"]),
                             "y_min_m": float(2*half_y-(active_patch["row_start_1m"]+active_patch["height_m"])),
                             "x_max_m": float(active_patch["col_start_1m"]+active_patch["width_m"]),
                             "y_max_m": float(2*half_y-active_patch["row_start_1m"]),
                             "source": "diagnostic_patch"}
        else:
            active_window = None
        def ground(xy):
            if active_window is None or not (active_window["x_min_m"] <= xy[0] <= active_window["x_max_m"]
                                             and active_window["y_min_m"] <= xy[1] <= active_window["y_max_m"]):
                return None
            col, row = (~transform) * (xy[0]+origin[0], xy[1]+origin[1])
            ri, ci = int(round(row)), int(round(col))
            if ri < 0 or ci < 0 or ri >= z.shape[0] or ci >= z.shape[1]:
                return None
            val = z[ri, ci]
            return float(val) if np.isfinite(val) else None

    if nodata_share <= .01 and dz < 5.0:
        raise RuntimeError(f"Predeclared slope/relief gate failed: relief={dz:.3f} m < 5 m")
    if nodata_share > .01 and active_patch is None:
        raise RuntimeError(f"Full tile DTM fails the fixed {nodata_share:.2%} nodata criterion and no qualifying diagnostic patch exists")
    allgeom = prep(__import__("shapely").union_all(footprints))
    # Geometry-only outdoor sample pair: scan fixed 10 m grid east/west of tile center.
    grid = np.arange(10, int(2*half_x)-9, 5, dtype=float)
    positions = []
    for yy in grid:
        for xx in grid:
            xy = (xx, yy)
            elev = ground(xy)
            if elev is None or allgeom.contains(Point(*xy)):
                continue
            local_z = elev-origin[2]
            positions.append((np.array([xx, yy, local_z+1.5]), elev))
    if len(positions) < 2:
        raise RuntimeError(f"Could not find two ground-supported outdoor points: {len(positions)}")
    # Deterministic geometry-only LOS selection: scan points in grid order, reject
    # pairs with a building-footprint intersection or insufficient terrain clearance.
    from shapely.geometry import LineString
    tx = rx = None
    for i, (p0, _) in enumerate(positions):
        for p1, _ in positions[i+1:]:
            distance = float(np.linalg.norm(p1[:2]-p0[:2]))
            if distance < 20 or distance > 60:
                continue
            if allgeom.intersects(LineString([p0[:2],p1[:2]])):
                continue
            line = np.linspace(p0, p1, max(3, int(distance/5)+1))
            ground_line = [ground(point[:2]) for point in line]
            if all(g is not None and float(p[2]) > g+.3 for p,g in zip(line,ground_line)):
                tx, rx = p0, p1
                break
        if tx is not None:
            break
    if tx is None or rx is None:
        raise RuntimeError("No geometry-only outdoor LOS pair passed the preregistered clearance rule")
    outdir = ROOT / "scene"
    outdir.mkdir(parents=True, exist_ok=True)
    surface_meta = {}
    for mat, polys in byclass.items():
        nv, nf = write_ply(outdir / f"{mat}.ply", polys)
        surface_meta[mat] = {"triangles": len(polys), "vertices": nv, "faces": nf,
                             "sha256": sha(outdir / f"{mat}.ply")}
    # A height-only one-metre terrain mesh is retained separately from the RF scene.
    # It is useful for inspection and point placement, but is intentionally never
    # listed as a Mitsuba shape because terrain is height-only by protocol.
    terrain_vertices = []
    terrain_faces = []
    nrows, ncols = z.shape
    sampled_rows = list(range(0, nrows, 2))
    sampled_cols = list(range(0, ncols, 2))
    index = {}
    for ii, i in enumerate(sampled_rows):
        for jj, j in enumerate(sampled_cols):
            if np.isfinite(z[i, j]):
                index[ii, jj] = len(terrain_vertices)
                terrain_vertices.append((float(j*transform.a),
                                         float(BOUNDS[3]-origin[1]+i*transform.e),
                                         float(z[i,j]-origin[2])))
    for ii in range(len(sampled_rows)-1):
        for jj in range(len(sampled_cols)-1):
            corners = [(ii,jj),(ii,jj+1),(ii+1,jj+1),(ii+1,jj)]
            if all(p in index for p in corners):
                a,b,c,d = (index[p] for p in corners)
                terrain_faces.extend([(a,b,c),(a,c,d)])
    terrain = outdir / "terrain_height_only.ply"
    with terrain.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(terrain_vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(terrain_faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in terrain_vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in terrain_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    height_mesh = {"vertices": len(terrain_vertices), "triangles": len(terrain_faces),
                   "resolution_m": 1.0, "terrain_in_rf_scene": False,
                   "sha256": sha(terrain)}

    mats = {"concrete": "concrete", "brick": "brick", "glass": "glass"}
    xml = ['<scene version="2.1.0">']
    for ident, kind in mats.items():
        xml += [f'<bsdf type="itu-radio-material" id="{ident}"><string name="type" value="{kind}"/>'
                '<float name="thickness" value="0.1"/></bsdf>']
    for mat in byclass:
        path = outdir / f"{mat}.ply"
        if path.stat().st_size == 0 or not byclass[mat]:
            continue
        xml.append(f'<shape type="ply" id="surface-{mat}"><string name="filename" value="{path.name}"/>'
                   f'<ref id="{mat}" name="bsdf"/></shape>')
    xml.append("</scene>")
    scene_xml = outdir / "scene.xml"
    scene_xml.write_text("\n".join(xml)+"\n")

    # Check the user and BS/RX points lie outdoors and at least 1.5 m above sampled terrain.
    point_ground = {"tx": ground(tx[:2]), "rx": ground(rx[:2])}
    outdoor_pass = all(e is not None and abs(p[2] - e - 1.5) < .02 and
                       not allgeom.contains(Point(*p[:2]))
                       for p,e in [(tx, point_ground["tx"]), (rx, point_ground["rx"])])
    # Report every footprint height source; no inferred OSM/default building heights.
    building_parts = sum(height_src.values())
    return {"height_model": height, "tile_packages": len(features),
            "building_parts_outside_scene_bounds": rejected_outside,
            "invalid_ground_surfaces_repaired": invalid_ground,
            "building_part_height_provenance": height_src,
            "selected_lod2_2_buildingparts": building_parts,
            "lod2_2_coverage_of_selected_geometry": 1.0,
            "surface_types": surface_counts, "surface_meshes": surface_meta,
            "outdoor_points_local_m": {"tx": tx.tolist(), "rx": rx.tolist()},
            "outdoor_ground_nap_m": point_ground, "outdoor_check_pass": outdoor_pass,
            "scene_xml_sha256": sha(scene_xml), "terrain_rf_interactions": False,
            "source_ground_mesh_separate": height_mesh,
            "active_height_window": active_window}


def smoke(meta):
    import mitsuba as mi
    import sionna.rt as rt
    # Do not force a GPU device; record the managed Mitsuba variant selected by this environment.
    scene = rt.load_scene(str(ROOT/"scene"/"scene.xml"), merge_shapes=False)
    scene.frequency = 5.53e9
    scene.tx_array = scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
        pattern="iso", polarization="V")
    tx = np.asarray(meta["outdoor_points_local_m"]["tx"], dtype=float)
    rx = np.asarray(meta["outdoor_points_local_m"]["rx"], dtype=float)
    scene.add(rt.Transmitter("tx", position=tx.tolist()))
    scene.add(rt.Receiver("rx", position=rx.tolist()))
    tic = time.perf_counter()
    paths = rt.PathSolver()(scene=scene, max_depth=0, los=True, specular_reflection=False,
                            diffuse_reflection=False, refraction=False, diffraction=False,
                            synthetic_array=False, samples_per_src=1000, seed=7)
    cir, delays = paths.cir(normalize_delays=False, out_type="numpy")
    import drjit as dr
    dr.sync_thread()
    elapsed = time.perf_counter()-tic
    values = cir.reshape(-1)
    finite = bool(np.isfinite(values.real).all() and np.isfinite(values.imag).all())
    nonzero = bool(np.any(np.abs(values)>0))
    return {"seconds": elapsed, "scene_loaded": True,
            "mitsuba_variant": mi.variant(), "cir_shape": list(cir.shape),
            "delay_shape": list(np.asarray(delays).shape),
            "finite": finite, "nonzero": nonzero,
            "coefficient_abs": float(np.abs(values).max()) if len(values) else 0.,
            "pass": finite and nonzero,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    t0=time.perf_counter()
    meta=download()
    meta["source_software"]={"sionna_rt": "2.0.1", "python": os.popen("python --version").read().strip()}
    meta["downloaded"]=True
    import rasterio
    dem = ROOT / "raw" / "ahn4_dtm_05m.tif"
    with rasterio.open(dem) as ds:
        z = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
        valid = z[np.isfinite(z)]
        patch = select_height_window(z[::2, ::2])
        dtm_audit = {"crs": str(ds.crs), "shape": list(z.shape),
                     "resolution_m": list(ds.res), "vertical_datum": "NAP",
                     "valid_cells": int(valid.size), "total_cells": int(z.size),
                     "nodata_fraction": float(1-valid.size/z.size),
                     "minimum_nap_m": float(valid.min()) if valid.size else None,
                     "maximum_nap_m": float(valid.max()) if valid.size else None,
                     "robust_p05_p95_relief_m": float(np.nanpercentile(valid,95)-np.nanpercentile(valid,5)) if valid.size else None,
                     "strict_full_tile_gate_pass": bool(valid.size and (1-valid.size/z.size) <= .01 and float(valid.max()-valid.min()) >= 5),
                     "diagnostic_100m_patch": patch,
                     "diagnostic_patch_gate_pass": patch is not None,
                     "terrain_used_for_height_only": True,
                     "terrain_added_to_rf_scene": False}
    meta["terrain_audit"] = dtm_audit
    if not dtm_audit["strict_full_tile_gate_pass"] and patch is None:
        # The preregistered stop rule prohibits any link or RT trace when the
        # full tile fails and no qualifying diagnostic patch exists.
        meta["status"] = "stopped_pretrace_no_qualifying_dtm_patch"
        meta["wall_seconds_total"] = time.perf_counter()-t0
        meta["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        meta["tracing_attempted"] = False
        report = ROOT/"results"/"feasibility.json"
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
        print(json.dumps(meta,indent=2,allow_nan=False),flush=True)
        return
    scene_meta=make_scene(meta)
    meta["scene"]=scene_meta
    if not scene_meta["outdoor_check_pass"]:
        raise RuntimeError("Predeclared outdoor/height check failed")
    trace=smoke(meta)
    meta["smoke"]=trace
    meta["wall_seconds_total"]=time.perf_counter()-t0
    meta["peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    meta["status"]="complete" if trace["pass"] else "failed"
    report=ROOT/"results"/"feasibility.json"
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
    print(json.dumps(meta,indent=2,allow_nan=False),flush=True)
    if meta["status"] != "complete":
        raise SystemExit(1)


if __name__=="__main__":
    main()
