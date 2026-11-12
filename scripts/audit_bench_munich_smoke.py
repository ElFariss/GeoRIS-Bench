"""One declared Phase 0b CityGML/DGM development smoke; no benchmark channels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

BOUNDS = (691383., 5334880., 691483., 5334980.)
DATA = {"690_5334.gml": "232c75a398ec7ee81736aeb72f7eda374f88b9431f8c35cdbee18a51ff029e87",
        "691_5334.tif": "ddfed9fdbe2e69e8f8143acb7f5a483f49e07b381b9f6fe4b9d52a40e02f381c"}
MATERIALS = {"WallSurface": "brick", "RoofSurface": "concrete", "GroundSurface": "concrete"}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rings(poly):
    rs = [poly.find("./{*}exterior/{*}LinearRing"),
          *poly.findall("./{*}interior/{*}LinearRing")]
    if rs[0] is None:
        return None
    out = []
    for r in rs:
        pos = r.findtext("./{*}posList")
        if pos is None:
            pos = " ".join(n.text for n in r.findall("./{*}pos"))
        xyz = np.fromstring(pos, sep=" ").reshape(-1, 3)
        if len(xyz) and np.array_equal(xyz[0], xyz[-1]):
            xyz = xyz[:-1]
        if len(xyz):
            xyz = xyz[np.r_[True, np.any(np.diff(xyz, axis=0) != 0, axis=1)]]
        out.append(xyz)
    return out


def triangles(face):
    outer = face[0]
    if len(outer) < 3:
        return []
    rel = outer - outer[0]
    normal = np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0)
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] == 0:
        return []
    keep = [i for i in range(3) if i != axis]
    nonzero = [r for r in face if len(r) >= 3 and Polygon(r[:, keep]).area > 0]
    if not nonzero or nonzero[0] is not outer:
        return []
    projected = Polygon(outer[:, keep], [r[:, keep] for r in nonzero[1:]])
    if not projected.is_valid:
        raise ValueError("nondegenerate CityGML face invalid in dominant projection")
    point_map = {tuple(p[keep]): p for r in nonzero for p in r}
    tris = list(constrained_delaunay_triangles(projected).geoms)
    if not tris or abs(sum(t.area for t in tris)-projected.area) > 1e-8*max(projected.area, 1.):
        raise ValueError("constrained triangulation area check failed")
    out = []
    for t in tris:
        v = np.stack([point_map[tuple(p)] for p in list(t.exterior.coords)[:3]])
        if np.dot(np.cross(v[1]-v[0],v[2]-v[0]), normal) < 0:
            v = v[[0,2,1]]
        out.append(v)
    return out


def read_buildings(path, patch):
    root_halo = patch.buffer(35, cap_style="square", join_style="mitre").envelope
    polys, footprints, roofs = defaultdict(list), [], []
    count = Counter()
    it = ET.iterparse(path, events=("start","end"))
    _, root = next(it)
    for event, elem in it:
        if event != "end" or elem.tag.rsplit("}",1)[-1] != "cityObjectMember":
            continue
        b = elem.find("./{*}Building")
        if b is None:
            elem.clear()
            continue
        ground = [rs for s in b.findall(".//{*}GroundSurface")
                  for p in s.findall(".//{*}Polygon") if (rs := rings(p)) is not None]
        areas = [Polygon(rs[0][:,:2], [r[:,:2] for r in rs[1:]])
                 for rs in ground if len(rs[0]) >= 3]
        picked = [p for p in areas if p.area>0 and p.intersects(root_halo)]
        if not picked:
            elem.clear()
            continue
        if not all(p.is_valid for p in picked):
            raise ValueError("invalid native building footprint in halo")
        count["buildings"] += 1
        footprints.extend(picked)
        for key in MATERIALS:
            for s in b.findall(".//{*}"+key):
                for p in s.findall(".//{*}Polygon"):
                    rs = rings(p)
                    tt = triangles(rs) if rs is not None else []
                    if not tt:
                        count["skipped_zero_area_or_empty_face"] += 1
                        continue
                    polys[key].extend(tt)
                    count[key] += 1
                    if key=="RoofSurface":
                        roofs.extend(poly for t in tt if (poly:=Polygon(t[:,:2])).area>0)
        elem.clear()
        if len(root)>500:
            root.clear()
    if not count["buildings"] or any(not polys[k] for k in MATERIALS):
        raise ValueError(f"missing LoD2 semantic surfaces {count}")
    return polys, unary_union(footprints), unary_union([*footprints,*roofs]), dict(count)


def ply(path, vertices, faces):
    with path.open("w") as f:
        f.write(f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
""")
        np.savetxt(f, vertices, fmt="%.7f")
        np.savetxt(f, faces, fmt="3 %d %d %d")


def build(raw, out):
    start = time.perf_counter()
    hashes = {f:sha(raw/f) for f in DATA}
    if hashes != DATA:
        raise ValueError(f"source hash mismatch {hashes}")
    patch = box(*BOUNDS)
    with rasterio.open(raw/"691_5334.tif") as r:
        if r.crs.to_epsg()!=25832 or r.res!=(1.,1.):
            raise ValueError(f"unexpected terrain grid {r.crs} {r.res}")
        w = from_bounds(*BOUNDS,transform=r.transform)
        z = r.read(1,window=w,masked=True).filled(np.nan)[::-1].astype(float)
    nodata = 1-float(np.isfinite(z).mean())
    if z.shape!=(100,100) or nodata > .01 or np.nanmax(z)-np.nanmin(z)<5.:
        raise ValueError("declared DTM extent/nodata/relief failed")
    origin = np.array([*BOUNDS[:2],float(np.nanmin(z))])
    terrain = z-origin[2]

    model, footprint, obstruction, counts = read_buildings(raw/"690_5334.gml",patch)
    elements = []
    for name, group in model.items():
        mat = MATERIALS[name]
        coords = np.asarray(group).reshape(-1,3)-origin
        indices = np.arange(len(coords)).reshape(-1,3)
        path = out/f"{name}.ply"
        ply(path,coords,indices)
        elements.append(f'''<bsdf type="itu-radio-material" id="mat-medium-{name}">
           <string name="type" value="{mat}"/><float name="thickness" value="0.1"/></bsdf>
         <shape type="ply" id="mesh-{name}"><string name="filename" value="{path.name}"/>
           <ref id="mat-medium-{name}" name="bsdf"/></shape>''')
    xml = out/"building_rf_scene.xml"
    xml.write_text('<scene version="2.1.0">'+"".join(elements)+'</scene>')

    a,b = np.meshgrid(np.arange(100)+.5,np.arange(100)+.5)
    coords = np.column_stack([a.ravel(),b.ravel(),terrain.ravel()])
    i = np.arange(10000).reshape(100,100)[:-1,:-1].ravel()
    faces = np.vstack([np.column_stack([i,i+1,i+101]),np.column_stack([i,i+101,i+100])])
    if np.isnan(coords[:,2]).any():
        valid = np.isfinite(coords[:,2])
        faces = faces[valid[faces].all(axis=1)]
        idx = np.cumsum(valid)-1
        faces = idx[faces]
        coords = coords[valid]
    ply(out/"terrain_height_only.ply",coords,faces)

    def pt(i,j):
        return Point(BOUNDS[0]+i+.5,BOUNDS[1]+j+.5)

    indices = [(i,j) for i in range(4,100,5) for j in range(4,100,5) if np.isfinite(terrain[j,i])]
    open_points = [(i,j) for i,j in indices if obstruction.distance(pt(i,j))>=1.]
    indoor = next(((i,j) for i,j in indices if footprint.contains(pt(i,j))),None)
    if indoor is None:
        raise ValueError("no in-footprint control")
    candidate = None
    for k,p in enumerate(open_points):
        for q in open_points[k+1:]:
            dis = np.linalg.norm(np.subtract(p,q))
            if not 15.<=dis<=35. or LineString([pt(*p),pt(*q)]).distance(obstruction)<1.:
                continue
            u = np.linspace(0,1,1+int(np.ceil(dis/.25)))
            xx = p[0]+u*(q[0]-p[0])
            yy = p[1]+u*(q[1]-p[1])
            # Use the maximum elevation of adjacent triangle vertices as
            # a conservative clearance test between raster sample centres.
            lx,ly = np.floor(xx).astype(int),np.floor(yy).astype(int)
            sample = np.max([terrain[np.minimum(ly+j,99),np.minimum(lx+i,99)] for i in (0,1) for j in (0,1)],axis=0)
            clearance = (1-u)*terrain[p[1],p[0]]+u*terrain[q[1],q[0]]+1.5-sample
            if np.isfinite(clearance).all() and clearance.min()>=.2:
                candidate = (p,q,float(clearance.min()))
                break
        if candidate:
            break
    if not candidate:
        raise ValueError("no geometry-only LOS fixture in fixed patch")
    p,q,margin = candidate

    def local(ij,h):
        i,j=ij
        return [i+.5,j+.5,float(terrain[j,i])+h]

    return {"source_sha256":hashes,"bounds_epsg25832":BOUNDS,"vertical_epsg":7837,"local_origin_m":origin.tolist(),
            "dtm_nodata_fraction":nodata,"dtm_min_max_m":[float(np.nanmin(z)),float(np.nanmax(z))],
            "dtm_mesh_triangles":len(faces),"terrain_in_rf_scene":False,"surface_counts":counts,
            "triangles_per_semantic":{k:len(v) for k,v in model.items()},"material_by_semantic":MATERIALS,
            "tx_local_m":local(p,1.5),"rx_local_m":local(q,1.5),"indoor_control_ground_local_m":local(indoor,0.),
            "ray_terrain_clearance_lower_bound_m":margin,"build_seconds":time.perf_counter()-start}


def smoke(out, result):
    from importlib.metadata import version
    import drjit as dr
    import mitsuba as mi
    import sionna.rt as rt

    if version("sionna-rt")!="2.0.1" or not mi.variant().startswith("cuda"):
        raise ValueError("expected pinned Sionna RT 2.0.1 CUDA")
    start = time.perf_counter()
    scene = rt.load_scene(str(out/"building_rf_scene.xml"),merge_shapes=False)
    height = mi.load_dict({"type":"scene","ground":{"type":"ply","filename":str(out/"terrain_height_only.ply")}})
    load_time=time.perf_counter()-start
    rf_shapes = [shape.id() for shape in scene.mi_scene.shapes()]
    if len(rf_shapes) != len(MATERIALS) or any("terrain" in name.lower() for name in rf_shapes):
        raise ValueError(f"unexpected RF scene inventory: {rf_shapes}")

    def down(s, xyz):
        h=s.ray_intersect(mi.Ray3f(mi.Point3f(float(xyz[0]),float(xyz[1]),1000),mi.Vector3f(0,0,-1)))
        return float(h.p.z[0]) if bool(h.is_valid()[0]) else None

    probes=[result["tx_local_m"],result["rx_local_m"]]
    measured=[down(height,p) for p in probes]
    residuals=[abs(h-(p[2]-1.5)) for p,h in zip(probes,measured,strict=True)]
    roof_control=down(scene.mi_scene,result["indoor_control_ground_local_m"])
    if max(residuals)>.02 or any(down(scene.mi_scene,p) is not None for p in probes):
        raise ValueError("declared outdoor/terrain check failed")
    if roof_control is None or roof_control<=result["indoor_control_ground_local_m"][2]+.5:
        raise ValueError("in-footprint control did not hit a building above the terrain")
    scene.frequency=5.53e9
    scene.tx_array=scene.rx_array=rt.PlanarArray(
        num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern="iso",polarization="VH")
    scene.add(rt.Transmitter("tx",position=probes[0]))
    scene.add(rt.Receiver("rx",position=probes[1]))
    start=time.perf_counter()
    paths=rt.PathSolver()(scene=scene,max_depth=0,los=True,specular_reflection=False,
                          diffuse_reflection=False,refraction=False,diffraction=False,
                          synthetic_array=False,samples_per_src=1000,max_num_paths_per_src=1000,seed=7)
    a,tau=paths.cir(normalize_delays=False,out_type="numpy")
    dr.sync_thread()
    if (a.shape!=(1,2,1,2,1,1) or tau.shape[-1]!=1 or not np.isfinite(tau).all() or np.min(tau)<=0 or np.ptp(tau)>1e-11
        or not np.isfinite(a).all() or any(abs(a[0,i,0,i,0,0])<=0 for i in (0,1))):
        raise ValueError(f"LOS/dual-pol smoke failed, CIR shape {a.shape}, delay shape {tau.shape}")
    result.update({"status":"PASS","sionna_rt":version("sionna-rt"),"mitsuba":version("mitsuba"),
                   "mitsuba_variant":mi.variant(),"geometry_load_seconds":load_time,
                   "ground_mesh_intersection_residual_m":residuals,"control_roof_height_local_m":roof_control,
                   "rf_shapes":rf_shapes,"cir_shape":list(a.shape),
                   "valid_positive_los_delay":True,"finite_nonzero_copol_ports":2,
                   "solve_seconds":time.perf_counter()-start,
                   "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--build-only",action="store_true")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    result=build(args.raw,args.out)
    if not args.build_only:
        smoke(args.out,result)
    (args.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
