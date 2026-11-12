"""Controlled Sionna 2.0.1 capability and native Munich geometry audit."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import resource
import time
from xml.etree import ElementTree as ET

import drjit as dr
import mitsuba as mi
import numpy as np
import sionna.rt as rt
from sionna.rt.constants import InteractionType


def hits(scene, xy):
    xy = np.asarray(xy)
    origin = mi.Point3f(mi.Float(xy[:, 0]), mi.Float(xy[:, 1]), 1000)
    hit = scene.mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(0, 0, -1)))
    z = hit.p.z.numpy().astype(float)
    z[~hit.is_valid().numpy()] = np.inf
    return z


def first_shape(scene, xy):
    h = scene.mi_scene.ray_intersect(
        mi.Ray3f(mi.Point3f(float(xy[0]), float(xy[1]), 1000), mi.Vector3f(0, 0, -1))
    )
    return h.shape[0].id() if bool(h.is_valid()[0]) else None


def geometry(layout, out, polygon):
    # Same native loader and default shape merging as trace_munich.base_scene.
    from matplotlib.path import Path as Polygon
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(rt.scene.munich)
    scene = rt.load_scene(str(source))
    users = np.asarray(layout["users"]).reshape(-1, 3)
    ris = np.asarray(layout["ris_center"])
    points = np.vstack((ris, users))
    z = hits(scene, points[:, :2])
    valid = (z < .5) & (z <= points[:, 2] - .2)
    invalid = np.flatnonzero(~valid[1:]) + 1
    inside = Polygon(polygon).contains_points(points[:, :2]) & (points[:, 2] < 16.86401)

    def demonstrate(point, require_footprint=False):
        d = np.arange(-45., 46.)
        dx, dy = np.meshgrid(d, d, indexing="ij")
        xy = point[:2] + np.column_stack((dx.ravel(), dy.ravel()))
        dist = np.linalg.norm(xy - point[:2], axis=1)
        offset = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
        ground = hits(scene, (xy[:, None] + offset).reshape(-1, 2)).reshape(-1, 9)
        good = (dist <= 45) & ((ground < .5) & (ground <= point[2] - .2)).all(axis=1)
        if require_footprint:
            good &= Polygon(polygon).contains_points(xy)
        candidates = np.lexsort((xy[:, 1], xy[:, 0], dist))
        selection = next((int(i) for i in candidates if good[i]), None)
        if selection is None:
            return None
        i = selection
        return {"position_m": [*map(float, xy[i]), float(point[2])],
                "moved_m": float(dist[i]), "height_3x3_m": ground[i].tolist(),
                "native_shape": first_shape(scene, xy[i])}

    checks = {"ris": demonstrate(ris, require_footprint=bool(valid.all())),
              "first_invalid_ue": demonstrate(points[invalid[0]]) if len(invalid) else None}
    x, y = np.arange(0, 116, .5), np.arange(-25, 96, .5)
    xx, yy = np.meshgrid(x, y)
    field = hits(scene, np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.pcolormesh(x, y, np.ma.masked_invalid(field), shading="nearest", cmap="viridis", vmin=0, vmax=28)
    fig.colorbar(image, ax=ax, label="Native downward first-hit height (m)")
    outline = np.vstack((polygon, polygon[0]))
    ax.plot(*outline.T, color="tab:orange", label="Disputed exported footprint")
    for mask, label, color in [(valid[1:], "Legacy UE: ground pass", "white"),
                               (~valid[1:], "Legacy UE: failed ground check", "red")]:
        ax.scatter(*users[mask, :2].T, s=9, c=color, label=label)
    ax.scatter(*ris[:2], s=50, c="cyan", marker="x", label="Legacy RIS")
    for point in checks.values():
        if point:
            ax.scatter(*point["position_m"][:2], s=72, c="magenta", marker="+")
    ax.set(aspect="equal", xlabel="Scene x (m)", ylabel="Scene y (m)",
           title="Munich audit: native surface heights near the legacy RIS")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "munich_native_height_audit.png", dpi=140)
    plt.close(fig)
    material_elements = ET.parse(source).getroot().findall("./bsdf")
    thickness = [float(f.attrib["value"]) for b in material_elements for f in b.findall("./float")
                 if f.attrib.get("name") == "thickness"]
    return {"probe_count": len(users), "export_footprint_below_roof_user_count": int(inside[1:].sum()),
            "export_footprint_below_roof_ris": bool(inside[0]),
            "native_ground_pass_ues": int(valid[1:].sum()), "native_ground_pass_ris": bool(valid[0]),
            "native_ue_above_or_on_first_surface_count": int((users[:, 2] >= z[1:]).sum()),
            "native_ris_hit_z_m": float(z[0]) if np.isfinite(z[0]) else None,
            "native_ris_hit_shape": first_shape(scene, ris[:2]),
            "native_ue_hit_z_counts": {str(float(v)): int(c) for v, c in zip(*np.unique(np.round(z[1:], 2), return_counts=True))},
            "first_invalid_ue_index": int(invalid[0]-1) if len(invalid) else None,
            "first_invalid_ue_hit_shape": first_shape(scene, points[invalid[0], :2]) if len(invalid) else None,
            "demo": checks, "source_material_count": len(material_elements),
            "source_material_thickness_values_m": sorted(set(thickness)), "source_materials_with_explicit_thickness": len(thickness)}


def array_scene(scene=None, dual=True):
    if scene is None:
        scene = rt.load_scene()
    scene.frequency = 5.53e9
    scene.tx_array = scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
        pattern="iso", polarization="VH" if dual else "V"
    )
    return scene


def solve(scene, **kwargs):
    opts = dict(max_depth=0, synthetic_array=False, los=True,
                specular_reflection=False, diffuse_reflection=False, refraction=False,
                diffraction=False, edge_diffraction=False, samples_per_src=20000,
                max_num_paths_per_src=40000, seed=7)
    opts.update(kwargs)
    start = time.perf_counter()
    paths = rt.PathSolver()(scene=scene, **opts)
    a, _ = paths.cir(normalize_delays=False, out_type="numpy")
    dr.sync_thread()
    return paths, a, time.perf_counter() - start


def probes(n):
    scene = array_scene()
    scene.add(rt.Transmitter("tx", position=[-12, 0, 1.5]))
    for k in range(n):
        scene.add(rt.Receiver(f"p{k}", position=[0, (k//8 - (n/8-1)/2)*.028, 1.5+(k%8-3.5)*.03]))
    seconds = []
    for _ in range(3):
        _, a, elapsed = solve(scene)
        seconds.append(elapsed)
    assert a.shape == (n, 2, 1, 2, 1, 1), a.shape
    j = a[:, :, 0, :, 0, 0]
    diag = np.abs(j[:, [0, 1], [0, 1]])
    cross = np.abs(j[:, [0, 1], [1, 0]])
    passed = bool(np.isfinite(j).all() and np.all(diag > 0))
    return {"probe_count": n, "cir_shape": list(a.shape), "seconds": seconds,
            "min_copol_abs": float(diag.min()), "max_crosspol_abs": float(cross.max()),
            "pass": passed, "peak_process_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def wall(out, thickness):
    # Rectangle normal to x: the only obstacle in a manufactured scene.
    ply = out / "synthetic_wall.ply"
    ply.write_text("""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0 -1 0
0 1 0
0 1 2
0 -1 2
3 0 1 2
3 0 2 3
""")
    xml = out / "synthetic_wall.xml"
    xml.write_text(f'''<scene version="2.1.0">
      <bsdf type="itu-radio-material" id="brick">
        <string name="type" value="brick"/>
        <float name="thickness" value="{thickness}"/>
      </bsdf>
      <shape type="ply" id="mesh-wall">
        <string name="filename" value="{ply.name}"/>
        <ref id="brick" name="bsdf"/>
      </shape></scene>''')
    s = array_scene(rt.load_scene(str(xml)), dual=False)
    s.add(rt.Transmitter("tx", position=[-4, -.15, 1]))
    s.add(rt.Receiver("rx", position=[4, .3, 1.05]))
    return s


def interactions(paths, a):
    return {"nonzero_cir": int(np.count_nonzero(a)), "coefficient_abs": np.abs(a[a != 0]).tolist(),
            "interaction_histogram": {str(int(x)): int(n) for x, n in zip(*np.unique(paths.interactions.numpy(), return_counts=True))}}


def capabilities(out):
    results = [probes(n) for n in (64, 128, 256)]
    thin, thick = wall(out, .05), wall(out, .20)
    off, a, _ = solve(thin, max_depth=1)
    dif, b, _ = solve(thin, max_depth=1, diffraction=True, edge_diffraction=True)
    t1, c, _ = solve(thin, max_depth=1, refraction=True)
    t2, d, _ = solve(thick, max_depth=1, refraction=True)
    return {"explicit_dual_pol_free_space": results,
            "wall_disabled": interactions(off, a), "wall_diffraction": interactions(dif, b),
            "wall_transmission_005m": interactions(t1, c), "wall_transmission_020m": interactions(t2, d),
            "diffraction_pass": bool(not np.count_nonzero(a) and np.count_nonzero(b)
                                     and (dif.interactions.numpy() == InteractionType.DIFFRACTION).any()),
            "transmission_pass": bool(not np.count_nonzero(a) and np.count_nonzero(c)
                                      and np.count_nonzero(d) and abs(abs(c).max()-abs(d).max()) > 1e-12)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["geometry", "capabilities"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--geometry-input", type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    versions = {n: version(n) for n in ("sionna-rt", "mitsuba", "drjit", "numpy")}
    if versions["sionna-rt"] != "2.0.1":
        raise RuntimeError(versions)
    if args.mode == "geometry":
        src = json.loads(args.geometry_input.read_text())
        result = geometry(src["layout"], out, np.asarray(src["footprint"]))
    else:
        result = capabilities(out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    (out / f"{args.mode}.json").write_text(json.dumps(
        {"versions": versions, "mitsuba_variant": mi.variant(), **result},
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
