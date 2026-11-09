"""Export one CoupledRIS-Bench preview case to ParaView (VTK XML, no VTK dependency).

Usage (GIS env with pyarrow):
  python scripts/export_case_paraview.py outputs/previews/bench-users-v1 02-munich-natural [--every 5]
Writes outputs/paraview/<case>/:
  terrain.vts      ground model (z = NaN where no data; scalar 'valid'), local coordinates
  buildings.vtp    footprints extruded from building ground to ground + height (readability only)
  sites.vtp        BS and RIS points plus the BS-RIS segment
  paths.vtp        one polyline per user over the whole clip
  users.pvd        time series of user points (open this and press Play)
Coordinates are shifted by the map's lower-left bound so ParaView keeps full precision; 'origin.json' records it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LINK = {"los": 0, "no_los_reflection_candidate": 1, "no_los": 2}
HEIGHT = {"pedestrian": 0, "elevated": 1, "drone": 2}


def arr(name, values, comps=1, kind="Float32"):
    v = " ".join(f"{x:.4f}" if kind.startswith("Float") else str(int(x)) for x in np.ravel(values))
    return f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{comps}" format="ascii">{v}</DataArray>'


def write_vtp(path, points, verts=(), lines=(), polys=(), point_data=None):
    def cells(groups):
        conn = [i for g in groups for i in g]
        offs = np.cumsum([len(g) for g in groups]) if groups else []
        return arr("connectivity", conn, kind="Int64") + arr("offsets", offs, kind="Int64")
    pd_xml = "".join(arr(k, v, kind=t) for k, (v, t) in (point_data or {}).items())
    path.write_text(
        '<?xml version="1.0"?><VTKFile type="PolyData" version="1.0" byte_order="LittleEndian"><PolyData>'
        f'<Piece NumberOfPoints="{len(points)}" NumberOfVerts="{len(verts)}" NumberOfLines="{len(lines)}" '
        f'NumberOfStrips="0" NumberOfPolys="{len(polys)}">'
        f'<PointData>{pd_xml}</PointData><Points>{arr("Points", points, 3)}</Points>'
        f'<Verts>{cells(list(verts))}</Verts><Lines>{cells(list(lines))}</Lines><Polys>{cells(list(polys))}</Polys>'
        '</Piece></PolyData></VTKFile>')


VIEW_SCRIPT = """# Load and style this case in ParaView: Tools -> Python Script Editor -> File -> Open this file -> Run,
# or start ParaView with: flatpak run org.paraview.ParaView --script=<path to this file>
import os
from paraview.simple import *
d = "__CASE_DIR__"  # absolute path, written at export (the Script Editor has no __file__)
v = GetActiveViewOrCreate("RenderView")
t0 = OpenDataFile(os.path.join(d, "terrain.vts"))
t = Threshold(Input=t0); t.Scalars = ["POINTS", "valid"]; t.LowerThreshold = 0.5; t.UpperThreshold = 1.5; t.AllScalars = 1  # hide no-data cells
td = Show(t, v); ColorBy(td, ("POINTS", "elevation_m")); td.Opacity = 0.6
b = OpenDataFile(os.path.join(d, "buildings.vtp")); bd = Show(b, v); bd.ColorArrayName = ["POINTS", ""]; bd.AmbientColor = bd.DiffuseColor = [0.75, 0.75, 0.8]; bd.Opacity = 0.55
p = OpenDataFile(os.path.join(d, "paths.vtp")); pd = Show(p, v); pd.ColorArrayName = ["POINTS", ""]; pd.DiffuseColor = [0.3, 0.3, 0.3]; pd.LineWidth = 1.5
s = OpenDataFile(os.path.join(d, "sites.vtp")); sd = Show(s, v); ColorBy(sd, ("POINTS", "site")); sd.SetRepresentationType("Points"); sd.PointSize = 18; sd.RenderPointsAsSpheres = 1
sl = Show(PassArrays(Input=s), v); sl.SetRepresentationType("Wireframe"); sl.ColorArrayName = ["POINTS", ""]; sl.DiffuseColor = sl.AmbientColor = [0.1, 0.1, 0.6]; sl.LineWidth = 3  # BS-RIS line
slut = GetColorTransferFunction("site"); slut.InterpretValuesAsCategories = 1
slut.Annotations = ["0", "BS", "1", "RIS"]; slut.IndexedColors = [0.1, 0.2, 0.75, 0.75, 0.1, 0.6]
sd.SetScalarBarVisibility(v, True)
u = OpenDataFile(os.path.join(d, "users.pvd")); ud = Show(u, v); ColorBy(ud, ("POINTS", "bs_link")); ud.SetRepresentationType("Points"); ud.PointSize = 10; ud.RenderPointsAsSpheres = 1
lut = GetColorTransferFunction("bs_link"); lut.InterpretValuesAsCategories = 1
lut.Annotations = ["0", "LOS", "1", "reflection candidate", "2", "no LOS"]; lut.IndexedColors = [0.1, 0.6, 0.3, 0.95, 0.6, 0.1, 0.8, 0.15, 0.2]
ud.SetScalarBarVisibility(v, True)
GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
v.CameraPosition = [0, -600, 700]; ResetCamera(v); Render(v)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("case")
    ap.add_argument("--every", type=int, default=5, help="export every Nth frame (default 5 = 100 ms)")
    ap.add_argument("--terrain-stride", type=int, default=2)
    ap.add_argument("--out", default="outputs/paraview", help="parent folder for the case export")
    a = ap.parse_args()
    root = Path(a.root)
    case_dir = root / "cases" / a.case
    meta = json.loads((case_dir / "case.json").read_text())
    geo = json.loads((root / "geometry" / f"{meta['map_id']}.json").read_text())
    out = Path(a.out) / a.case
    out.mkdir(parents=True, exist_ok=True)
    b = geo["bounds"]  # [xmin, xmax, ymin, ymax]
    xmin, ymin = (b[0], b[2]) if isinstance(b, list) else (b["xmin"], b["ymin"])
    o = np.array([xmin, ymin, 0.0])
    (out / "frames").mkdir(exist_ok=True)
    (out / "origin.txt").write_text(json.dumps({"crs": geo.get("crs"), "origin_xy": [xmin, ymin]}) + "\n")

    # Terrain as a structured grid.
    t = np.load(root / "geometry" / f"{meta['map_id']}-terrain.npz")
    s = a.terrain_stride
    xmax, ymax = (b[1], b[3]) if isinstance(b, list) else (b["xmax"], b["ymax"])
    ix = np.flatnonzero((t["x"] >= xmin - 20) & (t["x"] <= xmax + 20))[::s]   # crop to the map area + 20 m
    iy = np.flatnonzero((t["y"] >= ymin - 20) & (t["y"] <= ymax + 20))[::s]
    x, y = t["x"][ix] - xmin, t["y"][iy] - ymin
    z, valid = t["z"][np.ix_(iy, ix)].astype(float), t["valid"][np.ix_(iy, ix)]
    z = np.where(valid, z, np.nan)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])
    (out / "terrain.vts").write_text(
        '<?xml version="1.0"?><VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">'
        f'<StructuredGrid WholeExtent="0 {len(x)-1} 0 {len(y)-1} 0 0"><Piece Extent="0 {len(x)-1} 0 {len(y)-1} 0 0">'
        f'<PointData>{arr("valid", valid.ravel(), kind="Int8")}{arr("elevation_m", np.nan_to_num(z).ravel())}</PointData>'
        f'<Points>{arr("Points", np.nan_to_num(pts, nan=float(np.nanmin(z))), 3)}</Points></Piece></StructuredGrid></VTKFile>')

    # Buildings: extruded footprints (walls + roof), as in the PNG previews.
    bp, polys, bh = [], [], []
    for b in geo["buildings"]:
        poly = np.asarray(json.loads(b["polygon_xy"]) if isinstance(b["polygon_xy"], str) else b["polygon_xy"], float)
        if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if len(poly) < 3:
            continue
        g, h = float(b["ground_m"]), float(b["height_m"])
        n, k = len(poly), len(bp)
        for zz in (g, g + h):
            for px, py in poly:
                bp.append([px - xmin, py - ymin, zz]); bh.append(h)
        polys.append(list(range(k + n, k + 2 * n)))  # roof
        polys += [[k + i, k + (i + 1) % n, k + n + (i + 1) % n, k + n + i] for i in range(n)]  # walls
    write_vtp(out / "buildings.vtp", np.array(bp), polys=polys, point_data={"height_m": (np.array(bh), "Float32")})

    # Sites.
    bs, ris = np.array(meta["bs_xyz"], float) - o, np.array(meta["ris_xyz"], float) - o
    write_vtp(out / "sites.vtp", np.array([bs, ris]), verts=[[0], [1]], lines=[[0, 1]],
              point_data={"site": (np.array([0, 1]), "Int8")})

    # Users: per-frame points and whole-clip paths.
    df = pq.read_table(case_dir / "frames.parquet").to_pandas()
    df["link"] = df.bs_link.map(LINK).fillna(-1)
    df["hcls"] = df.height_class.map(HEIGHT).fillna(-1)
    paths_pts, paths = [], []
    for _, u in df.sort_values("frame").groupby("user_id"):
        k = len(paths_pts)
        paths_pts += (u[["x", "y", "z"]].to_numpy() - o).tolist()
        paths.append(list(range(k, k + len(u))))
    write_vtp(out / "paths.vtp", np.array(paths_pts), lines=paths)
    frames = sorted(df.frame.unique())[:: a.every]
    entries = []
    for f in frames:
        u = df[df.frame == f]
        p = u[["x", "y", "z"]].to_numpy() - o
        name = f"frames/users_{int(f):04d}.vtp"
        write_vtp(out / name, p, verts=[[i] for i in range(len(p))], point_data={
            "bs_link": (u.link.to_numpy(), "Int8"), "ris_los": (u.ris_los.astype(int).to_numpy(), "Int8"),
            "served": (u.served.astype(int).to_numpy(), "Int8"), "height_class": (u.hcls.to_numpy(), "Int8"),
            "user_id": (u.user_id.astype(int).to_numpy(), "Int32"), "height_agl_m": (u.height_agl_m.to_numpy(), "Float32")})
        entries.append(f'<DataSet timestep="{float(u.time_s.iloc[0]):.3f}" file="{name}"/>')
    (out / "users.pvd").write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1"><Collection>'
                                   + "".join(entries) + "</Collection></VTKFile>")
    (out / "view_case.py").write_text(VIEW_SCRIPT.replace("__CASE_DIR__", str(out.resolve())))
    print(json.dumps({"out": str(out), "frames": len(frames), "users": int(df.user_id.nunique()),
                      "buildings": len(geo["buildings"]), "codes": {"bs_link": LINK, "height_class": HEIGHT,
                                                                    "site": {"BS": 0, "RIS": 1}}}))


if __name__ == "__main__":
    main()
