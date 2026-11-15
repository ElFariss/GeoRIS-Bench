#!/usr/bin/env python3
"""Berlin held-out inputs: merged LoD2 GML (4 tiles, dedup by gml:id) and a 1 m DTM GeoTIFF from DGM1 xyz
(area bounds + 60 m), plus the Berlin row of heldout_inputs/source-manifest.json. Geometry validity only."""
import hashlib, json, zipfile
from pathlib import Path
import numpy as np
from atomwise_ris.bench import heldout_geometry_v3x as hg, berlin_lod2_v3x as bl

H = hg.H; S = H / "src"; C = H / "cache"
xmls = sorted((S / "berlin_gml").glob("LoD2_33_*_BE.xml"))
stats = bl.merge_tiles(xmls, C / "berlin_lod2_merged.gml")
x0, y0, x1, y1 = hg.AREAS["berlin"]["bounds"]; m = 60.
with zipfile.ZipFile(S / "DGM1_390_5818.zip") as z:
    name = [n for n in z.namelist() if n.endswith(".xyz")][0]
    with z.open(name) as f:
        a = np.loadtxt(f, usecols=(0, 1, 2))
sel = (a[:, 0] >= x0 - m) & (a[:, 0] < x1 + m) & (a[:, 1] >= y0 - m) & (a[:, 1] < y1 + m)
a = a[sel]
xs = np.unique(a[:, 0]); ys = np.unique(a[:, 1])
dx = float(np.median(np.diff(xs))); dy = float(np.median(np.diff(ys)))
assert abs(dx - 1) < 1e-6 and abs(dy - 1) < 1e-6, (dx, dy)
nx, ny = len(xs), len(ys)
grid = np.full((ny, nx), np.nan, dtype="float32")
ci = np.rint(a[:, 0] - xs[0]).astype(int); ri = np.rint(ys[-1] - a[:, 1]).astype(int)
grid[ri, ci] = a[:, 2]
import rasterio
from rasterio.transform import from_origin
tr = from_origin(xs[0] - 0.5, ys[-1] + 0.5, 1.0, 1.0)   # xyz are cell centres
with rasterio.open(C / "berlin.tif", "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
                   crs="EPSG:25833", transform=tr, nodata=np.nan) as d:
    d.write(grid, 1)
def ent(p, url, extra=None):
    return {"file": p.name, "url": url, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), **(extra or {})}
aud = {r["id"]: r for r in json.loads((hg.ROOT / "outputs/audits/bench-phase0b-20261108/map_downloads.json").read_text())}["berlin"]
for b in aud["buildings"] + [aud["terrain"]]:
    assert hashlib.sha256((S / b["file"]).read_bytes()).hexdigest() == b["sha256"], b["file"]
row = {"id": "berlin",
       "buildings": [ent(C / "berlin_lod2_merged.gml", "derived: merge of " + ", ".join(b["url"] for b in aud["buildings"]),
                         {"derived_from": [{k: b[k] for k in ("file", "url", "sha256")} for b in aud["buildings"]], **stats})],
       "terrain": ent(C / "berlin.tif", "derived: " + aud["terrain"]["url"],
                      {"derived_from": {k: aud["terrain"][k] for k in ("file", "url", "sha256")}, "grid": [ny, nx],
                       "nan_cells": int(np.isnan(grid).sum())}),
       "osm": ent(C / "berlin.osm", aud["osm"]["url"], {"audit_sha256": aud["osm"]["sha256"],
                  "note": "live OSM API snapshot fetched 2026-11-14; differs from the 2026-11-08 audit snapshot"})}
man = [r for r in json.loads((H / "source-manifest.json").read_text()) if r["id"] != "berlin"] + [row]
(H / "source-manifest.json").write_text(json.dumps(man, indent=2))
print(json.dumps({"merge": stats, "grid": [ny, nx], "nan": int(np.isnan(grid).sum()), "z": [float(np.nanmin(grid)), float(np.nanmax(grid))]}))
