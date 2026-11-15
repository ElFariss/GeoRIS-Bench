#!/usr/bin/env python3
"""Utrecht geometry-validity fix (cross-city declaration, preflight B, Fix U1).

The G5 world z-bound check failed on the first Utrecht scene: 3DBAG BuildingPart NL.IMBAG.Pand.0344100000052254-0
(3DBAG val3dity LoD2.2 error [302]) has Wall/Roof faces 2.96 m below its own GroundSurface (roof at -1.08 m NAP,
ground 1.88 m). Rule (geometry validity only, decided before any channel): in an LoD2.2 solid whose val3dity
list is non-empty, every Wall/Roof face with a vertex more than 0.5 m below the part's lowest GroundSurface vertex
is removed; nothing else changes. Writes derived tiles <tile>.v3x-valid.city.json.gz into heldout_inputs/cache and
updates the Utrecht building rows of heldout_inputs/source-manifest.json; logs every removed face and its area.
"""
import gzip, hashlib, json
from pathlib import Path
import numpy as np
from atomwise_ris.bench import heldout_geometry_v3x as hg

H = hg.H
man = json.loads((H / "source-manifest.json").read_text())
row = [r for r in man if r["id"] == "utrecht"][0]
log = {"rule": __doc__.split("Rule")[1].strip(), "tiles": []}
new_rows = []
for b in row["buildings"]:
    src = H / "src" / (b.get("original_file") or b["file"])
    d = json.load(gzip.open(src, "rt"))
    v = np.asarray(d["vertices"], float) * d["transform"]["scale"] + d["transform"]["translate"]
    removed = []
    for k, o in d["CityObjects"].items():
        if o.get("type") != "BuildingPart":
            continue
        parent = d["CityObjects"].get(o.get("parents", [None])[0], {})
        val = str(parent.get("attributes", {}).get("b3_val3dity_lod22", "[]"))
        if val in ("[]", "", "None"):
            continue
        for g in o.get("geometry", []):
            if g.get("lod") != "2.2" or g.get("type") != "Solid":
                continue
            sem = g["semantics"]
            ground = [v[[i for r in f for i in r]][:, 2].min()
                      for sh, sv in zip(g["boundaries"], sem["values"]) for f, s in zip(sh, sv)
                      if s is not None and sem["surfaces"][s]["type"] == "GroundSurface"]
            if not ground:
                continue
            z0 = min(ground)
            for si, (sh, sv) in enumerate(zip(g["boundaries"], sem["values"])):
                keep_f, keep_s = [], []
                for f, s in zip(sh, sv):
                    typ = sem["surfaces"][s]["type"] if s is not None else None
                    pts = v[[i for r in f for i in r]]
                    if typ in ("WallSurface", "RoofSurface") and pts[:, 2].min() < z0 - 0.5:
                        p = pts[:, :3]; area = 0.5 * np.linalg.norm(sum(np.cross(p[i] - p[0], p[i + 1] - p[0]) for i in range(1, len(p) - 1)))
                        removed.append({"part": k, "type": typ, "z_min": float(pts[:, 2].min()), "ground_z": float(z0), "area_m2": float(area),
                                        "xy": np.round(pts[:, :2].mean(0), 2).tolist(), "val3dity": val})
                        continue
                    keep_f.append(f); keep_s.append(s)
                g["boundaries"][si] = keep_f; sem["values"][si] = keep_s
    out = H / "cache" / src.name.replace(".city.json.gz", ".v3x-valid.city.json.gz")
    with gzip.open(out, "wt", encoding="utf8") as f:
        json.dump(d, f)
    log["tiles"].append({"tile": src.name, "derived": out.name, "removed_faces": removed})
    new_rows.append({"file": out.name, "url": "derived (Fix U1): " + b["url"], "bytes": out.stat().st_size,
                     "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "tile": b.get("tile"),
                     "original_file": src.name, "original_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                     "removed_faces": len(removed)})
row["buildings"] = new_rows
(H / "source-manifest.json").write_text(json.dumps(man, indent=2))
(H / "utrecht-fixU1.json").write_text(json.dumps(log, indent=2))
print(json.dumps([{t["tile"]: [(r["part"], r["type"], round(r["z_min"], 2), round(r["area_m2"], 2)) for r in t["removed_faces"]]} for t in log["tiles"]]))
