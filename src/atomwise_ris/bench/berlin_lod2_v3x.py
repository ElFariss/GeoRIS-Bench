"""Berlin LoD2 (CityGML 1.0, AdV profile, EPSG:25833 + DHHN2016) for the cross-city held-out test only.

Berlin LoD2 uses the same CityGML 1.0 AdV structure as the Bavarian (Munich) LoD2, so the unchanged Munich
parsers are reused:
- users_geometry._munich_buildings for the user/site geometry;
- the production trace script's _munich_scene for the ray-tracing scene (same mesh-repair policy and audit).
The only differences are the source file (one merged GML of the 4 Berlin tiles, deduplicated by gml:id),
the CRS label and the map id. docs/bench-ris-relevant-v3-crosscity-20261114.md.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from pathlib import Path


def berlin_buildings(files, bounds):
    from . import users_geometry as ug
    out, seen = [], set()
    for f in files:
        for b in ug._munich_buildings(Path(f), bounds):
            if b.id in seen:
                continue
            seen.add(b.id)
            b.height_source = b.height_source.replace("Bavaria", "Berlin")
            out.append(b)
    return out


def merge_tiles(xml_files, out_path):
    """One CityModel with the cityObjectMembers of all tiles (first occurrence of each Building gml:id kept)."""
    import xml.etree.ElementTree as ET
    head = None; seen = set(); members = []; dup = 0
    for f in xml_files:
        txt = Path(f).read_text(encoding="utf-8")
        if head is None:
            head = txt[: txt.index("<core:cityObjectMember>")]
        parts = txt.split("<core:cityObjectMember>")[1:]
        for p in parts:
            body = p.split("</core:cityObjectMember>")[0]
            i = body.find('gml:id="')
            gid = body[i + 8: body.index('"', i + 8)] if i >= 0 else None
            if gid in seen:
                dup += 1
                continue
            seen.add(gid)
            members.append("<core:cityObjectMember>" + body + "</core:cityObjectMember>")
    Path(out_path).write_text(head + "\n".join(members) + "\n</core:CityModel>\n", encoding="utf-8")
    ET.parse(out_path)  # must be well formed
    return {"members": len(members), "duplicates_dropped": dup}


def berlin_scene(T, raw_dir, scene_dir, case, rows, anchor_users=None):
    b = rows["buildings"][0]
    src = {"file": b["file"], "url": b.get("url", "derived: merged Berlin LoD2 tiles"), "bytes": int(b["bytes"]), "sha256": b["sha256"]}
    p = Path(raw_dir) / src["file"]
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != src["sha256"]:
        raise RuntimeError(f"Berlin merged LoD2 source missing or hash mismatch: {p}")
    code = textwrap.dedent(inspect.getsource(T._munich_scene))
    rep = [("def _munich_scene(", "def _berlin_scene("),
           ('    source = raw_dir / native.MUNICH["file"]\n    native.download(source, native.MUNICH["url"], native.MUNICH["bytes"], native.MUNICH["sha256"])\n',
            '    source = raw_dir / BERLIN_SOURCE["file"]\n'),
           ('f"munich_bldg_{len(per_building)}"', 'f"berlin_bldg_{len(per_building)}"'),
           ('"Munich scene build produced no LoD2 surfaces in case bounds"', '"Berlin scene build produced no LoD2 surfaces in case bounds"'),
           ('        "munich", per_class, per_building, (z_min_w, z_max_w), max_bldg_h', '        "berlin", per_class, per_building, (z_min_w, z_max_w), max_bldg_h'),
           ('"map": "munich",', '"map": "berlin",'), ('"source_crs": "EPSG:25832",', '"source_crs": "EPSG:25833",'),
           ('"source": native.MUNICH,', '"source": BERLIN_SOURCE,'),
           ('"scene_bounds_epsg25832_m": bounds,', '"scene_bounds_epsg25833_m": bounds, "origin_projected_m": origin.tolist(),')]
    for a, c in rep:
        if code.count(a) != 1:
            raise RuntimeError(f"Munich scene source changed; cannot derive the Berlin builder ({a[:40]!r})")
        code = code.replace(a, c)
    ns = dict(T.__dict__); ns["BERLIN_SOURCE"] = src
    exec(compile(code, "<berlin_scene from _munich_scene>", "exec"), ns)
    return ns["_berlin_scene"](Path(raw_dir), scene_dir, case, anchor_users=anchor_users)
