"""Declared, user-authorised admission of the held-out maps (Utrecht, Berlin) for the cross-city test only.

docs/bench-ris-relevant-v3-crosscity-20261114.md. Everything here is process-local: the frozen modules
(users, users_geometry, the production trace script) are not edited; their module-level tables are extended
in the running process. Held-out inputs live under outputs/runs/bench_ris_relevant_v3x_20261114/heldout_inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .production_freeze import ROOT

H = ROOT / "outputs/runs/bench_ris_relevant_v3x_20261114/heldout_inputs"
AREAS = {  # from configs/bench/phase0b-area-proposals-20261108.json (bounds_m: xmin, ymin, xmax, ymax)
    "utrecht": {"crs": "EPSG:28992", "bounds": (135145., 455634., 135585., 456074.), "kind": "netherlands"},
    "berlin": {"crs": "EPSG:25833", "bounds": (390788., 5818756., 391228., 5819196.), "kind": "berlin"},
}


def source_rows(city):
    return {r["id"]: r for r in json.loads((H / "source-manifest.json").read_text())}[city]


def admit(city):
    from . import users, users_geometry as ug
    if city not in AREAS:
        raise ValueError(city)
    if city not in ug.DEV_IDS:
        ug.DEV_IDS = tuple(ug.DEV_IDS) + (city,)
    ug.AREAS[city] = AREAS[city]
    if city not in users.DEVELOPMENT:
        users.DEVELOPMENT = tuple(users.DEVELOPMENT) + (city,)
    try:   # case compactor of scripts/build_bench_production_cases.py (frozen script; module table only)
        import build_bench_production_cases as bpc
        if city not in bpc.DEVELOPMENT:
            bpc.DEVELOPMENT = tuple(bpc.DEVELOPMENT) + (city,)
    except ImportError:
        pass
    if city == "berlin" and not getattr(ug, "_v3x_berlin", False):
        from . import berlin_lod2_v3x
        orig = ug._netherlands_buildings

        def buildings(files, bounds):
            if all(Path(f).suffix == ".gml" for f in files):
                return berlin_lod2_v3x.berlin_buildings(files, bounds)
            return orig(files, bounds)
        ug._netherlands_buildings = buildings
        ug._v3x_berlin = True


def admit_trace(T, city):
    """Extend the production trace script (module T) for a held-out city, in this process only."""
    rows = source_rows(city)
    orig_ref = T._load_v11_reference

    def ref(map_id):
        if map_id != city:
            return orig_ref(map_id)
        g = json.loads((H / "geometry" / f"{city}.json").read_text())
        with np.load(H / "geometry" / f"{city}-terrain.npz", allow_pickle=False) as t:
            tz = np.asarray(t["z"], dtype=np.float64)
        return {"bounds": [float(x) for x in g["bounds"]], "terrain_z_min_m": float(np.nanmin(tz)),
                "terrain_z_max_m": float(np.nanmax(tz)),
                "v11_max_building_height_m": float(max(b["height_m"] for b in g["buildings"])),
                "v11_ground_by_id": {b["id"]: float(b["ground_m"]) for b in g["buildings"]}}
    T._load_v11_reference = ref
    if city == "utrecht":
        T.THREE_DBAG_TILES[city] = [(b["file"], b["url"], int(b["bytes"]), b["sha256"]) for b in rows["buildings"]]
        builder = lambda raw, sd, case, anchor_users=None: T._three_dbag_scene(city, raw, sd, case, anchor_users=anchor_users)
    else:
        from . import berlin_lod2_v3x
        builder = lambda raw, sd, case, anchor_users=None: berlin_lod2_v3x.berlin_scene(T, raw, sd, case, rows, anchor_users=anchor_users)
    orig_build = T.build_scene

    def build_scene(case, root, out, production=False, scene_dir=None, anchor_users=None):
        if case["map_id"] != city:
            return orig_build(case, root, out, production=production, scene_dir=scene_dir, anchor_users=anchor_users)
        if scene_dir is None:
            scene_dir = root / "scenes" / city / case["id"] if production else root / "scenes" / city
        saved = T._rotterdam_scene
        T._rotterdam_scene = lambda raw, sd, c, anchor_users=None: builder(raw, sd, case, anchor_users=anchor_users)
        try:   # reuse the unchanged dispatch + endpoint preflight; the 'rotterdam' branch now calls the held-out builder
            return orig_build({**case, "map_id": "rotterdam"}, root, out, production=production,
                              scene_dir=scene_dir, anchor_users=anchor_users)
        finally:
            T._rotterdam_scene = saved
    T.build_scene = build_scene
