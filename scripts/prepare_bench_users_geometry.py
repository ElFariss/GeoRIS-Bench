#!/usr/bin/env python3
"""Build the checksum-verified development-only geometry and slope gate.

v1: one DTM-validity rule (4x4 stencil), stairs excluded, no v0 site pool (sites
are planned by the generator). Writes under ``--out`` (default: the v1 preview).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomwise_ris.bench.users_geometry import DEV_IDS, load_development_maps


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/previews/bench-users-v1/geometry"


def compare_baseline(out, baseline, summary):
    """Prove that a refreshed OSM snapshot did not change the saved v1 geometry gate."""
    import numpy as np

    previous=json.loads((baseline/"slope_gate.json").read_text())
    out_checks={}
    for k in DEV_IDS:
        a,b=summary["maps"][k],previous["maps"][k]
        keys=set(a)-{"checksummed_sources"}
        if any(a[key]!=b[key] for key in keys):
            raise ValueError(f"{k}: refreshed source changed v1 geometry gate fields: "
                             f"{sorted(key for key in keys if a[key]!=b[key])}")
        old_sources,new_sources=b["checksummed_sources"],a["checksummed_sources"]
        if {s:h for s,h in old_sources.items() if not s.endswith(".osm")} != {
                s:h for s,h in new_sources.items() if not s.endswith(".osm")}:
            raise ValueError(f"{k}: changed non-OSM source")
        for name in ("buildings","bounds","crs"):
            old=json.loads((baseline/f"{k}.json").read_text())[name]
            new=json.loads((out/f"{k}.json").read_text())[name]
            if old!=new:
                raise ValueError(f"{k}: refreshed source changed exported {name}")
        with np.load(baseline/f"{k}-terrain.npz",allow_pickle=False) as old, \
             np.load(out/f"{k}-terrain.npz",allow_pickle=False) as new:
            for field in ("x","y","z","valid"):
                np.testing.assert_array_equal(old[field],new[field])
        out_checks[k]={"equal_geometry_gate_fields":sorted(keys),
                       "equal_exports":["buildings","bounds","crs","terrain arrays"],
                       "osm_sha256_original":old_sources[f"{k}.osm"],
                       "osm_sha256_recovered":new_sources[f"{k}.osm"]}
    return out_checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--cache",default="/tmp/atomwise-phase0b-input")
    ap.add_argument("--source-manifest",type=Path)
    ap.add_argument("--baseline-geometry",type=Path)
    args=ap.parse_args()
    out=args.out
    cities=load_development_maps(cache_root=args.cache,area_ids=DEV_IDS,export_dir=out,build_sites=False,
                                 manifest_path=args.source_manifest)
    summary={"scope":"geometry/slope only; development maps only; v1 4x4 DTM validity rule","maps":{}}
    for map_id,city in cities.items():
        summary["maps"][map_id]={
            "crs":city.crs,"bounds_m":list(city.bounds),
            "checksummed_sources":city.source_hashes,
            "lod2_buildings":len(city.buildings),
            "building_height_sources":{
                key:sum(1 for b in city.buildings if key in b.height_source)
                for key in sorted({"native surfaces","partial shell","slab fallback"})},
            "street_segments_valid":int(len(city.street_segments)),
            "street_hotspots":int(len(city.hotspots_xy)),
            "street_slope":city.slope_summary(),
            "geometry_rejections":getattr(city,"geometry_rejections",{}),
            "facade_candidates":int(len(city.facade_xyz)),
            "roof_candidates":int(len(city.roof_xyz)),
            "wall_triangles":int(len(city.wall_triangles)),
            "embree_enabled":city._ray_intersector is not None,
        }
        print(map_id,json.dumps(summary["maps"][map_id]["street_slope"]),json.dumps(summary["maps"][map_id]["geometry_rejections"]),flush=True)
    if args.baseline_geometry:
        summary["v1_source_recovery_checks"]=compare_baseline(out,args.baseline_geometry,summary)
    path=out/"slope_gate.json"
    path.write_text(json.dumps(summary,indent=2)+"\n")
    print("GEOMETRY_GATE_WRITTEN",path,flush=True)


if __name__=="__main__": main()
