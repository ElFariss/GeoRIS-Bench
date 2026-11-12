#!/usr/bin/env python3
"""Make compact approved pilot inputs or sample new cases with frozen v1.1."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from atomwise_ris.bench.production_freeze import ROOT, sha256, verify_freeze

PREVIEW = ROOT / "outputs/previews/bench-users-v1.1"
DEVELOPMENT = ("munich", "rotterdam", "maastricht")
PILOT = ("01-munich-natural", "15-rotterdam-natural")
PILOT2 = (*PILOT, "39-maastricht-natural")
CHALLENGE_ORDER = ("shadow_edge", "shadow_edge_sloped", "all_no_los", "dense_crowd", "elevated_crowd")


def compact(source, case, frames):
    f = source / case["id"] / "frames.parquet"
    if sha256(f) != case["frames_sha256"]:
        raise ValueError(f"case Parquet failed hash validation: {f}")
    if case["map_id"] not in DEVELOPMENT:
        raise ValueError("only development maps are admitted")
    frame_count = int(case["frame_count"])
    neighbors = {j for i in frames for j in (max(i-1,0),i,min(i+1,frame_count-1))}
    cols = ["frame","time_s","user_id","x","y","z","served","bs_link","ris_los","height_class"]
    rows = pq.read_table(f, columns=cols, filters=[("frame","in",sorted(neighbors)),
                                                   ("served","==",True)]).to_pylist()
    keyed = {(row["frame"],row["user_id"]):row for row in rows}
    frames_data = []
    xyz = lambda x: np.asarray([x[k] for k in ("x","y","z")],dtype=float)
    for i in frames:
        users=[]
        for k in case["served_user_ids"]:
            row=keyed[(i,k)]
            u0=keyed[(max(0,i-1),k)]
            u1=keyed[(min(frame_count-1,i+1),k)]
            dt=float(u1["time_s"]-u0["time_s"])
            vel=(xyz(u1)-xyz(u0))/dt if dt>0 else np.zeros(3)
            users.append({"user_id":k,"xyz":xyz(row).tolist(),"velocity_xyz":vel.tolist(),
                          "bs_link":row["bs_link"],"ris_los":row["ris_los"],
                          "height_class":row["height_class"]})
        frames_data.append({"frame":i,"time_s":float(row["time_s"]),"users":users})
    bs,ris=np.asarray(case["bs_xyz"]),np.asarray(case["ris_xyz"])
    normal=bs-ris
    normal[2]=0
    normal/=np.linalg.norm(normal)
    return {"id":case["id"],"map_id":case["map_id"],"site_id":case["site_id"],
            "bs_xyz":bs.tolist(),"ris_xyz":ris.tolist(),"ris_normal_xyz":normal.tolist(),
            "bs_ris_geometry_los":case["bs_ris_los"],
            "frame_source_sha256":case["frames_sha256"],"frames":frames_data}


def approved_pilot(ids=PILOT):
    m=json.loads((PREVIEW/"case_manifest.json").read_text())
    cases={c["id"]:c for c in m["cases"]}
    out=[]
    for i in ids:
        x=cases[i]
        if json.loads((PREVIEW/"cases"/i/"case.json").read_text())!=x:
            raise ValueError(f"approved case metadata mismatch: {i}")
        out.append(compact(PREVIEW/"cases",x,[0]))
    return out


def new_cases(root, cache, count, n_frames):
    # Reuse the exact site pool and overlay whose bytes passed the v1.1 freeze.
    from atomwise_ris.bench import users
    from atomwise_ris.bench.ground_fill import GroundLayer
    from atomwise_ris.bench.users_geometry import load_development_maps
    from generate_bench_users_preview import natural, osm_point_hotspots, write_case

    pool=json.loads((PREVIEW/"site_pool.json").read_text())["maps"]
    all_cases=[]
    for im,name in enumerate(DEVELOPMENT):
        city=load_development_maps(cache_root=cache,area_ids=(name,),
                  manifest_path=PREVIEW/"recovered-source-manifest.json",build_sites=False)[name]
        city.ground_fill=GroundLayer.read(PREVIEW/"ground_fill"/f"{name}-footprint-ground.npz",
                                         PREVIEW/"ground_fill"/f"{name}-owners.json")
        city.sites=pool[name]
        generator=users.Generator(city,name)
        osm_point_hotspots(generator,cache)
        sites=generator.sites[:]
        for i in range(count):
            rng=np.random.default_rng(np.random.SeedSequence([users.SEED, 20261109, im, i]))
            t=users.times(rng)
            counts=Counter()
            generator.sites=[sites[i % len(sites)]]
            site=generator.choose_site(rng,counts)
            c=natural(generator,t,rng,counts)
            case_id=f"{name}-v11-{i:04d}"
            case=write_case(root,generator,site,c,t,case_id,"natural",None,counts,rng,{})
            frames=np.unique(np.rint(np.linspace(0,len(t)-1,n_frames)).astype(int)).tolist()
            all_cases.append(compact(root/"cases",case,frames))
    return all_cases


def challenge_on_map(g, category, t, rng, counts):
    from generate_bench_users_preview import natural
    if category.startswith("shadow"):
        res = g.shadow(t, rng, counts, slope_required=category.endswith("sloped"))
        if res is not None:
            site, c, change = res
            return site, c, {
                "shadow_user_id": 0,
                "first_shadow_transition_frame": change,
                "leader_ground_relief_m": float(np.ptp(c.ground[:, 0])),
            }
    elif category == "all_no_los":
        res = g.no_los_group(t, rng, counts)
        if res is not None:
            site, c = res
            return site, c, {}
    elif category == "dense_crowd":
        for _ in range(250):
            c = natural(g, t, rng, counts)
            if c.size >= 60:
                return g.choose_site(rng, counts), c, {}
            counts["challenge_density_below_60"] += 1
    elif category == "elevated_crowd":
        c = g.elevated(t, rng, counts)
        if c is not None:
            return g.choose_site(rng, counts), c, {}
    counts["challenge_preferred_map_unavailable"] += 1
    return None


def choose_challenge(generator, i, t, rng, counts):
    requested = CHALLENGE_ORDER[i % len(CHALLENGE_ORDER)]
    start_idx = i % len(CHALLENGE_ORDER)
    for cat_idx in range(len(CHALLENGE_ORDER)):
        category = CHALLENGE_ORDER[(start_idx + cat_idx) % len(CHALLENGE_ORDER)]
        for _ in range(20):
            res = challenge_on_map(generator, category, t, rng, counts)
            if res is not None:
                site, c, extra = res
                return site, c, extra, category, requested
    raise RuntimeError(f"All 5 challenge categories exhausted for trajectory {i} on map {getattr(generator, 'map_id', None)}")


def challenge_cases(root, cache, count, n_frames):
    from atomwise_ris.bench import users
    from atomwise_ris.bench.ground_fill import GroundLayer
    from atomwise_ris.bench.users_geometry import load_development_maps
    from generate_bench_users_preview import osm_point_hotspots, write_case

    pool = json.loads((PREVIEW / "site_pool.json").read_text())["maps"]
    all_cases = []
    for im, name in enumerate(DEVELOPMENT):
        city = load_development_maps(cache_root=cache, area_ids=(name,),
                  manifest_path=PREVIEW / "recovered-source-manifest.json", build_sites=False)[name]
        city.ground_fill = GroundLayer.read(PREVIEW / "ground_fill" / f"{name}-footprint-ground.npz",
                                           PREVIEW / "ground_fill" / f"{name}-owners.json")
        city.sites = pool[name]
        generator = users.Generator(city, name)
        osm_point_hotspots(generator, cache)

        for i in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([users.SEED, 202611092, im, i]))
            t = users.times(rng)
            counts = Counter()
            site, c, extra, scenario, requested = choose_challenge(generator, i, t, rng, counts)
            case_id = f"{name}-v11c-{i:04d}"
            case = write_case(root, generator, site, c, t, case_id, "challenge", scenario, counts, rng, extra)
            frames = np.unique(np.rint(np.linspace(0, len(t) - 1, n_frames)).astype(int)).tolist()
            compacted = {
                **compact(root / "cases", case, frames),
                "kind": "challenge",
                "scenario": scenario,
                "requested_scenario": requested,
            }
            if "shadow_user_id" in extra:
                compacted["shadow_user_id"] = extra["shadow_user_id"]
            if "first_shadow_transition_frame" in extra:
                compacted["first_shadow_transition_frame"] = extra["first_shadow_transition_frame"]
            all_cases.append(compacted)
    return all_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--generate-count", type=int, default=0)
    ap.add_argument("--frame-count", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=ROOT / ".cache/bench-phase0b-recovered-20261108")
    ap.add_argument("--lead-approved-size", action="store_true")
    ap.add_argument("--pilot2", action="store_true", help="select exactly the three approved pilot frame-0 cases")
    ap.add_argument("--challenge", action="store_true", help="generate the separate challenge subset")
    a = ap.parse_args()
    freeze = verify_freeze()
    if a.challenge:
        if not a.lead_approved_size:
            raise ValueError("--challenge requires --lead-approved-size")
        if not 1 <= a.generate_count <= 23:
            raise ValueError("--generate-count must be between 1 and 23 for challenge mode")
        if not 1 <= a.frame_count <= 16:
            raise ValueError("frame_count outside registered plan")
        if a.pilot2:
            raise ValueError("challenge mode cannot be combined with pilot2")
    else:
        if not 0 <= a.generate_count <= 120 or not 1 <= a.frame_count <= 16:
            raise ValueError("size outside the registered plan")
        if a.generate_count > 2 and not a.lead_approved_size:
            raise ValueError("do not generate the full size before the lead accepts the pilot")
        if a.pilot2 and a.generate_count:
            raise ValueError("pilot2 reuses approved placements; it cannot generate fresh cases")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.challenge:
        cases = challenge_cases(a.output.parent, a.cache, a.generate_count, a.frame_count)
        manifest = {
            "scope": "development",
            "subset": "challenge",
            "is_pilot": False,
            "freeze_sha256": freeze,
            "seed_word": 202611092,
            "challenge_order": list(CHALLENGE_ORDER),
            "cases": cases,
        }
        with a.output.open("x") as f:
            json.dump(manifest, f, allow_nan=False, indent=2)
            f.write("\n")
    else:
        cases = (new_cases(a.output.parent, a.cache, a.generate_count, a.frame_count)
                 if a.generate_count else approved_pilot(PILOT2 if a.pilot2 else PILOT))
        with a.output.open("x") as f:
            json.dump({"scope": "development", "is_pilot": not a.generate_count,
                       "freeze_sha256": freeze, "cases": cases}, f, allow_nan=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
