#!/usr/bin/env python3
"""Geometry-only generation of the RIS-relevant pilot (docs/bench-ris-relevant-scope-v1-20261112.md).

Wraps the frozen v1.1 generator: same map inputs, ground layer, hotspots, crowd model and
compaction; new site range, RIS-zone anchoring and served-user rule from
``atomwise_ris.bench.ris_relevant``. No tracing, no rates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from atomwise_ris.bench.production_freeze import ROOT, verify_freeze
from atomwise_ris.bench import ris_relevant as rr

PREVIEW = ROOT / "outputs/previews/bench-users-v1.1"


def write_case_rr(out, g, site, c, t, cid, served, counts, extra):
    """Mirror of the frozen preview ``write_case`` with an externally chosen served set."""
    from atomwise_ris.bench import users as u
    from generate_bench_users_preview import dump, table
    p = out / "cases" / cid
    p.mkdir(parents=True, exist_ok=True)
    f = len(t)
    labels = u.user_labels(g, site, c, t)
    if not np.isfinite(c.xyz).all() or not np.isfinite(c.ground).all():
        raise ValueError("invalid terrain or coordinates in case")
    ris_frames = labels["ris_los"].reshape(f, c.size)
    reach = u.ris_reachable(ris_frames)
    labels["served"] = np.tile(served, f)
    result = {"id": cid, "map_id": g.map_id, "kind": rr.SCOPE_NAME, "scenario": None,
              "duration_s": float(t[-1]), "dt_s": u.DT, "frame_count": f, "crowd_size": c.size,
              "group_sizes": np.unique(c.group, return_counts=True)[1].tolist(),
              "served_count": int(served.sum()), "served_user_ids": np.flatnonzero(served).tolist(),
              "movement": bool(np.any(c.speed > .015)),
              "users_moving": int(np.any(c.speed > .015, axis=0).sum()),
              "height_class_counts": dict(Counter(c.height.tolist())),
              "initial_link_counts": dict(Counter(labels["bs_link"][:c.size].tolist())),
              "all_frame_link_counts": dict(Counter(labels["bs_link"].tolist())),
              "labels_rows": len(labels["frame"]), "rejections": dict(counts),
              "distance_bs_ris_m": float(np.linalg.norm(site["bs_xyz"] - site["ris_xyz"])),
              "walk_slope_cap_deg": u.WALK_SLOPE_CAP_DEG, "site_distance_range_m": list(g.site_distance_m),
              "ris_los_users_first_frame": int(ris_frames[0].sum()),
              "ris_reachable_users": int(reach.sum()), "served_ris_los_first_frame": int(ris_frames[0][served].sum()),
              "served_ris_reachable": int(reach[served].sum()),
              "max_ground_relief_m": float(np.ptp(c.ground, axis=0).max()),
              "max_speed_mps": float(c.speed.max()), "protocol_seed": u.SEED, "seed_word": rr.SEED_WORD,
              **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in site.items()}, **extra}
    table(p / "frames.parquet", labels)
    result["frames_sha256"] = hashlib.sha256((p / "frames.parquet").read_bytes()).hexdigest()
    dump(p / "case.json", result)
    return result, labels


def load_generator(cache, map_id, distance):
    from atomwise_ris.bench.ground_fill import GroundLayer
    from atomwise_ris.bench.users_geometry import load_development_maps
    from atomwise_ris.bench import users as u
    from generate_bench_users_preview import osm_point_hotspots
    im = u.DEVELOPMENT.index(map_id)
    city = load_development_maps(cache_root=cache, area_ids=(map_id,),
                                 manifest_path=PREVIEW / "recovered-source-manifest.json", build_sites=False)[map_id]
    city.ground_fill = GroundLayer.read(PREVIEW / "ground_fill" / f"{map_id}-footprint-ground.npz",
                                        PREVIEW / "ground_fill" / f"{map_id}-owners.json")
    g = rr.RISRelevantGenerator(city, map_id, site_rng=rr.site_rng(im), distance_m=distance)
    osm_point_hotspots(g, cache)
    return g, im


def generate(out, cache, map_id, count):
    from atomwise_ris.bench import users as u
    from build_bench_production_cases import compact
    from generate_bench_users_preview import natural
    t0 = time.time()
    rule = "R1"
    try:
        g, im = load_generator(cache, map_id, rr.RULE_R1_M)
    except ValueError as e:                       # users.Generator raises when no site exists
        if "no valid BS/RIS sites" not in str(e):
            raise
        g = None
    if g is None or len(g.sites) < rr.MIN_SITES:
        rule = "R2"
        g, im = load_generator(cache, map_id, rr.RULE_R2_M)
        if len(g.sites) < rr.MIN_SITES:
            raise RuntimeError(f"STOP: fewer than {rr.MIN_SITES} sites under R2 ({len(g.sites)})")
    sites = g.sites[:]
    report = {"map_id": map_id, "map_index": im, "site_rule": rule, "distance_range_m": list(g.site_distance_m),
              "site_stats": g.site_stats, "sites": [{k: s[k] for k in ("site_id", "bs_support", "bs_ris_base_distance_m",
                                                                     "bs_los_share", "ris_los_share", "ris_zone_probe_count")}
                                                    for s in sites],
              "trajectories": {}, "failed_indices": []}
    print("SITES", rule, len(sites), json.dumps(g.site_stats), flush=True)
    cases = []
    for i in range(count):
        rng = rr.trajectory_rng(im, i)
        t = u.times(rng)
        frames = rr.selected_frames(len(t))
        counts = Counter()
        cid = rr.case_id(map_id, i)
        done = None
        for k in range(len(sites)):
            g.sites = [sites[(i + k) % len(sites)]]
            try:
                site = g.choose_site(rng, counts)
            except RuntimeError:
                counts["rr_site_draw_exhausted"] += 1
                continue
            for attempt in range(rr.CROWD_ATTEMPTS_PER_SITE):
                c = natural(g, t, rng, counts)
                elig = rr.quick_eligibility(g, site, c, frames)
                served = rr.schedule_ris_relevant(elig, rng, counts)
                if served is None:
                    continue
                done = (site, c, served, k, attempt, int(elig.sum()))
                break
            if done:
                break
        g.sites = sites
        if done is None:
            report["failed_indices"].append(i)
            report["trajectories"][cid] = {"status": "failed", "counts": dict(counts)}
            print("FAILED", cid, dict(counts), flush=True)
            continue
        site, c, served, k, attempt, n_elig = done
        extra = {"site_substitution_steps": k, "crowd_attempt": attempt, "eligible_users": n_elig,
                 "trajectory_index": i, "split": rr.split_of(i), "site_rule": rule}
        case, labels = write_case_rr(out, g, site, c, t, cid, served, counts, extra)
        # The stored labels must reproduce the eligibility used for scheduling.
        f, n = len(t), c.size
        bs = np.asarray(labels["bs_link"]).reshape(f, n)[frames]
        rl = np.asarray(labels["ris_los"]).reshape(f, n)[frames]
        if not rr.eligible_users(bs, rl)[served].all():
            raise RuntimeError(f"{cid}: stored labels disagree with the scheduling eligibility")
        comp = {**compact(out / "cases", case, frames.tolist()), "kind": rr.SCOPE_NAME,
                "split": rr.split_of(i), "trajectory_index": i, "site_rule": rule,
                "bs_ris_distance_m": case["bs_ris_distance_m"]}
        cases.append(comp)
        report["trajectories"][cid] = {"status": "ok", "site_id": site["site_id"], "substitution_steps": k,
                                       "crowd_attempt": attempt, "eligible_users": n_elig,
                                       "served": int(served.sum()), "crowd": int(c.size),
                                       "bs_ris_distance_m": case["bs_ris_distance_m"], "counts": dict(counts)}
        print("CASE", cid, site["site_id"], "k", k, "attempt", attempt, "elig", n_elig, "served", int(served.sum()),
              f"d1={case['bs_ris_distance_m']:.1f}", f"t={time.time()-t0:.0f}s", flush=True)
    report["elapsed_s"] = time.time() - t0
    return cases, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="cases.json path (refuses overwrite)")
    ap.add_argument("--map", default="rotterdam", choices=["rotterdam"])
    ap.add_argument("--count", type=int, default=rr.CASE_COUNT)
    ap.add_argument("--cache", type=Path, default=ROOT.parent / ".cache/bench-phase0b-recovered-20261108")
    a = ap.parse_args()
    if a.count != rr.CASE_COUNT:
        raise ValueError(f"the declaration fixes {rr.CASE_COUNT} trajectories")
    freeze = verify_freeze()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cases, report = generate(a.output.parent, a.cache, a.map, a.count)
    manifest = {"scope": "development", "subset": rr.SCOPE_NAME, "is_pilot": False, "freeze_sha256": freeze,
                "seed_word": rr.SEED_WORD, "declaration": "docs/bench-ris-relevant-scope-v1-20261112.md",
                "declaration_prerun_sha256": "ffb56cfd4ea036b1c6f1ae1bdedc3b2085e288ac9180385cc9af156a0eb381b4",
                "site_rule": report["site_rule"], "split": {k: list(v) for k, v in rr.SPLIT.items()},
                "failed_indices": report["failed_indices"], "cases": cases}
    with a.output.open("x") as f:
        json.dump(manifest, f, allow_nan=False, indent=2)
        f.write("\n")
    (a.output.parent / "generation_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("MANIFEST", a.output, len(cases), "failed", report["failed_indices"], flush=True)


if __name__ == "__main__":
    main()
