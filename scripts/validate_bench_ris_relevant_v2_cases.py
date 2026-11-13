#!/usr/bin/env python3
"""Geometry gate G1-G4 of docs/bench-ris-relevant-scope-v2-20261113.md (before tracing); v1 checks + radius."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from atomwise_ris.bench.production_freeze import ROOT, sha256, verify_freeze
from atomwise_ris.bench import ris_relevant_v2 as rr
from atomwise_ris.bench import users as u

PREVIEW = ROOT / "outputs/previews/bench-users-v1.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=ROOT.parent / ".cache/bench-phase0b-recovered-20261108")
    a = ap.parse_args()
    freeze = verify_freeze()
    m = json.loads(a.manifest.read_text())
    cases = m["cases"]
    root = a.manifest.parent / "cases"
    from atomwise_ris.bench.users_geometry import load_development_maps
    city = load_development_maps(cache_root=a.cache, area_ids=("rotterdam",),
                                 manifest_path=PREVIEW / "recovered-source-manifest.json", build_sites=False)["rotterdam"]
    lo, hi = (rr.RULE_R1_M if m["site_rule"] == "R1" else rr.RULE_R2_M)
    radius = float(m["radius_m"])
    if radius not in rr.RADIUS_LADDER_M:
        raise ValueError("radius not on the declared ladder")
    bounds = np.asarray(city.bounds, dtype=float)
    entries = recheck_agree = 0
    label_ok = 0
    g3_bad, g4_bad, sites = [], [], Counter()
    classes, d1, d2 = Counter(), [], []
    for c in cases:
        cj = json.loads((root / c["id"] / "case.json").read_text())
        pf = root / c["id"] / "frames.parquet"
        h = hashlib.sha256(pf.read_bytes()).hexdigest()
        if h != cj["frames_sha256"] or h != c["frame_source_sha256"]:
            raise ValueError(f"{c['id']}: frames.parquet hash mismatch")
        rows = pq.read_table(pf, columns=["frame", "user_id", "bs_link", "ris_los", "served", "ground_z_m", "z"]).to_pylist()
        keyed = {(r["frame"], r["user_id"]): r for r in rows}
        bs = np.asarray(c["bs_xyz"]); ris = np.asarray(c["ris_xyz"])
        dist = float(np.linalg.norm(bs - ris))
        vis_br = bool(np.ravel(city.los_batch(bs, ris[None])[0])[0])
        if not (lo <= dist <= hi) or not vis_br or not c["bs_ris_geometry_los"]:
            g3_bad.append(c["id"])
        sites[c["site_id"]] += 1
        d1.append(dist)
        if len(c["frames"]) != rr.N_FRAMES:
            g4_bad.append((c["id"], "frames"))
        for fr in c["frames"]:
            n = len(fr["users"])
            if not 2 <= n <= 4:
                g4_bad.append((c["id"], fr["frame"], "served_count", n))
            xyz = np.asarray([us["xyz"] for us in fr["users"]], dtype=float)
            if not np.isfinite(xyz).all() or not ((xyz[:, :2] >= bounds[:2]) & (xyz[:, :2] < bounds[2:])).all():
                g4_bad.append((c["id"], fr["frame"], "coords"))
            bvis = np.asarray(city.los_batch(bs, xyz)[0], dtype=bool)
            rvis = np.asarray(city.los_batch(ris, xyz)[0], dtype=bool)
            for j, us in enumerate(fr["users"]):
                r = keyed[(fr["frame"], us["user_id"])]
                entries += 1
                if not r["served"]:
                    g4_bad.append((c["id"], fr["frame"], "not_served_in_labels"))
                if not (np.isfinite(r["ground_z_m"]) and r["z"] > r["ground_z_m"]):
                    g4_bad.append((c["id"], fr["frame"], "ground"))
                label_ok += int(r["bs_link"] != u.LOS and bool(r["ris_los"]))
                dj = float(np.linalg.norm(xyz[j] - ris))
                label_ok -= int(dj > radius)
                recheck_agree += int((not bvis[j]) and rvis[j] and dj <= radius)
                classes[(r["bs_link"], bool(r["ris_los"]))] += 1
                d2.append(float(np.linalg.norm(xyz[j] - ris)))
    n_ok = len(cases)
    G = {
        "G1_generation": {"generated": n_ok, "declared": rr.CASE_COUNT, "failed_indices": m["failed_indices"],
                          "threshold": ">= 46", "pass": n_ok >= 46},
        "G2_served_rule": {"entries": entries, "label_share": label_ok / entries, "recheck_share": recheck_agree / entries,
                           "radius_m": radius, "threshold": "label+radius 100%, recheck >= 99%",
                           "pass": label_ok == entries and recheck_agree >= .99 * entries},
        "G3_site_rule": {"site_rule": m["site_rule"], "range_m": [lo, hi], "violations": g3_bad,
                         "distinct_sites": len(sites), "cases_per_site": dict(sites),
                         "threshold": "0 violations, >= 3 distinct sites", "pass": not g3_bad and len(sites) >= 3},
        "G4_users": {"violations": g4_bad[:50], "violation_count": len(g4_bad), "pass": not g4_bad},
    }
    split_counts = Counter(c["split"] for c in cases)
    out = {"status": "pass" if all(v["pass"] for v in G.values()) else "fail", "gates": G,
           "freeze_sha256": freeze, "manifest_sha256": sha256(a.manifest), "split_counts": dict(split_counts),
           "served_label_classes": {f"{k[0]}|ris_los={k[1]}": v for k, v in classes.items()},
           "d1_bs_ris_m_min_median_max": [min(d1), float(np.median(d1)), max(d1)],
           "d2_ris_user_m_q10_median_q90": np.percentile(d2, [10, 50, 90]).tolist(),
           "note": "G5 (mesh policy and endpoint preflight) is evaluated by the trace stage, fail-closed."}
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v["pass"] for k, v in G.items()}), out["status"])


if __name__ == "__main__":
    main()
