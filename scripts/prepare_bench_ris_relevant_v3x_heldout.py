#!/usr/bin/env python3
"""Held-out map preparation for the cross-city test (Utrecht / Berlin): geometry export, street-terrain (slope)
gate and ground-fill layer, with the rules the development maps used (generate_bench_users_preview.validated_cities,
ground_fill.prepare_map with the frozen development fill decision). docs/bench-ris-relevant-v3-crosscity-20261114.md.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from atomwise_ris.bench.production_freeze import ROOT
from atomwise_ris.bench import heldout_geometry_v3x as hg
from atomwise_ris.bench import ris_relevant_v3x as rr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True, choices=sorted(hg.AREAS))
    a = ap.parse_args()
    rr.configure(a.city); hg.admit(a.city)
    from atomwise_ris.bench.users_geometry import load_development_maps
    from atomwise_ris.bench import ground_fill
    H = hg.H
    city = load_development_maps(cache_root=H / "cache", area_ids=(a.city,), manifest_path=H / "source-manifest.json",
                                 export_dir=H / "geometry", build_sites=False)[a.city]
    street = np.asarray(city.street_segments)
    z, sl, valid = city.sample_ground(street.reshape(-1, 2))
    gate_ok = bool(len(street)) and bool(np.asarray(valid).all()) and bool(np.isfinite(z).all())
    length = np.linalg.norm(street[:, 1] - street[:, 0], axis=1)
    grade = np.degrees(np.arctan2(abs(z[1::2] - z[::2]), length))
    frozen = json.loads((ROOT / "outputs/previews/bench-users-v1.1/ground_fill/audit.json").read_text())
    approved = frozen["interpolated_use_enabled"]
    osm = H / "cache" / f"{a.city}.osm"
    if hashlib.sha256(osm.read_bytes()).hexdigest() != city.source_hashes[osm.name]:
        raise ValueError("protective OSM geometry hash mismatch")
    layer, stats = ground_fill.prepare_map(city, osm, allow_fill=approved is not False)
    stats["artifact"] = layer.write(H / "ground_fill", a.city)
    rep = {"city": a.city, "slope_gate_pass": gate_ok, "valid_street_segments": int(len(street)),
           "valid_street_length_m": float(length.sum()), "lod2_buildings": len(city.buildings),
           "facade_candidates": int(len(city.facade_xyz)), "roof_candidates": int(len(city.roof_xyz)),
           "street_grade_deg_q50_q90_q95_max": np.percentile(grade, [50, 90, 95, 100]).tolist(),
           "street_dtm_slope_deg_q50_q90_q95_max": np.percentile(sl, [50, 90, 95, 100]).tolist(),
           "geometry_rejections": getattr(city, "geometry_rejections", {}), "checksummed_sources": city.source_hashes,
           "ground_fill_counts": stats.get("counts"), "development_fill_decision": approved,
           "street_hotspots": int(len(city.hotspots_xy))}
    (H / f"{a.city}-prep.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps({k: rep[k] for k in ("city", "slope_gate_pass", "valid_street_segments", "lod2_buildings",
                                          "facade_candidates", "roof_candidates", "ground_fill_counts")}, default=str))
    if not gate_ok:
        raise SystemExit("street-terrain gate failed")


if __name__ == "__main__":
    main()
