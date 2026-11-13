"""RIS-relevant scope v1: a wrapper around the frozen v1.1 user generator.

Rules: docs/bench-ris-relevant-scope-v1-20261112.md (declared before generation).
Nothing in ``users.py`` is edited. ``RISRelevantGenerator`` subclasses ``users.Generator`` and
overrides only site planning (BS-RIS distance range), the post-draw site check, and group
anchoring (a declared 0.5 share of anchors in the RIS zone). ``eligible_users`` and
``schedule_ris_relevant`` replace the v1.1 served-slot rule (no negative-share slots).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import users as u

SCOPE_NAME = "ris_relevant_pilot"
SEED_WORD = 20261112                  # disjoint from {1,2,3,4,5,20261109,202611092}
PRIOR_SEED_WORDS = frozenset({1, 2, 3, 4, 5, 20261109, 202611092})
SITE_STREAM, TRAJ_STREAM = 0, 1
RULE_R1_M = (5., 20.)
RULE_R2_M = (5., 25.)                  # declared fallback, used only if R1 gives < MIN_SITES
MIN_SITES = 4
ZONE_ANCHOR_SHARE = .5
CROWD_ATTEMPTS_PER_SITE = 25
MIN_ELIGIBLE = 2
N_FRAMES = 16
CASE_COUNT = 48
SPLIT = {"train": (0, 33), "validation": (34, 40), "test": (41, 47)}   # inclusive; test is SEALED
CASE_PREFIX = "rr1"

assert SEED_WORD not in PRIOR_SEED_WORDS


def site_rng(im: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, SITE_STREAM, im]))


def trajectory_rng(im: int, i: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, TRAJ_STREAM, im, i]))


def split_of(index: int) -> str:
    for name, (lo, hi) in SPLIT.items():
        if lo <= index <= hi:
            return name
    raise ValueError(f"trajectory index outside the declared pilot: {index}")


def case_id(map_id: str, i: int) -> str:
    return f"{map_id}-{CASE_PREFIX}-{i:04d}"


def selected_frames(frame_count: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Production frame selection (same expression as build_bench_production_cases)."""
    return np.unique(np.rint(np.linspace(0, frame_count - 1, n_frames)).astype(int))


def eligible_users(bs_link_frames, ris_los_frames) -> np.ndarray:
    """Eligible iff BS-blocked (label != 'los') AND RIS-LOS at EVERY given frame. Shapes (frames, users)."""
    bs = np.asarray(bs_link_frames)
    ris = np.asarray(ris_los_frames, dtype=bool)
    if bs.shape != ris.shape or bs.ndim != 2:
        raise ValueError("bs_link and ris_los must both be (frames, users)")
    return ((bs != u.LOS) & ris).all(axis=0)


def schedule_ris_relevant(eligible, rng, counts=None) -> np.ndarray | None:
    """2-4 served users drawn only from eligible users; None when fewer than 2 are eligible.

    ``want`` is drawn exactly like v1.1 (rng.integers(2, 5)); there are no negative-share slots and
    no fallback pool.
    """
    eligible = np.asarray(eligible, dtype=bool)
    counts = Counter() if counts is None else counts
    pool = np.flatnonzero(eligible)
    if len(pool) < MIN_ELIGIBLE:
        counts["rr_crowd_too_few_eligible"] += 1
        return None
    want = int(rng.integers(2, 5))
    take = rng.choice(pool, min(want, len(pool)), replace=False)
    out = np.zeros(len(eligible), dtype=bool)
    out[take] = True
    return out


class RISRelevantGenerator(u.Generator):
    """v1.1 generator with the RIS-relevant site range and RIS-zone anchoring."""

    def __init__(self, city, map_id, site_rng=None, distance_m=RULE_R1_M):
        self.site_distance_m = tuple(float(x) for x in distance_m)
        self.site_zone = {}
        self.active_zone = None
        super().__init__(city, map_id, site_rng=site_rng)

    # ---- site planning: v1.1 procedure with self.site_distance_m -------------
    def plan_sites(self, rng):
        city = self.city
        lo, hi = self.site_distance_m
        stats = {"ris_candidates": 0, "bs_roof_candidates": 0, "bs_mast_candidates": 0,
                 "pairs_in_range": 0, "pairs_bs_ris_los": 0, "pairs_evaluated": 0,
                 "pairs_failing_coverage": 0, "pairs_eligible": 0, "distance_range_m": [lo, hi]}
        fac = np.asarray(city.facade_xyz, dtype=float).reshape(-1, 3)
        fac_ids = np.asarray(getattr(city, "facade_owner_ids", np.full(len(fac), None, dtype=object)), dtype=object)
        fac_g = np.empty(0)
        stats["ris_facade_points_total"] = len(fac)
        if len(fac):
            g, method = self.support_ground(fac_ids, fac, building_mounted=True)
            agl = fac[:, 2] - g
            stats["ris_facade_no_valid_ground_reference"] = int((~np.isfinite(g)).sum())
            keep = np.isfinite(g) & self.within(fac[:, :2]) & (agl >= u.RIS_AGL_M[0]) & (agl <= u.RIS_AGL_M[1])
            fac, fac_ids, fac_g = fac[keep], fac_ids[keep], g[keep]
            t = self._thin(fac, 10., rng)
            fac, fac_ids, fac_g = fac[t], fac_ids[t], fac_g[t]
        stats["ris_candidates"] = len(fac)
        roof = np.asarray(city.roof_xyz, dtype=float).reshape(-1, 3)
        roof_ids = np.asarray(getattr(city, "roof_owner_ids", np.full(len(roof), None, dtype=object)), dtype=object)
        roof_g = np.empty(0)
        if len(roof):
            g, _ = self.support_ground(roof_ids, roof, building_mounted=True)
            keep = np.isfinite(g) & self.within(roof[:, :2]) & ((roof[:, 2] - g) >= u.ROOF_BS_MIN_AGL_M)
            roof, roof_ids, roof_g = roof[keep], roof_ids[keep], g[keep]
            t = self._thin(roof, 15., rng)
            roof, roof_ids, roof_g = roof[t], roof_ids[t], roof_g[t]
        mast_xy = self.streets.xy[np.sort(rng.choice(len(self.streets.xy), min(80, len(self.streets.xy)), replace=False))]
        mg, _, mok = city.sample_ground(mast_xy)
        mok = np.asarray(mok, dtype=bool)
        mast = np.c_[mast_xy[mok], mg[mok] + u.MAST_BS_M]
        stats["bs_roof_candidates"], stats["bs_mast_candidates"] = len(roof), len(mast)
        anchors = ([(p + np.array([0., 0., u.ROOF_ANTENNA_MIN_M]), "rooftop", str(i), gz) for p, i, gz in zip(roof, roof_ids, roof_g)] +
                   [(p, "mast", None, p[2] - u.MAST_BS_M) for p in mast])
        if not len(fac) or not anchors:
            return [], stats
        pairs = []
        for a, (bs, _, _, _) in enumerate(anchors):
            d = np.linalg.norm(fac - bs, axis=1)
            near = np.flatnonzero((d >= lo) & (d <= hi))
            stats["pairs_in_range"] += len(near)
            if len(near):
                vis, _ = city.los_batch(bs, fac[near])
                pairs.extend((a, int(r)) for r in near[np.asarray(vis, dtype=bool)])
        stats["pairs_bs_ris_los"] = len(pairs)
        if not pairs:
            return [], stats
        chosen = np.sort(rng.choice(len(pairs), u.MAX_PAIR_EVAL, replace=False)) if len(pairs) > u.MAX_PAIR_EVAL else np.arange(len(pairs))
        pi = np.sort(rng.choice(len(self.streets.xy), min(u.MAX_PROBES, len(self.streets.xy)), replace=False))
        pxy = self.streets.xy[pi]
        pz, _, pok = city.sample_ground(pxy)
        pok = np.asarray(pok, dtype=bool)
        probes = np.c_[pxy[pok], pz[pok] + u.PROBE_AGL_M]
        bs_vis, ris_vis = {}, {}

        def evaluate(pair_indices):
            eligible = []
            for i in pair_indices:
                a, r = pairs[i]
                stats["pairs_evaluated"] += 1
                if a not in bs_vis:
                    bs_vis[a] = np.asarray(city.los_batch(anchors[a][0], probes)[0], dtype=bool)
                if r not in ris_vis:
                    ris_vis[r] = np.asarray(city.los_batch(fac[r], probes)[0], dtype=bool)
                b, ris = bs_vis[a], ris_vis[r]
                zone = ris & ~b
                shadow = int(zone.sum())
                if (b.mean() < u.BS_MIN_LOS_SHARE or ris.mean() < u.RIS_MIN_LOS_SHARE or
                        shadow < max(u.SHADOW_MIN_POINTS, u.SHADOW_MIN_SHARE * len(probes))):
                    stats["pairs_failing_coverage"] += 1
                    continue
                eligible.append((a, r, {"probe_count": len(probes), "bs_los_share": float(b.mean()),
                                        "ris_los_share": float(ris.mean()),
                                        "ris_covers_bs_shadow_share": float(shadow / len(probes)),
                                        "shadow_share_covered_by_ris": float(shadow / max(1, int((~b).sum())))},
                                 probes[zone, :2].copy()))
            stats["pairs_eligible"] += len(eligible)
            return eligible

        sites, used, ris_xy = [], Counter(), []

        def select(eligible, limit):
            for k in rng.permutation(len(eligible)):
                a, r, cov, zone_xy = eligible[int(k)]
                if used[a] >= u.MAX_SITES_PER_BS or any(np.linalg.norm(fac[r, :2] - q) < u.MIN_RIS_SEPARATION_M for q in ris_xy):
                    continue
                bs, support, building, bs_ground = anchors[a]
                used[a] += 1
                ris_xy.append(fac[r, :2].copy())
                site_refs = {}
                for which, mounted, ident, pos in (("bs", support == "rooftop", building, bs),
                                                   ("ris", True, str(fac_ids[r]), fac[r])):
                    site_refs.update(self._ground_record(which, mounted, ident, pos))
                sid = f"{self.map_id}-{CASE_PREFIX}-site-{len(sites):02d}"
                self.site_zone[sid] = zone_xy
                sites.append({"site_id": sid, "bs_xyz": bs.tolist(),
                              "ris_xyz": fac[r].tolist(), "bs_agl_m": float(bs[2] - bs_ground), "ris_agl_m": float(fac[r, 2] - fac_g[r]),
                              "bs_ground_z_m": float(bs_ground), "ris_ground_z_m": float(fac_g[r]),
                              **site_refs,
                              "bs_support": support, "bs_building_id": building, "ris_building_id": str(fac_ids[r]),
                              "ris_support": "facade", "bs_ris_los": True,
                              "bs_ris_base_distance_m": float(np.linalg.norm(bs - fac[r])),
                              "ris_zone_probe_count": int(len(zone_xy)), **cov})
                if len(sites) >= limit:
                    break

        select(evaluate(chosen), u.MAX_SITES)
        if getattr(city, "ground_fill", None) is not None and len(sites) < 6:
            unused = np.setdiff1d(np.arange(len(pairs)), chosen, assume_unique=True)
            stats["sites_before_repair"] = len(sites)
            if len(unused):
                extra = np.sort(rng.choice(unused, min(3000, len(unused)), replace=False))
                stats["repair_pairs_evaluated"] = len(extra)
                select(evaluate(extra), 6)
        stats["sites_selected"] = len(sites)
        return sites, stats

    # ---- v1.1 choose_site with the RIS-relevant range ----------------------
    def choose_site(self, rng, counts):
        lo, hi = self.site_distance_m
        for _ in range(25):
            site = self.sites[int(rng.integers(len(self.sites)))]
            bs = np.asarray(site["bs_xyz"], dtype=float).copy()
            ris = np.asarray(site["ris_xyz"], dtype=float).copy()
            offset = min(24., float(rng.lognormal(np.log(4), .75)))
            if site.get("bs_support") == "rooftop":
                offset = max(0., offset - u.ROOF_ANTENNA_MIN_M)
            bs[2] += offset
            distance = float(np.linalg.norm(bs - ris))
            if not lo <= distance <= hi:
                counts["site_distance_out_of_range_after_mast_draw"] += 1
                continue
            vis, _ = self.city.los_batch(bs, ris[None])
            if not bool(np.ravel(vis)[0]):
                counts["site_bs_ris_blocked_after_mast_draw"] += 1
                continue
            extra = {k: v for k, v in site.items() if k not in {"bs_xyz", "ris_xyz", "bs_agl_m", "ris_agl_m"}}
            bs_ground = float(site["bs_ground_z_m"])
            ris_ground = float(site["ris_ground_z_m"])
            self.active_zone = self.site_zone.get(site["site_id"])
            return {**extra, "site_id": str(site["site_id"]), "bs_xyz": bs, "ris_xyz": ris,
                    "bs_agl_m": float(bs[2] - bs_ground), "ris_agl_m": float(ris[2] - ris_ground),
                    "bs_ris_distance_m": distance}
        raise RuntimeError(f"{self.map_id}: no eligible BS/RIS after mast draws")

    # ---- anchoring: a declared share of anchors in the active RIS zone ------
    def anchor(self, rng, counts, isolated=False):
        zone = self.active_zone
        if zone is not None and len(zone) and rng.random() < ZONE_ANCHOR_SHARE:
            for _ in range(20):
                pt = zone[int(rng.integers(len(zone)))]
                d, i = self.streets.tree.query(pt + rng.normal(0, 7, 2))
                if d <= 25.:
                    counts["rr_zone_anchor"] += 1
                    return int(i)
                counts["rr_zone_anchor_snap_too_far"] += 1
            counts["rr_zone_anchor_fallback_v11"] += 1
        return super().anchor(rng, counts, isolated=isolated)


def quick_eligibility(g, site, crowd, frames):
    """Geometry-only eligibility at the selected frames (same LOS calls as users.Generator.link)."""
    xyz = crowd.xyz[frames]                      # (frames, users, 3)
    flat = xyz.reshape(-1, 3)
    bs_vis = np.asarray(g.city.los_batch(site["bs_xyz"], flat)[0], dtype=bool).reshape(len(frames), -1)
    ris_vis = np.asarray(g.city.los_batch(site["ris_xyz"], flat)[0], dtype=bool).reshape(len(frames), -1)
    return (~bs_vis & ris_vis).all(axis=0)
