"""RIS-relevant scope v3 (docs/bench-ris-relevant-scope-v3-20261113.md): v2 rules + front-half-space rule.

Wraps ``ris_relevant_v2`` (which wraps v1 and the frozen v1.1 generator). New: seed word, 120 cases,
84/18/18 split, and the eligibility/zone condition angle(user - RIS, panel normal) <= 80 deg, where the
panel normal is the horizontal unit vector RIS->BS (as in compact()).
"""
from __future__ import annotations

import numpy as np

from . import ris_relevant as v1
from . import ris_relevant_v2 as v2
from . import users as u
from .ris_relevant import (CROWD_ATTEMPTS_PER_SITE, MIN_ELIGIBLE, MIN_SITES, N_FRAMES, RULE_R1_M, RULE_R2_M,  # noqa: F401
                           ZONE_ANCHOR_SHARE, eligible_users, schedule_ris_relevant, selected_frames)

SCOPE_NAME = "ris_relevant_v3"
SEED_WORD = 202611133
PRIOR_SEED_WORDS = frozenset(v2.PRIOR_SEED_WORDS | {v2.SEED_WORD})
SITE_STREAM, TRAJ_STREAM = 0, 1
CASE_PREFIX = "rr3"
CASE_COUNT = 120
MIN_GENERATED = 114
SPLIT = {"train": (0, 83), "validation": (84, 101), "test": (102, 119)}
RADIUS_LADDER_M = (40., 50., 60.)
RADIUS_M = None
MAX_NORMAL_ANGLE_DEG = 80.
MIN_SITES = 6                         # amendment A1
SITE_DRAWS = 8                        # amendment A1: union of 8 planning draws
DECLARATION = "docs/bench-ris-relevant-scope-v3-20261113.md"
DECLARATION_PRERUN_SHA256 = "e35e08bf8159f9e61377737faa50d69dfb37c53a6a6358cfa6fcc6b401b75948"
assert SEED_WORD not in PRIOR_SEED_WORDS


def site_rng(im, k=0):
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, SITE_STREAM, im, k]))


def trajectory_rng(im, i):
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, TRAJ_STREAM, im, i]))


def case_id(map_id, i):
    return f"{map_id}-{CASE_PREFIX}-{i:04d}"


def split_of(index):
    for name, (lo, hi) in SPLIT.items():
        if lo <= index <= hi:
            return name
    raise ValueError(f"trajectory index outside the declared v3 scope: {index}")


def panel_normal(bs_xyz, ris_xyz):
    n = np.asarray(bs_xyz, float) - np.asarray(ris_xyz, float)
    n[2] = 0.
    return n / np.linalg.norm(n)


def front_ok(points, bs_xyz, ris_xyz, max_deg=MAX_NORMAL_ANGLE_DEG):
    """angle(point - RIS, panel normal) <= max_deg, for points of shape (..., 3)."""
    v = np.asarray(points, float) - np.asarray(ris_xyz, float)
    c = (v @ panel_normal(bs_xyz, ris_xyz)) / np.linalg.norm(v, axis=-1)
    return c >= np.cos(np.radians(max_deg))


class RISRelevantGeneratorV3(v2.RISRelevantGeneratorV2):
    def __init__(self, city, map_id, site_rng=None, distance_m=RULE_R1_M, radius_m=None):
        r = RADIUS_M if radius_m is None else radius_m
        self._im = u.DEVELOPMENT.index(map_id)
        super().__init__(city, map_id, site_rng=site_rng, distance_m=distance_m, radius_m=r)

    def plan_sites(self, rng):
        """Amendment A1: union of SITE_DRAWS unchanged v1 planning draws, v1.1 diversity rules across draws.

        ``rng`` (the single-draw stream) is not used; draw k uses site_rng(im, k)."""
        from collections import Counter
        kept, zones, used, ris_xy, stats = [], {}, Counter(), [], {"draws": []}
        for k in range(SITE_DRAWS):
            self.site_zone = {}
            sites_k, st = v1.RISRelevantGenerator.plan_sites(self, site_rng(self._im, k))
            stats["draws"].append({"k": k, "sites": len(sites_k), "pairs_eligible": st.get("pairs_eligible")})
            for s in sites_k:
                ris = np.asarray(s["ris_xyz"], float)[:2]
                anchor = tuple(np.round(np.asarray(s["bs_xyz"], float), 3))
                if used[anchor] >= u.MAX_SITES_PER_BS or any(np.linalg.norm(ris - q) < u.MIN_RIS_SEPARATION_M for q in ris_xy):
                    continue
                used[anchor] += 1
                ris_xy.append(ris)
                sid = f"{self.map_id}-{v1.CASE_PREFIX}-site-{len(kept):02d}"
                zones[sid] = self.site_zone[s["site_id"]]
                kept.append({**s, "site_id": sid, "planning_draw": k})
        self.site_zone = zones
        stats["sites_selected"] = len(kept)
        return kept, stats

    def _near_zones(self):
        super()._near_zones()
        zones, sites = {}, []
        for s in self.sites:
            sid = s["site_id"].replace(f"-{v2.CASE_PREFIX}-site-", f"-{CASE_PREFIX}-site-")
            z = self.site_zone[s["site_id"]]
            if len(z):
                g, _, ok = self.city.sample_ground(z)
                pts = np.c_[z, np.asarray(g) + u.PROBE_AGL_M]
                z = z[np.asarray(ok, bool) & front_ok(pts, s["bs_xyz"], s["ris_xyz"])]
            zones[sid] = z
            sites.append({**s, "site_id": sid, "ris_zone_probe_count": int(len(z)),
                          "ris_zone_definition": s["ris_zone_definition"] + f", within {MAX_NORMAL_ANGLE_DEG:g} deg of the panel normal"})
        self.sites, self.site_zone = sites, zones


RISRelevantGenerator = RISRelevantGeneratorV3


def quick_eligibility(g, site, crowd, frames):
    base = v2.quick_eligibility(g, site, crowd, frames)
    return base & front_ok(crowd.xyz[frames], site["bs_xyz"], site["ris_xyz"]).all(axis=0)
