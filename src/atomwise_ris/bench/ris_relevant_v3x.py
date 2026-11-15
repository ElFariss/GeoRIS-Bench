"""RIS-relevant v3 cross-city extension (docs/bench-ris-relevant-v3-crosscity-20261114.md).

The v3 rules are unchanged (``ris_relevant_v3``: union of 8 site draws, r = 40 m, front half-space 80 deg).
New here: a per-city seed word, ids ``<city>-rx3-NNNN``, 48 trajectories, and the CHECK (0-11) / TEST (12-47)
split. ``CITY`` must be set (``configure(city)``) before a generator is built; one city per process.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import ris_relevant as v1
from . import ris_relevant_v3 as v3
from . import users as u
from .ris_relevant import (CROWD_ATTEMPTS_PER_SITE, MIN_ELIGIBLE, N_FRAMES, RULE_R1_M, RULE_R2_M,  # noqa: F401
                           ZONE_ANCHOR_SHARE, eligible_users, schedule_ris_relevant, selected_frames)
from .ris_relevant_v3 import MAX_NORMAL_ANGLE_DEG, MIN_SITES, SITE_DRAWS, front_ok, panel_normal  # noqa: F401

SCOPE_NAME = "ris_relevant_v3x"
CITIES = {"munich": (0, 2026111401), "maastricht": (2, 2026111402), "utrecht": (4, 2026111403), "berlin": (3, 2026111404)}
HELDOUT = ("utrecht", "berlin")
PRIOR_SEED_WORDS = frozenset(v3.PRIOR_SEED_WORDS | {v3.SEED_WORD})
CASE_PREFIX = "rx3"
CASE_COUNT = 48
MIN_GENERATED = 46
SPLIT = {"check": (0, 11), "test": (12, 47)}
RADIUS_LADDER_M = (40.,)
RADIUS_M = 40.
DECLARATION = "docs/bench-ris-relevant-v3-crosscity-20261114.md"
DECLARATION_PRERUN_SHA256 = "5c9fde10560540b5a9cb1ecfa304e2733215ce8df5718aac22a063bbdae33266"
assert not (set(w for _, w in CITIES.values()) & PRIOR_SEED_WORDS) and len({w for _, w in CITIES.values()}) == 4

CITY = None
SEED_WORD = None
IM = None


def configure(city):
    global CITY, SEED_WORD, IM
    if city not in CITIES:
        raise ValueError(f"not a declared cross-city map: {city}")
    CITY, (IM, SEED_WORD) = city, CITIES[city]
    if city in HELDOUT and city not in u.DEVELOPMENT:
        # Declared, user-authorised one-shot admission of the held-out map, in this process only.
        u.DEVELOPMENT = tuple(u.DEVELOPMENT) + (city,)
    return city


def site_rng(im, k=0):
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, 0, im, k]))


def trajectory_rng(im, i):
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, 1, im, i]))


def case_id(map_id, i):
    return f"{map_id}-{CASE_PREFIX}-{i:04d}"


def split_of(index):
    for name, (lo, hi) in SPLIT.items():
        if lo <= index <= hi:
            return name
    raise ValueError(f"trajectory index outside the declared cross-city scope: {index}")


class RISRelevantGeneratorV3X(v3.RISRelevantGeneratorV3):
    def __init__(self, city, map_id, site_rng=None, distance_m=RULE_R1_M, radius_m=None):
        if map_id != CITY:
            raise ValueError("configure(city) first; one city per process")
        super().__init__(city, map_id, site_rng=site_rng, distance_m=distance_m, radius_m=RADIUS_M)

    def plan_sites(self, rng):
        """v3 Amendment A1 (union of SITE_DRAWS v1 draws), with the city's seed word."""
        kept, zones, used, ris_xy, stats = [], {}, Counter(), [], {"draws": []}
        for k in range(SITE_DRAWS):
            self.site_zone = {}
            sites_k, st = v1.RISRelevantGenerator.plan_sites(self, site_rng(IM, k))
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
        sites, zones = [], {}
        for s in self.sites:
            sid = s["site_id"].replace(f"-{v3.CASE_PREFIX}-site-", f"-{CASE_PREFIX}-site-")
            zones[sid] = self.site_zone[s["site_id"]]
            sites.append({**s, "site_id": sid})
        self.sites, self.site_zone = sites, zones


RISRelevantGenerator = RISRelevantGeneratorV3X
quick_eligibility = v3.quick_eligibility
