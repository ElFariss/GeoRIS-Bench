"""RIS-relevant scope v2 (docs/bench-ris-relevant-scope-v2-20261113.md): v1 rules + RIS-user radius r.

Wraps ``ris_relevant`` (v1), which wraps the frozen v1.1 generator. Re-exports the v1 names; the
overrides are the scope identity, the seed word, the RIS zone (street nodes within r that are
RIS-LOS and BS-blocked) and the radius term in the eligibility.
"""
from __future__ import annotations

import numpy as np

from . import ris_relevant as v1
from . import users as u
from .ris_relevant import (CASE_COUNT, CROWD_ATTEMPTS_PER_SITE, MIN_ELIGIBLE, MIN_SITES, N_FRAMES,  # noqa: F401
                           RULE_R1_M, RULE_R2_M, SPLIT, ZONE_ANCHOR_SHARE, eligible_users,
                           schedule_ris_relevant, selected_frames, split_of)

SCOPE_NAME = "ris_relevant_v2_pilot"
SEED_WORD = 20261113
PRIOR_SEED_WORDS = frozenset(v1.PRIOR_SEED_WORDS | {v1.SEED_WORD})
SITE_STREAM, TRAJ_STREAM = 0, 1
CASE_PREFIX = "rr2"
RADIUS_LADDER_M = (40., 50., 60.)
RADIUS_M = None                       # set by the builder from the ladder
DECLARATION = "docs/bench-ris-relevant-scope-v2-20261113.md"
DECLARATION_PRERUN_SHA256 = "92e63a3770e5b9d2d713393cc45589c1a95cc7f3fe72fc482ec5c1ec3d76ec3f"

assert SEED_WORD not in PRIOR_SEED_WORDS


def site_rng(im: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, SITE_STREAM, im]))


def trajectory_rng(im: int, i: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([u.SEED, SEED_WORD, TRAJ_STREAM, im, i]))


def case_id(map_id: str, i: int) -> str:
    return f"{map_id}-{CASE_PREFIX}-{i:04d}"


class RISRelevantGeneratorV2(v1.RISRelevantGenerator):
    """v1 generator; RIS zone = street nodes within radius r, RIS-LOS and BS-blocked at 1.5 m AGL."""

    def __init__(self, city, map_id, site_rng=None, distance_m=RULE_R1_M, radius_m=None):
        r = RADIUS_M if radius_m is None else radius_m
        if r is None:
            raise ValueError("v2 needs a radius from the declared ladder")
        self.radius_m = float(r)
        super().__init__(city, map_id, site_rng=site_rng, distance_m=distance_m)
        self._near_zones()

    def _near_zones(self):
        zones, sites = {}, []
        xy = np.asarray(self.streets.xy, dtype=float)
        for s in self.sites:
            sid = s["site_id"].replace(f"-{v1.CASE_PREFIX}-site-", f"-{CASE_PREFIX}-site-")
            ris = np.asarray(s["ris_xyz"], dtype=float)
            bs = np.asarray(s["bs_xyz"], dtype=float)
            near = np.flatnonzero(np.linalg.norm(xy - ris[:2], axis=1) <= self.radius_m)
            zone = np.empty((0, 2))
            if len(near):
                z, _, ok = self.city.sample_ground(xy[near])
                ok = np.asarray(ok, dtype=bool)
                pts = np.c_[xy[near][ok], np.asarray(z)[ok] + u.PROBE_AGL_M]
                if len(pts):
                    rv = np.asarray(self.city.los_batch(ris, pts)[0], dtype=bool)
                    bv = np.asarray(self.city.los_batch(bs, pts)[0], dtype=bool)
                    keep = rv & ~bv & (np.linalg.norm(pts - ris, axis=1) <= self.radius_m)
                    zone = pts[keep, :2]
            zones[sid] = zone
            sites.append({**s, "site_id": sid, "ris_zone_probe_count": int(len(zone)),
                          "ris_zone_definition": f"street nodes within {self.radius_m:g} m, RIS-LOS and BS-blocked at 1.5 m AGL"})
        self.sites, self.site_zone = sites, zones


RISRelevantGenerator = RISRelevantGeneratorV2


def quick_eligibility(g, site, crowd, frames):
    """v1 eligibility (BS-blocked & RIS-LOS at every selected frame) AND 3-D RIS-user distance <= r."""
    base = v1.quick_eligibility(g, site, crowd, frames)
    d = np.linalg.norm(crowd.xyz[frames] - np.asarray(site["ris_xyz"], dtype=float)[None, None, :], axis=-1)
    return base & (d <= g.radius_m).all(axis=0)
