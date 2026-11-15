"""Registration of ``ris_relevant_v3x`` (cross-city) next to, not inside, PRODUCTION_SCOPES and the v1-v3 registries.

One manifest per city, registered by SHA-256 after its geometry gate. ``install(module)`` is process-local. For a
held-out city, ``admit_heldout(city, *modules)`` extends the module-level DEVELOPMENT_MAPS tuples in this process only
(declared, user-authorised one-shot test). Declaration: docs/bench-ris-relevant-v3-crosscity-20261114.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import production_run
from . import ris_relevant_scope as v1scope
from . import ris_relevant_scope_v2 as v2scope
from . import ris_relevant_scope_v3 as v3scope
from .production_freeze import sha256
from .production_run import PRODUCTION_SCOPES
from .ris_relevant_v3x import (CASE_COUNT, CITIES, DECLARATION, DECLARATION_PRERUN_SHA256, HELDOUT, N_FRAMES,  # noqa: F401
                               SCOPE_NAME, SPLIT, case_id, split_of)

RIS_RELEVANT_V3X_SCOPES = (SCOPE_NAME,)
from .production_freeze import ROOT
REGISTRY_FILE = ROOT / "configs/bench/ris-relevant-v3x-registry-20261114.json"
# manifest sha256 -> (city, failed_indices); one entry is added per city after its geometry gate (registry file).
REGISTERED = ({h: (v["city"], tuple(v.get("failed_indices", ()))) for h, v in json.loads(REGISTRY_FILE.read_text()).items()}
              if REGISTRY_FILE.is_file() else {})


def pipeline_scopes():
    return (tuple(PRODUCTION_SCOPES) + v1scope.RIS_RELEVANT_SCOPES + v2scope.RIS_RELEVANT_V2_SCOPES
            + v3scope.RIS_RELEVANT_V3_SCOPES + RIS_RELEVANT_V3X_SCOPES)


def expected_cases(city, failed=()):
    return {case_id(city, i): city for i in range(CASE_COUNT) if i not in set(failed)}


def validate_manifest(m, freeze_sha256, city, failed):
    cases = m.get("cases", [])
    if (m.get("scope") != "development" or m.get("subset") != SCOPE_NAME or m.get("is_pilot") is not False
            or m.get("freeze_sha256") != freeze_sha256 or m.get("seed_word") != CITIES[city][1]
            or m.get("radius_m") != 40.0 or m.get("city") != city
            or {c["id"]: c["map_id"] for c in cases} != expected_cases(city, failed)):
        raise ValueError(f"invalid or unapproved ris_relevant_v3x manifest identity for {city}")
    for c in cases:
        idx = int(c["id"].rsplit("-", 1)[1])
        if (len(c.get("frames", [])) != N_FRAMES or c.get("kind") != SCOPE_NAME
                or c.get("split") != split_of(idx) or c.get("trajectory_index") != idx):
            raise ValueError(f"{c['id']}: frames/kind/split/index mismatch")


def registered_manifest(path, freeze_sha256, *, _registered=None):
    reg = REGISTERED if _registered is None else _registered
    digest = sha256(path)
    if digest in reg:
        city, failed = reg[digest]
        m = json.loads(Path(path).read_text())
        validate_manifest(m, freeze_sha256, city, failed)
        return m, SCOPE_NAME, digest
    return v3scope.registered_manifest(path, freeze_sha256)


def admit_heldout(city, *modules):
    if city not in HELDOUT:
        return
    for mod in (production_run, *modules):
        if hasattr(mod, "DEVELOPMENT_MAPS") and city not in mod.DEVELOPMENT_MAPS:
            mod.DEVELOPMENT_MAPS = tuple(mod.DEVELOPMENT_MAPS) + (city,)


def install(module) -> None:
    if hasattr(module, "registered_manifest"):
        module.registered_manifest = registered_manifest
    if hasattr(module, "PRODUCTION_SCOPES"):
        module.PRODUCTION_SCOPES = pipeline_scopes()


def split_document(cases):
    out = {"role": SCOPE_NAME, "method": "trajectory index split: check 0000-0011 (data check only), test 0012-0047 (SEALED)",
           "sealed": ["test"], "declaration": DECLARATION}
    for name in SPLIT:
        out[name] = sorted(c["id"] for c in cases if c["split"] == name)
    return out
