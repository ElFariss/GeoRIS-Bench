"""Registration of ``ris_relevant_v2_pilot`` next to (not inside) PRODUCTION_SCOPES and the v1 registry.

``registered_manifest`` admits the registered v2 manifest and otherwise delegates to the v1 function
(which delegates to ``production_pilot``). ``install`` is process-local, as in v1.
Declaration: docs/bench-ris-relevant-scope-v2-20261113.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import ris_relevant_scope as v1scope
from .production_freeze import sha256
from .production_run import PRODUCTION_SCOPES
from .ris_relevant_v2 import (CASE_COUNT, DECLARATION, DECLARATION_PRERUN_SHA256, N_FRAMES, SCOPE_NAME,  # noqa: F401
                              SEED_WORD, SPLIT, case_id, split_of)

RIS_RELEVANT_V2_SCOPES = (SCOPE_NAME,)
RIS_RELEVANT_V2_HASH = "951c42c3ed8fb8acb82a1ff1c953d210b4fc4bd5116043b4d4bf050d0206715b"
RIS_RELEVANT_V2_FAILED_INDICES = ()
RIS_RELEVANT_V2_RADIUS_M = 40.0
MAP = "rotterdam"


def expected_cases(failed=RIS_RELEVANT_V2_FAILED_INDICES, map_id=MAP):
    return {case_id(map_id, i): map_id for i in range(CASE_COUNT) if i not in set(failed)}


def pipeline_scopes():
    return tuple(PRODUCTION_SCOPES) + v1scope.RIS_RELEVANT_SCOPES + RIS_RELEVANT_V2_SCOPES


def validate_manifest(manifest, freeze_sha256, expected, radius):
    cases = manifest.get("cases", [])
    if (manifest.get("scope") != "development" or manifest.get("subset") != SCOPE_NAME
            or manifest.get("is_pilot") is not False or manifest.get("freeze_sha256") != freeze_sha256
            or manifest.get("seed_word") != SEED_WORD or manifest.get("radius_m") != radius
            or len(cases) != len(expected) or {c["id"]: c["map_id"] for c in cases} != expected):
        raise ValueError("invalid or unapproved ris_relevant_v2_pilot manifest identity")
    for c in cases:
        idx = int(c["id"].rsplit("-", 1)[1])
        if (len(c.get("frames", [])) != N_FRAMES or c.get("kind") != SCOPE_NAME
                or c.get("split") != split_of(idx) or c.get("trajectory_index") != idx):
            raise ValueError(f"{c['id']}: frames/kind/split/index mismatch")


def registered_manifest(path: Path, freeze_sha256: str, *, _hash=None, _failed=None, _radius=None):
    reg = RIS_RELEVANT_V2_HASH if _hash is None else _hash
    failed = RIS_RELEVANT_V2_FAILED_INDICES if _failed is None else _failed
    radius = RIS_RELEVANT_V2_RADIUS_M if _radius is None else _radius
    digest = sha256(path)
    if reg is not None and digest == reg:
        manifest = json.loads(Path(path).read_text())
        validate_manifest(manifest, freeze_sha256, expected_cases(failed), radius)
        return manifest, SCOPE_NAME, digest
    return v1scope.registered_manifest(path, freeze_sha256)


def install(module) -> None:
    if hasattr(module, "registered_manifest"):
        module.registered_manifest = registered_manifest
    if hasattr(module, "PRODUCTION_SCOPES"):
        module.PRODUCTION_SCOPES = pipeline_scopes()


def split_document(cases):
    out = {"role": SCOPE_NAME, "method": "trajectory index split: " +
           ", ".join(f"{k} {lo:04d}-{hi:04d}" for k, (lo, hi) in SPLIT.items()),
           "sealed": ["test"], "declaration": DECLARATION}
    for name in SPLIT:
        out[name] = sorted(c["id"] for c in cases if c["split"] == name)
    return out
