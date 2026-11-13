"""Registration of the RIS-relevant pilot scope, next to (not inside) PRODUCTION_SCOPES.

The existing registry (``production_pilot.PILOTS``, ``production_run.PRODUCTION_SCOPES``) is not
edited. Pipeline entry points for this scope are the new wrapper scripts
``scripts/run_bench_ris_relevant_{trace,channels,package}.py``; they call ``install(module)`` on the
unchanged production module *in their own process only*, so the production scripts keep their
exact behaviour for every existing scope.

Declaration: docs/bench-ris-relevant-scope-v1-20261112.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import production_pilot
from .production_freeze import sha256
from .production_run import PRODUCTION_SCOPES
from .ris_relevant import CASE_COUNT, N_FRAMES, SCOPE_NAME, SEED_WORD, SPLIT, case_id, split_of

RIS_RELEVANT_SCOPES = (SCOPE_NAME,)
DECLARATION = "docs/bench-ris-relevant-scope-v1-20261112.md"
DECLARATION_PRERUN_SHA256 = "ffb56cfd4ea036b1c6f1ae1bdedc3b2085e288ac9180385cc9af156a0eb381b4"

# Filled in once, after geometry generation and the geometry gate, before tracing.
RIS_RELEVANT_PILOT_HASH = "82dcc824feb37071367e07d6388d6341b028dd69e2089dbb1bd25fab13c1e46b"
RIS_RELEVANT_FAILED_INDICES = ()
RIS_RELEVANT_MAP = "rotterdam"


def expected_cases(failed=RIS_RELEVANT_FAILED_INDICES, map_id=RIS_RELEVANT_MAP):
    return {case_id(map_id, i): map_id for i in range(CASE_COUNT) if i not in set(failed)}


def pipeline_scopes():
    """Existing production scopes followed by the RIS-relevant scope (existing tuple untouched)."""
    return tuple(PRODUCTION_SCOPES) + RIS_RELEVANT_SCOPES


def validate_ris_relevant_manifest(manifest: dict, freeze_sha256: str, expected: dict) -> None:
    cases = manifest.get("cases", [])
    if (manifest.get("scope") != "development"
            or manifest.get("subset") != SCOPE_NAME
            or manifest.get("is_pilot") is not False
            or manifest.get("freeze_sha256") != freeze_sha256
            or manifest.get("seed_word") != SEED_WORD
            or len(cases) != len(expected)
            or {c["id"]: c["map_id"] for c in cases} != expected):
        raise ValueError("invalid or unapproved ris_relevant_pilot manifest identity")
    if any(len(c.get("frames", [])) != N_FRAMES for c in cases):
        raise ValueError(f"ris_relevant_pilot requires {N_FRAMES} frames per case")
    for c in cases:
        idx = int(c["id"].rsplit("-", 1)[1])
        if c.get("kind") != SCOPE_NAME or c.get("split") != split_of(idx) or c.get("trajectory_index") != idx:
            raise ValueError(f"{c['id']}: kind/split/index mismatch")


def registered_manifest(path: Path, freeze_sha256: str, *, _hash=None, _failed=None):
    """Drop-in for production_pilot.registered_manifest that also admits the RIS-relevant pilot."""
    registered = RIS_RELEVANT_PILOT_HASH if _hash is None else _hash
    failed = RIS_RELEVANT_FAILED_INDICES if _failed is None else _failed
    digest = sha256(path)
    if registered is not None and digest == registered:
        manifest = json.loads(Path(path).read_text())
        validate_ris_relevant_manifest(manifest, freeze_sha256, expected_cases(failed))
        return manifest, SCOPE_NAME, digest
    return production_pilot.registered_manifest(path, freeze_sha256)


def install(module) -> None:
    """Point one already-imported pipeline module at the extended registry (process-local)."""
    if hasattr(module, "registered_manifest"):
        module.registered_manifest = registered_manifest
    if hasattr(module, "PRODUCTION_SCOPES"):
        module.PRODUCTION_SCOPES = pipeline_scopes()


def split_document(cases: list[dict]) -> dict:
    out = {"role": SCOPE_NAME, "method": "trajectory index split: " +
           ", ".join(f"{k} {lo:04d}-{hi:04d}" for k, (lo, hi) in SPLIT.items()),
           "sealed": ["test"], "declaration": DECLARATION}
    for name in SPLIT:
        out[name] = sorted(c["id"] for c in cases if c["split"] == name)
    return out
