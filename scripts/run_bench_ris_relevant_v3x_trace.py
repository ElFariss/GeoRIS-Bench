#!/usr/bin/env python3
"""Trace entry point for ris_relevant_v3x (cross-city). Env: RR3X_CITY=<city>, RR3_SHARD=k/K.

Reuses run_bench_ris_relevant_v3_trace unchanged (1e6 samples, cap 2e6, shared direct trace, shard filter) and, in
this process only, points the registry at ris_relevant_scope_v3x and admits a declared held-out city.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomwise_ris.bench import ris_relevant_v3x, ris_relevant_scope_v3x as S  # noqa: E402

CITY = ris_relevant_v3x.configure(os.environ["RR3X_CITY"])
if CITY in ris_relevant_v3x.HELDOUT:
    from atomwise_ris.bench import heldout_geometry_v3x  # noqa: E402
    heldout_geometry_v3x.admit(CITY)
import run_bench_ris_relevant_v3_trace as W  # noqa: E402

S.install(W.T)
S.admit_heldout(CITY, W.T)
if CITY in ris_relevant_v3x.HELDOUT:
    heldout_geometry_v3x.admit_trace(W.T, CITY)
_only = os.environ.get("RR3X_ONLY")          # smoke preflight: restrict to the listed trajectory indices
if _only:
    _idx = {int(x) for x in _only.split(",")}
    _orig_verify = W.T.verify_case_manifest

    def _verify_only(path, production=False, smoke_unregistered=False):
        obj, cases = _orig_verify(path, production=production, smoke_unregistered=smoke_unregistered)
        cases = [c for c in cases if int(c["id"].rsplit("-", 1)[1]) in _idx]
        print(f"RR3X_ONLY {sorted(_idx)}: {len(cases)} cases", flush=True)
        return obj, cases
    W.T.verify_case_manifest = _verify_only

if __name__ == "__main__":
    W.T.main()
