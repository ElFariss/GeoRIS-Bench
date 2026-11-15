#!/usr/bin/env python3
"""Production gate for the ris_relevant_v3 scope (NEW, 2026-11-13).

Runs the unchanged scripts/verify_bench_production_gate.py in production mode with the scope registry
extended in this process only (as scripts/run_bench_ris_relevant_package.py does for packaging).
The gate checks schema, finiteness, loader/rescore consistency (<= 1e-10), LOS geometry, mesh
preflight and the stop rule over every case; its output holds integrity results only, no rate
statistics, so no test-split metric is produced.
Usage: same arguments as verify_bench_production_gate.py --production.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_bench_production_gate as gate  # noqa: E402
from atomwise_ris.bench import production_run, ris_relevant_scope_v3x  # noqa: E402

import os  # noqa: E402
from atomwise_ris.bench import ris_relevant_v3x  # noqa: E402
_CITY = ris_relevant_v3x.configure(os.environ["RR3X_CITY"])
ris_relevant_scope_v3x.admit_heldout(_CITY, gate)
gate.registered_manifest = ris_relevant_scope_v3x.registered_manifest
production_run.PRODUCTION_SCOPES = ris_relevant_scope_v3x.pipeline_scopes()

if __name__ == "__main__":
    gate.main()
