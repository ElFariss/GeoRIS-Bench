#!/usr/bin/env python3
"""Trace entry point for the ris_relevant_pilot scope.

Imports the unchanged scripts/run_bench_production_trace.py, extends its manifest registry and scope
tuple in this process only (atomwise_ris.bench.ris_relevant_scope.install), then runs its main().
Same CLI as run_bench_production_trace.py (use --production).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench_production_trace as trace  # noqa: E402
from atomwise_ris.bench import ris_relevant_scope  # noqa: E402

ris_relevant_scope.install(trace)

if __name__ == "__main__":
    trace.main()
