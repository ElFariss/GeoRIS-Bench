#!/usr/bin/env python3
"""Channel entry point for ris_relevant_v3x (cross-city). Env: RR3X_CITY=<city>.

Reuses run_bench_ris_relevant_v3_channels unchanged (shared-direct merge) with the v3x registry, process-local.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomwise_ris.bench import ris_relevant_v3x, ris_relevant_scope_v3x as S  # noqa: E402

CITY = ris_relevant_v3x.configure(os.environ["RR3X_CITY"])
import run_bench_ris_relevant_v3_channels as W  # noqa: E402

S.install(W.channels)
S.admit_heldout(CITY, W.channels)
channels = W.channels

if __name__ == "__main__":
    W.channels.main()
