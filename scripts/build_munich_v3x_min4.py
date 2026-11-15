#!/usr/bin/env python3
"""Non-destructive geometry-only v3x builder for Munich with MIN_SITES=4 (matching Rotterdam v3 & Maastricht v3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bench_ris_relevant_v3x_cases as b  # noqa: E402
from atomwise_ris.bench import ris_relevant_v3x as rr  # noqa: E402

rr.MIN_SITES = 4

if __name__ == "__main__":
    b.main()
