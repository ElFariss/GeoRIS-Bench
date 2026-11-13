#!/usr/bin/env python3
"""Channel entry point for the ris_relevant_pilot scope.

Imports the unchanged scripts/build_bench_production_channels.py, extends its manifest registry and
scope tuple in this process only, then runs its main(). Worker re-exec (os.execv on sys.argv)
re-enters this wrapper, so the extension survives restarts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bench_production_channels as channels  # noqa: E402
from atomwise_ris.bench import ris_relevant_scope  # noqa: E402

ris_relevant_scope.install(channels)

if __name__ == "__main__":
    channels.main()
