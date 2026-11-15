#!/usr/bin/env python3
"""Write channels/<map>/DONE for ris_relevant_v3 after both nodes' units are present (unchanged production check)."""
import sys
from pathlib import Path
import os
import run_bench_ris_relevant_v3x_channels as w
from atomwise_ris.bench.production_freeze import verify_freeze
c = w.channels
m, scope, h = c.registered_manifest(Path(sys.argv[1]), verify_freeze())
ok = c.update_map_completion_status(os.environ["RR3X_CITY"], m["cases"], Path(sys.argv[2]), Path(sys.argv[3]), h)
print("scope", scope, "map DONE:", ok)
sys.exit(0 if ok else 1)
