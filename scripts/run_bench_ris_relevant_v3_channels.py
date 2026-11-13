#!/usr/bin/env python3
"""Channel entry point for the ris_relevant_v3 scope (docs/bench-ris-relevant-scope-v3-20261113.md section 3).

Imports the unchanged scripts/build_bench_production_channels.py; in this process (and its forked unit children):
- extends the registry / scope tuple (ris_relevant_scope_v3.install);
- wraps frame_inputs: after the unchanged checks, every path row whose receiver is an external endpoint
  (BS port or user; link_rx >= n) is replaced by the rows of the shared direct trace of the same frame
  (<trace>/<map>/direct/<case>/frame-XXXX.npz, sha256 checked against its sidecar). Rows to the RIS elements
  are kept. Hence H_direct and the BS link classes are identical for 64/128/256.
Each node runs workers over the full manifest; a case is processed only where its trace record exists (node-local shards).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bench_production_channels as channels  # noqa: E402
from atomwise_ris.bench import ris_relevant_scope_v3  # noqa: E402
from atomwise_ris.bench.production_freeze import sha256  # noqa: E402

PER_PATH = ("link_tx", "link_rx", "path_index", "path_delay_s", "path_length_m", "path_length_is_exact", "path_jones",
            "path_theta_t", "path_phi_t", "path_theta_r", "path_phi_r", "path_type", "is_los", "has_reflection",
            "has_diffraction", "has_other_interaction", "interaction_enum_json", "interaction_enum_names_json")
_orig_frame_inputs = channels.frame_inputs


def merge_direct(t, n, direct):
    rows = len(t["link_rx"])
    for k in PER_PATH:
        if len(t[k]) != rows or len(direct[k]) != len(direct["link_rx"]):
            raise ValueError(f"per-path key {k} has an unexpected length")
    if not np.allclose(direct["tx_xyz"], t["tx_xyz"], rtol=0, atol=1e-9):
        raise ValueError("direct trace tx endpoints differ from the array trace")
    if not np.allclose(direct["rx_xyz"], t["rx_xyz"][n:], rtol=0, atol=1e-9):
        raise ValueError("direct trace rx endpoints differ from the array trace external receivers")
    keep = np.asarray(t["link_rx"]) < n
    out = dict(t)
    for k in PER_PATH:
        base = np.asarray(t[k])
        add = np.asarray(direct[k])
        if k == "link_rx":
            add = add + n
        if base.dtype.kind in "iub":
            add = add.astype(base.dtype)
        out[k] = np.concatenate([base[keep], add])
    return out


def frame_inputs_v3(p, case, frame, n, basis_centers):
    t, md, tx = _orig_frame_inputs(p, case, frame, n, basis_centers)
    p = Path(p)
    dp = Path(str(p).replace(f"/n{n}/", "/direct/"))
    side = json.loads(dp.with_suffix(".json").read_text())
    if side.get("sha256") != sha256(dp):
        raise ValueError(f"shared direct trace failed its sha256 check: {dp}")
    with np.load(dp, allow_pickle=False) as z:
        direct = {k: z[k] for k in z.files}
    return merge_direct(t, n, direct), md, tx


ris_relevant_scope_v3.install(channels)
channels.frame_inputs = frame_inputs_v3
if __name__ == "__main__":
    channels.main()
