#!/usr/bin/env python3
"""Audit hybrid gate evidence; optionally run a bounded direct-NEC microproof.

This command never writes to the historical `outputs/runs/hybrid_gates/` root.
The microproof is a 3x3 smoke experiment, not a full-size gate pass.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from atomwise_ris.hybrid.evidence import audit_legacy_gate_evidence, sha256_file


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="outputs/runs/hybrid_gates_audit_20261106",
        help="fresh output directory; existing files are never overwritten",
    )
    parser.add_argument(
        "--microproof", action="store_true",
        help="also run small 3x3/21-segment direct NEC P1/P2 convention probes",
    )
    args = parser.parse_args()

    if args.microproof:
        # Keep the optional small NEC probe on one CPU thread.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = audit_legacy_gate_evidence()
    audit.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "audit_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/atomwise_ris/hybrid/evidence.py"),
    })
    _write_new(out / "gate_evidence_audit.json", audit)

    if args.microproof:
        from atomwise_ris.hybrid.direct_validation import run_direct_nec_microproof

        micro = run_direct_nec_microproof()
        source_root = Path(__file__).resolve().parents[1]
        source_files = [
            Path(__file__).resolve(),
            source_root / "src/atomwise_ris/hybrid/direct_validation.py",
            source_root / "src/atomwise_ris/hybrid/embedded.py",
            source_root / "src/atomwise_ris/hybrid/operator.py",
            source_root / "src/atomwise_ris/hybrid/ray_coupling.py",
            source_root / "src/atomwise_ris/physics/nec.py",
        ]
        micro.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "python": platform.python_version(),
            "source_sha256": {str(p.relative_to(source_root)): sha256_file(p) for p in source_files},
        })
        _write_new(out / "direct_nec_microproof.json", micro)

    print(json.dumps({
        "all_passed": False,
        "optimizer_authorized": False,
        "audit_path": str(out / "gate_evidence_audit.json"),
        "microproof": str(out / "direct_nec_microproof.json") if args.microproof else None,
        "gate_statuses": {k: v["status"] for k, v in audit["gates"].items()},
    }, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
