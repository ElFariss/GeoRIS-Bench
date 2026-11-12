"""Shared helpers for full-production runs (per-case resume, markers, stop rule).

The layout these helpers serve is fixed in docs/bench-production-full-layout-20261109.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PRODUCTION_SCOPES = ("development_full", "development_challenge")
DEVELOPMENT_MAPS = ("munich", "rotterdam", "maastricht")
ARRAY_SIZES = (64, 128, 256)
# Preregistered fail-closed rule: a map stops when MORE than 2% of its cases fail.
STOP_RULE_FRACTION_NUM, STOP_RULE_FRACTION_DEN = 2, 100
# Physical cores on 3060-3 are CPU k and k+6; core 5 (CPUs 5 and 11) is reserved for timing.
FORBIDDEN_CPUS = frozenset({5, 11})


def stop_rule_exceeded(failed: int, total: int) -> bool:
    """True when failed/total > 2%, in exact integer arithmetic (120 cases: 3 fail; 23 cases: 1 fails)."""
    if total <= 0 or failed < 0 or failed > total:
        raise ValueError(f"invalid stop-rule counts: failed={failed} total={total}")
    return failed * STOP_RULE_FRACTION_DEN > STOP_RULE_FRACTION_NUM * total


def atomic_write_json(path: Path, value) -> None:
    """Write JSON through a same-directory temp file, fsync it, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def cases_by_map(cases: list[dict]) -> dict[str, list[dict]]:
    """Group manifest cases by map, in manifest order, refusing non-development maps."""
    out: dict[str, list[dict]] = {}
    for case in cases:
        if case["map_id"] not in DEVELOPMENT_MAPS:
            raise ValueError(f"non-development map in production manifest: {case['map_id']}")
        out.setdefault(case["map_id"], []).append(case)
    return out


def check_cpu_set(cpus) -> tuple[int, ...]:
    """Validate a worker CPU set: non-empty, integers, and never CPUs 5 or 11."""
    cpus = tuple(sorted({int(c) for c in cpus}))
    if not cpus or FORBIDDEN_CPUS.intersection(cpus):
        raise ValueError(f"worker CPU set must be non-empty and exclude 5 and 11: {cpus}")
    return cpus
