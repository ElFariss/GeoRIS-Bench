"""Conservative classification of the first hybrid-gate artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


FULL_SIZE_DIR = "outputs/runs/hybrid_gates_v2_20261107"
FULL_SIZE_PRIMARY_MESH = 81


def audit_full_size_p1_p2(root: str | Path = ".", prefix: str = "", required_p2_builder: str | None = None) -> dict[str, Any]:
    """Register the full-size P1/P2 artifacts honestly (docs/hybrid-gates-v2-20261107.md).

    A gate counts as passed at a mesh only if the artifact is not a smoke run, every criterion recorded in
    it passed, and the tolerances recorded in it equal the declared constants in ``full_validation``.
    The primary mesh (81) must pass; if a 161 artifact exists it must pass too for an unqualified pass.
    """
    from atomwise_ris.hybrid import full_validation as fv

    declared = {"P1": fv.THRESHOLDS_P1, "P2": fv.THRESHOLDS_P2}
    base = Path(root) / FULL_SIZE_DIR
    meshes: dict[int, dict[str, Any]] = {}
    for sub in sorted(base.glob(f"{prefix}seg*")) if base.is_dir() else []:
        try:
            seg = int(sub.name[len(prefix) + 3:])
        except ValueError:
            continue
        rec: dict[str, Any] = {"directory": str(sub)}
        for gate in ("P1", "P2"):
            path = sub / f"{gate.lower()}.json"
            if not path.is_file():
                rec[gate] = {"status": "not_run"}
                continue
            data = _load(path)
            failed = [c["id"] for c in data.get("criteria", []) if not c.get("passed")]
            tolerances_ok = data.get("thresholds") == declared[gate]
            smoke = bool(data.get("smoke"))
            builder_ok = gate != "P2" or required_p2_builder is None or data.get("builder") == required_p2_builder
            ok = (bool(data.get("criteria")) and not failed and tolerances_ok and not smoke
                  and bool(data.get("passed")) and builder_ok)
            rec[gate] = {
                "status": "passed" if ok else ("smoke_not_evidence" if smoke else "failed" if failed else "invalid_artifact"),
                "failed_criteria": failed, "tolerances_equal_declared": tolerances_ok,
                "segments_per_wire": data.get("segments_per_wire"), "sha256": sha256_file(path),
            }
            if gate == "P2":
                variant = data.get("variant_full_external_coupling", {})
                rec[gate]["variant_full_external_coupling_passed"] = variant.get("passed_p2v")
        meshes[seg] = rec
    out: dict[str, Any] = {"meshes": {str(k): v for k, v in meshes.items()}}
    for gate in ("P1", "P2"):
        primary = meshes.get(FULL_SIZE_PRIMARY_MESH, {}).get(gate, {"status": "not_run"})["status"]
        fine = meshes.get(161, {}).get(gate, {"status": "not_run"})["status"]
        if primary != "passed":
            status = "not_run" if primary == "not_run" else "failed_or_invalid_at_81"
        elif fine == "not_run":
            status = "passed_at_81_only"
        elif fine == "passed":
            status = "passed_at_81_and_161"
        else:
            status = "passed_at_81_failed_or_invalid_at_161"
        out[gate] = status
    return out


def audit_legacy_gate_evidence(root: str | Path = ".") -> dict[str, Any]:
    """Assess what the saved gate files actually demonstrate, without rerunning them."""
    root = Path(root)
    old = root / "outputs/runs/hybrid_gates"
    p1 = _load(old / "gate_P1.json")
    p2 = _load(old / "gate_P2.json")
    p3 = _load(old / "gate_P3.json")
    p4 = _load(old / "gate_P4.json")
    summary = _load(old / "gates_summary.json")
    action_path = root / "outputs/runs/coupled_wire_v1/action_mesh_audit.json"
    action_audit = _load(action_path)

    p3_z_paths = [
        root / "outputs/runs/coupling_halfwave64_v1/z_side8_segments41.npy",
        root / "outputs/runs/coupling_halfwave64_v1/z_side8_segments81.npy",
        root / "outputs/runs/coupled_wire_v1/z_reference161.npy",
    ]
    p3_z_hashes = {p.name: sha256_file(p) for p in p3_z_paths if p.is_file()}
    legacy_paths = [old / f"gate_P{i}.json" for i in range(1, 5)] + [old / "gates_summary.json"]

    full_size = audit_full_size_p1_p2(root)
    fix1 = audit_full_size_p1_p2(root, prefix="fix1_", required_p2_builder="fixed_nec_external_plus_open_gap")
    return {
        "audit_version": "2026-11-07-evidence-reclassification-v2",
        "full_size_p1_p2_v2": full_size,
        "full_size_p1_p2_v2_fix1_rerun_fixed_hybrid": fix1,
        "scope": "read-only evidence audit; no optimizer or training authorization",
        "legacy_summary_claimed_all_passed": bool(summary.get("all_passed")),
        "legacy_artifact_sha256": {p.name: sha256_file(p) for p in legacy_paths},
        "gates": {
            "P1": {
                "status": "not_verified_by_legacy_artifact",
                "legacy_claimed_pass": bool(p1.get("passed")),
                "observed": p1.get("metrics", {}),
                "reason": "The saved Schur residual compares two algebraic expressions built from the same NEC Z matrix. It verifies the reduction implementation, not independent direct-NEC port response under incident fields or loads.",
                "needed": "Compare reciprocal incident-field port excitation and loaded currents against independent NEC source-card solves on the full 64-port geometry; retain exact geometry, source, and mesh provenance.",
            },
            "P2": {
                "status": "not_verified_by_legacy_artifact",
                "legacy_claimed_pass": bool(p2.get("passed")),
                "observed": p2.get("metrics", {}),
                "reason": "The saved comparison constructs both channels from the same monolithic NEC Z matrix. It does not compare the hybrid field/excitation operators with an independent direct full-wave solve.",
                "needed": "Compare complex channels and state ordering against direct NEC on a declared free-space geometry, with independent direct and hybrid operator paths.",
            },
            "P3": {
                "status": "partial_mesh_evidence_action_error_observed",
                "legacy_claimed_pass": bool(p3.get("passed")),
                "legacy_observed": p3.get("metrics", {}),
                "mesh_matrix_sha256": p3_z_hashes,
                "action_audit_sha256": sha256_file(action_path),
                "action_audit_scope": action_audit.get("scope"),
                "action_audit_actions": action_audit.get("actions"),
                "action_audit_max_channel_relative_error": action_audit.get("max_channel_relative_error"),
                "action_audit_max_rate_absolute_error": action_audit.get("max_rate_absolute_error"),
                "reason": "Mesh matrices are genuine NEC artifacts, but the legacy gate samples random states and one-step best single/pair moves rather than completed returned optimizer actions. A separate existing returned-action audit covers 54 actions on one development pilot frame and reports a maximum rate shift of 0.643806341438097 bit/s/Hz; it is useful warning evidence, not a broad convergence gate.",
                "needed": "Rescore the frozen returned actions across the declared validation trajectories on matched meshes and report rank reversals and trajectory-level shifts without tuning a threshold after seeing results.",
            },
            "P4": {
                "status": "blocked_uncalibrated_operator",
                "legacy_claimed_pass": bool(p4.get("passed")),
                "observed": p4.get("metrics", {}),
                "reason": "The legacy path constructs incoming and outgoing scales using lambda/pi and probe-lattice H values, then checks only finiteness, channel magnitude, load sensitivity, and matrix rank. It contains no direct LOS channel comparison or phase/polarization normalization check at NEC wire positions.",
                "needed": "Use actual calibrated path fields at NEC segment locations and compare phase, polarization, incident port response, and outgoing transfer to independent direct NEC reference cases before adding urban multipath.",
            },
        },
        "optimizer_authorized": False,
        "training_authorized": False,
    }
