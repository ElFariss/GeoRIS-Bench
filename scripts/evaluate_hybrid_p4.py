"""Fail-closed P4 comparator: ray H_direct plus predicted open-gap correction vs NEC."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.p4_calibration import P4_LIMITS, complex_metrics, cross_polar_leakage


REQUIRED = (
    "absolute_mapping", "calibration_fit_applied", "frequency_hz", "port_order",
    "polarization_basis", "rt_incident_complex", "nec_incident_complex",
    "rt_incident_field_xyz", "nec_incident_field_xyz", "rt_direct_h",
    "nec_direct_h", "z_ee_free", "delta_z_open_gap_pred", "n_bs",
    "delta_z_provenance",
)
PROVENANCE_REQUIRED = (
    "origin", "builder", "builder_source_sha256", "ris_alone_response_sha256",
    "incident_field_sha256", "external_nec_sha256", "used_monolithic_truth",
    "calibration_fit_applied",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_builder_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    return _sha256(root / "src/atomwise_ris/hybrid/ray_coupling.py")


def _complex_array(value):
    """Decode JSON complex arrays as trailing [real, imag] pairs."""
    x = np.asarray(value, dtype=float)
    if x.ndim < 1 or x.shape[-1] != 2:
        raise ValueError("complex values must encode each scalar as [real, imag]")
    return x[..., 0] + 1j * x[..., 1]


def _direct(z_external: np.ndarray, n_bs: int) -> np.ndarray:
    n = len(z_external)
    if z_external.ndim != 2 or z_external.shape != (n, n) or not 0 < n_bs < n:
        raise ValueError("z_ee_free must be square and n_bs must partition external ports")
    return -100.0 * np.linalg.inv(z_external + 50.0 * np.eye(n, dtype=np.complex128))[n_bs:, :n_bs]


def _check_predicted_delta_provenance(prov: dict) -> str | None:
    missing = [k for k in PROVENANCE_REQUIRED if k not in prov]
    if missing:
        return "delta_z_provenance missing: " + ", ".join(missing)
    if prov.get("origin") != "ris_alone_fixed_builder":
        return "open-gap delta-Z must come from the RIS-alone fixed builder; monolithic truth is not a predictor"
    if prov.get("builder") != "build_spherical_nec_external_operators":
        return "unrecognized open-gap delta-Z builder"
    if prov.get("builder_source_sha256") != expected_builder_sha256():
        return "builder source hash does not match the current fixed builder"
    for field in ("ris_alone_response_sha256", "incident_field_sha256", "external_nec_sha256"):
        value = str(prov.get(field, ""))
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            return f"{field} must be a SHA-256 digest"
    if prov.get("used_monolithic_truth") is not False:
        return "monolithic truth-derived delta-Z is forbidden as a predictor"
    if prov.get("calibration_fit_applied") is not False:
        return "fitted delta-Z is prohibited"
    return None


def evaluate(data: dict) -> dict:
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        why = "historical truth-derived delta_z_open_gap is not an allowed prediction input" if "delta_z_open_gap" in data else "missing independently calibrated inputs"
        return {"status": "blocked", "gate": "P4_NOT_EVALUATED", "reason": why,
                "missing": missing, "criteria": P4_LIMITS}
    if data["calibration_fit_applied"] is not False:
        return {"status": "blocked", "gate": "P4_NOT_EVALUATED",
                "reason": "per-instance fitted calibration is prohibited", "criteria": P4_LIMITS}
    if not str(data["absolute_mapping"]).strip():
        return {"status": "blocked", "gate": "P4_NOT_EVALUATED",
                "reason": "absolute mapping convention must be stated", "criteria": P4_LIMITS}
    provenance_error = _check_predicted_delta_provenance(data["delta_z_provenance"])
    if provenance_error:
        return {"status": "blocked", "gate": "P4_NOT_EVALUATED",
                "reason": provenance_error, "criteria": P4_LIMITS}
    try:
        rt_i = _complex_array(data["rt_incident_complex"])
        nec_i = _complex_array(data["nec_incident_complex"])
        rt_e = _complex_array(data["rt_incident_field_xyz"])
        nec_e = _complex_array(data["nec_incident_field_xyz"])
        rt_direct = _complex_array(data["rt_direct_h"])
        nec_direct = _complex_array(data["nec_direct_h"])
        z_ee = _complex_array(data["z_ee_free"])
        dz_pred = _complex_array(data["delta_z_open_gap_pred"])
        n_bs = int(data["n_bs"])
        if z_ee.shape != dz_pred.shape:
            raise ValueError("z_ee_free and predicted delta-Z must have matching shape")
        d_free = _direct(z_ee, n_bs)
        d_pred = _direct(z_ee + dz_pred, n_bs)
        delta_d = d_pred - d_free
        corrected_direct = rt_direct + delta_d
        m_incident = complex_metrics(rt_i, nec_i)
        m_direct_ray = complex_metrics(rt_direct, nec_direct)
        m_direct_corrected = complex_metrics(corrected_direct, nec_direct)
        if rt_e.shape != nec_e.shape or rt_e.shape[-1] != 3:
            raise ValueError("incident field arrays must match and end in xyz dimension 3")
        unit = nec_e / np.maximum(np.linalg.norm(nec_e, axis=-1, keepdims=True), np.finfo(float).tiny)
        leakage = [cross_polar_leakage(v, u) for v, u in zip(rt_e.reshape(-1, 3), unit.reshape(-1, 3))]
        cross_max = float(max(leakage, default=float("nan")))
        passed = bool(m_incident["all_scalar_pass"] and m_direct_corrected["all_scalar_pass"]
                      and cross_max <= P4_LIMITS["cross_polar_leakage_max"])
        return {
            "status": "complete", "gate": "P4", "passed": passed,
            "scope": "single Munich LOS; no-fit ray direct plus RIS-alone fixed-builder open-gap correction vs monolithic NEC",
            "frequency_hz": float(data["frequency_hz"]),
            "absolute_mapping": data["absolute_mapping"],
            "calibration_fit_applied": False,
            "delta_z_provenance": data["delta_z_provenance"],
            "delta_h_open_gap_real": delta_d.real.tolist(),
            "delta_h_open_gap_imag": delta_d.imag.tolist(),
            "incident_response": m_incident,
            "direct_h_ray_vs_nec": m_direct_ray,
            "direct_h_ray_plus_predicted_open_gap_vs_nec": m_direct_corrected,
            "cross_polar_leakage_max": cross_max,
            "cross_polar_pass": bool(cross_max <= P4_LIMITS["cross_polar_leakage_max"]),
            "criteria": P4_LIMITS,
        }
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        return {"status": "blocked", "gate": "P4_NOT_EVALUATED",
                "reason": f"input validation/calculation failed: {exc}", "criteria": P4_LIMITS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON with absolute RT and independent NEC inputs")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = evaluate(json.loads(Path(args.input).read_text()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
