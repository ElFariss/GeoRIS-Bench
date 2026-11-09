"""Fixed, no-fit metrics for the P4 ray-tracer / NEC comparison."""
from __future__ import annotations

import numpy as np


P4_LIMITS = {
    "complex_relative_error_max": 0.01,
    "wrapped_phase_error_rad_max": 0.01,
    "copolar_amplitude_ratio": (0.99, 1.01),
    "cross_polar_leakage_max": 0.01,
}


def complex_metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    """Compare complex responses without fitting a scale or phase correction.

    `model` and `reference` must already use the same port order, polarization
    basis, source amplitude, and reference plane. This function deliberately
    does not align or normalize either input.
    """
    a = np.asarray(model, dtype=np.complex128)
    b = np.asarray(reference, dtype=np.complex128)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("model and reference must be nonempty arrays with identical shape")
    nr = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    if nr == 0.0:
        raise ValueError("reference response has zero norm")
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    phase_mask = np.abs(flat_b) > (np.finfo(float).eps * max(1.0, float(np.max(np.abs(flat_b)))))
    if not np.any(phase_mask):
        raise ValueError("reference has no nonzero entries for phase comparison")
    per_entry_phase = np.angle(flat_a[phase_mask] * np.conjugate(flat_b[phase_mask]))
    phase = float(np.max(np.abs(per_entry_phase)))
    metrics = {
        "complex_relative_error": float(np.linalg.norm(a - b) / nr),
        "copolar_amplitude_ratio": na / nr,
        "wrapped_phase_error_rad": phase,
    }
    metrics["complex_pass"] = metrics["complex_relative_error"] <= P4_LIMITS["complex_relative_error_max"]
    metrics["phase_pass"] = metrics["wrapped_phase_error_rad"] <= P4_LIMITS["wrapped_phase_error_rad_max"]
    lo, hi = P4_LIMITS["copolar_amplitude_ratio"]
    metrics["amplitude_pass"] = lo <= metrics["copolar_amplitude_ratio"] <= hi
    metrics["all_scalar_pass"] = bool(metrics["complex_pass"] and metrics["phase_pass"] and metrics["amplitude_pass"])
    return metrics


def cross_polar_leakage(field: np.ndarray, copolar_unit: np.ndarray) -> float:
    """Return the orthogonal-polarization amplitude ratio for 2/3-vector fields."""
    e = np.asarray(field, dtype=np.complex128).reshape(-1)
    u = np.asarray(copolar_unit, dtype=np.complex128).reshape(-1)
    if e.shape != u.shape or e.size not in (2, 3):
        raise ValueError("field and polarization unit vector must be matching 2D/3D vectors")
    un = float(np.linalg.norm(u))
    if un == 0.0:
        raise ValueError("polarization vector must be nonzero")
    u = u / un
    e_norm = float(np.linalg.norm(e))
    if e_norm == 0.0:
        return 0.0
    cross = e - u * np.vdot(u, e)
    return float(np.linalg.norm(cross) / e_norm)

