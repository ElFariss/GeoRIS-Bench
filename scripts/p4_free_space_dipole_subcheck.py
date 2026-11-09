"""No-fit free-space field-to-wire subcheck against the saved NEC port reference.

This tests a deliberately explicit sinusoidal-current thin-wire approximation.
The exact NEC self impedances and the reference channel convention are retained;
only the BS-to-UE mutual impedance is independently predicted from the dipole
Green function and Lorentz reciprocity. It is a subcheck, not the Munich P4 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.embedded import C0, zdip_E


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sinusoidal_current_profile(length: float, nseg: int, frequency_hz: float):
    """Center-fed sinusoidal thin-wire current, normalized to 1 A at the feed."""
    dl = length / nseg
    z = (np.arange(nseg) + 0.5) * dl - length / 2
    k = 2 * np.pi * frequency_hz / C0
    denom = np.sin(k * length / 2)
    if abs(denom) < 1e-10:
        raise ValueError("invalid sinusoidal-current normalization")
    current = np.sin(k * (length / 2 - np.abs(z))) / denom
    current /= current[nseg // 2]
    return z, np.full(nseg, dl), current


def predicted_mutual_impedance(tx: np.ndarray, rx: np.ndarray, length: float,
                               nseg: int, frequency_hz: float):
    """Predict Z_rx,tx (ohm) by source-segment fields and receive reciprocity."""
    zoff, dl, current = sinusoidal_current_profile(length, nseg, frequency_hz)
    src = tx[None, :] + np.column_stack([np.zeros(nseg), np.zeros(nseg), zoff])
    obs = rx[None, :] + np.column_stack([np.zeros(nseg), np.zeros(nseg), zoff])
    omega = 2 * np.pi * frequency_hz
    e_total = np.zeros((nseg, 3), dtype=np.complex128)
    for q in range(nseg):
        # ``zdip_E`` multiplies its ``p`` argument by k*eta, so its
        # implementation's effective source variable is omega times the
        # physical electric dipole moment I*dl/(j*omega).  Passing the
        # physical moment directly (as an earlier draft did) omits omega
        # and suppresses the field by 2*pi*f.  This equivalent current
        # moment has no fitted scale or phase.
        p = current[q] * dl[q] / 1j
        e_total += zdip_E(obs, src[q], p=p, frequency=frequency_hz)
    # The receive basis current is also normalized to 1 A at its feed.
    z_mut = -np.sum(current * dl * e_total[:, 2])
    return complex(z_mut), e_total


def run(reference: Path, segments_per_wire: int = 81) -> dict:
    with np.load(reference) as z:
        z_ee = np.asarray(z["z_ee_without_ris"], dtype=np.complex128)
        h_nec = np.asarray(z["h_direct_without_ris"], dtype=np.complex128)
        bs = np.asarray(z["bs_centers_xyz_m"], dtype=float)
        ue = np.asarray(z["ue_centers_xyz_m"], dtype=float)
    if z_ee.shape != (2, 2) or h_nec.shape != (1, 1) or bs.shape != (1, 3) or ue.shape != (1, 3):
        raise ValueError("reference does not contain the expected BS0, UE0 two-port geometry")
    frequency = 5.53e9
    length = .024
    zmut, e_rx_segments = predicted_mutual_impedance(
        bs[0], ue[0], length, segments_per_wire, frequency)
    z_model = np.diag(np.diag(z_ee)).astype(np.complex128)
    z_model[0, 1] = z_model[1, 0] = zmut
    h_model = -100.0 * np.linalg.inv(z_model + 50.0 * np.eye(2))[1:2, 0:1]
    err = h_model - h_nec
    rel = float(np.linalg.norm(err) / np.linalg.norm(h_nec))
    phase = float(np.abs(np.angle(h_model[0, 0] * np.conjugate(h_nec[0, 0]))))
    ratio = float(abs(h_model[0, 0]) / abs(h_nec[0, 0]))
    # z-directed receive port: this records the incident vector's horizontal
    # content at its NEC segment centres, but there is no NEC vector-field
    # reference in the saved artifact against which a cross-pol gate can score.
    total = np.linalg.norm(e_rx_segments, axis=1)
    cross_ratio = np.linalg.norm(e_rx_segments[:, :2], axis=1) / np.maximum(total, np.finfo(float).tiny)
    return {
        "status": "measured_subcheck",
        "gate": "P4_NOT_EVALUATED",
        "scope": "one free-space BS0-to-UE0 port pair; sinusoidal-current dipole field approximation; no RIS and no fitting",
        "frequency_hz": frequency,
        "wire_length_m": length,
        "segments_per_wire": segments_per_wire,
        "port_order": ["BS0", "UE0"],
        "channel_convention": "-100 * inv(Z_external + 50*I)[UE0,BS0]",
        "field_to_port_mapping": "source current moment I_q*dl with e^(-j*omega*t) convention; exact z-directed Hertzian Green field e^(-jkr), receive V_oc=-sum(I_rx*dl*E_z); sinusoidal current normalized to 1 A feed current",
        "green_function_source_variable": "zdip_E p argument is omega times the physical electric dipole moment; p=I*dl/j",
        "self_impedances_source": "NEC reference diagonal copied unchanged; no fitted or adjusted values",
        "z_mutual_predicted_ohm": {"real": zmut.real, "imag": zmut.imag},
        "z_mutual_nec_reference_ohm": {"real": float(z_ee[1, 0].real), "imag": float(z_ee[1, 0].imag)},
        "h_predicted": {"real": float(h_model[0, 0].real), "imag": float(h_model[0, 0].imag)},
        "h_nec_reference": {"real": float(h_nec[0, 0].real), "imag": float(h_nec[0, 0].imag)},
        "metrics": {
            "complex_relative_error": rel,
            "wrapped_phase_error_rad": phase,
            "copolar_amplitude_ratio": ratio,
            "complex_relative_error_limit": .01,
            "wrapped_phase_error_limit_rad": .01,
            "copolar_amplitude_ratio_limit": [.99, 1.01],
            "each_scalar_limit_pass": {
                "complex_relative_error": rel <= .01,
                "wrapped_phase_error_rad": phase <= .01,
                "copolar_amplitude_ratio": .99 <= ratio <= 1.01,
            },
        },
        "polarization_diagnostic": {
            "incident_field_projection": "z-directed wire basis",
            "max_horizontal_to_total_field_ratio_across_receive_segments": float(cross_ratio.max()),
            "cross_polar_gate": "not scoreable: saved NEC reference contains port impedance/channel, not vector field samples",
        },
        "limitation": "This checks an idealized sinusoidal thin-wire current model against the NEC two-port reference. The 24 mm wires are electrically short and NEC currents are not sinusoidal in this near-grazing 246 m geometry; the large residual is a failed approximation, not a P4 calibration score. It is not a Sionna-to-NEC comparison and cannot establish the full P4 gate.",
        "reference_path": str(reference),
        "reference_sha256": sha256(reference),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="outputs/runs/hybrid_track2_20261108/munich_los_nec_reference.npz")
    ap.add_argument("--segments", type=int, default=81)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = run(Path(args.reference), args.segments)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
