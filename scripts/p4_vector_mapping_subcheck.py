"""Run a no-fit NEC-current / Cartesian-field mapping subcheck (not full P4)."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.embedded import C0, ETA0, extract_ris_response
from atomwise_ris.hybrid.p4_vector_mapping import mutual_impedance_from_isolated_currents


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _h_from_z(z):
    return -100.0 * np.linalg.inv(z + 50.0 * np.eye(len(z), dtype=np.complex128))[-1:, :1]


def run(reference: Path, segments: int = 81):
    # Geometry-only read for prediction. No impedance, channel, or open-gap
    # truth from this artifact is passed to the mapping or NEC current solves.
    with np.load(reference) as z:
        bs_xyz = np.asarray(z["bs_centers_xyz_m"], dtype=float)
        ue_xyz = np.asarray(z["ue_centers_xyz_m"], dtype=float)
    if bs_xyz.shape != (1, 3) or ue_xyz.shape != (1, 3):
        raise ValueError("expected one BS0 and one UE0 position")

    frequency = 5.53e9
    length = 0.024
    # Separate single-wire solves avoid using the monolithic two-port matrix
    # to construct either the transmit or receive port-current basis.
    tx = extract_ris_response(bs_xyz, wire_length=length,
                              segments_per_wire=segments, frequency=frequency)
    rx = extract_ris_response(ue_xyz, wire_length=length,
                              segments_per_wire=segments, frequency=frequency)
    tx_tangent = np.tile([0.0, 0.0, 1.0], (len(tx.segment_lengths), 1))
    rx_tangent = np.tile([0.0, 0.0, 1.0], (len(rx.segment_lengths), 1))
    # Preserve the historical metre-labelled helper representation (SI
    # wavelength multiplied by NEC-normalized getters) for a reproducible
    # NEC-consistency result, then run the source-correct physical coordinates.
    z_normalized = mutual_impedance_from_isolated_currents(
        tx, rx, tx_tangent, rx_tangent, frequency)
    eps_nec = 8.854e-12
    mu_nec = 4.0 * np.pi * 1e-7
    c_nec = 1.0 / np.sqrt(mu_nec * eps_nec)
    eta_nec = np.sqrt(mu_nec / eps_nec)
    nec_to_si_coord = c_nec / C0
    tx_phys = replace(tx, segment_centers=tx.segment_centers * nec_to_si_coord,
                      segment_lengths=tx.segment_lengths * nec_to_si_coord,
                      wire_centers=tx.wire_centers * nec_to_si_coord)
    rx_phys = replace(rx, segment_centers=rx.segment_centers * nec_to_si_coord,
                      segment_lengths=rx.segment_lengths * nec_to_si_coord,
                      wire_centers=rx.wire_centers * nec_to_si_coord)
    z_pred = mutual_impedance_from_isolated_currents(
        tx_phys, rx_phys, tx_tangent, rx_tangent, frequency,
        speed_of_light_m_s=c_nec, intrinsic_impedance_ohm=eta_nec)
    z_physical_si = mutual_impedance_from_isolated_currents(
        tx_phys, rx_phys, tx_tangent, rx_tangent, frequency, C0, ETA0)

    # Target access is deliberately separated from prediction construction.
    with np.load(reference) as z:
        z_target = np.asarray(z["z_ee_without_ris"], dtype=np.complex128)
        h_target = np.asarray(z["h_direct_without_ris"], dtype=np.complex128)
    if z_target.shape != (2, 2) or h_target.shape != (1, 1):
        raise ValueError("reference target arrays have unexpected shape")

    z_pred_matrix = np.diag(np.diag(z_target)).astype(np.complex128)
    z_pred_matrix[0, 1] = z_pred_matrix[1, 0] = z_pred
    h_pred = _h_from_z(z_pred_matrix)
    z_ref = z_target[1, 0]
    h_ref = h_target[0, 0]
    zrel = abs(z_pred - z_ref) / abs(z_ref)
    zphase = abs(np.angle(z_pred * np.conjugate(z_ref)))
    zamp = abs(z_pred) / abs(z_ref)
    hrel = float(np.linalg.norm(h_pred - h_target) / np.linalg.norm(h_target))
    hphase = float(abs(np.angle(h_pred[0, 0] * np.conjugate(h_ref))))
    hamp = float(abs(h_pred[0, 0]) / abs(h_ref))
    return {
        "status": "measured_subcheck",
        "gate": "P4_NOT_EVALUATED",
        "scope": "free-space BS0-to-UE0 reciprocal port mapping; no RIS, no Sionna, no fitted scale or phase",
        "phasor_convention": "exp(+j omega t), outgoing Green factor exp(-j k r); NEC EFIE convention",
        "mapping": {
            "incoming": "Voc[n] = -sum_m I_mn^(1A port current) dl_m (t_m dot E_inc,m); Cartesian polarization projection, unconjugated reciprocity product",
            "outgoing": "sum_m dyadic free-space Hertzian-segment fields for complex NEC currents I_m dl_m t_m, including 1/r, 1/r^2, 1/r^3 terms",
            "transmit_receive_current_basis": "two independent isolated one-wire PyNEC solves; not the monolithic two-port matrix",
            "physical_coordinates": "current-centre/length getters converted from NEC wavelengths with SI wavelength are contracted; physical metres restored by c_NEC/C0 before the source-matched NEC map",
            "source_constants": {"epsilon_f_per_m": eps_nec, "mu_h_per_m": mu_nec,
                                 "c_nec_m_per_s": c_nec, "eta_nec_ohm": eta_nec,
                                 "coordinate_restore_ratio": nec_to_si_coord},
            "h_channel_convention": "-100 * inv(Z_external + 50*I)[UE0,BS0]",
        },
        "frequency_hz": frequency,
        "segments_per_wire": segments,
        "wire_length_m": length,
        "z_mutual_predicted_ohm": [float(z_pred.real), float(z_pred.imag)],
        "z_mutual_reference_ohm": [float(z_ref.real), float(z_ref.imag)],
        "z_mutual_metrics": {
            "complex_relative_error": float(zrel),
            "wrapped_phase_error_rad": float(zphase),
            "copolar_amplitude_ratio": float(zamp),
            "complex_relative_error_limit": 0.01,
            "wrapped_phase_error_limit_rad": 0.01,
            "copolar_amplitude_ratio_limit": [0.99, 1.01],
        },
        "coordinate_and_propagation_convention_checks": {
            "nec_normalized_centres_with_si_kernel": {
                "z_mutual_ohm": [float(z_normalized.real), float(z_normalized.imag)],
                "complex_relative_error": float(abs(z_normalized - z_ref) / abs(z_ref)),
                "wrapped_phase_error_rad": float(abs(np.angle(z_normalized * np.conjugate(z_ref)))),
                "interpretation": "NEC-consistency cancellation only; not a physical SI coordinate map",
            },
            "physical_centres_with_exact_si_kernel": {
                "z_mutual_ohm": [float(z_physical_si.real), float(z_physical_si.imag)],
                "complex_relative_error_vs_nec": float(abs(z_physical_si - z_ref) / abs(z_ref)),
                "wrapped_phase_error_rad_vs_nec": float(abs(np.angle(z_physical_si * np.conjugate(z_ref)))),
                "interpretation": "Physical SI propagation convention; expected to differ from NEC's source-derived propagation constant",
            },
        },
        "direct_channel_diagnostic": {
            "self_impedances": "copied unchanged from NEC reference only for this channel reconstruction diagnostic; no fitting",
            "complex_relative_error": hrel,
            "wrapped_phase_error_rad": hphase,
            "copolar_amplitude_ratio": hamp,
        },
        "reference_path": str(reference),
        "reference_sha256": sha256(reference),
        "prediction_limits": [
            "NEC isolated-wire current basis and free-space Green integration form a consistency subcheck, not an independent Sionna calibration.",
            "Only z-directed ports are present; the implemented API handles arbitrary Cartesian tangents, but no independent cross-polar field reference is available.",
            "No RIS, RT incident vectors, corrected Munich direct channel, open-gap prediction, or matching reflection NEC reference was available here.",
        ],
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
