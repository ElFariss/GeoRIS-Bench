"""No-fit convention-only replay of the short-wire mutual-impedance check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.embedded import C0, ETA0
from atomwise_ris.hybrid.p4_vector_mapping import (
    current_to_free_space_field,
    field_to_open_circuit_voltage,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sinusoidal_profile(length, n, frequency):
    dl = length / n
    z = (np.arange(n) + 0.5) * dl - length / 2
    k = 2 * np.pi * frequency / C0
    profile = np.sin(k * (length / 2 - np.abs(z))) / np.sin(k * length / 2)
    profile /= profile[n // 2]
    return z, np.full(n, dl), profile.astype(np.complex128)


def z_mutual(tx, rx, src_centers, rx_centers, tangents, lengths, f, c, eta):
    field = current_to_free_space_field(tx, src_centers, tangents, lengths,
                                        rx_centers, f, c, eta)
    return complex(field_to_open_circuit_voltage(rx, rx_centers, tangents,
                                                 lengths, field)[0])


def metrics(z, target):
    return {
        "z_mutual_ohm": [float(z.real), float(z.imag)],
        "complex_relative_error": float(abs(z - target) / abs(target)),
        "wrapped_phase_error_rad": float(abs(np.angle(z * np.conjugate(target)))),
        "copolar_amplitude_ratio": float(abs(z) / abs(target)),
    }


def run(reference: Path, segments=81):
    with np.load(reference) as data:
        bs = np.asarray(data["bs_centers_xyz_m"], dtype=float)[0]
        ue = np.asarray(data["ue_centers_xyz_m"], dtype=float)[0]
        z_ee = np.asarray(data["z_ee_without_ris"], dtype=np.complex128)
    f = 5.53e9
    n = segments
    length = 0.024
    zoffs, dl, current = sinusoidal_profile(length, n, f)
    tx_centers = bs[None, :] + np.column_stack([np.zeros(n), np.zeros(n), zoffs])
    rx_centers = ue[None, :] + np.column_stack([np.zeros(n), np.zeros(n), zoffs])
    tangents = np.tile([0.0, 0.0, 1.0], (n, 1))
    c_nec = 1.0 / np.sqrt(4.0 * np.pi * 1.0e-7 * 8.854e-12)
    eta_nec = np.sqrt((4.0 * np.pi * 1.0e-7) / 8.854e-12)
    target = z_ee[1, 0]

    # Correct current-moment convention under NEC source constants.
    z_nec = z_mutual(current, current, tx_centers, rx_centers, tangents, dl, f, c_nec, eta_nec)
    # Historical code passed I*dl/j into a kernel that expects I*dl. This
    # reproduces that error under the same source-derived NEC constants.
    z_wrong_moment = z_mutual(current / 1j, current, tx_centers, rx_centers,
                              tangents, dl, f, c_nec, eta_nec)
    # Physical SI convention, using physical coordinates and exact SI constants.
    z_si = z_mutual(current, current, tx_centers, rx_centers, tangents, dl, f, C0, ETA0)

    z_nec_matrix = np.diag(np.diag(z_ee)).astype(np.complex128)
    z_nec_matrix[0, 1] = z_nec_matrix[1, 0] = z_nec
    h_nec = -100.0 * np.linalg.inv(z_nec_matrix + 50 * np.eye(2))[1, 0]
    h_target = -100.0 * np.linalg.inv(z_ee + 50 * np.eye(2))[1, 0]
    historical_path = Path("outputs/runs/bench_tracks_p4_20261108/free_space_dipole_subcheck.json")
    historical = json.loads(historical_path.read_text()) if historical_path.exists() else None
    return {
        "status": "measured_subcheck",
        "gate": "P4_NOT_EVALUATED",
        "scope": "one free-space BS0-UE0 pair, sinusoidal-current thin-wire approximation; convention replay only",
        "criteria_unchanged": {"complex_relative_error_max": 0.01,
                               "wrapped_phase_error_rad_max": 0.01,
                               "copolar_amplitude_ratio": [0.99, 1.01]},
        "source_constants": {"PyNEC_source_version": "2.3.4",
                              "PyNEC_sdist_sha256": "d448d038d2bcb4265d511b2440a5d5ce26081269a776d45139fb8bf17a475df4",
                              "c_nec_m_per_s": float(c_nec), "eta_nec_ohm": float(eta_nec),
                              "c_si_m_per_s": C0, "eta_si_ohm": ETA0},
        "comparison_to_nec_mutual_impedance": {
            "historical_result_with_p_equals_I_dl_over_j": historical.get("metrics") if historical else None,
            "source_correct_current_moment_I_dl_with_NEC_constants": metrics(z_nec, target),
            "same_current_moment_under_exact_SI_constants": metrics(z_si, target),
            "same_NEC_constants_with_historical_extra_1_over_j": metrics(z_wrong_moment, target),
        },
        "direct_channel_with_nec_self_impedance_copied_unchanged": {
            "complex_relative_error": float(abs(h_nec - h_target) / abs(h_target)),
            "wrapped_phase_error_rad": float(abs(np.angle(h_nec * np.conjugate(h_target)))),
            "copolar_amplitude_ratio": float(abs(h_nec) / abs(h_target)),
        },
        "interpretation": "The `I*dl/j` error is a fixed 90-degree current-moment convention mistake. NEC's source-derived propagation constants are separately required for comparison with a NEC target; exact-SI propagation is reserved for a physical SI model and is not phase-aligned to NEC.",
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
