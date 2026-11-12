"""Small, direct-NEC hybrid consistency probes.

These deliberately use a 3x3 toy array. They are useful for checking units,
phase and port signs against independent NEC-2 solves, but cannot certify the
64-port urban model or optimizer behavior.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from atomwise_ris.hybrid.embedded import C0, extract_ris_response
from atomwise_ris.hybrid.operator import HybridCoupledNetwork
from atomwise_ris.hybrid.ray_coupling import build_spherical_dipole_operators
from atomwise_ris.physics.nec import WireGeometry


def _small_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ris = np.array(
        [[0.0, (j - 1) * 0.028, 1.0 + (i - 1) * 0.030]
         for i in range(3) for j in range(3)], dtype=float
    )
    bs = np.array([[-0.50, 0.0, 1.0]], dtype=float)
    ue = np.array([[0.50, 0.10, 1.0]], dtype=float)
    return ris, bs, ue


def _direct_sparameter(z: np.ndarray, bits: np.ndarray, loads: np.ndarray) -> complex:
    """One-BS/one-UE channel from the complete NEC port matrix and 50-ohm terminations."""
    n_ris = len(bits)
    rhs = np.zeros((n_ris + 2, 1), dtype=complex)
    rhs[n_ris, 0] = 1.0  # one-volt Thevenin source; ratio cancels its amplitude
    terminated = np.r_[loads[np.asarray(bits, dtype=int)], 50.0, 50.0]
    currents = np.linalg.solve(z + np.diag(terminated), rhs)
    return complex(-100.0 * currents[n_ris + 1, 0])


def run_direct_nec_microproof(segments_per_wire: int = 21) -> dict[str, Any]:
    """Run direct NEC P1/P2 probes on a 3x3 array and return measurements.

    The output is descriptive. It deliberately has no pass/fail thresholds and
    cannot promote the full-size P1/P2 gates.
    """
    if segments_per_wire < 3 or segments_per_wire % 2 != 1:
        raise ValueError("segments_per_wire must be odd and at least 3")

    from atomwise_ris.physics.nec import diagnostics

    frequency = 5.53e9
    side = 3
    ris, bs, ue = _small_geometry()
    wire_length = 0.024
    loads = np.array([1.0 - 10.8j, 1.0 + 129.2j], dtype=complex)
    response = extract_ris_response(
        ris, wire_length=wire_length, segments_per_wire=segments_per_wire,
        frequency=frequency,
    )

    # P1: an incident plane wave is solved independently by NEC and by the
    # port-current reciprocity integral. Angles use PyNEC's arrival convention.
    theta, phi = np.deg2rad(70.0), np.deg2rad(30.0)
    k = 2.0 * np.pi * frequency / C0
    k_arr = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    e_inc_z = -np.sin(theta) * np.exp(1j * k * (response.segment_centers @ k_arr))
    predicted_short = response.short_circuit_current_from_field(e_inc_z)

    geometry = WireGeometry(ris, np.full(side * side, wire_length), frequency=frequency)
    nec = geometry.context(segments_per_wire)
    nec.ex_card(1, 1, 1, 0, 70.0, 30.0, 0.0, 0, 0, 0)
    nec.xq_card(0)
    direct_all_currents = np.asarray(nec.get_structure_currents(0).get_current())
    mid = np.arange(side * side) * segments_per_wire + segments_per_wire // 2
    direct_short = direct_all_currents[mid]
    short_relative_error = float(
        np.linalg.norm(direct_short - predicted_short) / np.linalg.norm(direct_short)
    )

    voc_conventional = response.z_ris @ predicted_short
    voc_helper = response.open_circuit_voltage_from_field(e_inc_z)
    open_voltage_relative_error = float(
        np.linalg.norm(voc_helper + voc_conventional) / np.linalg.norm(voc_conventional)
    )

    loaded_current_errors = {}
    for label, bits in (
        ("all_off", np.zeros(side * side, dtype=np.int8)),
        ("all_on", np.ones(side * side, dtype=np.int8)),
        ("alternating", np.arange(side * side, dtype=np.int8) % 2),
    ):
        loaded_geometry = WireGeometry(
            ris, np.full(side * side, wire_length), frequency=frequency
        )
        loaded_nec = loaded_geometry.context(segments_per_wire, loads[np.asarray(bits, int)])
        loaded_nec.ex_card(1, 1, 1, 0, 70.0, 30.0, 0.0, 0, 0, 0)
        loaded_nec.xq_card(0)
        full_currents = np.asarray(loaded_nec.get_structure_currents(0).get_current())
        direct_loaded = full_currents[mid]
        predicted_loaded = -np.linalg.solve(
            response.z_ris + np.diag(loads[np.asarray(bits, int)]), voc_helper
        )
        loaded_current_errors[label] = float(
            np.linalg.norm(direct_loaded - predicted_loaded) / np.linalg.norm(direct_loaded)
        )

    # P2: compare the spherical Green-function hybrid approximation against a
    # complete, independently extracted 11-port NEC system (9 RIS + BS + UE).
    all_centers = np.vstack([ris, bs, ue])
    all_lengths = np.full(len(all_centers), wire_length)
    z_full, timing = WireGeometry(
        all_centers, all_lengths, frequency=frequency
    ).impedance(segments_per_wire)
    c_in, c_out, h_direct, a_matrix = build_spherical_dipole_operators(
        response, bs, ue, wire_length=wire_length, frequency=frequency
    )
    hybrid = HybridCoupledNetwork(a_matrix, c_out, c_in, h_direct, loads)
    channel_records = {}
    states = {
        "all_off": np.zeros(side * side, dtype=np.int8),
        "all_on": np.ones(side * side, dtype=np.int8),
        "alternating": np.arange(side * side, dtype=np.int8) % 2,
    }
    for label, bits in states.items():
        h_direct_nec = _direct_sparameter(z_full, bits, loads)
        h_hybrid = complex(hybrid.channel(bits)[0, 0])
        channel_records[label] = {
            "direct_nec": [h_direct_nec.real, h_direct_nec.imag],
            "spherical_hybrid": [h_hybrid.real, h_hybrid.imag],
            "relative_error": float(abs(h_hybrid - h_direct_nec) / abs(h_direct_nec)),
        }

    diag = diagnostics(response.z_ris)
    return {
        "scope": "3x3 thin-wire free-space microproof; descriptive only, not full P1/P2 gate",
        "solver": "direct PyNEC 2.3.4 EX1 plane-wave and complete 11-port MoM impedance solve",
        "geometry": {
            "ris_elements": side * side,
            "bs_ports": 1,
            "ue_ports": 1,
            "wire_length_m": wire_length,
            "wire_radius_m": 0.00005,
            "segments_per_wire": segments_per_wire,
            "extended_thin_wire_kernel_enabled": True,
            "segment_length_to_radius": wire_length / segments_per_wire / 0.00005,
            "segment_length_to_wavelength": (wire_length / segments_per_wire) / (C0 / frequency),
            "frequency_hz": frequency,
            "load_states_ohm": [[float(x.real), float(x.imag)] for x in loads],
        },
        "p1_direct_nec_port_consistency": {
            "plane_wave_angles_deg": {"theta_arrival": 70.0, "phi_arrival": 30.0},
            "short_circuit_current_relative_error": short_relative_error,
            "open_circuit_voltage_helper_vs_minus_Z_i_sc_relative_error": open_voltage_relative_error,
            "loaded_current_relative_errors": loaded_current_errors,
            "port_matrix_diagnostics": diag,
            "conclusion": "small-array convention probe only; not a 64-port gate pass",
        },
        "p2_direct_nec_spherical_comparison": {
            "full_nec_impedance_extraction": timing,
            "states": channel_records,
            "maximum_relative_channel_error": max(r["relative_error"] for r in channel_records.values()),
            "conclusion": "measured approximation error; no pass threshold applied",
        },
        "excluded": ["urban ray paths", "OpenSourceRIS calibration", "optimizer-selected actions", "test data"],
    }
