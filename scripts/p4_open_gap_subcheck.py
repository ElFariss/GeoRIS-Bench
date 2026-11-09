"""Predict Munich first-order open-gap scattering from RIS-only NEC inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from atomwise_ris.hybrid.embedded import C0, ETA0, extract_ris_response
from atomwise_ris.hybrid.p4_vector_mapping import (
    current_to_free_space_field,
    direct_channel_with_open_gap,
    mutual_impedance_from_isolated_currents,
)
from atomwise_ris.hybrid.ray_coupling import (
    impressed_segment_currents,
    open_gap_scatter_impedance,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def phase_error(a, b):
    return float(np.max(np.abs(np.angle(a * np.conjugate(b)))))


def amplitude_ratio(a, b):
    return float(np.linalg.norm(a) / np.linalg.norm(b))


def run(reference: Path, segments: int = 81):
    # Geometry-only inputs are read first. Truth matrices remain unopened until
    # every prediction below has been constructed.
    with np.load(reference) as data:
        ris_centers = np.asarray(data["ris_centers_xyz_m"], dtype=float)
        ext_centers = np.vstack((data["bs_centers_xyz_m"], data["ue_centers_xyz_m"])).astype(float)
    if ris_centers.shape != (64, 3) or ext_centers.shape != (2, 3):
        raise ValueError("expected the registered 64-element RIS and BS0/UE0 geometry")

    f = 5.53e9
    length = 0.024
    radius = 0.00005
    # The pinned PyNEC source defines eps and mu slightly differently from SI.
    # NEC's segment getters are wavelength-normalized. Correct their lengths
    # and centres back to the SI geometry before applying a Green kernel.
    eps_nec = 8.854e-12
    mu_nec = 4.0 * np.pi * 1e-7
    c_nec = 1.0 / np.sqrt(mu_nec * eps_nec)
    eta_nec = np.sqrt(mu_nec / eps_nec)
    nec_to_si_coord = c_nec / C0
    ris_raw = extract_ris_response(ris_centers, wire_length=length,
                                  segments_per_wire=segments, radius=radius, frequency=f)
    ris = replace(ris_raw,
                  segment_centers=ris_raw.segment_centers * nec_to_si_coord,
                  segment_lengths=ris_raw.segment_lengths * nec_to_si_coord,
                  wire_centers=ris_raw.wire_centers * nec_to_si_coord)
    # Independent one-wire port-current bases generate the vector incident
    # fields at each RIS segment. No two-port/truth reference enters this step.
    field_cols = []
    for p in range(2):
        ext_raw = extract_ris_response(ext_centers[p:p + 1], wire_length=length,
                                       segments_per_wire=segments, radius=radius, frequency=f)
        ext_one = replace(ext_raw,
                          segment_centers=ext_raw.segment_centers * nec_to_si_coord,
                          segment_lengths=ext_raw.segment_lengths * nec_to_si_coord,
                          wire_centers=ext_raw.wire_centers * nec_to_si_coord)
        tangent = np.tile([0.0, 0.0, 1.0], (len(ext_one.segment_lengths), 1))
        e_xyz = current_to_free_space_field(
            ext_one.iz[:, 0], ext_one.segment_centers, tangent,
            ext_one.segment_lengths, ris.segment_centers, f,
            speed_of_light_m_s=c_nec, intrinsic_impedance_ohm=eta_nec)
        field_cols.append(e_xyz)
    fields_xyz = np.stack(field_cols, axis=2)  # (RIS segments, XYZ, ext port)
    ris_tangents = np.tile([0.0, 0.0, 1.0], (len(ris.segment_lengths), 1))
    e_tangent = np.einsum("mi,mik->mk", ris_tangents, fields_xyz)
    u = ris.segment_lengths[:, None] * e_tangent
    continuous_currents = impressed_segment_currents(
        ris_centers, u, wire_length=length, segments_per_wire=segments,
        radius=radius, frequency=f)
    delta_z_pred = open_gap_scatter_impedance(ris, e_tangent, continuous_currents)

    # Calculate the unperturbed external port block from an external-only NEC
    # solve. This is the direct free-space baseline, not the monolithic RIS case.
    ext_pair = extract_ris_response(ext_centers, wire_length=length,
                                   segments_per_wire=segments, radius=radius, frequency=f)
    z_ee_free = np.asarray(ext_pair.z_ris, dtype=np.complex128)
    h_free, h_corrected = direct_channel_with_open_gap(z_ee_free, delta_z_pred, n_bs=1)

    # Only now read the separate monolithic run for scoring.
    # A convention-only direct-link cross-check: the same physical coordinate
    # map is evaluated under NEC constants and exact SI constants, with no fit.
    tx_raw = extract_ris_response(ext_centers[:1], wire_length=length,
                                 segments_per_wire=segments, radius=radius, frequency=f)
    rx_raw = extract_ris_response(ext_centers[1:], wire_length=length,
                                 segments_per_wire=segments, radius=radius, frequency=f)
    tx_phys = replace(tx_raw,
                      segment_centers=tx_raw.segment_centers * nec_to_si_coord,
                      segment_lengths=tx_raw.segment_lengths * nec_to_si_coord,
                      wire_centers=tx_raw.wire_centers * nec_to_si_coord)
    rx_phys = replace(rx_raw,
                      segment_centers=rx_raw.segment_centers * nec_to_si_coord,
                      segment_lengths=rx_raw.segment_lengths * nec_to_si_coord,
                      wire_centers=rx_raw.wire_centers * nec_to_si_coord)
    tx_tangent = np.tile([0.0, 0.0, 1.0], (len(tx_phys.segment_lengths), 1))
    rx_tangent = np.tile([0.0, 0.0, 1.0], (len(rx_phys.segment_lengths), 1))
    z_mutual_si = mutual_impedance_from_isolated_currents(
        tx_phys, rx_phys, tx_tangent, rx_tangent, f,
        speed_of_light_m_s=C0, intrinsic_impedance_ohm=ETA0)
    z_mutual_nec = mutual_impedance_from_isolated_currents(
        tx_phys, rx_phys, tx_tangent, rx_tangent, f,
        speed_of_light_m_s=c_nec, intrinsic_impedance_ohm=eta_nec)

    with np.load(reference) as data:
        z_open_ref = np.asarray(data["z_ee_monolithic_open"], dtype=np.complex128)
        z_free_ref = np.asarray(data["z_ee_without_ris"], dtype=np.complex128)
        h_open_ref = np.asarray(data["h_direct_monolithic_open"], dtype=np.complex128)
        h_free_ref = np.asarray(data["h_direct_without_ris"], dtype=np.complex128)
    delta_z_ref = z_open_ref - z_free_ref
    h_change_pred = h_corrected - h_free
    h_change_ref = h_open_ref - h_free_ref
    metrics = {
        "delta_z_complex_relative_error": relative_error(delta_z_pred, delta_z_ref),
        "delta_z_wrapped_phase_error_rad_max": phase_error(delta_z_pred, delta_z_ref),
        "delta_z_copolar_amplitude_ratio": amplitude_ratio(delta_z_pred, delta_z_ref),
        "direct_h_free_complex_relative_error": relative_error(h_free, h_free_ref),
        "direct_h_free_wrapped_phase_error_rad": phase_error(h_free, h_free_ref),
        "direct_h_free_copolar_amplitude_ratio": amplitude_ratio(h_free, h_free_ref),
        "direct_h_corrected_complex_relative_error": relative_error(h_corrected, h_open_ref),
        "direct_h_corrected_wrapped_phase_error_rad": phase_error(h_corrected, h_open_ref),
        "direct_h_corrected_copolar_amplitude_ratio": amplitude_ratio(h_corrected, h_open_ref),
        "direct_h_change_complex_relative_error": relative_error(h_change_pred, h_change_ref),
        "direct_h_change_wrapped_phase_error_rad": phase_error(h_change_pred, h_change_ref),
        "direct_h_change_copolar_amplitude_ratio": amplitude_ratio(h_change_pred, h_change_ref),
        "declared_complex_relative_error_limit": 0.01,
        "declared_wrapped_phase_error_limit_rad": 0.01,
        "declared_copolar_amplitude_ratio_limits": [0.99, 1.01],
    }
    return {
        "status": "measured_subcheck",
        "gate": "P4_NOT_EVALUATED",
        "scope": "geometry-only free-space LOS, RIS-alone predicted open-gap dZ; independent monolithic NEC target; not a ray-traced P4 result",
        "prediction": {
            "method": "isolated external NEC current bases -> Cartesian free-space dyadic fields at physical RIS segment centres using source-derived NEC c and eta -> projection on RIS wire tangents -> RIS-only impressed-segment NEC solve -> open_gap_scatter_impedance",
            "coordinate_conversion": "PyNEC normalized wavelength coordinates and lengths multiplied by c_NEC/C0 to restore the requested SI geometry before field/port integration",
            "nec_constants": {"epsilon_f_per_m": eps_nec, "mu_h_per_m": mu_nec,
                               "c_m_per_s": c_nec, "eta_ohm": eta_nec,
                               "coordinate_ratio_c_nec_over_c0": nec_to_si_coord},
            "delta_z_predicted": [[[float(x.real), float(x.imag)] for x in row] for row in delta_z_pred],
            "z_ee_free": [[[float(x.real), float(x.imag)] for x in row] for row in z_ee_free],
            "delta_z_used_monolithic_truth": False,
            "channel_correction": "H=-100*inv(Z_EE+50I)[UE,BS], with Z_EE=Z_EE_free+delta_z_pred",
        },
        "target": {
            "source": "independent monolithic NEC open/free two-port blocks, accessed after prediction construction",
            "reference_path": str(reference),
            "reference_sha256": sha256(reference),
        },
        "frequency_hz": f,
        "elements": len(ris_centers),
        "segments_per_wire": segments,
        "metrics": metrics,
        "source_convention_direct_link_check": {
            "reference_z_mutual_ohm": [float(z_free_ref[1, 0].real), float(z_free_ref[1, 0].imag)],
            "physical_geometry_exact_si_kernel_z_mutual_ohm": [float(z_mutual_si.real), float(z_mutual_si.imag)],
            "physical_geometry_nec_kernel_z_mutual_ohm": [float(z_mutual_nec.real), float(z_mutual_nec.imag)],
            "exact_si_vs_nec_reference_complex_relative_error": float(abs(z_mutual_si - z_free_ref[1, 0]) / abs(z_free_ref[1, 0])),
            "exact_si_vs_nec_reference_phase_error_rad": float(abs(np.angle(z_mutual_si * np.conjugate(z_free_ref[1, 0])))),
            "nec_kernel_vs_nec_reference_complex_relative_error": float(abs(z_mutual_nec - z_free_ref[1, 0]) / abs(z_free_ref[1, 0])),
            "nec_kernel_vs_nec_reference_phase_error_rad": float(abs(np.angle(z_mutual_nec * np.conjugate(z_free_ref[1, 0])))),
            "interpretation": "The SI kernel and NEC kernel are kept distinct; their source-derived difference is expected at this long range. Only the NEC kernel is compared to the NEC reference.",
        },
        "all_declared_scalar_checks_pass": all([
            metrics["delta_z_complex_relative_error"] <= 0.01,
            metrics["delta_z_wrapped_phase_error_rad_max"] <= 0.01,
            0.99 <= metrics["delta_z_copolar_amplitude_ratio"] <= 1.01,
            metrics["direct_h_corrected_complex_relative_error"] <= 0.01,
            metrics["direct_h_corrected_wrapped_phase_error_rad"] <= 0.01,
            0.99 <= metrics["direct_h_corrected_copolar_amplitude_ratio"] <= 1.01,
            metrics["direct_h_change_complex_relative_error"] <= 0.01,
            metrics["direct_h_change_wrapped_phase_error_rad"] <= 0.01,
            0.99 <= metrics["direct_h_change_copolar_amplitude_ratio"] <= 1.01,
        ]),
        "direct_h_change_predicted": [[[float(x.real), float(x.imag)] for x in row] for row in h_change_pred],
        "direct_h_change_reference": [[[float(x.real), float(x.imag)] for x in row] for row in h_change_ref],
        "not_scoreable": [
            "Sionna RT per-port incident vector fields and ray-traced direct channel are unavailable.",
            "No matching independent NEC reflection-case reference exists.",
            "No independent cross-polar vector-field target exists; the 1% cross-polar limit is not assessed.",
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
