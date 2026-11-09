"""No-fit vector field/port maps for the P4 thin-wire checks.

The phasor convention is exp(+j omega t), with outgoing Green functions
exp(-j k r). Lorentz reciprocity uses an unconjugated current-field product.
These helpers use global Cartesian vectors and support arbitrary segment axes.
They are a free-space NEC segment-current mapping, not a Sionna calibration.
"""
from __future__ import annotations

import numpy as np

from .embedded import C0, ETA0


def _segments(currents, centers, tangents, lengths):
    currents = np.asarray(currents, dtype=np.complex128)
    centers = np.asarray(centers, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    m = len(centers)
    if currents.ndim == 1:
        currents = currents[:, None]
    if (currents.ndim != 2 or currents.shape[0] != m or centers.shape != (m, 3)
            or tangents.shape != (m, 3) or lengths.shape != (m,)):
        raise ValueError("segment arrays must have matching (M,), (M,3), and (M,N) shapes")
    if not all(np.all(np.isfinite(a)) for a in (currents, centers, tangents, lengths)):
        raise ValueError("segment arrays must be finite")
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(lengths <= 0) or np.any(norms <= 0):
        raise ValueError("segment lengths and tangent norms must be positive")
    return currents, centers, tangents / norms[:, None], lengths


def field_to_open_circuit_voltage(currents_per_port_amp, centers_m, tangents_xyz,
                                  lengths_m, incident_e_xyz_v_m):
    """Map complex Cartesian incident E at segment centres to port Voc.

    V_oc[n] = -sum_m I_mn^(1 A port current) dl_m (t_m . E_m).
    No conjugation, projection onto z only, magnitude normalization, or phase
    alignment is applied. Shapes are E=(M,3), output=(N,).
    """
    currents, _, tangents, lengths = _segments(
        currents_per_port_amp, centers_m, tangents_xyz, lengths_m)
    e = np.asarray(incident_e_xyz_v_m, dtype=np.complex128)
    if e.shape != (len(lengths), 3) or not np.all(np.isfinite(e)):
        raise ValueError("incident E must be finite complex Cartesian (M,3)")
    projected = np.einsum("mi,mi->m", tangents, e)
    return -(currents * lengths[:, None]).T @ projected


def current_to_free_space_field(currents_a, centers_m, tangents_xyz, lengths_m,
                                observation_points_m, frequency_hz,
                                speed_of_light_m_s=C0, intrinsic_impedance_ohm=ETA0):
    """Radiate NEC segment currents using the exact free-space dyadic Green field.

    Each segment contributes its complex current moment I*dl*t. This includes
    1/r, 1/r^2, and 1/r^3 terms. It is exact for the Hertzian-segment integral
    approximation; finite-wire segment variation within a segment is not
    represented. Output shape is (P,3), for P observation points.
    """
    currents, centers, tangents, lengths = _segments(
        currents_a, centers_m, tangents_xyz, lengths_m)
    if currents.shape[1] != 1:
        raise ValueError("current_to_free_space_field accepts one current solution at a time")
    obs = np.asarray(observation_points_m, dtype=float)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim != 2 or obs.shape[1] != 3 or not np.all(np.isfinite(obs)):
        raise ValueError("observation points must have shape (P,3)")
    if (not np.isfinite(frequency_hz) or frequency_hz <= 0
            or not np.isfinite(speed_of_light_m_s) or speed_of_light_m_s <= 0
            or not np.isfinite(intrinsic_impedance_ohm) or intrinsic_impedance_ohm <= 0):
        raise ValueError("frequency, propagation speed, and intrinsic impedance must be positive and finite")

    k = 2.0 * np.pi * frequency_hz / speed_of_light_m_s
    out = np.zeros((len(obs), 3), dtype=np.complex128)
    for q, (center, tangent, length) in enumerate(zip(centers, tangents, lengths)):
        delta = obs - center
        r = np.linalg.norm(delta, axis=1)[:, None]
        if np.any(r <= 0):
            raise ValueError("observation point coincides with a source segment centre")
        rh = delta / r
        kr = k * r
        moment = currents[q, 0] * length
        a = (1j * k * intrinsic_impedance_ohm * moment / (4.0 * np.pi * r)
             * (1.0 + 1.0 / (1j * kr) - 1.0 / kr**2) * np.exp(-1j * kr))
        b = (intrinsic_impedance_ohm * moment / (2.0 * np.pi * r**2)
             * (1.0 + 1.0 / (1j * kr)) * np.exp(-1j * kr))
        radial_projection = np.einsum("pi,i->p", rh, tangent)[:, None]
        out += (a + b) * radial_projection * rh - a * tangent[None, :]
    return out


def mutual_impedance_from_isolated_currents(tx_response, rx_response,
                                            tx_tangents_xyz, rx_tangents_xyz,
                                            frequency_hz,
                                            speed_of_light_m_s=C0,
                                            intrinsic_impedance_ohm=ETA0):
    """First-order reciprocal Z_rx,tx from independently solved isolated ports."""
    tx_currents = np.asarray(tx_response.iz[:, 0], dtype=np.complex128)
    rx_currents = np.asarray(rx_response.iz[:, 0], dtype=np.complex128)
    e = current_to_free_space_field(
        tx_currents, tx_response.segment_centers, tx_tangents_xyz,
        tx_response.segment_lengths, rx_response.segment_centers, frequency_hz,
        speed_of_light_m_s, intrinsic_impedance_ohm)
    return complex(field_to_open_circuit_voltage(
        rx_currents, rx_response.segment_centers, rx_tangents_xyz,
        rx_response.segment_lengths, e)[0])


def direct_channel_with_open_gap(z_ee_free, delta_z_open_pred, n_bs,
                                 reference_impedance_ohm=50.0):
    """Apply a predicted RIS open-gap impedance correction to the direct block.

    Returns ``(H_free, H_corrected)`` using the declared power-wave coefficient
    convention ``-100 * inv(Z_EE + 50 I)[UE, BS]``. No field or channel fit is
    performed. The caller must keep target truth out of both input matrices.
    """
    z0 = np.asarray(z_ee_free, dtype=np.complex128)
    dz = np.asarray(delta_z_open_pred, dtype=np.complex128)
    if z0.ndim != 2 or z0.shape[0] != z0.shape[1] or dz.shape != z0.shape:
        raise ValueError("z_ee_free and predicted delta_z must be matching square matrices")
    if not isinstance(n_bs, (int, np.integer)) or n_bs <= 0 or n_bs >= len(z0):
        raise ValueError("n_bs must split nonempty BS and UE port blocks")
    if not np.all(np.isfinite(z0)) or not np.all(np.isfinite(dz)):
        raise ValueError("impedance matrices must be finite")
    if not np.isfinite(reference_impedance_ohm) or reference_impedance_ohm <= 0:
        raise ValueError("reference impedance must be positive and finite")
    ident = np.eye(len(z0), dtype=np.complex128) * reference_impedance_ohm
    try:
        h_free_all = -100.0 * np.linalg.solve(z0 + ident, np.eye(len(z0), dtype=np.complex128))
        h_corr_all = -100.0 * np.linalg.solve(z0 + dz + ident, np.eye(len(z0), dtype=np.complex128))
    except np.linalg.LinAlgError as exc:
        raise ValueError("terminated external impedance block is singular") from exc
    return (h_free_all[n_bs:, :n_bs], h_corr_all[n_bs:, :n_bs])
