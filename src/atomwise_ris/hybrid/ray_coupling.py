"""Coupling ray-traced multipath parameters and dipole Green's functions to hybrid operators.

Constructs C_in, C_out, and H_direct for:
1. Spherical-wave Green's function dipole coupling (free-space benchmark)
2. Far-field plane-wave angular-spectrum coupling (multipath ray tracing)
"""
from typing import Optional, Tuple, Union
import numpy as np
from atomwise_ris.hybrid.embedded import (
    C0, ETA0, zdip_E, EmbeddedRISResponse
)

def build_spherical_dipole_operators(
    ris_response: EmbeddedRISResponse,
    bs_centers: np.ndarray,
    ue_centers: np.ndarray,
    wire_length: float = 0.024,
    frequency: float = 5.53e9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact spherical-wave dipole hybrid operators C_in, C_out, H_direct, A.
    
    Evaluates mutual impedance between external dipoles and RIS segments using
    the exact Hertzian Green's function and thin-wire Lorentz reciprocity.
    
    Parameters
    ----------
    ris_response : EmbeddedRISResponse
        Pre-extracted RIS impedance and segment response.
    bs_centers : np.ndarray
        (M, 3) BS antenna coordinates.
    ue_centers : np.ndarray
        (K, 3) UE antenna coordinates.
    wire_length : float
        External dipole wire length in meters (default 24 mm).
    frequency : float
        Operating frequency in Hz.
        
    Returns
    -------
    c_in : np.ndarray, shape (N, M)
    c_out : np.ndarray, shape (K, N)
    h_direct : np.ndarray, shape (K, M)
    a_matrix : np.ndarray, shape (N, N)
    """
    bs_c = np.asarray(bs_centers, dtype=float)
    ue_c = np.asarray(ue_centers, dtype=float)
    n_bs = len(bs_c)
    n_ue = len(ue_c)
    n_ris = ris_response.n_elements
    
    # Segment distributions for isolated external dipoles
    # For small dipoles, sinusoidal or triangular segment distribution
    # Here we use the center segment of ris_response to get standard segment discretization
    # Or isolate a single wire with the same segment count
    s_per_wire = len(ris_response.segment_lengths) // n_ris
    dl_ext = ris_response.segment_lengths[:s_per_wire]
    # Relative z offsets of segments along wire
    z_rel = ris_response.segment_centers[:s_per_wire, 2] - ris_response.wire_centers[0, 2]
    
    # Isolated external wire current distribution
    # Single wire self impedance and current per unit port current:
    # We take wire 0 from iz / iy as a close approximation, or extract isolated dipole
    iz_ext = ris_response.iz[:s_per_wire, 0]
    # Normalize so mid-segment current is 1.0 (unit port current)
    mid_idx = s_per_wire // 2
    if abs(iz_ext[mid_idx]) > 0:
        iz_ext = iz_ext / iz_ext[mid_idx]
    z_ext_self = ris_response.z_ris[0, 0]
    
    # 1. Mutual impedance Z_ris_bs: shape (n_ris, n_bs)
    z_ris_bs = np.zeros((n_ris, n_bs), dtype=complex)
    sc_r = ris_response.segment_centers
    dl_r = ris_response.segment_lengths
    iz_r = ris_response.iz
    
    for m in range(n_bs):
        # Segments of BS antenna m
        sc_bs_m = bs_c[m:m+1] + np.column_stack([np.zeros(s_per_wire), np.zeros(s_per_wire), z_rel])
        # Field at RIS segments from BS antenna m
        e_at_ris = np.zeros(len(sc_r), dtype=complex)
        for seg_idx in range(s_per_wire):
            dE = zdip_E(sc_r, sc_bs_m[seg_idx], frequency=frequency)[:, 2]
            e_at_ris += dE * dl_ext[seg_idx] * iz_ext[seg_idx]
        # Reciprocity integral: v_oc = - sum (iz_r * dl_r * e_at_ris)
        z_ris_bs[:, m] = - (iz_r * dl_r[:, None]).T @ e_at_ris

    # 2. Mutual impedance Z_ue_ris: shape (n_ue, n_ris)
    z_ue_ris = np.zeros((n_ue, n_ris), dtype=complex)
    for k in range(n_ue):
        sc_ue_k = ue_c[k:k+1] + np.column_stack([np.zeros(s_per_wire), np.zeros(s_per_wire), z_rel])
        e_at_ris = np.zeros(len(sc_r), dtype=complex)
        for seg_idx in range(s_per_wire):
            dE = zdip_E(sc_r, sc_ue_k[seg_idx], frequency=frequency)[:, 2]
            e_at_ris += dE * dl_ext[seg_idx] * iz_ext[seg_idx]
        z_ue_ris[k, :] = - (iz_r * dl_r[:, None]).T @ e_at_ris

    # 3. Direct mutual impedance Z_ue_bs: shape (n_ue, n_bs)
    z_ue_bs = np.zeros((n_ue, n_bs), dtype=complex)
    for k in range(n_ue):
        sc_ue_k = ue_c[k:k+1] + np.column_stack([np.zeros(s_per_wire), np.zeros(s_per_wire), z_rel])
        for m in range(n_bs):
            sc_bs_m = bs_c[m:m+1] + np.column_stack([np.zeros(s_per_wire), np.zeros(s_per_wire), z_rel])
            e_at_ue = np.zeros(s_per_wire, dtype=complex)
            for seg_idx in range(s_per_wire):
                dE = zdip_E(sc_ue_k, sc_bs_m[seg_idx], frequency=frequency)[:, 2]
                e_at_ue += dE * dl_ext[seg_idx] * iz_ext[seg_idx]
            z_ue_bs[k, m] = - np.sum(iz_ext * dl_ext * e_at_ue)

    # External impedance block termination (Z_EE + 50 I)
    # External ports: first n_bs are BS, next n_ue are UE
    n_ext = n_bs + n_ue
    z_ee = np.diag(np.full(n_ext, z_ext_self))
    z_ee[n_bs:, :n_bs] = z_ue_bs
    z_ee[:n_bs, n_bs:] = z_ue_bs.T
    
    e_block = z_ee + 50.0 * np.eye(n_ext)
    ei_block = np.linalg.inv(e_block)
    
    # Combined Z_RE: shape (n_ris, n_ext)
    z_re = np.hstack([z_ris_bs, z_ue_ris.T])
    z_er = z_re.T
    
    # Exact Schur complement terms
    a_matrix = ris_response.z_ris - z_re @ ei_block @ z_er
    c_out = - 100.0 * (ei_block @ z_er)[n_bs:]
    c_in = (z_re @ ei_block)[:, :n_bs]
    h_direct = - 100.0 * ei_block[n_bs:, :n_bs]
    
    return c_in, c_out, h_direct, a_matrix


def build_multipath_operators(
    ris_response: EmbeddedRISResponse,
    bs_paths: dict,
    ue_paths: dict,
    h_direct_traced: np.ndarray,
    frequency: float = 5.53e9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build hybrid operators C_in, C_out from ray-traced multipath parameters.
    
    Parameters
    ----------
    ris_response : EmbeddedRISResponse
        Pre-extracted RIS impedance and segment response.
    bs_paths : dict
        Dict with keys 'power', 'phase', 'delay', 'aoa_az', 'aoa_el' for BS-to-RIS links.
        Each array has shape (M, P).
    ue_paths : dict
        Dict with keys 'power', 'phase', 'delay', 'aod_az', 'aod_el' for RIS-to-UE links.
        Each array has shape (K, Q).
    h_direct_traced : np.ndarray
        (K, M) complex direct channel from ray tracer.
    frequency : float
        Frequency in Hz.
        
    Returns
    -------
    c_in : np.ndarray, shape (N, M)
    c_out : np.ndarray, shape (K, N)
    h_direct : np.ndarray, shape (K, M)
    """
    n_elements = ris_response.n_elements
    # Calibration factor between electric field / voltage and 50-ohm power wave
    # eta0 = 376.73 ohm, Z0 = 50 ohm
    scale_factor = 1.0 / np.sqrt(50.0 / ETA0)
    
    # 1. Build C_in: shape (N, M)
    bs_power = np.asarray(bs_paths['power']) # (M, P) in dB
    bs_phase = np.asarray(bs_paths['phase']) # (M, P) in deg
    bs_delay = np.asarray(bs_paths['delay']) # (M, P) in s
    bs_aoa_az = np.asarray(bs_paths['aoa_az']) # (M, P) in deg
    bs_aoa_el = np.asarray(bs_paths['aoa_el']) # (M, P) in deg
    
    m_sources = bs_power.shape[0]
    c_in = np.zeros((n_elements, m_sources), dtype=complex)
    
    for m in range(m_sources):
        for p in range(bs_power.shape[1]):
            pw = bs_power[m, p]
            if np.isnan(pw) or pw < -200:
                continue
            amp = 10.0 ** (pw / 20.0)
            tau = bs_delay[m, p]
            ph = np.radians(bs_phase[m, p]) - 2.0 * np.pi * frequency * tau
            alpha_p = amp * np.exp(1j * ph)
            
            # Zenith and azimuth in radians
            th = np.radians(bs_aoa_el[m, p])
            az = np.radians(bs_aoa_az[m, p])
            
            # Embedded effective length for all N elements
            h_theta = ris_response.effective_length(th, az) # (N,)
            c_in[:, m] += alpha_p * h_theta * scale_factor

    # 2. Build C_out: shape (K, N)
    ue_power = np.asarray(ue_paths['power']) # (K, Q) in dB
    ue_phase = np.asarray(ue_paths['phase']) # (K, Q) in deg
    ue_delay = np.asarray(ue_paths['delay']) # (K, Q) in s
    ue_az = np.asarray(ue_paths.get('aoa_az', ue_paths.get('aod_az'))) # (K, Q)
    ue_el = np.asarray(ue_paths.get('aoa_el', ue_paths.get('aod_el'))) # (K, Q)
    
    k_users = ue_power.shape[0]
    c_out = np.zeros((k_users, n_elements), dtype=complex)
    
    for k in range(k_users):
        for q in range(ue_power.shape[1]):
            pw = ue_power[k, q]
            if np.isnan(pw) or pw < -200:
                continue
            amp = 10.0 ** (pw / 20.0)
            tau = ue_delay[k, q]
            ph = np.radians(ue_phase[k, q]) - 2.0 * np.pi * frequency * tau
            beta_q = amp * np.exp(1j * ph)
            
            th = np.radians(ue_el[k, q])
            az = np.radians(ue_az[k, q])
            
            # Embedded effective length
            h_theta = ris_response.effective_length(th, az) # (N,)
            # Reciprocity: C_out has scaling -100 * (1/100) * scale = -scale or +scale depending on convention
            c_out[k, :] += beta_q * h_theta * scale_factor

    h_direct = np.asarray(h_direct_traced, dtype=complex).copy()
    return c_in, c_out, h_direct


# ---------------------------------------------------------------------------
# Spherical free-space hybrid with an NEC-derived external block and the passive
# (open-gap) scattering of the RIS wires. Derivation: docs/hybrid-gates-v2-20261107.md,
# section "P2 rerun after model fix". The legacy builder above remains the default
# dipole path; nothing here uses the 70-port truth network.
# ---------------------------------------------------------------------------

def _field_z(obs: np.ndarray, src: np.ndarray, moments: np.ndarray, frequency: float) -> np.ndarray:
    """Ez at ``obs`` points from z-directed Hertzian elements at ``src`` with moments dl*I."""
    out = np.zeros(len(obs), dtype=complex)
    for s in range(len(src)):
        out += zdip_E(obs, src[s], frequency=frequency)[:, 2] * moments[s]
    return out


def external_field_vectors(ris_response: EmbeddedRISResponse, ext_response: EmbeddedRISResponse,
                           frequency: float = 5.53e9) -> np.ndarray:
    """(M, K) tangential field at the RIS segment centres from unit current on each external antenna.

    Each external wire uses its own isolated, mid-normalised segment current shape from the
    external-antennas-alone NEC solve.
    """
    k = ext_response.n_elements
    s = len(ext_response.segment_lengths) // k
    cols = []
    for p in range(k):
        blk = slice(p * s, (p + 1) * s)
        shape = ext_response.iz[blk, p].copy()
        mid = s // 2
        if abs(shape[mid]) > 0:
            shape = shape / shape[mid]
        cols.append(_field_z(ris_response.segment_centers, ext_response.segment_centers[blk],
                             ext_response.segment_lengths[blk] * shape, frequency))
    return np.column_stack(cols)


def impressed_segment_currents(wire_centers: np.ndarray, voltages: np.ndarray, wire_length: float = 0.024,
                               segments_per_wire: int = 81, radius: float = 0.00005,
                               frequency: float = 5.53e9) -> np.ndarray:
    """NEC currents (M, K) of the continuous-wire structure for K sets of simultaneous segment voltages.

    ``voltages`` is (M, K), M = n_wires * segments_per_wire, ordered wire by wire. Column k is applied
    as EX voltage sources on every segment at once (NEC superposes them). With voltage ``dl * E`` per
    segment this is the induced current for an incident field E sampled at the segment centres
    (the convention checked against NEC plane waves in gate P1.4).
    """
    from atomwise_ris.physics.nec import WireGeometry

    centers = np.asarray(wire_centers, dtype=float)
    n = len(centers)
    s = int(segments_per_wire)
    v = np.asarray(voltages, dtype=complex)
    if v.shape[0] != n * s:
        raise ValueError("voltages must have n_wires * segments_per_wire rows")
    nec = WireGeometry(centers, np.full(n, wire_length), radius=radius, frequency=frequency).context(s)
    cols = []
    for k in range(v.shape[1]):
        for q in range(n):
            for j in range(s):
                x = v[q * s + j, k]
                nec.ex_card(0, q + 1, j + 1, 0, float(x.real), float(x.imag), 0, 0, 0, 0)
        nec.xq_card(0)
        cols.append(np.asarray(nec.get_structure_currents(k).get_current()).copy())
    return np.column_stack(cols)


def open_gap_scatter_impedance(ris_response: EmbeddedRISResponse, field_vectors: np.ndarray,
                               continuous_currents: np.ndarray) -> np.ndarray:
    """K x K change of the external impedance block when the RIS ports are open (structural scattering).

    ``continuous_currents[:, p]`` = I_sc,p = S u_p from :func:`impressed_segment_currents` with
    u_p = dl * E_p. Returns dZ = -U^T (I_sc - I_Z i_sc), i_sc = I_Y^T U (reciprocity-symmetrised).
    """
    u = ris_response.segment_lengths[:, None] * np.asarray(field_vectors, dtype=complex)
    i_sc = ris_response.iy.T @ u
    i_oc = np.asarray(continuous_currents, dtype=complex) - ris_response.iz @ i_sc
    dz = -u.T @ i_oc
    return 0.5 * (dz + dz.T)


def _schur_operators(z_ris: np.ndarray, z_re: np.ndarray, z_ee: np.ndarray, n_bs: int):
    n_ext = len(z_ee)
    ei = np.linalg.inv(z_ee + 50.0 * np.eye(n_ext))
    z_er = z_re.T
    a = z_ris - z_re @ ei @ z_er
    c_out = -100.0 * (ei @ z_er)[n_bs:]
    c_in = (z_re @ ei)[:, :n_bs]
    h_direct = -100.0 * ei[n_bs:, :n_bs]
    return c_in, c_out, h_direct, a


def build_spherical_nec_external_operators(
    ris_response: EmbeddedRISResponse,
    ext_response: EmbeddedRISResponse,
    n_bs: int,
    continuous_currents: Optional[np.ndarray] = None,
    frequency: float = 5.53e9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Hybrid operators with the external block from NEC of the external antennas alone.

    Parameters
    ----------
    ris_response : RIS alone (64 wires), as in the legacy builder.
    ext_response : the BS + UE wires alone (BS first), from ``extract_ris_response``.
    n_bs : number of BS ports.
    continuous_currents : optional (M, K) result of :func:`impressed_segment_currents` for the fields
        of :func:`external_field_vectors`. If given, the passive scattering of the open-gap RIS
        wires is added to the external block, so ``h_direct`` is the channel with the RIS present
        and its ports open. If None, only the NEC external block is used.

    Returns
    -------
    c_in, c_out, h_direct, a_matrix, info
    """
    e = external_field_vectors(ris_response, ext_response, frequency)
    weight = (ris_response.iz * ris_response.segment_lengths[:, None]).T
    z_re = -weight @ e
    z_ee = np.asarray(ext_response.z_ris, dtype=complex).copy()
    info: dict = {"open_gap_scatter_included": continuous_currents is not None}
    if continuous_currents is not None:
        dz = open_gap_scatter_impedance(ris_response, e, continuous_currents)
        z_ee = z_ee + dz
        info["delta_z_open_gap"] = dz
    c_in, c_out, h_direct, a = _schur_operators(ris_response.z_ris, z_re, z_ee, n_bs)
    info["z_re"] = z_re
    info["z_ee"] = z_ee
    return c_in, c_out, h_direct, a, info
