"""Full-size (64 RIS + 4 BS + 2 UE = 70 port) direct-NEC validation of hybrid gates P1 and P2.

Criteria and tolerances are declared in ``docs/hybrid-gates-v2-20261107.md`` before any 64-wire run and
are encoded in ``THRESHOLDS_P1`` / ``THRESHOLDS_P2`` below. Do not loosen them after seeing results.

Independence: the reference side consists of NEC solves in which sources (EX) and terminations (LD) are
cards applied to the structure. The model side is the 70x70 port impedance matrix (unit-voltage solves,
inverted from the feed admittance), the Schur reduction of ``CoupledNetwork`` / ``HybridCoupledNetwork``,
Woodbury updates and the segment-current reciprocity integrals of ``EmbeddedRISResponse``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from atomwise_ris.hybrid.embedded import C0, EmbeddedRISResponse, zdip_E
from atomwise_ris.hybrid.operator import HybridCoupledNetwork
from atomwise_ris.physics.network import CoupledNetwork

LOADS = np.array([1.0 - 10.8j, 1.0 + 129.2j], dtype=complex)
N_BS = 4
N_UE = 2
WIRE_LENGTH = 0.024
RADIUS = 0.00005
FREQUENCY = 5.53e9
REFERENCE_OHM = 50.0

# Declared in docs/hybrid-gates-v2-20261107.md. Keys map to criterion ids.
THRESHOLDS_P1 = {
    "P1.0a_reciprocity_relative_max": 1.0e-6,
    "P1.0b_min_resistance_eigenvalue_ohm_gt": 0.0,
    "P1.0c_max_scattering_singular_value_max": 1.0001,
    "P1.0d_condition_number_lt": 1.0e5,
    "P1.1_loaded_current_vector_relative_max": 1.0e-6,
    "P1.2_reduced_channel_relative_max": 1.0e-6,
    "P1.3_woodbury_field_relative_max": 1.0e-6,
    "P1.3_woodbury_se_abs_max_bps_hz": 1.0e-4,
    "P1.4_plane_wave_short_circuit_relative_max": 1.0e-3,
    "P1.5_open_circuit_identity_relative_max": 1.0e-5,
    "P1.6_plane_wave_loaded_current_relative_max": 1.0e-3,
}
THRESHOLDS_P2 = {
    "P2.0_truth_state_channel_relative_max": 1.0e-6,
    "P2.1_channel_relative_error_max": 0.05,
    "P2.2_worst_user_se_abs_diff_max_bps_hz": 0.5,
    "P2.3_spearman_min": 0.95,
    "P2.4_regret_max_bps_hz": 0.5,
    "P2.5_truth_modulability_min": 0.01,
}
RANDOM_SEEDS = (20261107, 20261108, 20261109, 20261110, 20261111)
PLANE_WAVE_ANGLES_DEG = {"A1": (70.0, 30.0), "A2": (90.0, 180.0), "A3": (60.0, 200.0)}
LOADED_PLANE_WAVE_ANGLES = ("A1", "A2")


# --------------------------------------------------------------------------- generic numerics

def rel_err(test: np.ndarray, ref: np.ndarray) -> float:
    ref = np.asarray(ref)
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(np.asarray(test) - ref) / denom) if denom > 0 else float("inf")


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with tie handling."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def criterion(cid: str, value: float, tolerance: float, op: str) -> dict[str, Any]:
    """op in {'<=', '<', '>=', '>'}; NaN never passes."""
    value = float(value)
    if math.isnan(value):
        passed = False
    else:
        passed = {"<=": value <= tolerance, "<": value < tolerance,
                  ">=": value >= tolerance, ">": value > tolerance}[op]
    return {"id": cid, "value": value, "tolerance": float(tolerance), "op": op, "passed": bool(passed)}


def network_currents(z: np.ndarray, terminations: np.ndarray, excitation_ports: Sequence[int]) -> np.ndarray:
    """(n_ports, n_exc) port currents for 1 V series sources, from the port impedance matrix."""
    n = len(z)
    rhs = np.zeros((n, len(excitation_ports)), dtype=complex)
    for col, port in enumerate(excitation_ports):
        rhs[port, col] = 1.0
    return np.linalg.solve(z + np.diag(terminations), rhs)


def terminations_for(bits: np.ndarray, loads: np.ndarray = LOADS, n_ext: int = N_BS + N_UE) -> np.ndarray:
    return np.r_[loads[np.asarray(bits, dtype=int)], np.full(n_ext, REFERENCE_OHM, dtype=complex)]


def declared_states(n: int) -> dict[str, np.ndarray]:
    states: dict[str, np.ndarray] = {
        "all_off": np.zeros(n, dtype=np.int8),
        "all_on": np.ones(n, dtype=np.int8),
        "alternating": (np.arange(n) % 2).astype(np.int8),
    }
    for i, seed in enumerate(RANDOM_SEEDS):
        states[f"R{i + 1}"] = np.random.default_rng(seed).integers(0, 2, n).astype(np.int8)
    return states


def excitation_ports(n_ris: int, ris_ports: Sequence[int] | None = None) -> dict[str, list[int]]:
    if ris_ports is None:
        ris_ports = [0, 27, n_ris - 1] if n_ris >= 28 else [0, n_ris // 2, n_ris - 1]
    return {
        "bs": [n_ris + m for m in range(N_BS)],
        "ue": [n_ris + N_BS],
        "ris": [int(p) for p in ris_ports],
    }


# --------------------------------------------------------------------------- NEC drivers (top-level: picklable)

def _geometry(centers: np.ndarray, lengths: np.ndarray, radius: float, frequency: float):
    from atomwise_ris.physics.nec import WireGeometry
    return WireGeometry(np.asarray(centers, dtype=float), np.asarray(lengths, dtype=float),
                        radius=radius, frequency=frequency)


def task_impedance(centers, lengths, radius, frequency, segments):
    """70-port impedance via unit-voltage solves. Returns (Z, timing)."""
    return _geometry(centers, lengths, radius, frequency).impedance(segments)


def task_ris_response(centers, wire_length, radius, frequency, segments):
    from atomwise_ris.hybrid.embedded import extract_ris_response
    return extract_ris_response(np.asarray(centers, dtype=float), wire_length=wire_length,
                                segments_per_wire=segments, radius=radius, frequency=frequency)


def task_loaded_solve(centers, lengths, radius, frequency, segments, terminations, ports):
    """NEC loaded solve: LD cards on the terminations, one EX voltage source per port in ``ports``.

    Returns the (n_wires, len(ports)) feed-segment currents for 1 V sources.
    """
    g = _geometry(centers, lengths, radius, frequency)
    n_wires = len(centers)
    nec = g.context(segments, np.asarray(terminations, dtype=complex))
    mid = np.arange(n_wires) * segments + segments // 2
    cols = []
    for run, p in enumerate(ports):
        nec.ex_card(0, int(p) + 1, segments // 2 + 1, 0, 1, 0, 0, 0, 0, 0)
        nec.xq_card(0)
        res = nec.get_structure_currents(run)
        if abs(res.get_frequency() - frequency) > 1:
            raise RuntimeError("NEC frequency mismatch")
        cols.append(np.asarray(res.get_current())[mid].copy())
    return np.array(cols).T


def task_plane_wave(centers, lengths, radius, frequency, segments, terminations, angles_deg):
    """NEC plane-wave (EX type 1) feed-segment currents; ``terminations`` may be None (short circuit)."""
    g = _geometry(centers, lengths, radius, frequency)
    n_wires = len(centers)
    loads = None if terminations is None else np.asarray(terminations, dtype=complex)
    nec = g.context(segments, loads)
    mid = np.arange(n_wires) * segments + segments // 2
    cols = []
    for run, (theta, phi) in enumerate(angles_deg):
        nec.ex_card(1, 1, 1, 0, float(theta), float(phi), 0.0, 0, 0, 0)
        nec.xq_card(0)
        cols.append(np.asarray(nec.get_structure_currents(run).get_current())[mid].copy())
    return np.array(cols).T


# --------------------------------------------------------------------------- hybrid operator assembly

def _field_z(obs: np.ndarray, src: np.ndarray, weights: np.ndarray, frequency: float) -> np.ndarray:
    """Ez at ``obs`` points from z-dipole elements at ``src`` with moments ``weights`` (= dl * I)."""
    out = np.zeros(len(obs), dtype=complex)
    for s in range(len(src)):
        out += zdip_E(obs, src[s], frequency=frequency)[:, 2] * weights[s]
    return out


def spherical_blocks(resp: EmbeddedRISResponse, bs: np.ndarray, ue: np.ndarray,
                     frequency: float = FREQUENCY) -> dict[str, np.ndarray]:
    """Mutual impedance blocks used by ``build_spherical_dipole_operators`` (same formulas).

    Returns z_re (n_ris, n_ext), z_ee_mutual (n_ext, n_ext, zero diagonal; exact pairwise dipole
    mutual impedance including BS-BS and UE-UE) and the scalar external self impedance.
    """
    ext = np.vstack([np.asarray(bs, dtype=float), np.asarray(ue, dtype=float)])
    n_ris = resp.n_elements
    s_per = len(resp.segment_lengths) // n_ris
    dl_ext = resp.segment_lengths[:s_per]
    z_rel = resp.segment_centers[:s_per, 2] - resp.wire_centers[0, 2]
    iz_ext = resp.iz[:s_per, 0].copy()
    mid = s_per // 2
    if abs(iz_ext[mid]) > 0:
        iz_ext = iz_ext / iz_ext[mid]
    w_ext = dl_ext * iz_ext
    offs = np.column_stack([np.zeros(s_per), np.zeros(s_per), z_rel])
    ext_sc = [c[None, :] + offs for c in ext]
    weight_r = (resp.iz * resp.segment_lengths[:, None]).T  # (N, M)
    z_re = np.zeros((n_ris, len(ext)), dtype=complex)
    for p in range(len(ext)):
        z_re[:, p] = -weight_r @ _field_z(resp.segment_centers, ext_sc[p], w_ext, frequency)
    z_ee = np.zeros((len(ext), len(ext)), dtype=complex)
    for a in range(len(ext)):
        for b in range(len(ext)):
            if a != b:
                z_ee[a, b] = -np.sum(w_ext * _field_z(ext_sc[a], ext_sc[b], w_ext, frequency))
    return {"z_re": z_re, "z_ee_mutual": z_ee, "z_self": resp.z_ris[0, 0]}


def assemble_operators(z_ris: np.ndarray, z_re: np.ndarray, z_ee: np.ndarray,
                       n_bs: int = N_BS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Schur/termination algebra of ``build_spherical_dipole_operators``. Returns (c_in, c_out, h_direct, A)."""
    n_ext = len(z_ee)
    ei = np.linalg.inv(z_ee + REFERENCE_OHM * np.eye(n_ext))
    z_er = z_re.T
    a = z_ris - z_re @ ei @ z_er
    c_out = -2.0 * REFERENCE_OHM * (ei @ z_er)[n_bs:]
    c_in = (z_re @ ei)[:, :n_bs]
    h_direct = -2.0 * REFERENCE_OHM * ei[n_bs:, :n_bs]
    return c_in, c_out, h_direct, a


def z_ee_as_in_code_under_test(blocks: dict[str, np.ndarray], n_bs: int = N_BS) -> np.ndarray:
    """External block exactly as ``build_spherical_dipole_operators`` builds it (no BS-BS/UE-UE coupling)."""
    m = blocks["z_ee_mutual"]
    n_ext = len(m)
    z = np.diag(np.full(n_ext, blocks["z_self"], dtype=complex))
    z_ue_bs = m[n_bs:, :n_bs]
    z[n_bs:, :n_bs] = z_ue_bs
    z[:n_bs, n_bs:] = z_ue_bs.T
    return z


def z_ee_full_external(blocks: dict[str, np.ndarray]) -> np.ndarray:
    """Diagnostic variant: all pairwise external mutual impedances."""
    return blocks["z_ee_mutual"] + np.diag(np.full(len(blocks["z_ee_mutual"]), blocks["z_self"], dtype=complex))


# --------------------------------------------------------------------------- JSON helpers

def jc(x: Any) -> Any:
    """Make numpy/complex values JSON-serialisable."""
    if isinstance(x, dict):
        return {str(k): jc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jc(v) for v in x]
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {"re": x.real.tolist(), "im": x.imag.tolist()}
        return x.tolist()
    if isinstance(x, (complex, np.complexfloating)):
        return {"re": float(x.real), "im": float(x.imag)}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# --------------------------------------------------------------------------- evaluation (given NEC outputs)

def evaluate_p1(
    *, z_full: np.ndarray, n_ris: int, states: dict[str, np.ndarray],
    loaded: dict[str, np.ndarray], ports: dict[str, list[int]],
    woodbury: dict[str, dict[str, Any]], resp: EmbeddedRISResponse,
    pw_short: np.ndarray, pw_loaded: dict[tuple[str, str], np.ndarray],
    loads: np.ndarray = LOADS, thresholds: dict[str, float] = THRESHOLDS_P1,
) -> dict[str, Any]:
    """Compare model-side predictions with the NEC outputs and apply the declared criteria.

    ``loaded[state]`` has columns in the order bs ports, ue ports, ris ports.
    ``woodbury[name]`` = {'bits', 'parent', 'flip', 'field_woodbury', 'se_moves'} for F1/F2.
    ``pw_short``: (n_ris, n_angles) NEC short-circuit currents in PLANE_WAVE_ANGLES_DEG order.
    ``pw_loaded[(state, angle)]``: (n_ris,) NEC loaded plane-wave port currents.
    """
    from atomwise_ris.physics.nec import diagnostics

    exc = ports["bs"] + ports["ue"] + ports["ris"]
    n_bs_cols = len(ports["bs"])
    diag = diagnostics(z_full)
    crits = [
        criterion("P1.0a", diag["reciprocity_relative"], thresholds["P1.0a_reciprocity_relative_max"], "<="),
        criterion("P1.0b", diag["minimum_resistance_eigenvalue_ohm"], thresholds["P1.0b_min_resistance_eigenvalue_ohm_gt"], ">"),
        criterion("P1.0c", diag["maximum_scattering_singular_value"], thresholds["P1.0c_max_scattering_singular_value_max"], "<="),
        criterion("P1.0d", diag["condition_number"], thresholds["P1.0d_condition_number_lt"], "<"),
    ]

    net = CoupledNetwork(z_full, loads)
    hyb = HybridCoupledNetwork.from_monolithic_z(z_full, loads, n_ris)
    per_state: dict[str, Any] = {}
    cur_errors: list[float] = []
    chan_errors: list[float] = []
    for name, sol in loaded.items():
        bits = states[name]
        term = terminations_for(bits, loads)
        pred = network_currents(z_full, term, exc)
        e_cols = [rel_err(sol[:, c], pred[:, c]) for c in range(sol.shape[1])]
        h_nec = -2.0 * REFERENCE_OHM * sol[n_ris + N_BS:n_ris + N_BS + N_UE, :n_bs_cols]
        e_net = rel_err(net.channel(bits), h_nec)
        e_hyb = rel_err(hyb.channel(bits), h_nec)
        per_state[name] = {"bits": bits.tolist(), "current_vector_rel_err_by_excitation": dict(zip(map(str, exc), e_cols)),
                           "max_current_vector_rel_err": max(e_cols),
                           "channel_rel_err_coupled_network": e_net, "channel_rel_err_hybrid_from_monolithic": e_hyb}
        cur_errors.append(max(e_cols))
        chan_errors.extend([e_net, e_hyb])
    crits.append(criterion("P1.1", max(cur_errors), thresholds["P1.1_loaded_current_vector_relative_max"], "<="))
    crits.append(criterion("P1.2", max(chan_errors), thresholds["P1.2_reduced_channel_relative_max"], "<="))

    wood_out: dict[str, Any] = {}
    f_err, se_err = [], []
    for name, info in woodbury.items():
        h_nec = -2.0 * REFERENCE_OHM * loaded[name][n_ris + N_BS:n_ris + N_BS + N_UE, :n_bs_cols]
        field_nec = h_nec @ net.w
        e_field = rel_err(info["field_woodbury"], field_nec)
        rec: dict[str, Any] = {"parent": info["parent"], "flip": list(info["flip"]), "field_rel_err": e_field}
        f_err.append(e_field)
        if info.get("se_moves") is not None:
            se_nec = float(net.score(field_nec))
            rec["se_moves"] = float(info["se_moves"])
            rec["se_direct_nec"] = se_nec
            rec["se_abs_diff"] = abs(float(info["se_moves"]) - se_nec)
            se_err.append(rec["se_abs_diff"])
        wood_out[name] = rec
    crits.append(criterion("P1.3a", max(f_err), thresholds["P1.3_woodbury_field_relative_max"], "<="))
    crits.append(criterion("P1.3b", max(se_err) if se_err else float("nan"), thresholds["P1.3_woodbury_se_abs_max_bps_hz"], "<="))

    # P1.4-P1.6: plane-wave incidence on the RIS-alone structure (reciprocity vs NEC EX1).
    k = 2.0 * np.pi * resp.frequency / C0
    pw_records: dict[str, Any] = {}
    short_errors, ocv_errors = [], []
    for col, (aname, (th_deg, ph_deg)) in enumerate(PLANE_WAVE_ANGLES_DEG.items()):
        th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
        k_arr = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        e_z = -np.sin(th) * np.exp(1j * k * (resp.segment_centers @ k_arr))
        isc = resp.short_circuit_current_from_field(e_z)
        voc_helper = resp.open_circuit_voltage_from_field(e_z)
        voc_conv = resp.z_ris @ isc
        e_short = rel_err(isc, pw_short[:, col])
        e_ocv = float(np.linalg.norm(voc_helper + voc_conv) / np.linalg.norm(voc_conv))
        short_errors.append(e_short)
        ocv_errors.append(e_ocv)
        pw_records[aname] = {"theta_deg": th_deg, "phi_deg": ph_deg,
                             "short_circuit_rel_err": e_short, "open_circuit_identity_rel_err": e_ocv, "loaded": {}}
        for (sname, an), nec_cur in pw_loaded.items():
            if an != aname:
                continue
            zl = loads[np.asarray(states[sname], dtype=int)]
            pred = -np.linalg.solve(resp.z_ris + np.diag(zl), voc_helper)
            pw_records[aname]["loaded"][sname] = rel_err(pred, nec_cur)
    loaded_pw = [v for a in pw_records.values() for v in a["loaded"].values()]
    crits.append(criterion("P1.4", max(short_errors), thresholds["P1.4_plane_wave_short_circuit_relative_max"], "<="))
    crits.append(criterion("P1.5", max(ocv_errors), thresholds["P1.5_open_circuit_identity_relative_max"], "<="))
    crits.append(criterion("P1.6", max(loaded_pw) if loaded_pw else float("nan"),
                           thresholds["P1.6_plane_wave_loaded_current_relative_max"], "<="))
    return {
        "gate": "P1", "criteria": crits, "passed": all(c["passed"] for c in crits),
        "z_diagnostics": diag, "thresholds": thresholds,
        "per_state_network_vs_direct_nec": per_state, "woodbury": wood_out, "plane_wave": pw_records,
    }


def exhaustive_scores(net: Any, off_bits: np.ndarray) -> dict[str, Any]:
    """Single-flip and pair-flip worst-user SE from ``off_bits`` through the network's own Woodbury moves."""
    st = net.state(off_bits)
    singles = np.asarray(net.moves(st), dtype=float)
    pairs = np.asarray(net.moves(st, pairs=True), dtype=float)
    return {"singles": singles, "pairs": pairs, "pair_index": np.asarray(net.pairs)}


def select_best_states(truth: CoupledNetwork, n: int) -> dict[str, Any]:
    off = np.zeros(n, dtype=np.int8)
    scores = exhaustive_scores(truth, off)
    i = int(np.argmax(scores["singles"]))
    j = int(np.argmax(scores["pairs"]))
    pair = scores["pair_index"][j]
    best_single = off.copy()
    best_single[i] = 1
    best_pair = off.copy()
    best_pair[pair] = 1
    return {"single_index": i, "pair_indices": [int(pair[0]), int(pair[1])],
            "bits_single": best_single, "bits_pair": best_pair, "scores": scores}


def evaluate_p2(
    *, z_full: np.ndarray, resp: EmbeddedRISResponse, bs: np.ndarray, ue: np.ndarray,
    states: dict[str, np.ndarray], best: dict[str, Any], direct_best: dict[str, np.ndarray],
    n_ris: int, loads: np.ndarray = LOADS, frequency: float = FREQUENCY,
    thresholds: dict[str, float] = THRESHOLDS_P2,
) -> dict[str, Any]:
    """P2: factored spherical hybrid vs 70-port NEC network, plus attribution diagnostics.

    ``direct_best[name]``: (n_wires, 4) NEC loaded currents for BS1 / BP (BS-port sources).
    """
    from atomwise_ris.hybrid.ray_coupling import build_spherical_dipole_operators

    truth = CoupledNetwork(z_full, loads)
    c_in, c_out, h_dir, a = build_spherical_dipole_operators(resp, bs, ue, wire_length=WIRE_LENGTH, frequency=frequency)
    hyb = HybridCoupledNetwork(a, c_out, c_in, h_dir, loads)
    blocks = spherical_blocks(resp, bs, ue, frequency)
    # Cross-check that the diagnostic re-assembly reproduces the code under test.
    ops_chk = assemble_operators(resp.z_ris, blocks["z_re"], z_ee_as_in_code_under_test(blocks))
    reassembly_err = max(rel_err(ops_chk[0], c_in), rel_err(ops_chk[1], c_out), rel_err(ops_chk[2], h_dir), rel_err(ops_chk[3], a))
    c_in_f, c_out_f, h_dir_f, a_f = assemble_operators(resp.z_ris, blocks["z_re"], z_ee_full_external(blocks))
    hyb_full = HybridCoupledNetwork(a_f, c_out_f, c_in_f, h_dir_f, loads)

    all_states = dict(states)
    all_states["best_single"] = best["bits_single"]
    all_states["best_pair"] = best["bits_pair"]
    off = all_states["all_off"]
    h_off_t, h_off_h, h_off_f = truth.channel(off), hyb.channel(off), hyb_full.channel(off)

    def se(net: Any, h: np.ndarray, w: np.ndarray) -> float:
        return float(net.score(h @ w))

    per_state: dict[str, Any] = {}
    for name, bits in all_states.items():
        h_t, h_h, h_f = truth.channel(bits), hyb.channel(bits), hyb_full.channel(bits)
        rec = {
            "bits": bits.tolist(),
            "channel_rel_err": rel_err(h_h, h_t),
            "se_truth_own_w": se(truth, h_t, truth.w), "se_hybrid_own_w": se(hyb, h_h, hyb.w),
            "se_truth_common_w": se(truth, h_t, truth.w), "se_hybrid_common_w": se(hyb, h_h, truth.w),
            "modulation_fidelity_rel_err": (rel_err(h_h - h_off_h, h_t - h_off_t)
                                            if np.linalg.norm(h_t - h_off_t) > 0 else None),
            "diagnostic_full_external": {
                "channel_rel_err": rel_err(h_f, h_t),
                "se_hybrid_own_w": se(hyb_full, h_f, hyb_full.w),
                "modulation_fidelity_rel_err": (rel_err(h_f - h_off_f, h_t - h_off_t)
                                                if np.linalg.norm(h_t - h_off_t) > 0 else None),
            },
        }
        rec["se_abs_diff_own_w"] = abs(rec["se_hybrid_own_w"] - rec["se_truth_own_w"])
        rec["se_abs_diff_common_w"] = abs(rec["se_hybrid_common_w"] - rec["se_truth_common_w"])
        per_state[name] = rec

    # P2.0: truth network vs direct NEC on the searched states
    p20 = []
    for name, sol in direct_best.items():
        h_nec = -2.0 * REFERENCE_OHM * sol[n_ris + N_BS:n_ris + N_BS + N_UE, :N_BS]
        p20.append(rel_err(truth.channel(all_states[name]), h_nec))

    # rankings and regret
    hyb_scores = exhaustive_scores(hyb, off)
    hyb_full_scores = exhaustive_scores(hyb_full, off)
    t = best["scores"]
    rho_single = spearman(hyb_scores["singles"], t["singles"])
    rho_pair = spearman(hyb_scores["pairs"], t["pairs"])
    hs, hp = int(np.argmax(hyb_scores["singles"])), int(np.argmax(hyb_scores["pairs"]))
    regret_single = float(t["singles"].max() - t["singles"][hs])
    regret_pair = float(t["pairs"].max() - t["pairs"][hp])
    f_regret_single = float(t["singles"].max() - t["singles"][int(np.argmax(hyb_full_scores["singles"]))])
    f_regret_pair = float(t["pairs"].max() - t["pairs"][int(np.argmax(hyb_full_scores["pairs"]))])

    contrast = float(np.linalg.norm(truth.channel(all_states["all_on"]) - h_off_t) / np.linalg.norm(h_off_t))
    crits = [
        criterion("P2.0", max(p20) if p20 else float("nan"), thresholds["P2.0_truth_state_channel_relative_max"], "<="),
        criterion("P2.1", max(r["channel_rel_err"] for r in per_state.values()), thresholds["P2.1_channel_relative_error_max"], "<="),
        criterion("P2.2", max(r["se_abs_diff_own_w"] for r in per_state.values()), thresholds["P2.2_worst_user_se_abs_diff_max_bps_hz"], "<="),
        criterion("P2.3a", rho_single, thresholds["P2.3_spearman_min"], ">="),
        criterion("P2.3b", rho_pair, thresholds["P2.3_spearman_min"], ">="),
        criterion("P2.4a", regret_single, thresholds["P2.4_regret_max_bps_hz"], "<="),
        criterion("P2.4b", regret_pair, thresholds["P2.4_regret_max_bps_hz"], "<="),
        criterion("P2.5", contrast, thresholds["P2.5_truth_modulability_min"], ">="),
    ]

    # Variant (declared in the doc before any 64-wire run): the same P2.1-P2.4 limits applied to the hybrid
    # with the exact external block (BS-BS and UE-UE mutual coupling). Reported separately; it never
    # changes the status of the gate for the code as written.
    fe = {name: rec["diagnostic_full_external"] for name, rec in per_state.items()}
    var_se = {name: abs(fe[name]["se_hybrid_own_w"] - per_state[name]["se_truth_own_w"]) for name in fe}
    variant_crits = [
        criterion("P2v.1", max(r["channel_rel_err"] for r in fe.values()), thresholds["P2.1_channel_relative_error_max"], "<="),
        criterion("P2v.2", max(var_se.values()), thresholds["P2.2_worst_user_se_abs_diff_max_bps_hz"], "<="),
        criterion("P2v.3a", spearman(hyb_full_scores["singles"], t["singles"]), thresholds["P2.3_spearman_min"], ">="),
        criterion("P2v.3b", spearman(hyb_full_scores["pairs"], t["pairs"]), thresholds["P2.3_spearman_min"], ">="),
        criterion("P2v.4a", f_regret_single, thresholds["P2.4_regret_max_bps_hz"], "<="),
        criterion("P2v.4b", f_regret_pair, thresholds["P2.4_regret_max_bps_hz"], "<="),
    ]
    block_err = {
        "A": rel_err(hyb.a, truth.a), "C_in": rel_err(hyb.f, truth.f),
        "C_out": rel_err(hyb.c, truth.c), "H_direct": rel_err(hyb.d, truth.d),
        "diagnostic_full_external": {
            "A": rel_err(hyb_full.a, truth.a), "C_in": rel_err(hyb_full.f, truth.f),
            "C_out": rel_err(hyb_full.c, truth.c), "H_direct": rel_err(hyb_full.d, truth.d)},
    }
    z_re_truth = z_full[:n_ris, n_ris:]
    return {
        "gate": "P2", "criteria": crits, "passed": all(c["passed"] for c in crits), "thresholds": thresholds,
        "variant_full_external_coupling": {
            "description": "hybrid with exact BS-BS/UE-UE mutual coupling; NOT the code under test; same limits",
            "criteria": variant_crits, "passed_p2v": all(c["passed"] for c in variant_crits),
            "shared_preconditions_passed": bool(crits[0]["passed"] and crits[-1]["passed"]),
            "per_state_se_abs_diff_own_w": var_se},
        "precoder_protocol_primary": "each network uses its own RZF of channel(all-OFF)",
        "per_state": per_state,
        "best_states": {"single_index": best["single_index"], "pair_indices": best["pair_indices"],
                        "truth_se_best_single": float(t["singles"].max()), "truth_se_best_pair": float(t["pairs"].max())},
        "ranking": {"spearman_single": rho_single, "spearman_pair": rho_pair,
                    "hybrid_chosen_single": hs, "hybrid_chosen_pair": [int(x) for x in hyb_scores["pair_index"][hp]],
                    "regret_single": regret_single, "regret_pair": regret_pair,
                    "diagnostic_full_external": {"spearman_single": spearman(hyb_full_scores["singles"], t["singles"]),
                                                 "spearman_pair": spearman(hyb_full_scores["pairs"], t["pairs"]),
                                                 "regret_single": f_regret_single, "regret_pair": f_regret_pair}},
        "diagnostics_non_gating": {
            "operator_block_relative_errors": block_err,
            "z_re_block_relative_error_spherical_vs_nec": rel_err(blocks["z_re"], z_re_truth),
            "z_ee_mutual_rel_err_bs_bs": rel_err(blocks["z_ee_mutual"][:N_BS, :N_BS] ,
                                                 z_full[n_ris:n_ris + N_BS, n_ris:n_ris + N_BS]
                                                 - np.diag(np.diag(z_full[n_ris:n_ris + N_BS, n_ris:n_ris + N_BS]))),
            "z_self_external_used": blocks["z_self"],
            "z_self_external_nec_bs_mean": np.diag(z_full)[n_ris:n_ris + N_BS].mean(),
            "reassembly_vs_code_under_test_rel_err": reassembly_err,
            "common_precoder_se_max_abs_diff": max(r["se_abs_diff_common_w"] for r in per_state.values()),
            "modulation_fidelity_max": max((r["modulation_fidelity_rel_err"] or 0.0) for r in per_state.values()),
        },
    }


def external_only_diagnostic(
    *, z_full: np.ndarray, resp: EmbeddedRISResponse, bs: np.ndarray, ue: np.ndarray,
    segments: int, n_ris: int, frequency: float = FREQUENCY,
) -> dict[str, Any]:
    """Non-gating attribution of the direct-channel error (cheap: only the 6 external wires are solved).

    Separates (a) the error of the dipole Green-function model of the external block from (b) the
    scattering by the 64 open-gap RIS wires that the truth network always contains but the hybrid
    ``H_direct`` does not model.
    """
    ext = np.vstack([np.asarray(bs, dtype=float), np.asarray(ue, dtype=float)])
    z6, _ = task_impedance(ext, np.full(len(ext), WIRE_LENGTH), RADIUS, frequency, segments)
    n_ext = len(ext)
    ei6 = np.linalg.inv(z6 + REFERENCE_OHM * np.eye(n_ext))
    d6 = -2.0 * REFERENCE_OHM * ei6[N_BS:, :N_BS]
    truth = CoupledNetwork(z_full, LOADS)
    blocks = spherical_blocks(resp, bs, ue, frequency)
    z_model_full = z_ee_full_external(blocks)
    z_model_code = z_ee_as_in_code_under_test(blocks)
    d_model_full = -2.0 * REFERENCE_OHM * np.linalg.inv(z_model_full + REFERENCE_OHM * np.eye(n_ext))[N_BS:, :N_BS]
    d_model_code = -2.0 * REFERENCE_OHM * np.linalg.inv(z_model_code + REFERENCE_OHM * np.eye(n_ext))[N_BS:, :N_BS]
    z_ee_truth = z_full[n_ris:, n_ris:]
    nd = float(np.linalg.norm(truth.d))
    return {
        "scope": "non-gating attribution; external wires solved alone in NEC (6 wires)",
        "d_reference_norm": nd,
        "external_block_rel_err_model_full_vs_nec_6wire": rel_err(z_model_full, z6),
        "external_block_rel_err_nec_6wire_vs_nec_70port": rel_err(z6, z_ee_truth),
        "external_block_rel_err_model_full_vs_nec_70port": rel_err(z_model_full, z_ee_truth),
        "direct_channel_rel_err_vs_nec70": {
            "code_under_test_model": rel_err(d_model_code, truth.d),
            "model_with_full_external_coupling": rel_err(d_model_full, truth.d),
            "nec_external_only_6wire": rel_err(d6, truth.d)},
        "direct_channel_rel_err_vs_nec6wire": {
            "code_under_test_model": rel_err(d_model_code, d6),
            "model_with_full_external_coupling": rel_err(d_model_full, d6)},
        "reading": ("direct_channel_rel_err_vs_nec70['nec_external_only_6wire'] is the part of the truth direct "
                    "channel caused by the 64 open-gap RIS wires; direct_channel_rel_err_vs_nec6wire"
                    "['model_with_full_external_coupling'] is the pure dipole-model error of the external block."),
    }


# --------------------------------------------------------------------------- P2 rerun after the model fix

def task_external_response(ext_centers, wire_length, radius, frequency, segments):
    """External antennas (BS first, then UE) solved alone in NEC, no RIS wires."""
    from atomwise_ris.hybrid.embedded import extract_ris_response
    return extract_ris_response(np.asarray(ext_centers, dtype=float), wire_length=wire_length,
                                segments_per_wire=segments, radius=radius, frequency=frequency)


def task_open_gap_currents(ris_centers, field_vectors, segment_lengths, wire_length, radius, frequency, segments):
    """I_sc,p = S u_p for u_p = dl * E_p on the RIS-alone continuous structure (6 solves)."""
    from atomwise_ris.hybrid.ray_coupling import impressed_segment_currents
    u = np.asarray(segment_lengths)[:, None] * np.asarray(field_vectors)
    return impressed_segment_currents(ris_centers, u, wire_length=wire_length, segments_per_wire=segments,
                                      radius=radius, frequency=frequency)


def _network_report(net: Any, truth: CoupledNetwork, all_states: dict[str, np.ndarray], off: np.ndarray,
                    truth_scores: dict[str, Any]) -> dict[str, Any]:
    h_off_t, h_off_h = truth.channel(off), net.channel(off)
    per: dict[str, Any] = {}
    for name, bits in all_states.items():
        h_t, h_h = truth.channel(bits), net.channel(bits)
        se_t = float(truth.score(h_t @ truth.w))
        se_h = float(net.score(h_h @ net.w))
        den = np.linalg.norm(h_t - h_off_t)
        per[name] = {
            "bits": bits.tolist(), "channel_rel_err": rel_err(h_h, h_t),
            "se_truth_own_w": se_t, "se_hybrid_own_w": se_h, "se_abs_diff_own_w": abs(se_h - se_t),
            "se_abs_diff_common_w": abs(float(net.score(h_h @ truth.w)) - se_t),
            "modulation_fidelity_rel_err": (rel_err(h_h - h_off_h, h_t - h_off_t) if den > 0 else None),
        }
    sc = exhaustive_scores(net, off)
    t = truth_scores
    hs, hp = int(np.argmax(sc["singles"])), int(np.argmax(sc["pairs"]))
    ranking = {
        "spearman_single": spearman(sc["singles"], t["singles"]), "spearman_pair": spearman(sc["pairs"], t["pairs"]),
        "hybrid_chosen_single": hs, "hybrid_chosen_pair": [int(x) for x in sc["pair_index"][hp]],
        "regret_single": float(t["singles"].max() - t["singles"][hs]),
        "regret_pair": float(t["pairs"].max() - t["pairs"][hp]),
    }
    blocks = {"A": rel_err(net.a, truth.a), "C_in": rel_err(net.f, truth.f),
              "C_out": rel_err(net.c, truth.c), "H_direct": rel_err(net.d, truth.d)}
    return {"per_state": per, "ranking": ranking, "block_relative_errors": blocks}


def evaluate_p2_fixed(
    *, z_full: np.ndarray, resp: EmbeddedRISResponse, ext_resp: EmbeddedRISResponse,
    continuous_currents: np.ndarray, bs: np.ndarray, ue: np.ndarray,
    states: dict[str, np.ndarray], best: dict[str, Any], direct_best: dict[str, np.ndarray],
    n_ris: int, loads: np.ndarray = LOADS, frequency: float = FREQUENCY,
    thresholds: dict[str, float] = THRESHOLDS_P2,
) -> dict[str, Any]:
    """P2 for the fixed hybrid (NEC external block + open-gap scattering). Limits are THRESHOLDS_P2, unchanged.

    The truth ``z_full`` is used only for scoring; it enters no hybrid operator.
    """
    from atomwise_ris.hybrid.ray_coupling import (
        build_spherical_dipole_operators, build_spherical_nec_external_operators)

    truth = CoupledNetwork(z_full, loads)
    c_in, c_out, h_dir, a, info = build_spherical_nec_external_operators(
        resp, ext_resp, N_BS, continuous_currents=continuous_currents, frequency=frequency)
    gate_net = HybridCoupledNetwork(a, c_out, c_in, h_dir, loads)
    c_in_a, c_out_a, h_dir_a, a_a, _ = build_spherical_nec_external_operators(
        resp, ext_resp, N_BS, continuous_currents=None, frequency=frequency)
    ext_only = HybridCoupledNetwork(a_a, c_out_a, c_in_a, h_dir_a, loads)
    lc = build_spherical_dipole_operators(resp, bs, ue, wire_length=WIRE_LENGTH, frequency=frequency)
    legacy = HybridCoupledNetwork(lc[3], lc[1], lc[0], lc[2], loads)

    all_states = dict(states)
    all_states["best_single"] = best["bits_single"]
    all_states["best_pair"] = best["bits_pair"]
    off = all_states["all_off"]
    t = best["scores"]
    reports = {
        "fixed_nec_external_plus_open_gap": _network_report(gate_net, truth, all_states, off, t),
        "diagnostic_nec_external_only": _network_report(ext_only, truth, all_states, off, t),
        "diagnostic_legacy_dipole": _network_report(legacy, truth, all_states, off, t),
    }
    gate = reports["fixed_nec_external_plus_open_gap"]
    per, rk = gate["per_state"], gate["ranking"]

    p20 = []
    for name, sol in direct_best.items():
        h_nec = -2.0 * REFERENCE_OHM * sol[n_ris + N_BS:n_ris + N_BS + N_UE, :N_BS]
        p20.append(rel_err(truth.channel(all_states[name]), h_nec))
    contrast = float(np.linalg.norm(truth.channel(all_states["all_on"]) - truth.channel(off))
                     / np.linalg.norm(truth.channel(off)))
    crits = [
        criterion("P2.0", max(p20) if p20 else float("nan"), thresholds["P2.0_truth_state_channel_relative_max"], "<="),
        criterion("P2.1", max(r["channel_rel_err"] for r in per.values()), thresholds["P2.1_channel_relative_error_max"], "<="),
        criterion("P2.2", max(r["se_abs_diff_own_w"] for r in per.values()), thresholds["P2.2_worst_user_se_abs_diff_max_bps_hz"], "<="),
        criterion("P2.3a", rk["spearman_single"], thresholds["P2.3_spearman_min"], ">="),
        criterion("P2.3b", rk["spearman_pair"], thresholds["P2.3_spearman_min"], ">="),
        criterion("P2.4a", rk["regret_single"], thresholds["P2.4_regret_max_bps_hz"], "<="),
        criterion("P2.4b", rk["regret_pair"], thresholds["P2.4_regret_max_bps_hz"], "<="),
        criterion("P2.5", contrast, thresholds["P2.5_truth_modulability_min"], ">="),
    ]
    dz = info["delta_z_open_gap"]
    return {
        "gate": "P2", "builder": "fixed_nec_external_plus_open_gap",
        "criteria": crits, "passed": all(c["passed"] for c in crits), "thresholds": thresholds,
        "precoder_protocol_primary": "each network uses its own RZF of channel(all-OFF)",
        "per_state": per, "ranking": rk, "block_relative_errors": gate["block_relative_errors"],
        "best_states": {"single_index": best["single_index"], "pair_indices": best["pair_indices"],
                        "truth_se_best_single": float(t["singles"].max()),
                        "truth_se_best_pair": float(t["pairs"].max())},
        "diagnostic_networks_non_gating": {k: v for k, v in reports.items() if k.startswith("diagnostic")},
        "fix_info": {"delta_z_open_gap_norm": float(np.linalg.norm(dz)),
                     "z_ee_free_norm": float(np.linalg.norm(ext_resp.z_ris)),
                     "z_re_rel_err_vs_nec_truth_block_non_gating": rel_err(info["z_re"], z_full[:n_ris, n_ris:]),
                     "z_ee_rel_err_vs_nec_truth_block_non_gating": rel_err(info["z_ee"], z_full[n_ris:, n_ris:]),
                     "z_ee_free_only_rel_err_vs_nec_truth_block_non_gating": rel_err(ext_resp.z_ris, z_full[n_ris:, n_ris:]),
                     "delta_z_rel_err_vs_truth_difference_non_gating": rel_err(dz, z_full[n_ris:, n_ris:] - ext_resp.z_ris)},
    }
