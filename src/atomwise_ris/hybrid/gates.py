"""Implementation and runner for numerical validation gates P1 to P4.

Declared tolerances match docs/hybrid-gates.md.
Gate results are saved to outputs/runs/hybrid_gates/.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from atomwise_ris.physics.nec import geometry, diagnostics
from atomwise_ris.hybrid.embedded import C0, ETA0
from atomwise_ris.hybrid.operator import HybridCoupledNetwork

# Pre-declared gate thresholds (must match docs/hybrid-gates.md)
THRESHOLDS_P1 = {
    'reciprocity_relative': 1.0e-6,
    'min_resistance_eigenvalue_ohm': 0.0,
    'max_scattering_singular_value': 1.0001,
    'condition_number': 1.0e5,
    'schur_reduction_relative_error': 1.0e-10,
}

THRESHOLDS_P2 = {
    'factored_hybrid_max_relative_error': 0.05, # 5.0%
    'state_contrast_minimum': 0.01,              # 1.0%
    'ranking_correlation_minimum': 0.95,
}

THRESHOLDS_P3 = {
    'mesh_81_vs_161_channel_relative_error': 0.015, # 1.5%
    'mesh_41_vs_81_channel_relative_error': 0.035,  # 3.5%
    'mesh_81_vs_161_rate_absolute_error': 0.05,     # 0.05 bps/Hz
}

THRESHOLDS_P4 = {
    'max_channel_magnitude': 1.0,
    'load_sensitivity_contrast_minimum': 0.005,     # 0.5%
    'mimo_rank_minimum': 2,
}


def run_gate_p1(
    z_path: str = 'outputs/runs/coupling_halfwave64_v1/z_side8_segments81.npy'
) -> Dict[str, Any]:
    """Execute Gate P1: Multiport Network & Port Consistency."""
    z = np.load(z_path)
    z_ris = z[:64, :64]
    
    diag = diagnostics(z_ris)
    loads = np.array([1.0 - 10.8j, 1.0 + 129.2j])
    net = HybridCoupledNetwork.from_monolithic_z(z, loads, n_ris=64)
    
    # Test Schur reduction on sampled states
    rng = np.random.default_rng(42)
    schur_errors = []
    for _ in range(10):
        b = rng.integers(0, 2, size=64)
        h_factored = net.channel(b)
        zl_full = np.r_[loads[b], np.full(6, 50.0)]
        rhs = np.eye(70)[:, 64:68]
        sol = np.linalg.solve(z + np.diag(zl_full), rhs)
        h_direct = -100.0 * sol[68:]
        err = float(np.linalg.norm(h_factored - h_direct) / np.linalg.norm(h_direct))
        schur_errors.append(err)
        
    passed = bool(
        diag['reciprocity_relative'] <= THRESHOLDS_P1['reciprocity_relative']
        and diag['minimum_resistance_eigenvalue_ohm'] > THRESHOLDS_P1['min_resistance_eigenvalue_ohm']
        and diag['maximum_scattering_singular_value'] <= THRESHOLDS_P1['max_scattering_singular_value']
        and diag['condition_number'] < THRESHOLDS_P1['condition_number']
        and max(schur_errors) <= THRESHOLDS_P1['schur_reduction_relative_error']
    )
    
    return {
        'gate': 'P1',
        'passed': passed,
        'metrics': {
            'reciprocity_relative': diag['reciprocity_relative'],
            'minimum_resistance_eigenvalue_ohm': diag['minimum_resistance_eigenvalue_ohm'],
            'maximum_scattering_singular_value': diag['maximum_scattering_singular_value'],
            'condition_number': diag['condition_number'],
            'max_schur_reduction_error': max(schur_errors),
            'schur_errors': schur_errors,
        },
        'thresholds': THRESHOLDS_P1,
    }


def run_gate_p2(
    z_path: str = 'outputs/runs/coupling_halfwave64_v1/z_side8_segments81.npy'
) -> Dict[str, Any]:
    """Execute Gate P2: Free-Space Benchmark vs 70-Port NEC-2."""
    z = np.load(z_path)
    loads = np.array([1.0 - 10.8j, 1.0 + 129.2j])
    
    # 1. Full monolithic 70-port network
    net_full = HybridCoupledNetwork.from_monolithic_z(z, loads, n_ris=64)
    
    # 2. Factored hybrid operator isolating RIS array and external antenna arrays
    e_bs = z[64:68, 64:68] + 50.0 * np.eye(4)
    e_ue = z[68:70, 68:70] + 50.0 * np.eye(2)
    ei_bs = np.linalg.inv(e_bs)
    ei_ue = np.linalg.inv(e_ue)
    
    c_hyb = - 100.0 * (ei_ue @ z[68:70, :64])
    f_hyb = z[:64, 64:68] @ ei_bs
    d_hyb = 100.0 * (ei_ue @ z[68:70, 64:68] @ ei_bs)
    a_hyb = z[:64, :64]
    
    net_hyb = HybridCoupledNetwork(a_hyb, c_hyb, f_hyb, d_hyb, loads, w=net_full.w)
    
    # Test states: all-OFF, all-ON, best-single, best-pair, 10 random
    s_off = net_full.state(np.zeros(64, int))
    best_single_idx = int(np.argmax(net_full.moves(s_off, pairs=False)))
    best_pair_idx = int(np.argmax(net_full.moves(s_off, pairs=True)))
    best_pair = net_full.pairs[best_pair_idx]
    
    b_single = np.zeros(64, int); b_single[best_single_idx] = 1
    b_pair = np.zeros(64, int); b_pair[best_pair] = 1
    
    rng = np.random.default_rng(42)
    test_states = [
        ('all-OFF', np.zeros(64, int)),
        ('all-ON', np.ones(64, int)),
        ('best-single', b_single),
        ('best-pair', b_pair),
    ] + [(f'rand_{i}', rng.integers(0, 2, size=64)) for i in range(10)]
    
    rel_errors = []
    full_scores = []
    hyb_scores = []
    
    for name, b in test_states:
        h_f = net_full.channel(b)
        h_h = net_hyb.channel(b)
        err = float(np.linalg.norm(h_f - h_h) / np.linalg.norm(h_f))
        rel_errors.append(err)
        
        score_f = float(net_full.score(h_f @ net_full.w))
        score_h = float(net_hyb.score(h_h @ net_full.w))
        full_scores.append(score_f)
        hyb_scores.append(score_h)
        
    # Contrast between all-ON and all-OFF
    h_off = net_hyb.channel(np.zeros(64, int))
    h_on = net_hyb.channel(np.ones(64, int))
    contrast = float(np.linalg.norm(h_on - h_off) / np.linalg.norm(h_off))
    
    # Spearman rank correlation
    rank_f = np.argsort(np.argsort(full_scores))
    rank_h = np.argsort(np.argsort(hyb_scores))
    corr = float(np.corrcoef(rank_f, rank_h)[0, 1])
    
    passed = bool(
        max(rel_errors) <= THRESHOLDS_P2['factored_hybrid_max_relative_error']
        and contrast >= THRESHOLDS_P2['state_contrast_minimum']
        and corr >= THRESHOLDS_P2['ranking_correlation_minimum']
    )
    
    return {
        'gate': 'P2',
        'passed': passed,
        'metrics': {
            'max_relative_error': max(rel_errors),
            'mean_relative_error': float(np.mean(rel_errors)),
            'modulation_contrast': contrast,
            'ranking_spearman_correlation': corr,
            'test_states_count': len(test_states),
        },
        'thresholds': THRESHOLDS_P2,
    }


def run_gate_p3(
    z41_path: str = 'outputs/runs/coupling_halfwave64_v1/z_side8_segments41.npy',
    z81_path: str = 'outputs/runs/coupling_halfwave64_v1/z_side8_segments81.npy',
    z161_path: str = 'outputs/runs/coupled_wire_v1/z_reference161.npy',
) -> Dict[str, Any]:
    """Execute Gate P3: Mesh Discretization Fidelity (41 vs 81 vs 161)."""
    z41 = np.load(z41_path)
    z81 = np.load(z81_path)
    z161 = np.load(z161_path)
    loads = np.array([1.0 - 10.8j, 1.0 + 129.2j])
    
    net41 = HybridCoupledNetwork.from_monolithic_z(z41, loads, n_ris=64)
    net81 = HybridCoupledNetwork.from_monolithic_z(z81, loads, n_ris=64)
    net161 = HybridCoupledNetwork.from_monolithic_z(z161, loads, n_ris=64)
    
    s_off = net81.state(np.zeros(64, int))
    best_single = int(np.argmax(net81.moves(s_off, pairs=False)))
    best_pair = net81.pairs[int(np.argmax(net81.moves(s_off, pairs=True)))]
    
    b_single = np.zeros(64, int); b_single[best_single] = 1
    b_pair = np.zeros(64, int); b_pair[best_pair] = 1
    
    rng = np.random.default_rng(77)
    test_states = [
        ('all-OFF', np.zeros(64, int)),
        ('all-ON', np.ones(64, int)),
        ('best-single', b_single),
        ('best-pair', b_pair),
    ] + [(f'rand_{i}', rng.integers(0, 2, size=64)) for i in range(20)]
    
    errs_81_161 = []
    errs_41_81 = []
    rate_errs_81_161 = []
    
    for _, b in test_states:
        h41 = net41.channel(b)
        h81 = net81.channel(b)
        h161 = net161.channel(b)
        
        e_81_161 = float(np.linalg.norm(h81 - h161) / np.linalg.norm(h161))
        e_41_81  = float(np.linalg.norm(h41 - h81)  / np.linalg.norm(h81))
        
        r81 = float(net81.score(h81 @ net81.w))
        r161 = float(net161.score(h161 @ net161.w))
        r_err = abs(r81 - r161)
        
        errs_81_161.append(e_81_161)
        errs_41_81.append(e_41_81)
        rate_errs_81_161.append(r_err)
        
    passed = bool(
        max(errs_81_161) <= THRESHOLDS_P3['mesh_81_vs_161_channel_relative_error']
        and max(errs_41_81) <= THRESHOLDS_P3['mesh_41_vs_81_channel_relative_error']
        and max(rate_errs_81_161) <= THRESHOLDS_P3['mesh_81_vs_161_rate_absolute_error']
    )
    
    return {
        'gate': 'P3',
        'passed': passed,
        'metrics': {
            'max_channel_error_81_vs_161': max(errs_81_161),
            'max_channel_error_41_vs_81': max(errs_41_81),
            'max_rate_error_81_vs_161': max(rate_errs_81_161),
            'mean_channel_error_81_vs_161': float(np.mean(errs_81_161)),
            'test_states_count': len(test_states),
        },
        'thresholds': THRESHOLDS_P3,
    }


def run_gate_p4(
    scenario_npz_path: str = 'data/processed/munich_specular_v3/scenario.npz',
    deepmimo_dir: str = 'data/raw/deepmimo_scenarios/munich_specular_v3_bs_fullpaths',
    z_ris_path: str = 'outputs/runs/coupling_halfwave64_v1/z_side8_segments81.npy',
    frequency: float = 5.53e9
) -> Dict[str, Any]:
    """Execute Gate P4: Single-Path & Multi-Path Urban Case (Munich Scene)."""
    z = np.load(z_ris_path)
    z_ris = z[:64, :64]
    
    s = np.load(scenario_npz_path)
    d_dir = Path(deepmimo_dir)
    
    lam = C0 / frequency
    scale = 1.0 / np.sqrt(50.0 / ETA0)
    
    # 1. Build C_in for single dominant path (LOS) and multipath sum
    c_in_single = np.zeros((64, 4), dtype=complex)
    c_in_multi  = np.zeros((64, 4), dtype=complex)
    
    for tx in range(4):
        pwr = np.load(d_dir / f'power_t000_tx{tx:03d}_r001.npz')['power'][:64]
        phs = np.load(d_dir / f'phase_t000_tx{tx:03d}_r001.npz')['phase'][:64]
        tau = np.load(d_dir / f'delay_t000_tx{tx:03d}_r001.npz')['delay'][:64]
        el  = np.load(d_dir / f'aoa_el_t000_tx{tx:03d}_r001.npz')['aoa_el'][:64]
        
        # Path 0 (LOS)
        amp0 = 10.0 ** (pwr[:, 0] / 20.0)
        ph0 = np.radians(phs[:, 0]) - 2.0 * np.pi * frequency * tau[:, 0]
        th0 = np.radians(el[:, 0])
        h_eff0 = (lam / np.pi) * np.sin(th0)
        c_in_single[:, tx] = amp0 * np.exp(1j * ph0) * h_eff0 * scale
        
        # Multipath sum
        for p in range(4):
            valid = ~np.isnan(pwr[:, p]) & (pwr[:, p] > -200)
            amp_p = np.zeros(64); amp_p[valid] = 10.0 ** (pwr[valid, p] / 20.0)
            ph_p = np.zeros(64); ph_p[valid] = np.radians(phs[valid, p]) - 2.0 * np.pi * frequency * tau[valid, p]
            th_p = np.zeros(64); th_p[valid] = np.radians(el[valid, p])
            h_eff_p = (lam / np.pi) * np.sin(th_p)
            c_in_multi[:, tx] += amp_p * np.exp(1j * ph_p) * h_eff_p * scale

    # 2. Build C_out from traced probe-to-user response
    # Frame 0
    h_probe0 = s['H'][0] # shape (2, 64)
    c_out = h_probe0 * ((lam / np.pi) / np.sqrt(lam**2 / (4.0 * np.pi))) * np.sqrt(ETA0 / 50.0)
    d0 = s['D'][0] # shape (2, 4)
    
    loads = np.array([1.0 - 10.8j, 1.0 + 129.2j])
    net_single = HybridCoupledNetwork(z_ris, c_out, c_in_single, d0, loads)
    net_multi  = HybridCoupledNetwork(z_ris, c_out, c_in_multi,  d0, loads)
    
    h_off_s = net_single.channel(np.zeros(64, int))
    h_on_s  = net_single.channel(np.ones(64, int))
    h_off_m = net_multi.channel(np.zeros(64, int))
    h_on_m  = net_multi.channel(np.ones(64, int))
    
    max_h_mag = float(np.abs(h_off_m).max())
    has_nan = bool(np.isnan(h_off_m).any() or np.isinf(h_off_m).any())
    contrast_m = float(np.linalg.norm(h_on_m - h_off_m) / np.linalg.norm(h_off_m))
    contrast_s = float(np.linalg.norm(h_on_s - h_off_s) / np.linalg.norm(h_off_s))
    mimo_rank = int(np.linalg.matrix_rank(h_off_m))
    
    passed = bool(
        (not has_nan)
        and max_h_mag < THRESHOLDS_P4['max_channel_magnitude']
        and contrast_m >= THRESHOLDS_P4['load_sensitivity_contrast_minimum']
        and mimo_rank >= THRESHOLDS_P4['mimo_rank_minimum']
    )
    
    return {
        'gate': 'P4',
        'passed': passed,
        'metrics': {
            'max_channel_magnitude': max_h_mag,
            'has_nan_or_inf': has_nan,
            'multipath_load_sensitivity_contrast': contrast_m,
            'single_path_load_sensitivity_contrast': contrast_s,
            'mimo_rank': mimo_rank,
            'mean_c_in_magnitude': float(np.abs(c_in_multi).mean()),
            'mean_c_out_magnitude': float(np.abs(c_out).mean()),
        },
        'thresholds': THRESHOLDS_P4,
    }


def run_all_gates(out_dir: str = 'outputs/runs/hybrid_gates_audit_20261106') -> Dict[str, Any]:
    """Fail closed on the saved claims; old probes are not independent gates.

    The original per-gate functions are retained for reproducibility of the
    legacy calculations, but their `passed` flags do not authorize training or
    hybrid optimization. New numerical runs belong in the separately dated
    direct-validation workflow.
    """
    from atomwise_ris.hybrid.evidence import audit_legacy_gate_evidence

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = audit_legacy_gate_evidence()
    result['all_passed'] = False
    result['failing_gate'] = 'P1'
    result['interpretation'] = 'Legacy pass flags are superseded by this evidence audit.'
    summary_path = out / 'gate_evidence_audit.json'
    if summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite prior audit: {summary_path}')
    summary_path.write_text(json.dumps(result, indent=2) + '\n')
    return result
