"""Hybrid Ray-Tracing + NEC-2 Multiport Electromagnetic Channel Module."""

from atomwise_ris.hybrid.operator import HybridCoupledNetwork, HybridState
from atomwise_ris.hybrid.embedded import (
    EmbeddedRISResponse,
    extract_ris_response,
    zdip_E,
    C0,
    ETA0,
)
from atomwise_ris.hybrid.ray_coupling import (
    build_spherical_dipole_operators,
    build_multipath_operators,
    build_spherical_nec_external_operators,
)
from atomwise_ris.hybrid.gates import (
    run_gate_p1,
    run_gate_p2,
    run_gate_p3,
    run_gate_p4,
    run_all_gates,
    THRESHOLDS_P1,
    THRESHOLDS_P2,
    THRESHOLDS_P3,
    THRESHOLDS_P4,
)

__all__ = [
    'HybridCoupledNetwork',
    'HybridState',
    'EmbeddedRISResponse',
    'extract_ris_response',
    'zdip_E',
    'C0',
    'ETA0',
    'build_spherical_dipole_operators',
    'build_multipath_operators',
    'build_spherical_nec_external_operators',
    'run_gate_p1',
    'run_gate_p2',
    'run_gate_p3',
    'run_gate_p4',
    'run_all_gates',
    'THRESHOLDS_P1',
    'THRESHOLDS_P2',
    'THRESHOLDS_P3',
    'THRESHOLDS_P4',
]
