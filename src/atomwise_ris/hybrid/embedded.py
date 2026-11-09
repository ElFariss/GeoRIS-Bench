"""Embedded element pattern and electromagnetic reciprocity integrals.

Extracts segment currents from PyNEC WireGeometry and computes:
- Full segment current distribution matrices I_Y and I_Z = I_Y @ Z_RIS
- Embedded vector effective length h_eff(theta, phi)
- Mutual impedance and open-circuit voltages by Lorentz reciprocity
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

C0 = 299792458.0
ETA0 = 376.730313668

def zdip_E(
    robs: np.ndarray,
    rsrc: np.ndarray,
    p: float = 1.0,
    frequency: float = 5.53e9
) -> np.ndarray:
    """Exact near, intermediate and far electric field of a z-directed Hertzian dipole.
    
    Parameters
    ----------
    robs : np.ndarray
        (..., 3) observation points in meters.
    rsrc : np.ndarray
        (3,) or (..., 3) source dipole position in meters.
    p : float
        Electric dipole moment (p = I * dl / (j * omega)).
    frequency : float
        Operating frequency in Hz.
        
    Returns
    -------
    np.ndarray
        (..., 3) complex electric field vector [Ex, Ey, Ez] in V/m.
    """
    lam = C0 / frequency
    k = 2.0 * np.pi / lam
    d = robs - rsrc
    r = np.linalg.norm(d, axis=-1, keepdims=True)
    # Avoid zero-division at source point
    r_safe = np.maximum(r, 1e-12)
    rh = d / r_safe
    ct = rh[..., 2:3] # z component (cos theta)
    
    kr = k * r_safe
    jkr = 1j * kr
    
    # Standard dyadic Green's function for z-directed electric dipole
    # A corresponds to transverse radiation term, B to radial term
    A = (1j * k * ETA0 * p / (4.0 * np.pi * r_safe)) * (1.0 + 1.0 / jkr - 1.0 / (kr ** 2)) * np.exp(-1j * kr)
    B = (ETA0 * p / (2.0 * np.pi * (r_safe ** 2))) * (1.0 + 1.0 / jkr) * np.exp(-1j * kr)
    
    E = ((A + B) * ct) * rh
    E[..., 2:3] -= A
    return E

@dataclass
class EmbeddedRISResponse:
    """Multiport segment and port response extracted from NEC-2."""
    z_ris: np.ndarray             # (N, N) port impedance matrix
    y_ris: np.ndarray             # (N, N) port admittance matrix
    segment_centers: np.ndarray   # (M, 3) segment centers in meters
    segment_lengths: np.ndarray   # (M,) segment lengths in meters
    iy: np.ndarray                # (M, N) segment currents per unit port voltage
    iz: np.ndarray                # (M, N) segment currents per unit port current
    wire_centers: np.ndarray      # (N, 3) element wire center coordinates
    frequency: float = 5.53e9
    
    @property
    def n_elements(self) -> int:
        return len(self.wire_centers)

    def effective_length(self, theta_rad: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
        """Compute embedded vector effective length h_eff,theta for all N elements.
        
        Parameters
        ----------
        theta_rad : np.ndarray
            Zenith angles in radians (0 at +z, pi/2 at xy plane).
        phi_rad : np.ndarray
            Azimuth angles in radians.
            
        Returns
        -------
        np.ndarray
            (N, ...) complex effective length in meters along theta-hat.
            (h_eff,phi is identically zero for z-directed wires).
        """
        k = 2.0 * np.pi * self.frequency / C0
        th = np.asarray(theta_rad)
        ph = np.asarray(phi_rad)
        
        # Unit vector pointing toward the arrival source
        # k_arr = [sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]
        kx = np.sin(th) * np.cos(ph)
        ky = np.sin(th) * np.sin(ph)
        kz = np.cos(th)
        k_dir = np.stack([kx, ky, kz], axis=0) # (3, ...)
        
        # Array center
        r_ris = np.mean(self.wire_centers, axis=0)
        
        # Segment relative positions: (M, 3)
        rel_pos = self.segment_centers - r_ris
        
        # Spatial phase factor: exp(j * k * dot(rel_pos, k_dir))
        # rel_pos is (M, 3), k_dir is (3, ...)
        phase = np.exp(1j * k * np.tensordot(rel_pos, k_dir, axes=(1, 0))) # (M, ...)
        
        # Dipole projection onto z-directed wires: z_hat . theta_hat = -sin(theta)
        # By Lorentz reciprocity: h_eff,theta = sin(theta) * sum_m (dl_m * I_Z[m, n] * phase[m])
        # Shape: (N, M) @ (M, ...) -> (N, ...)
        weight = (self.iz * self.segment_lengths[:, None]).T # (N, M)
        h_theta = np.sin(th) * np.tensordot(weight, phase, axes=(1, 0)) # (N, ...)
        return h_theta

    def open_circuit_voltage_from_field(self, e_inc_z: np.ndarray) -> np.ndarray:
        """Compute induced open-circuit voltage v_oc at each RIS port from incident Ez field.
        
        v_oc = - sum_m (I_Z[m, n] * dl_m * E_inc_z[m])
        """
        weight = (self.iz * self.segment_lengths[:, None]).T # (N, M)
        return - weight @ e_inc_z

    def short_circuit_current_from_field(self, e_inc_z: np.ndarray) -> np.ndarray:
        """Compute short-circuit current i_sc at each RIS port from incident Ez field.
        
        i_sc = sum_m (I_Y[m, n] * dl_m * E_inc_z[m])
        """
        weight = (self.iy * self.segment_lengths[:, None]).T # (N, M)
        return weight @ e_inc_z


def extract_ris_response(
    wire_centers: np.ndarray,
    wire_length: float = 0.024,
    segments_per_wire: int = 21,
    radius: float = 0.00005,
    frequency: float = 5.53e9
) -> EmbeddedRISResponse:
    """Extract complete segment current distribution and impedance matrix using PyNEC.
    
    Parameters
    ----------
    wire_centers : np.ndarray
        (N, 3) center coordinates of all RIS wires.
    wire_length : float
        Length of each dipole wire in meters (default 24 mm).
    segments_per_wire : int
        Odd number of segments per wire (e.g. 21, 41, 81).
    radius : float
        Wire radius in meters (default 50 um).
    frequency : float
        Frequency in Hz (default 5.53 GHz).
        
    Returns
    -------
    EmbeddedRISResponse
    """
    from atomwise_ris.physics.nec import WireGeometry
    
    centers = np.asarray(wire_centers, dtype=float)
    n_elements = len(centers)
    lengths = np.full(n_elements, float(wire_length))
    
    g = WireGeometry(centers, lengths, radius=radius, frequency=frequency)
    n = g.context(segments=segments_per_wire)
    lam = C0 / frequency
    
    cols = []
    sc = None
    dl = None
    for q in range(n_elements):
        n.ex_card(0, q + 1, segments_per_wire // 2 + 1, 0, 1.0, 0.0, 0, 0, 0, 0)
        n.xq_card(0)
        c = n.get_structure_currents(q)
        cols.append(np.asarray(c.get_current()).copy())
        if q == 0:
            sc = np.column_stack([
                np.asarray(getattr(c, f'get_current_segment_center_{a}')()) * lam
                for a in 'xyz'
            ])
            dl = np.asarray(c.get_current_segment_length()) * lam
            
    iy = np.array(cols).T # (M, N)
    mid_indices = np.arange(n_elements) * segments_per_wire + segments_per_wire // 2
    y_ris = iy[mid_indices, :]
    z_ris = np.linalg.inv(y_ris)
    iz = iy @ z_ris
    
    return EmbeddedRISResponse(
        z_ris=z_ris,
        y_ris=y_ris,
        segment_centers=sc,
        segment_lengths=dl,
        iy=iy,
        iz=iz,
        wire_centers=centers,
        frequency=frequency
    )
