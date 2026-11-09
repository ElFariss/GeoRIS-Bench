"""Absolute Jones-path to NEC-current/voltage bridge for P4 registered paths."""
from dataclasses import replace
import numpy as np

from .embedded import C0, ETA0, extract_ris_response

FREQ = 5.53e9
NEC_C = 1 / np.sqrt(4*np.pi*1e-7 * 8.854e-12)
NEC_ETA = np.sqrt(4*np.pi*1e-7 / 8.854e-12)


def response(centers):
    """Restore the physical metre coordinates of the pinned PyNEC getter."""
    raw = extract_ris_response(centers, segments_per_wire=81, frequency=FREQ)
    scale = NEC_C / C0
    return replace(raw, segment_centers=raw.segment_centers*scale,
                   segment_lengths=raw.segment_lengths*scale)


def vh_basis(outward):
    """Global e_theta/e_phi (zero radio orientation, tr38901_2 V/H)."""
    x,y,z = np.asarray(outward, dtype=float)
    rho = np.hypot(x,y)
    if rho < 1e-12:
        raise ValueError("azimuth undefined at pole")
    return np.array([[z*x/rho, -y/rho], [z*y/rho, x/rho], [-rho, 0.]])


def path_geometry(tx, rx, reflected):
    if reflected:
        image = tx * [1,1,-1]
        b = image + (rx-image)*(-image[2]/(rx[2]-image[2]))
        if not (np.abs(b[:2]) < 1000).all():
            raise ValueError("specular point off the controlled plate")
        dep, arr = b-tx, b-rx
        length = np.linalg.norm(dep)+np.linalg.norm(arr)
    else:
        dep,arr=rx-tx,tx-rx
        length=np.linalg.norm(dep)
    return float(length), dep/np.linalg.norm(dep), arr/np.linalg.norm(arr)


def field_from_jones(row, tx, rx, source_response, observation_points,
                     reflected=False, nec_medium=True):
    """V/m per one ampere at the source feed; no response fitting."""
    if row is None:
        return np.zeros((len(observation_points),3), dtype=complex)
    length, dep, arr = path_geometry(tx, rx, reflected)
    if abs(length - C0*row["delay_s"]) > .0002:
        raise ValueError("traced delay inconsistent with the registered path geometry")
    a = np.asarray(row["jones"],float)
    if a.shape!=(2,2,2) or not np.isfinite(a).all():
        raise ValueError("invalid Jones matrix")
    jones = a[...,0]+1j*a[...,1]
    k = 2*np.pi*FREQ/(NEC_C if nec_medium else C0)
    eta = NEC_ETA if nec_medium else ETA0
    tx_phase = np.exp(1j*k*((source_response.segment_centers-tx)@dep))
    moment_z = np.sum(source_response.iz[:,0]*source_response.segment_lengths*tx_phase)
    moment = np.array([0.,0.,moment_z])
    electric = (-1j*k*eta/(C0/FREQ) * np.exp(-1j*k*length)
                * (vh_basis(arr) @ jones @ vh_basis(dep).T @ moment))
    phase = np.exp(1j*k*((observation_points-rx)@arr))
    return phase[:,None]*electric


def isolated_nec_fields(source, probes):
    """Independent PyNEC NE-card electric field, normalized by solved feed current."""
    from atomwise_ris.physics.nec import WireGeometry
    n = WireGeometry(source[None,:], np.array([.024]), frequency=FREQ).context(81)
    n.ex_card(0,1,41,0,1,0,0,0,0,0)
    vectors=[]
    for i, (x,y,z) in enumerate(probes):
        n.ne_card(0,1,1,1,float(x),float(y),float(z),0.,0.,0.)
        result = n.get_near_field_pattern(i)
        xyz = np.array([getattr(result,f"get_{a}")()[0] for a in "xyz"])
        np.testing.assert_allclose(xyz,[x,y,z],rtol=0,atol=2e-9)
        vectors.append([getattr(result,f"get_field_{a}")()[0] for a in "xyz"])
    current = complex(n.get_structure_currents(0).get_current()[40])
    if abs(current)==0:
        raise ValueError("zero NEC source feed current")
    return np.asarray(vectors)/current
