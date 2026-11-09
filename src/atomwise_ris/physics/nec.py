"""NEC-2 thin-wire MoM port extraction, with SI inputs and explicit conventions.

This is a free-space PEC wire reference, NOT a calibrated OpenSourceRIS PCB.
Voltage sources and loads occupy the center segment. Y columns are currents
for a unit impressed voltage with all other gap voltages zero. Z=inv(Y).
The complete network includes the transmitting and receiving antennas, so
radiation, structural scattering and all inter-port coupling are solved together.
"""
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class WireGeometry:
    centers: np.ndarray
    lengths: np.ndarray
    radius: float = 0.00005
    frequency: float = 5.53e9

    def context(self, segments=11, loads=None):
        import PyNEC
        if segments < 3 or segments % 2 != 1:
            raise ValueError('Use an odd number of segments >=3')
        n = PyNEC.nec_context()
        g = n.get_geometry()
        for q, ((x,y,z), length) in enumerate(zip(self.centers, self.lengths)):
            g.wire(q+1,segments,x,y,z-length/2,x,y,z+length/2,self.radius,1,1)
        n.geometry_complete(0)
        n.set_extended_thin_wire_kernel(True)
        n.fr_card(0,1,self.frequency/1e6,0)  # NEC FR card uses MHz.
        n.pt_card(0,0,0,0)
        if loads is not None:
            for q, load in enumerate(loads):
                if abs(load):
                    n.ld_card(4,q+1,segments//2+1,segments//2+1,float(load.real),float(load.imag),0)
        return n

    def impedance(self, segments=11):
        start = time.perf_counter()
        n = self.context(segments)
        size = len(self.centers)
        columns=[]
        for q in range(size):
            n.ex_card(0,q+1,segments//2+1,0,1,0,0,0,0,0)
            n.xq_card(0)
            c=n.get_structure_currents(q)
            if abs(c.get_frequency()-self.frequency)>1:
                raise RuntimeError('NEC frequency mismatch')
            columns.append(np.asarray(c.get_current())[np.arange(size)*segments+segments//2].copy())
        y=np.asarray(columns).T
        return np.linalg.inv(y), {'seconds':time.perf_counter()-start,
            'segments_per_wire':segments,'unknowns':segments*size}

    def loaded_current(self, loads, source=0, segments=11):
        n=self.context(segments,loads)
        n.ex_card(0,source+1,segments//2+1,0,1,0,0,0,0,0)
        n.xq_card(0)
        return np.asarray(n.get_structure_currents(0).get_current())[
            np.arange(len(self.centers))*segments+segments//2].copy()

def scattering(z, reference_ohm=50.):
    eye=np.eye(len(z))
    # S=(Z-Z0 I) @ inv(Z+Z0 I), without explicit right inverse.
    return np.linalg.solve((z+reference_ohm*eye).T,(z-reference_ohm*eye).T).T

def diagnostics(z):
    s=scattering(z)
    return {'reciprocity_relative':float(np.linalg.norm(z-z.T)/np.linalg.norm(z)),
            'minimum_resistance_eigenvalue_ohm':float(np.linalg.eigvalsh((z+z.conj().T)/2).min()),
            'maximum_scattering_singular_value':float(np.linalg.svd(s,compute_uv=False)[0]),
            'condition_number':float(np.linalg.cond(z))}

def geometry(side=8, users=None):
    # Panel lies in yz; wires are z-polarized and do not overlap (24<30 mm).
    cells=np.array([[0,(j-(side-1)/2)*.028,1+(i-(side-1)/2)*.030]
                    for i in range(side) for j in range(side)])
    bs=np.array([[-.50,-.35+(j-1.5)*.0271,1.04] for j in range(4)])
    if users is None:
        users=np.array([[-.40,.30,.98],[-.48,.38,1.03]])
    centers=np.concatenate([cells,bs,users])
    return WireGeometry(centers,np.r_[np.full(len(cells),.024),np.full(6,.024)])
