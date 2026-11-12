"""Physics-sensitive checks for the reused production MoM basis."""
import importlib.util
import unittest

import numpy as np

from atomwise_ris.bench.production_channel import RisBasis, ray_z_projection, aggregate_classes
from atomwise_ris.hybrid.p4_rt_bridge import FREQ, field_from_jones, path_geometry
from atomwise_ris.hybrid.ray_coupling import impressed_segment_currents
from atomwise_ris.physics.nec import WireGeometry


class ProductionChannelTest(unittest.TestCase):
    def test_vertical_projection_equals_registered_p4_cartesian_map(self):
        from types import SimpleNamespace
        from atomwise_ris.hybrid.embedded import C0
        tx=np.array([4.,-2.,2.])
        rx=np.array([8.,8.,4.])
        length, dep, arr=path_geometry(tx,rx,False)
        offsets=np.array([-.004,0.,.004])
        current=np.array([.15+.01j,1.,.15-.01j])
        j=np.array([[.3+.2j,.1-.1j],[-.1+.3j,.1+.4j]])
        source=SimpleNamespace(segment_centers=tx+np.c_[np.zeros(3),np.zeros(3),offsets],
                               iz=current[:,None],segment_lengths=np.full(3,.004))
        row={"delay_s":length/C0,"jones":np.stack([j.real,j.imag],axis=-1)}
        expected=field_from_jones(row,tx,rx,source,rx[None])[0,2]
        z,_=ray_z_projection(j[None],np.array([np.arccos(dep[2])]),
                  np.array([np.arccos(arr[2])]),np.array([length]),offsets,current*.004)
        np.testing.assert_allclose(z,[expected],atol=1e-14,rtol=1e-12)

    def test_mixed_and_blocked_link_classes_are_not_geometric_guesses(self):
        # 2 RIS ports, one BS source, two UEs. UE0 has reflected RIS
        # response; UE1 sees only diffraction on its BS link.
        s=np.array([0,0,1,0]); d=np.array([0,1,0,4]); k=np.array([2,3,2,3])
        r=aggregate_classes(s,d,k,n=2,n_bs=1,k=2)
        self.assertEqual(r,{"bs_to_ris":"mixed","bs_to_user":["blocked","diffraction_only"],
                            "ris_to_user":["reflection_only","blocked"]})

    @unittest.skipUnless(importlib.util.find_spec("PyNEC"),"requires PyNEC")
    def test_reused_factorization_matches_independent_open_gap_rhs(self):
        centers=np.array([[0.,0.,1.],[0.,.028,1.]])
        seg=21
        z,_=WireGeometry(centers,np.full(2,.024),frequency=FREQ).impedance(seg)
        b=RisBasis(centers,seg,z)
        rng=np.random.default_rng(2)
        e=rng.normal(size=(2*seg,2))+1j*rng.normal(size=(2*seg,2))
        v=b.length*e
        raw=impressed_segment_currents(centers,v,segments_per_wire=seg)
        expected=-v.T@(raw-b.iz@(b.iy.T@v))
        expected=(expected+expected.T)/2
        np.testing.assert_allclose(b.open_gap(e),expected,atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(b.open_gap(2*e),4*expected,atol=1e-13,rtol=1e-10)
