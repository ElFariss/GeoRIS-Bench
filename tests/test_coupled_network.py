import unittest
import numpy as np
from atomwise_ris.physics.network import CoupledNetwork

class CoupledTests(unittest.TestCase):
    def test_reduction_and_every_pair(self):
        rng=np.random.default_rng(7);n=4;v=rng.normal(size=(n+6,n+6));x=rng.normal(size=(n+6,n+6))
        z=v@v.T+10*np.eye(n+6)+1j*(x+x.T);loads=np.array([1+5j,1-5j]);ev=CoupledNetwork(z,loads)
        for number in range(2**n):
            b=np.array([(number>>i)&1 for i in range(n)],np.int8);state=ev.state(b)
            full=-100*np.linalg.solve(z+np.diag(np.r_[loads[b],np.full(6,50)]),np.eye(n+6)[:,n:n+4])[-2:]
            np.testing.assert_allclose(ev.channel(b),full,atol=1e-12)
            for ix,score in zip(ev.pairs,ev.moves(state,True)):
                new=ev.flip(state,ix);direct=ev.state(new.bits)
                np.testing.assert_allclose(new.field,direct.field,atol=1e-12)
                self.assertAlmostEqual(float(score),float(ev.score(direct.field)),places=10)
            for i,score in enumerate(ev.moves(state)):
                direct=ev.flip(state,[i]);self.assertAlmostEqual(float(score),float(ev.score(direct.field)),places=10)
    def test_distinct(self):
        e=CoupledNetwork(50*np.eye(10,dtype=complex),np.array([1+5j,1-5j]));s=e.state(np.zeros(4,int))
        with self.assertRaises(ValueError):e.flip(s,[0,0])
if __name__=='__main__':unittest.main()
