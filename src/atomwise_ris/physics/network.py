"""Exact passive multiport termination and rank-one/rank-two RIS edits.

Port order: RIS cells, four BS ports, two UE ports. All external ports use
50-ohm real references; H maps incident BS power waves to received UE waves.
No empirical aperture multiplier or OpenSourceRIS gamma is used here.
"""
from dataclasses import dataclass
import numpy as np
from atomwise_ris.channels.precoding import precoder

@dataclass
class NetworkState:
    bits: np.ndarray
    inverse: np.ndarray
    field: np.ndarray

class CoupledNetwork:
    def __init__(self,z,loads,w=None,coupled=True):
        self.n=len(z)-6;self.loads=np.asarray(loads);self.z=np.asarray(z).copy()
        if not coupled:
            # Model mismatch ablation only: ground-truth evaluation stays coupled.
            rr=self.z[:self.n,:self.n];rr[:]=np.diag(np.diag(rr))
        n=self.n;e=self.z[n:,n:]+50*np.eye(6);ei=np.linalg.inv(e)
        self.a=self.z[:n,:n]-self.z[:n,n:]@ei@self.z[n:,:n]
        self.c=-100*(ei@self.z[n:,:n])[-2:]
        self.f=(self.z[:n,n:]@ei)[:,:4]
        self.d=-100*ei[-2:,:4]
        self.w=precoder(self.channel(np.zeros(n,dtype=np.int8))) if w is None else w
        self.fw=self.f@self.w;self.dw=self.d@self.w
        self.noise=1.380649e-23*290*1e6*10**(.7)
        self.pairs=np.array(np.triu_indices(n,1)).T
    def channel(self,bits):
        return self.d+self.c@np.linalg.solve(self.a+np.diag(self.loads[np.asarray(bits,int)]),self.f)
    def state(self,bits):
        b=np.asarray(bits,np.int8).copy();inv=np.linalg.inv(self.a+np.diag(self.loads[b]))
        return NetworkState(b,inv,self.dw+self.c@inv@self.fw)
    def score(self,fields):
        power=np.abs(fields)**2;signal=np.diagonal(power,axis1=-2,axis2=-1)
        # Sum interferers directly: total-minus-desired loses weak interference
        # when desired power is much larger, which biases high-SNR ranking.
        off=1-np.eye(power.shape[-1])
        interference=(power*off).sum(axis=-1)
        return np.log2(1+signal/(self.noise+interference)).min(-1)
    def moves(self,state,pairs=False,subset=None):
        t=state.inverse;l=self.c@t;r=t@self.fw
        delta=self.loads[1-state.bits]-self.loads[state.bits]
        den=1/delta+np.diag(t)
        if not pairs:
            if subset is None:
                fields=state.field[None]-np.einsum('ki,ij->ikj',l,r)/den[:,None,None]
            else:
                idx=np.asarray(subset,dtype=int)
                fields=state.field[None]-np.einsum('ki,ij->ikj',l[:,idx],r[idx])/den[idx,None,None]
        else:
            if subset is None:
                i,j=self.pairs.T
            else:
                sub=np.asarray(subset,dtype=int)
                idx_i,idx_j=np.triu_indices(len(sub),1)
                i,j=sub[idx_i],sub[idx_j]
            det=den[i]*den[j]-t[i,j]*t[j,i]
            ui=(den[j,None]*r[i]-t[i,j,None]*r[j])/det[:,None]
            uj=(den[i,None]*r[j]-t[j,i,None]*r[i])/det[:,None]
            fields=state.field[None]-l[:,i].T[:,:,None]*ui[:,None,:]-l[:,j].T[:,:,None]*uj[:,None,:]
        return np.asarray(self.score(fields))
    def _pair_scores(self,state,i,j):
        """Exact Woodbury rank-2 scores for toggling the cells (i[m],j[m]) together."""
        t=state.inverse;l=self.c@t;r=t@self.fw
        delta=self.loads[1-state.bits]-self.loads[state.bits]
        den=1/delta+np.diag(t)
        det=den[i]*den[j]-t[i,j]*t[j,i]
        ui=(den[j,None]*r[i]-t[i,j,None]*r[j])/det[:,None]
        uj=(den[i,None]*r[j]-t[j,i,None]*r[i])/det[:,None]
        fields=state.field[None]-l[:,i].T[:,:,None]*ui[:,None,:]-l[:,j].T[:,:,None]*uj[:,None,:]
        return np.asarray(self.score(fields))
    def moves_pairs(self,state,pairs):
        """Exact scores for an explicit (m,2) array of distinct-cell pairs (any order, any subset)."""
        p=np.asarray(pairs,dtype=int)
        if p.size==0:return np.empty(0,dtype=float)
        if p.ndim!=2 or p.shape[1]!=2:raise ValueError('pairs must have shape (m,2)')
        if p.min()<0 or p.max()>=self.n or np.any(p[:,0]==p[:,1]):
            raise ValueError('pairs need two distinct in-range cell indices')
        return self._pair_scores(state,p[:,0],p[:,1])
    def moves_subspace(self,state,atom_indices):
        sub=np.asarray(atom_indices,dtype=int)
        if len(sub)<2:
            return np.empty(0,dtype=float),np.empty((0,2),dtype=int)
        idx_i,idx_j=np.triu_indices(len(sub),1)
        i,j=sub[idx_i],sub[idx_j]
        pairs=np.column_stack([i,j])
        return self._pair_scores(state,i,j),pairs
    def moves_anchors(self,state,anchor_indices):
        anc=np.asarray(anchor_indices,dtype=int)
        if len(anc)==0:
            return np.empty(0,dtype=float),np.empty((0,2),dtype=int)
        mask=np.zeros(self.n,dtype=bool);mask[anc]=True
        pairs=self.pairs[mask[self.pairs[:,0]]|mask[self.pairs[:,1]]]
        return self._pair_scores(state,pairs[:,0],pairs[:,1]),pairs
    def score_multi_flips(self,state,flip_lists):
        return np.asarray([float(self.score(self.flip(state,ix).field)) for ix in flip_lists],dtype=float)
    def flip(self,state,indices):
        ix=np.asarray(indices,dtype=int).reshape(-1)
        if len(ix)==0:return state
        if len(set(ix.tolist()))!=len(ix):raise ValueError('Distinct flips required')
        delta=self.loads[1-state.bits[ix]]-self.loads[state.bits[ix]]
        small=np.diag(1/delta)+state.inverse[np.ix_(ix,ix)]
        inv=state.inverse-state.inverse[:,ix]@np.linalg.solve(small,state.inverse[ix])
        b=state.bits.copy();b[ix]=1-b[ix]
        return NetworkState(b,inv,self.dw+self.c@inv@self.fw)
    def features(self,state,first=-1,gains=None):
        """Return per-cell policy features, reusing caller-scored flip gains when supplied."""
        n=self.n
        gain=self.moves(state)-self.score(state.field) if gains is None else np.asarray(gains,dtype=float)
        t=state.inverse;l=self.c@t;r=t@self.fw
        # Analytical full-network score, port interaction and geometry features.
        coords=np.column_stack((np.arange(n)%8,np.arange(n)//8))/7
        interaction=np.abs(t).sum(1)-np.abs(np.diag(t))
        features=np.column_stack([state.bits,coords,gain,np.real(np.diag(t)),np.imag(np.diag(t)),
           interaction,l.real.T,l.imag.T,r.real,r.imag,np.arange(n)==first])
        return features.astype(np.float32),gain
