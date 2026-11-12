from dataclasses import dataclass,field
import numpy as np
from atomwise_ris.channels.precoding import precoder,C0


@dataclass
class Evaluator:
    direct: np.ndarray              # [K,K] = D @ W
    contribution: np.ndarray        # [N,K,K], before Gamma
    gamma: np.ndarray               # [N,2] or [2]
    noise_w: float
    cell_positions: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n=self.contribution.shape[0]
        self.gamma=np.broadcast_to(self.gamma,(self.n,2)).copy()
        self.base=self.direct + np.einsum('n,nkj->kj',self.gamma[:,0],self.contribution)
        self.delta=(self.gamma[:,1]-self.gamma[:,0])[:,None,None]*self.contribution
        self.pairs=np.column_stack(np.triu_indices(self.n,1))

    @classmethod
    def from_scenario(cls,s,index,response,aperture_multiplier=1.0,precoder_response=None):
        if s.metadata['frequency_hz'] != response.frequency_hz:
            raise ValueError('Propagation and response frequencies differ')
        area=s.metadata['cell_area_m2']
        factor=4*np.pi*area/(C0/response.frequency_hz)**2 * aperture_multiplier
        reference_gamma=response.gamma if precoder_response is None else precoder_response
        off=np.broadcast_to(reference_gamma,(s.G.shape[0],2))[:,0]
        # Fixed ALL-OFF reference handles blocked direct paths without silently
        # setting the corresponding user's transmit stream to zero.
        reference=s.D[index]+np.einsum('kn,n,nm->km',s.H[index],off,s.G)*factor
        w=precoder(reference,s.metadata['power_w'])
        c=np.einsum('kn,nj->nkj',s.H[index],s.G @ w)*factor
        return cls(s.D[index]@w,c,response.gamma,s.metadata['noise_w'],s.cell_positions,
                   {'aperture_factor':factor,'precoder_power_w':float(np.sum(abs(w)**2)),
                    'precoder':'RZF on all-OFF reference effective channel, fixed for episode'})

    def field(self,bits):
        return self.base + np.einsum('...n,nkj->...kj',bits,self.delta)

    def full_field(self,bits):
        g=np.where(np.asarray(bits)[..., :,None,None],self.gamma[:,1,None,None],self.gamma[:,0,None,None])
        return self.direct + np.sum(g*self.contribution,axis=-3)

    def rates(self,u):
        p=np.abs(u)**2
        desired=np.diagonal(p,axis1=-2,axis2=-1)
        # Sum interferers directly avoids catastrophic cancellation at high SNR.
        off=1-np.eye(p.shape[-1])
        interference=(p*off).sum(axis=-1)
        return np.log1p(desired/(interference+self.noise_w))/np.log(2)

    def score_field(self,u):
        return self.rates(u).min(axis=-1)

    def score(self,bits):
        return self.score_field(self.field(bits))

    def flip_fields(self,bits,u=None):
        if u is None: u=self.field(bits)
        return u+(1-2*np.asarray(bits,dtype=float))[:,None,None]*self.delta

    def pair_fields(self,bits,u=None):
        if u is None: u=self.field(bits)
        d=(1-2*np.asarray(bits,dtype=float))[:,None,None]*self.delta
        return u+d[self.pairs[:,0]]+d[self.pairs[:,1]]

    def subspace_pair_fields(self,bits,u=None,subset=None):
        """Pair fields for a subset of atom indices, returning (fields, pair_indices)."""
        if u is None: u=self.field(bits)
        d=(1-2*np.asarray(bits,dtype=float))[:,None,None]*self.delta
        if subset is None:
            return u+d[self.pairs[:,0]]+d[self.pairs[:,1]],self.pairs
        sub=np.asarray(subset,dtype=np.intp)
        if len(sub)<2:
            return np.empty((0,u.shape[-2],u.shape[-1]),dtype=u.dtype),np.empty((0,2),dtype=int)
        i_idx,j_idx=np.triu_indices(len(sub),1)
        pairs=np.column_stack([sub[i_idx],sub[j_idx]])
        return u+d[pairs[:,0]]+d[pairs[:,1]],pairs

    def offspring_fields(self,base_bits,candidates,u=None):
        """Vectorized evaluation of candidates perturbed from base_bits."""
        if u is None: u=self.field(base_bits)
        cand=np.asarray(candidates,dtype=np.int8)
        if cand.ndim==1:
            cand=cand[None,:]
        diff=np.bitwise_xor(cand,np.asarray(base_bits,dtype=np.int8))
        d=(1-2*np.asarray(base_bits,dtype=float))[:,None,None]*self.delta
        delta_u=np.tensordot(diff.astype(d.dtype),d,axes=([1],[0]))
        return u+delta_u

    def features(self,bits,u=None,first=-1,memory_frequency=None,recent=None,gains=None):
        """Shared observation contract for MLP and attention; all CSI is exact here.

        Per-state normalization is analytic, never fitted on evaluation data.
        Pair conditioning keeps the ORIGINAL parent's objective/flip gains;
        first-marked token allows learning harmful-first but useful joint edits.
        """
        if u is None: u=self.field(bits)
        if gains is None:
            score=self.score_field(u)
            gains=self.score_field(self.flip_fields(bits,u))-score
        scale=max(float(np.linalg.norm(u)),np.sqrt(self.noise_w))
        c=self.contribution/scale
        d=self.delta/scale
        xyz=self.cell_positions-self.cell_positions.mean(0)
        xyz/=max(float(np.max(abs(xyz))),1e-9)
        g=self.gamma
        selected=(np.arange(self.n)==first).astype(float)
        mf=np.zeros(self.n) if memory_frequency is None else memory_frequency
        rec=np.zeros(self.n) if recent is None else recent
        global_rates=np.broadcast_to(self.rates(u),(self.n,u.shape[0]))
        x=np.column_stack([bits,xyz,g.real,g.imag,c.real.reshape(self.n,-1),c.imag.reshape(self.n,-1),
                           d.real.reshape(self.n,-1),d.imag.reshape(self.n,-1),
                           gains/(np.std(gains)+1e-6),selected,mf,rec,global_rates])
        return x.astype(np.float32),gains
