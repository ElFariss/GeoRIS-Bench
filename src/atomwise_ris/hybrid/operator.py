"""Factored hybrid channel operator and fast rank-1 / rank-2 state updates.

The hybrid channel operator evaluates:
    H_eff(b) = H_direct + C_out @ inv(Z_RIS + Z_L(b)) @ C_in
where:
    Z_RIS: (N, N) multiport impedance matrix of the coupled RIS array
    Z_L(b): diag(loads[b]), state-dependent load impedances
    C_in:  (N, M) incident excitation operator from M transmit sources
    C_out: (K, N) reradiation operator to K receive users
    H_direct: (K, M) direct transmission matrix (RIS-independent)

This class implements exact Woodbury rank-1 and rank-2 updates, allowing
O(N) single-cell evaluation and O(N^2) pairwise evaluation without
re-inverting the N x N impedance matrix.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from atomwise_ris.channels.precoding import precoder

@dataclass
class HybridState:
    """State of the RIS network for a specific binary configuration."""
    bits: np.ndarray
    inverse: np.ndarray
    field: np.ndarray

class HybridCoupledNetwork:
    """Factored hybrid electromagnetic network evaluator.
    
    Parameters
    ----------
    z_ris : np.ndarray
        (N, N) complex impedance matrix of the RIS.
    c_out : np.ndarray
        (K, N) reradiation / receive operator.
    c_in : np.ndarray
        (N, M) excitation / incident operator.
    h_direct : np.ndarray
        (K, M) direct path channel.
    loads : np.ndarray
        Complex load impedances for states 0 and 1, e.g. [z0, z1].
    w : Optional[np.ndarray]
        (M, K) precoding matrix. If None, regularized zero forcing is computed
        from the all-OFF (zeros) channel.
    noise : Optional[float]
        Noise power in Watts. Default is 290K, 1MHz, 7dB NF.
    """
    def __init__(
        self,
        z_ris: np.ndarray,
        c_out: np.ndarray,
        c_in: np.ndarray,
        h_direct: np.ndarray,
        loads: np.ndarray,
        w: Optional[np.ndarray] = None,
        noise: Optional[float] = None,
    ):
        self.a = np.asarray(z_ris, dtype=complex).copy()
        self.c = np.asarray(c_out, dtype=complex).copy()
        self.f = np.asarray(c_in, dtype=complex).copy()
        self.d = np.asarray(h_direct, dtype=complex).copy()
        self.loads = np.asarray(loads, dtype=complex).copy()
        self.n = len(self.a)
        
        # Dimensions
        self.k = self.c.shape[0] # number of user receive streams
        self.m = self.f.shape[1] # number of BS transmit antennas
        
        # Reference precoder
        self.w = precoder(self.channel(np.zeros(self.n, dtype=np.int8))) if w is None else np.asarray(w, dtype=complex)
        self.fw = self.f @ self.w
        self.dw = self.d @ self.w
        
        # Thermal noise: 290 K, 1 MHz, 7 dB noise figure = k_B * T * B * 10^(0.7)
        self.noise = 1.380649e-23 * 290.0 * 1e6 * (10.0 ** 0.7) if noise is None else float(noise)
        self.pairs = np.array(np.triu_indices(self.n, 1)).T

    def channel(self, bits: np.ndarray) -> np.ndarray:
        """Compute the K x M channel matrix for given binary bits."""
        b = np.asarray(bits, dtype=int)
        zl = np.diag(self.loads[b])
        return self.d + self.c @ np.linalg.solve(self.a + zl, self.f)

    def state(self, bits: np.ndarray) -> HybridState:
        """Compute full state including inverse and precoded stream field."""
        b = np.asarray(bits, dtype=np.int8).copy()
        zl = np.diag(self.loads[b])
        inv = np.linalg.inv(self.a + zl)
        field = self.dw + self.c @ inv @ self.fw
        return HybridState(b, inv, field)

    def score(self, fields: np.ndarray) -> np.ndarray:
        """Compute minimum user rate (bps/Hz) under fixed precoder."""
        power = np.abs(fields) ** 2
        signal = np.diagonal(power, axis1=-2, axis2=-1)
        off = 1.0 - np.eye(power.shape[-1])
        interference = (power * off).sum(axis=-1)
        sinr = signal / (self.noise + interference)
        return np.log2(1.0 + sinr).min(-1)

    def moves(self, state: HybridState, pairs: bool = False, subset: Optional[np.ndarray] = None) -> np.ndarray:
        """Rank-1 or rank-2 Woodbury evaluation of all single or pairwise bit flips."""
        t = state.inverse
        l = self.c @ t
        r = t @ self.fw
        delta = self.loads[1 - state.bits] - self.loads[state.bits]
        den = 1.0 / delta + np.diag(t)
        
        if not pairs:
            if subset is None:
                fields = state.field[None] - np.einsum('ki,ij->ikj', l, r) / den[:, None, None]
            else:
                idx = np.asarray(subset, dtype=int)
                fields = state.field[None] - np.einsum('ki,ij->ikj', l[:, idx], r[idx]) / den[idx, None, None]
        else:
            if subset is None:
                i, j = self.pairs.T
            else:
                sub = np.asarray(subset, dtype=int)
                idx_i, idx_j = np.triu_indices(len(sub), 1)
                i, j = sub[idx_i], sub[idx_j]
            det = den[i] * den[j] - t[i, j] * t[j, i]
            ui = (den[j, None] * r[i] - t[i, j, None] * r[j]) / det[:, None]
            uj = (den[i, None] * r[j] - t[j, i, None] * r[i]) / det[:, None]
            fields = state.field[None] - l[:, i].T[:, :, None] * ui[:, None, :] - l[:, j].T[:, :, None] * uj[:, None, :]
        return np.asarray(self.score(fields))

    def moves_pairs(self, state: HybridState, pairs: np.ndarray) -> np.ndarray:
        """Exact rank-2 Woodbury scores for an explicit (m, 2) array of distinct-cell pairs."""
        p = np.asarray(pairs, dtype=int)
        if p.size == 0:
            return np.empty(0, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('pairs must have shape (m,2)')
        if p.min() < 0 or p.max() >= self.n or np.any(p[:, 0] == p[:, 1]):
            raise ValueError('pairs need two distinct in-range cell indices')
        i, j = p[:, 0], p[:, 1]
        t = state.inverse
        l = self.c @ t
        r = t @ self.fw
        delta = self.loads[1 - state.bits] - self.loads[state.bits]
        den = 1.0 / delta + np.diag(t)
        det = den[i] * den[j] - t[i, j] * t[j, i]
        ui = (den[j, None] * r[i] - t[i, j, None] * r[j]) / det[:, None]
        uj = (den[i, None] * r[j] - t[j, i, None] * r[i]) / det[:, None]
        fields = state.field[None] - l[:, i].T[:, :, None] * ui[:, None, :] - l[:, j].T[:, :, None] * uj[:, None, :]
        return np.asarray(self.score(fields))

    def flip(self, state: HybridState, indices: Union[int, list, np.ndarray]) -> HybridState:
        """Perform exact Sherman-Morrison-Woodbury rank-k update for bit flip(s)."""
        ix = np.asarray(indices, dtype=int).reshape(-1)
        if len(ix) == 0:
            return state
        if len(set(ix.tolist())) != len(ix):
            raise ValueError('Distinct flips required')
        delta = self.loads[1 - state.bits[ix]] - self.loads[state.bits[ix]]
        small = np.diag(1.0 / delta) + state.inverse[np.ix_(ix, ix)]
        inv = state.inverse - state.inverse[:, ix] @ np.linalg.solve(small, state.inverse[ix])
        b = state.bits.copy()
        b[ix] = 1 - b[ix]
        return HybridState(b, inv, self.dw + self.c @ inv @ self.fw)

    @classmethod
    def from_monolithic_z(cls, z: np.ndarray, loads: np.ndarray, n_ris: int, n_bs: int = 4, n_ue: int = 2, w=None):
        """Construct hybrid network from a monolithic (N+6) x (N+6) impedance matrix."""
        e = z[n_ris:, n_ris:] + 50.0 * np.eye(n_bs + n_ue)
        ei = np.linalg.inv(e)
        a = z[:n_ris, :n_ris] - z[:n_ris, n_ris:] @ ei @ z[n_ris:, :n_ris]
        c = -100.0 * (ei @ z[n_ris:, :n_ris])[-n_ue:]
        f = (z[:n_ris, n_ris:] @ ei)[:, :n_bs]
        d = -100.0 * ei[-n_ue:, :n_bs]
        return cls(a, c, f, d, loads, w=w)
