"""Pair shortlist selectors for exact 2-flip search (Track 3 infrastructure).

A selector chooses which of the ``n(n-1)/2`` pairs of ``ev.pairs`` get an exact
rank-2 Woodbury score (``ev.moves_pairs``). A selector never returns a value or
a move: it returns pair *indices* into ``ev.pairs`` only, and every shortlisted
pair is rescored exactly by the evaluator before the search may use it.

Interface (all selectors)::

    select(ev, state, single_gains, k, rng) -> np.ndarray of unique int indices into ev.pairs

``single_gains`` is ``ev.moves(state) - current_value`` (length n), or ``None``
when the selector declares ``needs_gains = False``. ``rng`` is a
``numpy.random.Generator`` owned by the selector (never the search's own rng).

Everything here is numpy-only. The neural adapter takes an arbitrary numpy
scoring callable; weights are exported from torch elsewhere, never imported
here, so inference carries no framework overhead.

M0 (``analytic_topk``) formula
------------------------------
Notation for the current state: ``t`` cached inverse, ``l = C t`` (K x n),
``r = t F W`` (n x K), ``F`` the K x K field, ``den_i = 1/delta_i + t_ii``,
``det_ij = den_i den_j - t_ij t_ji``. A single flip changes the field by
``-l_i r_i / den_i`` and a pair flip by ``-(l_i u_i + l_j u_j)`` with
``u_i = (den_j r_i - t_ij r_j)/det_ij``, ``u_j = (den_i r_j - t_ji r_i)/det_ij``.

The score is ``S = min_k log2(1 + SINR_k)``. Let ``k*`` be the minimum user and
``G_m = 2 c_m conj(F[k*, m])`` the Wirtinger gradient of ``S`` with respect to
that row of ``F`` (``c_m = dS/dP_m``; the desired stream has weight
``1/(ln2 (1+sinr) D)``, interferers ``-sig/(ln2 (1+sinr) D^2)``, ``D = noise +
interference``). With ``h_i = sum_m G_m r[i, m]`` and ``l_i = l[k*, i]`` the
linearised score change is

    lin_i  = -Re[ l_i h_i / den_i ]
    lin_ij = -Re[ (l_i (den_j h_i - t_ij h_j) + l_j (den_i h_j - t_ji h_i)) / det_ij ]

and the M0 priority is the exact single-flip gains plus the linearised pair
interaction correction (which is exactly zero when ``t_ij = 0``):

    m0_ij = g_i + g_j + (lin_ij - lin_i - lin_j)

It is O(n^2) elementwise with no K x K field per pair. It is a heuristic
ranking only; the winner is always rescored exactly.
"""
from dataclasses import dataclass

import numpy as np

_LN2 = float(np.log(2.0))


def _topk_indices(scores, k):
    """Unique indices of the k largest scores, sorted ascending (deterministic)."""
    p = len(scores)
    if k >= p:
        return np.arange(p)
    return np.sort(np.argpartition(-np.asarray(scores), k - 1)[:k])


def _state_arrays(ev, state):
    t = state.inverse
    l = ev.c @ t
    r = t @ ev.fw
    delta = ev.loads[1 - state.bits] - ev.loads[state.bits]
    den = 1.0 / delta + np.diag(t)
    return t, l, r, den


def m0_priority(ev, state, single_gains):
    """Return the M0 priority of every pair in ``ev.pairs`` (see module docstring)."""
    t, l, r, den = _state_arrays(ev, state)
    field = np.asarray(state.field)
    power = np.abs(field) ** 2
    signal = np.diagonal(power).copy()
    interference = power.sum(1) - signal
    denom = float(ev.noise) + interference
    sinr = signal / denom
    ks = int(np.argmin(np.log2(1.0 + sinr)))
    c_sig = 1.0 / (_LN2 * (1.0 + sinr[ks]) * denom[ks])
    c_int = -signal[ks] / (_LN2 * (1.0 + sinr[ks]) * denom[ks] ** 2)
    c = np.full(field.shape[1], c_int)
    c[ks] = c_sig
    grad = 2.0 * c * np.conj(field[ks])
    h = r @ grad                       # (n,)
    lk = l[ks]                         # (n,)
    lin_single = -np.real(lk * h / den)
    i, j = ev.pairs.T
    det = den[i] * den[j] - t[i, j] * t[j, i]
    lin_pair = -np.real((lk[i] * (den[j] * h[i] - t[i, j] * h[j])
                         + lk[j] * (den[i] * h[j] - t[j, i] * h[i])) / det)
    g = np.asarray(single_gains, dtype=float)
    return g[i] + g[j] + (lin_pair - lin_single[i] - lin_single[j])


def analytic_topk(ev, state, single_gains, k, rng=None):
    """M0: top-k pairs by exact single gains plus a linearised interaction term from ``t``."""
    return _topk_indices(m0_priority(ev, state, single_gains), int(k))


analytic_topk.needs_gains = True
analytic_topk.name = "m0"


def random_topk(ev, state, single_gains, k, rng):
    """Uniform random shortlist of k distinct pairs."""
    p = len(ev.pairs)
    if k >= p:
        return np.arange(p)
    return np.sort(rng.choice(p, size=int(k), replace=False))


random_topk.needs_gains = False
random_topk.name = "rand"


@dataclass
class PairFeatures:
    """Atom-level and pair-level (permutation-symmetric) inputs for a pair model."""
    atom: np.ndarray    # (n, D) per-atom features x_i
    edge: np.ndarray    # (P, E) symmetric pair features e_ij
    pairs: np.ndarray   # (P, 2) indices into the atoms, i < j

    def flat(self):
        """(P, 3D+E): [x_i + x_j, |x_i - x_j|, x_i * x_j, e_ij]; invariant to swapping i and j."""
        a, b = self.atom[self.pairs[:, 0]], self.atom[self.pairs[:, 1]]
        return np.concatenate([a + b, np.abs(a - b), a * b, self.edge], axis=1)


_STATIC_FEATURE_CACHE = {}


def _zrr(ev):
    z = getattr(ev, "z", None)
    return np.asarray(z)[:ev.n, :ev.n] if z is not None else np.asarray(ev.a)


def _cached_grid_coords(n, w):
    key = ("grid", int(n), int(w))
    coords = _STATIC_FEATURE_CACHE.get(key)
    if coords is None:
        idx = np.arange(n, dtype=np.float32)
        coords = (
            ((idx % w) / max(w - 1, 1)).astype(np.float32),
            ((idx // w) / max((n - 1) // w, 1)).astype(np.float32),
        )
        if len(_STATIC_FEATURE_CACHE) > 64:
            _STATIC_FEATURE_CACHE.clear()
        _STATIC_FEATURE_CACHE[key] = coords
    return coords


def atom_features(ev, state, single_gains, grid_width=8):
    """Per-atom features: bit, grid x/y, single gain, t_ii (re/im), |t_i.| row sum, l and r (re/im).

    ``grid_width`` is the number of cells per row used to place atom ``a`` at
    ``(a % w, a // w)``, matching ``CoupledNetwork.features``.
    """
    n = ev.n
    t = state.inverse
    l = ev.c @ t
    r = t @ ev.fw
    gx, gy = _cached_grid_coords(n, int(grid_width))
    diag_t = np.diagonal(t)
    abs_t = np.abs(t)
    interaction = abs_t.sum(1) - np.diagonal(abs_t)
    k = l.shape[0]
    x = np.empty((n, 7 + 4 * k), dtype=np.float32)
    x[:, 0] = state.bits
    x[:, 1] = gx
    x[:, 2] = gy
    x[:, 3] = single_gains
    x[:, 4] = diag_t.real
    x[:, 5] = diag_t.imag
    x[:, 6] = interaction
    x[:, 7:7 + k] = l.real.T
    x[:, 7 + k:7 + 2 * k] = l.imag.T
    x[:, 7 + 2 * k:7 + 3 * k] = r.real
    x[:, 7 + 3 * k:7 + 4 * k] = r.imag
    return x


def pair_features(ev, state, single_gains, grid_width=8, pairs=None):
    """Vectorised symmetric pair features. Never uses the true pair gain.

    ``e_ij`` = [Re Z_RR[i,j], Im Z_RR[i,j], |Z_RR[i,j]|, element distance (cells),
    g_i + g_j, g_i * g_j, Re t_ij, Im t_ij], with ``t_ij`` symmetrised as
    ``(t_ij + t_ji)/2`` and ``Z_RR[i,j]`` read as ``(Z_ij + Z_ji)/2``.
    """
    w = int(grid_width)
    n = ev.n
    t = state.inverse
    g = np.asarray(single_gains, dtype=np.float32)
    raw_z = getattr(ev, "z", None)
    if pairs is None and raw_z is not None:
        ckey = ("pf", id(raw_z), n, w)
        cached = _STATIC_FEATURE_CACHE.get(ckey)
        if cached is None or cached[0] is not raw_z:
            pr = ev.pairs
            i = np.ascontiguousarray(pr[:, 0])
            j = np.ascontiguousarray(pr[:, 1])
            flat_ij = (i * n + j).astype(np.intp)
            flat_ji = (j * n + i).astype(np.intp)
            z = _zrr(ev)
            zij = 0.5 * (z[i, j] + z[j, i])
            dist = np.hypot((i % w) - (j % w), (i // w) - (j // w))
            static4 = np.column_stack([zij.real, zij.imag, np.abs(zij), dist]).astype(np.float32)
            cached = (raw_z, pr, i, j, flat_ij, flat_ji, static4)
            if len(_STATIC_FEATURE_CACHE) > 64:
                _STATIC_FEATURE_CACHE.clear()
            _STATIC_FEATURE_CACHE[ckey] = cached
        _, pr, i, j, flat_ij, flat_ji, static4 = cached
        edge = np.empty((len(pr), 8), dtype=np.float32)
        edge[:, :4] = static4
        gi, gj = g[i], g[j]
        edge[:, 4] = gi + gj
        edge[:, 5] = gi * gj
        t_flat = t.reshape(-1)
        tij = 0.5 * (t_flat[flat_ij] + t_flat[flat_ji])
        edge[:, 6] = tij.real
        edge[:, 7] = tij.imag
        return PairFeatures(atom_features(ev, state, g, w), edge, pr)

    pr = ev.pairs if pairs is None else np.asarray(pairs, dtype=int)
    i, j = pr[:, 0], pr[:, 1]
    z = _zrr(ev)
    zij = 0.5 * (z[i, j] + z[j, i])
    tij = 0.5 * (t[i, j] + t[j, i])
    dist = np.hypot((i % w) - (j % w), (i // w) - (j // w))
    edge = np.column_stack([zij.real, zij.imag, np.abs(zij), dist, g[i] + g[j], g[i] * g[j],
                            tij.real, tij.imag]).astype(np.float32)
    return PairFeatures(atom_features(ev, state, g, w), edge, pr)


class neural_topk:
    """Adapter from a numpy scoring callable to the selector interface.

    ``score_fn(PairFeatures) -> (P,)`` array; the top-k by score are shortlisted.
    ``score_fn`` must be pure numpy (e.g. :class:`NumpyPairMLP`).
    """
    needs_gains = True
    name = "neural"

    def __init__(self, score_fn, grid_width=8):
        self.score_fn = score_fn
        self.grid_width = int(grid_width)

    def __call__(self, ev, state, single_gains, k, rng=None):
        s = np.asarray(self.score_fn(pair_features(ev, state, single_gains, self.grid_width)), dtype=float)
        if s.shape != (len(ev.pairs),):
            raise ValueError("neural scorer must return one score per pair")
        return _topk_indices(s, int(k))


class CascadeTopK:
    """M0 shortlist followed by neural reranking of only the shortlisted pairs.

    ``shortlist`` controls M0 candidates and ``k`` controls exact pair scans.
    The returned indices always refer to the original ``ev.pairs`` array.
    """
    needs_gains = True
    name = "m0_mlp_cascade"

    def __init__(self, score_fn, shortlist=256, grid_width=8):
        self.score_fn = score_fn
        self.shortlist = int(shortlist)
        self.grid_width = int(grid_width)
        if self.shortlist < 1:
            raise ValueError("shortlist must be positive")

    def __call__(self, ev, state, single_gains, k, rng=None):
        cand = analytic_topk(ev, state, single_gains, min(self.shortlist, len(ev.pairs)))
        pf = pair_features(ev, state, single_gains, self.grid_width, ev.pairs[cand])
        scores = np.asarray(self.score_fn(pf), dtype=float)
        if scores.shape != (len(cand),):
            raise ValueError("cascade scorer must return one score per shortlisted pair")
        return np.sort(cand[_topk_indices(scores, min(int(k), len(cand)))])


def _cached_triu_flat_indices(n):
    key = ("triu_flat", int(n))
    flat_ij = _STATIC_FEATURE_CACHE.get(key)
    if flat_ij is None:
        i, j = np.triu_indices(int(n), k=1)
        flat_ij = (i * int(n) + j).astype(np.intp)
        if len(_STATIC_FEATURE_CACHE) > 64:
            _STATIC_FEATURE_CACHE.clear()
        _STATIC_FEATURE_CACHE[key] = flat_ij
    return flat_ij


class NumpyFactoredPairHead:
    """Symmetric factored pair scorer: u_i dot v_j + u_j dot v_i + edge bias.

    Parameters are an exported atom projection (with optional 1-layer per-element
    GELU encoder and skip projection) and symmetric edge weights. All-pairs
    interaction is evaluated as one N x N matrix product plus upper-triangle gather.
    """
    def __init__(
        self,
        atom_w,
        edge_w,
        bias=0.0,
        *,
        v_w=None,
        atom_mean=None,
        atom_scale=None,
        edge_mean=None,
        edge_scale=None,
        enc_w=None,
        enc_b=None,
        u_skip_w=None,
        v_skip_w=None,
        u_re_w=None,
        v_re_w=None,
        u_im_w=None,
        v_im_w=None,
        y_mean=0.0,
        y_scale=1.0,
    ):
        self.atom_w = np.asarray(atom_w, dtype=np.float32)
        self.v_w = None if v_w is None else np.asarray(v_w, dtype=np.float32)
        self.edge_w = np.asarray(edge_w, dtype=np.float32).reshape(-1)
        self.bias = float(bias)
        if self.atom_w.ndim != 2 or self.edge_w.ndim != 1:
            raise ValueError("invalid factored-head dimensions")
        if self.v_w is not None and self.v_w.shape != self.atom_w.shape:
            raise ValueError("v_w shape must match atom_w shape")
        self.atom_mean = None if atom_mean is None else np.asarray(atom_mean, dtype=np.float32)
        self.atom_scale = None if atom_scale is None else np.asarray(atom_scale, dtype=np.float32)
        self.atom_inv_scale = None if self.atom_scale is None else (1.0 / self.atom_scale).astype(np.float32)
        self.edge_mean = None if edge_mean is None else np.asarray(edge_mean, dtype=np.float32).reshape(-1)
        self.edge_scale = None if edge_scale is None else np.asarray(edge_scale, dtype=np.float32).reshape(-1)
        self.enc_w = None if enc_w is None else np.asarray(enc_w, dtype=np.float32)
        self.enc_b = None if enc_b is None else np.asarray(enc_b, dtype=np.float32)
        self.u_skip_w = None if u_skip_w is None else np.asarray(u_skip_w, dtype=np.float32)
        self.v_skip_w = None if v_skip_w is None else np.asarray(v_skip_w, dtype=np.float32)
        self.u_re_w = None if u_re_w is None else np.asarray(u_re_w, dtype=np.float32)
        self.v_re_w = None if v_re_w is None else np.asarray(v_re_w, dtype=np.float32)
        self.u_im_w = None if u_im_w is None else np.asarray(u_im_w, dtype=np.float32)
        self.v_im_w = None if v_im_w is None else np.asarray(v_im_w, dtype=np.float32)
        self.y_mean = float(y_mean)
        self.y_scale = float(y_scale)
        # Fold linear edge standardization into effective edge weights/bias for zero-copy runtime dot product.
        if self.edge_mean is not None and self.edge_scale is not None:
            self._edge_w_eff = (self.edge_w / self.edge_scale).astype(np.float32)
            self._bias_eff = float(self.bias - np.dot(self.edge_mean / self.edge_scale, self.edge_w))
            self._t_re_mean = float(self.edge_mean[6]) if self.edge_mean.size > 6 else 0.0
            self._t_re_inv = float(1.0 / self.edge_scale[6]) if self.edge_scale.size > 6 else 1.0
            self._t_im_mean = float(self.edge_mean[7]) if self.edge_mean.size > 7 else 0.0
            self._t_im_inv = float(1.0 / self.edge_scale[7]) if self.edge_scale.size > 7 else 1.0
        else:
            self._edge_w_eff = self.edge_w
            self._bias_eff = self.bias
            self._t_re_mean = 0.0
            self._t_re_inv = 1.0
            self._t_im_mean = 0.0
            self._t_im_inv = 1.0

    @classmethod
    def random(cls, atom_dim, edge_dim, rank=16, enc_dim=0, seed=0):
        rng = np.random.default_rng(seed)
        if int(enc_dim) > 0:
            enc_w = rng.normal(0, atom_dim ** -0.5, (atom_dim, int(enc_dim))).astype(np.float32)
            enc_b = np.zeros(int(enc_dim), dtype=np.float32)
            u_w = rng.normal(0, int(enc_dim) ** -0.5, (int(enc_dim), int(rank))).astype(np.float32)
            v_w = rng.normal(0, int(enc_dim) ** -0.5, (int(enc_dim), int(rank))).astype(np.float32)
            return cls(u_w, np.zeros(edge_dim, dtype=np.float32), 0.0, v_w=v_w, enc_w=enc_w, enc_b=enc_b)
        u_w = rng.normal(0, atom_dim ** -0.5, (atom_dim, int(rank))).astype(np.float32)
        v_w = rng.normal(0, atom_dim ** -0.5, (atom_dim, int(rank))).astype(np.float32)
        return cls(u_w, np.zeros(edge_dim, dtype=np.float32), 0.0, v_w=v_w)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        kw = {}
        for key in (
            "v_w",
            "atom_mean",
            "atom_scale",
            "edge_mean",
            "edge_scale",
            "enc_w",
            "enc_b",
            "u_skip_w",
            "v_skip_w",
            "u_re_w",
            "v_re_w",
            "u_im_w",
            "v_im_w",
        ):
            if key in d.files:
                kw[key] = d[key]
        if "y_mean" in d.files:
            kw["y_mean"] = float(d["y_mean"])
        if "y_scale" in d.files:
            kw["y_scale"] = float(d["y_scale"])
        return cls(d["atom_w"], d["edge_w"], float(d["bias"]), **kw)

    def __call__(self, pf):
        in_atom_dim = self.enc_w.shape[0] if self.enc_w is not None else self.atom_w.shape[0]
        if in_atom_dim != pf.atom.shape[1] or self.edge_w.shape[0] != pf.edge.shape[1]:
            raise ValueError("factored-head input dimensions do not match features")
        a = pf.atom
        if self.atom_mean is not None:
            a = np.clip((a - self.atom_mean) * self.atom_inv_scale, -10.0, 10.0)
        if self.enc_w is not None:
            h = _gelu(a @ self.enc_w + self.enc_b)
        else:
            h = a
        u = h @ self.atom_w
        if self.u_skip_w is not None:
            u = u + a @ self.u_skip_w
        if self.v_w is None:
            m = u @ u.T
        else:
            v = h @ self.v_w
            if self.v_skip_w is not None:
                v = v + a @ self.v_skip_w
            m = u @ v.T
            m = m + m.T

        n = pf.atom.shape[0]
        pairs = pf.pairs
        is_canonical_triu = (
            len(pairs) == n * (n - 1) // 2
            and len(pairs) > 0
            and int(pairs[0, 0]) == 0
            and int(pairs[0, 1]) == 1
            and int(pairs[-1, 0]) == n - 2
            and int(pairs[-1, 1]) == n - 1
        )
        if is_canonical_triu:
            flat_ij = _cached_triu_flat_indices(n)
            pair_score = m.ravel()[flat_ij]
        else:
            i, j = pairs[:, 0], pairs[:, 1]
            pair_score = m[i, j]

        if self.u_re_w is not None and self.v_re_w is not None:
            ur = h @ self.u_re_w
            vr = h @ self.v_re_w
            mr = ur @ vr.T
            mr = mr + mr.T
            r_pair = mr.ravel()[flat_ij] if is_canonical_triu else mr[pairs[:, 0], pairs[:, 1]]
            e_re = (pf.edge[:, 6] - self._t_re_mean) * self._t_re_inv
            pair_score = pair_score + r_pair * e_re
        if self.u_im_w is not None and self.v_im_w is not None:
            ui = h @ self.u_im_w
            vi = h @ self.v_im_w
            mi = ui @ vi.T
            mi = mi + mi.T
            im_pair = mi.ravel()[flat_ij] if is_canonical_triu else mi[pairs[:, 0], pairs[:, 1]]
            e_im = (pf.edge[:, 7] - self._t_im_mean) * self._t_im_inv
            pair_score = pair_score + im_pair * e_im

        raw = pair_score + (pf.edge @ self._edge_w_eff + self._bias_eff)
        if self.y_scale != 1.0 or self.y_mean != 0.0:
            return raw * self.y_scale + self.y_mean
        return raw


def exact_pair_labels(ev, bits):
    """Return exact per-pair gain Δij and synergy Sij from a parent bit pattern.

    Sij = Δij - Δi - Δj, where all gains are relative to the same parent.
    Uses the evaluator's exact ``moves`` and ``moves_pairs`` Woodbury paths.
    """
    state = ev.state(np.asarray(bits, dtype=np.int8))
    base = float(ev.score(state.field))
    singles = np.asarray(ev.moves(state), dtype=float) - base
    pair_values = np.asarray(ev.moves_pairs(state, ev.pairs), dtype=float)
    if pair_values.shape != (len(ev.pairs),):
        raise ValueError("moves_pairs returned unexpected shape")
    i, j = ev.pairs.T
    gains = pair_values - base
    return {"bits": np.asarray(state.bits, dtype=np.int8), "base": base,
            "single_gains": singles, "pair_gains": gains,
            "synergy": gains - singles[i] - singles[j],
            "pairs": np.asarray(ev.pairs, dtype=np.int32)}


SELECTORS = {"m0": analytic_topk, "rand": random_topk}


# ---- numpy inference models (weights exported from torch; random init for cost measurement) ----

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class NumpyPairMLP:
    """Symmetric pairwise MLP over ``PairFeatures.flat()`` (hidden GELU layers, scalar output)."""

    def __init__(self, layers):
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.x_mean = self.x_scale = None
        self.y_mean, self.y_scale = 0.0, 1.0

    @classmethod
    def random(cls, in_dim, hidden=(64, 64), seed=0):
        rng = np.random.default_rng(seed)
        dims = [in_dim, *hidden, 1]
        return cls([(rng.normal(0, dims[a] ** -0.5, (dims[a], dims[a + 1])), np.zeros(dims[a + 1]))
                    for a in range(len(dims) - 1)])

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        if "activation" in d and str(d["activation"]) != "gelu_tanh_approx":
            raise ValueError("unsupported exported MLP activation")
        n = len([k for k in d.files if k.startswith("w")])
        model = cls([(d[f"w{a}"], d[f"b{a}"]) for a in range(n)])
        if "x_mean" in d:
            model.x_mean = np.asarray(d["x_mean"], np.float32)
            model.x_scale = np.asarray(d["x_scale"], np.float32)
            model.y_mean, model.y_scale = float(d["y_mean"]), float(d["y_scale"])
        return model

    def __call__(self, pf):
        x = pf.flat() if isinstance(pf, PairFeatures) else np.asarray(pf, np.float32)
        if self.x_mean is not None:
            x = np.clip((x - self.x_mean) / self.x_scale, -10, 10)
        for a, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if a < len(self.layers) - 1:
                x = _gelu(x)
        return x[:, 0] * self.y_scale + self.y_mean


class NumpyAttentionPairScorer:
    """Post-norm transformer encoder over atoms plus a symmetric pair head over all pairs."""

    def __init__(self, params):
        self.p = {k: ([{kk: np.asarray(vv, np.float32) for kk, vv in layer.items()} for layer in v]
                      if k == "layers" else
                      (np.asarray(v, np.float32) if k != "heads" else int(v))) for k, v in params.items()}

    @classmethod
    def random(cls, atom_dim, edge_dim, d=32, heads=4, layers=2, ff=64, head_hidden=64, seed=0):
        rng = np.random.default_rng(seed)

        def mat(a, b):
            return rng.normal(0, a ** -0.5, (a, b))
        enc = [dict(wq=mat(d, d), wk=mat(d, d), wv=mat(d, d), wo=mat(d, d), bo=np.zeros(d),
                    g1=np.ones(d), b1=np.zeros(d), f1=mat(d, ff), fb1=np.zeros(ff),
                    f2=mat(ff, d), fb2=np.zeros(d), g2=np.ones(d), b2=np.zeros(d)) for _ in range(layers)]
        return cls(dict(heads=heads, win=mat(atom_dim, d), bin=np.zeros(d), layers=enc,
                        h1=mat(3 * d + edge_dim, head_hidden), hb1=np.zeros(head_hidden),
                        h2=mat(head_hidden, 1), hb2=np.zeros(1)))

    def encode(self, atom):
        p = self.p
        x = np.asarray(atom, np.float32) @ p["win"] + p["bin"]
        n, d = x.shape
        h = p["heads"]
        dh = d // h
        for L in p["layers"]:
            q = (x @ L["wq"]).reshape(n, h, dh).transpose(1, 0, 2)
            k = (x @ L["wk"]).reshape(n, h, dh).transpose(1, 0, 2)
            v = (x @ L["wv"]).reshape(n, h, dh).transpose(1, 0, 2)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(dh).astype(np.float32)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = (att @ v).transpose(1, 0, 2).reshape(n, d) @ L["wo"] + L["bo"]
            x = _layer_norm(x + o, L["g1"], L["b1"])
            x = _layer_norm(x + _gelu(x @ L["f1"] + L["fb1"]) @ L["f2"] + L["fb2"], L["g2"], L["b2"])
        return x

    def __call__(self, pf):
        p = self.p
        hid = self.encode(pf.atom)
        a, b = hid[pf.pairs[:, 0]], hid[pf.pairs[:, 1]]
        z = np.concatenate([a + b, np.abs(a - b), a * b, pf.edge], axis=1)
        return (_gelu(z @ p["h1"] + p["hb1"]) @ p["h2"] + p["hb2"])[:, 0]
