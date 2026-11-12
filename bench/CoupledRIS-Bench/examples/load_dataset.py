"""Small reference loader for a packaged CoupledRIS-Bench pilot.

Example::

    ev = load_network("outputs/.../channels/channel_realizations.h5", "munich/64/case-1/0")
    rate = rate_for_bits(ev, bits)

The pilot package stores labels separately; ``verify_label_rate`` performs a
direct score readback against the parent-state rate in
optimization/single_flip_gains.h5.
"""
from pathlib import Path

import h5py
import numpy as np

from atomwise_ris.hybrid.operator import HybridCoupledNetwork


def _as_complex(ds):
    a = np.asarray(ds[()])
    if np.iscomplexobj(a): return a.astype(np.complex128, copy=False)
    if a.dtype.fields and {"r", "i"}.issubset(a.dtype.fields): return a["r"] + 1j * a["i"]
    if a.ndim and a.shape[-1] == 2: return a[..., 0] + 1j * a[..., 1]
    raise ValueError(f"{ds.name}: unsupported complex encoding")


def load_network(channel_h5, channel_key):
    """Load the self-contained hybrid operator group into HybridCoupledNetwork."""
    def _read_from_group(g):
        z = _as_complex(g["Z_effective"])
        ci, co, hd = (_as_complex(g[k]) for k in ("C_in", "C_out", "H_direct"))
        loads = _as_complex(g["loads"])
        if loads.shape == (z.shape[0], 2): loads = loads[0]
        if loads.shape != (2,): raise ValueError("loads must contain OFF/ON complex impedances")
        w = _as_complex(g["precoder"])
        noise = float(g.attrs["noise"])
        return HybridCoupledNetwork(z, co, ci, hd, loads, w=w, noise=noise)

    if isinstance(channel_h5, (h5py.File, h5py.Group)):
        return _read_from_group(channel_h5[str(channel_key).strip("/")])
    with h5py.File(channel_h5, "r") as f:
        return _read_from_group(f[str(channel_key).strip("/")])


def rate_for_bits(channel_h5, channel_key, bits):
    ev = load_network(channel_h5, channel_key)
    b = np.asarray(bits, dtype=np.int8)
    if b.shape != (ev.n,) or np.any((b != 0) & (b != 1)):
        raise ValueError(f"bits must be a binary vector with shape ({ev.n},)")
    return float(ev.score(ev.state(b).field))


def verify_label_rate(channel_h5, channel_key, optimization_dir, parent="all_off", atol=1e-10):
    """Recompute and compare one stored base rate; raise on tolerance failure."""
    ev = load_network(channel_h5, channel_key)
    labels_h5 = Path(optimization_dir) / "single_flip_gains.h5" if Path(optimization_dir).is_dir() else Path(optimization_dir)
    with h5py.File(labels_h5, "r") as f:
        g = f[f"{str(channel_key).strip('/')}/{parent}"]
        bits, expected = g["bits"][()], float(g.attrs["base_rate_bps_hz"])
    actual = float(ev.score(ev.state(bits).field))
    error = abs(actual - expected)
    if error > atol: raise AssertionError(f"rate round-trip error {error:.3g} exceeds {atol:g}")
    return {"expected": expected, "actual": actual, "absolute_error": error, "atol": atol}
