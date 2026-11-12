"""ME-GA / GA-RISC: a reimplementation from the published specification.

Source: F. Zardi, P. Oliveri, A. Massa, "Memory-Enhanced Dynamic Evolutionary Control of
Reconfigurable Intelligent Surfaces", IEEE TAP 2024 (arXiv 2309.04353), Sect. III, Eqs. (12)-(17)
and Fig. 2. This is NOT a reproduction: the paper does not publish code, the stagnation
threshold delta, the selection/crossover/mutation operators, the memory size or the memory
eviction rule. Every such choice is made in `MEGAParams`/this module and is recorded, with the
paper location and the reason, in docs/me-ga-reimplementation.md.

Methods
  me_ga        deadline-truncated ME-GA (best-so-far kept), memory operators on
  ga_risc      identical but the Learning/Remembering operators are off (never touches memory)
  me_ga_paper  full Gamma x V budget with the stagnation stop, NO deadline cut; true runtime recorded

Per-generation order (Fig. 2 and the text of Sect. III): selection+crossover+mutation (adaptive
rho, psi) -> fitness evaluation -> population replacement (nu) -> effectiveness self-evaluation
(Theta) -> Memory Learning (Theta >= 0) or Memory Remembering (Theta < 0) -> stop test.

Evaluation never returns a value that differs from a direct solve: the population is scored with
batched stacked linear solves (and, for children close to a base individual, a vectorised rank-m
Woodbury update); both are verified against `CoupledNetwork.state(bits).field` in the tests.
"""
from dataclasses import dataclass, replace
from time import perf_counter_ns
import numpy as np
from atomwise_ris.optimization.search import (SearchResult, Memory, ME_GA_METHODS,
                                              _score_completed_by_deadline)

_CALL_RESERVE_NS = 150_000   # fixed per-call overhead allowance in the deadline guard (small chunks cost more per individual)
_EPS_T = 1e-12   # floor on the worst-user rate before Phi = 1/T (guards a zero rate)


@dataclass(frozen=True)
class MEGAParams:
    """Paper values (Sect. IV) plus every unspecified choice, each marked in the doc table."""
    gamma: int = 100                 # population size (paper)
    V: int = 100                     # maximum iterations (paper)
    W: int = 3                       # observation window (paper)
    rho: tuple = (0.02, 0.06)        # mutation range (paper)
    psi: tuple = (0.6, 0.95)         # crossover range (paper)
    nu_max: float = 0.2              # replacement fraction (paper)
    kappa_max: float = 0.2           # restore probability (paper)
    beta_max: float = 0.2            # store probability (paper)
    sigma_max: float = 1.0           # paper; our sigma is the normalised binary variance, max = 1
    stagnation_delta: float | None = None  # Paper gives no value; None uses the published V cap only.
    capacity: int = 12               # ASSUMPTION: |Psi|, size-matched to the repo's `Memory`
    tournament: int = 2              # ASSUMPTION: binary tournament selection
    elitism: int = 0                 # ASSUMPTION: none (the paper lists no elitism)
    kappa_rule: str = 'magnitude'    # DEVIATION: see `restore_probability`
    theta_series: str = 'iteration'  # DEVIATION: Phi_best in Eq.15 = per-iteration best ('running' = literal TeX formula)
    stagnation_stop: bool = True
    ga_risc_replacement: bool = True # GA-RISC ablates only Psi; population replacement stays (ASSUMPTION)
    chunk: int = 25                  # deadline granularity of a population evaluation (engineering)
    woodbury_max_m: int = 24         # evaluation engineering only
    woodbury_min_children: int = 12  # evaluation engineering only


# ----------------------------------------------------------------------------------------------
# Population evaluators
# ----------------------------------------------------------------------------------------------
class SpecularPopulation:
    """Specular/separable evaluator: scoring a stack of bit vectors is one batched contraction."""
    def __init__(self, ev):
        self.ev = ev; self.n = ev.n; self.kind = 'specular'

    def score(self, bits, base=None):
        return np.asarray(self.ev.score(np.asarray(bits, dtype=np.int8)), dtype=float)


class CoupledPopulation:
    """Coupled multiport evaluator with batched stacked solves and an optional Woodbury path.

    `method`: 'solve' (batched LU, 2 right-hand sides; default engine), 'inverse' (stacked
    np.linalg.inv, the reference), 'woodbury' (rank-m update from a base state for children
    within `m_max` flips, solve for the rest) or 'auto' (Woodbury only when at least
    `min_children` children are close enough to amortise the base inverse).
    """
    def __init__(self, ev, method='auto', m_max=24, min_children=12):
        if method not in ('solve', 'inverse', 'woodbury', 'auto'):
            raise ValueError(method)
        self.ev = ev; self.n = ev.n; self.kind = 'coupled'; self.method = method
        self.m_max = int(m_max); self.min_children = int(min_children)
        self._ix = np.arange(self.n); self._base_key = None; self._base = None
        self.woodbury_children = 0; self.solved_children = 0

    def fields_solve(self, bits):
        ev = self.ev; b = np.asarray(bits, dtype=np.int64); g = len(b)
        m = np.broadcast_to(ev.a, (g, self.n, self.n)).copy()
        m[:, self._ix, self._ix] += ev.loads[b]
        x = np.linalg.solve(m, np.broadcast_to(ev.fw, (g,) + ev.fw.shape))
        return ev.dw + ev.c @ x

    def fields_inverse(self, bits):
        ev = self.ev; b = np.asarray(bits, dtype=np.int64); g = len(b)
        m = np.broadcast_to(ev.a, (g, self.n, self.n)).copy()
        m[:, self._ix, self._ix] += ev.loads[b]
        return ev.dw + ev.c @ np.linalg.inv(m) @ ev.fw

    def _base_state(self, base_bits):
        key = np.asarray(base_bits, dtype=np.int8).tobytes()
        if key != self._base_key:
            self._base = self.ev.state(base_bits); self._base_key = key
        return self._base

    def fields_woodbury(self, bits, base_bits):
        """Exact fields of `bits` from the base state (children differing from it in <= m_max atoms)."""
        ev = self.ev; b = np.asarray(bits, dtype=np.int8); b0 = np.asarray(base_bits, dtype=np.int8)
        st = self._base_state(b0); t = st.inverse
        diff = b != b0; m = diff.sum(1); g = len(b); mm = int(m.max()) if g else 0
        out = np.broadcast_to(st.field, (g,) + st.field.shape).copy()
        if mm == 0:
            return out
        order = np.argsort(~diff, axis=1, kind='stable')[:, :mm]
        valid = np.arange(mm)[None, :] < m[:, None]
        delta = ev.loads[1 - b0] - ev.loads[b0]
        l = ev.c @ t; r = t @ ev.fw
        s = np.zeros((g, mm, mm), dtype=complex)
        s += t[order[:, :, None], order[:, None, :]]
        s[:, np.arange(mm), np.arange(mm)] += 1 / delta[order]
        pair = valid[:, :, None] & valid[:, None, :]
        s = np.where(pair, s, np.eye(mm, dtype=complex)[None])      # identity on padded rows/cols: exact no-op
        rhs = np.where(valid[:, :, None], r[order], 0)
        lft = np.where(valid[None, :, :], l[:, order], 0)            # (k, g, mm)
        x = np.linalg.solve(s, rhs)
        return out - np.einsum('kgm,gmj->gkj', lft, x)

    def score(self, bits, base=None):
        b = np.asarray(bits, dtype=np.int8)
        if self.method == 'inverse':
            self.solved_children += len(b); return np.asarray(self.ev.score(self.fields_inverse(b)), dtype=float)
        if self.method == 'solve' or base is None or len(b) == 0:
            self.solved_children += len(b); return np.asarray(self.ev.score(self.fields_solve(b)), dtype=float)
        m = (b != np.asarray(base, dtype=np.int8)).sum(1)
        near = m <= self.m_max
        need = 1 if self.method == 'woodbury' else self.min_children
        if int(near.sum()) < need:
            self.solved_children += len(b); return np.asarray(self.ev.score(self.fields_solve(b)), dtype=float)
        out = np.empty(len(b), dtype=float)
        out[near] = self.ev.score(self.fields_woodbury(b[near], base))
        self.woodbury_children += int(near.sum())
        if (~near).any():
            out[~near] = self.ev.score(self.fields_solve(b[~near])); self.solved_children += int((~near).sum())
        return out


def make_population(ev, method='auto'):
    """Pick the population evaluator for either physics (duck-typed on the multiport attributes)."""
    if hasattr(ev, 'loads') and hasattr(ev, 'fw'):
        return CoupledPopulation(ev, method=method)
    return SpecularPopulation(ev)


# ----------------------------------------------------------------------------------------------
# Operators and indicators (each is a pure function so the tests can pin it down)
# ----------------------------------------------------------------------------------------------
def population_variance(pop):
    """Eq. (14) for binary genes, normalised: mean per-gene variance / 0.25, so sigma_max = 1."""
    p = np.asarray(pop, dtype=float).mean(axis=0)
    return float(np.mean(p * (1 - p)) / 0.25)


def adaptive_rates(sigma, p):
    """Eq. (13): rho_v falls and psi_v rises with population variance; both stay in range."""
    s = min(max(sigma / p.sigma_max, 0.0), 1.0)
    rho = p.rho[1] - (p.rho[1] - p.rho[0]) * s
    psi = p.psi[0] + (p.psi[1] - p.psi[0]) * s
    return rho, psi


def replacement_fraction(sigma, p):
    """nu_v = nu_max (1 - sigma_v / sigma_max)."""
    return p.nu_max * (1 - min(max(sigma / p.sigma_max, 0.0), 1.0))


def effectiveness(phi_series, v, W):
    """Eq. (15) on the per-iteration-best series (index 0 = initial population). The window is
    truncated to the available history for v < W (terms with v - k < 0 are dropped)."""
    return float(sum((phi_series[v] - phi_series[v - k]) / 2.0 ** k for k in range(1, min(W, v) + 1)))


def store_probability(theta, theta_max, p):
    """Eq. (16) for theta >= 0. theta <= theta_max, so the ratio is in [0, 1]; theta_max = 0 gives 0."""
    if theta_max <= 0:
        return 0.0
    return float(min(max(p.beta_max * theta / theta_max, 0.0), p.beta_max))


def restore_probability(theta, thetas, p):
    """Eq. (17) for theta < 0, with a DOCUMENTED DEVIATION. Read literally (theta_max = max theta over
    the window) the ratio is negative when any earlier theta was positive, exceeds 1 when all were
    negative, and is 0/0 at theta_max = 0, so the literal rule is not a probability. Rules:
      'magnitude'        kappa_max * |theta| / max|theta_seen|            (default; always in (0, kappa_max])
      'negative_extreme' kappa_max * theta / min(theta_seen)              (mirror of Eq. 17 for the negative branch)
      'literal_clipped'  kappa_max * theta / theta_max clipped to [0, kappa_max]; 0 unless theta_max < 0
    """
    seen = np.asarray(thetas, dtype=float)
    if p.kappa_rule == 'magnitude':
        ratio = abs(theta) / max(float(np.max(np.abs(seen))), 1e-300)
    elif p.kappa_rule == 'negative_extreme':
        ratio = theta / float(np.min(seen))
    elif p.kappa_rule == 'literal_clipped':
        tmax = float(np.max(seen)); ratio = theta / tmax if tmax < 0 else 0.0
    else:
        raise ValueError(p.kappa_rule)
    return float(min(max(p.kappa_max * ratio, 0.0), p.kappa_max))


def make_children(pop, scores, rho, psi, rng, tournament=2):
    """Binary-tournament selection, one-point crossover (prob. psi per pair), gene-wise bit-flip mutation (prob. rho)."""
    g, n = pop.shape
    k = max(2, int(tournament))
    cand = rng.integers(0, g, (g + (g % 2), k))
    win = cand[np.arange(len(cand)), np.argmax(scores[cand], axis=1)]
    par = pop[win]
    p1, p2 = par[0::2], par[1::2]
    cross = rng.random(len(p1)) < psi
    cut = rng.integers(1, max(n, 2), len(p1))
    swap = (np.arange(n)[None, :] >= cut[:, None]) & cross[:, None]
    kids = np.empty_like(par)
    kids[0::2] = np.where(swap, p2, p1); kids[1::2] = np.where(swap, p1, p2)
    kids = kids[:g]
    kids ^= (rng.random((g, n)) < rho).astype(kids.dtype)
    return kids


# ----------------------------------------------------------------------------------------------
# The controller
# ----------------------------------------------------------------------------------------------
def _phi(t):
    return 1.0 / max(float(t), _EPS_T)


def run_me_ga(popeval, seed, params=None, memory=None, use_memory=True, deadline_ns=None, start_ns=None):
    """One time step of ME-GA / GA-RISC. Returns a dict (bits, value, evaluations, trace, timings, info).

    `memory` is a `Memory` (mutated in place, so it persists across frames) or None. With
    `use_memory=False` the memory is neither read nor written. `deadline_ns=None` disables the cut.
    """
    p = params or MEGAParams()
    start = perf_counter_ns() if start_ns is None else start_ns
    n = popeval.n; G = int(p.gamma); rng = np.random.default_rng(seed)
    timings = {'setup': 0., 'features': 0., 'inference': 0., 'evaluation': 0., 'memory': 0., 'other': 0.}
    trace = []; info = {'generations': 0, 'theta': [], 'beta': [], 'kappa': [], 'event': [], 'sigma': [],
                        'rho': [], 'psi': [], 'nu_count': [], 'stop': None, 'psi_size_start': 0,
                        'psi_size_end': 0, 'stored': 0, 'restored': 0, 'chunks_discarded': 0}
    if use_memory and memory is None:
        # ME-GA still has a functioning within-step memory when callers do not
        # provide cross-frame state. The memory is local and discarded on return.
        memory = Memory(capacity=int(p.capacity))
    memory = memory if use_memory else None
    state = {'count': 0, 'best': None, 'bestv': -np.inf, 'per_ind': 0, 'late': False,
             'calibrated': False, 'single_ns': 0}
    known = {}                                     # Psi entry bytes -> rate on THIS time step's channel
    if memory is not None:
        info['psi_size_start'] = len(memory.patterns)

    def evaluate(bits, bucket, base=None, first_mandatory=False):
        """Score `bits` in chunks. A chunk finishing after the deadline is discarded, not counted."""
        bits = np.asarray(bits, dtype=np.int8); vals = np.empty(len(bits)); done = 0
        while done < len(bits):
            size = min(int(p.chunk), len(bits) - done)
            mandatory = first_mandatory and state['count'] == 0 and done == 0
            if mandatory:
                size = 1
            else:
                if not state['calibrated']:
                    size = min(size, 8)             # first chunk is small: it calibrates the per-individual cost
                if deadline_ns is not None:
                    # the single mandatory evaluation carries one-off overhead, so it only bounds an uncalibrated chunk
                    est = (int(state['per_ind'] * size * 1.5) if state['calibrated'] else 2 * state['single_ns']) + _CALL_RESERVE_NS
                    if perf_counter_ns() + est > deadline_ns:
                        state['late'] = True; break
            t = perf_counter_ns(); v = popeval.score(bits[done:done + size], base); ended = perf_counter_ns()
            timings[bucket] += (ended - t) / 1e6
            if mandatory:
                state['single_ns'] = ended - t
            elif size >= 8:
                state['per_ind'] = (ended - t) / size; state['calibrated'] = True
            if deadline_ns is not None and not mandatory and not _score_completed_by_deadline(ended, deadline_ns):
                state['late'] = True; info['chunks_discarded'] += 1; break
            vals[done:done + size] = v; state['count'] += size
            j = int(np.argmax(v))
            if v[j] > state['bestv']:
                state['bestv'] = float(v[j]); state['best'] = bits[done + j].copy()
                trace.append(((ended - start) / 1e6, state['bestv'], state['count']))
            done += size
        return vals[:done], done

    def finish():
        elapsed = (perf_counter_ns() - start) / 1e6
        timings['other'] = max(0., elapsed - sum(timings.values()))
        if memory is not None:
            info['psi_size_end'] = len(memory.patterns)
        info['evaluations'] = state['count']
        return {'bits': state['best'], 'value': state['bestv'], 'evaluations': state['count'], 'trace': trace,
                'timings': timings, 'info': info, 'elapsed_ms': elapsed}

    # Eq. (12): random initial population (memory does NOT seed it); its fitness is needed by selection.
    pop = rng.integers(0, 2, (G, n), dtype=np.int8)
    scores, done = evaluate(pop, 'evaluation', first_mandatory=True)
    if done < G:
        info['stop'] = 'deadline'; return finish()
    phi_it = [_phi(scores.max())]            # per-iteration best series, index 0 = S_0
    phi_run = [phi_it[0]]                    # running best (Phi_best of the TeX), for the stagnation test only
    thetas = []; sigma = population_variance(pop); ops_ns = 0

    for v in range(1, int(p.V) + 1):
        if deadline_ns is not None and perf_counter_ns() + int(ops_ns * 1.5) + int((state['per_ind'] if state['calibrated'] else state['single_ns']) * p.chunk * 1.5) + _CALL_RESERVE_NS > deadline_ns:
            info['stop'] = 'deadline'; break
        t0 = perf_counter_ns()
        rho, psi = adaptive_rates(sigma, p)                  # sigma of the population the operators act on
        kids = make_children(pop, scores, rho, psi, rng, p.tournament)
        e = int(p.elitism); base_idx = int(np.argmax(scores))
        base_bits = pop[base_idx].copy()
        if e > 0:
            elite = np.argsort(scores)[::-1][:e]; kids[:e] = pop[elite]
        ops_ns = max(ops_ns, perf_counter_ns() - t0)
        ks, done = evaluate(kids[e:], 'evaluation', base=base_bits)
        if done < len(kids) - e:
            info['stop'] = 'deadline'; break
        new_scores = np.empty(G); new_scores[e:] = ks
        if e > 0:
            new_scores[:e] = scores[elite]
        pop, scores = kids, new_scores
        info['generations'] = v
        jb = int(np.argmax(scores)); iter_best = pop[jb].copy(); iter_best_score = float(scores[jb])
        phi_it.append(_phi(iter_best_score)); phi_run.append(min(phi_run[-1], phi_it[-1]))
        sigma = population_variance(pop)                      # variance of S_v: drives nu_v
        info['sigma'].append(sigma); info['rho'].append(rho); info['psi'].append(psi)
        # Population replacement: the nu_v fraction with worst fitness is replaced by random individuals.
        k = int(round(replacement_fraction(sigma, p) * G))
        if use_memory is False and not p.ga_risc_replacement:
            k = 0
        info['nu_count'].append(k)
        if k > 0:
            worst = np.argsort(scores)[:k]; rnd = rng.integers(0, 2, (k, n), dtype=np.int8)
            rs, done = evaluate(rnd, 'evaluation')
            if done < k:
                info['stop'] = 'deadline'; break
            pop[worst] = rnd; scores[worst] = rs
        theta = effectiveness(phi_it if p.theta_series == 'iteration' else phi_run, v, int(p.W))
        thetas.append(theta); info['theta'].append(theta)
        u = rng.random()                                      # always drawn so ME-GA and GA-RISC share a stream
        beta = kappa = 0.0; event = None
        if memory is not None:
            tm = perf_counter_ns()
            if theta >= 0:                                    # Memory Learning
                beta = store_probability(theta, max(thetas), p)
                if u < beta:
                    memory.remember(iter_best); known[np.asarray(iter_best, dtype=np.int8).tobytes()] = iter_best_score
                    event = 'stored'; info['stored'] += 1
                timings['memory'] += (perf_counter_ns() - tm) / 1e6
            else:                                             # Memory Remembering
                kappa = restore_probability(theta, thetas, p)
                if u < kappa and len(memory.patterns) > 0:
                    unknown = [b for b in memory.patterns if np.asarray(b, dtype=np.int8).tobytes() not in known]
                    timings['memory'] += (perf_counter_ns() - tm) / 1e6
                    if unknown:   # rescoring on the current channel is required for a meaningful "best of Psi"
                        us, done = evaluate(np.stack(unknown), 'memory', base=iter_best)
                        for b, val in zip(unknown[:done], us):
                            known[np.asarray(b, dtype=np.int8).tobytes()] = float(val)
                        if done < len(unknown):
                            info['stop'] = 'deadline'; break
                    tm = perf_counter_ns()
                    vals = [known[np.asarray(b, dtype=np.int8).tobytes()] for b in memory.patterns]
                    bestm = memory.patterns[int(np.argmax(vals))]; bestm_val = float(max(vals))
                    wi = int(np.argmin(scores)); pop[wi] = bestm; scores[wi] = bestm_val
                    event = 'restored'; info['restored'] += 1
                    timings['memory'] += (perf_counter_ns() - tm) / 1e6
                else:
                    timings['memory'] += (perf_counter_ns() - tm) / 1e6
        info['beta'].append(beta); info['kappa'].append(kappa); info['event'].append(event)
        if p.stagnation_stop and p.stagnation_delta is not None and v >= p.W:
            window = np.mean([phi_run[v - j] for j in range(1, int(p.W) + 1)])
            if abs(phi_run[v] - window) < p.stagnation_delta:
                info['stop'] = 'stagnation'; break
    else:
        info['stop'] = 'V'
    return finish()


# ----------------------------------------------------------------------------------------------
# search(...) adapters with the conventions of search.py and coupled_search.py
# ----------------------------------------------------------------------------------------------
def _spec(method):
    if method not in ME_GA_METHODS:
        raise ValueError(f'Not an ME-GA method: {method}')
    return {'use_memory': method != 'ga_risc', 'enforce_deadline': method != 'me_ga_paper'}


def _as_memory(memory, capacity=12):
    """A `Memory` persists across frames (mutated in place); a plain list is a read-only local copy."""
    if isinstance(memory, Memory) or (memory is not None and hasattr(memory, 'patterns')):
        return memory
    mem = Memory(capacity=int(capacity))
    if memory is not None:
        mem.patterns = [np.asarray(b, dtype=np.int8).copy() for b in memory][:mem.capacity]
    return mem


def search_specular(ev, method, budget_ms, seed, initial=None, memory=None, params=None, evaluation='auto', info=None):
    """`initial` is ignored on purpose: Eq. (12) draws the initial population at random."""
    spec = _spec(method); start = perf_counter_ns()
    deadline = start + int(budget_ms * 1e6) if spec['enforce_deadline'] else None
    p = params or MEGAParams()
    out = run_me_ga(make_population(ev), seed, p, _as_memory(memory, p.capacity) if spec['use_memory'] else None,
                    spec['use_memory'], deadline, start)
    if info is not None:
        info.update(out['info'])
    missed = (out['elapsed_ms'] > budget_ms) if spec['enforce_deadline'] else None
    return SearchResult(out['bits'], out['value'], out['elapsed_ms'], out['evaluations'], out['trace'],
                        out['timings'], missed)


def search_coupled(ev, method, deadline, seed, memory=(), trace=None, params=None, evaluation='auto', info=None):
    """Returns (bits, value, evaluations, breakdown) like coupled_search.search; `deadline` is absolute ns."""
    spec = _spec(method); start = perf_counter_ns()
    p = params or MEGAParams()
    pop = make_population(ev, evaluation)
    out = run_me_ga(pop, seed, p, _as_memory(memory, p.capacity) if spec['use_memory'] else None, spec['use_memory'],
                    deadline if spec['enforce_deadline'] else None, start)
    if trace is not None:
        trace.extend(out['trace'])
    if info is not None:
        info.update(out['info'])
        if isinstance(pop, CoupledPopulation):
            info.update(woodbury_children=pop.woodbury_children, solved_children=pop.solved_children)
    t = out['timings']
    breakdown = {'features_ms': 0., 'inference_ms': 0., 'memory_ms': t['memory'],
                 'evaluation_ms': t['evaluation'], 'operators_ms': t['other']}
    return out['bits'].copy(), out['value'], out['evaluations'], breakdown


def per_evaluation_cost(popeval, rng_seed=0, sizes=(25, 100), reps=15, base=True):
    """Measured microseconds per individual for batch sizes `sizes` (median over `reps`)."""
    rng = np.random.default_rng(rng_seed); out = {}
    for size in sizes:
        bits = rng.integers(0, 2, (size, popeval.n), dtype=np.int8)
        ref = bits[0].copy(); bits[1:] = np.where(rng.random((size - 1, popeval.n)) < 0.1, 1 - ref, ref)
        popeval.score(bits, ref)
        ts = []
        for _ in range(reps):
            t = perf_counter_ns(); popeval.score(bits, ref); ts.append(perf_counter_ns() - t)
        out[size] = float(np.median(ts)) / size / 1e3
    return out
