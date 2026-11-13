"""NSGA-II for binary RIS configuration: a deadline-aware evolutionary reference baseline.

Source: K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective genetic
algorithm: NSGA-II", IEEE Trans. Evol. Comput. 6(2):182-197, 2002, doi 10.1109/4235.996017.
Design and hyper-parameter rule: docs/nsga2-baseline-20261113.md.

Formulation
  chromosome   b in {0,1}^N (one bit per RIS atom)
  objectives   the K per-user rates R_k(b), all maximised, under the SAME fixed precoder and the SAME
               physics as every other method (the evaluator's own `fw`/`dw`/`c`/`loads`/`noise`)
  returned     the best max-min individual (largest min_k R_k) seen in ANY evaluation of the call
               (anytime: the population front is the search state, the reported action is max-min)

Algorithm (Sect. III of the paper): P_0 -> fast non-dominated sort + crowding distance -> per
generation: binary tournament with the crowded-comparison operator, crossover (prob. pc; uniform or
two-point), gene-wise bit-flip mutation (rate 1/N), evaluate Q_t, R_t = P_t u Q_t, elitist
environmental selection by fronts, the last front truncated by crowding distance.

Initial population: the all-OFF state plus P-1 uniformly random individuals (`seed_all_off=True`).
NOTE: the ME-GA reimplementation (me_ga.py) draws its initial population purely at random (Eq. 12,
no all-OFF seed); the all-OFF seed here follows the NSGA-II brief of 2026-11-13. The all-OFF
individual is also the single "mandatory" first evaluation, exactly like ME-GA's first individual.

Evaluation reuses ME-GA's `CoupledPopulation` (batched stacked solves, and for children within
`m_max` flips of the current best a vectorised rank-m Woodbury update); `CoupledRatePopulation`
only changes the final reduction from min_k to the full per-user vector. min_k of these rates is
bit-identical to `ev.score(fields)` (same operations), verified in tests/test_nsga2.py.

Deadline: ME-GA's chunked guard with amendment A1 (adaptive chunk size, conservative max per-individual
cost x 1.5 + call reserve; a chunk that completes after the deadline is discarded; best-so-far is kept).
Existing methods are untouched: nothing here is imported by search.py / coupled_search.py.
"""
from dataclasses import dataclass, asdict
from time import perf_counter_ns
import numpy as np
from atomwise_ris.optimization.me_ga import CoupledPopulation, SpecularPopulation
from atomwise_ris.optimization.search import SearchResult, _score_completed_by_deadline

NSGA2_METHODS = ('nsga2',)
_CALL_RESERVE_NS = 150_000     # same per-call overhead allowance as me_ga.py


@dataclass(frozen=True)
class NSGA2Params:
    population: int = 50             # P (even); chosen by the declared rule, see CHOSEN_PARAMS
    crossover_prob: float = 0.9      # pc; chosen by the declared rule
    crossover: str = 'uniform'       # fixed a priori: 'uniform' (gene swap prob. 0.5) | 'two_point'
    mutation_rate: float | None = None   # None -> 1/N (Deb et al. 2002, binary-coded runs)
    seed_all_off: bool = True        # all-OFF + (P-1) random
    max_generations: int | None = None   # None: until the deadline (no deadline -> 100)
    dedup: bool = True               # identical genomes are scored once per call (exact reuse)
    chunk: int = 25                  # deadline granularity (same as MEGAParams.chunk)
    woodbury_max_m: int = 24         # same as MEGAParams
    woodbury_min_children: int = 12  # same as MEGAParams


# Hyper-parameters locked by the rule of docs/nsga2-baseline-20261113.md (free-space dev only).
# Benchmarks call nsga2 with these values; changing them requires a new dated declaration.
CHOSEN_PARAMS = NSGA2Params(population=20, crossover_prob=0.9)   # rule s.3 on v1 train, 2026-11-13 (tuning_selection.json)


# ----------------------------------------------------------------------------------------------
# Per-user rate evaluators
# ----------------------------------------------------------------------------------------------
def user_rates(ev, fields):
    """Per-user rates with exactly the operations of CoupledNetwork.score / HybridCoupledNetwork.score
    without the final min, so `user_rates(ev, f).min(-1) == ev.score(f)` bit for bit."""
    power = np.abs(fields) ** 2
    signal = np.diagonal(power, axis1=-2, axis2=-1)
    off = 1.0 - np.eye(power.shape[-1])
    interference = (power * off).sum(axis=-1)
    return np.log2(1.0 + signal / (ev.noise + interference))


class CoupledRatePopulation(CoupledPopulation):
    """ME-GA's coupled population evaluator, returning the (g, K) per-user rate matrix."""

    def rates(self, bits, base=None):
        b = np.asarray(bits, dtype=np.int8); ev = self.ev
        if self.method == 'inverse':
            self.solved_children += len(b); return user_rates(ev, self.fields_inverse(b))
        if self.method == 'solve' or base is None or len(b) == 0:
            self.solved_children += len(b); return user_rates(ev, self.fields_solve(b))
        m = (b != np.asarray(base, dtype=np.int8)).sum(1)
        near = m <= self.m_max
        need = 1 if self.method == 'woodbury' else self.min_children
        if int(near.sum()) < need:
            self.solved_children += len(b); return user_rates(ev, self.fields_solve(b))
        k = ev.c.shape[0]
        out = np.empty((len(b), k), dtype=float)
        out[near] = user_rates(ev, self.fields_woodbury(b[near], base))
        self.woodbury_children += int(near.sum())
        if (~near).any():
            out[~near] = user_rates(ev, self.fields_solve(b[~near])); self.solved_children += int((~near).sum())
        return out


class SpecularRatePopulation(SpecularPopulation):
    """Specular/separable evaluator: per-user rates from the evaluator's own `rates`."""

    def rates(self, bits, base=None):
        return np.asarray(self.ev.rates(self.ev.field(np.asarray(bits, dtype=np.int8))), dtype=float)


def make_rate_population(ev, method='auto', params=None):
    p = params or CHOSEN_PARAMS
    if hasattr(ev, 'loads') and hasattr(ev, 'fw'):
        return CoupledRatePopulation(ev, method=method, m_max=p.woodbury_max_m, min_children=p.woodbury_min_children)
    return SpecularRatePopulation(ev)


# ----------------------------------------------------------------------------------------------
# NSGA-II operators (pure functions; pinned down by the tests). All objectives are MAXIMISED.
# ----------------------------------------------------------------------------------------------
def dominates_matrix(f):
    """D[i, j] = individual i Pareto-dominates individual j (>= in every objective, > in one)."""
    f = np.asarray(f, dtype=float)
    ge = (f[:, None, :] >= f[None, :, :]).all(-1)
    gt = (f[:, None, :] > f[None, :, :]).any(-1)
    return ge & gt


def fast_non_dominated_sort(f):
    """Deb et al. (2002) fast non-dominated sort, O(M n^2). Returns (fronts list of index arrays, rank)."""
    f = np.asarray(f, dtype=float); n = len(f)
    if n == 0:
        return [], np.empty(0, dtype=int)
    d = dominates_matrix(f)
    count = d.sum(0).astype(np.int64)          # n_p: number of individuals dominating p
    rank = np.full(n, -1, dtype=int); fronts = []
    current = np.flatnonzero(count == 0); r = 0
    while current.size:
        rank[current] = r; fronts.append(current)
        count = count - d[current].sum(0)
        current = np.flatnonzero((count == 0) & (rank < 0)); r += 1
    return fronts, rank


def crowding_distance(f):
    """Crowding distance of one front (Deb et al. 2002, Fig. 4): boundary points get +inf."""
    f = np.asarray(f, dtype=float); n, k = f.shape
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf; return dist
    for m in range(k):
        o = np.argsort(f[:, m], kind='stable'); fm = f[o, m]; span = fm[-1] - fm[0]
        dist[o[0]] = np.inf; dist[o[-1]] = np.inf
        if span > 0:
            dist[o[1:-1]] += (fm[2:] - fm[:-2]) / span
    return dist


def environmental_selection(f, size):
    """Elitist truncation of R_t to `size`: whole fronts, the last one by descending crowding distance.
    Returns (selected indices, their rank, their crowding distance computed on their full front)."""
    fronts, _ = fast_non_dominated_sort(f)
    sel, rk, cd = [], [], []
    for r, fr in enumerate(fronts):
        c = crowding_distance(f[fr]); room = size - len(sel)
        if room <= 0:
            break
        if len(fr) <= room:
            sel.extend(fr.tolist()); rk.extend([r] * len(fr)); cd.extend(c.tolist())
        else:
            o = np.argsort(-c, kind='stable')[:room]
            sel.extend(fr[o].tolist()); rk.extend([r] * room); cd.extend(c[o].tolist())
    return np.asarray(sel, dtype=int), np.asarray(rk, dtype=int), np.asarray(cd, dtype=float)


def crowded_tournament(rank, crowd, rng, count):
    """Binary tournament with the crowded-comparison operator (lower rank, then larger crowding; coin on ties)."""
    n = len(rank)
    a = rng.integers(0, n, count); b = rng.integers(0, n, count); coin = rng.random(count) < 0.5
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    return np.where(better | (tie & coin), a, b)


def make_offspring(pop, rank, crowd, rng, pc, kind='uniform', pm=None):
    """|P| children: crowded tournament, crossover with probability pc, then bit-flip mutation (rate pm)."""
    g, n = pop.shape
    pm = 1.0 / n if pm is None else float(pm)
    par = pop[crowded_tournament(rank, crowd, rng, g + (g % 2))]
    p1, p2 = par[0::2], par[1::2]; h = len(p1)
    cross = rng.random(h) < pc
    if kind == 'uniform':
        mask = rng.random((h, n)) < 0.5
    elif kind == 'two_point':
        cuts = np.sort(rng.integers(0, n + 1, (h, 2)), axis=1)
        ix = np.arange(n)[None, :]
        mask = (ix >= cuts[:, :1]) & (ix < cuts[:, 1:])
    else:
        raise ValueError(kind)
    mask &= cross[:, None]
    kids = np.empty_like(par)
    kids[0::2] = np.where(mask, p2, p1); kids[1::2] = np.where(mask, p1, p2)
    kids = kids[:g]
    kids ^= (rng.random((g, n)) < pm).astype(kids.dtype)
    return kids


# ----------------------------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------------------------
def run_nsga2(popeval, seed, params=None, deadline_ns=None, start_ns=None):
    """One NSGA-II search. Returns a dict (bits, value, evaluations, trace, timings, info, elapsed_ms).

    `value` = min_k R_k(bits) of the best max-min individual evaluated; `deadline_ns=None` disables the cut.
    """
    p = params or CHOSEN_PARAMS
    start = perf_counter_ns() if start_ns is None else start_ns
    n = popeval.n; P = int(p.population)
    if P < 4 or P % 2:
        raise ValueError('population must be even and >= 4')
    pm = 1.0 / n if p.mutation_rate is None else float(p.mutation_rate)
    max_gen = p.max_generations if p.max_generations is not None else (None if deadline_ns is not None else 100)
    rng = np.random.default_rng(seed)
    timings = {'setup': 0., 'evaluation': 0., 'operators': 0., 'selection': 0., 'other': 0.}
    trace = []
    info = {'generations': 0, 'stop': None, 'chunks_discarded': 0, 'duplicates_reused': 0,
            'front0_size': [], 'params': asdict(p)}
    state = {'count': 0, 'best': None, 'bestv': -np.inf, 'per_ind': 0, 'calibrated': False, 'single_ns': 0}
    cache = {}
    # A2 end reserve: max(0.3 ms, 1 % of the budget) -- the 0.3 ms floor is search.policy_reserve_ns's
    reserve_ns = max(300_000, int(0.01 * (deadline_ns - start))) if deadline_ns is not None else 0
    inv_ns = [0]

    def prepare_base(kids, base):
        """A2: compute the Woodbury base inverse explicitly, under the guard; None = solve path only."""
        if not hasattr(popeval, '_base_state') or popeval.method not in ('auto', 'woodbury'):
            return base
        if int(((kids != base).sum(1) <= popeval.m_max).sum()) < (1 if popeval.method == 'woodbury' else popeval.min_children):
            return base                                       # auto mode will solve anyway; no inverse needed
        if np.asarray(base, dtype=np.int8).tobytes() == popeval._base_key:
            return base
        if deadline_ns is not None:
            est = inv_ns[0] if inv_ns[0] else 4 * state['single_ns']
            if perf_counter_ns() + int(1.5 * est) + reserve_ns > deadline_ns:
                return None
        t = perf_counter_ns(); popeval._base_state(base); dt = perf_counter_ns() - t
        inv_ns[0] = max(inv_ns[0], dt); timings['evaluation'] += dt / 1e6
        return base

    def evaluate_unique(bits, base=None, first_mandatory=False):
        """Score rows in chunks; returns (rates[:done], done).

        Deadline guard (amendment A1 of docs/nsga2-baseline-20261113.md): the per-individual cost estimate is
        the single mandatory evaluation's cost until calibrated, then the MAX per-individual cost seen over
        chunks of >= 4 individuals (solve and Woodbury chunks differ); the chunk is shrunk to what fits in the
        remaining time at 1.5 x that estimate plus the call reserve. A chunk ending after the deadline is
        discarded and not counted (as in ME-GA)."""
        bits = np.asarray(bits, dtype=np.int8); out = None; done = 0
        while done < len(bits):
            size = min(int(p.chunk), len(bits) - done)
            mandatory = first_mandatory and state['count'] == 0 and done == 0
            if mandatory:
                size = 1
            else:
                if not state['calibrated']:
                    size = min(size, 8)
                if deadline_ns is not None:
                    # A2: est(size) = 1.5 * per * size + single-call cost + end reserve
                    per = state['per_ind'] if state['calibrated'] else state['single_ns']
                    room = deadline_ns - perf_counter_ns() - state['single_ns'] - reserve_ns
                    fit = int(room // max(1.5 * per, 1.0))
                    if fit < 1:
                        break
                    size = min(size, fit)
            t = perf_counter_ns(); r = popeval.rates(bits[done:done + size], base); ended = perf_counter_ns()
            timings['evaluation'] += (ended - t) / 1e6
            if mandatory:
                state['single_ns'] = ended - t
            elif size >= 4:
                cost = (ended - t) / size
                state['per_ind'] = cost if not state['calibrated'] else max(state['per_ind'], cost)
                state['calibrated'] = True
            if deadline_ns is not None and not mandatory and not _score_completed_by_deadline(ended, deadline_ns):
                info['chunks_discarded'] += 1; break
            if out is None:
                out = np.empty((len(bits), r.shape[1]))
            out[done:done + size] = r; state['count'] += size
            mm = r.min(1); j = int(np.argmax(mm))
            if mm[j] > state['bestv']:
                state['bestv'] = float(mm[j]); state['best'] = bits[done + j].copy()
                trace.append(((ended - start) / 1e6, state['bestv'], state['count']))
            done += size
        return (out[:done] if out is not None else np.empty((0, 0))), done

    def evaluate(bits, base=None, first_mandatory=False):
        """Exact reuse of already-scored genomes (dedup), then chunked scoring of the new ones.
        Returns (rates of all rows, complete flag)."""
        bits = np.asarray(bits, dtype=np.int8)
        if not p.dedup:
            r, done = evaluate_unique(bits, base, first_mandatory)
            return r, done == len(bits)
        keys = [row.tobytes() for row in bits]
        todo, seen = [], set()
        for i, kk in enumerate(keys):
            if kk not in cache and kk not in seen:
                todo.append(i); seen.add(kk)
        info['duplicates_reused'] += len(bits) - len(todo)
        if todo:
            r, done = evaluate_unique(bits[todo], base, first_mandatory)
            for i, row in zip(todo[:done], r):
                cache[keys[i]] = row
            if done < len(todo):
                return None, False
        return np.stack([cache[kk] for kk in keys]), True

    def finish():
        elapsed = (perf_counter_ns() - start) / 1e6
        timings['other'] = max(0., elapsed - sum(timings.values()))
        info['evaluations'] = state['count']
        return {'bits': state['best'], 'value': state['bestv'], 'evaluations': state['count'], 'trace': trace,
                'timings': timings, 'info': info, 'elapsed_ms': elapsed}

    pop = rng.integers(0, 2, (P, n), dtype=np.int8)
    if p.seed_all_off:
        pop[0] = 0
    f, ok = evaluate(pop, first_mandatory=True)
    if not ok:
        info['stop'] = 'deadline'; return finish()
    t = perf_counter_ns()
    sel, rank, crowd = environmental_selection(f, P)
    pop, f = pop[sel], f[sel]
    sel_ns = perf_counter_ns() - t; timings['selection'] += sel_ns / 1e6
    ops_ns = 0
    gen = 0
    while max_gen is None or gen < max_gen:
        if deadline_ns is not None:
            per = state['per_ind'] if state['calibrated'] else state['single_ns']
            if perf_counter_ns() + int(ops_ns * 1.5) + int(per * 1.5) + state['single_ns'] + reserve_ns > deadline_ns:   # >= 1 child fits (A1/A2)
                info['stop'] = 'deadline'; break
        t0 = perf_counter_ns()
        kids = make_offspring(pop, rank, crowd, rng, p.crossover_prob, p.crossover, pm)
        base = pop[int(np.argmax(f.min(1)))].copy()          # Woodbury base: current best max-min member
        ops = perf_counter_ns() - t0; ops_ns = max(ops_ns, ops); timings['operators'] += ops / 1e6
        base = prepare_base(kids, base)
        kf, ok = evaluate(kids, base)
        if not ok:
            info['stop'] = 'deadline'; break
        if deadline_ns is not None and perf_counter_ns() + int(sel_ns * 1.5) + reserve_ns > deadline_ns:
            info['stop'] = 'deadline'; break                  # children already counted in best-so-far
        t = perf_counter_ns()
        allp = np.concatenate([pop, kids]); allf = np.concatenate([f, kf])
        sel, rank, crowd = environmental_selection(allf, P)
        pop, f = allp[sel], allf[sel]
        s = perf_counter_ns() - t; sel_ns = max(sel_ns, s); timings['selection'] += s / 1e6
        gen += 1; info['generations'] = gen; info['front0_size'].append(int((rank == 0).sum()))
    else:
        info['stop'] = 'max_generations'
    return finish()


# ----------------------------------------------------------------------------------------------
# search(...) adapters with the conventions of me_ga.search_coupled / me_ga.search_specular
# ----------------------------------------------------------------------------------------------
def _check(method):
    if method not in NSGA2_METHODS:
        raise ValueError(f'Not an NSGA-II method: {method}')


def search_coupled(ev, method, deadline, seed, memory=(), trace=None, params=None, evaluation='auto', info=None):
    """Returns (bits, value, evaluations, breakdown) like coupled_search.search; `deadline` is absolute ns.
    `memory` is accepted for call compatibility and ignored (NSGA-II has no cross-frame memory)."""
    _check(method); start = perf_counter_ns()
    p = params or CHOSEN_PARAMS
    pop = make_rate_population(ev, evaluation, p)
    out = run_nsga2(pop, seed, p, deadline, start)
    if trace is not None:
        trace.extend(out['trace'])
    if info is not None:
        info.update(out['info'])
        if isinstance(pop, CoupledPopulation):
            info.update(woodbury_children=pop.woodbury_children, solved_children=pop.solved_children)
    t = out['timings']
    breakdown = {'features_ms': 0., 'inference_ms': 0., 'memory_ms': 0., 'evaluation_ms': t['evaluation'],
                 'operators_ms': t['operators'] + t['selection'] + t['other'], 'selection_ms': t['selection'],
                 'generations': out['info']['generations'], 'stop': out['info']['stop']}
    return out['bits'].copy(), out['value'], out['evaluations'], breakdown


def search_specular(ev, method, budget_ms, seed, initial=None, memory=None, params=None, evaluation='auto', info=None):
    """SearchResult like me_ga.search_specular; `initial`/`memory` ignored."""
    _check(method); start = perf_counter_ns()
    p = params or CHOSEN_PARAMS
    out = run_nsga2(make_rate_population(ev, evaluation, p), seed, p, start + int(budget_ms * 1e6), start)
    if info is not None:
        info.update(out['info'])
    return SearchResult(out['bits'], out['value'], out['elapsed_ms'], out['evaluations'], out['trace'],
                        out['timings'], out['elapsed_ms'] > budget_ms)
