"""Anytime, deadline-aware pair-search ablation ladder for coupled RIS control.

The public method IDs are ``pair_plus_l1`` through ``pair_plus_l6``. Each level
adds exactly one declared mechanism to its predecessor. ``NO_L2_METHODS`` is a
separate development round: each adds one mechanism to a base without L2
screening, with the base's scan overhead reduced. ``FIXED_L2_METHODS`` add a
like-for-like restart screen to that same base. The module is kept
independent of the matched-v1 dispatcher so existing controllers remain
byte-for-byte unchanged.
"""
from bisect import insort
from collections import deque
from time import perf_counter_ns

import numpy as np

from atomwise_ris.optimization.pair_select import CascadeTopK, SELECTORS
from atomwise_ris.optimization.search import _score_completed_by_deadline


METHODS = tuple(f"pair_plus_l{i}" for i in range(1, 7))
NO_L2_METHODS = ("pair_plus_b0", "pair_plus_b3", "pair_plus_b4", "pair_plus_b4o", "pair_plus_b5", "pair_plus_b6")
# Share of earlier restarts' single-flip values a restart must reach to earn pair search.
FIXED_L2_METHODS = {"pair_plus_b2": 0.50, "pair_plus_b2t": 0.75}
# b0 with both components that beat it in the no-L2 round: restart density and recombination.
COMBINED_METHODS = ("pair_plus_b35",)
# b35 with an exact-pair shortlist: the selector ranks pairs, only shortlisted pairs are scored
# (exactly, by ``ev.moves_pairs``), and a full scan runs only if no shortlisted pair improves.
SELECTOR_SPECS = {**{f"pair_plus_b35_m0_k{k}": ("m0", k) for k in (8, 32, 128, 512)},
                  **{f"pair_plus_b35_rand_k{k}": ("rand", k) for k in (32, 128)}}
SELECTOR_METHODS = tuple(SELECTOR_SPECS)
M1_METHODS = tuple(f"pair_plus_b35_m1_k{k}" for k in (8, 32, 128))
ALL_METHODS = METHODS + NO_L2_METHODS + tuple(FIXED_L2_METHODS) + COMBINED_METHODS + SELECTOR_METHODS + M1_METHODS


def _spec(method):
    """Component flags. Levels are cumulative; ``b`` methods add one component to b0."""
    if method in METHODS:
        level = int(method.rsplit("l", 1)[1])
        return dict(fast=False, screen=level >= 2, density=level >= 3, tie_break=level >= 4, escape=level >= 4,
                    escape_first_basin_only=False, recombine=level >= 5, memory=level >= 6, screen_quantile=None)
    if method in NO_L2_METHODS:
        name = method.rsplit("_", 1)[1]
        return dict(fast=True, screen=False, density=name == "b3", tie_break=name in ("b4", "b4o"),
                    escape=name in ("b4", "b4o"), escape_first_basin_only=name == "b4o",
                    recombine=name == "b5", memory=name == "b6", screen_quantile=None)
    if method in FIXED_L2_METHODS:
        return dict(fast=True, screen=False, density=False, tie_break=False, escape=False,
                    escape_first_basin_only=False, recombine=False, memory=False,
                    screen_quantile=FIXED_L2_METHODS[method])
    if method in COMBINED_METHODS:
        return dict(fast=True, screen=False, density=True, tie_break=False, escape=False,
                    escape_first_basin_only=False, recombine=True, memory=False, screen_quantile=None)
    if method in SELECTOR_SPECS:
        return dict(_spec("pair_plus_b35"), selector=SELECTOR_SPECS[method])
    if method in M1_METHODS:
        return dict(_spec("pair_plus_b35"), selector=("m1cascade", int(method.rsplit("k", 1)[1])))
    raise ValueError(f"unknown pair-plus method: {method!r}")
_SCREEN_MARGIN = 0.50
_TABU_TENURE = 5
_TABU_ESCAPE_LIMIT = 8
_MEMORY_CAPACITY = 4
_MEMORY_MAX_FLIPS = 20


class PairPlusMemory:
    """Small trajectory-scoped elite store for L6.

    Values are intentionally not carried across frames: every pattern is
    rescored against the current frame before it can seed search.
    """

    def __init__(self, capacity=_MEMORY_CAPACITY):
        self.capacity = int(capacity)
        self.trajectory = None
        self.patterns = []

    def enter(self, trajectory):
        if trajectory != self.trajectory:
            self.patterns = []
            self.trajectory = trajectory

    def remember(self, bits, value=None):
        b = np.asarray(bits, dtype=np.int8).reshape(-1).copy()
        if b.size == 0 or not np.isin(b, (0, 1)).all():
            raise ValueError("memory patterns must be nonempty binary vectors")
        self.patterns = [p for p in self.patterns if not np.array_equal(p, b)]
        self.patterns.insert(0, b)
        self.patterns = self.patterns[:self.capacity]


def _user2_rate(ev, field):
    """Return the second-user rate when the evaluator exposes the RIS 2x2 link."""
    f = np.asarray(field)
    if not hasattr(ev, "noise") or f.ndim != 2 or f.shape[0] < 2 or f.shape[1] < 2:
        return None
    power = np.abs(f) ** 2
    desired = float(power[1, 1])
    interference = float(np.sum(power[1]) - desired)
    return float(np.log2(1.0 + desired / (float(ev.noise) + interference)))


def _lex_better(value, second, incumbent, incumbent_second, tie_break):
    if value > incumbent + 1e-12:
        return True
    return bool(tie_break and abs(value - incumbent) <= 1e-12
                and second is not None
                and (incumbent_second is None or second > incumbent_second + 1e-12))


def _score_state(ev, bits):
    state = ev.state(np.asarray(bits, dtype=np.int8))
    return state, float(ev.score(state.field))


def _elite_recombine(elites, rng, n):
    """Uniformly recombine two distinct cached local optima."""
    if len(elites) < 2:
        raise ValueError("elite recombination needs two distinct parents")
    i, j = rng.choice(len(elites), size=2, replace=False)
    mask = rng.random(n) < 0.5
    return np.where(mask, elites[i][0], elites[j][0]).astype(np.int8)


def search(ev, method, deadline_ns, seed, memory=None, trace=None, parent_recorder=None,
           selector_model=None, selector_shortlist=256):
    """Run one ladder level.

    Parameters
    ----------
    ev : CoupledNetwork-compatible object
        Must implement ``state``, ``score``, ``moves``, ``moves_subspace`` and
        ``flip``. ``moves(state, True)`` returns all pair-toggle scores.
    method : str
        One of :data:`ALL_METHODS`.
    deadline_ns : int
        Absolute ``perf_counter_ns`` deadline.
    seed : int
        Per-frame deterministic seed.
    memory : PairPlusMemory or iterable, optional
        Used only by the memory methods (L6, b6). They read at most four patterns and only applies
        rank-k updates of at most twenty bit flips.
    trace : list, optional
        Appended with ``(elapsed_ms, best_min_rate, evaluations)`` improvements.

    Returns ``(bits, value, evaluations, breakdown)``. A late candidate scan or
    completed-action verification is never admitted as a search improvement.
    """
    spec = _spec(method)
    start = perf_counter_ns()
    rng = np.random.default_rng(int(seed))
    trace = [] if trace is None else trace

    def emit_parent(state, category):
        if parent_recorder is not None:
            parent_recorder(np.asarray(state.bits, dtype=np.int8).copy(), category)

    n = int(ev.n)
    pair_count = n * (n - 1) // 2
    tie_break = spec["tie_break"]
    fast = spec["fast"]
    selector = spec.get("selector")
    stats = {
        "setup_ms": 0.0, "memory_ms": 0.0, "single_scan_ms": 0.0,
        "pair_scan_ms": 0.0, "verification_ms": 0.0,
        "single_scans": 0, "pair_scans": 0, "restarts": 0,
        "subspace_scans": 0, "subspace_scan_ms": 0.0,
        "screened_out": 0, "tabu_moves": 0, "recombinations": 0,
        "elite_pool_max_size": 0,
        "secondary_tie_moves": 0,
        "memory_rescored": 0, "memory_skipped_unbounded": 0,
        "late_scans": 0, "deadline_missed": False, "restart_densities": [],
    }

    if selector is not None:
        if selector[0] == "m1cascade":
            if selector_model is None:
                raise ValueError("M1 selector method requires selector_model")
            sel_fn, sel_k = CascadeTopK(selector_model, shortlist=selector_shortlist), int(selector[1])
        else:
            sel_fn, sel_k = SELECTORS[selector[0]], int(selector[1])
        # Separate stream: selector randomness never perturbs the search's restart draws.
        sel_rng = np.random.default_rng([int(seed), 0x5E1EC7])
        stats.update({"shortlist_scans": 0, "fallback_scans": 0, "selector_ms": 0.0,
                      "shortlist_scan_ms": 0.0, "selector_single_scans": 0})
    short_ns = 0  # largest measured selector + shortlist-scan duration

    # A mandatory all-OFF feasible action provides a valid fallback for every
    # budget, including budgets shorter than one complete pair scan.
    t0 = perf_counter_ns()
    best, bestv = _score_state(ev, np.zeros(n, dtype=np.int8))
    best2 = _user2_rate(ev, best.field) if tie_break else None
    count = 1
    stats["setup_ms"] += (perf_counter_ns() - t0) / 1e6
    trace.append(((perf_counter_ns() - start) / 1e6, bestv, count))
    if perf_counter_ns() > deadline_ns:
        stats["deadline_missed"] = True
        return best.bits.copy(), bestv, count, stats

    def record(state, value, second):
        nonlocal best, bestv, best2
        if _lex_better(value, second, bestv, best2, tie_break):
            best, bestv, best2 = state, float(value), second
            trace.append(((perf_counter_ns() - start) / 1e6, bestv, count))

    def full_candidate(state, indices, batch_value, current_value, current_second, force=False):
        """Verify one batch-selected toggle with the complete state evaluator."""
        t = perf_counter_ns()
        candidate = ev.flip(state, indices)
        value = float(ev.score(candidate.field))
        second = _user2_rate(ev, candidate.field) if tie_break else None
        ended = perf_counter_ns()
        stats["verification_ms"] += (ended - t) / 1e6
        if not _score_completed_by_deadline(ended, deadline_ns):
            stats["deadline_missed"] = True
            return None
        if abs(value - float(batch_value)) > 1e-7:
            raise AssertionError("rank-k candidate score disagrees with the full evaluator")
        if _lex_better(value, second, current_value, current_second, tie_break):
            record(candidate, value, second)
            return candidate, value, second
        if force:
            return candidate, value, second
        return None

    # Seed L6 from a small, bounded set of prior-frame elites. Large rank-k
    # updates are skipped instead of falling back to an unbounded inverse.
    current = best
    currentv = bestv
    current2 = best2
    if spec["memory"] and memory is not None and perf_counter_ns() < deadline_ns:
        tmem = perf_counter_ns()
        patterns = list(getattr(memory, "patterns", memory))[:_MEMORY_CAPACITY]
        for bits in patterns:
            b = np.asarray(bits, dtype=np.int8).reshape(-1)
            if b.size != n or not np.isin(b, (0, 1)).all():
                continue
            changed = np.flatnonzero(b != best.bits)
            if len(changed) == 0 or len(changed) > _MEMORY_MAX_FLIPS:
                stats["memory_skipped_unbounded"] += 1
                continue
            if perf_counter_ns() + max(150_000, len(changed) * 35_000) >= deadline_ns:
                stats["deadline_missed"] = True
                break
            candidate = ev.flip(best, changed)
            value = float(ev.score(candidate.field))
            second = _user2_rate(ev, candidate.field) if tie_break else None
            ended = perf_counter_ns()
            if not _score_completed_by_deadline(ended, deadline_ns):
                stats["deadline_missed"] = True
                break
            count += 1
            stats["memory_rescored"] += 1
            record(candidate, value, second)
            if _lex_better(value, second, currentv, current2, tie_break):
                current, currentv, current2 = candidate, value, second
        stats["memory_ms"] = (perf_counter_ns() - tmem) / 1e6

    # Track local optima for L5 recombination and L6 cross-frame memory.
    elites = []

    def remember_elite(state, value):
        if not (spec["recombine"] or spec["memory"]):
            return
        bits = state.bits.copy()
        if any(np.array_equal(bits, b) for b, _ in elites):
            return
        elites.append((bits, float(value)))
        elites.sort(key=lambda item: item[1], reverse=True)
        del elites[8:]
        stats["elite_pool_max_size"] = max(stats["elite_pool_max_size"], len(elites))

    def single_climb(state, value):
        nonlocal count
        while perf_counter_ns() < deadline_ns:
            # Keep a scale-aware reserve for full state construction and scoring.
            if deadline_ns - perf_counter_ns() <= (500_000 if fast else max(200_000, n * 20_000)):
                break
            t = perf_counter_ns()
            scores = np.asarray(ev.moves(state), dtype=float).reshape(-1)
            ended = perf_counter_ns()
            stats["single_scan_ms"] += (ended - t) / 1e6
            if not _score_completed_by_deadline(ended, deadline_ns):
                stats["late_scans"] += 1
                stats["deadline_missed"] = True
                break
            stats["single_scans"] += 1
            count += len(scores)
            max_score = float(np.max(scores))
            current_second = _user2_rate(ev, state.field) if tie_break else None
            if max_score < value - 1e-12:
                emit_parent(state, "single_optimum")
                break
            tied = np.flatnonzero(np.abs(scores - max_score) <= 1e-12)
            j = int(np.min(tied))
            if tie_break and (len(tied) > 1 or abs(max_score - value) <= 1e-12):
                user2 = []
                for k in sorted(map(int, tied))[:8]:
                    if deadline_ns - perf_counter_ns() <= 250_000:
                        break
                    state_k = ev.flip(state, [k])
                    user2.append((_user2_rate(ev, state_k.field), k))
                if not user2:
                    break
                second_best, j = max(user2, key=lambda item: (-np.inf if item[0] is None else item[0], -item[1]))
                if (abs(max_score - value) <= 1e-12
                        and (second_best is None or current_second is None
                             or second_best <= current_second + 1e-12)):
                    break
                stats["secondary_tie_moves"] += 1
            elif max_score <= value + 1e-12:
                emit_parent(state, "single_optimum")
                break
            candidate = full_candidate(state, [j], scores[j], value, current_second)
            if candidate is None:
                break
            state, value, current_second = candidate
        return state, value

    def full_best_pair(state, value, second, allow_nonimprove=False, tabu=None):
        nonlocal count
        # Pair scans are O(n^2); do not start one unless there is room for an
        # initial measured-scale scan plus a verified state update.
        estimate = 2_500_000 if fast else max(1_500_000, pair_count * 1_500)
        if deadline_ns - perf_counter_ns() <= estimate + 250_000:
            return None
        t = perf_counter_ns()
        scores = np.asarray(ev.moves(state, True), dtype=float).reshape(-1)
        ended = perf_counter_ns()
        stats["pair_scan_ms"] += (ended - t) / 1e6
        if not _score_completed_by_deadline(ended, deadline_ns):
            stats["late_scans"] += 1
            stats["deadline_missed"] = True
            return None
        stats["pair_scans"] += 1
        count += len(scores)
        if len(scores) != pair_count:
            raise ValueError("full pair scan returned an unexpected candidate count")
        # L4 ties use the second user's actual rate after max-min equality; the
        # deterministic pair-index order resolves any remaining tie.
        if fast:
            # Same selection without a full sort of every pair on every scan.
            top = [int(i) for i in np.flatnonzero(np.abs(scores - np.max(scores)) <= 1e-12)[:8]]
            order = top
        else:
            order = np.lexsort((np.arange(len(scores)), -scores))
            top = [int(i) for i in order[:8] if abs(float(scores[i]) - float(scores[order[0]])) <= 1e-12]
        if tie_break and len(top) > 1:
            user2 = []
            for k in top:
                if deadline_ns - perf_counter_ns() <= 250_000:
                    break
                pair = ev.pairs[k]
                state2 = ev.flip(state, pair)
                user2.append((_user2_rate(ev, state2.field), k))
            if user2:
                _, pick = max(user2, key=lambda item: (-np.inf if item[0] is None else item[0], -item[1]))
            else:
                pick = int(order[0])
        else:
            pick = int(order[0])
        pair = ev.pairs[pick]
        key = tuple(sorted(map(int, pair)))
        if tabu is not None and key in tabu:
            if fast:
                order = np.lexsort((np.arange(len(scores)), -scores))
            admissible = next((int(i) for i in order if tuple(sorted(map(int, ev.pairs[i]))) not in tabu), None)
            if admissible is not None:
                pick = admissible
                pair = ev.pairs[pick]
                key = tuple(sorted(map(int, pair)))
        value2 = float(scores[pick])
        candidate_second = None
        if tie_break and abs(value2 - value) <= 1e-12:
            candidate0 = ev.flip(state, pair)
            candidate_second = _user2_rate(ev, candidate0.field)
        tied_improvement = tie_break and _lex_better(value2, candidate_second, value, second, True)
        if value2 > value + 1e-12 or tied_improvement or allow_nonimprove:
            if tied_improvement and abs(value2 - value) <= 1e-12:
                stats["secondary_tie_moves"] += 1
            return full_candidate(state, pair, value2, value, second, force=allow_nonimprove)
        return None

    def best_pair(state, value, second, allow_nonimprove=False, tabu=None):
        if selector is None or allow_nonimprove:
            return full_best_pair(state, value, second, allow_nonimprove, tabu)
        return shortlist_pair(state, value, second, tabu)

    def shortlist_pair(state, value, second, tabu):
        """Exactly score only the selector's shortlist; full scan if none of it improves."""
        nonlocal count, short_ns
        # Room for one measured-scale shortlist pass plus a verified state update. The first pass
        # assumes 0.4 ms; later passes use 1.5x the slowest pass seen.
        if deadline_ns - perf_counter_ns() <= max(400_000, int(1.5 * short_ns)) + 250_000:
            return None
        t = perf_counter_ns()
        gains = None
        if sel_fn.needs_gains:
            singles = np.asarray(ev.moves(state), dtype=float).reshape(-1)
            count += len(singles)
            stats["selector_single_scans"] += 1
            gains = singles - value
        idx = np.unique(np.asarray(sel_fn(ev, state, gains, sel_k, sel_rng), dtype=int))
        mid = perf_counter_ns()
        stats["selector_ms"] += (mid - t) / 1e6
        if len(idx) == 0 or idx[0] < 0 or idx[-1] >= pair_count:
            raise ValueError("selector returned an invalid pair index")
        cand = ev.pairs[idx]
        scores = np.asarray(ev.moves_pairs(state, cand), dtype=float).reshape(-1)
        ended = perf_counter_ns()
        stats["shortlist_scan_ms"] += (ended - mid) / 1e6
        short_ns = max(short_ns, ended - t)
        if not _score_completed_by_deadline(ended, deadline_ns):
            stats["late_scans"] += 1
            stats["deadline_missed"] = True
            return None
        stats["shortlist_scans"] += 1
        count += len(scores)
        if len(scores) != len(idx):
            raise ValueError("shortlist scan returned an unexpected candidate count")
        # Lowest pair index among exact ties, as the full scan resolves them.
        pick = int(np.flatnonzero(np.abs(scores - np.max(scores)) <= 1e-12)[0])
        if float(scores[pick]) > value + 1e-12:
            return full_candidate(state, ev.pairs[idx[pick]], float(scores[pick]), value, second)
        if len(idx) >= pair_count:
            return None  # shortlist was the whole neighbourhood: nothing improves
        before = stats["pair_scans"]
        result = full_best_pair(state, value, second, False, tabu)
        if stats["pair_scans"] > before:
            stats["fallback_scans"] += 1
        return result

    def pair_climb(state, value, second=None, allow_escape=True):
        # Complete strict 1-opt descent before every full 2-opt scan. Callers have
        # already climbed, so the fast base skips the repeated scan.
        if not fast:
            state, value = single_climb(state, value)
        second = _user2_rate(ev, state.field) if tie_break else None
        escapes = 0
        basin_best = value
        tabu = deque(maxlen=_TABU_TENURE)
        while perf_counter_ns() < deadline_ns:
            pair_scans_before = stats["pair_scans"]
            improved = best_pair(state, value, second, allow_nonimprove=False)
            if improved is not None:
                state, value, second = improved
                state, value = single_climb(state, value)
                second = _user2_rate(ev, state.field) if tie_break else None
                remember_elite(state, value)
                # L4 resets on any improvement of the current state, so climbing back after
                # an escape can keep one basin going for the whole deadline. The fast base
                # resets only on a new basin best, so the escape limit really applies.
                if not fast or value > basin_best + 1e-12:
                    escapes = 0
                basin_best = max(basin_best, value)
                continue
            if stats["pair_scans"] > pair_scans_before:
                emit_parent(state, "pair_optimum")
            if (not spec["escape"] or not allow_escape or escapes >= _TABU_ESCAPE_LIMIT
                    or perf_counter_ns() >= deadline_ns):
                break
            jumped = best_pair(state, value, second, allow_nonimprove=True, tabu=set(tabu))
            if jumped is None:
                break
            old_bits = state.bits.copy()
            state, value, second = jumped
            diff = np.flatnonzero(old_bits != state.bits)
            tabu.append(tuple(sorted(map(int, diff))))
            stats["tabu_moves"] += 1
            escapes += 1
            state, value = single_climb(state, value)
            second = _user2_rate(ev, state.field) if tie_break else None
        remember_elite(state, value)
        return state, value

    # L6 memory can seed the first basin; otherwise begin from all-OFF.
    current, currentv = single_climb(current, currentv)
    current, currentv = pair_climb(current, currentv)

    restart_no = 0
    one_opt_values = []  # sorted single-flip values of earlier random restarts
    while deadline_ns - perf_counter_ns() > (1_100_000 if fast else max(350_000, n * 30_000)):
        stats["restarts"] += 1
        restart_no += 1
        if spec["recombine"] and len(elites) >= 2 and restart_no % 3 == 0:
            bits = _elite_recombine(elites, rng, n)
            stats["recombinations"] += 1
        elif spec["density"]:
            density = (18.0 / 64.0) if restart_no % 2 else 0.5
            stats["restart_densities"].append(float(density))
            bits = (rng.random(n) < density).astype(np.int8)
        else:
            if spec["screen"]:
                stats["restart_densities"].append(0.5)
            bits = rng.integers(0, 2, n, dtype=np.int8)
        t = perf_counter_ns()
        state, value = _score_state(ev, bits)
        ended = perf_counter_ns()
        if not _score_completed_by_deadline(ended, deadline_ns):
            stats["deadline_missed"] = True
            break
        count += 1
        emit_parent(state, "random_start")
        second = _user2_rate(ev, state.field) if tie_break else None
        record(state, value, second)
        # L2 screens a single-flip basin before paying for a full pair scan.
        state, value = single_climb(state, value)
        second = _user2_rate(ev, state.field) if tie_break else None
        if spec["screen_quantile"] is not None:
            # Compare single-flip values only with each other. L2 compares them with a
            # pair-refined best, which rejected about 99% of restarts at 100 ms.
            cutoff = (one_opt_values[int(spec["screen_quantile"] * len(one_opt_values))]
                      if one_opt_values else -np.inf)
            insort(one_opt_values, value)
            if value < cutoff:
                stats["screened_out"] += 1
                continue
        if spec["screen"] and value < bestv - _SCREEN_MARGIN:
            # A cheap, exact top-eight subspace probe can rescue a basin whose
            # one-flip value is weak but whose ranked 2-flip neighborhood is
            # promising. Only those restarts earn the full O(n^2) pair scan.
            if deadline_ns - perf_counter_ns() > max(350_000, n * 25_000):
                t = perf_counter_ns()
                singles = np.asarray(ev.moves(state), dtype=float).reshape(-1)
                ended = perf_counter_ns()
                stats["single_scan_ms"] += (ended - t) / 1e6
                if _score_completed_by_deadline(ended, deadline_ns):
                    stats["single_scans"] += 1
                    count += len(singles)
                    anchors = np.argsort(singles)[::-1][:min(8, n)]
                    t = perf_counter_ns()
                    sub_scores, _ = ev.moves_subspace(state, anchors)
                    ended = perf_counter_ns()
                    stats["subspace_scan_ms"] += (ended - t) / 1e6
                    if _score_completed_by_deadline(ended, deadline_ns):
                        stats["subspace_scans"] += 1
                        count += len(sub_scores)
                        if len(sub_scores) == 0 or float(np.max(sub_scores)) < bestv - _SCREEN_MARGIN:
                            stats["screened_out"] += 1
                            current, currentv, current2 = best, bestv, best2
                            continue
                    else:
                        stats["late_scans"] += 1
                        stats["deadline_missed"] = True
                        break
                else:
                    stats["late_scans"] += 1
                    stats["deadline_missed"] = True
                    break
            else:
                stats["screened_out"] += 1
                current, currentv, current2 = best, bestv, best2
                continue
        state, value = pair_climb(state, value, second, allow_escape=not spec["escape_first_basin_only"])
        current, currentv = state, value
        current2 = _user2_rate(ev, state.field) if tie_break else None

    if spec["memory"] and memory is not None:
        for bits, val in elites:
            memory.remember(bits, val)
        memory.remember(best.bits, bestv)

    # Final full-channel verification is required even if the last search scan
    # ended close to the deadline. It never promotes a late candidate.
    # The fast base leaves this to its caller, as `pair` does: the benchmark rescores every
    # returned action directly inside the timed section.
    verified_value = bestv
    if not fast:
        t = perf_counter_ns()
        verified = ev.state(best.bits)
        verified_value = float(ev.score(verified.field))
        stats["verification_ms"] += (perf_counter_ns() - t) / 1e6
        if abs(verified_value - bestv) > 1e-7:
            raise AssertionError("returned best action failed full coupled-network verification")
    stats["deadline_missed"] = bool(stats["deadline_missed"] or perf_counter_ns() > deadline_ns)
    return best.bits.copy(), verified_value, count, stats

