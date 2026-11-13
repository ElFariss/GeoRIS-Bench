"""Registry of reference solvers added after the matched_v2 freeze (NEW, 2026-11-13).

search.py / coupled_search.py / me_ga.py are hash-locked by outputs/frozen/matched_v2.json and are not edited.
This module is a new dispatch path: methods in NEW_METHODS go to their module, every other name goes to the
unchanged coupled_search.search with the caller's arguments, so existing methods behave byte-for-byte as before.
"""
from atomwise_ris.optimization import coupled_search, nsga2
from atomwise_ris.optimization.search import ME_GA_METHODS

NEW_METHODS = tuple(nsga2.NSGA2_METHODS)
EVOLUTIONARY_BASELINES = tuple(ME_GA_METHODS) + NEW_METHODS


def search_coupled(ev, method, deadline, seed, memory=(), trace=None, **kw):
    """(bits, value, evaluations, breakdown) for any coupled method; `deadline` is absolute perf_counter ns."""
    if method in nsga2.NSGA2_METHODS:
        return nsga2.search_coupled(ev, method, deadline, seed, memory, trace, params=kw.get('params'))
    return coupled_search.search(ev, method, deadline, seed, memory=memory, trace=trace, **kw)
