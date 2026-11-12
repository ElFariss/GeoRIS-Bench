"""CPU controllers; every online action is timed. No post-deadline improvements.

Model construction/loading is offline. Each run reports overruns. A candidate
counts only when its entire evaluation finished before the deadline. Initial
configuration evaluation is mandatory and is reported even if budget too small.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np

@dataclass
class SearchResult:
    bits: np.ndarray
    value: float
    elapsed_ms: float
    evaluations: int
    trace: list
    breakdown_ms: dict
    deadline_missed: bool

@dataclass
class Memory:
    capacity: int = 12
    trajectory: object = None
    patterns: list = field(default_factory=list)

    def enter(self,trajectory):
        if trajectory != self.trajectory:
            self.patterns=[]; self.trajectory=trajectory

    def remember(self,bits):
        if isinstance(bits,(list,tuple)) and len(bits)>0 and isinstance(bits[0],(list,tuple,np.ndarray)):
            for b in bits: self.remember(b)
            return
        b=np.asarray(bits,dtype=np.int8).copy()
        self.patterns=[a for a in self.patterns if not np.array_equal(a,b)]
        self.patterns.insert(0,b);self.patterns=self.patterns[:self.capacity]

    def remember_population(self,population):
        if not population: return
        if isinstance(population[0],tuple):
            for b,_ in sorted(population,key=lambda x:x[1],reverse=True):
                self.remember(b)
        else:
            for b in population: self.remember(b)

def softmax(x):
    x=np.asarray(x);z=np.exp(x-np.max(x));return z/z.sum()

def local_optimum(ev,bits,pairs=False,max_steps=1000):
    """Offline reference and label generation; NEVER called outside timed loop in benchmarks."""
    b=np.asarray(bits,dtype=np.int8).copy();u=ev.field(b);v=ev.score_field(u)
    for _ in range(max_steps):
        fs=ev.flip_fields(b,u);vs=ev.score_field(fs);j=int(np.argmax(vs))
        if vs[j]>v+1e-12:
            b[j]^=1;u=fs[j];v=vs[j];continue
        if pairs:
            fs=ev.pair_fields(b,u);vs=ev.score_field(fs);j=int(np.argmax(vs))
            if vs[j]>v+1e-12:
                b[ev.pairs[j]]^=1;u=fs[j];v=vs[j];continue
        break
    return b,float(v)

def _anchor_pairs_fast(ev,anchors):
    mask=np.zeros(ev.n,dtype=bool);mask[anchors]=True
    return ev.pairs[mask[ev.pairs[:,0]]|mask[ev.pairs[:,1]]]

def _conditional_pair_proposals(first,scores,residual,existing=(),limit=4,n=None):
    """Bounded learned second-bit proposals; zero residual is a strict no-op."""
    scores=np.asarray(scores,dtype=float).reshape(-1);residual=np.asarray(residual,dtype=float).reshape(-1)
    if scores.size!=residual.size or (n is not None and scores.size!=n):
        return np.empty((0,2),dtype=int)
    if scores.size<2 or not np.any(np.abs(residual)>1e-8):
        return np.empty((0,2),dtype=int)
    first=int(first)
    order=np.argsort(scores)[::-1]
    order=order[order!=first][:min(max(0,int(limit)),scores.size-1)]
    known={tuple(sorted(map(int,pair))) for pair in existing}
    pairs=[(first,int(j)) for j in order if tuple(sorted((first,int(j)))) not in known]
    return np.asarray(pairs,dtype=int).reshape(-1,2)

def _analytical_residual_rank(analytical,output,weight):
    """Blend a residual policy without counting its analytical input twice."""
    analytical=np.asarray(analytical,dtype=float)
    output=np.asarray(output,dtype=float)
    return analytical+float(weight)*(output-analytical)

def _score_completed_by_deadline(ended_ns,deadline_ns):
    """Protocol count eligibility: a candidate counts only if its full score finished in time."""
    return int(ended_ns)<=int(deadline_ns)

PROTOCOL_PATH=Path(__file__).resolve().parents[3]/'configs'/'experiments'/'matched_v1.json'
GA_METHODS=('memory_ga','uniform_ga','analytical_ga','analytical_no_memory_ga','mlp_ga','attention_ga',
            'attention_no_memory_ga','attention_no_physics_ga','attention_no_conditioning_ga')
BASELINE_METHODS=('greedy','pair','random')
NEURAL_METHODS=('mlp_ga','attention_ga','attention_no_memory_ga','attention_no_physics_ga','attention_no_conditioning_ga')
POPULATION_METHOD='population_ga'
# Zardi et al. reimplementation (src/atomwise_ris/optimization/me_ga.py). Deliberately NOT in GA_METHODS:
# they are not matched-protocol variants, are never registered by matched_v1, and take no shared config.
ME_GA_METHODS=('me_ga','ga_risc','me_ga_paper')

def variant_spec(method):
    """Ranking signal and memory switch of a GA variant; the only two ways variants may differ."""
    if method not in GA_METHODS:raise ValueError(f'Not a GA variant: {method}')
    ranking='uniform' if method=='uniform_ga' else 'memory' if method=='memory_ga' else \
            'neural' if method in NEURAL_METHODS else 'analytical'
    return {'ranking':ranking,'memory':not method.endswith('_no_memory_ga')}

@dataclass(frozen=True)
class SearchConfig:
    """Shared GA step limits from the frozen matched protocol (one object drives every GA variant)."""
    name:str
    specular:dict
    coupled:dict
    shared:dict
    protocol_sha256:str
    def specular_limits(self,budget_ms):
        s=self.specular
        t_cap=min(budget_ms*s['t_cap_fraction_short'],s['t_cap_max_ms_short']) if budget_ms<=12.0 else budget_ms*s['t_cap_fraction_long']
        return {'P0':int(s['P0']),'P1':int(s['P1']),'n_starts':int(s['n_starts']),'max_pol':int(s['max_pol']),'t_cap_ms':float(t_cap)}
    def coupled_limits(self):
        return {'K':int(self.coupled['K']),'hop_K':int(self.coupled['hop_K']),'hop_idx':[int(i) for i in self.shared['hop_idx']]}

def load_search_config(name='medium',path=None):
    """Load small/medium/large limits from the registered protocol file (never edited by code)."""
    p=Path(path) if path is not None else PROTOCOL_PATH
    raw=p.read_bytes();protocol=json.loads(raw)
    sc=protocol['shared_search_config']
    if name not in sc['specular'] or name not in sc['coupled']:raise ValueError(f'Unknown search config {name!r}')
    return SearchConfig(name,dict(sc['specular'][name]),dict(sc['coupled'][name]),dict(sc['all_variants']),hashlib.sha256(raw).hexdigest())

def policy_reserve_ns(policy,measured_ms=0.,floor_ns=300000):
    """Time to keep free before invoking a neural ranking pass so its overhead stays inside the deadline.

    Uses the larger of the policy's calibrated cost (`policy.expected_ms`, measured on a train case during warm-up)
    and the largest cost observed in this run, with 25 percent headroom, never below `floor_ns`.
    """
    hint=float(getattr(policy,'expected_ms',0.) or 0.)
    return max(int(floor_ns),int(1.25e6*max(hint,measured_ms)))

def search(ev,method,budget_ms,seed,initial=None,memory=None,policy=None,
           population_size=12,p_pair=.5,config=None,legacy=False,
           conditional_pair_policy=False):
    """Dispatch. Default: GA variants run the shared matched config. `legacy=True` is the only way to reach per-method limits."""
    if method in ME_GA_METHODS:   # explicit opt-in only; ignores config/legacy/policy/initial (see me_ga.py)
        from atomwise_ris.optimization import me_ga
        return me_ga.search_specular(ev,method,budget_ms,seed,initial,memory)
    if method == POPULATION_METHOD:
        if legacy or config is not None:raise ValueError('population_ga has its own fixed controller and no matched config')
        return _search_population(ev,budget_ms,seed,initial,memory)
    if legacy:
        if config is not None:raise ValueError('legacy mode cannot be combined with a matched config')
        return _search_legacy(ev,method,budget_ms,seed,initial,memory,policy,population_size,p_pair)
    if method in GA_METHODS:
        cfg=config if config is not None else load_search_config('medium')
        return _search_matched(ev,method,budget_ms,seed,initial,memory,policy,cfg,
                               conditional_pair_policy=conditional_pair_policy)
    # greedy/pair/random have no per-method limits and identical counting; unchanged behaviour.
    return _search_legacy(ev,method,budget_ms,seed,initial,memory,policy,population_size,p_pair)

def _search_population(ev,budget_ms,seed,initial=None,memory=None,population_size=12):
    """Standalone population-12 tournament GA baseline under a complete CPU deadline.

    This is deliberately separate from the matched ranking-guidance family: each
    generation uses tournament selection, uniform crossover, and bit mutation.
    Every objective value comes from the evaluator's full candidate score path.
    """
    start=perf_counter_ns(); deadline=start+int(budget_ms*1e6)
    rng=np.random.default_rng(seed); n=ev.n; size=max(4,int(population_size))
    timings={'setup':0.,'features':0.,'inference':0.,'evaluation':0.,'memory':0.,'other':0.}
    zero=np.zeros(n,dtype=np.int8) if initial is None else np.asarray(initial,dtype=np.int8).copy()
    candidates=[zero]
    remembered=getattr(memory,'patterns',memory if memory is not None else ())
    candidates.extend(np.asarray(b,dtype=np.int8).copy() for b in list(remembered)[:8])
    candidates=candidates[:size]
    while len(candidates)<size:
        candidates.append(rng.integers(0,2,n,dtype=np.int8))
    scores=[]; count=0; best=zero.copy(); bestv=-np.inf
    candidate_cost_ns=300_000
    completion_reserve_ns=250_000
    trace=[]
    def evaluate(bits,mandatory=False):
        nonlocal count,best,bestv,candidate_cost_ns
        if not mandatory and deadline-perf_counter_ns()<=max(
                completion_reserve_ns, int(candidate_cost_ns*1.6)+completion_reserve_ns):return None
        t=perf_counter_ns(); value=float(ev.score(bits)); ended=perf_counter_ns()
        candidate_cost_ns=max(candidate_cost_ns,ended-t)
        timings['evaluation']+=(ended-t)/1e6
        if ended<=deadline or mandatory:
            count+=1
            if value>bestv:
                best=np.asarray(bits,dtype=np.int8).copy();bestv=value
                trace.append(((ended-start)/1e6,bestv,count))
            return value
        return None
    for i,bits in enumerate(candidates):
        v=evaluate(bits,mandatory=(i==0))
        if v is None:break
        scores.append(v)
    candidates=candidates[:len(scores)]
    while deadline-perf_counter_ns()>max(completion_reserve_ns,int(candidate_cost_ns*1.6)+completion_reserve_ns):
        order=np.argsort(scores)[::-1]
        elite=candidates[int(order[0])].copy(); next_pop=[elite]; next_scores=[float(scores[int(order[0])])]
        while len(next_pop)<size and perf_counter_ns()<deadline:
            def tournament():
                ix=rng.integers(0,len(candidates),size=min(2,len(candidates)))
                return candidates[int(ix[np.argmax(np.asarray(scores)[ix])])]
            p1=tournament();p2=tournament()
            child=(np.where(rng.integers(0,2,n,dtype=bool),p1,p2).astype(np.int8,copy=False)
                   if rng.random()<0.25 else p1.copy())
            n_mut=int(rng.choice((1,2)))
            child[rng.choice(n,size=min(n,n_mut),replace=False)]^=1
            value=evaluate(child)
            if value is None:break
            next_pop.append(child.copy()); next_scores.append(value)
        if len(next_pop)<2:break
        candidates,scores=next_pop,next_scores
    elapsed=(perf_counter_ns()-start)/1e6
    timings['other']=max(0.,elapsed-sum(timings.values()))
    return SearchResult(best,bestv,elapsed,count,trace,timings,elapsed>budget_ms)

def _search_legacy(ev,method,budget_ms,seed,initial=None,memory=None,policy=None,
                   population_size=12,p_pair=.5):
    """Pre-matched controller with hand-set per-method limits; kept for reproducing legacy runs and for the non-GA baselines."""
    start=perf_counter_ns();deadline=start+int(budget_ms*1e6)
    if budget_ms <= 1.2:
        safety_margin = int(max(0.35, budget_ms * 0.35) * 1e6)
    elif budget_ms <= 5.5:
        safety_margin = int(max(0.55, budget_ms * 0.12) * 1e6)
    else:
        safety_margin = int(max(0.65, min(1.1, budget_ms * 0.08)) * 1e6)
    work_deadline = deadline - safety_margin

    timings={'setup':0.,'features':0.,'inference':0.,'evaluation':0.,'memory':0.,'other':0.}
    rng=np.random.default_rng(seed)
    b=np.zeros(ev.n,dtype=np.int8) if initial is None else np.asarray(initial,dtype=np.int8).copy()
    best=b.copy();u=ev.field(b);v=float(ev.score_field(u));bestv=v;count=1
    trace=[((perf_counter_ns()-start)/1e6,bestv,count)]
    timings['setup']=(perf_counter_ns()-start)/1e6

    def accept(candidate,field,value,already_counted=False):
        nonlocal best,bestv,count
        if not already_counted:count+=1
        if perf_counter_ns()<=deadline and value>bestv:
            best=candidate.copy();bestv=float(value);trace.append(((perf_counter_ns()-start)/1e6,bestv,count))

    ga=method in ('memory_ga','analytical_ga','mlp_ga','attention_ga','attention_no_physics_ga','attention_no_conditioning_ga','attention_no_memory_ga')
    if not ga:
        estimated_iteration_ns=100000
        while perf_counter_ns()+estimated_iteration_ns<work_deadline:
            iteration_start=perf_counter_ns()
            if method in ('greedy','pair'):
                t=perf_counter_ns();fs=ev.flip_fields(b,u);values=ev.score_field(fs)
                timings['evaluation']+=(perf_counter_ns()-t)/1e6;count+=ev.n
                if perf_counter_ns()>deadline:break
                j=int(np.argmax(values))
                if values[j]>v+1e-12:
                    b[j]^=1;u=fs[j];v=float(values[j]);accept(b,u,v,already_counted=True)
                    estimated_iteration_ns=max(estimated_iteration_ns,int((perf_counter_ns()-iteration_start)*1.2))
                    continue
                if method=='pair':
                    if perf_counter_ns()+700000>work_deadline:break
                    t=perf_counter_ns();fs=ev.pair_fields(b,u);values=ev.score_field(fs)
                    timings['evaluation']+=(perf_counter_ns()-t)/1e6;count+=len(ev.pairs)
                    if perf_counter_ns()>deadline:break
                    j=int(np.argmax(values))
                    if values[j]>v+1e-12:
                        b[ev.pairs[j]]^=1;u=fs[j];v=float(values[j]);accept(b,u,v,already_counted=True)
                        estimated_iteration_ns=max(estimated_iteration_ns,int((perf_counter_ns()-iteration_start)*1.2))
                        continue
                b=rng.integers(0,2,ev.n,dtype=np.int8)
                t=perf_counter_ns();u=ev.field(b);v=float(ev.score_field(u));timings['evaluation']+=(perf_counter_ns()-t)/1e6
                accept(b,u,v)
            elif method=='random':
                q=rng.integers(0,2,(16,ev.n),dtype=np.int8)
                t=perf_counter_ns();fu=ev.field(q);fv=ev.score_field(fu);timings['evaluation']+=(perf_counter_ns()-t)/1e6
                count+=len(q)-1;j=int(np.argmax(fv));accept(q[j],fu[j],fv[j])
            else:raise ValueError(method)
            estimated_iteration_ns=max(estimated_iteration_ns,int((perf_counter_ns()-iteration_start)*1.2))
    else:
        # Fast path for ultra-tight 1 ms budgets: 1-opt step + memory check + neural policy pass
        if budget_ms <= 1.2:
            t = perf_counter_ns()
            fs = ev.flip_fields(b, u)
            vs = ev.score_field(fs)
            timings['evaluation'] += (perf_counter_ns() - t) / 1e6
            count += ev.n
            j = int(np.argmax(vs))
            if vs[j] > v + 1e-12:
                b[j] ^= 1; u = fs[j]; v = float(vs[j])
                accept(b, u, v, already_counted=True)

            recent = np.zeros(ev.n)
            memory_freq = np.zeros(ev.n)
            if memory is not None and len(memory.patterns) > 0:
                t = perf_counter_ns()
                fu = ev.field(memory.patterns[0]); fv = float(ev.score_field(fu))
                accept(memory.patterns[0], fu, fv)
                timings['memory'] += (perf_counter_ns() - t) / 1e6

            analytical_gains = vs - v
            norm_gains = analytical_gains / (np.std(analytical_gains) + 1e-6)
            if method in ('mlp_ga', 'attention_ga', 'attention_no_physics_ga', 'attention_no_conditioning_ga', 'attention_no_memory_ga') and perf_counter_ns() + 300000 < deadline:
                t = perf_counter_ns()
                x, _ = ev.features(best, u, memory_frequency=memory_freq, recent=recent, gains=analytical_gains)
                timings['features'] += (perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()
                if policy is not None:
                    logits = policy(x, norm_gains)
                else:
                    logits = norm_gains
                timings['inference'] += (perf_counter_ns() - t) / 1e6

            if memory is not None:
                t = perf_counter_ns()
                memory.remember(best)
                timings['memory'] += (perf_counter_ns() - t) / 1e6

            elapsed = (perf_counter_ns() - start) / 1e6
            timings['other'] = max(0., elapsed - sum(timings.values()))
            return SearchResult(best, bestv, elapsed, count, trace, timings, elapsed > budget_ms)

        # Method-specific active anchor subspace and evolutionary basin parameters
        if method == 'memory_ga':
            P0, P1, n_starts, max_pol, t_cap = 4, 3, 0, 1, min(budget_ms * 0.45, 3.2) if budget_ms <= 12.0 else budget_ms * 0.50
        elif method == 'attention_no_memory_ga':
            P0, P1, n_starts, max_pol, t_cap = 5, 4, 1, 1, min(budget_ms * 0.50, 3.6) if budget_ms <= 12.0 else budget_ms * 0.55
        elif method == 'analytical_ga':
            P0, P1, n_starts, max_pol, t_cap = 5, 4, 1, 2, min(budget_ms * 0.52, 3.8) if budget_ms <= 12.0 else budget_ms * 0.58
        elif method in ('attention_no_physics_ga', 'attention_no_conditioning_ga'):
            P0, P1, n_starts, max_pol, t_cap = 5, 4, 1, 2, min(budget_ms * 0.55, 4.0) if budget_ms <= 12.0 else budget_ms * 0.60
        elif method == 'mlp_ga':
            P0, P1, n_starts, max_pol, t_cap = 6, 5, 2, 2, min(budget_ms * 0.58, 4.4) if budget_ms <= 12.0 else budget_ms * 0.65
        else:  # attention_ga
            P0, P1, n_starts, max_pol, t_cap = 6, 5, 4, 4, min(budget_ms * 0.65, 5.4) if budget_ms <= 12.0 else budget_ms * 0.72

        if budget_ms > 20.0:
            n_starts += 2
            max_pol += 1

        def greedy_1opt(b_in, u_in, v_in):
            b_c = b_in.copy(); u_c = u_in; v_c = v_in
            fs = None; vs = None
            while perf_counter_ns() + 25000 < work_deadline:
                t = perf_counter_ns()
                fs = ev.flip_fields(b_c, u_c); vs = ev.score_field(fs)
                timings['evaluation'] += (perf_counter_ns() - t) / 1e6
                j = int(np.argmax(vs))
                if vs[j] > v_c + 1e-12:
                    b_c[j] ^= 1; u_c = fs[j]; v_c = float(vs[j])
                    accept(b_c, u_c, v_c, already_counted=False)
                    nonlocal count
                    count += ev.n - 1
                else:
                    count += ev.n
                    break
            return b_c, u_c, v_c, fs, vs

        def climb_anchor_2opt(b_in, u_in, v_in, P, use_pol=False):
            nonlocal count
            moved_pair = False
            b_c, u_c, v_c, fs, vs = greedy_1opt(b_in, u_in, v_in)
            if fs is None:
                t = perf_counter_ns()
                fs = ev.flip_fields(b_c, u_c); vs = ev.score_field(fs)
                timings['evaluation'] += (perf_counter_ns() - t) / 1e6; count += ev.n
            gains = vs - v_c
            norm_g = gains / (np.std(gains) + 1e-6)
            logits = norm_g
            if use_pol and method in ('mlp_ga', 'attention_ga', 'attention_no_physics_ga', 'attention_no_conditioning_ga', 'attention_no_memory_ga'):
                t = perf_counter_ns()
                x, _ = ev.features(b_c, u_c, gains=gains)
                timings['features'] += (perf_counter_ns() - t) / 1e6
                t = perf_counter_ns()
                if policy is not None:
                    logits = policy(x, norm_g)
                timings['inference'] += (perf_counter_ns() - t) / 1e6

            while perf_counter_ns() + 120000 < work_deadline:
                dr = ev.rates(fs) - ev.rates(u_c)[None, :]
                sum_g = dr.sum(axis=-1); norm_sg = sum_g / (np.std(sum_g) + 1e-6)
                norm_mg = (vs - v_c) / (np.std(vs - v_c) + 1e-6)
                sc = (0.15 * logits + 0.55 * norm_sg + 0.45 * norm_mg) if (use_pol and policy is not None) else (0.55 * norm_sg + 0.45 * norm_mg)
                order = np.argsort(sc)[::-1]
                anchors = np.concatenate([order[:P], order[-2:]])
                ap = _anchor_pairs_fast(ev, anchors)
                t = perf_counter_ns()
                d = (1 - 2 * b_c.astype(float))[:, None, None] * ev.delta
                fs2 = u_c + d[ap[:, 0]] + d[ap[:, 1]]
                vs2 = ev.score_field(fs2)
                timings['evaluation'] += (perf_counter_ns() - t) / 1e6; count += len(ap)
                j2 = int(np.argmax(vs2))
                if vs2[j2] > v_c + 1e-12:
                    b_c[ap[j2]] ^= 1; u_c = fs2[j2]; v_c = float(vs2[j2]); moved_pair = True
                    accept(b_c, u_c, v_c, already_counted=True)
                    b_c, u_c, v_c, fs, vs = greedy_1opt(b_c, u_c, v_c)
                    if fs is None: break
                else:
                    break
            return b_c, u_c, v_c, moved_pair, logits

        # Step 1: Climb from initial state (all-OFF) with anchor-partner 2-opt and neural policy
        b0, u0, v0, moved_pair, logits = climb_anchor_2opt(b, u, v, P0, use_pol=True)
        bestu = u0 if np.array_equal(best, b0) else ev.field(best)

        # Step 2: Memory Retrieval & Rescoring (Zardi, Oliveri, & Massa, IEEE TAP 2024)
        if memory is not None and len(memory.patterns) > 0 and perf_counter_ns() + 60000 < work_deadline:
            t = perf_counter_ns()
            for cand in memory.patterns[:2]:
                if perf_counter_ns() + 40000 >= work_deadline: break
                fu = ev.field(cand); fv = float(ev.score_field(fu))
                count += 1
                if fv > bestv and perf_counter_ns() <= deadline:
                    best = cand.copy(); bestv = fv; bestu = fu
                    trace.append(((perf_counter_ns() - start) / 1e6, bestv, count))
            timings['memory'] += (perf_counter_ns() - t) / 1e6

        # Step 3: Pareto-Ridge Multi-Basin Exploration (only needed when 1-opt optimum admits improving pairs)
        if moved_pair and perf_counter_ns() + 200000 < work_deadline:
            seen = {b0.tobytes()}
            # Antipodal all-ON start
            b1 = np.ones(ev.n, dtype=np.int8)
            t = perf_counter_ns(); u1 = ev.field(b1); v1 = float(ev.score_field(u1))
            timings['evaluation'] += (perf_counter_ns() - t) / 1e6
            accept(b1, u1, v1)
            if v1 >= bestv: bestu = u1
            b1, u1, v1, _, _ = greedy_1opt(b1, u1, v1)
            if v1 >= bestv: bestu = u1
            if b1.tobytes() not in seen and v1 >= bestv - 3e-3 and (perf_counter_ns() - start) < t_cap * 1e6:
                seen.add(b1.tobytes())
                b1, u1, v1, _, _ = climb_anchor_2opt(b1, u1, v1, P1, use_pol=False)
                if v1 >= bestv: bestu = u1

            starts = [rng.integers(0, 2, ev.n, dtype=np.int8) for _ in range(n_starts)]
            diff = np.flatnonzero(b0 != b1)
            if len(diff) >= 4 and method == 'attention_ga':
                d_ord = diff[np.argsort(logits[diff])[::-1]]
                c = b0.copy(); k = max(2, len(d_ord) // 2); c[d_ord[:k]] = b1[d_ord[:k]]
                starts.append(c)

            for st in starts:
                if (perf_counter_ns() - start) > t_cap * 1e6 or perf_counter_ns() + 180000 >= work_deadline:
                    break
                bs = st.copy()
                t = perf_counter_ns(); us = ev.field(bs); vs = float(ev.score_field(us))
                timings['evaluation'] += (perf_counter_ns() - t) / 1e6
                accept(bs, us, vs)
                if vs >= bestv: bestu = us
                bs, us, vs, _, _ = greedy_1opt(bs, us, vs)
                if vs >= bestv: bestu = us
                if bs.tobytes() not in seen and vs >= bestv - 3e-3 and (perf_counter_ns() - start) < (t_cap - 0.15) * 1e6 and perf_counter_ns() + 180000 < work_deadline:
                    seen.add(bs.tobytes())
                    bs, us, vs, _, _ = climb_anchor_2opt(bs, us, vs, P1, use_pol=False)
                    if vs >= bestv: bestu = us

            # Bounded verification polish on the winning basin elite
            bestu = ev.field(best)
            for _ in range(max_pol):
                if (perf_counter_ns() - start) > (t_cap + 0.85) * 1e6 or perf_counter_ns() + 350000 >= work_deadline:
                    break
                t = perf_counter_ns()
                fs_p = ev.pair_fields(best, bestu); vs_p = ev.score_field(fs_p)
                timings['evaluation'] += (perf_counter_ns() - t) / 1e6; count += len(ev.pairs)
                jp = int(np.argmax(vs_p))
                if vs_p[jp] > bestv + 1e-12:
                    bp = best.copy(); bp[ev.pairs[jp]] ^= 1; up = fs_p[jp]; vp = float(vs_p[jp])
                    accept(bp, up, vp, already_counted=True)
                    bestu = up
                    bp, up, vp, _, _ = greedy_1opt(bp, up, vp)
                    bestu = ev.field(best)
                else:
                    break

    # Memory write is online; fast single-insert
    if memory is not None:
        t=perf_counter_ns()
        memory.remember(best)
        timings['memory']+=(perf_counter_ns()-t)/1e6
    elapsed=(perf_counter_ns()-start)/1e6
    timings['other']=max(0.,elapsed-sum(timings.values()))
    return SearchResult(best,bestv,elapsed,count,trace,timings,elapsed>budget_ms)


def _search_matched(ev,method,budget_ms,seed,initial,memory,policy,config,
                    conditional_pair_policy=False):
    """Matched-protocol GA. Every variant executes the same steps with the same limits.

    Variants differ only in (a) the ranking signal used wherever atoms are ranked
    (anchors, crossover order) and (b) whether memory is supplied. Counting rule:
    one evaluation per candidate configuration scored; neural inference is time only.
    """
    spec=variant_spec(method);rank_kind=spec['ranking']
    if not spec['memory']:memory=None
    lim=config.specular_limits(budget_ms)
    P0,P1,n_starts,max_pol,t_cap=lim['P0'],lim['P1'],lim['n_starts'],lim['max_pol'],lim['t_cap_ms']
    w_logit=float(config.shared['neural_logit_weight_specular']);w_mem=float(config.shared['memory_bias_weight'])
    run_crossover=bool(config.shared['basin_crossover'])
    start=perf_counter_ns();deadline=start+int(budget_ms*1e6)
    if budget_ms <= 1.2:
        safety_margin = int(max(0.35, budget_ms * 0.35) * 1e6)
    elif budget_ms <= 5.5:
        safety_margin = int(max(0.55, budget_ms * 0.12) * 1e6)
    else:
        safety_margin = int(max(0.65, min(1.1, budget_ms * 0.08)) * 1e6)
    work_deadline = deadline - safety_margin

    timings={'setup':0.,'features':0.,'inference':0.,'evaluation':0.,'memory':0.,'other':0.}
    rng=np.random.default_rng(seed)
    b=np.zeros(ev.n,dtype=np.int8) if initial is None else np.asarray(initial,dtype=np.int8).copy()
    best=b.copy();u=ev.field(b);v=float(ev.score_field(u));bestv=v;count=1
    trace=[((perf_counter_ns()-start)/1e6,bestv,count)]
    timings['setup']=(perf_counter_ns()-start)/1e6

    def record(candidate,value):
        """Single point where an improvement is stored and traced."""
        nonlocal best,bestv
        best=np.asarray(candidate).copy();bestv=float(value)
        trace.append(((perf_counter_ns()-start)/1e6,bestv,count))

    def accept(candidate,value,already_counted=False):
        nonlocal count
        ended=perf_counter_ns()
        if ended>deadline:return
        if not already_counted:count+=1
        if value>bestv:record(candidate,value)

    mem_std=np.zeros(ev.n)
    if rank_kind=='memory' and memory is not None and len(memory.patterns)>1:
        t=perf_counter_ns();mem_std=np.std(np.asarray(memory.patterns,dtype=float),axis=0)
        timings['memory']+=(perf_counter_ns()-t)/1e6

    seen_ms=[0.]
    pair_eval_ns=[1_000_000]
    def neural_logits(b_in,u_in,gains,norm_g,first=-1):
        """Policy pass; time is paid from the same deadline and reported separately."""
        t=perf_counter_ns();x,_=ev.features(b_in,u_in,first=first,gains=gains)
        t1=perf_counter_ns();timings['features']+=(t1-t)/1e6
        out=policy(x,norm_g);t2=perf_counter_ns();timings['inference']+=(t2-t1)/1e6
        seen_ms[0]=max(seen_ms[0],(t2-t)/1e6)
        return out

    # Fast path for 1 ms budgets: identical for all variants (1-opt, memory check, one ranking pass).
    if budget_ms <= 1.2:
        b0=b.copy();u0=u;v0=v
        t=perf_counter_ns();fs=ev.flip_fields(b,u);vs=ev.score_field(fs)
        ended=perf_counter_ns();timings['evaluation']+=(ended-t)/1e6
        if ended<=deadline:count+=ev.n
        j=int(np.argmax(vs))
        if vs[j]>v+1e-12:
            b[j]^=1;u=fs[j];v=float(vs[j]);accept(b,v,already_counted=True)
        if memory is not None and len(memory.patterns)>0:
            t=perf_counter_ns();fu=ev.field(memory.patterns[0]);fv=float(ev.score_field(fu))
            accept(memory.patterns[0],fv);timings['memory']+=(perf_counter_ns()-t)/1e6
        if perf_counter_ns()+(policy_reserve_ns(policy) if rank_kind=='neural' and policy is not None else 300000)<deadline:
            gains=vs-v0;norm_g=gains/(np.std(gains)+1e-6)
            if rank_kind=='uniform':rng.random(ev.n)
            elif rank_kind=='neural' and policy is not None:neural_logits(b0,u0,gains,norm_g)
        if memory is not None:
            t=perf_counter_ns();memory.remember(best);timings['memory']+=(perf_counter_ns()-t)/1e6
        elapsed=(perf_counter_ns()-start)/1e6
        timings['other']=max(0.,elapsed-sum(timings.values()))
        return SearchResult(best,bestv,elapsed,count,trace,timings,elapsed>budget_ms)

    def greedy_1opt(b_in,u_in,v_in):
        nonlocal count
        b_c=b_in.copy();u_c=u_in;v_c=v_in;fs=None;vs=None
        while perf_counter_ns()+25000<work_deadline:
            t=perf_counter_ns();fs=ev.flip_fields(b_c,u_c);vs=ev.score_field(fs)
            ended=perf_counter_ns();timings['evaluation']+=(ended-t)/1e6
            if ended<=deadline:count+=ev.n
            j=int(np.argmax(vs))
            if vs[j]>v_c+1e-12:
                b_c[j]^=1;u_c=fs[j];v_c=float(vs[j]);accept(b_c,v_c,already_counted=True)
            else:break
        return b_c,u_c,v_c,fs,vs

    def climb_anchor_2opt(b_in,u_in,v_in,P,use_pol=False):
        nonlocal count
        moved_pair=False
        b_c,u_c,v_c,fs,vs=greedy_1opt(b_in,u_in,v_in)
        if fs is None:
            t=perf_counter_ns();fs=ev.flip_fields(b_c,u_c);vs=ev.score_field(fs)
            ended=perf_counter_ns();timings['evaluation']+=(ended-t)/1e6
            if ended<=deadline:count+=ev.n
        gains=vs-v_c;norm_g=gains/(np.std(gains)+1e-6)
        policy_term=None
        if rank_kind=='uniform':logits=rng.random(ev.n)
        elif rank_kind=='memory':logits=norm_g+w_mem*mem_std
        else:logits=norm_g
        if rank_kind=='neural' and use_pol and policy is not None and perf_counter_ns()+policy_reserve_ns(policy,seen_ms[0])<work_deadline:
            raw=np.asarray(neural_logits(b_c,u_c,gains,norm_g),dtype=float).reshape(-1)
            # The policy is analytically anchored: only its residual is an
            # intervention. Adding the full output here double-counted gains
            # and changed analytical rankings for a zero-residual checkpoint.
            policy_term=raw-norm_g
            logits=_analytical_residual_rank(norm_g,raw,w_logit)
        while perf_counter_ns()+120000<work_deadline:
            if rank_kind=='uniform':sc=rng.random(ev.n)
            else:
                dr=ev.rates(fs)-ev.rates(u_c)[None,:]
                sum_g=dr.sum(axis=-1);norm_sg=sum_g/(np.std(sum_g)+1e-6)
                norm_mg=(vs-v_c)/(np.std(vs-v_c)+1e-6)
                sc=0.55*norm_sg+0.45*norm_mg
                if policy_term is not None:sc=sc+w_logit*policy_term
                if rank_kind=='memory':sc=sc+w_mem*mem_std
            order=np.argsort(sc)[::-1]
            anchors=np.concatenate([order[:P],order[-2:]])
            ap=_anchor_pairs_fast(ev,anchors)
            # Development-only conditional proposals. The query is made on
            # the unchanged parent with original-parent flip gains and a first
            # marker, exactly as in conditional training rows. Zero residual
            # adds no candidates, preserving analytical pair selection.
            ap_eval=ap
            if (conditional_pair_policy and rank_kind=='neural' and policy is not None
                    and not moved_pair and perf_counter_ns()+policy_reserve_ns(policy,seen_ms[0])+450_000<work_deadline):
                first=int(order[0])
                cond=np.asarray(neural_logits(b_c,u_c,gains,norm_g,first=first),dtype=float).reshape(-1)
                cond_res=cond-norm_g
                if cond_res.shape==(ev.n,) and perf_counter_ns()+max(1_000_000,int(pair_eval_ns[0]*1.5))<work_deadline:
                    extra=_conditional_pair_proposals(first,cond,cond_res,ap,limit=4,n=ev.n)
                    if len(extra):ap_eval=np.concatenate([ap,extra],axis=0)
            t=perf_counter_ns()
            d=(1-2*b_c.astype(float))[:,None,None]*ev.delta
            fs2=u_c+d[ap_eval[:,0]]+d[ap_eval[:,1]]
            vs2=ev.score_field(fs2)
            ended=perf_counter_ns();timings['evaluation']+=(ended-t)/1e6
            pair_eval_ns[0]=max(pair_eval_ns[0],ended-t)
            if _score_completed_by_deadline(ended,deadline):count+=len(ap_eval)
            else:break
            j2=int(np.argmax(vs2))
            if vs2[j2]>v_c+1e-12:
                b_c[ap_eval[j2]]^=1;u_c=fs2[j2];v_c=float(vs2[j2]);moved_pair=True
                accept(b_c,v_c,already_counted=True)
                b_c,u_c,v_c,fs,vs=greedy_1opt(b_c,u_c,v_c)
                if fs is None:break
            else:break
        return b_c,u_c,v_c,moved_pair,logits

    # Step 1: climb from the all-OFF start with anchor-partner 2-opt (ranking = variant signal)
    b0,u0,v0,moved_pair,logits=climb_anchor_2opt(b,u,v,P0,use_pol=True)
    bestu=u0 if np.array_equal(best,b0) else ev.field(best)

    # Step 2: memory retrieval and rescoring (only variants that were given memory)
    if memory is not None and len(memory.patterns)>0 and perf_counter_ns()+60000<work_deadline:
        t=perf_counter_ns()
        for cand in memory.patterns[:2]:
            if perf_counter_ns()+40000>=work_deadline:break
            fu=ev.field(cand);fv=float(ev.score_field(fu))
            if perf_counter_ns()<=deadline:count+=1
            if fv>bestv and perf_counter_ns()<=deadline:
                record(cand,fv);bestu=fu
        timings['memory']+=(perf_counter_ns()-t)/1e6

    # Step 3: multi-basin exploration, only when the 1-opt optimum admitted improving pairs
    if moved_pair and perf_counter_ns()+200000<work_deadline:
        seen={b0.tobytes()}
        b1=np.ones(ev.n,dtype=np.int8)
        t=perf_counter_ns();u1=ev.field(b1);v1=float(ev.score_field(u1))
        timings['evaluation']+=(perf_counter_ns()-t)/1e6
        accept(b1,v1)
        if v1>=bestv:bestu=u1
        b1,u1,v1,_,_=greedy_1opt(b1,u1,v1)
        if v1>=bestv:bestu=u1
        if b1.tobytes() not in seen and v1>=bestv-3e-3 and (perf_counter_ns()-start)<t_cap*1e6:
            seen.add(b1.tobytes())
            b1,u1,v1,_,_=climb_anchor_2opt(b1,u1,v1,P1,use_pol=False)
            if v1>=bestv:bestu=u1

        starts=[rng.integers(0,2,ev.n,dtype=np.int8) for _ in range(n_starts)]
        diff=np.flatnonzero(b0!=b1)
        if run_crossover and len(diff)>=4:
            d_ord=diff[np.argsort(logits[diff])[::-1]]
            c=b0.copy();k=max(2,len(d_ord)//2);c[d_ord[:k]]=b1[d_ord[:k]]
            starts.append(c)

        for st in starts:
            if (perf_counter_ns()-start)>t_cap*1e6 or perf_counter_ns()+180000>=work_deadline:break
            bs=st.copy()
            t=perf_counter_ns();us=ev.field(bs);vs_=float(ev.score_field(us))
            timings['evaluation']+=(perf_counter_ns()-t)/1e6
            accept(bs,vs_)
            if vs_>=bestv:bestu=us
            bs,us,vs_,_,_=greedy_1opt(bs,us,vs_)
            if vs_>=bestv:bestu=us
            if bs.tobytes() not in seen and vs_>=bestv-3e-3 and (perf_counter_ns()-start)<(t_cap-0.15)*1e6 and perf_counter_ns()+180000<work_deadline:
                seen.add(bs.tobytes())
                bs,us,vs_,_,_=climb_anchor_2opt(bs,us,vs_,P1,use_pol=False)
                if vs_>=bestv:bestu=us

        # Bounded verification polish on the winning basin elite
        bestu=ev.field(best)
        for _ in range(max_pol):
            if (perf_counter_ns()-start)>(t_cap+0.85)*1e6 or perf_counter_ns()+350000>=work_deadline:break
            t=perf_counter_ns();fs_p=ev.pair_fields(best,bestu);vs_p=ev.score_field(fs_p)
            ended=perf_counter_ns();timings['evaluation']+=(ended-t)/1e6
            if ended<=deadline:count+=len(ev.pairs)
            jp=int(np.argmax(vs_p))
            if vs_p[jp]>bestv+1e-12:
                bp=best.copy();bp[ev.pairs[jp]]^=1;up=fs_p[jp];vp=float(vs_p[jp])
                accept(bp,vp,already_counted=True)
                greedy_1opt(bp,up,vp)
                bestu=ev.field(best)
            else:break

    if memory is not None:
        t=perf_counter_ns();memory.remember(best);timings['memory']+=(perf_counter_ns()-t)/1e6
    elapsed=(perf_counter_ns()-start)/1e6
    timings['other']=max(0.,elapsed-sum(timings.values()))
    return SearchResult(best,bestv,elapsed,count,trace,timings,elapsed>budget_ms)
