"""Complete-controller deadline-aware search on a coupled multiport network."""
from time import perf_counter_ns
import numpy as np
from atomwise_ris.optimization.search import (GA_METHODS,POPULATION_METHOD,ME_GA_METHODS,load_search_config,
    variant_spec,policy_reserve_ns,_conditional_pair_proposals,_score_completed_by_deadline)

def search(ev,method,deadline,seed,memory=(),policy=None,scaling=None,config=None,legacy=False,trace=None,
           conditional_pair_policy=False):
    """Dispatch. GA variants run the shared matched config unless `legacy=True` (explicit, reproduces old runs).

    `trace`, if given, is filled with (elapsed_ms, value, evaluations_so_far) for every improvement.
    Returns (bits, value, evaluations, breakdown) in every mode.
    """
    if method in ME_GA_METHODS:   # explicit opt-in only; ignores config/legacy/policy/scaling (see me_ga.py)
        from atomwise_ris.optimization import me_ga
        return me_ga.search_coupled(ev,method,deadline,seed,memory,trace)
    if method == POPULATION_METHOD:
        if legacy or config is not None:raise ValueError('population_ga has its own fixed controller and no matched config')
        return _search_population(ev,deadline,seed,memory,trace)
    if legacy:
        if config is not None:raise ValueError('legacy mode cannot be combined with a matched config')
        return _search_legacy(ev,method,deadline,seed,memory,policy,scaling,trace)
    if method in GA_METHODS or method in ('greedy','pair','random'):
        cfg=config if config is not None else load_search_config('medium')
        return _search_matched(ev,method,deadline,seed,memory,policy,scaling,cfg,trace,
                               conditional_pair_policy=conditional_pair_policy)
    raise ValueError(method)

def _search_population(ev,deadline,seed,memory=(),trace=None,population_size=12):
    """Standalone population-12 tournament/crossover/mutation baseline."""
    rng=np.random.default_rng(seed); start=perf_counter_ns(); n=ev.n; size=max(4,int(population_size))
    trace=[] if trace is None else trace
    full_cost_ns=300_000;mutation_cost_ns=150_000;completion_reserve_ns=300_000
    initial_eval_start=perf_counter_ns()
    best=ev.state(np.zeros(n,dtype=np.int8)); bestv=float(ev.score(best.field)); count=1
    full_cost_ns=max(full_cost_ns,perf_counter_ns()-initial_eval_start)
    trace.append(((perf_counter_ns()-start)/1e6,bestv,count))
    breakdown={'features_ms':0.,'inference_ms':0.,'memory_ms':0.}
    population=[best.bits.copy()]
    population.extend(np.asarray(b,dtype=np.int8).copy() for b in list(memory)[:8])
    population=population[:size]
    while len(population)<size:population.append(rng.integers(0,2,n,dtype=np.int8))
    values=[bestv];states=[best]
    def score_candidate(bits,parent=None,parent_bits=None,mutation_only=False):
        nonlocal best,bestv,count,full_cost_ns,mutation_cost_ns
        estimate=mutation_cost_ns if mutation_only else full_cost_ns
        if deadline-perf_counter_ns()<=max(200_000,int(estimate*1.75)+completion_reserve_ns):return None
        t=perf_counter_ns()
        if mutation_only:
            changed=np.flatnonzero(np.asarray(bits,dtype=np.int8)!=parent_bits)
            st=ev.flip(parent,changed)
        else:st=ev.state(bits)
        value=float(ev.score(st.field));ended=perf_counter_ns()
        cost=ended-t
        if mutation_only:mutation_cost_ns=max(mutation_cost_ns,cost)
        else:full_cost_ns=max(full_cost_ns,cost)
        if ended>deadline:return None
        count+=1
        if value>bestv:
            best=st;bestv=value;trace.append(((ended-start)/1e6,bestv,count))
        return value,st
    for bits in population[1:]:
        scored=score_candidate(bits)
        if scored is None:break
        value,st=scored;values.append(value);states.append(st)
    population=population[:len(values)]
    states=states[:len(values)]
    while len(population)>=2 and deadline-perf_counter_ns()>max(200_000,int(full_cost_ns*1.75)+completion_reserve_ns):
        elite=int(np.argmax(values))
        next_pop=[population[elite].copy()];next_states=[states[elite]]
        next_values=[float(values[elite])]
        while len(next_pop)<size and deadline-perf_counter_ns()>max(
                200_000,int(full_cost_ns*1.75)+completion_reserve_ns):
            def tournament_index():
                ix=rng.integers(0,len(population),size=min(2,len(population)))
                return int(ix[np.argmax(np.asarray(values)[ix])])
            i1,i2=tournament_index(),tournament_index()
            p1,p2=population[i1],population[i2]
            crossed=rng.random()<0.25
            child=(np.where(rng.integers(0,2,n,dtype=bool),p1,p2).astype(np.int8,copy=False)
                   if crossed else p1.copy())
            n_mut=int(rng.choice((1,2)))
            child[rng.choice(n,size=min(n,n_mut),replace=False)]^=1
            scored=score_candidate(child,parent=states[i1],parent_bits=p1,mutation_only=not crossed)
            if scored is None:break
            value,st=scored
            next_pop.append(child.copy());next_states.append(st);next_values.append(value)
        if len(next_pop)<2:break
        population,states,values=next_pop,next_states,next_values
    return best.bits.copy(),bestv,count,breakdown

def _search_legacy(ev,method,deadline,seed,memory=(),policy=None,scaling=None,trace=None,fixed_counting=False):
    """Pre-matched controller with per-method limits (legacy reproduction and non-GA baselines).

    `fixed_counting` stops a flipped candidate that was already scored inside a batch from being counted twice.
    """
    rng=np.random.default_rng(seed);start=perf_counter_ns();reserve=150_000
    trace=[] if trace is None else trace
    best=ev.state(np.zeros(ev.n,np.int8));bestv=float(ev.score(best.field));count=1
    trace.append(((perf_counter_ns()-start)/1e6,bestv,count))
    breakdown={'features_ms':0.,'inference_ms':0.,'memory_ms':0.}

    def record(st,value):
        nonlocal best,bestv
        best=st;bestv=value;trace.append(((perf_counter_ns()-start)/1e6,value,count))

    def improve(st,value):
        if perf_counter_ns()<deadline and value>bestv:record(st,value)

    def accept(st,value,counted=False):
        nonlocal count
        if perf_counter_ns()>deadline:return
        if not counted:count+=1
        improve(st,value)

    ga=method.endswith('_ga')

    if not ga:
        # Original baseline algorithms (preserve exact pair/greedy/random reference behavior)
        t_mem=perf_counter_ns()
        for bits in memory:
            if perf_counter_ns()+150_000>=deadline:break
            st=ev.state(bits);accept(st,float(ev.score(st.field)))
        breakdown['memory_ms']=(perf_counter_ns()-t_mem)/1e6
        st=best;v=bestv;estimate=800_000
        while perf_counter_ns()+estimate+reserve<deadline:
            t=perf_counter_ns()
            if method in ['greedy','pair']:
                scores=ev.moves(st);count+=ev.n;j=int(np.argmax(scores));indices=[j]
                if scores[j]<=v+1e-10:
                    if method=='pair' and perf_counter_ns()+2_000_000+reserve<deadline:
                        scores=ev.moves(st,True);count+=len(scores);j=int(np.argmax(scores));indices=ev.pairs[j]
                    if scores[j]<=v+1e-10:
                        st=ev.state(rng.integers(0,2,ev.n,dtype=np.int8));v=float(ev.score(st.field));accept(st,v)
                        estimate=max(estimate,int((perf_counter_ns()-t)*1.25));continue
                if perf_counter_ns()+300_000<deadline:
                    st=ev.flip(st,indices);v=float(ev.score(st.field));accept(st,v,counted=fixed_counting)
            elif method=='random':
                q=ev.state(rng.integers(0,2,ev.n,dtype=np.int8));accept(q,float(ev.score(q.field)))
            else:raise ValueError(method)
            estimate=max(estimate,int((perf_counter_ns()-t)*1.25))
        return best.bits.copy(),bestv,count,breakdown

    # Memory-Enhanced Dynamic Evolutionary Controller (Zardi, Oliveri, & Massa, IEEE TAP 2024)
    budget_ms=(deadline-start)/1e6
    if method=='attention_ga':
        K,w_res,hop_K,hop_idx,extra_hop=28,0.12,20,[0,1],True
    elif method=='mlp_ga':
        K,w_res,hop_K,hop_idx,extra_hop=24,0.07,18,[0,1],False
    elif method=='analytical_ga':
        K,w_res,hop_K,hop_idx,extra_hop=20,0.0,16,[0,1],False
    elif method=='attention_no_physics_ga':
        K,w_res,hop_K,hop_idx,extra_hop=22,0.05,14,[0,2],False
    elif method=='attention_no_memory_ga':
        K,w_res,hop_K,hop_idx,extra_hop=24,0.12,16,[0,1],False
    else: # memory_ga
        K,w_res,hop_K,hop_idx,extra_hop=18,0.0,14,[0,1],False

    cached_logits=None
    def get_logits(st,v,allow_policy=True):
        nonlocal breakdown,cached_logits,count
        rem_ms=(deadline-perf_counter_ns())/1e6
        if allow_policy and policy is not None and method not in ('memory_ga','analytical_ga') and rem_ms>0.45:
            ft=perf_counter_ns();x,g=ev.features(st);count+=ev.n
            if scaling is not None:x=(x-scaling[0])/scaling[1]
            breakdown['features_ms']+=(perf_counter_ns()-ft)/1e6
            gn=g/max(float(np.std(g)),1e-4)
            it=perf_counter_ns();out=np.asarray(policy(x,gn),dtype=float).squeeze()
            breakdown['inference_ms']+=(perf_counter_ns()-it)/1e6
            res=out-gn;res_norm=res/max(float(np.std(res)),1e-4)
            cached_logits=gn+w_res*res_norm
            return cached_logits
        g=ev.moves(st)-v;count+=ev.n
        if method=='memory_ga' and len(memory)>0:
            return g+0.1*np.std(np.array(memory),axis=0)
        return g

    # Fast 1-ms path: 1-opt step + Woodbury memory rescoring + single policy pass
    if budget_ms<=1.5:
        if perf_counter_ns()+180_000<deadline:
            sc=ev.moves(best);count+=ev.n;j=int(np.argmax(sc))
            if sc[j]>bestv+1e-10 and perf_counter_ns()+120_000<deadline:
                st1=ev.flip(best,[j]);v1=float(ev.score(st1.field));accept(st1,v1)
        t_mem=perf_counter_ns()
        for bits in memory[:2]:
            if perf_counter_ns()+180_000>=deadline:break
            diff=np.flatnonzero(bits!=best.bits)
            if len(diff)==0:continue
            stm=ev.flip(best,diff) if len(diff)<=20 else ev.state(bits)
            vm=float(ev.score(stm.field));accept(stm,vm)
        breakdown['memory_ms']+=(perf_counter_ns()-t_mem)/1e6
        if policy is not None and method not in ('memory_ga','analytical_ga') and perf_counter_ns()+320_000<deadline:
            get_logits(best,bestv,allow_policy=True)
        return best.bits.copy(),bestv,count,breakdown

    def climb(st,v,sub_K,use_pol=False):
        nonlocal count
        while perf_counter_ns()+200_000<deadline:
            while perf_counter_ns()+150_000<deadline:
                sc=ev.moves(st);count+=ev.n;j=int(np.argmax(sc))
                if sc[j]>v+1e-10:
                    st=ev.flip(st,[j]);v=float(ev.score(st.field))
                    improve(st,v)
                else:break
            if perf_counter_ns()+350_000>=deadline:break
            logits=get_logits(st,v,allow_policy=(use_pol and st is best))
            sub=np.argsort(logits)[::-1][:sub_K]
            sc_sub,p_sub=ev.moves_subspace(st,sub);count+=len(sc_sub)
            if len(sc_sub)==0:break
            bp=int(np.argmax(sc_sub))
            if sc_sub[bp]>v+1e-10 and perf_counter_ns()+150_000<deadline:
                st=ev.flip(st,p_sub[bp]);v=float(ev.score(st.field))
                improve(st,v)
            else:break
        return st,v

    # Step 1: Climb from initial all-OFF reference state
    st,v=climb(best,bestv,K,use_pol=True)

    # Step 2: Fast 1-Opt Screened Random Immigrant when trapped on all-OFF (e.g. strict 2-opt trap)
    if np.all(best.bits==0) and (deadline-perf_counter_ns())/1e6>4.2:
        st_r=ev.state(rng.integers(0,2,ev.n,dtype=np.int8));vr=float(ev.score(st_r.field));accept(st_r,vr)
        while perf_counter_ns()+180_000<deadline:
            sc=ev.moves(st_r);count+=ev.n;j=int(np.argmax(sc))
            if sc[j]>vr+1e-10:
                st_r=ev.flip(st_r,[j]);vr=float(ev.score(st_r.field))
                improve(st_r,vr)
            else:break
        if vr>bestv-1.2:
            for _ in range(4):
                if perf_counter_ns()+900_000>=deadline:break
                sc2=ev.moves(st_r,True);count+=len(sc2);bp=int(np.argmax(sc2))
                if sc2[bp]>vr+1e-10:
                    st_r=ev.flip(st_r,ev.pairs[bp]);vr=float(ev.score(st_r.field))
                    improve(st_r,vr)
                    while perf_counter_ns()+180_000<deadline:
                        sc=ev.moves(st_r);count+=ev.n;j=int(np.argmax(sc))
                        if sc[j]>vr+1e-10:
                            st_r=ev.flip(st_r,[j]);vr=float(ev.score(st_r.field))
                            improve(st_r,vr)
                        else:break
                else:break

    # Step 3: Fast Woodbury Memory Rescoring & Subspace Refinement
    if len(memory)>0 and perf_counter_ns()+400_000<deadline:
        t_mem=perf_counter_ns()
        mem_depth=4 if budget_ms<=15.0 else 6
        for b_mem in memory[:mem_depth]:
            if perf_counter_ns()+450_000>=deadline:break
            diff=np.flatnonzero(b_mem!=best.bits)
            if len(diff)==0:continue
            st_m=ev.flip(best,diff) if len(diff)<=20 else ev.state(b_mem)
            vm=float(ev.score(st_m.field));accept(st_m,vm)
            st_m,vm=climb(st_m,vm,min(K,20),use_pol=False)
        breakdown['memory_ms']+=(perf_counter_ns()-t_mem)/1e6

    # Step 4: Policy/Gain-Guided 2-Bit Basin Hopping
    if perf_counter_ns()+500_000<deadline:
        g=ev.moves(best)-bestv;count+=ev.n
        ranked=np.argsort(g)[::-1]
        cand=ev.flip(best,ranked[hop_idx]);cv=float(ev.score(cand.field));accept(cand,cv)
        cand,cv=climb(cand,cv,hop_K,use_pol=False)

    # Step 5: Attention-Guided Secondary Basin Exploration (for attention_ga)
    if extra_hop and policy is not None and perf_counter_ns()+650_000<deadline:
        base_hop=best
        pol_logits=get_logits(base_hop,bestv,allow_policy=True)
        ranked_pol=np.argsort(pol_logits)[::-1]
        cand2=ev.flip(base_hop,ranked_pol[:2]);cv2=float(ev.score(cand2.field));accept(cand2,cv2)
        cand2,cv2=climb(cand2,cv2,hop_K,use_pol=False)
        if perf_counter_ns()+650_000<deadline:
            cand3=ev.flip(base_hop,ranked_pol[1:3]);cv3=float(ev.score(cand3.field));accept(cand3,cv3)
            cand3,cv3=climb(cand3,cv3,hop_K,use_pol=False)

    return best.bits.copy(),bestv,count,breakdown


def _search_matched(ev,method,deadline,seed,memory,policy,scaling,config,trace,
                    conditional_pair_policy=False):
    """Matched-protocol controller.

    Evaluation counting: one per candidate configuration scored. Single-flip scores are cached per
    state, so ranking reuses scores the search already paid for; neural feature extraction
    recomputes them but that is reported as time, not as evaluations. Variants differ only in
    ranking signal (`rank`) and in whether `memory` is supplied.
    """
    ga=method in GA_METHODS
    rng=np.random.default_rng(seed);start=perf_counter_ns();reserve=150_000
    trace=[] if trace is None else trace
    best=ev.state(np.zeros(ev.n,np.int8));bestv=float(ev.score(best.field));count=1
    trace.append(((perf_counter_ns()-start)/1e6,bestv,count))
    breakdown={'features_ms':0.,'inference_ms':0.,'memory_ms':0.}

    def record(st,value):
        """Single point where an improvement is stored and traced."""
        nonlocal best,bestv
        best=st;bestv=value;trace.append(((perf_counter_ns()-start)/1e6,value,count))

    def improve(st,value):
        if perf_counter_ns()<deadline and value>bestv:record(st,value)

    def accept(st,value,counted=False):
        nonlocal count
        if perf_counter_ns()>deadline:return
        if not counted:count+=1
        improve(st,value)

    if not ga:
        # Baselines: same search as legacy; a flipped candidate already scored in a batch is not recounted.
        t_mem=perf_counter_ns()
        for bits in memory:
            if perf_counter_ns()+150_000>=deadline:break
            st=ev.state(bits);accept(st,float(ev.score(st.field)))
        breakdown['memory_ms']=(perf_counter_ns()-t_mem)/1e6
        st=best;v=bestv;estimate=800_000
        while perf_counter_ns()+estimate+reserve<deadline:
            t=perf_counter_ns()
            if method in ['greedy','pair']:
                scores=ev.moves(st);count+=ev.n;j=int(np.argmax(scores));indices=[j]
                if scores[j]<=v+1e-10:
                    if method=='pair' and perf_counter_ns()+2_000_000+reserve<deadline:
                        scores=ev.moves(st,True);count+=len(scores);j=int(np.argmax(scores));indices=ev.pairs[j]
                    if scores[j]<=v+1e-10:
                        st=ev.state(rng.integers(0,2,ev.n,dtype=np.int8));v=float(ev.score(st.field));accept(st,v)
                        estimate=max(estimate,int((perf_counter_ns()-t)*1.25));continue
                if perf_counter_ns()+300_000<deadline:
                    st=ev.flip(st,indices);v=float(ev.score(st.field));accept(st,v,counted=True)
            elif method=='random':
                q=ev.state(rng.integers(0,2,ev.n,dtype=np.int8));accept(q,float(ev.score(q.field)))
            else:raise ValueError(method)
            estimate=max(estimate,int((perf_counter_ns()-t)*1.25))
        return best.bits.copy(),bestv,count,breakdown

    spec=variant_spec(method);rank_kind=spec['ranking']
    if not spec['memory']:memory=()
    memory=list(memory)
    lim=config.coupled_limits();K,hop_K,hop_idx=lim['K'],lim['hop_K'],lim['hop_idx']
    w_res=float(config.shared['neural_residual_weight_coupled']);w_mem=float(config.shared['memory_bias_weight'])
    secondary_hops=bool(config.shared['secondary_hops'])
    budget_ms=(deadline-start)/1e6

    mem_std=np.zeros(ev.n)
    if rank_kind=='memory' and len(memory)>1:
        t=perf_counter_ns();mem_std=np.std(np.array(memory),axis=0);breakdown['memory_ms']+=(perf_counter_ns()-t)/1e6

    seen_ms=[0.]   # largest observed neural pass (features + inference) in this run
    conditional_pair_ns=[1_000_000]
    flips=[None,None]   # (state, single-flip scores) of the most recently scored state
    ranks=[None,None,None]   # (state, policy allowed, logits)

    def single_scores(st):
        nonlocal count
        if flips[0] is st:return flips[1]
        sc=ev.moves(st)
        if perf_counter_ns()<=deadline:count+=ev.n
        flips[0]=st;flips[1]=sc
        return sc

    def rank(st,v,allow_policy):
        """Variant ranking over atoms (higher = better candidate); the only place variants differ."""
        nonlocal count
        if rank_kind=='uniform':return rng.random(ev.n)
        if ranks[0] is st and ranks[1]==allow_policy:return ranks[2]
        rem_ms=(deadline-perf_counter_ns())/1e6
        if rank_kind=='neural' and allow_policy and policy is not None and rem_ms>policy_reserve_ns(policy,seen_ms[0],450_000)/1e6:
            # Reuse the same full single-flip scores used by analytical ranking;
            # feature extraction must not silently score a duplicate candidate set.
            gains=single_scores(st)-v
            ft=perf_counter_ns();x,g=ev.features(st,gains=gains)
            if scaling is not None:x=(x-scaling[0])/scaling[1]
            breakdown['features_ms']+=(perf_counter_ns()-ft)/1e6
            gn=g/max(float(np.std(g)),1e-4)
            it=perf_counter_ns();out=np.asarray(policy(x,gn),dtype=float).squeeze()
            breakdown['inference_ms']+=(perf_counter_ns()-it)/1e6
            seen_ms[0]=max(seen_ms[0],(perf_counter_ns()-ft)/1e6)
            res=out-gn;res_norm=res/max(float(np.std(res)),1e-4)
            logits=gn+w_res*res_norm
        else:
            g=single_scores(st)-v
            logits=g/max(float(np.std(g)),1e-4)
            if rank_kind=='memory':logits=logits+w_mem*mem_std
        ranks[0]=st;ranks[1]=allow_policy;ranks[2]=logits
        return logits

    # Fast 1-ms path: identical sequence for every variant (1-opt step, memory rescoring, one ranking pass).
    if budget_ms<=1.5:
        if perf_counter_ns()+180_000<deadline:
            sc=single_scores(best);j=int(np.argmax(sc))
            if sc[j]>bestv+1e-10 and perf_counter_ns()+120_000<deadline:
                st1=ev.flip(best,[j]);v1=float(ev.score(st1.field));accept(st1,v1,counted=True)
        t_mem=perf_counter_ns()
        for bits in memory[:2]:
            if perf_counter_ns()+180_000>=deadline:break
            diff=np.flatnonzero(bits!=best.bits)
            if len(diff)==0:continue
            stm=ev.flip(best,diff) if len(diff)<=20 else ev.state(bits)
            vm=float(ev.score(stm.field));accept(stm,vm)
        breakdown['memory_ms']+=(perf_counter_ns()-t_mem)/1e6
        if perf_counter_ns()+(policy_reserve_ns(policy,0.,320_000) if rank_kind=='neural' and policy is not None else 320_000)<deadline:rank(best,bestv,True)
        return best.bits.copy(),bestv,count,breakdown

    def climb(st,v,sub_K,use_pol=False):
        nonlocal count
        conditional_queried=False
        while perf_counter_ns()+200_000<deadline:
            while perf_counter_ns()+150_000<deadline:
                sc=single_scores(st);j=int(np.argmax(sc))
                if sc[j]>v+1e-10:
                    st=ev.flip(st,[j]);v=float(sc[j]);improve(st,v)
                else:break
            if perf_counter_ns()+350_000>=deadline:break
            logits=rank(st,v,use_pol and st is best)
            sub=np.argsort(logits)[::-1][:sub_K]
            sc_sub,p_sub=ev.moves_subspace(st,sub)
            if perf_counter_ns()<=deadline:count+=len(sc_sub)
            if len(sc_sub)==0:break
            best_score=float(np.max(sc_sub));best_pair=p_sub[int(np.argmax(sc_sub))];cached_pair_state=None
            # Optional development policy: condition the same ranker on the
            # first anchor, using gains from this original parent state. Exact
            # pair scores are evaluated only for a bounded set of new pairs.
            if (conditional_pair_policy and not conditional_queried and use_pol
                    and rank_kind=='neural' and policy is not None
                    and perf_counter_ns()+policy_reserve_ns(policy,seen_ms[0],450_000)+350_000<deadline):
                conditional_queried=True
                first=int(sub[0]);gains=single_scores(st)-v
                gn=gains/max(float(np.std(gains)),1e-4)
                ft=perf_counter_ns();x,_=ev.features(st,first=first,gains=gains)
                if scaling is not None:x=(x-scaling[0])/scaling[1]
                breakdown['features_ms']+=(perf_counter_ns()-ft)/1e6
                it=perf_counter_ns();out=np.asarray(policy(x,gn),dtype=float).reshape(-1)
                breakdown['inference_ms']+=(perf_counter_ns()-it)/1e6
                seen_ms[0]=max(seen_ms[0],(perf_counter_ns()-ft)/1e6)
                residual=out-gn
                if out.shape==(ev.n,) and perf_counter_ns()+350_000<deadline:
                    proposed=_conditional_pair_proposals(first,out,residual,p_sub,limit=4,n=ev.n)
                    for pair in proposed:
                        if perf_counter_ns()+350_000>=deadline:break
                        if perf_counter_ns()+max(1_000_000,int(conditional_pair_ns[0]*1.5))>=deadline:break
                        t_pair=perf_counter_ns();candidate=ev.flip(st,pair);value=float(ev.score(candidate.field));ended_pair=perf_counter_ns()
                        conditional_pair_ns[0]=max(conditional_pair_ns[0],ended_pair-t_pair)
                        if not _score_completed_by_deadline(ended_pair,deadline):break
                        count+=1
                        if value>best_score:
                            best_score=value;best_pair=np.asarray(pair,dtype=int);cached_pair_state=candidate
                        improve(candidate,value)
            if best_score>v+1e-10 and perf_counter_ns()+150_000<deadline:
                st=cached_pair_state if cached_pair_state is not None else ev.flip(st,best_pair)
                v=best_score;improve(st,v)
            else:break
        return st,v

    # Step 1: climb from the all-OFF reference state
    st,v=climb(best,bestv,K,use_pol=True)

    # Step 2: random immigrant when trapped on all-OFF
    if np.all(best.bits==0) and (deadline-perf_counter_ns())/1e6>4.2:
        st_r=ev.state(rng.integers(0,2,ev.n,dtype=np.int8));vr=float(ev.score(st_r.field));accept(st_r,vr)
        def climb_1opt(s_,v_):
            while perf_counter_ns()+180_000<deadline:
                sc=single_scores(s_);j=int(np.argmax(sc))
                if sc[j]>v_+1e-10:
                    s_=ev.flip(s_,[j]);v_=float(ev.score(s_.field));improve(s_,v_)
                else:break
            return s_,v_
        st_r,vr=climb_1opt(st_r,vr)
        if vr>bestv-1.2:
            for _ in range(4):
                if perf_counter_ns()+900_000>=deadline:break
                sc2=ev.moves(st_r,True)
                if perf_counter_ns()<=deadline:count+=len(sc2)
                bp=int(np.argmax(sc2))
                if sc2[bp]>vr+1e-10:
                    st_r=ev.flip(st_r,ev.pairs[bp]);vr=float(ev.score(st_r.field));improve(st_r,vr)
                    st_r,vr=climb_1opt(st_r,vr)
                else:break

    # Step 3: memory rescoring and subspace refinement (empty for no-memory variants)
    if len(memory)>0 and perf_counter_ns()+400_000<deadline:
        t_mem=perf_counter_ns()
        mem_depth=4 if budget_ms<=15.0 else 6
        for b_mem in memory[:mem_depth]:
            if perf_counter_ns()+450_000>=deadline:break
            diff=np.flatnonzero(b_mem!=best.bits)
            if len(diff)==0:continue
            st_m=ev.flip(best,diff) if len(diff)<=20 else ev.state(b_mem)
            vm=float(ev.score(st_m.field));accept(st_m,vm)
            st_m,vm=climb(st_m,vm,min(K,20),use_pol=False)
        breakdown['memory_ms']+=(perf_counter_ns()-t_mem)/1e6

    # Step 4: basin hop on the variant's own top-ranked atoms
    if perf_counter_ns()+500_000<deadline:
        ranked=np.argsort(rank(best,bestv,True))[::-1]
        cand=ev.flip(best,ranked[hop_idx]);cv=float(ev.score(cand.field));accept(cand,cv)
        cand,cv=climb(cand,cv,hop_K,use_pol=False)

    # Step 5: secondary hops, run for every variant with its own ranking
    if secondary_hops and perf_counter_ns()+650_000<deadline:
        base_hop=best
        ranked=np.argsort(rank(base_hop,bestv,True))[::-1]
        cand2=ev.flip(base_hop,ranked[:2]);cv2=float(ev.score(cand2.field));accept(cand2,cv2)
        cand2,cv2=climb(cand2,cv2,hop_K,use_pol=False)
        if perf_counter_ns()+650_000<deadline:
            cand3=ev.flip(base_hop,ranked[1:3]);cv3=float(ev.score(cand3.field));accept(cand3,cv3)
            cand3,cv3=climb(cand3,cv3,hop_K,use_pol=False)

    return best.bits.copy(),bestv,count,breakdown
