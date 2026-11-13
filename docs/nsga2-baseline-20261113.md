# NSGA-II reference baseline: design, tuning rule and readiness criteria (2026-11-13)

Declared 2026-11-13 ~03:50 WIB, **before any tuning run**. The rules in sections 1–5 are not changed after the
pre-run hash (`nsga2-baseline-20261113.prerun.sha256`); results are appended below them.

## 1. Method

- Source: Deb, Pratap, Agarwal, Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II",
  IEEE TEVC 6(2), 2002, doi 10.1109/4235.996017.
- Module: `src/atomwise_ris/optimization/nsga2.py` (new). Method name `nsga2`.
- Chromosome b ∈ {0,1}^N. Objectives: the K per-user rates R_k(b), all maximised, under the same fixed
  precoder and the same evaluator object as every other method. Returned action: the best max-min
  individual (largest min_k R_k) seen in any evaluation of the call (anytime).
- Operators: fast non-dominated sort, crowding distance, binary tournament with the crowded-comparison
  operator, elitist (μ+λ) environmental selection, **uniform crossover** (gene swap prob. 0.5) applied with
  probability pc, bit-flip mutation at rate **1/N**. Identical genomes are scored once per call (exact reuse).
- Initial population: **all-OFF + (P−1) uniformly random** individuals; the all-OFF individual is the single
  mandatory first evaluation. *Difference from ME-GA:* `me_ga.py` draws its initial population purely at random
  (Eq. 12, no all-OFF seed; see `outputs/frozen/matched_v2.json` `known_baseline_behaviour`), so the brief's
  "all-OFF plus random, exactly as ME-GA" cannot both hold. The explicit all-OFF seed is used.
- Evaluation: ME-GA's `CoupledPopulation` (batched stacked solves; vectorised rank-m Woodbury for children
  within 24 flips of the current best member, used when ≥ 12 children qualify), returning per-user rates.
  min_k of these rates is bit-identical to `ev.score`.
- Deadline: ME-GA's chunked guard (chunk 25, calibrated per-individual cost × 1.5 + 0.15 ms reserve; a chunk
  that ends after the deadline is discarded); the selection step is also guarded. Clock: the runner's
  `search_elapsed_ms`.
- Frozen / hash-locked files are not edited. Registration is by new code paths only:
  `GeoRIS-Bench/src/atomwise_ris/optimization/reference_methods.py` (coupled dispatch: nsga2, else the
  unchanged `coupled_search.search`) and `paper-b-neural-pair-search/src/atomwise_ris/bench/search_adapter_nsga2.py`
  (nsga2, else the unchanged `search_adapter.run_search`). The free-space (Paper A) path was cancelled by the lead
  on 2026-11-13 before any free-space run.

## 2. Tuning data (development only)

- RIS-relevant pilot **v1, train split** `rotterdam-rr1-0000` … `0033` (34 trajectories), all non-flagged
  frames, arrays 64 / 128 / 256 (separate cells), budgets 5 / 10 / 50 / 100 ms, seeds 7 and 19,
  search seed `seed × 1000003 + frame_row_index` (as the benchmark runner).
- Never opened: test split 0041–0047, Berlin/Utrecht, natural-scope test, challenge, `coupled_wire_test_*`.
- Node labkc-comm3, interpreter `/mnt/home/faris/projects/atom-wise-ris/.venv-em/bin/python` (3.12.3,
  numpy 2.2.6), single-thread BLAS, three processes pinned to CPUs 1, 2, 3 (SMT siblings 7, 8, 9 idle).
  CPU 5 / 11 (benchmark timing core) are not used.

## 3. Grid and selection rule

Grid: P ∈ {20, 50, 100} × pc ∈ {0.7, 0.9} (6 configs). Everything else fixed as in section 1.

Per (config, array, budget) cell: **M1** = mean over trajectories of the within-trajectory mean worst-user SE
(frames and seeds averaged within a trajectory); miss fraction with a miss = `search_elapsed_ms > budget`.

1. **Feasible**: miss fraction ≤ 0.5 % in every one of the 12 (array, budget) cells. If no config is feasible,
   use those feasible in the three 50 ms cells; if none, the config with the fewest total misses.
2. **Score** S = mean over the three arrays of M1 at **50 ms** (primary budget).
3. Choose the feasible config with the largest S. If the top two differ by < 0.002 bit/s/Hz, choose the one with
   the larger mean M1 over all 12 cells; if still within 0.002, the smaller P; then the larger pc.
4. The chosen values are written into `CHOSEN_PARAMS` in `nsga2.py`; its sha256 then goes into
   `NSGA2_READY.json`. No later change without a new dated declaration.

## 4. Readiness criteria (declared before the run)

- R1: all `tests/test_nsga2.py` tests pass with the comm3 interpreter (non-dominated sort and crowding
  correctness, deadline respect, per-seed determinism, evaluator rates = direct solves at rtol 1e-9).
- R2: the tuning run completes with zero rescoring errors (the runner stops if |reported − rescored| > 1e-7).
- `NSGA2_READY.json` is written when R1 and R2 hold and the rule of section 3 has been applied.

## 5. Validation report (descriptive)

After the lock, nsga2 with `CHOSEN_PARAMS` runs once on v1 **validation** 0034–0040 (all frames, 3 arrays,
4 budgets, the 10 registered seeds), one process pinned to CPU 4 (sibling 10 idle), CPU 5 untouched.
Reported: M1 worst-user SE, M2 latency (mean / p95 of `search_elapsed_ms`), M3 evaluations per call and misses
per cell. V1: miss fraction ≤ 0.5 % per (array, budget) cell; reported as pass/fail, never used to retune.
v1 failed its leverage gate, so these are development numbers only.

---

## Results of the tuning run (appended 2026-11-13 08:20 WIB; sections 1–5 above unchanged, pre-run sha256 `d7b3d34d…c817`)

- Train grid finished on labkc-comm3 (CPUs 1/2/3; 3 × 26,112 rows; 0 rescoring errors → **R2 PASS**). **R1 PASS** (13/13 on the comm3 interpreter).
- No config met the all-12-cells miss rule (every config misses at 256 / 5 and 10 ms, see A1), so rule step 1 fell to the
  three 50 ms cells (all six configs feasible there).
- S (mean M1 at 50 ms over 64/128/256): P20/pc0.9 6.4203, P20/pc0.7 6.4197, P50/pc0.9 6.4087, P50/pc0.7 6.4070,
  P100/pc0.9 6.3980, P100/pc0.7 6.3964. Top two within 0.002 → mean12 6.3924 vs 6.3914 (within 0.002) → smaller P tie →
  larger pc. **Chosen: P = 20, pc = 0.9** (full table: `runs/tune/tuning_selection.json` in the comm3 workspace).

## Amendment A1: deadline-guard fix (declared 2026-11-13 08:20 WIB, after tuning, before any validation run)

The tuning rows showed a defect in the deadline guard copied from ME-GA's logic (engineering, not a hyper-parameter):
(a) at N = 256 the first, uncalibrated 8-individual chunk was admitted on an estimate of 2 × the single-evaluation cost
but costs ~3 ×, so it finished after the deadline and was discarded (100 % misses at 256 / 10 ms, evaluations = 1);
(b) a 25-individual chunk that did not fit ended the search early (e.g. 128 / 10 ms with P = 50 stopped at ~2.2 ms);
(c) calibration on a cheap Woodbury chunk under-estimated later solve chunks (P20 at 128 / 100 ms: 13 / 1088 misses).

Fix in `nsga2.py` (only `evaluate_unique` and the per-generation guard): the chunk is shrunk to what fits in the
remaining time at 1.5 × the per-individual estimate + 0.15 ms reserve; the estimate is the single-evaluation cost until
calibrated, then the maximum per-individual cost over chunks of ≥ 4; a generation starts if ≥ 1 child fits.
**Hyper-parameters stay P = 20, pc = 0.9 (no retune, whatever the re-check shows).**

R3 (re-check, declared now): the locked module on the same **train** grid cells (seeds 7, 19; CPUs 1/2/3); reported per
cell; criterion: miss fraction ≤ 0.5 % in every (array, budget) cell. If R3 fails in some cell, `NSGA2_READY.json` is
still written and the failing cells are listed in it; the benchmark's own 0.5 % rule then excludes nsga2 there.

### R3 result for A1 (appended 2026-11-13 08:30 WIB): **FAIL**

Train re-check of the A1 module (sha256 `a85b8555…d996`): misses 64/50 ms 339/1088, 64/100 ms 852/1088 (overruns ≤ 0.52 ms),
128/5 ms 14/1088, 256/5 ms 49/1088, 256/10 ms 31/1088, 256/50 ms 84/1088 and 256/100 ms 74/1088 (max 108.2 ms); the other cells
had ≤ 1 miss. A1 filled the budget to within ~0.1 ms and under-counted (i) the fixed cost of small end-of-search chunks, which
also switch from the Woodbury to the solve path below 12 children, and (ii) the full base-state inverse that the first Woodbury
call on a new base computes (≈ several ms at N = 256).

## Amendment A2 (declared 2026-11-13 08:30 WIB, before its re-check; hyper-parameters unchanged, P = 20, pc = 0.9)

- Chunk estimate = 1.5 × per-individual cost × size + one single-call cost + end reserve; end reserve = max(0.3 ms,
  1 % of the budget) (the 0.3 ms floor is `search.policy_reserve_ns`'s, used by the other methods).
- The Woodbury base inverse is computed explicitly before a generation's children, under its own guard
  (estimate: max measured, initially 4 × the single-call cost); if it does not fit, that generation uses the solve path.
- Generation and selection guards use the same reserve.
- R3′ = R3 rerun with the A2 module on the same train cells; same criterion and the same consequence (no retune; failing cells
  listed in `NSGA2_READY.json`). The descriptive validation run of section 5 is rerun with the A2 module after R3′; the partial
  validation run of the A1 module was stopped and is not reported.

### R3′ (A2 module, sha256 `fec0a3d7…7260`), train, appended 2026-11-13 08:35 WIB: **PASS** — 0 misses in all 12 cells (13,056 rows).

`NSGA2_READY.json` written 2026-11-13 ~08:35 WIB on labkc-comm3 (`/var/tmp/atomwise-bench-ris-relevant-20261112/nsga2/`); the copy of
this document next to it is the version up to R3′ (its sha256 is recorded in the ready file); this section was appended after.

### Section 5 validation (v1 validation 0034–0040, 7 trajectories × 16 frames, 0 flagged, 10 seeds; CPU 4, sibling 10 idle), appended 08:45 WIB

Descriptive; development numbers only (v1 failed its leverage gate). M1 = worst-user SE (trajectory mean), gain = M1 − all-OFF,
M2 = `search_elapsed_ms`, M3 = evaluations per call. **V1 PASS** (0 misses in every cell; 1,120 runs per cell).

| Array | Budget ms | M1 bit/s/Hz | gain vs all-OFF | M2 mean ms | M2 p95 ms | M3 evals (mean) | misses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5 | 4.6606 | +0.0897 | 4.52 | 4.69 | 38 | 0/1120 |
| 64 | 10 | 4.6841 | +0.1132 | 9.43 | 9.59 | 78 | 0/1120 |
| 64 | 50 | 4.7422 | +0.1713 | 49.36 | 49.50 | 530 | 0/1120 |
| 64 | 100 | 4.7633 | +0.1923 | 98.97 | 99.15 | 1168 | 0/1120 |
| 128 | 5 | 5.0962 | +0.0287 | 3.86 | 4.09 | 8 | 0/1120 |
| 128 | 10 | 5.1107 | +0.0432 | 8.87 | 9.15 | 20 | 0/1120 |
| 128 | 50 | 5.1470 | +0.0795 | 48.70 | 49.01 | 112 | 0/1120 |
| 128 | 100 | 5.1658 | +0.0983 | 98.21 | 98.47 | 231 | 0/1120 |
| 256 | 5 | 4.4693 | +0.0000 | 2.96 | 3.10 | 1 | 0/1120 |
| 256 | 10 | 4.4836 | +0.0143 | 5.35 | 5.57 | 2 | 0/1120 |
| 256 | 50 | 4.5284 | +0.0591 | 44.97 | 46.17 | 19 | 0/1120 |
| 256 | 100 | 4.5416 | +0.0723 | 94.32 | 95.58 | 40 | 0/1120 |

At N = 256, 5 ms only the mandatory all-OFF evaluation fits (nsga2 returns all-OFF); at 50 ms the initial population of 20 is
barely complete, so NSGA-II there is close to random sampling around all-OFF.
