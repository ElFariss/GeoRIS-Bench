# CoupledRIS-Bench RIS-relevant scope v2: pilot declaration (2026-11-13 WIB)

**Status:** declared **before** any v2 generation, tracing or channel build. Sections 1–9 are fixed and never loosened.
Results are appended only below "Results".

## 0. Why v2: the v1 failure, stated openly

The v1 pilot (`docs/bench-ris-relevant-scope-v1-20261112.md`; Rotterdam, 48 trajectories, seed word 20261112)
**FAILED** its pre-registered leverage gate on validation (0034–0040).

| v1 gate | 64 | 128 | 256 | v1 threshold |
|---|---:|---:|---:|---|
| L1a / L1b (pathless served user-frames / flagged frames) | 0% / 0% ✓ | 0% / 0% ✓ | 0% / 0% ✓ | ≤ 2% / ≤ 5% |
| L2 median (best − all OFF) | 0.0741 ✓ | 0.1401 ✓ | **0.1340 ✗** | ≥ 0.05 / 0.07 / 0.14 |
| L3 median (all OFF − no RIS) | **0.0038 ✗** | **0.0036 ✗** | **0.0059 ✗** | ≥ 0.06 / 0.08 / 0.14 |

**Diagnosis (descriptive, from the v1 results section):**
- v1 brought the RIS close to the BS (d1 median 18.9 m). It did not bring the users close to the RIS: d2 had a median of 121 m
  over all served user-frames, and 89–232 m by class on validation.
- At the best setting the RIS path stayed 33–52 dB below the BS's diffracted or reflected paths.
- The ideal-plate bound for a BS-blocked user is `A²/(λ² d1²)·(d/d2)²`. With d2 ≈ d it no longer depends on d2, but the
  *actual* path loss of the BS's NLOS paths at short range does: users near the RIS are those whose diffracted BS path is
  weakest compared with the RIS path.

**v1 is closed:**
- Its verdict, thresholds and data are unchanged.
- Its sealed test (rr1 0041–0047) stays unopened.
- v2 is a new rule set with a new seed word, as v1 §8.6 requires.

## 1. Scope identity

- **Scope name:** `ris_relevant_v2_pilot`, in the new modules `src/atomwise_ris/bench/ris_relevant_v2.py` and
  `ris_relevant_scope_v2.py`. These wrap the v1 wrapper, which wraps the frozen v1.1 generator.
- **Unchanged:** `PRODUCTION_SCOPES`, the v1 registry, and every frozen or hash-locked file (the v1 list in §0 of the v1 doc).
- **Case ids:** `rotterdam-rr2-0000` … `rotterdam-rr2-0047`.
- **Manifest:** `outputs/runs/bench_ris_relevant_v2_pilot_20261113/cases.json`. Its SHA-256 is registered after the geometry gate.

## 2. Map

**Rotterdam** (`im = 1`), for the v1 reasons. One map only. Berlin, Utrecht, the sealed splits, the
`development_challenge` scope and `coupled_wire_test_*` are not opened.

## 3. Site rule: identical to v1 R1

- **R1:** 3-D BS–RIS distance 5 m ≤ d1 ≤ 20 m with a clear segment. Candidates, coverage rule, diversity rule, the 25-draw
  post-mast-draw recheck and the R2 (5–25 m) fallback ladder are exactly v1 §3, with MIN_SITES = 4.
- **Seed:** site planning uses the new seed word (Section 6), so the planned sites may differ from v1.

## 4. Users and the served-user rule

**Crowd:** the v1.1 natural crowd with the v1 anchoring rule (share **0.5** of group anchors in the RIS zone).
In v2 the **RIS zone** of a site is the set of street nodes (all `Streets.xy` vertices) that are:
- within horizontal distance **r** of the RIS;
- at 3-D distance ≤ r from the RIS at 1.5 m above ground;
- RIS-LOS **and** BS-blocked by `city.los_batch` at 1.5 m AGL.

This is geometry only. In v1 the zone was the map-wide BS-shadow probes.

**Eligibility:** a user is eligible if, at **every one of the 16 selected frames**:
- `bs_link != "los"` (BS-blocked), **and**
- `ris_los == True`, **and**
- the 3-D RIS–user distance is **≤ r**.

**Schedule:** as v1. Serve `min(rng.integers(2, 5), n_eligible)` eligible users chosen uniformly without replacement,
and only when at least 2 are eligible. There are no negative-share slots and no fallback pool. The retry rule is as v1:
25 crowd draws per site, with sites rotated from `i mod n_sites`.

**The radius r is chosen by geometry feasibility only, before tracing.**
- Ladder: **r ∈ (40, 50, 60) m**, tried in that order.
- At each r, generation runs with the declared seeds.
- The first r is accepted if it gives (a) ≥ 46 of 48 trajectories generated, and (b) ≥ 3 distinct sites used.
- Only these two counts (and wall time) decide. No channel, rate or trace output exists at that point.
- The manifest of the accepted r is the one traced. The manifests of rejected radii are kept with a `rejected_` prefix and never traced.
- If none of 40/50/60 passes: **STOP** (no wider radius under this declaration).
- The smallest feasible radius is preferred because the leverage is expected to rise as d2 falls. This is fixed now.

**Post-trace path rule:** as v1. A frame is flagged if any served user has a zero BS channel and a zero RIS channel.
Flagged frames are excluded from L2 and from the benchmark.

## 5. Size and split

- **Size:** **48 trajectories** × 16 frames × arrays 64 / 128 / 256.
- **Split by index:** train 0000–0033 (34), validation 0034–0040 (7), **test 0041–0047 (7, SEALED)**.
  Test trajectories are generated, traced and packaged. No metric, gate or design decision may read them.
- **Budget:** v1 needed 2 h 45 min from chain start to package on 3060-3 (trace 34 min; channels 4 workers).
  v2 uses the same chain.

## 6. Seeds

- New `SeedSequence` domain word **`20261113`**, disjoint from {1, 2, 3, 4, 5, 20261109, 202611092, 20261112}.
- Site planning: `[users.SEED, 20261113, 0, 1]`.
- Trajectory i: `[users.SEED, 20261113, 1, 1, i]`.
- The same seeds are used at every radius of the ladder.
- Package label pair sampling: `--seed 20261113`.

## 7. GEOMETRY GATE (before tracing; all must hold at the accepted r)

- **G1:** ≥ 46 / 48 generated.
- **G2:** 100% of served user-frames satisfy BS-blocked and RIS-LOS in the stored labels, **and** a 3-D RIS–user distance ≤ r
  at the compacted positions. An independent `los_batch` recheck agrees on ≥ 99%.
- **G3:** 100% of cases have post-draw d1 within the rule and a clear BS–RIS segment; ≥ 3 distinct sites.
- **G4:** 2–4 served users; all positions finite, in bounds, above ground.
- **G5:** trace preflight (five mesh-policy checks plus endpoint preflight) passes. This is evaluated by the trace stage.

## 8. LEVERAGE GATE (validation 0034–0040 only, after the package; all cells must hold)

The definitions are those of the v1 leverage run (`paper-b-neural-pair-search/scripts/ris_leverage_rr_20261113.py`):
- *all OFF* = every element OFF, with the stored RZF precoder of the all-OFF channel;
- *best* = b35 with long deadlines (300 / 500 / 800 ms, seed 7), never below all OFF;
- gains are in worst-user SE, bit/s/Hz.

| Gate | Quantity (validation frames that are not flagged) | 64 | 128 | 256 |
|---|---|---:|---:|---:|
| **L1a** | pathless served user-frames / validation served user-frames | ≤ 2% | ≤ 2% | ≤ 2% |
| **L1b** | flagged frames / validation frames | ≤ 5% | ≤ 5% | ≤ 5% |
| **L2** | median (best − all OFF) | ≥ 0.05 | ≥ 0.07 | ≥ 0.14 |

- **The L2 thresholds are the v1 thresholds** (the natural-scope non-zero-frame medians × √10), unchanged.
- **L3 is dropped, and why:** L3 (all OFF − no RIS) measures what an *unconfigured* RIS adds, compared with having no RIS.
  The benchmark's objective is to choose a RIS configuration under a deadline, starting from the all-OFF reference, so
  the quantity that matters is how much the setting can change the worst-user SE (L2).
  The all-OFF scattering of a passive panel is a property of the hardware, not of any search method.
  It is still reported (median and mean), but it decides nothing.
  This change is declared here, before any v2 data exist. It does not reinterpret the v1 verdict.
- **Reported, not gating:** the share of frames with best − all OFF > 0.1 and > 0.5; the mean and p90; all OFF − no RIS
  (median and mean); the RIS(best) − direct power per class; the d1 and d2 distributions.
- **PASS** = all nine cells hold. Otherwise **FAIL**, with the failing cells listed.

## 9. Declared later benchmark (only if the leverage gate PASSES; details in a separate dated benchmark declaration)

- **Methods:** greedy, pair, b3, b5, b35, M0 (analytic shortlist), ME-GA, NSGA-II, and the learned factored pair heads.
  The heads are trained on train 0000–0033 and selected on validation only.
- **Primary metrics:**
  1. worst-user SE (bit/s/Hz);
  2. optimization latency, mean and p95 (ms);
  3. computational budget: SE at hard deadlines of 5 / 10 / 50 / 100 ms, together with the number of channel/score
     evaluations each method spends.
- The test split is opened only after that benchmark declaration is frozen.

## 10. STOP RULES

1. Geometry: no feasible radius in the ladder, or G1–G4 fail at the accepted r → stop before tracing.
2. Sites: R2 still yields fewer than 4 sites → stop.
3. Run: the unchanged production rule (more than 2% of a map's cases fail → halt; with 48 cases, one failure halts).
   One retry is allowed for infrastructure failures only.
4. Budget: projected wall time > 10 h after the first 6 channel-complete trajectories → stop and re-plan. Free disk < 5 GB → stop.
5. Leverage FAIL → v2 is not promoted and no RIS-relevant search benchmark is run on it. Thresholds are never changed after the run.
6. Any read of rr2 0041–0047 for a metric → report it and discard the affected analysis.

## 11. Execution

1. Generate locally (`scripts/build_bench_ris_relevant_v2_cases.py --radius r`, ladder in order).
2. Validate (`scripts/validate_bench_ris_relevant_v2_cases.py`).
3. Register the hash.
4. Use the new 3060-3 workspace `/var/tmp/atomwise-bench-ris-relevant-v2-20261113`.
5. Run the chain `scripts/run_bench_ris_relevant_v2_pilot.sh`, which mirrors v1: trace on GPU 0 / CPUs 3,9, 4 channel workers, package.
6. Run the leverage stage on validation only and append the verdict below.

---

## Results

### Geometry gate (appended 2026-11-13 08:15 WIB, before any tracing)

**Radius ladder:** **r = 40 m was accepted at the first rung.**
- It gave 48 / 48 trajectories and 3 distinct sites.
- 50 m and 60 m were not generated, because the ladder stops at the first feasible radius.
- Generation took 65 s locally.

**Site rule:** **R1 (5–20 m) gave fewer than 4 sites** with the new site seed, so the declared fallback **R2 (5–25 m)** was used.
- R2 planned 4 sites: base d1 18.4 m (mast), 24.9 m (mast), 24.8 m (rooftop), 20.2 m (rooftop).
- Site 00 was never used, because no eligible crowd formed within its 40 m zone. The other sites took 23 / 12 / 13 cases.

**Manifest:** `outputs/runs/bench_ris_relevant_v2_pilot_20261113/cases.json`, SHA-256
`951c42c3ed8fb8acb82a1ff1c953d210b4fc4bd5116043b4d4bf050d0206715b`. It is registered in `ris_relevant_scope_v2.py`
with `RIS_RELEVANT_V2_RADIUS_M = 40.0`.

| Gate | Value | Result |
|---|---|---|
| G1 | 48 / 48 generated | PASS |
| G2 | 2,016 served user-frames; labels and radius 100%; independent recheck 100% | PASS |
| G3 | 0 violations of 5–25 m; post-draw d1 20.1 / 24.7 / 25.0 m (min / median / max); 3 sites | PASS |
| G4 | 0 violations | PASS |
| G5 | evaluated by the trace stage | pending |

**Descriptive values:**
- d2: 21.5 / 32.6 / 39.3 m (q10 / median / q90). v1 had a median of 121 m.
- Served user-frames by class:
  - `no_los_reflection_candidate`: 1,333;
  - `no_los`: 683.
- Split: 34 / 7 / 7 (test sealed).
- Generation effort: 14 of 48 trajectories needed a site substitution, and 27 a crowd redraw.

**Caveat:** because of the R2 fallback, d1 is about 5 m longer than in v1 (median 24.7 m against 18.9 m). Following the 1/d1²
scaling of the bound, this costs about 2.3 dB.

### Run and leverage gate (appended 2026-11-13 11:15 WIB)

**Run:** `/var/tmp/atomwise-bench-ris-relevant-v2-20261113` on 3060-3, job `a02dfe17-9b62-4258-8356-e1203231b18b`.
- Chain started 01:12:23Z.
- Trace: 48 / 48 passed, 0 failed.
- **G5: PASS.** `all_passed` in all 48 trace records; scene `f730314f…`, identical to natural production and v1.
- Channels: 144 / 144 units, 0 FAILED.
- PILOT_DONE at 04:00:03Z (2 h 48 min).
- Package: `output_sha256sums` verified.
  - `channel_realizations.h5` sha256 prefix `6b197eadadcbc496`
  - `ris_relevant_split.json` sha256 prefix `3c1269501d334ab9`
- Production gate, through `scripts/run_bench_ris_relevant_v2_gate.py`: **PASS**.
  - All 9 checks pass; 48 / 48 cases included.
  - LOS geometry agreement 12,096 / 12,096.

**Leverage stage:** `paper-b-neural-pair-search/scripts/rr2_leverage_stage_20261113.sh`. These are v2 copies of the v1 leverage scripts with rr2 ids. In the summary, L3 is non-gating; the L1a/L1b/L2 thresholds are verbatim.
- Validation only: 0034–0040, 112 frames per array. No test-split file was read.
- Scored 2026-11-13T04:00:58Z.
- Files: `outputs/runs/bench_ris_relevant_v2_pilot_20261113/leverage_20261113/`.

**Verdict: FAIL.** The only failing cell is **L2 at 128**: 0.0558 < 0.07.

| Gate | 64 | 128 | 256 | Threshold (64 / 128 / 256) |
|---|---:|---:|---:|---|
| L1a pathless served user-frames | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | ≤ 2% |
| L1b flagged frames | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | ≤ 5% |
| L2 median (best − all OFF), bit/s/Hz | 0.1068 ✓ | **0.0558 ✗** | 0.8580 ✓ | ≥ 0.05 / 0.07 / 0.14 |

**Reported, not gating** (304 served user-frames per array, 0 pathless, 0 flagged):

| Quantity | 64 | 128 | 256 |
|---|---:|---:|---:|
| Best − all OFF, mean / p90 | 0.298 / 0.715 | 0.373 / 1.144 | 1.229 / 2.598 |
| Frames with best − all OFF > 0.1 / > 0.5 | 70.5% / 28.6% | 41.1% / 29.5% | 100% / 88.4% |
| All OFF − no RIS, median / mean (former L3) | 0.0067 / 0.0534 | 0.0009 / 0.0121 | 0.0311 / 0.0376 |
| Per-trajectory median best − all OFF, 0034…0040 | 0.06, 0.21, 0.64, 0.79, 0.11, 0.10, 0.01 | 0.03, 0.26, 0.63, 1.37, 0.06, 0.01, 0.01 | 0.52, 0.57, 0.65, 1.59, 2.75, 0.69, 1.70 |

**Reading (descriptive, decides nothing):**
- Compared with v1, leverage rose sharply at 64 (median 0.074 → 0.107) and at 256 (0.134 → 0.858). At 256, every validation frame gains more than 0.1.
- At 128 the distribution is bimodal across trajectories:
  - 0035–0037 gain 0.26–1.37;
  - 0034, 0038, 0039 and 0040 gain ≤ 0.06.
  - So the median of 112 frames falls just below the threshold, while the mean (0.373) is high.
- With 7 validation trajectories, the 128 cell rests on very few placements.

**Under this declaration (stop rule §10.5):**
- v2 is **not** promoted, and no RIS-relevant search benchmark is run on it.
- The thresholds are not changed, and the sealed test (rr2 0041–0047) stays unopened.
- Any further variant needs a new dated declaration and a new seed word.
