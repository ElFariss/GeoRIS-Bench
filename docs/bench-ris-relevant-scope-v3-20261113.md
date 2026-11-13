# CoupledRIS-Bench RIS-relevant scope v3: declaration (2026-11-13 WIB)

**Status:** declared **before** any v3 generation, tracing or channel build. Sections 1–10 are fixed and never loosened.
Results are appended only below "Results".

## 0. Why v3: v1 and v2 failed, stated openly

| Pilot | Validation | Verdict | Failing cells |
|---|---|---|---|
| v1 (`…-v1-20261112.md`, seed 20261112) | 7 traj (rr1 0034–0040) | FAIL | L3 at 64 / 128 / 256 (unconfigured-RIS benefit); L2 at 256: 0.134 < 0.14 |
| v2 (`…-v2-20261113.md`, seed 20261113, r = 40 m) | 7 traj (rr2 0034–0040) | FAIL | L2 at 128: 0.0558 < 0.07 (64: 0.107 ✓, 256: 0.858 ✓) |

The v1 and v2 verdicts stand. Their sealed tests (rr1/rr2 0041–0047) were never opened.
The diagnostics below used only train+validation (indices 0000–0040).

### D1. Tracer consistency (H_direct must not depend on the RIS size)

**Finding 1: H_direct differs between the 64-, 128- and 256-element traces of the same case, frame and user.**
- Measured on the packages over 656 frames per version (41 trajectories × 16).
- **No** frame has identical H_direct across sizes.

| | v1 | v2 |
|---|---:|---:|
| Median relative difference ‖d₁₂₈ − d₆₄‖ / ‖d₆₄‖ | 0.57 | 0.29 |
| Median relative difference, 256 vs 64 | 0.97 | 0.70 |
| Per-user direct-power difference, 128 vs 64: median / p90 | 4.2 / 18.4 dB | 1.4 / 4.7 dB |

This explains the array-dependent no-RIS rates and link-class counts seen in v1.

**Finding 2: convergence test.**
- The criterion was written before the run (`v3/D1_CRITERION.txt`, sha256 `760019d5…d8b`).
- Setup: 6 v2 validation frames (rr2 0034–0039, 17 users); direct-only traces (receivers = BS ports + users); path cap 2e6.
- Reference: 1e6 samples, seed 7.

| Samples per source | ≤ 1 dB from reference (seed 7 / seed 8) | Seed-to-seed ≤ 1 dB | Class agreement | Converged |
|---|---:|---:|---:|---|
| 1e4 | 64.7% / 64.7% | 100% | 94.1% | no |
| 1e5 | 94.1% / 94.1% | 100% | 100% | no (below 95%) |
| 1e6 | 100% (reference) | 100% | 100% | **yes** |

- At 1e4, 3 of 17 users had no direct path at all; at 1e5 and 1e6, 2 of 17.

**Finding 3: the path cap causes the array dependence.**
- Production trace at 1e4 samples and cap `max_num_paths_per_src = 40000`, compared with the direct-only trace at the same samples:
  - 64-element receiver set: H_direct agrees within 1 dB for **100%** of users;
  - 256-element receiver set: **35%**;
  - 256-element receiver set with cap 2e6: **94%**.
- Even at 1e6 samples and cap 2e6, one of 7 users in rr2-0034 differs by 5.4 dB between the 256-receiver trace and the
  direct-only trace.
- **So H_direct is traced once per frame, with external receivers only, and shared by all three array sizes.**
- Solve time at 1e6 samples is 0.1–1.2 s (direct-only) and about 1.5 s (256 receivers). Cost is not a constraint.

### D2. Why some v2 trajectories have no headroom at 128

**Data:** b35 with long deadlines (300 / 500 / 800 ms) on all 41 v2 train+validation trajectories × 16 frames × 3 sizes.
- Pooled over these 41 trajectories, the median best − all OFF is **0.149 / 0.248 / 0.659** at 64 / 128 / 256, all above the L2 thresholds.
- The 7-trajectory validation median of 0.056 at 128 was dominated by placement.

**Result by site:**

| v2 site | Trajectories | Median gain 64 / 128 / 256 | Served user-frames behind the panel plane | Median max angle user vs panel normal |
|---|---:|---|---:|---:|
| 01 | 20 | 0.202 / 0.461 / 0.959 | 0% | 82.6° |
| 02 | 10 | **0.070 / 0.058** / 0.499 | **99%** | 108.2° |
| 03 | 11 | 0.221 / 0.211 / 0.575 | 0% | 51.9° |

**The separating geometry-only quantity is the angle between the RIS→user direction and the panel normal.**
- Trajectory level: trajectories whose served users all stay within about 82° of the normal have median gains
  0.398 / 0.451 / 0.716 (17 trajectories). That is at least 5.1× each L2 threshold.
- The share of user-frames behind the panel plane has Spearman ρ = −0.37 / −0.44 / −0.11 with the per-trajectory gain.
- The panel normal points horizontally at the BS (unchanged since v1.1). A user behind that plane sees the panel edge-on or from behind.
- d2 and d1·d2/(d1+d2) correlate *positively* (ρ 0.2–0.5) and are confounded with site. They are not used.
- Files: `outputs/runs/bench_ris_relevant_v3_20261113/diagnostics/`.

## 1. Scope identity

- **Scope name:** `ris_relevant_v3`, in the new modules `ris_relevant_v3.py` and `ris_relevant_scope_v3.py`.
  These wrap v2, which wraps v1, which wraps the frozen v1.1 generator.
- **Unchanged:** `PRODUCTION_SCOPES`, the v1/v2 registries, and every frozen or hash-locked file.
- **Ids:** `rotterdam-rr3-0000` … `rotterdam-rr3-0119`.
- **Manifest:** `outputs/runs/bench_ris_relevant_v3_20261113/cases.json`. Its hash is registered after the geometry gate.

## 2. Map, sites, users (v2 rules plus one new rule)

- **Map:** Rotterdam (`im = 1`), as before.
- **Sites:** exactly v2/v1 R1 (5–20 m, clear BS–RIS segment, v1.1 coverage and diversity), with the R2 fallback (5–25 m) if fewer than 4 sites.
- **Crowd:** v2 (v1.1 natural crowd; 0.5 of anchors in the RIS zone within r, RIS-LOS and BS-blocked).
- **Eligible user, at every one of the 16 selected frames, all of:**
  1. BS-blocked;
  2. RIS-LOS;
  3. 3-D RIS–user distance ≤ r;
  4. **new: angle between (user − RIS) and the panel normal ≤ 80°** (front half-space with a 10° margin from grazing; from D2).
  The RIS-zone anchor nodes use the same four conditions at 1.5 m AGL.
- **Schedule and retries:** as v2 (2–4 served users, eligible pool only, 25 crowd draws per site, rotating sites).
- **Radius ladder:** r ∈ (40, 50, 60) m. Accept the first r with **≥ 114 / 120** generated and ≥ 3 distinct sites;
  otherwise STOP. Decided on generation counts only.

## 3. Tracer (from D1)

- Array traces (64 / 128 / 256 receivers on the panel): `samples_per_src = 1e6`, `max_num_paths_per_src = 2e6`,
  seed 7, max_depth 2, LOS + specular + diffraction (otherwise unchanged).
- **Shared direct trace:** one trace per frame with **external receivers only** (BS ports + users), at 1e6 samples and cap 2e6.
  - Its BS↔user and user↔user paths **replace** those rows in all three array traces when channels are built
    (new wrapper `run_bench_ris_relevant_v3_channels.py`).
  - So H_direct and the BS link classes are identical across array sizes by construction.
  - Integrity check **G6**, after packaging: per frame, H_direct is bit-identical across the three sizes (train+validation+test,
    integrity only; no rate is computed on test).
- Each trace shard is a subset of the one registered manifest, by trajectory index. Shard outputs are merged into one trace
  directory. Every record keeps the full-manifest hash.

## 4. Size, split, seeds

- **Size:** **120 trajectories** × 16 frames × 64 / 128 / 256.
- **Split by index:** train 0000–0083 (84), validation 0084–0101 (18), **test 0102–0119 (18, SEALED)**.
- **Seeds:** new domain word **`202611133`**, disjoint from {1, 2, 3, 4, 5, 20261109, 202611092, 20261112, 20261113}.
  - Sites: `[SEED, 202611133, 0, 1]`.
  - Trajectory i: `[SEED, 202611133, 1, 1, i]`.
  - Package labels: `--seed 202611133`.

## 5. GEOMETRY GATE (before tracing, at the accepted r)

- **G1:** ≥ 114 / 120 generated.
- **G2:** 100% of served user-frames satisfy all four eligibility conditions in the stored labels and compacted positions;
  independent `los_batch` recheck ≥ 99%.
- **G3:** d1 rule and clear segment for 100% of cases; ≥ 3 distinct sites.
- **G4:** 2–4 served users; finite, in bounds, above ground.
- **G5:** trace mesh and endpoint preflight passes (trace stage).
- **G6:** shared-H_direct identity (Section 3), after packaging.

## 6. LEVERAGE GATE (validation 0084–0101 only: 18 trajectories × 16 frames = 288 frames per size), scored PER ARRAY SIZE

The definitions are those of v1/v2: best = b35 at 300 / 500 / 800 ms (seed 7), never below all OFF; worst-user SE in bit/s/Hz.

| Cell (per size n) | Quantity | 64 | 128 | 256 |
|---|---|---:|---:|---:|
| L1a | pathless served user-frames / served user-frames | ≤ 2% | ≤ 2% | ≤ 2% |
| L1b | flagged frames / frames | ≤ 5% | ≤ 5% | ≤ 5% |
| L2 | median (best − all OFF) on unflagged frames | ≥ 0.05 | ≥ 0.07 | ≥ 0.14 |

- **The thresholds are unchanged from v1/v2.** There is no L3; the reason is given in v2 §8 (it measures what an
  unconfigured RIS adds, which is not the search objective).
- **Per-size verdict, declared up front:**
  - an array size whose L1a, L1b and L2 cells all pass is **promoted and benchmarked**;
  - a size that fails is **excluded** from the benchmark and reported as failed, with its values.
- Reported but not gating: the former L3 (median and mean), shares of gains > 0.1 and > 0.5, per-site medians, and the d1/d2/angle distributions.
- **Why v3 differs from v1/v2:**
  1. a validation set 2.6× larger (18 trajectories), so the median no longer rests on 7 placements;
  2. the D1 tracer fix (converged samples, a shared direct trace);
  3. the D2 front-half-space rule.

## 7. Declared benchmark (promoted sizes only; details in a separate dated benchmark declaration, frozen before test)

- **Methods:** greedy, pair, b3, b5, b35, M0, ME-GA, NSGA-II, learned factored pair heads (trained on train, selected on validation).
- **Primary metrics:**
  1. worst-user spectral efficiency;
  2. optimization latency (mean / p95 ms);
  3. computational budget: SE at 5 / 10 / 50 / 100 ms deadlines, with evaluation counts.

## 8. Compute (declared)

- **Workspace:** `/mnt/home/faris/atomwise-bench-ris-relevant-v3-20261113` on labkc-comm3, plus the same layout under `~` on labkc-comm2.
- **Trace:** 4 GPUs (2 per node), 4 shards by index mod 4. comm3 takes shards 0 and 1; comm2 takes shards 2 and 3.
- **Channels:** each node builds the channels of its own shards. comm2 results (channel units and trace records) are copied
  back to comm3 through the controller as they finish, so the run is resumable.
- **CPUs:** 5 and 11 are left free on both nodes.
- **Not used:** labkc-l40 (no inter-node transfer path and no PyNEC environment within the 20-minute limit) and labkc-comm1.
- **Basis cache:** not used. The NEC basis is a live PyNEC factorization and cannot be serialized, so each worker builds its own.

## 9. STOP RULES

1. No feasible r, or G1–G4 fail → no tracing.
2. Fewer than 4 sites under R2 → stop.
3. More than 2% of cases fail (the production rule; with 120 cases, 3 failures halt). One retry for infrastructure failures only.
4. Free disk < 5 GB → stop.
5. A failing size is excluded (Section 6). Thresholds never change.
6. Any metric read of rr3 0102–0119 → report it and discard the affected analysis.

## 10. Execution

1. Generate and run the geometry gate locally.
2. Register the hash.
3. Stage to both nodes, then run trace shards, channels and the copy-back.
4. Merge, assemble, package (comm3) and write `PILOT_DONE`.
5. Score the leverage gate (validation only) and write `LEVERAGE_GATE.json`.
6. Append the results below.

---

## Results

### Event 1, 13:26 WIB: stop rule 9.2 fired at site planning (no trajectory generated, nothing traced)

**What fired:** with the v3 site stream `[SEED, 202611133, 0, 1]`, R1 gave fewer than 4 sites, and R2 gave **1** site.
Under §9.2 the v3 build, as declared above, **stops**. The log is in
`outputs/runs/bench_ris_relevant_v3_20261113/stopped_siteplan_r40/generation.log`.

**Cause:** the v1.1 site planner is a single random draw. It thins facade and roof points at random and evaluates at most
1,000 BS–RIS pairs. With the v1 and v2 streams it found exactly 4 sites; with this stream it found 1.
This is a property of one random planning draw, not of the map. It also means v1/v2 rested on only 3–4 placements.

## Amendment A1 (13:30 WIB, before any v3 trajectory is generated; gates and thresholds untouched)

**A1.1 Site pool.**
- The site pool is the **union of 8 independent planning draws**, site streams `[SEED, 202611133, 0, 1, k]` for k = 0…7.
  Each draw is the unchanged v1 planner at R1.
- Deduplication, in stream order: a site is kept only if its RIS is ≥ 25 m from every kept RIS and its BS anchor has
  < 2 kept sites (the v1.1 diversity rules).
- Required: **≥ 6 sites** (was 4). If R1 gives fewer than 6, the same union is run at R2.
  If R2 gives fewer than 6: STOP, with no further amendment.

**A1.2** Everything else in §§1–9 is unchanged: user rules, front-half-space rule, radius ladder, tracer, sizes, split,
seeds for trajectories, the G and L gates and their thresholds, and the stop rules.

**Rationale:**
- More placements make the validation median less dependent on 3–4 sites. That is the v2 lesson from D2.
- The change uses geometry only; no rate or channel output exists.

### Geometry gate under A1 (appended 13:40 WIB, before any tracing)

**Sites:**
- **R1 holds:** the union of 8 draws gave **14 sites** (2, 1, 3, 3, 5, 3, 2, 2 per draw after their own diversity rule).
- Only **4 sites** formed eligible crowds; the others have empty front-half-space zones within 40 m.
  Cases per site: 79 / 30 / 8 / 3. Validation uses 3 sites (12 / 5 / 1 trajectories).
- So placement diversity is still limited (caveat).

**Radius ladder:** **r = 40 m accepted at the first rung:** 120 / 120 generated (≥ 114), 4 sites (≥ 3).

**Manifest:** `outputs/runs/bench_ris_relevant_v3_20261113/cases.json`, SHA-256
`58efc89e39d7a335bfae970cdfc3811552fd7b2a4a37fbb04494dd497cd6893e`, registered in `ris_relevant_scope_v3.py`
(`RIS_RELEVANT_V3_RADIUS_M = 40.0`).

| Gate | Value | Result |
|---|---|---|
| G1 | 120 / 120 | PASS |
| G2 | 4,800 served user-frames; labels, radius and front-half-space 100%; LOS recheck 100% | PASS |
| G3 | 0 violations of 5–20 m; d1 7.8 / 13.7 / 18.7 m (min / median / max); 4 sites | PASS |
| G4 | 0 violations | PASS |

**Descriptive values:**
- d2: 26.7 / 32.3 / 36.4 m (q10 / median / q90).
- Served classes: `no_los` 1,082, `no_los_reflection_candidate` 3,718.
- Generation effort: 99 of 120 trajectories needed a site substitution.

**Smoke test on 3060-3** (case rr3-0000, train split; integrity only):
- Trace with the v3 wrapper: 68 s for 16 frames × 3 sizes, plus the shared direct trace.
- After the v3 merge, the external-receiver path rows are **identical** for 64 / 128 / 256 in all 3 frames checked.
  Before the merge, the raw array traces had 355 / 362 / 346 such rows in frame 0.
- One 64-element channel unit was built: basis 211 s, unit OK.

### Event 2, 07:15Z: infrastructure failure in the new v3 trace wrapper (fixed; one retry under §9.3)

**What happened:** each of the 4 trace shards failed its second case at `scene_build` (rr3-0004, -0005, -0006, -0007)
and stopped, per the stop rule.

**Cause:** a code bug in the new wrapper.
- `trace_direct` added a count-0 entry to the module-level `ARRAY_SHAPES` for `extract_npz`, and left it there.
- The scene builder iterates `ARRAY_SHAPES`, so it then built a 0-element panel and failed.

**Fix:** the entry is now added only around the `extract_npz` call (`try`/`finally`).
- The fixed wrapper's sha256 is `e9436bc4…5337`.
- The merge script now ignores `--retry-failed` archive files and supersedes a stale merged FAILED record once the case passes
  (sha256 `44c1cbfb…60ac`).
- `STAGED_SHA256SUMS` was updated on both nodes.

**Retry:**
- The shards were restarted with `--retry-failed` at about 07:16Z. This is the single infrastructure retry §9.3 allows.
- Records of cases that had already passed (rr3-0000 … -0003, traced before the faulty entry existed) are reused through the unchanged resume check.
- Channel workers were not restarted and keep their NEC bases.
- Originals are in `archive_bug1/` on each node.

### Run, package and leverage gate (appended 2026-11-13 18:20 WIB)

**Run:**
- **Nodes:** labkc-comm3 (shards 0, 1), with workspace `/mnt/home/faris/atomwise-bench-ris-relevant-v3-20261113`, and labkc-comm2
  (shards 2, 3), with workspace `/home/faris/atomwise-bench-ris-relevant-v3-20261113`. Both ran 4 GPUs' worth of trace shards
  and 5 channel workers each; CPUs 5 and 11 were unused.
- **Trace:** 120 / 120 passed, 0 failed (after the Event 2 retry). Done 08:12Z (comm3) and about 08:20Z (comm2).
  Each node merged its two shards; the final merge of all 4 shards was at 11:12Z.
- **Channels:** 357 / 360 units.
  - **rr3-0069 (train) failed at all 3 sizes** (`RuntimeError: Unknown exception` in the channel stage; genuine, not retried).
  - 1 of 120 = 0.83% ≤ 2%, so the stop rule was not exceeded, and the case is excluded (`failed_cases` in `channel_run.json`).
  - Workers finished at 10:01Z (comm3) and 10:22Z (comm2).
- **Copy-back:** comm2 outputs were shipped in 2 batches (1.54 GB at 08:14Z; the rest at 11:10Z).
- **Package:** assemble at 11:12:55Z, package at 11:13:49Z, then **PILOT_DONE**.
  Split file: `ris_relevant_v3_split.json`.
- **Production gate** (integrity, all cases): **PASS**.
  - 119 / 120 included; stop rule not exceeded.
  - LOS geometry agreement 0.9928; mesh preflight passed.

**G6: FAIL.**
- The shared-direct merge makes the *external path rows* identical for 64 / 128 / 256 (smoke test, 3 frames).
- But the channel builder's `d` term (`network_from_group(...).d`, which the leverage scripts use as "no RIS") still differs
  across sizes in all 1,616 train+validation frames checked. Relative difference: median 0.72, max 1.77; per-user power 3.5 / 6.3 dB (median / p90).
- So `d` is **not** the BS→user path channel alone. It contains a term that depends on the array, and D1's package-level
  comparison was confounded by it.
- The tracer findings of D1 still stand: sampling at 1e4 is not converged, and the 40,000 path cap truncates paths for
  256 receivers. Both were fixed in the traces.
- The source of the array-dependent `d` term is **not yet investigated**.
- **Consequence:** the "no RIS" rate and the former L3 are not comparable across sizes. The L-gate below does not use them:
  L2 compares the best setting with all OFF within one array size.

**Leverage gate** (validation rr3 0084–0101 only: 18 trajectories, 288 frames per size, 800 served user-frames per size;
scored 2026-11-13T11:15:21Z; no test file read):

| Cell | 64 | 128 | 256 | Threshold |
|---|---:|---:|---:|---|
| L1a pathless served user-frames | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | ≤ 2% |
| L1b flagged frames | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | ≤ 5% |
| L2 median (best − all OFF), bit/s/Hz | **0.3018 ✓** | **0.4183 ✓** | **0.7820 ✓** | ≥ 0.05 / 0.07 / 0.14 |
| **Per-size verdict** | **PASS, promoted** | **PASS, promoted** | **PASS, promoted** | |

**Reported, not gating:**

| Quantity | 64 | 128 | 256 |
|---|---:|---:|---:|
| Best − all OFF, mean / p90 | 0.450 / 1.089 | 0.534 / 1.379 | 1.097 / 2.760 |
| Frames > 0.1 / > 0.5 | 73.6% / 29.2% | 79.9% / 38.9% | 90.3% / 62.5% |
| All OFF − "no RIS" median (see G6) | −0.0004 | 0.0028 | 0.0109 |

**Verdict:** all three sizes pass L1a, L1b and L2 and are promoted for the declared benchmark (§7).

**Open caveats:**
1. G6 failed: the `d` term depends on array size (to be investigated before any no-RIS claim).
2. Only 4 sites were used, and validation comes from 3 of them (12 / 5 / 1 trajectories).
3. One train case (0069) is excluded.

The sealed test (rr3 0102–0119) stays unopened until a benchmark declaration is frozen.

**Files:**
- `outputs/runs/bench_ris_relevant_v3_20261113/leverage_20261113/`: `LEVERAGE_GATE.json`, `summary.json`,
  `production_gate.json`, `channel_run.json`, split, logs.
- Diagnostics: `…/diagnostics/`.
- Remote: `$V/LEVERAGE_GATE.json`, `$V/PILOT_DONE`, `$V/ris_relevant_v3_split.json`.
