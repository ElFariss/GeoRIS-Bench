# CoupledRIS-Bench RIS-relevant scope v1: pilot declaration (2026-11-12 WIB)

**Status:** declared before any geometry generation, tracing or channel build for this scope.
Sections 1–8 are the pre-registered rules and pass/fail criteria. They are never loosened after this point.
Results are appended only in the "Results" sections at the end, with the time they were added.

**Why:** the natural-scope leverage diagnostic
(`paper-b-neural-pair-search/docs/bench-ris-leverage-diagnostic-20261112.md`, validation only) found that
49% of frames have worst-user rate 0 (a served user without any traced path), and that the facade RIS
(median 112 m from the BS) arrives 40–60 dB below the BS paths. The median best − all-OFF gain on non-zero
frames is 0.015 / 0.022 / 0.045 bit/s/Hz at 64 / 128 / 256 elements. The user chose option 1 of that
diagnostic: a new scope in which the RIS is close to the BS and every served user depends on it geometrically.
The natural scope is unchanged and stays as Paper A's "a realistic facade RIS barely matters" result.

**What is not touched:** the frozen v1.1 generator and every file in
`configs/bench/users-v1.1-freeze-20261109.json` (lock `e152e78c…0ecd`, verified before every stage);
`hybrid/operator.py`; `scripts/package_bench_production.py`; the phase-0 audit scripts;
`bench/CoupledRIS-Bench/physics/` (Z_RIS is reused read-only); `matched_v1/v2.json`; `outputs/frozen`;
the hash-locked docs; the Paper B declaration. The natural and challenge production workspaces on 3060-3
are not written. Berlin, Utrecht, the natural sealed test split (trajectory index ≥ 102), the
`development_challenge` scope and `coupled_wire_test_*` are not opened.

---

## 1. Scope identity

- **Scope name:** `ris_relevant_pilot` (registered in the new module `src/atomwise_ris/bench/ris_relevant_scope.py`,
  next to, not inside, `PRODUCTION_SCOPES`; the existing tuple `("development_full", "development_challenge")`
  and every behaviour of those scopes stay byte-for-byte identical).
- **Case ids:** `rotterdam-rr1-0000` … `rotterdam-rr1-0047`.
- **Manifest:** `outputs/runs/bench_ris_relevant_pilot_20261112/cases.json`. Its SHA-256 is registered in
  `ris_relevant_scope.py` after geometry generation and before tracing (the same procedure as the
  natural and challenge manifests).

## 2. Map choice

**Map: Rotterdam** (development map index `im = 1`). Reasons, all fixed before this declaration:
1. In the natural-scope diagnostic Rotterdam had the fewest zero-rate frames (28%, against 56% Munich and 64% Maastricht).
2. Its v1.1 site pool has 16 planned sites (Munich has 6), so it has the most facade/roof candidates for a short-range site rule.
3. Its scene already passed all five mesh-policy checks in natural production, so a pilot failure cannot be blamed on the mesh.

One map only. Munich and Maastricht are not generated in this pilot.

## 3. Site rule (geometry only, never rates)

The v1.1 `Generator.plan_sites` procedure is reused unchanged except for the BS–RIS distance range.
It is implemented in a subclass in a new file (`RISRelevantGenerator`); `users.py` is not edited.

- **Candidates:** exactly as v1.1. RIS on a facade point at 3–30 m above local ground (10 m thinning);
  BS on a rooftop point (≥ 6 m AGL, antenna ≥ 1.5 m above the roof, 15 m thinning) or a 12 m street mast (80 street samples).
- **Distance rule R1:** 3-D BS–RIS distance **5 m ≤ d1 ≤ 20 m**, with a clear BS–RIS segment (`city.los_batch`).
- **Coverage rule:** as v1.1: the BS sees ≥ 10% of street probes, the RIS sees ≥ 10%, and the RIS sees
  ≥ max(25, 5% of probes) probes that the BS cannot see (probes at 1.5 m AGL, up to 1200).
- **Diversity:** as v1.1: ≤ 16 sites, ≤ 2 sites per BS anchor, RIS points ≥ 25 m apart, plus the v1.1
  repair pass (up to 3000 extra pairs) when fewer than 6 sites are found.
- **Mast/antenna draw:** as v1.1 `choose_site` (lognormal offset, median 4 m, cap 24 m, roof offset reduced by 1.5 m).
  The 3-D distance **after** the draw must also satisfy 5–20 m and the BS–RIS segment must stay clear (25 draws).
- **Fallback ladder (declared now):** if R1 yields fewer than 4 sites, use **R2: 5 m ≤ d1 ≤ 25 m**.
  If R2 also yields fewer than 4 sites, the pilot STOPS (no further widening).
- **Stored per site:** the RIS-zone probes (street probes that are RIS-LOS and BS-blocked) are stored with
  the site and used only for crowd anchoring (Section 4).
- **RIS orientation:** unchanged (`compact()`: horizontal unit vector from RIS towards BS).

## 4. Users and the served-user rule

**Crowd:** the v1.1 natural crowd model (`Generator.natural`: group counts, Zipf sizes, cruise speeds,
routes, height classes, slope cap) is unchanged, with one declared change to anchoring.
Each group anchor is replaced, with probability **0.5**, by a RIS-zone probe of the active site drawn uniformly,
with the v1.1 jitter N(0, 7 m) and the v1.1 snap to a street node within 25 m. Otherwise the v1.1 anchor is used.
Without this, a natural crowd almost never puts two users in the BS shadow of a RIS that is ≤ 20 m from the BS.

**Selected frames:** the 16 frames `unique(rint(linspace(0, F-1, 16)))` of the clip, as in production.

**Eligibility (geometry labels only):** a user is eligible to be served if, at **every one of the 16 selected frames**:
- `bs_link != "los"` (BS-blocked: label `no_los` or `no_los_reflection_candidate`), **and**
- `ris_los == True`.

**Schedule:** draw `want = rng.integers(2, 5)` (2–4, as v1.1). If at least 2 users are eligible, serve
`min(want, n_eligible)` eligible users chosen uniformly without replacement. There are **no negative-share slots**
(v1.1 `SERVED_NEGATIVE_SHARE = 0.15` is replaced by 0) and no fallback to non-eligible users.

**Retry rule:** with fewer than 2 eligible users, the crowd is redrawn (same trajectory rng) up to 25 times per site.
Sites are tried in the order `(i + k) mod n_sites`, k = 0, 1, …, so trajectory `i` starts at site `i mod n_sites`
(the production round-robin). If every site is exhausted, the trajectory is recorded as a generation failure.
Rules are never relaxed per trajectory.

**After tracing, the traced-path rule:** a served user-frame is **pathless** if its BS→user channel and its RIS→user
channel are both exactly zero (no traced path at all; the definition used in the diagnostic). Every frame with a
pathless served user is **flagged** (`rr_pathless_frame = true`) in a sidecar. Flagged frames are excluded from the
leverage medians (L2, L3) and from any search benchmark on this scope. Their share is bounded by gate L1.
No user is re-scheduled after tracing.

## 5. Size, seeds and split

**Size: 48 trajectories × 16 frames × 3 arrays (64/128/256) = 768 frames, 2,304 array-frames.**

Budget (stage costs from `docs/bench-production-full-20261109.md` §5.1, 360-trajectory production):
- channel conversion 164,301 s / 360 = 456 s wall per trajectory with 3 two-CPU workers;
- ray tracing 12,100 s / 360 ≈ 34 s per trajectory (GPU 0), which runs concurrently with the channels;
- one-time NEC basis setup about 3,750 s.

This pilot runs **4** channel workers (CPU pairs 0,6 / 1,7 / 2,8 / 4,10; the trace on 3,9; CPUs 5 and 11 stay reserved).
The workers claim (case, array) units by `flock`, so the per-trajectory wall time is about 456 × 3/4 ≈ 342 s.
Projection: 48 × 342 s + 3,750 s ≈ 20,200 s ≈ **5.6 h**, plus packaging (about 0.1 h). This is inside the 6 h target.

**Seeds:** a new `SeedSequence` domain word **`20261112`**. It is disjoint from {1, 2, 3, 4, 5, 20261109, 202611092}.
- Site planning: `SeedSequence([users.SEED, 20261112, 0, im])` with `im = 1`.
- Trajectory `i`: `SeedSequence([users.SEED, 20261112, 1, im, i])`, i = 0 … 47.

With `users.SEED = 202611081501`, the second word differs from every earlier stream, so no stream is reused.

**Split, by trajectory index (all 16 frames of a trajectory in one part):**

| Part | Indices | Trajectories |
|---|---|---:|
| train | 0000–0033 | 34 |
| validation | 0034–0040 | 7 |
| **test (SEALED)** | **0041–0047** | **7** |

The test part is sealed. Its trajectories are generated, traced and packaged, because that cannot be avoided.
No metric, plot, gate value or design decision may read them. The gates in Sections 6–7 use only the
geometry labels of all trajectories (G-gates) or the validation part (L-gates).
Opening the test part needs a separate dated declaration.

## 6. GEOMETRY GATE (evaluated before tracing, all must hold)

- **G1 (generation):** ≥ 46 of 48 trajectories are generated within the retry rule. Failed indices stay empty
  and the split is not renumbered.
- **G2 (served-user rule):** 100% of served user × selected-frame entries satisfy `bs_link != los` and `ris_los`
  in the stored labels. An independent recheck with `city.los_batch` at the compacted positions
  (BS→user blocked and RIS→user clear) agrees on ≥ 99% of the entries.
- **G3 (site rule):** 100% of cases have a post-draw 3-D BS–RIS distance within the declared range (R1, or R2 if the
  ladder fell back) and a clear BS–RIS segment on recheck. At least **3** distinct sites are used.
- **G4 (users):** every case serves 2–4 users. Every served position is finite and inside the map bounds,
  with ground z finite and user z > ground z.
- **G5 (existing mesh and terrain checks):** the trace preflight for Rotterdam passes all five fail-closed
  mesh-policy checks of `docs/bench-production-full-20261109.md` §6.1.5 (terrain bounds, ground reference
  ≤ 2.5 m, outward normals, dropped area ≤ 0.5%, emitted area ≤ 0.1%), and the endpoint preflight passes for every case.
  G5 is evaluated by the trace stage itself, which fails closed. A G5 failure halts the pilot.

G1–G4 are recorded below before tracing starts. If any of G1–G4 fails, nothing is traced.

## 7. LEVERAGE GATE (validation part only, after channels; all must hold at each array size)

The definitions are those of `paper-b-neural-pair-search/scripts/ris_leverage_20261112.py`:
- *no RIS* = `H_direct` only;
- *all OFF* = every element OFF, with the stored RZF precoder of the all-OFF channel;
- *best* = b35 with long deadlines (300 / 500 / 800 ms), never below all OFF.

Gains are in worst-user rate, bit/s/Hz.

**Derivation of the thresholds from the ideal-plate bound.**
- The bound is `A²d²/(λ²d1²d2²)`. For a BS-blocked user, d ≈ d2, so it scales as 1/d1².
- Moving d1 from the natural median of 112 m to ≤ 20 m raises the RIS-to-direct power ratio by
  ≥ 20·log10(112/20) = **15 dB** (19 dB at d1 = 12.5 m).
- The RIS path is still ≥ 22 dB below the direct path even after that shift (natural blocked/RIS-LOS ratios
  of −55.6 / −52.8 / −40.9 dB plus 15 dB). In this small-ratio regime the rate change the setting can cause
  is linear in the amplitude ratio a = 10^(r/20).
- We require a **conservative 10 dB** of the ≥ 15 dB shift, i.e. a factor √10 = 3.16 in amplitude and in gain,
  applied to the natural-scope values.

| Gate | Quantity (validation frames that are not flagged) | 64 | 128 | 256 |
|---|---|---:|---:|---:|
| **L1a** | pathless served user-frames / all validation served user-frames | ≤ 2% | ≤ 2% | ≤ 2% |
| **L1b** | flagged frames / validation frames (natural: 49%) | ≤ 5% | ≤ 5% | ≤ 5% |
| **L2** | median (best − all OFF); natural non-zero-frame median 0.015 / 0.022 / 0.045, × 3.16 | ≥ 0.05 | ≥ 0.07 | ≥ 0.14 |
| **L3** | median (all OFF − no RIS); natural mean 0.019 / 0.025 / 0.043, × 3.16, tested on the median | ≥ 0.06 | ≥ 0.08 | ≥ 0.14 |

- **Reported, not gating:** the share of frames with best − all OFF > 0.1; the median RIS-to-direct power ratio
  per user class; the d1 and d2 distributions.
- **Outcome:** **PASS** = L1a, L1b, L2 and L3 hold at all three array sizes. Anything else is a **FAIL**,
  with the failing cells listed.

## 8. STOP RULES

1. **Geometry:** if G1–G4 fail, stop before tracing. Report; do not change the rules or the seed under this declaration.
2. **Site ladder:** if R2 still yields fewer than 4 sites, stop.
3. **Run (unchanged production rule):** a map halts when more than 2% of its cases fail
   (`production_run.stop_rule_exceeded`; with 48 cases, one failed case halts the run). An infrastructure failure
   (GPU free-memory guard, host OOM, killed process) may be retried once on the same case, as in the 2026-11-11
   case-0073 precedent. A genuine trace or channel failure is not retried.
4. **Budget:** if, after the first 6 channel-complete trajectories, the projected wall time exceeds 10 h, stop and re-plan.
5. **Disk:** stop if free space on the workspace volume drops below 5 GB.
6. **Leverage:** if the leverage gate FAILS, the scope is **not** promoted to production and the Paper B search
   benchmark is not run on it as a "RIS-relevant" benchmark. The result is reported as is. A new rule set needs a
   new dated declaration and a new seed domain word. Thresholds are never changed after the run.
7. **Sealed test:** any accidental read of indices 0041–0047 is reported, and the affected analysis is discarded.

## 9. Execution plan

1. Generate the geometry locally (AMD-kc) with `scripts/build_bench_ris_relevant_cases.py`, then validate it with
   `scripts/validate_bench_ris_relevant_cases.py` (G1–G4) and record the result below.
2. Register the manifest hash.
3. Sync to a new workspace on 3060-3, `/var/tmp/atomwise-bench-ris-relevant-20261112`, which never overwrites the production workspaces.
4. Run the trace (GPU 0, CPUs 3,9) and the 4-worker channel build with resume, then packaging, as one background chain.
5. Evaluate the leverage gate on validation only, as a separate step.

---

## Results: geometry gate (appended 2026-11-12 22:10 WIB, before any tracing)

**Run:**
- Generated locally on AMD-kc with `scripts/build_bench_ris_relevant_cases.py` in 27 s wall.
- Validated with `scripts/validate_bench_ris_relevant_cases.py`.
- Files: `outputs/runs/bench_ris_relevant_pilot_20261112/{cases.json, generation_report.json, generation.log, geometry_gate.json}`.

**Manifest:** SHA-256 `82dcc824feb37071367e07d6388d6341b028dd69e2089dbb1bd25fab13c1e46b`, registered as
`RIS_RELEVANT_PILOT_HASH` in `src/atomwise_ris/bench/ris_relevant_scope.py`.

**Sites:**
- Rule **R1** (5–20 m) holds, so the fallback ladder was not used.
- Site planning found exactly **4** sites (the minimum the ladder allows):
  - 937 RIS facade candidates;
  - 3,178 BS–RIS pairs in range, of which 697 have a clear segment;
  - 5 pairs passed coverage;
  - the repair pass ran because fewer than 6 sites were found.
- Base distances: 19.1 m (mast), 18.8 m (rooftop), 19.9 m (rooftop), 12.8 m (mast).

| Gate | Value | Threshold | Result |
|---|---|---|---|
| G1 generation | 48 / 48 generated, 0 failed | ≥ 46 | PASS |
| G2 served rule | 2,192 served user-frames: labels 100%, independent LOS recheck 100% | 100% / ≥ 99% | PASS |
| G3 site rule | 0 violations; post-draw d1 13.2 / 19.0 / 20.0 m (min / median / max); 4 sites used (12 / 12 / 11 / 13 cases) | 0, ≥ 3 sites | PASS |
| G4 users | 0 violations (2–4 served, finite, in bounds, above ground) | 0 | PASS |
| G5 mesh / endpoint preflight | evaluated by the trace stage (fail-closed) | 5 checks | pending |

**Geometry gate G1–G4: PASS.**

**Descriptive values (not gating):**
- Split: 34 train / 7 validation / 7 test (sealed).
- Served user-frames by class:
  - `no_los` / RIS-LOS: 1,922;
  - `no_los_reflection_candidate` / RIS-LOS: 270.
- RIS–user distance d2: 74 / 121 / 290 m (q10 / median / q90). This is large, so most of the leverage must come from the
  short d1 (the bound scales as 1/d1² for BS-blocked users).
- Generation effort:
  - 1 of 48 trajectories needed a site substitution;
  - 8 needed a crowd redraw;
  - every other trajectory succeeded on its first crowd.

**Caveats for the run:**
- One site (base 19.9 m) sits at the edge of the range.
- With only 4 sites, the pilot measures leverage for 4 placements. It says nothing about placement diversity.
- Packaging labels use `--seed 20261112` (the label pair-sampling seed of the frozen packager).

## Launch record (appended 2026-11-12 23:00 WIB)

**Where and what:**
- **Workspace:** `/var/tmp/atomwise-bench-ris-relevant-20261112` on 3060-3, a new directory.
  - It holds a copy of the production workspace code, which is byte-identical to GeoRIS-Bench for every pipeline file checked, plus the new files.
  - The physics directory is `diff -r` identical to production.
  - `STAGED_SHA256SUMS` covers 227 code files and is checked at every chain start.
- **Chain:** `scripts/run_bench_ris_relevant_pilot.sh`, started 15:14:31Z as job `9b7c67ca-18fe-48f8-b73d-24ceca6a7033`.
  - Trace: GPU 0, CPUs 3,9.
  - Channels: 4 workers on CPUs 0,6 / 1,7 / 2,8 / 4,10.
  - Then assemble, then package with `--seed 20261112`.

**G5 (trace preflight): PASS.**
- Rotterdam scene SHA-256 `f730314f…c8b0`, identical to the natural-production scene.
- Mesh preflight `all_passed: true`: outward normals 100%, dropped area 0.0, emitted-area relative difference 3.2e-8, building base difference 0.0 m.
- Endpoint preflight passes for every case.

**Trace:** complete at 15:48:38Z, 48 passed, 0 failed.

**Channels at 16:00Z:** 0 FAILED units, with done units per array as below.

| Array | Done | Basis setup | Per-unit time |
|---:|---:|---:|---:|
| 64 | 40 / 48 | 227 s | — |
| 128 | 3 / 48 | 1,850 s | about 225 s |
| 256 | 4 / 48 | 2,150–2,163 s | about 210 s per worker |

The leverage gate (Section 7) has not been evaluated. It needs the finished package and is run on validation only.

## Results: leverage gate (appended 2026-11-13 03:15 WIB)

**Verdict: FAIL.** The failing cells are L3 at 64, 128 and 256, and L2 at 256. Under stop rule 8.6 the scope is not
promoted, and the Paper B search benchmark is not run on it as a RIS-relevant benchmark. The thresholds above are unchanged.

**Run:**
- `paper-b-neural-pair-search/scripts/ris_leverage_rr_20261113.py` and `ris_leverage_rr_summary_20261113.py` (definitions of
  `ris_leverage_20261112.py`; best = b35 at 300 / 500 / 800 ms, seed 7, never below all OFF).
- Ran on 3060-3, CPUs 0–4 and 7, in `/var/tmp/atomwise-bench-ris-relevant-search-20261113/runs/leverage/`; scored at 2026-11-12T18:03:31Z.
- Validation only: 7 trajectories (0034–0040) × 16 frames = 112 frames per array. No test-split file was read for any metric.
- Package: `output_sha256sums` verified (14 files, all OK).
  - `channel_realizations.h5` sha256 `78cf01f7f77f792fd0fab5cb57606a898abc7725aa701cf4ee73b31682929742`
  - `ris_relevant_split.json` sha256 `f14bf2f234d9bd6f128ea8cb125fe2f09713ef9e02e5bfa0908b5bd4bed836ff`
  - sidecar sha256 `ecb1a0539a26853dda48fe45c6ca19e7f5fabb667ce291423994aa61cc0bafeb`
- **Production gate:** `verify_bench_production_gate.py --production` through the new wrapper
  `scripts/run_bench_ris_relevant_gate.py` (registry extended in-process) gives **PASS**.
  - All 9 checks pass; 48 / 48 cases included and 0 failing.
  - LOS geometry agreement is 13,149 / 13,152 (0.99977), and the loader rate error is 0.0.
  - The first two invocations stopped at the freeze check, because the code tree's root lacked the v1.1 freeze files.
    The passing run uses the pilot workspace's own `src/`, the same tree the pilot chain used.

| Gate | 64 | 128 | 256 | Threshold (64 / 128 / 256) |
|---|---:|---:|---:|---|
| L1a pathless served user-frames | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | ≤ 2% / 2% / 2% |
| L1b flagged frames | 0.0% ✓ | 0.0% ✓ | 0.0% ✓ | ≤ 5% / 5% / 5% |
| L2 median (best − all OFF), bit/s/Hz | 0.0741 ✓ | 0.1401 ✓ | 0.1340 ✗ | ≥ 0.05 / 0.07 / 0.14 |
| L3 median (all OFF − no RIS), bit/s/Hz | 0.0038 ✗ | 0.0036 ✗ | 0.0059 ✗ | ≥ 0.06 / 0.08 / 0.14 |

Counts: 368 served user-frames per array, 0 pathless, and 0 flagged frames.

**Reported, not gating** (validation, 112 frames per array, none flagged):

| Quantity | 64 | 128 | 256 |
|---|---:|---:|---:|
| Best − all OFF, mean | 0.273 | 0.180 | 0.280 |
| Best − all OFF, p90 | 0.956 | 0.419 | 0.704 |
| Frames with best − all OFF > 0.1 | 41.1% | 58.9% | 51.8% |
| Frames with best − all OFF > 0.5 | 24.1% | 4.5% | 18.8% |
| All OFF − no RIS, mean | 0.0579 | 0.0445 | 0.0300 |
| All ON − all OFF, median | -0.0153 | -0.0018 | -0.0101 |
| Worst-user rate, median: no RIS / all OFF / best | 1.330 / 1.338 / 2.149 | 3.973 / 3.987 / 4.102 | 2.372 / 2.361 / 2.981 |
| Frames with worst-user rate 0 at all OFF | 0.0% | 0.0% | 0.0% |
| BS-blocked / RIS-LOS: user-frames | 93 | 108 | 180 |
| BS-blocked / RIS-LOS: RIS(best) − direct power, median dB | -52.4 | -46.7 | -33.4 |
| BS-blocked / RIS-LOS: ideal-plate bound vs LOS, median dB | -28.8 | -22.2 | -15.6 |
| BS-blocked / RIS-LOS: d2 median, m | 232 | 230 | 89 |
| BS reflection-only / RIS-LOS: user-frames | 275 | 260 | 188 |
| BS reflection-only / RIS-LOS: RIS(best) − direct power, median dB | -41.0 | -36.8 | -34.3 |
| BS reflection-only / RIS-LOS: ideal-plate bound vs LOS, median dB | -29.0 | -22.4 | -16.0 |
| BS reflection-only / RIS-LOS: d2 median, m | 89 | 89 | 90 |
| BS–RIS distance d1 over the 7 validation cases: min / median / max, m | 13.6 / 18.9 / 19.7 | 13.6 / 18.9 / 19.7 | 13.6 / 18.9 / 19.7 |

**Reading (descriptive, decides nothing):**
- **The setting does matter.** Median best − all OFF is 0.074 / 0.140 / 0.134 bit/s/Hz, against 0.015 / 0.022 / 0.045 in the natural
  scope, and 41–59% of frames gain more than 0.1.
- **The RIS does not help at all OFF.** All OFF − no RIS has a median of only 0.004–0.006, so L3 fails by a factor of 15–25.
  The L3 thresholds were scaled from a natural-scope *mean* (0.019 / 0.025 / 0.043) and are tested on the median. In this scope
  the all-OFF − no-RIS mean is 0.058 / 0.045 / 0.030, but the rule is the median and is not changed.
- **L2 at 256 misses narrowly** (0.134 against 0.14).
- **No served user is pathless or RIS-only:** no BS-blocked user has a zero direct channel. Even at the best setting, the RIS path
  stays 33–52 dB below the BS's diffracted or reflected paths, and the ideal-plate bound is −16 to −29 dB.
- **Caveat:** the link-class label counts in `metadata.json` differ between arrays for the same 368 user-frames (BS-blocked
  93 / 108 / 180). This affects only the descriptive class split and is not investigated here.

**Under this declaration:**
- The scope is not promoted, and no search benchmark is run on it as RIS-relevant.
- The sealed test (0041–0047) stays unopened.
- A new rule set, for example an L3-free gate or a gate on the mean, needs a new dated declaration and a new seed domain word (§8.6).

**Files:**
- `outputs/runs/bench_ris_relevant_pilot_20261112/leverage_20261113/`: `summary.json`, `leverage_gate.json`,
  `production_gate.json`, `stage.log`, `package_hashes.txt`.
- Per-frame JSONL stays on 3060-3 in `runs/leverage/a*_w*.jsonl`.
