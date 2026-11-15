# CoupledRIS-Bench RIS-relevant v3: cross-city held-out extension (declared 2026-11-14 14:40 WIB)

**Status:** declared **before** any cross-city generation, data download, scene build, tracing or channel build.
Sections 1–9 are fixed and never loosened. Results are appended only below "Results".

**Purpose:** a one-shot cross-city generalisation test for Paper A, on four cities that no method has seen:
**Munich, Maastricht, Utrecht, Berlin**. All methods (and the v3 rules) are frozen from Rotterdam
(`docs/bench-ris-relevant-scope-v3-20261113.md`). No training, validation, tuning or design choice uses these cities.

**Authorisation for the held-out maps:**
- The user explicitly authorised opening the held-out maps **Berlin and Utrecht** on 2026-11-14, for this declared
  one-shot test only.
- Their sources are downloaded and their scenes built for the first time here.
- Their geometry is changed only as far as the existing fail-closed mesh-policy checks require
  (`docs/bench-production-mesh-policy-20261109.md`).
- No method or design choice may be made from their channels.

**Unchanged:** every frozen or hash-locked file, the v1/v2/v3 registries, and `PRODUCTION_SCOPES`. Held-out map ids are
admitted only in-process, by the new cross-city wrappers (Section 8).

## 1. Rules: v3, unchanged

Per city, the rules of v3 §§2–3 with Amendment A1 apply as written:
- **Sites:** union of 8 v1 planning draws at R1, BS–RIS distance 5–20 m with a clear segment; v1.1 coverage and diversity
  rules; ≥ 6 sites; if fewer, the same union at R2 (5–25 m); if still fewer than 6, the city **fails**.
- **Users:** v1.1 natural crowd; 0.5 of anchors in the RIS zone. A user is eligible if, at all 16 selected frames, it is:
  - BS-blocked; and
  - RIS-LOS; and
  - within a 3-D RIS–user distance of **r = 40 m** (the v3-accepted radius, fixed here; no ladder); and
  - within **80°** of the panel normal.
- **Schedule:** 2–4 served users from the eligible pool only, with 25 crowd draws per site and sites rotated.
- **Tracer:** Sionna RT 2.0.1, max_depth 2, LOS + specular + diffraction, `samples_per_src = 1e6`,
  `max_num_paths_per_src = 2e6`, seed 7. One shared direct trace per frame (external receivers only) is merged into all
  three array traces (v3 §3).
- **Channels:** NEC-2 coupling with the existing Z_RIS for 64 / 128 / 256 elements, unchanged.

## 2. Cities, order, seeds, ids

| Order | City | Map index im | Seed word | Ids | Source status |
|---:|---|---:|---|---|---|
| 1 | Munich | 0 | **2026111401** | `munich-rx3-0000..0047` | development map; scenes and mesh checks exist |
| 2 | Maastricht | 2 | **2026111402** | `maastricht-rx3-0000..0047` | development map; scenes and mesh checks exist |
| 3 | Utrecht | 4 | **2026111403** | `utrecht-rx3-0000..0047` | held-out; 3DBAG LoD2 + AHN DTM + OSM (as Rotterdam) |
| 4 | Berlin | 3 | **2026111404** | `berlin-rx3-0000..0047` | held-out; Berlin LoD2 CityGML + DGM1 + OSM |

- **Disjointness:** the seed words are disjoint from {1, 2, 3, 4, 5, 20261109, 202611092, 20261112, 20261113, 202611133}
  and from each other.
- **Streams:** site draws `[SEED, w, 0, im, k]` (k = 0…7); trajectory i `[SEED, w, 1, im, i]`; package labels `--seed w`.
- **Areas:** the 440 m × 440 m areas of `configs/bench/phase0b-area-proposals-20261108.json` (Berlin EPSG:25833 Friedrichstadt;
  Utrecht EPSG:28992 Lombok–Kanaalstraat).

## 3. Size and split (per city)

- **Size:** **48 trajectories** × 16 frames × 64 / 128 / 256 elements.
- **CHECK split, indices 0000–0011 (12 trajectories, 192 frames per array):** scored **only** for the data check (Section 5).
- **TEST split, indices 0012–0047 (36 trajectories), SEALED:** generated, traced and packaged. No metric, gate or design
  decision reads it until the benchmark agent runs the frozen methods.
- There is no train or validation split.

## 4. Geometry and mesh gate (per city, before tracing; all must hold)

- **G1:** ≥ 46 / 48 trajectories generated (95%, as v3's 114 / 120).
- **G2:** 100% of served user-frames satisfy all four eligibility conditions; independent LOS recheck ≥ 99%.
- **G3:** d1 rule and clear BS–RIS segment for 100% of cases; ≥ 3 distinct sites used.
- **G4:** 2–4 served users; positions finite, in bounds, above ground.
- **G5:** trace preflight passes all five mesh-policy checks plus the endpoint preflight (evaluated by the trace stage, fail-closed).
- For Berlin and Utrecht only, the v1.1 street-terrain (slope) gate and ground-fill audit of `generate_bench_users_preview.py`
  must also pass, as they did for the development maps.

## 5. DATA CHECK (per city and per array size; check split only)

The definitions and thresholds are those of v3 §6:

| Cell | Quantity on check frames | 64 | 128 | 256 |
|---|---|---:|---:|---:|
| L1a | pathless served user-frames / served user-frames | ≤ 2% | ≤ 2% | ≤ 2% |
| L1b | flagged frames / frames | ≤ 5% | ≤ 5% | ≤ 5% |
| L2 | median (best − all OFF), best = b35 at 300 / 500 / 800 ms, seed 7 | ≥ 0.05 | ≥ 0.07 | ≥ 0.14 |

- **Per-array verdict:** a (city, array) whose three cells pass is **ready** for the benchmark. A failing (city, array)
  is excluded and reported as failed, with its values.
- There is no L3, and no "no RIS" comparison: the v3 G6 finding (the channel builder's `d` term depends on array size)
  is still open.
- The production integrity gate (`verify_bench_production_gate.py` through the wrapper) must also PASS.

## 6. Failure, stop and cutoff rules

1. A city fails, and is reported and excluded, if any of these fail: generation (fewer than 6 sites under R2, or G1–G4), the
   held-out source or slope gates, G5, the production gate, or the stop rule. Its arrays are also excluded if their data
   check fails.
2. **Stop rule (production, unchanged):** more than 2% of a city's cases fail → that city halts. With 48 cases, one
   failed case halts the city. One retry is allowed for infrastructure failures only; genuine failures are not retried.
3. **Hard cutoffs:**
   - Munich and Maastricht packaged and checked by **23:00 WIB 14 Nov**;
   - Utrecht and Berlin by **04:00 WIB 15 Nov**.
   - A city not ready by its cutoff is reported as **not completed**; the others are not delayed.
4. Nothing is loosened after any result. A failure is reported as is.
5. Any metric read of a test-split trajectory → report it and discard the affected analysis.

## 7. Outputs per city

- **Package:** under the city's run root.
- **Split file:** `ris_relevant_split.json` (check / test).
- **`<workspace>/<city>/READY.json`:** package path, split file, per-array check verdict and medians, gate status.
  Written only when the city is packaged and checked.

## 8. Compute and code (declared)

**Workspaces:**
- labkc-comm3: `/mnt/home/faris/atomwise-bench-ris-relevant-v3x-20261114` (code root + `<city>/` run roots).
- labkc-comm2: the same layout under `~/atomwise-bench-ris-relevant-v3x-20261114`. Finished comm2 cases and packages are
  copied to comm3.

**Placement:**
- One city per node at a time. Trace on both GPUs of that node (2 index shards); channel workers on CPU pairs 0,6 / 1,7 / 2,8,
  plus 3,9 after the trace.
- **CPUs 4, 5, 10 and 11 are never used on comm3 or comm2** (benchmark timing cores).
- labkc-l40 is used only if a channel environment works there quickly; labkc-comm1 is not used.

**New code (wrapping v3, which wraps v2, v1 and the frozen v1.1 generator):**
- `ris_relevant_v3x.py` — per-city seed words, cities, the check/test split.
- `ris_relevant_scope_v3x.py` — registry by manifest hash per city.
- `build_/validate_bench_ris_relevant_v3x_cases.py`.
- `run_bench_ris_relevant_v3x_{trace,channels,package,gate}.py`, plus node/final shell scripts.
- For held-out maps: a source fetch and geometry preparation wrapper that admits `berlin` and `utrecht` in-process.

## 9. Benchmark (separate agent; methods frozen from Rotterdam)

- **Methods:** greedy, pair, b3, b5, b35, M0, ME-GA, NSGA-II, learned heads (Rotterdam-trained, frozen).
- **Metrics:**
  1. worst-user SE;
  2. latency (mean / p95 ms);
  3. SE at 5 / 10 / 50 / 100 ms with evaluation counts.
- Run on the TEST split of each ready (city, array).

---

## Results

### Preflight requirement added by the user (14:45 WIB, before any full run)

A full 48-trajectory run starts only after three preflight stages pass:
- **A:** geometry gate G1–G4.
- **B:** mesh preflight G5.
- **C:** smoke run on check trajectories 0000 and 0001, all 16 frames, 64 / 128 / 256 elements. Checks:
  - trace and channel stages pass;
  - `H_direct_uncorrected` bit-identical between 64 and 128, and within 1% at 256;
  - no pathless served user;
  - no path-cap truncation;
  - all values finite;
  - loader rate error ≤ 1e-10;
  - projected package size per trajectory ≤ 1.6 GB / 48.
  - L2 is recorded descriptively only.

Integrity failures are treated as bugs: fixed, recorded here, and the smoke rerun. Low headroom is reported, never a reason to change rules.

**Package budget (user constraint):** ≤ 1.6 GB per city package, no fields beyond the v3 schema. The city script projects
`channel HDF5 × 1.12` (the v3 package/HDF5 ratio 1.1135) and stops **before** packaging if over budget.

### Preflight A results (14:50 WIB)

| City | Site rule | Sites planned → used | Generated | G1–G4 | Manifest sha256 |
|---|---|---|---|---|---|
| Munich | R1 < 6; **R2: 4 < 6** | – | – | **FAIL (stop rule 6.1): city excluded** | – |
| Maastricht | R1 | 17 → 4 (check uses 3) | 48 / 48 | PASS (2,112 served user-frames, 100% / 100%) | `ee602f85…ff82` |
| Utrecht | R1 < 6 → **R2** (5–25 m) | 11 → 8 (check uses 7) | 48 / 48 | PASS (2,096; 100% / 100%) | `e22ce496…59d6` |
| Berlin | R1 | 12 → 7 (check uses 6) | 48 / 48 | PASS (1,920; 100% / 100%) | `aa27f7ed…a75c` |

d1 (min / median / max) and d2 (q10 / median / q90), in m:
- Maastricht: d1 13.4 / 15.0 / 19.0; d2 24.7 / 31.4 / 36.8.
- Utrecht: d1 7.2 / 24.2 / 25.0; d2 14.7 / 29.4 / 37.5.
- Berlin: d1 8.3 / 15.0 / 19.8; d2 29.4 / 37.4 / 39.4.

The manifests are registered in `configs/bench/ris-relevant-v3x-registry-20261114.json`.

**Held-out sources** (fetched 14 Nov on 3060-3 from the URLs recorded in the 8 Nov audit
`outputs/audits/bench-phase0b-20261108/map_downloads.json`):
- **Hashes:** all four Berlin LoD2 zips, the Berlin DGM1 zip, both Utrecht 3DBAG tiles and the Utrecht AHN GeoTIFF
  **match the audit SHA-256s byte for byte**.
- **OSM:** both OSM files are new live API snapshots (sha256 recorded in `heldout_inputs/source-manifest.json`). The OSM API
  has no frozen release.
- **Berlin terrain:** one 2 km × 2 km DGM1 sheet (390_5818) covers the whole area. The neighbouring sheet names do not exist,
  because the sheet grid is 2 km.

**Held-out preparation** (`scripts/prepare_bench_ris_relevant_v3x_heldout.py`), applying the development-map rules:
- Utrecht:
  - street-terrain gate PASS (356 segments);
  - 1,101 LoD2 buildings;
  - ground fill with the frozen development decision (`interpolated_use_enabled`): 249,142 cells filled.
- Berlin:
  - street-terrain gate PASS (853 segments);
  - 132 LoD2 buildings (perimeter blocks);
  - 0 cells filled, because DGM1 has no gaps.

**Berlin geometry handling (validity only):**
- **Format conversion:** the DGM1 xyz cell centres were converted losslessly to a 1 m GeoTIFF over the area plus 60 m
  (560 × 560, 0 NaN).
- **Tile merge:** the four LoD2 tiles were merged into one CityGML file (3,370 members, 0 duplicate gml:ids).
- **Parsers:** Berlin LoD2 is CityGML 1.0 (AdV), as Bavaria, so the unchanged Munich parsers are reused
  (`berlin_lod2_v3x.py`).

**Code fixes during A** (integrity bugs, no rule change):
1. `ris_relevant_v3x` site planning is the v3 A1 union with the city seed word (unchanged rule).
2. The held-out map id is admitted in-process in `users`, `users_geometry` and the case compactor of
   `build_bench_production_cases.py`. The latter refused it with "only development maps are admitted" on the first Utrecht attempt.

### Preflight B, Utrecht: Fix U1 (15:00 WIB; geometry validity only, decided before any channel exists)

**The failure:** the first Utrecht smoke trace (rr3x 0000, on comm2) failed G5 `z_bounds_within_terrain_plus_height`.
- Mesh z_min was −1.08 m, against an allowed −0.14 m (terrain minimum 0.36 m − 0.5 m).
- The other four checks passed: 548 / 548 building bases matched with 0.0 m difference, outward normals 100%,
  dropped area 0, emitted-area difference 4.9e-8.

**Cause:** the 3DBAG LoD2.2 solid `NL.IMBAG.Pand.0344100000052254-0`, which 3DBAG itself flags `b3_val3dity_lod22 = [302]`,
has Wall/Roof faces 2.96 m **below its own GroundSurface** (ground 1.88 m, roof face at −1.08 m).

**Rule U1, applied to both Utrecht tiles** (`scripts/prepare_bench_ris_relevant_v3x_utrecht_tiles.py`):
- In a BuildingPart whose 3DBAG val3dity list is non-empty, remove every Wall/Roof face that has a vertex more than 0.5 m below
  the part's lowest GroundSurface vertex. Nothing else changes.
- Effect: 26 faces in 6 building parts removed, about 150 m² in total (log: `heldout_inputs/utrecht-fixU1.json`).
- The derived tiles `*.v3x-valid.city.json.gz` (hashes in `source-manifest.json`, with the original hashes alongside) feed both
  the user geometry and the RF scene.

**After U1:**
- The Utrecht preparation and generation were rerun with the declared seeds.
- Preflight A: PASS again. 48 / 48; R2; 11 sites planned → 7 used; 2,032 served user-frames at 100% / 100%;
  d1 6.9 / 22.3 / 25.0 m; d2 17.8 / 29.4 / 36.3 m.
- New manifest `4a22fade…73df`, registered. The pre-U1 manifest is kept in `superseded_U1/`; it was never traced beyond the failed smoke case.

**Compute event, 15:00 WIB:** labkc-comm2 and labkc-l40 stopped answering SSH (connection timed out at the gateway).
The Berlin smoke started on comm2 at 07:55Z has an unknown state; nothing from comm2 is used. All work continues on labkc-comm3.

### Preflight B results (comm3, trace-only smoke, 08:00–08:02Z)

All five G5 mesh checks pass on cases 0000 and 0001 for both held-out cities. Fresh traces were run on comm3 GPUs (CPUs 3 and 9); nothing from comm2 was used.

| City | Case | G5 | Building bases checked |
|---|---|---|---|
| Utrecht (post-U1) | 0000 | pass | 516 |
| Utrecht (post-U1) | 0001 | pass | 484 |
| Berlin | 0000 | pass | 516 |
| Berlin | 0001 | pass | 484 |

Maastricht passed G5 on 0000 and 0001 during its smoke (07:43Z).


### Takeover integrity audit (14 Nov 16:49 WIB, before full-build launch)

The user requested a fresh transcript, handoff, live-job and builder audit. Maastricht's original smoke had passed at 15:23 WIB, but no full build had started. A separate stricter CHECK-only gate was added at `scripts/rr3x_strict_smoke_check.py` (SHA-256 `969a9df99e1eb7afcbb3d804b51f776414d00a590c1f5396045254999037a925`) without editing the existing scientific rules or staged source files. It requires exact declared frame coverage, city/source identity and hashes, trace metadata/cap evidence, finite channels, loader agreement and no pathless served users. Maastricht 0000/0001 passed 128/128 trace records and 96/96 channel frames. Utrecht/Berlin must pass this gate on their own smoke artifacts before proceeding.

All 389 original staged hashes passed before launch. The continuation in `ops-takeover-20261114/` enforces the existing deadlines and size budget, preserves failures, excludes timing CPUs, and publishes READY only after checks. Maastricht full build began 16:49:07 WIB, followed by Utrecht then Berlin. Munich remains excluded; no criteria were relaxed. No method was tuned, and the audit opened no TEST channel metrics. See the takeover section of HANDOFF-20261114.md and workspace-notes/audits/20261114-takeover/ for reproducible evidence.

### Preflight C, Maastricht (smoke on 0000/0001, comm3; SMOKE.json written 08:23Z): PASS

- **Integrity checks:** values finite; loader rate error max 0.0; H_direct_uncorrected 64 == 128 (32/32 frames); 256 within 1% (max relative difference 0.0049); no served user without paths; max path index 18, below the cap.
- **Size:** projected 25.4 MB per trajectory, within the 1.6e9/48 budget.
- **L2 (descriptive only, not gating):** 0.677 / 0.802 / 2.04 for 64 / 128 / 256.
- **Operations:** the result was read late (11:00Z), so comm3 sat idle for 2.6 h. The full run was launched at about 11:05Z.
