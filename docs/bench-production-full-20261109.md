# CoupledRIS-Bench full production declaration — 2026-11-09 WIB

Preregistered declaration for full production of CoupledRIS-Bench across development maps.
Authorized following user approval and successful verification of the three-map pilot gate
(`outputs/runs/bench_production_pilot2_20261109/pilot_gate.json`, `status: pass`).

This declaration is recorded before generating any production case manifest or running compute.
No held-out geography (Berlin, Utrecht) or legacy test data is opened or generated.
The frozen v1.1 generator lock (`e152e78cb7b66204ea4080339c29a80a53c94685f8c3743b4eddd0cbab6e0ecd`)
is verified before all pipeline operations.

---

## 1. Production size

- **Development maps:** 3 maps (`munich`, `rotterdam`, `maastricht`).
- **Trajectories:** 120 natural trajectories per development map = 360 trajectories total.
- **Selected frames:** 16 approximately equally spaced original 20 ms samples per trajectory:
  - 1,920 frames per map;
  - 5,760 frames across all 3 maps.
- **Arrays:** 64, 128, and 256 element RIS arrays evaluated at 5.53 GHz:
  - 5,760 array-frames per map;
  - 17,280 array-frames across all 3 maps.
- **Base station & served users:**
  - 4-port BS array (vertical dipoles spaced half-wavelength horizontally);
  - 2 to 4 served users per frame scheduled under the frozen v1.1 scheduler.
- **Challenge subset and declared share:**
  - In the frozen v1.1 generator design (`docs/bench-users-v1-design-20261108.md`, `scripts/generate_bench_users_preview.py`),
    the declared challenge subset contains 8 cases across categories (`shadow_edge`, `shadow_edge_sloped`,
    `all_no_los`, `dense_crowd`, `elevated_crowd`) out of 50 preview cases, establishing a declared
    challenge share of **16.0%** (8 / 50).
  - In full production, this share corresponds to 16.0% of cases (~19 cases per map, ~58 cases total).
  - *Driver status note:* The current non-pilot driver `scripts/build_bench_production_cases.py` samples
    `natural(generator, t, rng, counts)` for each trajectory index, generating 120 natural cases per map
    (0% challenge share in the baseline driver output). This constraint and its counts are audited
    and reported in the manifest receipt.

---

## 2. Seed streams and disjointness proof

- **Base seed:** `users.SEED = 202611081501` (anchored in `src/atomwise_ris/bench/users.py`).
- **Production seed rule:**
  ```python
  rng = np.random.default_rng(np.random.SeedSequence([users.SEED, 20261109, im, i]))
  ```
  where:
  - `users.SEED = 202611081501`: immutable base entropy word;
  - `20261109`: production stream domain separator;
  - `im`: map index (`0` for Munich, `1` for Rotterdam, `2` for Maastricht);
  - `i`: trajectory index (`0` to `119`).

### Disjointness proof against preview and pilots

The preview and pilot pipelines in `scripts/generate_bench_users_preview.py` and `docs/bench-users-v1.1-freeze-20261109.md`
allocated the second entropy word as follows:
- `1`: natural preview cases (`[users.SEED, 1, im, case_index]`);
- `2`: challenge preview cases (`[users.SEED, 2, challenge_index]`);
- `3`: anonymous bulk natural draws (`[users.SEED, 3, im, draw_index]`);
- `4`: bulk parquet row permutation (`[users.SEED, 4]`);
- `5`: site planning pool (`[users.SEED, 5, im]`).

The pilots reused fixed preview cases (`01-munich-natural`, `15-rotterdam-natural`, `39-maastricht-natural`),
which draw exclusively from stream `1`.

Because the production seed stream uses `20261109` as its second word, and `20261109 ∉ {1, 2, 3, 4, 5}`,
the collision probability in NumPy's `SeedSequence` hashing is bounded by the 128-bit hash collision
resistance ($\ll 10^{-30}$). The production seed sequence cannot collide with any preview, bulk, or pilot draw.

---

## 3. Labeling protocol

Labels follow lines 10–20 of `docs/bench-production-pilot-20261109.md` exactly:
- **Selected labeled frames:** 4 selected frames per trajectory (frames approximately equally spaced across the trajectory duration).
- **Parents:** 2 parents per selected frame:
  1. `all-OFF` reference;
  2. Bounded single greedy configuration.
- **Pair coverage:**
  - **64 elements:** enumerate all 2,016 pairs for each parent;
  - **128 elements:** enumerate all 8,128 pairs for each parent;
  - **256 elements:**
    - 4,096 deterministic uniform sampled pairs per ordinary parent;
    - Full-pair audit: enumerate all 32,640 pairs for one parent on every tenth trajectory within each map (fixed index `i % 10 == 0`, first labeled frame).
- **Label metrics:** Flip gains in bit/s/Hz and pair synergy relative to the same reference parent.

---

## 4. Splits and leak-prevention policy

### 4.1 Leave-environment-out rotation (geographical cross-validation)
- **Fold 1 (`munich_test`):** Train on Rotterdam + Maastricht (240 trajectories), evaluate on Munich (120 trajectories).
- **Fold 2 (`rotterdam_test`):** Train on Munich + Maastricht (240 trajectories), evaluate on Rotterdam (120 trajectories).
- **Fold 3 (`maastricht_test`):** Train on Munich + Rotterdam (240 trajectories), evaluate on Maastricht (120 trajectories).

### 4.2 Within-map trajectory split (in-environment evaluation)
Trajectories within each map are grouped to ensure that all 16 frames of any single trajectory belong to the same split:
- **Train (70%):** 84 trajectories per map (indices `0000`–`0083`, 252 total across 3 maps);
- **Validation (15%):** 18 trajectories per map (indices `0084`–`0101`, 54 total across 3 maps);
- **Test (15%):** 18 trajectories per map (indices `0102`–`0119`, 54 total across 3 maps).

### 4.3 Boundaries
- Held-out test cities (Berlin, Utrecht) are never opened, parsed, or generated.
- No trajectory is split across train/validation/test folds, eliminating temporal auto-correlation leakage.

---

## 5. Budget caps and resource feasibility

Computed from observed stage costs recorded in `outputs/runs/bench_production_pilot2_20261109/pilot_gate.json`:

### 5.1 Wall time cap (projected wall time ≤ 72 h)
- **Measured pilot per-stage projections:**
  - Ray tracing solver: 12,099.6 s (~3.36 h, on GPU 0);
  - Trace path export: 4,897.6 s (~1.36 h);
  - Scene load and port setup: 665.4 s (~0.18 h);
  - Channel conversion: 164,301.2 s (~45.64 h, on CPU cores excluding 5 and 11);
  - One-time NEC factorization setup: 3,747.6 s (~1.04 h);
  - Label generation (conservative audit rate): 143.5 s (~0.04 h).
- **Total projected wall time:** ~185,855 s ≈ **51.63 h**.
- **Budget cap verification:** 51.63 h ≤ 72.0 h (**20.37 h headroom**, ~28.3% margin). Passes.

### 5.2 Storage cap (projected storage ≤ 60% of free space on each volume)
- **Trace volume (`/mnt/home` on 3060-3):**
  - Free space recorded at pilot gate: 210,330,648,576 bytes (~210.33 GB);
  - 60% budget cap: 126,198,389,145 bytes (~126.20 GB);
  - Projected raw path bytes: 3,828,702,720 bytes (~3.83 GB);
  - Projected usage share: **1.82%** of free space. Passes.
- **Package volume (`/var/tmp` on 3060-3):**
  - Free space recorded at pilot gate: 72,680,919,040 bytes (~72.68 GB);
  - 60% budget cap: 43,608,551,424 bytes (~43.61 GB);
  - Projected channel HDF5: 8,126,701,440 bytes (~8.13 GB);
  - Projected labels HDF5: 1,047,849,120 bytes (~1.05 GB);
  - Projected total package size without working overhead: ~13,003,253,280 bytes (~13.00 GB);
  - Projected usage share: **17.89%** of free space. Passes.

Both volumes comfortably satisfy the $\le 60\%$ storage headroom policy.

---

## 6. Pass criteria and stop rules

### 6.1 Per-case and per-map acceptance criteria
1. **Complete HDF5 schema:** Packages cover all 3 maps, all 3 arrays, and all declared cases and frames without missing datasets or attributes.
2. **Finite values:** Zero NaNs, infinities, or unphysical values in channels, matrices, and labels.
3. **Loader consistency:** Recomputed dense mutual-coupling rates reproduce stored values within absolute tolerance $\le 10^{-10}$ bit/s/Hz.
4. **LOS sanity:** Traced BS–RIS link maintains valid LOS on all three maps; binary LOS agreement between traced paths and geometry labels across served groups $\ge 75\%$.
5. **Fail-closed mesh policy preflight:** Every map passes all 5 geometric integrity checks:
   - Terrain bounds: $z \in [\text{terrain\_min} - 0.5\text{ m}, \text{terrain\_max} + \text{height} + 0.5\text{ m}]$;
   - Ground reference match: Building base differences within $\le 2.5\text{ m}$;
   - Normal orientation: 100% outward-oriented Newell normals;
   - Dropped positive area share $\le 0.5\%$ per class;
   - Emitted area conservation relative difference $\le 0.1\%$ per class.

### 6.2 Fail-closed stop rule
If more than **2% of cases in any map** fail any validation check (i.e. $> 2$ failed cases out of 120 trajectories for that map), execution on that map is halted immediately, a `FAILED` marker is written, and the failure is escalated. Processing on unaffected maps continues only under lead approval.

---

## Amendment 2026-11-09: separate challenge subset (lead decision)

Following the lead's decision to preserve the declared 16.0% challenge share while keeping the natural 360-case production manifest untouched, a separate challenge manifest of 23 trajectories per development map (69 trajectories total, bringing the overall full production count to 360 + 69 = 429 trajectories, 23 / (120 + 23) = 16.1%) has been generated and registered.

### 1. Challenge subset specification
- **Size:** 23 trajectories per development map (Munich, Rotterdam, Maastricht) = 69 trajectories total; 16 selected frames per trajectory = 1,104 frames total; 3,312 array-frames across 64/128/256 elements.
- **Seed stream rule:**
  ```python
  rng = np.random.default_rng(np.random.SeedSequence([users.SEED, 202611092, im, i]))
  ```
  where `users.SEED = 202611081501`, `202611092` is the challenge domain separator word (disjoint from preview/bulk words {1, 2, 3, 4, 5} and natural word 20261109), `im` is the map index (0, 1, 2), and `i ∈ [0, 22]`.
- **Category order:** `CHALLENGE_ORDER = ("shadow_edge", "shadow_edge_sloped", "all_no_los", "dense_crowd", "elevated_crowd")`. Trajectory `i` first attempts category `CHALLENGE_ORDER[i % 5]`.
- **Substitution rule and attempt limit:** Generation is confined strictly to the target map (no multi-map fallback). Each category is attempted up to **20 attempts** on the same trajectory `rng`. If 20 attempts fail, the generator substitutes cyclically to the next category in `CHALLENGE_ORDER` with the same 20-attempt limit on the same `rng`. If all 5 categories fail after 20 attempts each (100 total attempts), generation fails closed. (In the production run, 0 substitutions occurred: all 69 cases satisfied their primary requested scenario).

### 2. Scope, manifest identity, and isolation policy
- **Manifest path:** `outputs/runs/bench_production_full_20261109/challenge_cases.json`
- **SHA-256:** `999d221ef4b3ac46ad7a66871cd176470d38cd08faac051159826b4575205f91`
- **Registered scope:** `development_challenge` in `src/atomwise_ris/bench/production_pilot.py`.
- **Labeling protocol:** Identical to the natural production protocol (Section 3): 4 labeled frames per trajectory, 2 parents (all-OFF and greedy), all pairs at 64 and 128 elements, 4,096 uniform sampled pairs at 256 elements with full-pair audit on `i % 10 == 0`.
- **Isolation policy:** The challenge subset is strictly its own scope and manifest. It is **never mixed into the natural splits or the leave-environment-out folds**: it serves as an independently reported stress-test evaluation benchmark.

### 3. Acceptance criteria and separate stop rule
- **Per-case validation:** Challenge trajectories use the exact same per-case geometric, schema, and channel checks as natural cases.
- **Separate per-map 2% stop rule:** A distinct 2% stop rule applies independently to each map's challenge subset. With 23 cases per map, a 2% failure threshold means that $> 0.46$ failures, i.e. **any single failed case**, immediately halts that map's challenge execution, writes a `FAILED` marker, and escalates.

### 4. Updated resource projection
- **Total production trajectories:** 360 natural + 69 challenge = 429 trajectories (6,864 frames; 20,592 array-frames).
- **Projected wall time:** Serial runtime scales from natural 51.63 h as:
  $$\text{Wall time} \approx 51.63\text{ h} \times \frac{429}{360} \approx 61.5\text{ h}$$
  This remains well within the **72.0 h budget cap** (10.5 h headroom, ~14.6% margin).
