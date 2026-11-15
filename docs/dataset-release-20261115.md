# GeoRIS-Bench 1.0 Release Record (2026-11-15)

**Public version:** GeoRIS-Bench **1.0** (Mendeley Data, DOI 10.17632/td3w4cwbg5.1), files `GeoRIS-Bench-1.0-{rotterdam,maastricht,utrecht}.zip`. Internal build scopes `ris_relevant_v3` (Rotterdam) and `ris_relevant_v3x` (held-out cities) map to this release; earlier internal scopes v1/v2 were never released.

**Identifier mapping (repository ↔ public files).** The public files use clean names; the repository code and run records use the internal ones. Mapping: trajectory ids `rotterdam-rr3-NNNN` → `rotterdam-NNNN`, `maastricht-rx3-NNNN` → `maastricht-NNNN`, `utrecht-rx3-NNNN` → `utrecht-NNNN`; scope/kind `ris_relevant_v3`, `ris_relevant_v3x` → `georis_bench_1.0`; `ris_relevant_split.json` → `splits.json`; `format` prefix `coupledris-bench` → `georis-bench`. Removed from the public files: absolute lab paths (→ `<build-root>`), `*_host_only` trace paths, the internal package sidecar and the map-layer planning note. Channel, label and trajectory data are otherwise byte-identical; the build (`finalize.py`, `verify.py`, `build_bundle_v2.sh`) checked 1:1 HDF5 structure, identical arrays on a random sample of 30 frames per city, rate recomputation within 3.6e-12 bit/s/Hz, and no remaining internal identifiers.

**Scope:** Coupled 1-bit RIS multiuser urban benchmark (`ris_relevant_v3` + `ris_relevant_v3x`)  
**Released cities:** Rotterdam (training/validation/sealed test) and two held-out cities, Maastricht and Utrecht (12 check + 36 sealed test trajectories each), giving **216 trajectories** (Rotterdam 120, of which 119 are in frozen v3 and `rr3-0069` was rebuilt as a replacement; Maastricht 48; Utrecht 48), 16 frames per trajectory, $N \in \{64, 128, 256\}$. The two held-out cities serve as the cross-domain (unseen-city) test.  
**Status:** Structural, dense-loader, geometry-agreement and leverage checks pass for all three cities at all array sizes. Public redistribution remains gated by licensing cleanup and path sanitisation (Sect. 6).

---

## 1. Inventory & Integrity Summary

| City | Scope | Splits (`tr/v/te` or `ch/te`) | Valid / Expected | Package Size (`v3x`) | Geometry Agreement (`>= 0.75`) | Production Gate & Max Loader Err | Ready Arrays |
|---|---|---|---|---|---|---|---|
| **Rotterdam** | `v3` | `83(+1 repl.) / 18 / 18` | `119 / 120` (+`rr3-0069` replacement) | Frozen `v3` (`318`-entry `STAGED_SHA256SUMS`) | `99.28%` (`28,212 / 28,416`) | `pass`; recovery err `1.78e-15` | `64, 128, 256` |
| **Maastricht** | `v3x` | `12 / 36` | `48 / 48` | `1,213.9 MB` (`1,213,869,636 B`) | `99.95%` (`12,666 / 12,672`) | `pass`; `0.0` | `64, 128, 256` |
| **Utrecht** | `v3x` (post-U1) | `12 / 36` | `48 / 48` | `1,212.0 MB` (`1,211,984,423 B`) | `99.56%` (`12,138 / 12,192`) | `pass`; `0.0` | `64, 128, 256` |

---

## 2. Leverage Check (pre-declared)

Thresholds: $L_2 = \mathrm{median}(\text{best} - \text{all-OFF}) \ge 0.050, 0.070, 0.140$ bit/s/Hz for $N=64, 128, 256$; $L_{1a} \le 0.02$ pathless user-frame share; $L_{1b} \le 0.05$ flagged-frame share.

- **Rotterdam (`validation 0084–0101`):** `L1a = 0.0`, `L1b = 0.0` across all arrays; `L2 = 0.302 / 0.418 / 0.782` (`PASS` on 64/128/256). `L3` (`alloff - noris`): `-0.0004 / +0.0028 / +0.0109`.
- **Maastricht (`check 0000–0011`):** `L1a = 0.0`, `L1b = 0.0`; `L2 = 0.266 / 0.241 / 0.347` (`PASS` on 64/128/256).
- **Utrecht (`check 0000–0011`, post-U1):** `L1a = 0.0`, `L1b = 0.0`; `L2 = 0.130 / 0.417 / 0.293` (`PASS` on 64/128/256).

---

## 3. Replacement Record

1. **Rotterdam `rr3-0069` Replacement (`rotterdam-rr3-recovery-seed2026111502-i0069`):**
   - Original `rotterdam-rr3-0069` exclusion in frozen `v3` (`119/120`) is preserved untouched so prior Rotterdam benchmark denominators remain exact.
   - Rebuilt separately as a new trajectory under predetermined `seed_word = 2026111502` (`16` expected frames `[0, 23, ..., 349]`): `0` external wire collisions, `0` pathless served users, `288/288` (`100.00%`) geometry agreement, `max_dense_check_error = 1.15e-14`, `max_loader_rate_error = 1.78e-15`, `status = pass`.
   - Recovered HDF5 SHA-256 (`64 / 128 / 256`):
     - `64`: `9066a0110dc58839d0bff47ceb58663fe2aafaded1d54d5e26a68e68a42f1ca3`
     - `128`: `fe4ccc079ebdfa203a6d801586ffa94680d515d4386d5a438e279434cea7bda7`
     - `256`: `806adc35c2a00fc02c0f81238b207d65930564ef23f9261f1e8102805a086f53`

---

## 4. Package Checksums & Remote Paths

### Remote Package Locations (`labkc-comm3`)
- `maastricht`: `/mnt/home/faris/atomwise-bench-ris-relevant-v3x-20261114/maastricht/channels/package`
- `utrecht`: `/mnt/home/faris/atomwise-bench-ris-relevant-v3x-20261114/utrecht/channels/package`

### Primary Artifact SHA-256 Hashes (`v3x` Cities)

| City | `channel_realizations.h5` | `metadata.json` | `package/provenance.json` | `package/trajectories.parquet` | `ris_relevant_split.json` |
|---|---|---|---|---|---|
| **Maastricht** | `c4a49024a5fc4ce39f60da1b62151c56f47f9aea564679953dcac597d9504d24` | `5eff7e9ee6e5fe380c0b9596ef2cfd36df5bd67638a5193dd68b5de550204fbf` | `5f2ad343966373be184907f5e3eb097ee1957ac45f1187a27cd1c1da29e82139` | `0c2354aa82b1510600fc699482adea421fcd43c4ab4481b46c50878fc044801a` | `ad2136a799bdddf7c843b5d808d52a362ebf0b965661427a71dff17edec91658` |
| **Utrecht** | `c89f14b8519a8613c366531ca16fd7aad67147fce0acbe2a3fc99f8acc775c24` | `b52b890864a08a6b14be65af4611d7293434ec86ad9c4dd1f1cbb98e1f3a48c3` | `3d557f4db99c6eaeefe702a92e85294986879df7126dcee08d9172c640b3b583` | `97d8d46345ec981477a4d49af48fdebe3846a8f864fdfa945ee454eb667f26ee` | `61c820c010148a7c36034757a306fd595c5693bf1a514193bc02ec1a28e3b42a` |

Full per-file manifests are in `paper-a-ris-physics/outputs/runs/dataset_5city_audit_20261115/<city>/` (`STAGED_SHA256SUMS` for Rotterdam; `output_sha256sums` for `v3x` cities).

---

## 5. Future Work: Munich and Berlin

Munich and Berlin were generated under the same `v3x` rules but are **not part of this release**; they are planned for a future version.

- **Munich.** Its pre-declared leverage check failed at every array size (`L2 = 0.030 / 0.051 / 0.050` against `0.05 / 0.07 / 0.14`). The cause is the user-selection rule, not a build fault: 92.9% of check user-frames have a blocked direct path but a strong reflected base-station link (median baseline about 9.28 bit/s/Hz at 64 elements), so the RIS changes the worst-user rate materially in only about a quarter of frames.
- **Berlin.** Its leverage check passed at 64 and 128 elements (`0.105 / 0.088`) and failed at 256 (`0.130 < 0.140`). It is held back so the release contains only cities that pass at every array size.
- **Not used.** The post-hoc addenda (`leverage_gate_v2_urban_canyon.json`, `leverage_gate_v2_uniform008.json`) and the 15 Nov replacement of selected check trajectories (Munich 0000/0003/0004/0011, Berlin 0008/0009; originals in `<city>/backup2_units/` on comm3) change only the check sample and leave the sealed test data unchanged, so they do not establish RIS relevance and are not used.
- **Planned fix (v4).** Declare, before any rebuild, a user-selection rule that also requires a weak reflected base-station link (based on geometry or path strength, never on achieved rates); regenerate **all** trajectories of both cities (check and test) under it; run the check once; then run the sealed test once.

---

## 6. Provenance, Licensing & Known Caveats (Required Before Public Redistribution)

Per [`dataset-card/LICENSES/README.md`](dataset-card/LICENSES/README.md), [`dataset-card/LICENSES/phase0b-map-sources.md`](dataset-card/LICENSES/phase0b-map-sources.md), and `ALL_DATA_AUDIT.json`:
1. **Upstream map & terrain licenses (preserve individual notices):**
   - **3DBAG (Rotterdam, Maastricht, Utrecht):** CC BY 4.0 — *3DBAG by the 3D geoinformation research group and 3DGI, CC BY 4.0*.
   - **AHN4 0.5 m DTM (Dutch scenes):** CC0 1.0 (*Actueel Hoogtebestand Nederland (AHN4), DTM 0.5 m, via PDOK*).
   - **OpenStreetMap layers / fallbacks:** ODbL 1.0 (*© OpenStreetMap contributors*; keep geography separable if share-alike applies).
   - **PyNEC 2.3.4 / NEC++:** Numeric impedance arrays (`Z_RIS`) only; do not bundle GPL solver binaries/source inside data packages.
2. **Scientific / structural caveats:**
   - `H_direct` includes load-independent structural scattering of the RIS aperture; `H_direct_uncorrected` is stored separately (`g6_note`).
   - Integrity audits certify structural/numerical correctness (`complete`, `finite`, dense-loader consistency `<= 1e-10`, geometry agreement), **not** city-scale measured RF validation or inclusion of vegetation/vehicles/street furniture.
   - Public bundle release requires sanitizing absolute lab filesystem paths and private repository links in metadata sidecars (`release_ready: false` until sanitized and attribution files are bundled).
