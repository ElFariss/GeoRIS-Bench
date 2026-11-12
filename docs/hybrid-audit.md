# Hybrid DeepMIMO / ray-tracing + NEC-2 physics track: Stage 1 audit

Date: 2026-11-06. Scope: physics only (no optimizer results, no optimizer claims). Repository
state audited: `main` at `f4a8797` plus the untracked files listed by `git status`.
Every statement below is tagged **[V-file]** (verified by reading or computing from a local file in
this session), **[V-web]** (verified from a fetched web source; the fetch tool summarises pages with a
small model, so wording is paraphrased and exact license text should be re-read before any legal use),
**[V-num]** (verified by a numerical experiment in this session) or **[UNVERIFIED]**.

## 1. Which ray data exist locally, and do BS-to-RIS and RIS-to-UE links exist?

Both link types **exist** for the existing Munich scene, but only at the 64 ray-traced probe
positions, not at the NEC wire positions. Nothing needs to be traced to *start* (LOS-only checks);
new tracing is needed for a production hybrid because the probe lattice differs from the NEC lattice.

| Artifact | Content | Status |
|---|---|---|
| `data/raw/acquisition/munich_specular_v3/bs/paths.npz` | Sionna RT 2.0.1 per-path complex coefficient `a[rx,rx_ant,tx,tx_ant,path,time]`, delay `tau`, positions. 4 BS antennas to 256 receivers = 64 RIS probes + 192 user positions | [V-file] |
| `.../ris_00 ... ris_56/paths.npz` | 64 RIS probes as transmitters (8 per file) to 192 user positions | [V-file] |
| `data/raw/deepmimo_scenarios/munich_specular_v3_{bs,ris_XX}_fullpaths/` | DeepMIMO 4.0.5 export of the same traces: per-path `power`, `phase`, `delay`, `aoa_az/aoa_el`, `aod_az/aod_el`, `inter` (interaction code), `inter_pos` (up to 3 interaction points), `rx_pos`, `tx_pos`, `params.json` | [V-file] |
| `data/processed/munich_specular_v3/scenario.npz` | Coherent sums `G[64,4]` (BS to probe), `H[96,2,64]` (probe to user), `D[96,2,4]` (BS to user); 24 trajectories x 4 steps | [V-file] |
| `data/processed/munich_specular_{compute_v3,test_v1}`, `coupled_wire_test_v1` | independent re-trace, and the **frozen untouched test** data. The hybrid track does not read the test data | [V-file] |
| `data/processed/coupled_wire_v1`, `outputs/runs/coupling_halfwave64_v1/z_side8_segments{21,41,81}.npy` | existing free-space NEC-2 frames (70-port Z: 64 cells + 4 BS + 2 UE) | [V-file] |

Computed from the export (BS antenna 0; other antennas identical to within one path) [V-num]:

* BS to the 64 RIS probes: exactly 4 paths per link (one transmitter has one link with 5), LOS present in
  100 % of links. Interaction codes: LOS, ground reflection, two single building reflections, one
  double reflection class. Total unfolded path lengths 298 to 308 m, so the **incident field at the RIS
  is a sum of about four plane waves** (aperture 0.33 m, Fraunhofer distance 2D^2/lambda about 4 m, so
  the incident side is far-field to better than 0.002 rad of curvature phase).
* Last-interaction (reflection point) to RIS distances for the cell-0 paths: 4.84, 8.21, 4.87 m
  (cell 0 sample). For flat specular reflectors the wavefront is that of the image source at the
  unfolded length, so these are not a plane-wave validity limit, but they are a *geometric-optics*
  validity flag (reflector patch size vs first Fresnel zone is not checked by the tracer).
* BS to users (direct, RIS-free): median 2 paths, LOS in 69 to 70 % of links, up to 9 paths.
* Probe to users: 14 to 34 paths per link (median 22), LOS in 100 % of links. Users are 10 to 60 m
  from the RIS, so the **outgoing side is only partly far-field** (at 10 m the curvature phase across
  the aperture is about 0.16 rad). Element-resolved (sampled) phases are needed on this side.
* Conventions measured from the LOS path of a known geometry [V-num]:
  `aoa_az` is the azimuth of the direction **toward the source**; `aod_az` is the azimuth of the
  propagation direction; `aoa_el` and `aod_el` are **zenith angles from +z** (not elevations; the
  LOS arrival has 69.98 deg = 90 - 20.02 deg elevation); `power` equals 20 log10 |a| (matches
  Friis exactly, -96.78852 dB); `phase` omits the carrier delay phase, so
  `a = 10^(power/20) exp(j phase) exp(-j 2 pi f tau)`. Delay is float32, which limits reconstructed
  phase to about 1e-3 relative error (the repository's own parity files report 1.3e-3 and 1.5e-4).
* The ray-traced RIS is **virtual**: probes are isotropic, V-polarised nodes, not scene meshes
  (`scripts/trace_munich.py`), so the traced direct and multipath channels contain **no RIS
  scattering**. No double counting arises from the traces themselves [V-file]. A scene with a facade
  or panel mesh would need the panel footprint removed from the direct paths.

What is **missing** for the hybrid: (a) ray data at the NEC element positions: the probes use a
20 mm x 13 mm lattice, the NEC reference uses 28 mm x 30 mm pitch with 24 mm z-wires, and 24 mm wires
cannot sit at 13 mm vertical pitch without overlap; (b) element patterns (these come from NEC);
(c) the H-polarised component is irrelevant for z-directed wires (a z-wire couples only to the
theta-hat component) so V-V coefficients suffice; (d) no per-path data exist for the frozen test.

## 2. DeepMIMO

* Pinned: `deepmimo==4.0.5` in `pyproject.toml` (PyPI wheel uploaded 2026-08-25 per the saved PyPI JSON;
  Python >= 3.11; numpy < 2.3) [V-file]; exports written here carry `"version": "4.0.5"` [V-file].
* What it is [V-web]: an open-source Python package that converts ray-tracer outputs (Wireless InSite,
  Sionna RT via `sionna_exporter`, AODT) into a common scenario format, loads datasets
  (`Dataset`, `MacroDataset`, `DynamicDataset`) and generates channels. The docs describe an OSM to
  Sionna RT to DeepMIMO pipeline that uses **Sionna RT for the ray tracing**; DeepMIMO itself does not trace.
* Multi-link [V-web]: multiple TX/RX pairs are supported through `MacroDataset`; the scenario spec shows
  a BS-to-BS example using TX set 1 with RX set 1. Path quantities (power, phase, delay, AoA/AoD,
  interaction type and position) are available. Loading truncates to `MAX_PATHS` (docs: default 10);
  the 4.0.5 converter had a constant of 25 (repository `data/README.md` and `trace.json`)
  [V-file/V-web, the two defaults differ and were not reconciled]. The repository works around this by
  raising the converter capacity to the saved path count.
* **Arbitrary RIS location: not provided.** No page I could read documents RIS support, RIS links, or
  user-to-user links (overview, datasets capability page, converter, scenario spec; the six most recent
  release notes mention no RIS). Pre-computed scenario databases hold BS-to-UE-grid links, so BS-to-RIS and
  RIS-to-UE links for a new RIS position must be generated by tracing with the RIS positions included as
  both transmitters and receivers, exactly as `scripts/trace_munich.py` does. [V-web for the docs;
  the statement about hosted scenario contents is an inference, **[UNVERIFIED]**]
* **License inconsistency in 4.0.5 [V-web]:** the `LICENSE` file at tag `v4.0.5` is Apache-2.0
  (copyright 2025 DeepMIMO; "Update LICENSE to Apache 2.0" committed 2025-10-19), while `pyproject.toml`
  at the same tag and the PyPI metadata say `GPL-2.0-or-later` (also seen in the local saved PyPI JSON
  and in the pinned-commit `pyproject.toml`); `main` now says Apache-2.0. v4.0.0 release notes say the
  license moved from CC BY-NC-SA 4.0 (V3) to Apache 2.0 (V4). **Precaution taken:** the hybrid module
  does not import DeepMIMO; it reads the exported `.npz` files with numpy. Acquisition scripts that do
  import it already exist outside this track.
* **Scenario-asset licenses in the DeepMIMO database: [UNVERIFIED].** The docs pages carry no license
  or citation text and the scenario catalogue page did not render. Treat hosted scenarios as
  redistribution-unverified.

## 3. Sionna RT version and RIS status

* Pinned: `sionna-rt==2.0.1`, `mitsuba==3.8.0`, `drjit==1.3.1` (`pyproject.toml`, `requirements-lock.txt`;
  2.0.1 released 2026-03-31; the latest at retrieval was 2.2.0; Apache-2.0 for code) [V-file, V-web].
* RIS status [V-web]: the technical report states that, as of version 2.1, Sionna RT does not support
  RISs, that RIS support was added in 0.18 and that it survives in the Sionna 0.19.2 branch. The report
  documents 2.1, one minor version above the pin, so I also listed the **v2.0.1 source tree** through the
  GitHub contents API: `src/sionna/rt` has no RIS module and `radio_devices` contains only
  `radio_device.py`, `receiver.py`, `transmitter.py`. Conclusion: **no built-in RIS in the pinned
  version; the NEC-2 hybrid is custom either way** (the 0.19.2 model is a simplified ideal-reflector
  phase-profile device, not used).
* Convention evidence [V-num]: the saved free-space check gives magnitude error below 1e-4 and zero
  phase offset for a horizontal V-V LOS link, and the LOS path of the elevated BS (105 m) to a 3 m
  probe matches Friis to the saved precision with export phase 0 deg, so the V-V coefficient of an
  isotropic V antenna pair has **no extra sign or conjugation** for LOS. The docs I fetched do not
  describe the isotropic V pattern convention ([UNVERIFIED] from docs; verified empirically only).

## 4. Candidate scenes for tier T1 (two to three) and rights

Rights statements are the weakest part of the audit. What is verified: Sionna RT code is Apache-2.0
(README and PyPI) [V-web]; OpenStreetMap data is ODbL 1.0 with attribution and share-alike for derived
databases ("if you alter or build upon our data, you may distribute the result only under the same
license") [V-web]. What is **not** verified: any statement about the built-in scene assets. The
Sionna README and docs (`scene.rst`) list the scenes and give no creator, source-data, license or
attribution text, and the first 100 kB of the `munich.xml`, `etoile.xml` and `simple_street_canyon.xml`
files contain no license or credit comments (later bytes unread).

| Scene | Why | Geometry origin | Rights status |
|---|---|---|---|
| 1. Sionna `munich` (existing traces) | Already traced with the same RIS, BS and users; reusable for P4 LOS-only and for one T1 scene without new tracing | Mesh files are named after real buildings (for example church and company names), which indicates OSM-derived geometry **[UNVERIFIED inference]** | Repo is Apache-2.0; ODbL obligations of any OSM-derived meshes **[UNVERIFIED]**. Do not redistribute raw meshes; publishing derived numbers needs a legal decision |
| 2. Sionna `simple_street_canyon` | Synthetic urban canyon (buildings `building_1..6`, `floor`); controlled multipath with no third-party names | Looks hand-modelled; authorship **[UNVERIFIED]** | Same Apache-2.0 repo license, no OSM indications; cleanest of the built-ins |
| 3a. Newly generated OSM scene (DeepMIMO OSM pipeline: Overpass API to Mitsuba scene, no Blender) | Gives a third, independent real-city site with a license that **can** be verified | OpenStreetMap | ODbL 1.0 verified: credit OSM, share-alike on derived databases; whether ray data count as a Produced Work or a derivative database is a legal question **[UNVERIFIED]** |
| 3b. Sionna `etoile` or `florence` (fallback) | Real-city scenes without new generation | likely OSM-derived **[UNVERIFIED]** | as scene 1 |

DeepMIMO-hosted pre-computed scenarios are not proposed: they hold BS-to-grid links, they carry no
verified license, and they cannot supply BS-to-RIS and RIS-to-UE paths for an arbitrary RIS position.
Release recommendation: publish scripts, configuration and checksums rather than meshes or raw path
files until a rights decision exists (as the plan, section 8.4, already requires).

## 5. How the existing Munich export can be reused

1. **Incident side, scene 1.** The four BS-to-RIS paths per link (LOS, ground, two building reflections)
   are far-field plane waves at the RIS. Their amplitude, delay and arrival direction are in the export
   (`power`, `phase`, `delay`, `aoa_az`, `aoa_el` with the conventions above) and map directly onto
   the plane-wave incident operator of the derivation. The V-V coefficient is exactly the theta-hat
   field a z-wire sees, so no polarization information is lost.
2. **Direct term.** `D` (BS to user) is RIS-free; it is the ray-traced part of the baseline channel.
   The NEC structural scattering of the open-circuited wires is **not** in it and must be added or
   explicitly neglected (derivation, section 8).
3. **Outgoing side.** The 14 to 34 paths per probe-to-user link give the field at the user per unit
   source at the probe. Because the probes sit on a different lattice from the NEC wires, three options
   exist, in order of preference: (a) trace again with probes at the NEC element centers (cheap relative
   to the five-million-ray runs, GPU needed); (b) re-reference LOS paths geometrically (exact:
   rescale 1/r and rotate the phase by the exact distance difference) and use per-path
   plane-wave steering for reflections with the cross-validation error reported; (c) interpolate the
   channel over the aperture with a plane-wave least-squares fit (the 20 mm and 13 mm pitches are
   below lambda/2 = 27 mm, so the lattice is above the Nyquist spacing in both directions;
   leave-out error would have to be measured, **not done**).
   For this stage only option (b) for LOS is used (gate P4).
4. **What cannot be reused.** The frozen test data; the `H` sums for a different lattice without one
   of the above conversions; `G` and `H` as if they were port excitations (they are isotropic-probe
   transfer coefficients, not wire port voltages: derivation sections 3 to 5).

## 6. Compute and environment audit

* Local machine: 12 threads, 15 GiB RAM with about 1.3 GiB available and swap full; /home 97 % full
  (8.3 GB free). NEC-2 solves above 21 to 41 segments per wire for 64 wires are not run locally.
* `compute status`: 3060-1 up (not used, contract), 3060-2 unknown (never used), 3060-3 unknown
  (`lab-vpn` down, not used), rx6600 up via Tailscale. rx6600 read-only check: 16 CPUs, 60 GiB RAM
  (about 42 GiB available), `/home` 96 % full with 14 GB free, load average about 4, Python 3.12.3 and
  `uv` present, no project environment. The existing production NEC-2 records give about 290 s for the
  161-segment 70-port extraction (11,270 unknowns), so the RIS-only 64-wire extractions at 81, 121 and
  161 segments are single-core jobs of minutes needing 0.4 to 1.7 GB.
* PyNEC 2.3.4 (`.venv-em`) reports lengths and positions in wavelengths in
  `get_current_segment_*`; the near-field card returns the field of the structure currents **only**
  (the elementary-source field is not included) [V-num]; EX card type 4 (elementary current source) and
  type 1 (plane wave, angles are the direction *from which* the wave arrives) are available [V-num].
  These were measured, not read from documentation, and are re-tested in the unit tests.

## 7. Unverified items (consolidated)

1. License of the built-in Sionna scene assets and of any OSM-derived meshes; whether derived path
   data are redistributable.
2. DeepMIMO scenario-database licenses; hosted-scenario link contents.
3. Which DeepMIMO `MAX_PATHS` default applies to a given call (10 in docs, 25 in the 4.0.5 converter).
4. The isotropic V pattern convention in the Sionna docs (verified only empirically for LOS and
   from the saved parity files; reflected-path polarization bookkeeping is Sionna's and is taken as given).
5. The legal consequence of the DeepMIMO 4.0.5 Apache-2.0 versus GPL-2.0-or-later mismatch.
6. NEC-2/PyNEC solver licensing (stated in `THIRD_PARTY.md`, not re-audited here).
