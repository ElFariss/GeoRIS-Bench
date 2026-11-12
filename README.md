# GeoRIS-Bench

A geospatially grounded electromagnetic benchmark for RIS-aided wireless communications. Its working name in the code and records is **CoupledRIS-Bench**.

This repository holds the **code** that builds the benchmark, the loader, and the shared physics and reference-solver code. The data itself is released separately. The pipeline combines:
- open LoD2 3D city models and LiDAR ground models;
- ray tracing with Sionna RT;
- an NEC-2 model of mutual coupling between RIS elements.

> **Status: work in progress.**
> - **Production has run** on three development maps (Munich, Rotterdam, Maastricht):
>   - natural scope `development_full`: 359 trajectories with channels at 64, 128 and 256 elements;
>   - stress-test scope `development_challenge`: 69 trajectories;
>   - Berlin and Utrecht stay sealed as held-out maps.
> - **Open issue (2026-11-12): in the natural scope the RIS barely changes the worst-user rate.**
>   - About half of the validation frames have a served user with no traced path.
>   - The small RIS sits about 112 m from the base station, so its path arrives 40–60 dB below the base station's own paths.
>   - A RIS-relevant scope is being considered. The diagnostic is in `../paper-b-neural-pair-search/docs/bench-ris-leverage-diagnostic-20261112.md`.
> - The production stage and most of `docs/` were added on 2026-11-12. Their lab-specific defaults are now overridable (see *Running on your own machine*).

## Pipeline

| Stage | Entry point | What it does |
|---|---|---|
| Map audit | `scripts/fetch_bench_map_audit.py`, `scripts/audit_bench_area_coverage.py` | Downloads the LoD2, terrain and OSM sources and pins their hashes. Checks coverage of each map area. |
| Geometry | `scripts/prepare_bench_users_geometry.py` | Builds buildings, terrain, streets and slope per map. |
| Ground fill | `src/atomwise_ris/bench/ground_fill.py` | Fills a separate, flagged ground layer under building footprints. It is used only as the base height of building-mounted base-station and RIS sites and of rooftop users. |
| Users and sites | `scripts/generate_bench_users_preview.py` (`src/atomwise_ris/bench/users.py`) | Places base-station and RIS sites, clustered heavy-tailed crowds, height classes and walking trajectories on terrain. Seeded, with rejection counts. Frozen at v1.1 (`configs/bench/users-v1.1-freeze-20261109.json`, checked by `bench/production_freeze.py`). |
| Validation | `scripts/validate_bench_users_preview.py` | Recomputes ground, slope and link labels independently and checks the declared rules. |
| Figures and 3D view | `scripts/render_bench_users_preview.py`, `scripts/export_case_paraview.py` | Case maps, cards and distribution plots; a ParaView export. |
| RIS impedance | `scripts/run_bench_zris_extract.py`, `scripts/compare_bench_zris_mesh.py`, `scripts/validate_bench_zris_bundle.py` | Computes the NEC-2 RIS-only impedance matrix `Z_RIS` for 64, 128 and 256 elements, then checks mesh convergence, reciprocity and passivity. |
| Ray→wire calibration | `scripts/p4_*.py`, `scripts/evaluate_hybrid_p4.py`, `src/atomwise_ris/hybrid/` | Converts ray-traced fields to NEC-2 port voltages without any fitted scale or phase (gate P4). |
| RT scene builder | `scripts/build_bench_phase0b_scene.py`, `scripts/run_bench_rt_preview_smoke.py` | Builds LoD2 scenes for Sionna RT, with per-element probes at the NEC lattice. |
| Production cases | `scripts/build_bench_production_cases.py` (`src/atomwise_ris/bench/production_pilot.py`) | Registers the natural and challenge case manifests by hash. |
| Production trace | `scripts/run_bench_production_trace.py` | Ray-traces every frame with probes at the element centres (`max_depth` 2, diffraction on). |
| Production channels | `scripts/build_bench_production_channels.py` (`src/atomwise_ris/bench/production_channel.py`) | Builds `H_direct`, `C_in`, `C_out` and `Z_effective` per frame. Per-case resume. |
| Package and gate | `scripts/package_bench_production.py`, `scripts/verify_bench_production_gate.py` | Writes HDF5, Parquet and JSON with flip-gain labels, splits and provenance, then checks the gate. |

Shell wrappers for the lab runs are `scripts/run_bench_*.sh`. `scripts/run_bench_users_preview.sh` runs geometry → users → validation → figures in order.

## Channel model

The channel is `H = H_direct + C_out (Z_RIS + Z_L(b))^-1 C_in`, where:
- `Z_RIS` comes from NEC-2 and models element coupling;
- `C_in`, `C_out` and `H_direct` come from ray tracing through the validated conversion;
- `Z_L(b)` is the load set by the binary RIS configuration `b`.

The rate is the minimum user rate under a fixed regularized zero-forcing precoder (`channels/precoding.py`), evaluated by `hybrid/operator.py` (`HybridCoupledNetwork`) with exact rank-1 and rank-2 updates.

## Python package

The code lives in the `atomwise_ris` **namespace package**: there is no top-level `src/atomwise_ris/__init__.py`. The two research repositories add their own modules to the same package.

| Module | What it holds |
|---|---|
| `bench/` | Users, ground fill, production cases, channels, freeze check. |
| `hybrid/` | Channel operator, embedded-element model, ray coupling, P1–P4 gate machinery. |
| `physics/` | NEC-2 interface and the free-space coupled network. |
| `channels/` | Reference precoder. |
| `optimization/`, `evaluation/` | Reference solvers for the benchmark's search task: greedy, pair, ME-GA, pair-plus b3/b5/b35, M0 and the pair-selector inference. Their default limits come from the frozen `configs/experiments/matched_v1.json`. |

**Used by:**
- `../paper-a-ris-physics`: the physics study.
- `../paper-b-neural-pair-search`: neural pair search and the search benchmark.

## Install and test

```bash
uv venv && uv pip install -e ".[dataset,gis]"
PYTHONPATH=src:scripts python -m unittest discover -s tests
```

- **Optional extras:** ray tracing needs the `simulation` extra (Sionna RT 2.0.1, GPU recommended). NEC-2 extraction needs PyNEC.
- **Tests that need data:** some tests read packaged artifacts under `outputs/` and `bench/` and skip when those are absent.
- **Legacy gate audits** read the free-space wire-bench runs (`outputs/runs/coupled_wire_v1`, `coupling_halfwave64_v1`) from Paper A. In the workspace they are relative symlinks, gitignored with the rest of `outputs/`.

## Running on your own machine

The original runs used two lab GPU nodes (3060-1 and 3060-3) and a large-memory CPU node (rx6600). The shell wrappers in `scripts/run_bench_*.sh` default to portable settings:

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON` | `python3` | The interpreter |
| `BENCH_REQUIRE_HOST` | unset (no check) | Optional host guard. The lab runs required the production node. |
| `BENCH_HIGHMEM_LOCK` | `/var/tmp/atomwise-bench-round2-highmem.lock` | Shared lock that keeps the high-memory jobs apart |

The `taskset` CPU lists keep the lab's timing core (CPUs 5 and 11) free. Change them for your CPU layout.

**Provenance-locked files** are byte-identical to what produced the released data, because their SHA-256 hashes are recorded in its provenance and in the v1.1 freeze lock. They are:
- the v1.1 user generator;
- `hybrid/operator.py`;
- `scripts/package_bench_production.py`;
- the phase-0 audit scripts and wrappers;
- `bench/CoupledRIS-Bench/physics/`.

Three of the wrappers, `run_bench_phase0_rt.sh`, `run_bench_phase0b_scene.sh` and `run_bench_munich_smoke.sh`, still name the lab interpreter path. Edit a copy to rerun them; don't edit them in place.

The dated records in `docs/` keep the node names and lab `/var/tmp` workspaces of each run, with access details removed.

## Reproducibility

Every stage is seeded and records the hashes of its inputs, so rerunning the pipeline reproduces the same cases: the same sites, users, groups, trajectories and labels.

Regeneration is exact **up to floating-point rounding**, not bit-for-bit across machines:
- Different CPUs and math libraries can round the last binary digit differently. Continuous columns such as angles may differ by about 1e-14 between machines, far below any physical effect.
- Different `pyarrow` versions can lay out Parquet files differently, even for identical data.

So a regenerated file can have a different SHA-256 than the released one while holding the same data. **The released dataset files are the reference:**
- Their SHA-256 hashes are listed in the dataset's `provenance.json`.
- The production pipeline only ever reads cases from a registered, hash-checked manifest; it never regenerates them.
- To compare a regenerated case with the release, compare the values with a tolerance (for example `numpy.allclose` with `atol=1e-9`), not the file hashes.

## Documentation

- `docs/`: dated design and protocol records. Criteria are declared before each run.
  - Two of them are hash-locked by the v1.1 freeze: `bench-users-v1-design-20261108.md` and `bench-tracks-ground-v1.1-20261108.md`. Do not move or edit them.
- `docs/dataset-card/`: the dataset README, including the ground model and gap filling, and the source license notes.
- `bench/CoupledRIS-Bench/`: the release folder (per-array physics metadata, loader example, provenance, licenses).

## Sources and licensing

- Map data: Bavarian LoD2 and DGM1, 3DBAG and AHN, and OpenStreetMap (ODbL). See `docs/dataset-card/LICENSES/`.
- The license for this code has not yet been chosen.
