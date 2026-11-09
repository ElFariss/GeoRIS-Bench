# GeoRIS-Bench

A geospatially grounded electromagnetic benchmark for RIS-aided wireless communications.

This repository holds the **code** that builds the benchmark; the data itself is released separately. The pipeline combines:
- open LoD2 3D city models and LiDAR ground models;
- ray tracing with Sionna RT;
- an NEC-2 model of mutual coupling between RIS elements.

> **Status: work in progress.** This snapshot covers the stages that are validated so far: map preparation, ground fill, the user and site generator (v1.1, frozen), RIS impedance extraction, the ray-to-wire calibration (gate P4) and the ray-tracing scene builder. The production stage (channels, optimization labels, splits and loader) will be added when it passes its pilot.

## Pipeline

| Stage | Entry point | What it does |
|---|---|---|
| Map audit | `scripts/fetch_bench_map_audit.py`, `scripts/audit_bench_area_coverage.py` | Downloads the LoD2, terrain and OSM sources and pins their hashes. Checks coverage of each map area. |
| Geometry | `scripts/prepare_bench_users_geometry.py` | Builds buildings, terrain, streets and slope per map. |
| Ground fill | `src/atomwise_ris/bench/ground_fill.py` | Fills a separate, flagged ground layer under building footprints, used only as the base height of building-mounted BS and RIS sites and of rooftop users. |
| Users and sites | `scripts/generate_bench_users_preview.py` (`src/atomwise_ris/bench/users.py`) | Places BS and RIS sites, clustered heavy-tailed crowds, height classes and walking trajectories on terrain. Seeded, with rejection counts. |
| Validation | `scripts/validate_bench_users_preview.py` | Recomputes ground, slope and link labels independently and checks the declared rules. |
| Figures | `scripts/render_bench_users_preview.py` | Renders case maps, cards and distribution plots. |
| 3D view | `scripts/export_case_paraview.py` | Exports a case to ParaView (terrain, buildings, sites, animated users). |
| RIS impedance | `scripts/run_bench_zris_extract.py`, `scripts/compare_bench_zris_mesh.py`, `scripts/validate_bench_zris_bundle.py` | Computes the NEC-2 RIS-only impedance matrix `Z_RIS` for 64, 128 and 256 elements, then checks mesh convergence, reciprocity and passivity. |
| Ray→wire calibration | `scripts/p4_*.py`, `scripts/evaluate_hybrid_p4.py`, `src/atomwise_ris/hybrid/` | Converts ray-traced fields to NEC-2 port voltages without any fitted scale or phase. Validated against independent NEC references for line-of-sight and a controlled reflection. |
| RT scene builder | `scripts/build_bench_phase0b_scene.py`, `scripts/run_bench_rt_preview_smoke.py` | Builds LoD2 scenes for Sionna RT, with per-element probes at the NEC lattice. |

`scripts/run_bench_users_preview.sh` runs geometry → users → validation → figures in order.

## Channel model

The channel is `H = H_direct + C_out (Z_RIS + Z_L(b))^-1 C_in`, where:
- `Z_RIS` comes from NEC-2 and models element coupling;
- `C_in`, `C_out` and `H_direct` come from ray tracing through the validated conversion;
- `Z_L(b)` is the load set by the binary RIS configuration `b`.

## Install and test

```bash
uv venv && uv pip install -e ".[dataset,gis]"
PYTHONPATH=src:scripts python -m unittest discover -s tests
```

Ray tracing needs the `simulation` extra (Sionna RT 2.0.1, GPU recommended). NEC-2 extraction needs PyNEC.

## Documentation

- `docs/methodology/`: dated design and protocol records. Criteria are declared before each run.
- `docs/dataset-card/`: the dataset README, including the ground model and gap filling, and the source license notes.

## Sources and licensing

- Map data: Bavarian LoD2 and DGM1, 3DBAG and AHN, and OpenStreetMap (ODbL). See `docs/dataset-card/LICENSES/`.
- The license for this code has not yet been chosen.
