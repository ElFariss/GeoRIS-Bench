# CoupledRIS-Bench production pilot — preregistered 2026-11-09 WIB

This is a development-only pilot and a proposed size, not authorization to run
the full build. No held-out geography or old test data is used. Verify
the approved v1.1 freeze before every stage. Keep the preview labels intact.

**Proposed size for lead approval:** 120 natural trajectories per development
map, 16 approximately equally spaced original 20 ms samples per trajectory,
4 BS ports, the frozen scheduler's 2–4 served users and arrays 64/128/256.
Munich and Rotterdam are the initial two-map scope (240 cases, 3,840 frames,
11,520 array-frames); Maastricht is a separate addition only after lead
acceptance (360 total cases, 5,760 frames, 17,280 array-frames). Trace these
same selected frames for all arrays. For labels, use four selected frames per
trajectory and two parents per selected frame (all-OFF, bounded single
greedy). Enumerate all 2,016 / 8,128 pairs at 64/128. At 256 use 4,096
deterministic uniform sampled pairs per ordinary parent; enumerate all 32,640
for one parent on every tenth trajectory within each map (fixed index,
first labeled frame). Label gains in bit/s/Hz and pair synergy against the
same parent. This projected size is conditional on pilot storage/time and
not a release decision.

**Pilot:** approved preview cases 01-munich-natural and 15-rotterdam-natural,
frame 0 only; include all served users, all three sizes. Pilot label all six
array-frames, use two parents, and include a full-pair audit at 256 on one
parent per case. Use Sionna RT 2.0.1, 5.53 GHz, both V/H, explicit arrays,
max interaction depth 2, 10,000 samples and a 40,000 path capacity per
source, fixed seed 7, no refraction/transmission, no diffuse reflection;
LOS, specular and wedge diffraction on. The pinned solver defaults keep
edge diffraction off and enable the diffraction lit region. Raw per-path
files and native meshes remain on the approved lab host.
Treat RIS→UE as the reciprocal reverse of the same UE→RIS traced paths
in this passive medium. Use the native map origin and 4 vertical BS wires
spaced half a wavelength horizontally, the panel normal oriented toward BS
from the registered centre, and the declared 5 cm facade clearance used in
the preceding smoke. The production geometry label comparison records that
offset and the different BS array. No Maastricht trace is authorized here.

**Electrical convention:** 50 Ω external ports; archived ideal PEC wires,
5.53 GHz, 161-segment matrices for 64/128 and 81 for 256; per-frame
all-OFF-reference RZF fixed across bit evaluations. Temperature 290 K,
1 MHz, 7 dB noise figure. Use the fixed load model from the free-space
coupled benchmark; this is not a measured physical switch model. Apply the
absolute polarized P4 port map to each ray. Use exact RIS-only impressed
segment currents for the pilot open-gap term; terminate the full external
block and store the resulting Schur complement with each frame. No
uncalibrated legacy centre-only gain, no omitted open gap.

**Gate:** (1) the complete packaged schema must cover each declared city,
array, case and frame, with no NaNs or infinities and valid link/path
provenance; (2) the loader and independent dense recomputation reproduce a
stored bit-state rate to absolute 1e-10 bit/s/Hz; (3) each map's BS–RIS has
at least one traced LOS among its BS/element combinations, and binary LOS
agreement against geometry labels across served BS→UE and RIS→UE groups is
at least 75% overall (aggregate class is LOS if any corresponding explicit
link has LOS); reflection-only, diffraction-only, mixed and fully blocked
are reported separately without treating geometry candidates as verified
paths; (4) report measured elapsed time, peak GPU memory and stored bytes
per case and array, with geometry and reusable NEC setup separated from
marginal per-frame costs; (5) project the registered size from measured
stage costs, with an explicit storage headroom and resource decision.
No full run is launched even if all checks pass.

Adjacent 4-port BS antennas are below the far-field port-map regime. Retain
their exact free-space NEC mutual terms for traced direct links under 0.3 m,
adding traced reflected/diffracted terms separately. No external self-ray is
used. Use a reusable RIS MoM factorization per array; verify its driven port
admittance against the archived impedance before evaluating a frame.

## Pilot 2 amendment — 2026-11-09 WIB, before rerun

The lead accepted the corrected Maastricht scene and the user authorized
one three-map pilot. Add approved v1.1 case `39-maastricht-natural`, site
`maastricht-v1-site-07`, frame 0, served users 10, 12 and 55. Its original
frame Parquet SHA-256 is
`5b4886673d7ed2b33544ee6185f5643b9503aa5422637a36c2959024f39a66fe`.
The original Munich and Rotterdam case records are identical to pilot 1.
The new three-case manifest, at
`outputs/runs/bench_production_pilot2_20261109/pilot_cases.json`, has SHA-256
`5b129264c6f81d55bffd6d2edfe3db4a58b8938e3484564506779e6228fc5827`.
All three registered arrays and both label parents apply to the new case:
nine array-frames, no new placement generation, no test maps.

The [mesh policy](bench-production-mesh-policy-20261109.md) applies equally
to all three maps. It is a hard dependency for ray tracing. All previous
pilot artifacts and the original failure are retained. The solver, port
locations, wire meshes, electrical conventions, pair coverage, loader
tolerance, and the 75% aggregate LOS threshold are unchanged. The per-map
BS–RIS LOS check now covers all three registered maps; the previous smoke
is not grounds for waiving that check. Geometry failures stop before any
GPU trace; if geometry passes but link sanity fails, finish the bounded
chain and record the failed gate.

Project 120 trajectories per map, 16 sampled frames per trajectory, three
arrays (360 trajectories, 5,760 map-frames, 17,280 array-frames). Keep four
labeled frames and two parents; 256-element all-pair audits follow the
original production schedule. Report observed per-city timing, sampled GPU
peaks, raw and derived sizes and label cost, separating one-time NEC setup
and geometry build. The production label projection may conservatively charge
the denser pilot audit rate; no scale run is authorized.
