# P4 calibrated continuation — declared 2026-11-08 23:17 WIB

Use the 64 RIS + BS0 + UE0 centres from the registered Munich development
reference at 5.53 GHz, 81 segments, 24 mm vertical PEC wires. Keep the
previous 1% complex, 0.01 rad phase, [0.99, 1.01] co-polar ratio, and 1%
cross-polar limits. No fit, per-link scale or phase alignment. One
bounded implementation fix per failure; preserve measured failures.

The pinned Sionna RT 2.0.1 run will use CPU LLVM on 3060-3, excluding
logical CPUs 5 and 11, with CUDA hidden. Only P4-specific directories are
used. Trace both external directions to all 64 NEC centres and to the other
external antenna, with one colocated V/H pair at each endpoint. Use the
original built-in Munich LOS, and first-order Munich reflections as an
unscored diagnostic if there is no independent urban reflection reference.
Separately trace a controlled z=0 plane using the same port centres and a
PEC-limit material. A finite plate is equivalent here only for first-order
specular rays with reflection points on the plate, with edges, diffuse
scattering and transmission disabled. Score it against a separate NEC
perfect-ground solve where available; it cannot certify a Munich building
reflection.

Use Sionna's passband path coefficient (before carrier delay phase) and
global departure/reverse-arrival spherical basis. Its isotropic coefficient
includes wavelength over 4*pi times path spreading. For a unit NEC port
current, replace the isotropic transmit pattern with the segment-current
moment projected in the departure basis. Divide by Sionna's wavelength,
and multiply by -j*k*eta to recover a Cartesian electric field in V/m.
Use unmodified physical port coordinates. Apply exp(-j*k*R) with the
source-derived NEC wave number for the NEC comparison and record the physical
SI case separately. R is computed from registered coordinates for a
verified LOS or plane image path; it must agree with Sionna's traced delay to
within 0.2 mm. This is a source-derived difference between numerical medium
conventions, not a phase fit. Use the verified path's wavefront direction
across each receiving wire, then its isolated or embedded reciprocal NEC
current basis to compute open-circuit voltage. RIS-only impressed-segment
currents produce an independent open-gap correction. Targets do not enter
the prediction. Score independent incident NEC vector fields and ports
when available. If the matched urban reflection reference is missing or a
required metric fails, do not label the full gate as passed.

## Executed registered LOS

The trace ran in tmux `p4-registered-20261108` in
`/var/tmp/atomwise-p4-registered-20261108` on 3060-3, CPU affinity
`0,6`, CUDA hidden. It wrote DONE at 23:20:05 WIB. The checked compact
Jones/absolute-delay result had SHA-256
`3e6c1a4803aea37abb1bb020ab2a952b9419617f21188fb9bafc4a73c9dc37c6`.
The original urban mode reports only path counts; no raw urban path file or
mesh was copied. The registered LOS and controlled z=0 plane each returned
one ray for all 130 ordered external-source / receiver pairs. Munich's first
order inventory had three paths per BS-to-RIS pair, five per UE-to-RIS pair,
and two per external-to-external pair (LOS included).

The local tmux scorer wrote DONE and a checked
`outputs/runs/p4_calibrated_20261108/registered_comparison.json`, SHA-256
`b3cc7a3ff762f7e98c14749d76dfbc42be3de7139d66591158463a95794b7d6b`.
Field references use independent PyNEC NE-card Cartesian E, divided by the
solved source feed current. The monolithic NEC RIS/external block and
open-structure direct channel are scoring targets only. Each ray-driven
open-gap prediction uses only independently solved RIS current matrices
and the traced incident fields. The external block's off-diagonal entries
come from the reciprocal ray-derived open-circuit voltage; its self
impedances come from two isolated NEC antennas.

| Registered Munich LOS measurement | Complex relative error | Max wrapped phase |
|---|---:|---:|
| 130 per-port incident field projections, worst port | 0.0360% | 0.000338 rad |
| 64 × 2 RIS port voltage block | 0.0427% | 0.000386 rad |
| Uncorrected ray direct vs free NEC | 0.0176% | 8.01e-6 rad |
| Ray direct + independent open-gap correction vs monolithic NEC | 0.0176% | 8.04e-6 rad |
| Predicted open-gap impedance matrix vs monolithic difference | 0.1246% | 0.005570 rad |

Worst cross-polar leakage was 0.00372%; individual field amplitude ratios
were 1.0001154–1.0001240. All registered LOS limits passed. Exact-SI
propagation without the NEC numerical-medium convention is recorded
separately; the direct mutual phase difference is about 0.3335 rad, as
predicted from source constants. No phase or scale was fitted to a target.
The controlled reflection is a separate test; the original full gate
requires an independent matching urban reflection reference.

### Reproduction and checks

- Trace entry point: `scripts/p4_trace_registered_ports.py --geometry geometry.json --out trace_compact.json`.
  Use Sionna RT 2.0.1, LLVM CPU only, `taskset -c 0,6`. The staged geometry is a
  whitelist export of only the registered reference's three coordinate arrays.
- The voltage/field converter is `src/atomwise_ris/hybrid/p4_rt_bridge.py`.
  `scripts/p4_score_registered_ports.py` scores LOS and writes per-port fields,
  voltages and impedances. The local command and hashes are in
  `outputs/runs/p4_calibrated_20261108/run.log` and
  `.cache/p4-calibrated-stage/score.sh`.
- The full verdict aggregator is `scripts/p4_gate_registered_suite.py`.
  It deliberately cannot substitute the controlled plane for the registered
  urban reflection.
- Remote receipt check:
  `ssh -o BatchMode=yes -o StrictHostKeyChecking=yes <3060-3> 'cd /var/tmp/atomwise-p4-registered-20261108 && cat DONE && sha256sum -c trace.sha256'`.
  Do not rerun either completed job.
