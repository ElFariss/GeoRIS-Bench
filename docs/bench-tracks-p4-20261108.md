# Track P4 calibration check (declared 2026-11-08, WIB)

## Fixed gate and comparison cases

This is a development-only P4 check. It uses the already frozen engineering limits from `docs/hybrid-track2-20261108.md`: complex relative error <= 1%, wrapped phase error <= 0.01 rad, co-polar amplitude ratio in [0.99, 1.01], and cross-polar leakage <= 1%. No scale, phase, antenna-gain, or per-frame fit is allowed. A convention bridge may be applied only if independently specified from the source definitions before inspecting comparison results.

Primary geometry is the registered Munich direct LOS link in `outputs/runs/hybrid_track2_20261108/munich_los_nec_reference.npz`: 5.53 GHz; one BS port, the 64-element 8x8 NEC lattice, and the previously selected training UE. Port order, reference plane, field basis, and amplitude normalization must match exactly. The independent monolithic NEC open reference is the only allowed comparison target; its `delta_z_open_structure` is reference truth and is never an input to the predicted correction.

The declared reflection case is the same BS/UE and RIS geometry with first-order specular reflection enabled (maximum interaction depth 1, LOS enabled, diffuse reflection and diffraction disabled, transmission disabled). The reflection case is reported as a separate physical case. It is gate-scored only if a separate independent NEC reference for that same reflected geometry and port convention is available; otherwise it is explicitly diagnostic and P4 remains incomplete for that case.

For each case the intended comparison is Sionna's per-NEC-centre, polarization-resolved incident response against independent NEC port/field response. The corrected direct channel is `H_RT,direct + D(Z_EE,free + dZ_pred) - D(Z_EE,free)`, with `dZ_pred` produced by the fixed RIS-alone builder from its own inputs. Both uncorrected and corrected values are compared to the independent NEC direct/open reference. The correction must be independently predicted; the monolithic NEC open-minus-free `delta_z_open_structure` cannot be reused as a prediction.

## Stop rules

One bounded measurement/check round is allowed. Missing absolute field-to-port conversion, external NEC incident basis, predicted open-gap term, or a same-case independent NEC reference blocks scoring. Such missing inputs are not replaced by arbitrary scale factors, fitted phase, or legacy truth. If an available comparison fails the declared tolerances, preserve the measured error and stop after one documented diagnosis/fix attempt; do not alter limits. No held-out data, optimization, or training is used.

## Existing evidence before this round

The prior v3 evaluator correctly failed closed because the available Munich artifact has independent NEC open/free truth but lacks the RT direct channel, incident response on both sides, absolute mapping, polarization-resolved NEC/RT fields, and an independently predicted open-gap `dZ`. The Phase 0 dual-polarization probe demonstrates the Sionna array output's two polarization channels in free space; it does not supply an electric-field-to-NEC-port conversion or a Munich comparison. These are recorded as prerequisites, not assumed to be satisfied.

## Bounded P4 round outcome

The fresh preflight found that the permitted 3060-3 node had GPU1 available (11,909 MiB free, 0% utilization) but did not have Sionna RT in system Python or any of the checked existing research environments. GPU0 was already at 87% utilization, so it was not considered. No remote job or install was launched. The available Phase 0 capability record (`explicit_dual_pol_free_space`) is a free-space dual-channel smoke, not an absolute Munich calibration.

The evaluator was rerun against a copy of the preserved v3 truth-only input. It returned `blocked / P4_NOT_EVALUATED`, rejecting the monolithic truth-derived `delta_z_open_gap` and listing the missing absolute mapping, port order/basis, RT and NEC incident fields, RT direct channel, and independent predicted open-gap term. No numerical calibration error can be reported because there are no valid model/reference arrays to compare; recording a fabricated value would be misleading. The declared reflected case also lacks an independent same-geometry NEC reference. The limits remain unchanged and P4 is open.

- Gate preflight: `outputs/runs/bench_tracks_p4_20261108/preflight.json`, SHA-256 `28899fb91897996239023af71972e34a0ecf78c504569a3e556e53b2e00286b0`.
- Re-evaluated fail-closed output: `outputs/runs/bench_tracks_p4_20261108/gate_preflight.json`, SHA-256 `112c991509f6994f992f7310c9b50cfeb5d5bc9cb249d28940b78ddf0ee9911d`.
- Exact copied legacy input: `outputs/runs/bench_tracks_p4_20261108/legacy_truth_only_input.json`, SHA-256 `6a4ad664c9cce5fbb12c3297e218e66c7f7fa01d1498bf37554a5679271d3102`.
- Evaluator source SHA-256 `c03c509234a222bf2059af4058fb5dc0c240c3ebc35d18f6e08babef75427826`; independent NEC reference SHA-256 `fe5b23c71c55caee8dba1beb583b3698b2347bdfd1a04a78a5dafb5b8a7dbfc7`; Phase 0 capability record SHA-256 `eabe133daa0aa90cdda1b81e281b5e2732f788f370353c290b53767e32c39e1a`.
- Reproduction: `PYTHONPATH=src:scripts python scripts/evaluate_hybrid_p4.py --input outputs/runs/bench_tracks_p4_20261108/legacy_truth_only_input.json --output outputs/runs/bench_tracks_p4_20261108/gate_preflight.json`.

Next P4 attempt requires the pinned Sionna RT 2.0.1 runtime on an allowed 3060-3 GPU, polarization-resolved fields at all 64 NEC centers, an independently derived absolute RT-to-port convention, a fixed-builder predicted open-gap term with source hashes, and a matching NEC reference for any reflection case claimed as scored.

## Bounded no-fit analytic subcheck (19:10 WIB)

I ran the saved free-space dipole subcheck against the independent BS+UE-alone
NEC reference. Before execution, I inspected its source and corrected its
Green-function source argument: `zdip_E` applies its `p` argument as an
omega-scaled electric dipole moment, so the prior `I*dl/(j*omega)` argument
under-scaled the predicted field by `omega`. The corrected equivalent current
moment is `I*dl/j`; it introduces no scale or phase adjustment. This repair is
limited to `scripts/p4_free_space_dipole_subcheck.py`; no shared physics
function or frozen source was changed.

For the 24 mm z-directed BS/UE wires at 5.53 GHz and the registered 246 m
near-grazing link, the sinusoidal-current approximation predicts
`Z_mut = 0.00207445 + j0.00058191 ohm`; NEC gives
`-0.00114930 + j0.00209639 ohm`. Under the already declared port convention,
the predicted direct channel has complex relative error 148.98%, phase error
1.7988 rad and amplitude ratio 0.9012. All three scalar thresholds fail. This
is a failed idealized dipole-current approximation, **not** a P4 gate score:
it has no RIS, no Sionna comparison, no independently predicted open-gap
term, and no reference vector fields for cross-polar scoring. No tolerance was
changed and no fit was applied.

The diagnostic makes the absolute-conversion risk concrete: the assumed
sinusoidal current profile and far-field-style free-space model do not
reproduce this particular NEC link. It does not identify which convention or
wire-current assumption explains the phase difference, so it must not be used
to calibrate the hybrid operator. Full P4 remains blocked pending the listed
independent Sionna fields and fixed-builder open-gap inputs; the reflection
case remains unscored without a matching independent NEC reference.

Evidence: `outputs/runs/bench_tracks_p4_20261108/free_space_dipole_subcheck.json`
(SHA-256 `9c2374fa316d803d8fbbd1deb6a656db675bc87c5b1b975a5fbd0adf3ef50f6d`);
NEC reference NPZ SHA-256
`fe5b23c71c55caee8dba1beb583b3698b2347bdfd1a04a78a5dafb5b8a7dbfc7`.
Reproduce with:

```bash
PYTHONPATH=src:scripts python3 scripts/p4_free_space_dipole_subcheck.py \
  --out outputs/runs/bench_tracks_p4_20261108/free_space_dipole_subcheck.json
```
