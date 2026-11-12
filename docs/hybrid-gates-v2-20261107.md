# Hybrid physics gates P1 and P2 at full size: declaration, 2026-11-07

Status of this section: **declared before any 64-wire run**. Tolerances and pass criteria below are fixed and are not to be loosened after seeing results. If a criterion fails, the exact numbers and a diagnosis are appended to the results section, and the gate stays failed. This document supersedes the P1/P2 sections of `hybrid-gates.md` (historical proposal) and responds to the findings in `hybrid-audit-20261106.md`: the legacy P1/P2 artifacts were self-comparisons. No hybrid optimizer or training is authorized by anything here, even if both gates pass.

Only the free-space benchmark geometry is used (`physics/nec.geometry()`): 64 z-directed 24 mm wires (8x8, 28 mm x 30 mm pitch, panel in the yz plane at x=0), 4 BS wires (x=-0.5 m), 2 UE wires (default positions), radius 50 um, 5.53 GHz, NEC-2 extended thin-wire kernel, loads OFF = 1-10.8j ohm, ON = 1+129.2j ohm, 50 ohm real references on all BS/UE ports. 70 ports in total. No development, validation, pilot or test dataset is read or scored. Mesh: **81 segments per wire is the primary mesh**; 161 is run if affordable as a corroboration. 241 is excluded (below the NEC-2 extended-kernel ratio, see the audit).

## Independence of the comparisons

* **Reference (truth) side**: PyNEC solves in which the physical sources and terminations are NEC cards (`EX` voltage sources, `LD` type 4 loads, `EX` type 1 plane waves) applied to the structure. A 50 ohm termination is an `LD` card; the 1 V Thevenin source is an `EX` card in series with that `LD`.
* **Model side**: the 70x70 port impedance matrix Z (assembled from 70 unit-voltage NEC solves, then inverted from the feed-segment admittance), the Schur reduction in `CoupledNetwork` and `HybridCoupledNetwork.from_monolithic_z`, and Woodbury updates; for P1-B also `EmbeddedRISResponse` (segment currents, reciprocity integrals).
* The NEC loaded solve builds and factors a different linear system from the one used for Z. A mismatch therefore reveals errors in port indexing, source or load conventions, series-source handling, the Schur reduction or the Woodbury update. Both sides use the same NEC kernel, so P1 does not test the physics of the NEC kernel itself (the mesh/kernel question is P3).

## Gate P1: port consistency at full size

State set S1 (10 states, 64 bits each, fixed in advance): all-OFF; all-ON; alternating (`i mod 2`); R1..R5 = `numpy.random.default_rng(seed).integers(0,2,64)` with seeds 20261107..20261111; F1 = R1 with bit 17 flipped; F2 = R1 with bits 5 and 42 flipped. F1 and F2 are produced through `CoupledNetwork.flip` (Woodbury), and the direct NEC solve is then run on the resulting bit vector.

Excitation set per state: voltage source at BS ports 0,1,2,3; at UE port 0; at RIS ports 0, 27, 63 (8 solves; all other ports terminated by their load or 50 ohm).

| ID | Criterion | Tolerance | Justification |
|---|---|---|---|
| P1.0a | Z reciprocity `||Z-Z^T||_F/||Z||_F` (70x70) | <= 1e-6 | Carried over from the earlier gate text: NEC-2 double-precision residuals are 1e-8 to 1e-7; 1e-6 still catches indexing or asymmetry errors. |
| P1.0b | `lambda_min((Z+Z^H)/2)` | > 0 ohm | Passive radiating array. |
| P1.0c | `sigma_max(S)`, 50 ohm | <= 1.0001 | Passivity of the S-matrix. |
| P1.0d | `cond(Z)` | < 1e5 | Bounds the error amplification of the solves below. |
| P1.1 | Relative error of the full 70-port current vector, direct NEC loaded solve vs `solve(Z + diag(T), e_p)`, every state and every excitation (80 solves) | **max <= 1e-6** | Algebraically the two systems are identical; the residual is round-off in the factorizations. The round-off is bounded by about `cond x eps`; with `cond(Z) < 1e5` (P1.0d) and eps = 2.2e-16 that is below 3e-11 even before allowing for factorization growth, and the observed NEC reciprocity residual is 1e-8 to 1e-7. 1e-6 therefore leaves a margin of at least 10 over the worst plausible round-off while still catching any indexing, sign or termination error (which would show up at 1e-2 or larger). |
| P1.2 | Reduced 2x4 channel: `CoupledNetwork.channel(bits)` and `HybridCoupledNetwork.from_monolithic_z(...).channel(bits)` vs `-100 * I_ue` of the four direct BS-source NEC solves; Frobenius relative error, all 10 states | max <= 1e-6 | Same arithmetic argument. Verifies H = -100 I_UE (power-wave scaling and port order) independently. |
| P1.3 | Woodbury: for F1 and F2, `CoupledNetwork.flip(...)` field vs `H_nec(bits) @ W` (direct NEC), relative; and the SE obtained through `CoupledNetwork.moves(...)` for the F1 flip vs the SE of the direct field | field <= 1e-6 ; SE abs. diff. <= 1e-4 bit/s/Hz | Exact rank-k update. The SE bound allows for a modest amplification of a 1e-6 field error through the rate formula for a non-nulled state (R1 is a random state, SE of order 1 to 3). |
| P1.4 | RIS-only 64-wire structure, plane-wave incidence (3 incidences below): NEC `EX` type 1 port short-circuit currents vs `EmbeddedRISResponse.short_circuit_current_from_field` (reciprocity integral of the segment current distribution); relative error of the 64-vector | **max <= 1e-3** | The two quantities are equal in the continuum. The discrete difference is the field sampling at segment centers vs NEC's segment-integrated delta-gap formulation, O((k Delta)^2); at 21 segments the 3x3 probe gave 3.4e-4 and the error decreases at least quadratically with refinement, so 1e-3 is conservative for 81 or 161 segments. |
| P1.5 | Same structure: open-circuit identity `v_oc(helper)` vs `-Z_RIS i_sc` relative error | <= 1e-5 | Observed 3.6e-7 on the toy; same threshold as the existing regression test. |
| P1.6 | Same structure, loaded: for OFF, ON, alternating and R1, NEC `LD`+`EX1` port currents vs `-(Z_RIS + Z_L)^-1 v_oc` (2 incidences x 4 states), relative error | max <= 1e-3 | Same as P1.4; the loaded currents are linear in the incident excitation error. |

Plane-wave incidences (PyNEC arrival convention, theta-polarized, as in the microproof): A1 = (theta 70 deg, phi 30 deg), A2 = (90, 180), A3 = (60, 200). P1.6 is run for A1 and A2.

**P1 passes iff P1.0a-d, P1.1-P1.6 all hold at the primary mesh (81).** If the 161 run is also completed it must also hold for the gate to be reported as "passed at 81 and 161"; a failure at 161 only is reported as mesh sensitivity, and P1 is then labelled "passed at 81 only".

## Gate P2: factored spherical hybrid vs complete 70-port NEC

Hybrid path under test (code as is, not modified): `extract_ris_response` (RIS alone, 64 wires) -> `build_spherical_dipole_operators` (exact Hertzian dipole Green function with the segment current distribution of RIS wire 0; external self-impedance `Z_RIS[0,0]`; no BS-BS or UE-UE mutual coupling; RIS-alone `Z_RIS`) -> `HybridCoupledNetwork`. Truth: `CoupledNetwork` on the 70-port NEC Z (validated against direct loaded NEC solves by P1.1/P1.2, and for the best states re-verified directly, P2.0).

States (9): all-OFF, all-ON, R1..R5 (same bits as in P1), BS1 = best single flip from all-OFF, BP = best pair flip from all-OFF. BS1 and BP are found by exhaustive search (`CoupledNetwork.moves` over 64 singles and 2016 pairs) on the truth network, using the truth network's own precoder. The same exhaustive search is also run on the hybrid network to obtain hybrid-chosen singles and pairs.

Rate metric: worst-user SE, `CoupledNetwork.score`, noise 290 K, 1 MHz, 7 dB NF. **Precoder protocol (primary)**: each network uses its own reference-channel RZF, `precoder(channel(all-OFF))`, which is what both network classes do by default and what every benchmark does. This is the model-consistent protocol of the optimizer pipeline. Consequence: SE(all-OFF) is an interference-null (noise-limited, ~18.8 bit/s/Hz on the earlier reference Z) for both networks. A **common-precoder** variant (the truth precoder applied to both channels) is computed and reported as **non-gating** diagnostics, because it measures the absolute channel error at an interference null (a 3% channel error caps the SE near 10 bit/s/Hz) and is expected to be much more severe than the model-consistent protocol; its numbers are reported unaltered whichever way they fall.

| ID | Criterion | Tolerance | Justification |
|---|---|---|---|
| P2.0 | Truth validity: for BS1 and BP, `CoupledNetwork` channel vs direct NEC loaded solves (4 BS excitations), Frobenius relative | <= 1e-6 | Reference confirmed on the states that were selected by search. |
| P2.1 | Complex channel Frobenius relative error `||H_hyb-H_nec||_F/||H_nec||_F` for **each of the 9 states** | **max <= 0.05** | Inherited from the pre-declared proposal text (before any measurement). The 3x3 toy probe gave 2.5 to 3.3% with one BS and one UE; the 5% limit allows for modest extra error from 4 BS and 2 UE wires and a 64-wire array. |
| P2.2 | Worst-user SE difference `|SE_hyb(b) - SE_nec(b)|`, primary protocol, each of the 9 states | **max <= 0.5 bit/s/Hz** | First-order rate sensitivity to a relative amplitude error eps is 2 eps / ln 2 = 2.9 eps (0.14 bit/s/Hz for eps = 5%). The limit is ~3.5 times that to allow amplification where leakage is the difference of nearly cancelling terms. It is not intended to resolve the 0.1 bit/s/Hz optimizer contrasts. |
| P2.3 | Ranking: Spearman rank correlation between hybrid and truth SE over (a) all 64 single flips from all-OFF and (b) all 2016 pair flips from all-OFF | both >= 0.95 | The unchanged proposal text. A hybrid model that mis-ranks candidate moves would mislead any search. |
| P2.4 | Regret: truth SE of the hybrid-chosen best single / best pair vs truth SE of the truth-chosen one: `SE_nec(best by nec) - SE_nec(best by hybrid)` | both <= 0.5 bit/s/Hz | A move chosen on the hybrid must be nearly as good under truth; same magnitude as P2.2, since a regret cannot be required tighter than the SE accuracy. |
| P2.5 | Truth modulability `||H_nec(ON)-H_nec(OFF)||_F/||H_nec(OFF)||_F` | >= 0.01 | Precondition from the proposal text: the benchmark must exercise the RIS. |

**P2 passes iff P2.0-P2.5 all hold at the primary mesh (81).** Same 161 rule as for P1.

Non-gating diagnostics recorded for attribution only: (i) block errors of the hybrid operators A, C_in, C_out, H_direct against the Schur-reduced truth blocks; (ii) the modulation-fidelity error `||(H_hyb(b)-H_hyb(OFF)) - (H_nec(b)-H_nec(OFF))|| / ||H_nec(b)-H_nec(OFF)||`; (iii) a diagnostic variant of the hybrid with the full external block (exact BS-BS and UE-UE dipole mutual coupling, which the code under test omits) to see how much of the error that omission explains; (iv) the common-precoder SE comparison. None of these can turn a failed criterion into a pass.

**Variant P2v (added before any 64-wire run, after a 3x3, 11-segment smoke run of the pipeline that is not gate evidence).** The smoke run showed that `build_spherical_dipole_operators` omits BS-BS and UE-UE mutual coupling, although the four BS wires are only about half a wavelength apart; in the smoke case this produced a ~47% direct-channel error, against ~4% when the exact external mutual coupling is included. To tell the project whether a small code fix would pass, the same limits P2.1-P2.4 are also applied to a hybrid whose external block contains the exact pairwise dipole mutual impedances (assembled in `full_validation.py`, same Green function, same segment current distribution). This is reported as **P2v.1-P2v.4**, with P2.0 and P2.5 shared. P2v is **not** the gate: the gate P2 is the status of the code as written. If P2 fails and P2v passes, the correct statement is "P2 failed; a specific, bounded code change (adding the external mutual coupling to `build_spherical_dipole_operators`) is expected to pass and must be adopted and the gate rerun before the hybrid is used". No limit differs between P2 and P2v.

**Plane-wave variant**: not part of P2. `build_multipath_operators` needs an absolute amplitude/phase calibration (`scale_factor`) that the audit lists as unverified; checking it is the job of P4 (calibrated LOS check). The spherical hybrid in P2 is the physically derived reference.

## Execution and provenance

* Runner: `scripts/hybrid_gates_v2.py`, library `src/atomwise_ris/hybrid/full_validation.py`, tests `tests/test_hybrid_full_validation.py`.
* Compute: rx6600 (`/var/tmp/atomwise-ris-gates-20261107`, project-local `.venv-em` with PyNEC), tmux with log and DONE/FAILED markers. Nothing at 64 wires is run on the laptop (21 segments or smaller only, for tests and a smoke run that is not gate evidence).
* Output: `outputs/runs/hybrid_gates_v2_20261107/seg81/{p1.json,p2.json,summary.json,z_full.npy,z_ris_alone.npy}` (and `seg161/`). Each JSON stores the geometry, state bits, per-solve numbers, the declared tolerances in force, pass/fail per criterion, source hashes and timing.
* `src/atomwise_ris/hybrid/evidence.py` is extended only to register the new artifacts; it reports a gate as passed only if the artifact's own declared criteria all passed and its recorded tolerances equal the ones in this document.

## Results

Run 2026-11-07 on rx6600 (`labkc1-amd`, workspace `/var/tmp/atomwise-ris-gates-20261107`, PyNEC from the project `.venv-em`, 8 worker processes). Artifacts: `outputs/runs/hybrid_gates_v2_20261107/seg81/` (`p1.json`, `p2.json`, `summary.json`, `diag_external_only.json`, `z_full.npy`, `z_ris_alone.npy`, `nec_raw.npz` holding every raw NEC output so the evaluation can be rerun with `--reuse-raw`). The `external_only_diagnostic` function and the `--only-diag` flag were added after the 81-segment run (non-gating, appended code; the gating code and the declared limits were not touched), so `source_sha256` of `full_validation.py` inside `p1.json`/`p2.json` differs from the current file only by that appended function. A 3x3, 11-segment smoke run (not gate evidence) preceded the run, was used to debug the pipeline, and motivated the variant P2v described above.

**Verdict at the primary mesh (81): P1 PASSES. P2 FAILS (criterion P2.1).** P2v (the diagnostic variant with exact external coupling) also fails P2.1 and passes the other criteria. Nothing here authorizes the hybrid optimizer.

### Mesh 81 segments per wire (labkc1-amd, NEC wall 89.99181527498877 s, Z extraction 39 s, 5670 unknowns)

**P1** : PASS

| Criterion | Measured | Declared | Result |
|---|---|---|---|
| P1.0a | 5.58242e-08 | <= 1e-06 | pass |
| P1.0b | 0.542101 | > 0 | pass |
| P1.0c | 0.986633 | <= 1.0001 | pass |
| P1.0d | 13.6274 | < 100000 | pass |
| P1.1 | 3.25427e-13 | <= 1e-06 | pass |
| P1.2 | 2.30966e-13 | <= 1e-06 | pass |
| P1.3a | 1.93113e-13 | <= 1e-06 | pass |
| P1.3b | 2.46025e-13 | <= 0.0001 | pass |
| P1.4 | 5.65369e-05 | <= 0.001 | pass |
| P1.5 | 3.04975e-08 | <= 1e-05 | pass |
| P1.6 | 5.65558e-05 | <= 0.001 | pass |

Z diagnostics: reciprocity 5.582e-08, min eig Re(Z) 0.5421 ohm, max sv S 0.986633, cond 13.63.

P1.1/P1.2 per state (max current-vector error over 8 excitations; reduced-channel errors of CoupledNetwork / HybridCoupledNetwork.from_monolithic_z):

| State | P1.1 max | CoupledNetwork H | Hybrid H |
|---|---|---|---|
| F1 | 2.42e-13 | 2.08e-13 | 2.08e-13 |
| F2 | 3.25e-13 | 2.31e-13 | 2.31e-13 |
| R1 | 2.07e-13 | 1.78e-13 | 1.78e-13 |
| R2 | 2.82e-13 | 1.82e-13 | 1.82e-13 |
| R3 | 2.72e-13 | 1.75e-13 | 1.75e-13 |
| R4 | 3.19e-13 | 2.22e-13 | 2.22e-13 |
| R5 | 2.59e-13 | 2.22e-13 | 2.22e-13 |
| all_off | 2.47e-13 | 1.55e-13 | 1.55e-13 |
| all_on | 2.44e-13 | 1.88e-13 | 1.88e-13 |
| alternating | 2.14e-13 | 2.19e-13 | 2.19e-13 |

Woodbury:
- F1: parent R1, flip [17], field rel err 1.40e-13, SE moves 1.461264 vs direct 1.461264 (abs diff 2.27e-13)
- F2: parent R1, flip [5, 42], field rel err 1.93e-13, SE moves 1.869960 vs direct 1.869960 (abs diff 2.46e-13)

Plane wave (RIS-alone 64 wires):

| Incidence | short-circuit rel err | open-circuit identity | loaded: R1, all_off, all_on, alternating |
|---|---|---|---|
| A1 (70,30) | 5.20e-05 | 3.05e-08 | 5.20e-05, 5.20e-05, 5.18e-05, 5.20e-05 |
| A2 (90,180) | 5.65e-05 | 1.14e-08 | 5.64e-05, 5.66e-05, 5.64e-05, 5.65e-05 |
| A3 (60,200) | 4.52e-05 | 1.52e-08 | n/a |

**P2** : FAIL

| Criterion | Measured | Declared | Result |
|---|---|---|---|
| P2.0 | 2.11632e-13 | <= 1e-06 | pass |
| P2.1 | 0.464898 | <= 0.05 | **FAIL** |
| P2.2 | 0.362376 | <= 0.5 | pass |
| P2.3a | 0.998855 | >= 0.95 | pass |
| P2.3b | 0.999008 | >= 0.95 | pass |
| P2.4a | 0.0376268 | <= 0.5 | pass |
| P2.4b | 0 | <= 0.5 | pass |
| P2.5 | 0.69934 | >= 0.01 | pass |

Per state (own precoder primary; full-ext = diagnostic variant):

| State | channel err | SE truth | SE hybrid | |dSE| | modulation err | full-ext channel err | full-ext SE | full-ext modulation err | common-W |dSE| |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.4634 | 1.539 | 1.177 | 0.362 | 0.1881 | 0.0718 | 1.269 | 0.0209 | 0.97 |
| R2 | 0.4461 | 1.922 | 1.565 | 0.357 | 0.2070 | 0.0695 | 1.719 | 0.0239 | 1.67 |
| R3 | 0.4529 | 2.000 | 1.809 | 0.191 | 0.1895 | 0.0702 | 1.939 | 0.0209 | 1.48 |
| R4 | 0.4649 | 1.745 | 1.685 | 0.060 | 0.1963 | 0.0727 | 1.662 | 0.0225 | 0.85 |
| R5 | 0.4647 | 1.998 | 1.710 | 0.288 | 0.1933 | 0.0727 | 1.836 | 0.0211 | 1.43 |
| all_off | 0.4223 | 18.752 | 18.531 | 0.221 | - | 0.0693 | 18.540 | - | 15.95 |
| all_on | 0.4382 | 0.178 | 0.082 | 0.096 | 0.1913 | 0.0646 | 0.136 | 0.0210 | 0.11 |
| best_pair | 0.4224 | 14.012 | 13.774 | 0.238 | 0.3407 | 0.0695 | 13.815 | 0.0431 | 11.12 |
| best_single | 0.4242 | 11.588 | 11.289 | 0.299 | 0.1707 | 0.0696 | 11.340 | 0.0138 | 8.88 |

Search on truth from all-OFF: best single flip index 49 (SE 11.5883), best pair [7, 33] (SE 14.0122). Hybrid-chosen: single 54, pair [7, 33]. Spearman single 0.99886, pair 0.99901; regret single 0.0376, pair 0.0000.

Variant P2v (exact BS-BS/UE-UE coupling; NOT the gate): FAIL

| Criterion | Measured | Declared | Result |
|---|---|---|---|
| P2v.1 | 0.0727482 | <= 0.05 | **FAIL** |
| P2v.2 | 0.270224 | <= 0.5 | pass |
| P2v.3a | 0.999084 | >= 0.95 | pass |
| P2v.3b | 0.999461 | >= 0.95 | pass |
| P2v.4a | 0.0376268 | <= 0.5 | pass |
| P2v.4b | 0 | <= 0.5 | pass |

Operator block errors vs Schur-reduced truth (code under test / full-external variant): A 3.36e-04 / 2.84e-04; C_in 0.1876 / 0.0193; C_out 0.0378 / 0.0095; H_direct 0.4791 / 0.0739. Z_RE block (spherical vs NEC) 0.0166; BS-BS mutual 0.0330; common-precoder max |dSE| 15.95.

External-only attribution (6-wire NEC):
- external block, model(full) vs NEC 6-wire: 0.0167; NEC 6-wire vs 70-port external block: 0.0011
- direct channel vs NEC-70: code_under_test_model: 0.4791
- direct channel vs NEC-70: model_with_full_external_coupling: 0.0739
- direct channel vs NEC-70: nec_external_only_6wire: 0.0473
- direct channel vs NEC 6-wire: code_under_test_model: 0.4720
- direct channel vs NEC 6-wire: model_with_full_external_coupling: 0.0598


### Diagnosis of the P2 failure (81 segments)

1. **Primary cause (code under test).** `build_spherical_dipole_operators` builds the external block with only the UE-BS mutual terms; BS-BS and UE-UE couplings are set to zero. The four BS wires are about half a wavelength apart, so the coupling is large. This makes the `H_direct` block 47.9% wrong and the per-state channel 42.2-46.5% wrong (P2.1 limit 5%). Adding the exact pairwise dipole mutual impedances (same Green function and current distribution) reduces the direct-channel block error to 7.4% and the per-state channel error to 6.5-7.3% (P2v.1, still above 5%).
2. **Remaining 7.3% (P2v).** The 6-wire NEC attribution shows two comparable contributions to the direct channel (norm of the truth direct block is 0.0210): (a) the dipole model of the external block is 1.7% off in `Z_EE` (vs a 6-wire NEC solve) but that becomes a 6.0% error in `H_direct`, because `H_direct` is a small off-diagonal element of `inv(Z_EE + 50 I)`; (b) the truth network always contains the 64 open-gap RIS wires, which scatter and change the BS-UE direct channel by 4.7% relative to a 6-wire structure; the hybrid `H_direct` has no such term by construction (the RIS-state dependence comes only from the factored path).
3. **The RIS-dependent part is accurate.** With the exact external block: `A` 2.8e-4, `C_in` 1.9%, `C_out` 0.95%, RIS-modulation error ||(H(b)-H(OFF)) - truth|| / ||truth modulation|| of 1.4-4.3%. Under the code under test the modulation error is 17-34%, `C_in` 18.8%, `C_out` 3.8% (the BS-BS omission also corrupts the BS-side excitation).
4. **Model-consistent SE and ranking are good, absolute channel is not.** P2.2 (max 0.362 vs 0.5), P2.3 (Spearman 0.9989/0.9990) and P2.4 (regret 0.038/0.0) pass with each network using its own precoder. The common-precoder protocol (non-gating) differs by up to 15.9 bit/s/Hz at the all-OFF interference null, which is what a 42% direct-channel error does to a precoder designed on the other model. The P2.2 limit of 0.5 is generous relative to the 0.1 bit/s/Hz contrasts of the optimizer study; the hybrid model must not be used to resolve those differences without further evidence.
5. The failure is not a numerical artifact: P2.0 and P1.1-P1.3 show that the truth network reproduces direct NEC to 1e-13.

### What would change the verdict (not applied, and not allowed to change this declaration)

* A code change that includes the BS-BS/UE-UE mutual coupling is necessary but, per P2v.1, not sufficient at the declared 5% limit.
* The 5% limit applies to the total channel including the direct term, and the direct term is partly set by quantities that the urban pipeline will not take from the dipole model (the ray tracer will supply `H_direct`). A re-declared gate that scores the RIS-mediated path only (`C_out (A + Z_L)^-1 C_in`, errors 1.4-4.3% above) with an external block and `H_direct` taken from NEC would be a different test; if it is wanted, its limits must be declared before it is run and it must be reported as a new gate, not as a pass of the one above.

## Run manifest and resume

| Item | Value |
|---|---|
| Host | rx6600 (`<rx6600>`, tailscale), tmux `hybrid-gates-v2` |
| Workspace | `/var/tmp/atomwise-ris-gates-20261107` (copy of `src/`, `scripts/hybrid_gates_v2.py`, this doc); python `/var/tmp/atomwise-ris-gen-20261106/.venv-em/bin/python` |
| Entry point | `bash /var/tmp/atomwise-ris-gates-20261107/run.sh` runs seg 81 (8 workers) then seg 161 (3 workers) |
| Logs / markers | `logs/seg81.log`, `logs/seg161.log`, `logs/status`, `logs/seg{81,161}.{DONE,FAILED}`, `logs/ALL_FINISHED` |
| Output | `outputs/runs/hybrid_gates_v2_20261107/seg{81,161}/` on rx6600; seg81 synced locally |
| Seg 81 | finished 20:00 WIB, rc=0, about 90 s of NEC time |
| Seg 161 | started 20:00 WIB; Z extraction took 240 s; expected total about 30-60 min |

Check: `ssh <rx6600> 'cat /var/tmp/atomwise-ris-gates-20261107/logs/status; ls /var/tmp/atomwise-ris-gates-20261107/logs; tail -3 /var/tmp/atomwise-ris-gates-20261107/logs/seg161.log'`

On completion: `rsync -a <rx6600>:/var/tmp/atomwise-ris-gates-20261107/outputs/runs/hybrid_gates_v2_20261107/seg161 outputs/runs/hybrid_gates_v2_20261107/`, then `PYTHONPATH=src .venv-em/bin/python scripts/hybrid_gates_v2.py --segments 161 --out outputs/runs/hybrid_gates_v2_20261107/seg161 --only-diag`, then append the seg-161 table here (the same table generator is just a formatter of `p1.json`, `p2.json`, `diag_external_only.json`). Rule from the declaration: P1/P2 are reported as "passed at 81 only" until the 161 artifact exists, and a 161 failure is reported as mesh sensitivity. If seg161 FAILED, read the tail of `logs/seg161.log`; rerun with a fresh `--out`, since the script refuses to overwrite.

## Open items and risks

* P1 tests that the network algebra, port order, termination handling, Woodbury updates and the segment-current/reciprocity machinery agree with independent NEC solves. It does not test the NEC kernel itself (mesh/kernel convergence is P3) and the plane-wave check is on the RIS-alone structure.
* Only one geometry (default users) and one frequency are covered. Random states are 5 fixed seeds; best states come from one-step search from all-OFF.
* The plane-wave `build_multipath_operators` path (absolute calibration) is untouched: it belongs to P4.

## P2 rerun after model fix (declared 2026-11-07, before the model code was changed)

**Why.** The first run (above) failed P2.1 at 81 and 161 segments. The diagnosis attributed the error to (i) the omitted BS-BS/UE-UE coupling plus the Hertzian-dipole model of the external block, and (ii) the structural scattering of the 64 open-gap RIS wires, which the factored model's `H_direct` lacks. The project owner approved a bounded fix and one rerun.

**Limits are unchanged.** P2.0-P2.5 keep exactly the definitions and limits declared above (P2.1 channel error <= 0.05 for each of the 9 states; P2.2 <= 0.5 bit/s/Hz with each network's own precoder; P2.3 Spearman >= 0.95; P2.4 regret <= 0.5 bit/s/Hz; P2.0 <= 1e-6; P2.5 >= 0.01). The state set, seeds, the exhaustive search for the best single and pair states on the truth, and the 81 (primary) / 161 (corroboration) mesh rule are unchanged. `THRESHOLDS_P2` in code is not edited. The rerun is the gate for the fixed code. **If any criterion fails, it is reported as a failure with exact numbers and no further model change is made in this round.** The first-run artifacts stay in `seg81`/`seg161` as the gate result for the code as it was. The rerun artifacts go to `fix1_seg81` and `fix1_seg161`. P1 is re-evaluated from the same raw NEC outputs for completeness (its criteria and limits are unchanged and do not depend on the hybrid builder).

**Model change (in `ray_coupling.py`; the old dipole path stays selectable and is the default for existing callers).**

a. *External block from NEC of the external antennas alone.* `Z_EE,free` (6x6: BS-BS, UE-UE, BS-UE and self impedances) is the port impedance matrix of the 4 BS + 2 UE wires solved by NEC without any RIS wire (`extract_ris_response` on the external geometry, 81/161 segments). This is a quantity that is physically available for a known BS/UE geometry. The current shape of each external wire in the Green-function integrals is that wire's own normalized segment current from the same solve (instead of the shape of RIS wire 0).

b. *Passive (open-gap) scattering of the RIS wires in the direct term.* Derivation, using only the RIS-alone structure and the external-dipole fields, never the 70-port truth.

Let the RIS-alone structure (64 continuous wires, ports at the centre segments) have the segment-level admittance operator `S` (`S_ij` = current in segment i per unit voltage across segment j; symmetric by reciprocity). The embedded port quantities used by the hybrid are `I_Y = S[:, ports]`, `Z_RIS = (S[ports, ports])^-1`, `I_Z = I_Y Z_RIS`. An external antenna `p` carrying unit current produces the tangential field `E_p` at the RIS segment centres (Hertzian-dipole Green function, the same one that builds `Z_RE`). With `dl` the segment lengths and `u_p = dl * E_p` (equivalent segment EMFs, as in the plane-wave short-circuit check P1.4):

1. Current induced on the structure with all port gaps **continuous** (short-circuited ports): `I_sc,p = S u_p`.
2. Short-circuit port currents: `i_sc,p = I_Y^T u_p`.
3. With the gaps **open** (port current forced to zero) the extra gap voltage `v = -Z_RIS i_sc,p` drives `I_Y v`, so the induced current is `I_oc,p = I_sc,p - I_Z i_sc,p`.
4. By reciprocity the open-circuit voltage induced at external port `a` by the field re-radiated from `I_oc,p` is `dZ[a,p] = - u_a^T I_oc,p`.
5. The external block with the RIS ports open is `Z_EE = Z_EE,free + dZ`, and equivalently `dZ = -U^T S U + Z_RE^T Z_RIS^-1 Z_RE` (the second term is the port-circuit part already contained in the Schur complement `A`; the first term is the scattering of the continuous wires, which is the extra piece that port-level data cannot give).
6. Then the unchanged algebra of derivation section 2.3 applies with `E = Z_EE + 50 I`: `D = -100 [E^-1]_UE,BS`, `C_out`, `C_in`, `A = Z_RIS - Z_RE E^-1 Z_ER`. `H_direct = D` is then the channel with the RIS present and its ports open, as the truth network defines it, and `ZL -> infinity` gives `H = D` exactly.

`I_sc,p = S u_p` is obtained by one NEC solve of the RIS-alone structure with a voltage source `u_p[j]` simultaneously on every segment j (NEC superposes `EX` sources; the linearity is unit-tested); 6 solves (one per external port), no 70-port run. The same sampling convention (segment-centre field times `dl` as segment voltage) is the one validated against NEC `EX1` plane waves in P1.4 (5e-5). The scattering is first order in the external wire currents (external wires carry the isolated current shape); that approximation is part of what the gate tests. Note: `hybrid-derivation.md` section 7 states that an open-circuited wire's backscatter is negligible (about 1e-6); the first run measured 4.7% of the direct channel for these half-wave-class wires, so section 7 is wrong for this geometry and is annotated.

**What the fixed hybrid may and may not use.** Allowed: RIS-alone NEC (Z_RIS, segment currents, the 6 impressed-field solves), external-antennas-alone NEC, the Hertzian-dipole Green function. Not allowed and not used: any element of the 70-port `Z` or of the truth network, in any operator. The truth is used only for scoring, exactly as before (the builder functions do not take it as an argument).

**Reporting.** Gate = fixed builder (a+b). Non-gating diagnostics in the same artifact: legacy dipole builder (the first-run code), and (a)-only. All three get per-state channel error, block errors (A, C_in, C_out, H_direct), own-precoder SE, modulation fidelity, ranking and regret. Mesh rule: gate status is the 81-segment result; 161 is corroboration.

### Rerun results (2026-11-07, rx6600, tmux `hybrid-gates-fix1`, markers `logs/fix1_seg{81,161}.DONE`)

Code state: `ray_coupling.py` gained `external_field_vectors`, `impressed_segment_currents`, `open_gap_scatter_impedance`, `build_spherical_nec_external_operators` (legacy builder untouched and still the default dipole path); `full_validation.py` gained `evaluate_p2_fixed`; `scripts/hybrid_gates_v2.py --reuse-raw --builder fixed` re-evaluates the stored truth NEC outputs (`nec_raw.npz`, byte-identical copies of the first run) and adds two small NEC jobs per mesh (external antennas alone: under 1 s; RIS-alone impressed-field solve with 6 right-hand sides: 23 s at 81, about 2.7 min at 161), stored in `nec_raw_fix1.npz`. Artifacts: `outputs/runs/hybrid_gates_v2_20261107/fix1_seg81/` and `fix1_seg161/` (synced to the laptop with sha256 verification against the remote files; all OK). No limit and no `THRESHOLDS_P2` value was changed; the artifacts record the unchanged thresholds. The truth network is used only for scoring and for the non-gating comparisons marked as such.

**Verdict: the fixed hybrid passes every P2 criterion at 81 (primary) and at 161 segments. P1 is unchanged (pass at both).** This is a pass of the fixed code after a fix that was designed knowing the first-run failure; it is not an independent first-try pass, and the gate of the code as first written remains FAILED in `seg81`/`seg161`.

#### 81 (primary)

| P2 criterion | Measured | Declared | Result |
|---|---|---|---|
| P2.0 | 2.11632e-13 | <= 1e-06 | pass |
| P2.1 | 0.00981866 | <= 0.05 | pass |
| P2.2 | 0.0663104 | <= 0.5 | pass |
| P2.3a | 0.999817 | >= 0.95 | pass |
| P2.3b | 0.99992 | >= 0.95 | pass |
| P2.4a | 0 | <= 0.5 | pass |
| P2.4b | 0 | <= 0.5 | pass |
| P2.5 | 0.69934 | >= 0.01 | pass |

P1 (re-evaluated from the same raw NEC outputs): P1.0a 5.58e-08; P1.0b 0.542; P1.0c 0.987; P1.0d 13.6; P1.1 3.25e-13; P1.2 2.31e-13; P1.3a 1.93e-13; P1.3b 2.46e-13; P1.4 5.65e-05; P1.5 3.05e-08; P1.6 5.66e-05

| State | channel err | SE truth | SE hybrid | abs dSE | modulation err |
|---|---|---|---|---|---|
| R1 | 0.0055 | 1.539 | 1.545 | 0.006 | 0.0206 |
| R2 | 0.0077 | 1.922 | 1.934 | 0.011 | 0.0225 |
| R3 | 0.0071 | 2.000 | 1.998 | 0.002 | 0.0207 |
| R4 | 0.0051 | 1.745 | 1.745 | 0.000 | 0.0217 |
| R5 | 0.0048 | 1.998 | 1.998 | 0.000 | 0.0211 |
| all_off | 0.0097 | 18.752 | 18.775 | 0.023 | - |
| all_on | 0.0095 | 0.178 | 0.179 | 0.001 | 0.0209 |
| best_pair | 0.0098 | 14.012 | 13.946 | 0.066 | 0.0376 |
| best_single | 0.0096 | 11.588 | 11.585 | 0.003 | 0.0198 |

Truth best single 49 (SE 11.5883), best pair [7, 33] (SE 14.0122); hybrid chose single 49, pair [7, 33]. Spearman 0.99982/0.99992, regret 0.0000/0.0000.

| Block rel. error vs Schur-reduced truth | A | C_in | C_out | H_direct | worst channel err |
|---|---|---|---|---|---|
| fixed (gate): NEC external + open-gap | 2.57e-04 | 0.0209 | 0.0049 | 0.0010 | 0.0098 |
| diagnostic_legacy_dipole | 3.36e-04 | 0.1876 | 0.0378 | 0.4791 | 0.4649 |
| diagnostic_nec_external_only | 2.57e-04 | 0.0209 | 0.0049 | 0.0473 | 0.0461 |

Fix info (non-gating, truth used only for these comparisons): |dZ_open| 0.2324 ohm vs |Z_EE,free| 215.3; dZ vs truth difference 0.0215; Z_EE rel err 2.30e-05 (free only 1.07e-03); Z_RE rel err 0.0168.


#### Mesh 161 segments per wire: P1 PASS, P2 (fixed hybrid) PASS

| P2 criterion | Measured | Declared | Result |
|---|---|---|---|
| P2.0 | 7.64445e-13 | <= 1e-06 | pass |
| P2.1 | 0.0102411 | <= 0.05 | pass |
| P2.2 | 0.0699845 | <= 0.5 | pass |
| P2.3a | 0.999817 | >= 0.95 | pass |
| P2.3b | 0.999913 | >= 0.95 | pass |
| P2.4a | 0 | <= 0.5 | pass |
| P2.4b | 0 | <= 0.5 | pass |
| P2.5 | 0.698971 | >= 0.01 | pass |

P1 (re-evaluated from the same raw NEC outputs): P1.0a 9.8e-09; P1.0b 0.54; P1.0c 0.987; P1.0d 13.7; P1.1 1.57e-12; P1.2 8.23e-13; P1.3a 5.84e-13; P1.3b 2.99e-12; P1.4 1.62e-05; P1.5 5.33e-09; P1.6 1.63e-05

| State | channel err | SE truth | SE hybrid | abs dSE | modulation err |
|---|---|---|---|---|---|
| R1 | 0.0057 | 1.552 | 1.558 | 0.006 | 0.0214 |
| R2 | 0.0080 | 1.926 | 1.938 | 0.012 | 0.0235 |
| R3 | 0.0074 | 1.995 | 1.993 | 0.002 | 0.0216 |
| R4 | 0.0053 | 1.743 | 1.743 | 0.000 | 0.0227 |
| R5 | 0.0050 | 1.995 | 1.995 | 0.000 | 0.0220 |
| all_off | 0.0101 | 18.757 | 18.780 | 0.023 | - |
| all_on | 0.0099 | 0.177 | 0.178 | 0.001 | 0.0218 |
| best_pair | 0.0102 | 14.020 | 13.950 | 0.070 | 0.0394 |
| best_single | 0.0100 | 11.592 | 11.589 | 0.003 | 0.0205 |

Truth best single 49 (SE 11.5920), best pair [7, 33] (SE 14.0195); hybrid chose single 49, pair [7, 33]. Spearman 0.99982/0.99991, regret 0.0000/0.0000.

| Block rel. error vs Schur-reduced truth | A | C_in | C_out | H_direct | worst channel err |
|---|---|---|---|---|---|
| fixed (gate): NEC external + open-gap | 2.68e-04 | 0.0218 | 0.0051 | 0.0011 | 0.0102 |
| diagnostic_legacy_dipole | 3.44e-04 | 0.1875 | 0.0375 | 0.4791 | 0.4651 |
| diagnostic_nec_external_only | 2.68e-04 | 0.0218 | 0.0051 | 0.0494 | 0.0482 |

Fix info (non-gating, truth used only for these comparisons): |dZ_open| 0.2416 ohm vs |Z_EE,free| 214.6; dZ vs truth difference 0.0224; Z_EE rel err 2.50e-05 (free only 1.12e-03); Z_RE rel err 0.0176.


Reading: the fixed hybrid cuts the worst per-state channel error from 46.5% to 0.98% (81) and 1.02% (161); the direct-channel block error falls from 47.9% to 0.10% with both parts of the fix, and only to 4.7-4.9% with the NEC external block alone, so the open-gap scattering term accounts for the remainder as predicted (the (a)-only network would also have met the 5% limit narrowly, at 4.6% and 4.8%, but it is a diagnostic and not the gate). The residual 1.0% channel error is dominated by the dipole-model `Z_RE` coupling (1.7% block error, `C_in` 2.1%, `C_out` 0.5%) and by the first-order treatment of the scattering (dZ within 2.2% of the truth difference). RIS-modulation fidelity is 2.0-2.4% (3.8-3.9% for the best pair). Best single and best pair chosen by the hybrid coincide with the truth's.

### Remaining risks after the rerun

* The pass certifies the *free-space benchmark* hybrid, in which the external block comes from NEC of the BS/UE antennas alone and `Z_RE` from the Hertzian-dipole Green function. In the urban pipeline the BS/UE coupling to the RIS comes from ray tracing; P4 (calibrated LOS check) is still open. The open-gap scattering term has to be carried into any pipeline that supplies `H_direct` from a ray tracer: that `H_direct` has no RIS wires, whereas the benchmark `D` includes the 4.7% structural term. The model gives it as `D(Z_EE,free + dZ) - D(Z_EE,free)`.
* One geometry, one frequency, 5 fixed random states, best states from a one-step search from all-OFF; the per-state P2.1 error (about 1%) has 5x headroom against the 5% limit, but SE differences of the order of 0.1 bit/s/Hz (the contrasts of the optimizer study) are still not resolved with confidence: the worst P2.2 value is 0.066 (81) and 0.070 (161) bit/s/Hz, on the best-pair state.
* The mesh sensitivity of the fix is small between 81 and 161 (channel error 0.98% vs 1.02%), but this is the same NEC kernel on both sides; kernel validity is P3.
* P3 and P4 remain unevaluated here; nothing in this document authorizes hybrid optimization or training.
