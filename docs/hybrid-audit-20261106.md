# Hybrid gate evidence audit — 2026-11-06

This audit supersedes the `passed` labels in `outputs/runs/hybrid_gates/` for gate authorization. The historical JSON files remain unchanged and are hashed in `outputs/runs/hybrid_gates_audit_20261106_v3/gate_evidence_audit.json`. The original P1 and P2 results verify internal algebraic consistency, not the required independent physical comparisons. P3 contains useful mesh data, but its state selection and separate returned-action coverage do not establish the declared validation gate. P4 checks finite values and heuristic magnitudes without a calibrated LOS comparison. **No hybrid optimizer or training is authorized by these results.**

## Direct small-array evidence

`outputs/runs/hybrid_gates_audit_20261106_v3/direct_nec_microproof.json` records a direct PyNEC 2.3.4 comparison on a free-space 3×3 RIS with one BS and one UE wire, 5.53 GHz, and 21 segments per wire. This is a convention and implementation probe, not evidence for the 64-element urban system.

The EX1 plane-wave induced port current differs from the reciprocity integral by `3.4333e-4` relative. Loaded-current comparisons for all-OFF, all-ON, and alternating states differ by `3.3285e-4` to `3.4413e-4`. The open-circuit reciprocity identity residual is `3.6239e-7`. These observed values are recorded as regression evidence, not as predeclared pass criteria.

The factored spherical hybrid channel was compared with a separately extracted complete 11-port NEC impedance solve. Relative complex-channel errors were `2.4982%` (all-OFF), `3.3428%` (all-ON), and `2.6015%` (alternating). This is the measured approximation error for this toy geometry; no threshold is inferred from it.

## Mesh interpretation and thin-wire kernel

The local NEC wrapper enables NEC-2's extended thin-wire kernel. For a 24 mm wire of radius 0.05 mm, the segment-length/radius ratio is `480/N`: 5.93 at 81 segments, 2.98 at 161, and 1.99 at 241. The NEC-2 Part I theory manual states that its ordinary thin-wire approximation has less than 1% error for Δ/a above 8 and that the extended approximation is applicable to shorter/thicker segments with less than 1% error for Δ/a above 2. Those are statements about the kernel approximation under the manual's assumptions; they are not a global impedance, channel, or rate-error guarantee. Thus 81 and 161 satisfy that stated extended-kernel ratio condition, while 241 is just below it. Refining the segmentation is not automatically more accurate once the model moves outside that stated condition.

For these parameters, `ka` is about 0.00585 and each segment is below 0.1 wavelength, but those facts do not validate every geometry interaction or prove convergence of the coupled channel. The 161-segment dataset is a higher-resolution reference mesh, not ground truth. Final convergence on the frozen physical actions, in a justified kernel regime, remains pending. Source: [NEC-2 Part I: Theory, §3 “Evaluation of the Fields”](https://www.nec2.org/other/nec2prt1.pdf#page=21), especially pp. 21–24.

## Status by gate

- **P1 — not verified at full size.** The legacy Schur residual compares algebraic forms using the same impedance matrix. The small direct probe above verifies conventions only; the complete 64-port direct-source comparison remains unrun.
- **P2 — not verified at full size.** The toy direct-NEC comparison now measures the spherical approximation error, but there is no full-size independent direct-NEC benchmark or justified error acceptance criterion.
- **P3 — partial, with a material warning.** Existing returned-action mesh audit covers 54 actions on one pilot frame and observed maximum absolute rate change of `0.643806 bit/s/Hz` between the compared meshes. It is not broad validation-trajectory evidence. Legacy random and one-step states do not substitute for completed returned actions.
- **P4 — blocked.** The saved path operator is uncalibrated against direct LOS phase, polarization, incident port response, and outgoing transfer at NEC wire locations. Urban multipath integration must wait for those checks.

No pilot or test samples were used by this audit. It reads the existing action-mesh artifact and legacy gate artifacts, and runs only the stated 3×3 free-space probe. The gate audit exits with status 2 intentionally because no full gate is currently established.
