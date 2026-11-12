# Hybrid Ray-Tracing + NEC-2 Physics Validation Gates

**Document version:** 1.0  
**Date:** 2026-11-06  
**Scope:** Pre-declared numeric validation gates P1–P4 for the hybrid channel model. Numerical tolerances are declared **before** running evaluations and must not be retroactively loosened.

> **Superseded for gate authorization.** The audit in [`hybrid-audit-20261106.md`](hybrid-audit-20261106.md) found that the saved P1/P2 artifacts are self-comparisons, P3 lacks broad completed-action validation, and P4 lacks a calibrated LOS comparison. The thresholds below are historical proposal text; they are not evidence that the gates passed and do not authorize hybrid optimization or training. The full-size P1/P2 gates are now declared and measured in [`hybrid-gates-v2-20261107.md`](hybrid-gates-v2-20261107.md).

---

## 1. Principles of Gate Design

1. **A Priori Declaration:** All pass criteria and numeric tolerances are justified by electromagnetic theory, floating-point numerical conditioning, or established benchmark convergence before computing gate results.
2. **Strict Non-Loosening Policy:** If a gate fails, the failure must be diagnosed and reported with exact numbers; thresholds are never adjusted post-hoc to force a pass.
3. **Hierarchy of Verification:**
   - **P1 (Port Consistency):** Microscopic circuit and network consistency on the multiport impedance matrix.
   - **P2 (Free-Space Benchmark):** Factored hybrid operator vs complete 70-port Method of Moments full-wave solution.
   - **P3 (Mesh Fidelity):** Numerical convergence under MoM segment refinement (41 vs 81 vs 161 segments) on both random and optimized states.
   - **P4 (Integrated Urban Case):** Full coupling with ray-traced Munich paths, testing passivity, sensitivity, and multipath coherence.

---

## 2. Gate P1: Multiport Network & Port Consistency

### 2.1 Criteria and Numerical Tolerances

| Metric | Pass Condition | Theoretical / Numerical Justification |
|---|---|---|
| **Reciprocity Residual** | $\frac{\|Z - Z^T\|_F}{\|Z\|_F} \le 1.0 \times 10^{-6}$ | By Lorentz reciprocity in isotropic media, $Z = Z^T$. NEC-2 floating-point precision yields residuals around $10^{-8} - 10^{-7}$. $10^{-6}$ catches broken indexing or asymmetry. |
| **Passivity (Resistance Spectrum)** | $\lambda_{\min}\left(\frac{Z + Z^H}{2}\right) > 0\ \Omega$ | In a passive antenna array in free space, radiated power $P_{\text{rad}} = \frac{1}{2} \mathbf{i}^H \operatorname{Re}(Z) \mathbf{i} > 0$ for all $\mathbf{i} \ne \mathbf{0}$. $\lambda_{\min} > 0$ guarantees strictly passive behavior. |
| **Passivity (Scattering Matrix)** | $\sigma_{\max}(S) \le 1.0001$ | For $S = (Z - Z_0 I)(Z + Z_0 I)^{-1}$ with $Z_0 = 50\ \Omega$, energy conservation requires $\|S\|_2 \le 1.0$. The $0.0001$ tolerance accounts for matrix inversion conditioning. |
| **Condition Number** | $\kappa(Z) < 1.0 \times 10^5$ | For an $8 \times 8$ wire array with $d \ge 0.5 \lambda$, $\kappa(Z)$ is typically $10 - 20$. $\kappa < 10^5$ guarantees stable, unamplified error propagation in solves. |
| **Schur Reduction vs Direct Solve** | $\frac{\|H_{\text{factored}}(\mathbf{b}) - H_{\text{direct}}(\mathbf{b})\|_F}{\|H_{\text{direct}}(\mathbf{b})\|_F} \le 1.0 \times 10^{-10}$ | The Schur complement formula $D + C [A + Z_L(\mathbf{b})]^{-1} F$ is an exact algebraic reduction. In IEEE 754 float64, it must match the full $(N+6)\times(N+6)$ solve to within $10^{-10}$. |

---

## 3. Gate P2: Free-Space Benchmark vs 70-Port NEC-2

### 3.1 Criteria and Numerical Tolerances

Gate P2 benchmarks the hybrid operator against a reference geometry where both the full-wave 70-port NEC-2 solve and the hybrid operators can be computed directly.

| Metric | Pass Condition | Theoretical / Numerical Justification |
|---|---|---|
| **Spherical-Wave Dipole vs Full NEC-2** | $\frac{\|H_{\text{hybrid, spherical}}(\mathbf{b}) - H_{\text{NEC-70}}(\mathbf{b})\|_F}{\|H_{\text{NEC-70}}(\mathbf{b})\|_F} \le 0.05$ (5.0%) | In full 70-port NEC-2, external BS and UE dipoles interact with the RIS and with each other through multi-bounce wire currents. Factoring external coupling with Green's functions isolates the RIS. At $r \sim 0.5\text{ m}$, the missing higher-order wire-to-wire multi-reflections account for $< 3-5\%$ error. |
| **State Modulability / Contrast** | $\frac{\|H(\text{all-ON}) - H(\text{all-OFF})\|_F}{\|H(\text{all-OFF})\|_F} \ge 0.01$ (1.0%) | The load variation between $1 - j 10.8\ \Omega$ and $1 + j 129.2\ \Omega$ must produce measurable channel modulation ($> 1\%$). |
| **State Ranking Preservation** | Spearman rank correlation $\rho \ge 0.95$ across test states | The hybrid model must rank candidate RIS states (all-OFF, all-ON, random configurations, single-cell best, pair-cell best) consistently with the full-wave simulation. |
| **Far-Field vs Spherical Agreement** | $\frac{\|H_{\text{plane}}(\mathbf{b}) - H_{\text{spherical}}(\mathbf{b})\|_F}{\|H_{\text{spherical}}(\mathbf{b})\|_F} \le 0.15$ (15.0%) | For external antennas at $0.5\text{ m}$ (within the aperture Fresnel zone $d < 2D^2/\lambda = 3.04\text{ m}$), plane-wave phase curvature error is non-zero but bounded. |

---

## 4. Gate P3: Mesh Discretization Fidelity

### 4.1 Criteria and Numerical Tolerances

Gate P3 tests convergence under progressive segment refinement of the Method of Moments thin-wire kernel (41 segments vs 81 segments vs 161 segments per wire).

| Metric | Pass Condition | Theoretical / Numerical Justification |
|---|---|---|
| **81 vs 161 Segment Channel Relative Error** | $\max_{\mathbf{b}} \frac{\|H_{81}(\mathbf{b}) - H_{161}(\mathbf{b})\|_F}{\|H_{161}(\mathbf{b})\|_F} \le 0.015$ (1.5%) | Thin-wire MoM current convergence between segment lengths $\Delta \ell \approx 0.3\text{ mm}$ (81 segments) and $0.15\text{ mm}$ (161 segments). In prior baseline audits, random states had max error $0.74\%$. We set 1.5% across both random and optimized states. |
| **41 vs 81 Segment Channel Relative Error** | $\max_{\mathbf{b}} \frac{\|H_{41}(\mathbf{b}) - H_{81}(\mathbf{b})\|_F}{\|H_{81}(\mathbf{b})\|_F} \le 0.035$ (3.5%) | Coarser mesh (41 segments) shows monotonic convergence toward the 161-segment reference. |
| **Worst-User SE Rate Error** | $\max_{\mathbf{b}} \|\text{Rate}_{81}(\mathbf{b}) - \text{Rate}_{161}(\mathbf{b})\| \le 0.05\text{ bps/Hz}$ | A channel error $< 1.5\%$ translates to $< 0.02 - 0.05\text{ bps/Hz}$ spectral efficiency difference at nominal SNR. |
| **Audit on Optimized Actions** | Error on best-single and best-pair states $\le 0.015$ | Refinement stability must hold on extreme/optimized states, not merely on random states. |

---

## 5. Gate P4: Single-Path & Multi-Path Urban Case (Munich Scene)

### 5.1 Criteria and Numerical Tolerances

Gate P4 integrates the ray-traced Munich urban propagation paths (`scenario.npz` and DeepMIMO export) with the 64-element NEC-2 $Z_{\text{RIS}}$ matrix.

| Metric | Pass Condition | Theoretical / Numerical Justification |
|---|---|---|
| **Numerical Validity** | No NaNs, Infs, or exact zeros in $H_{\text{eff}}$ | All matrices must be well-conditioned and finite. |
| **Physical Passivity** | $\max_{k, m} |H_{\text{eff}, km}| < 1.0$ (and total power gain $< 0\text{ dB}$) | Path loss over $> 300\text{ m}$ distance in free space/urban environment is $\ll -60\text{ dB}$. Unphysical amplification ($|H| > 1$) would indicate an invalid port normalization or scale factor. |
| **Load Sensitivity in Multipath** | $\frac{\|H_{\text{eff}}(\mathbf{1}) - H_{\text{eff}}(\mathbf{0})\|_F}{\|H_{\text{eff}}(\mathbf{0})\|_F} \ge 0.005$ (0.5%) | The RIS-mediated component must produce a detectable modulation on the user channel in the presence of direct and multipath propagation. |
| **Single-Path to Multipath Monotonicity** | Coherent addition of reflection paths | Verifies that adding building and ground reflections modifies $H_{\text{eff}}$ without catastrophic numerical instability. |
| **Channel Rank & Condition** | $\operatorname{rank}(H_{\text{eff}}) = \min(N_{\text{UE}}, N_{\text{BS}}) = 2$ | The $2 \times 4$ MIMO channel has full row rank (independent spatial degrees of freedom). |

---

## 6. Execution and Reporting Protocol

1. Each gate script must output structured JSON artifacts recording exact values, pass/fail flags, and execution metadata into `outputs/runs/hybrid_gates/`.
2. If any gate fails its declared tolerance, execution halts, the exact failure is recorded in the final report, and no subsequent optimizer stages are run.
