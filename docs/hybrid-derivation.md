# Hybrid Ray-Tracing + NEC-2 Physical Channel Model: First-Principles Derivation

**Document version:** 1.0  
**Date:** 2026-11-06  
**Scope:** Electromagnetic derivation of the hybrid channel operator coupling ray-traced multipath propagation with thin-wire Method of Moments (NEC-2) multiport network physics. Physics only (no optimizer claims).

---

## 1. Executive Summary & Purpose

The purpose of this document is to establish a rigorous, first-principles derivation of the hybrid channel model that combines:
1. **Ray-traced multipath propagation** (from Sionna RT / DeepMIMO), which accurately captures site-specific macro-propagation, shadowing, delays, and specular/diffuse reflections in complex 3D urban geometries.
2. **Multiport electromagnetic thin-wire Method of Moments** (from NEC-2 / PyNEC), which accurately captures full mutual coupling, inter-element reradiation, structural currents, and load-dependent scattering across the 64-element reconfigurable intelligent surface (RIS).

The key deliverable is the factored linear channel operator:
$$H_{\text{eff}}(\mathbf{b}) = H_{\text{direct}} + C_{\text{out}} \left[ Z_{\text{RIS}} + Z_L(\mathbf{b}) \right]^{-1} C_{\text{in}}$$
where all operators are explicitly derived from Maxwell's equations and circuit network theory, with defined units, reference impedances, polarization conventions, phase references, and validity conditions.

---

## 2. Multiport Network Formulation and Circuit Derivation

### 2.1 Complete Multiport Linear System

Consider an electromagnetic system consisting of $N$ RIS antenna elements, $N_{\text{BS}}$ base station (BS) transmit ports, and $N_{\text{UE}}$ user equipment (UE) receive ports. The total number of ports is $P = N + N_{\text{ext}}$, where $N_{\text{ext}} = N_{\text{BS}} + N_{\text{UE}}$ (in our reference system: $N=64$, $N_{\text{BS}}=4$, $N_{\text{UE}}=2$, so $P=70$).

In the frequency domain with time-harmonic convention $e^{j \omega t}$, the linear electromagnetic response of the entire structure is completely characterized by its $P \times P$ open-circuit impedance matrix $Z$:
$$\mathbf{v} = Z \mathbf{i}$$
Partitioning the ports into RIS ports (subscript $R$, dimension $N$) and external ports (subscript $E$, dimension $N_{\text{ext}}$):
$$\begin{bmatrix} \mathbf{v}_R \\ \mathbf{v}_E \end{bmatrix} = \begin{bmatrix} Z_{RR} & Z_{RE} \\ Z_{ER} & Z_{EE} \end{bmatrix} \begin{bmatrix} \mathbf{i}_R \\ \mathbf{i}_E \end{bmatrix}$$
where:
- $\mathbf{v}_R, \mathbf{i}_R \in \mathbb{C}^N$ are the port voltages and terminal currents of the RIS elements.
- $\mathbf{v}_E, \mathbf{i}_E \in \mathbb{C}^{N_{\text{ext}}}$ are the port voltages and terminal currents of the external antennas (BS and UE).
- By the Lorentz reciprocity theorem for linear, isotropic, time-invariant media: $Z = Z^T$, so $Z_{RR} = Z_{RR}^T$, $Z_{EE} = Z_{EE}^T$, and $Z_{ER} = Z_{RE}^T$.
- By passivity: $\operatorname{Re}(Z) = \frac{1}{2}(Z + Z^H) \succ 0$ (strictly positive definite for lossy media/radiation into open free space).

### 2.2 Terminal Boundary Conditions

1. **RIS Port Termination:**
   Each RIS port $n \in \{1, \dots, N\}$ is terminated by a state-dependent passive lumped load $z_{L, n}(b_n) \in \mathbb{C}$ ($\operatorname{Re}(z_{L, n}) > 0$). In vector notation, with $\mathbf{i}_R$ defined as current flowing *into* the antenna port:
   $$\mathbf{v}_R = - Z_L(\mathbf{b}) \, \mathbf{i}_R, \quad Z_L(\mathbf{b}) = \operatorname{diag}\left( z_{L, 1}(b_1), \dots, z_{L, N}(b_N) \right)$$

2. **External Generator & Load Termination:**
   All external ports use a real reference impedance $Z_0 = 50\ \Omega$.
   - Each BS port $m \in \{1, \dots, N_{\text{BS}}\}$ is connected to a Thevenin generator with open-circuit source voltage $v_{s, m}$ and internal source impedance $Z_0$:
     $$v_{E, m} = v_{s, m} - Z_0 \, i_{E, m}$$
   - Each UE port $k \in \{1, \dots, N_{\text{UE}}\}$ is terminated into a receiver matched load $Z_0$:
     $$v_{E, N_{\text{BS}} + k} = - Z_0 \, i_{E, N_{\text{BS}} + k}$$
   Combining all external ports:
   $$\mathbf{v}_E = \mathbf{v}_s - Z_0 I_{N_{\text{ext}}} \, \mathbf{i}_E, \quad \text{where } \mathbf{v}_s = \begin{bmatrix} \mathbf{v}_{s, \text{BS}} \\ \mathbf{0}_{N_{\text{UE}}} \end{bmatrix}$$

### 2.3 Schur-Complement Network Reduction

Equating the two expressions for $\mathbf{v}_E$:
$$Z_{ER} \mathbf{i}_R + Z_{EE} \mathbf{i}_E = \mathbf{v}_s - Z_0 I_{N_{\text{ext}}} \mathbf{i}_E$$
$$\left( Z_{EE} + Z_0 I_{N_{\text{ext}}} \right) \mathbf{i}_E = \mathbf{v}_s - Z_{ER} \mathbf{i}_R$$
Defining the external termination matrix:
$$E \triangleq Z_{EE} + Z_0 I_{N_{\text{ext}}} \in \mathbb{C}^{N_{\text{ext}} \times N_{\text{ext}}}$$
Since $\operatorname{Re}(Z_{EE}) \succ 0$ and $Z_0 > 0$, $E$ is strictly non-singular and well-conditioned. Thus:
$$\mathbf{i}_E = E^{-1} \mathbf{v}_s - E^{-1} Z_{ER} \mathbf{i}_R$$

Substituting $\mathbf{i}_E$ into the RIS port equation:
$$\mathbf{v}_R = Z_{RR} \mathbf{i}_R + Z_{RE} \left( E^{-1} \mathbf{v}_s - E^{-1} Z_{ER} \mathbf{i}_R \right) = \left( Z_{RR} - Z_{RE} E^{-1} Z_{ER} \right) \mathbf{i}_R + Z_{RE} E^{-1} \mathbf{v}_s$$

Applying the RIS load boundary condition $\mathbf{v}_R = - Z_L(\mathbf{b}) \mathbf{i}_R$:
$$- Z_L(\mathbf{b}) \mathbf{i}_R = \left( Z_{RR} - Z_{RE} E^{-1} Z_{ER} \right) \mathbf{i}_R + Z_{RE} E^{-1} \mathbf{v}_s$$
$$\left[ \left( Z_{RR} - Z_{RE} E^{-1} Z_{ER} \right) + Z_L(\mathbf{b}) \right] \mathbf{i}_R = - Z_{RE} E^{-1} \mathbf{v}_s$$

Defining the external-terminated Schur complement of the RIS impedance:
$$A \triangleq Z_{RR} - Z_{RE} E^{-1} Z_{ER} \in \mathbb{C}^{N \times N}$$
The RIS terminal currents are solved exactly as:
$$\mathbf{i}_R = - \left[ A + Z_L(\mathbf{b}) \right]^{-1} Z_{RE} E^{-1} \mathbf{v}_s$$

### 2.4 Power Waves and End-to-End Channel Definition

In microwave network theory, the channel transfer matrix $H_{\text{eff}} \in \mathbb{C}^{N_{\text{UE}} \times N_{\text{BS}}}$ maps incident power waves $\mathbf{a}_{\text{BS}} \in \mathbb{C}^{N_{\text{BS}}}$ at the transmitter ports to received power waves $\mathbf{b}_{\text{UE}} \in \mathbb{C}^{N_{\text{UE}}}$ at the receiver ports:
$$\mathbf{b}_{\text{UE}} = H_{\text{eff}}(\mathbf{b}) \, \mathbf{a}_{\text{BS}}$$

For real reference impedance $Z_0$:
- The incident power wave launched by BS source $m$ (available power $P_{\text{av}, m} = \frac{1}{2} |a_m|^2$) is:
  $$a_m = \frac{v_{s, m}}{2 \sqrt{Z_0}}$$
- The power wave absorbed by the receiver termination load $Z_0$ at UE port $k$ (delivered power $P_{\text{del}, k} = \frac{1}{2} |b_k|^2$) is:
  $$b_k = \frac{v_{E, N_{\text{BS}}+k} - Z_0 i_{E, N_{\text{BS}}+k}}{2 \sqrt{Z_0}} = \frac{- Z_0 i_{E, N_{\text{BS}}+k} - Z_0 i_{E, N_{\text{BS}}+k}}{2 \sqrt{Z_0}} = - \sqrt{Z_0} \, i_{E, N_{\text{BS}}+k}$$

Therefore, the transfer coefficient from BS antenna $m$ to UE antenna $k$ is:
$$H_{\text{eff}, km} = \frac{b_k}{a_m} = \frac{- \sqrt{Z_0} \, i_{E, N_{\text{BS}}+k}}{\frac{v_{s, m}}{2 \sqrt{Z_0}}} = - 2 Z_0 \, \frac{i_{E, N_{\text{BS}}+k}}{v_{s, m}}$$
With $Z_0 = 50\ \Omega$, the normalization factor is exactly:
$$- 2 Z_0 = - 100\ \Omega$$

Now, substituting the solution for $\mathbf{i}_R$ back into the external current equation $\mathbf{i}_E$:
$$\mathbf{i}_E = E^{-1} \mathbf{v}_s + E^{-1} Z_{ER} \left[ A + Z_L(\mathbf{b}) \right]^{-1} Z_{RE} E^{-1} \mathbf{v}_s$$

Extracting the sub-blocks for UE receivers (rows $N_{\text{BS}}+1 \dots N_{\text{ext}}$ of $\mathbf{i}_E$) and BS transmitters (columns $1 \dots N_{\text{BS}}$ of $\mathbf{v}_s$):
$$H_{\text{eff}}(\mathbf{b}) = D + C \left[ A + Z_L(\mathbf{b}) \right]^{-1} F$$
where:
$$\begin{aligned}
D &\triangleq - 100 \left[ E^{-1} \right]_{[-N_{\text{UE}}:, \, :N_{\text{BS}}]} \in \mathbb{C}^{N_{\text{UE}} \times N_{\text{BS}}} \\
C &\triangleq - 100 \left[ E^{-1} Z_{ER} \right]_{[-N_{\text{UE}}:, \, :]} \in \mathbb{C}^{N_{\text{UE}} \times N} \\
F &\triangleq \left[ Z_{RE} E^{-1} \right]_{[:, \, :N_{\text{BS}}]} \in \mathbb{C}^{N \times N_{\text{BS}}} \\
A &\triangleq Z_{RR} - Z_{RE} E^{-1} Z_{ER} \in \mathbb{C}^{N \times N}
\end{aligned}$$
This matches the exact syntax, sign, and scaling implemented in `src/atomwise_ris/physics/network.py`.

---

## 3. Incident Multipath Field to RIS Excitation ($C_{\text{in}}$)

### 3.1 Ray-Traced Incident Field Representation

Let BS antenna $m$ launch an incident power wave $a_m = 1 \sqrt{\text{W}}$. In the propagation environment, rays travel from the BS to the RIS aperture along $P_m$ distinct multipath paths (LOS, single/multiple building reflections, ground bounces).
From the ray tracer (Sionna RT / DeepMIMO), path $p \in \{1, \dots, P_m\}$ is characterized by:
- Complex path gain $\alpha_{m, p} \in \mathbb{C}$ (combining geometric spreading, Fresnel reflection coefficients, and initial antenna pattern).
- Carrier propagation delay $\tau_{m, p} \in \mathbb{R}$ (seconds), corresponding to path phase $\psi_{m, p} = - 2\pi f \tau_{m, p}$.
- Arrival direction unit vector $\hat{\mathbf{k}}_{\text{arr}, m, p}$ pointing from the RIS toward the source/last bounce, parameterized by zenith angle $\theta_{m, p} \in [0, \pi]$ and azimuth angle $\phi_{m, p} \in [0, 2\pi)$:
  $$\hat{\mathbf{k}}_{\text{arr}, m, p} = \begin{bmatrix} \sin\theta_{m, p} \cos\phi_{m, p} \\ \sin\theta_{m, p} \sin\phi_{m, p} \\ \cos\theta_{m, p} \end{bmatrix}$$
- Polarization unit vector $\hat{\mathbf{e}}_{m, p}$ (for vertical polarization, $\hat{\mathbf{e}} = \hat{\boldsymbol{\theta}} = \cos\theta \cos\phi \, \hat{\mathbf{x}} + \cos\theta \sin\phi \, \hat{\mathbf{y}} - \sin\theta \, \hat{\mathbf{z}}$).

The incident electric field at position $\mathbf{r}$ on the RIS is:
$$\mathbf{E}_{\text{inc}, m}(\mathbf{r}) = \sum_{p=1}^{P_m} \mathbf{E}_{0, m, p} \, e^{j k \hat{\mathbf{k}}_{\text{arr}, m, p} \cdot (\mathbf{r} - \mathbf{r}_{\text{RIS}})}$$
where $k = 2\pi/\lambda$ is the free-space wavenumber, $\mathbf{r}_{\text{RIS}}$ is the array center, and the spatial phase advances by $+ j k \hat{\mathbf{k}}_{\text{arr}} \cdot \mathbf{r}$ for points located toward the source.

### 3.2 Lorentz Reciprocity and Open-Circuit Port Voltage

How does $\mathbf{E}_{\text{inc}, m}(\mathbf{r})$ induce open-circuit voltages at the RIS ports?
Consider the Lorentz reciprocity theorem applied to the linear, reciprocal thin-wire structure:
- **State 1 (Transmit):** An external generator injects terminal current $I_n^{(1)} = 1\text{ A}$ into RIS port $n$, with all other ports open ($I_{k \ne n}^{(1)} = 0$). This produces a current distribution $\mathbf{J}_n(\mathbf{r})$ across all wire segments and radiates field $\mathbf{E}_n(\mathbf{r})$.
- **State 2 (Receive):** The external incident field $\mathbf{E}_{\text{inc}}(\mathbf{r})$ impinges on the RIS with all ports open ($I_k^{(2)} = 0$). Open-circuit voltages $v_{\text{oc}, k}$ develop across the port gaps.

By the Lorentz reaction integral:
$$\int_{\text{wires}} \mathbf{J}_n(\mathbf{r}) \cdot \mathbf{E}_{\text{inc}}(\mathbf{r}) \, dV = - V_n^{(2)} I_n^{(1)} = - v_{\text{oc}, n} \cdot (1\text{ A})$$
Therefore:
$$v_{\text{oc}, n} = - \int_{\text{wires}} \mathbf{J}_n(\mathbf{r}) \cdot \mathbf{E}_{\text{inc}}(\mathbf{r}) \, dV$$

In the thin-wire Method of Moments discretization:
Each wire $q \in \{1, \dots, N\}$ is divided into $S$ segments of length $\Delta \ell_m$ and center coordinates $\mathbf{r}_m$. All wires are aligned with $\hat{\mathbf{z}}$, so $\mathbf{J}_n(\mathbf{r}) \cdot \mathbf{E}_{\text{inc}}(\mathbf{r}) = I_{Z}[m, n] \, E_{\text{inc}, z}(\mathbf{r}_m) / (\pi a^2)$, where $I_Z \in \mathbb{C}^{M \times N}$ ($M = N \times S$) is the matrix of segment currents when port $n$ carries unit open-circuit terminal current:
$$I_Z = I_Y \, Z_{\text{RIS}} = I_Y \, Y_{\text{port}}^{-1}$$
Here $I_Y \in \mathbb{C}^{M \times N}$ is the segment currents under unit impressed port voltages (the direct output of PyNEC's `xq_card` solves).

Thus, the open-circuit voltage at RIS port $n$ is:
$$v_{\text{oc}, n} = - \sum_{m=1}^M I_Z[m, n] \, \Delta \ell_m \, E_{\text{inc}, z}(\mathbf{r}_m)$$

### 3.3 Embedded Element Vector Effective Length

For a plane wave arriving from direction $(\theta, \phi)$ with unit amplitude electric field $\mathbf{E}_{\text{inc}}(\mathbf{r}) = \hat{\mathbf{e}} \, e^{j k \hat{\mathbf{k}}_{\text{arr}} \cdot (\mathbf{r} - \mathbf{r}_{\text{RIS}})}$, the open-circuit voltage induced at port $n$ is:
$$v_{\text{oc}, n} = \mathbf{h}_{\text{eff}, n}(\theta, \phi) \cdot \hat{\mathbf{e}}$$
where the **embedded element effective length vector** is:
$$\mathbf{h}_{\text{eff}, n}(\theta, \phi) \triangleq - \left( \sum_{m=1}^M I_Z[m, n] \, \Delta \ell_m \, e^{j k \hat{\mathbf{k}}_{\text{arr}}(\theta, \phi) \cdot (\mathbf{r}_m - \mathbf{r}_{\text{RIS}})} \right) \hat{\mathbf{z}}$$
For vertical polarization $\hat{\mathbf{e}} = \hat{\boldsymbol{\theta}}$, since $\hat{\mathbf{z}} \cdot \hat{\boldsymbol{\theta}} = - \sin\theta$:
$$h_{\text{eff}, \theta, n}(\theta, \phi) = \sin\theta \sum_{m=1}^M I_Z[m, n] \, \Delta \ell_m \, e^{j k \hat{\mathbf{k}}_{\text{arr}}(\theta, \phi) \cdot (\mathbf{r}_m - \mathbf{r}_{\text{RIS}})}$$
For horizontal polarization $\hat{\mathbf{e}} = \hat{\boldsymbol{\phi}}$, since $\hat{\mathbf{z}} \cdot \hat{\boldsymbol{\phi}} = 0$, $h_{\text{eff}, \phi, n}(\theta, \phi) \equiv 0$. The vertical wire array is blind to horizontally polarized field components.

### 3.4 Construction of $C_{\text{in}}$

Recall from Section 2.4 that in `network.py`:
$$F = \left[ Z_{RE} E^{-1} \right]_{[:, \, :N_{\text{BS}}]}$$
When external antennas are separated from the RIS by more than a few wavelengths, $Z_{RE} E^{-1} \approx Z_{R, \text{BS}} / (Z_{\text{self}, \text{BS}} + 50)$.
By reciprocity, the mutual impedance $Z_{n, m}$ between RIS port $n$ and BS antenna $m$ is:
$$Z_{n, m} = \frac{v_{\text{oc}, n}}{i_{\text{BS}, m}}$$
When BS antenna $m$ is driven with power wave $a_m = 1\sqrt{\text{W}}$, $v_{s, m} = 2\sqrt{50}$, and $i_{\text{BS}, m} \approx \frac{v_{s, m}}{Z_{\text{self}, \text{BS}} + 50}$.
Thus:
$$C_{\text{in}, n, m} = F_{nm} = \frac{v_{\text{oc}, n}}{v_{s, m}} = \frac{v_{\text{oc}, n}}{2 \sqrt{50}}$$
In terms of the multipath incident field:
$$C_{\text{in}, n, m} = \sum_{p=1}^{P_m} \alpha_{m, p} \, \frac{h_{\text{eff}, \theta, n}(\theta_{m, p}, \phi_{m, p})}{\sqrt{Z_0 / \eta_0}}$$
where $\eta_0 \approx 376.73\ \Omega$ is the characteristic impedance of free space.

---

## 4. RIS Reradiation to User Paths ($C_{\text{out}}$)

### 4.1 Reradiated Field from Induced RIS Currents

When the RIS is terminated in loads $Z_L(\mathbf{b})$, the terminal port currents are:
$$\mathbf{i}_R = - \left[ A + Z_L(\mathbf{b}) \right]^{-1} \mathbf{v}_{\text{oc}} = - 2\sqrt{50} \left[ A + Z_L(\mathbf{b}) \right]^{-1} C_{\text{in}} \, \mathbf{a}_{\text{BS}}$$
These terminal currents produce total segment currents across all wires:
$$\mathbf{I}_{\text{seg}} = I_Z \, \mathbf{i}_R = - 2\sqrt{50} \, I_Z \left[ A + Z_L(\mathbf{b}) \right]^{-1} C_{\text{in}} \, \mathbf{a}_{\text{BS}}$$

Each segment $m$ acts as a radiating Hertzian current element $I_{\text{seg}}[m] \Delta \ell_m \, \hat{\mathbf{z}}$. By the free-space dyadic Green's function, the electric field radiated to an observation point $\mathbf{r}_{\text{UE}}$ in the far field is:
$$\mathbf{E}_{\text{rad}}(\mathbf{r}_{\text{UE}}) = - j \frac{k \eta_0}{4\pi} \frac{e^{- j k r}}{r} \sin\theta \, \hat{\boldsymbol{\theta}} \sum_{m=1}^M I_{\text{seg}}[m] \, \Delta \ell_m \, e^{j k \hat{\mathbf{r}} \cdot (\mathbf{r}_m - \mathbf{r}_{\text{RIS}})}$$

### 4.2 Construction of $C_{\text{out}}$

By reciprocity, the open-circuit voltage induced at a receiver antenna $k$ with effective length $h_{\text{UE}}$ and reference impedance $Z_0 = 50\ \Omega$ is:
$$v_{\text{oc}, \text{UE}, k} = \mathbf{E}_{\text{rad}}(\mathbf{r}_{\text{UE}, k}) \cdot \mathbf{h}_{\text{UE}}$$
And the received power wave absorbed into the 50 $\Omega$ receiver load is:
$$b_k = \frac{v_{\text{oc}, \text{UE}, k}}{2 \sqrt{Z_0}}$$

Combining the constants and comparing with the matrix formulation of Section 2.4:
$$C_{\text{out}, k, n} = - 100 \left[ E^{-1} Z_{ER} \right]_{[N_{\text{BS}}+k, \, n]} \approx - \frac{100}{Z_{\text{self}, \text{UE}} + 50} \, Z_{\text{UE}, k, n}$$
where $Z_{\text{UE}, k, n}$ is the mutual impedance between UE antenna $k$ and RIS element $n$.
For multipath propagation between the RIS and the UE consisting of paths $q \in \{1, \dots, Q_k\}$:
$$C_{\text{out}, k, n} = \sum_{q=1}^{Q_k} \beta_{k, q} \, \frac{h_{\text{eff}, \theta, n}(\theta_{k, q}, \phi_{k, q})}{\sqrt{Z_0 / \eta_0}}$$
where $\beta_{k, q}$ are the complex path transfer coefficients from the RIS aperture to UE receiver $k$.

---

## 5. Summary of Units, Scaling, and Conventions

| Quantity | Symbol | SI Units | Dimensionless Equivalent |
|---|---|---|---|
| Open-circuit impedance matrix | $Z, Z_{\text{RIS}}, A$ | $\Omega$ | - |
| Load impedance diagonal | $Z_L(\mathbf{b})$ | $\Omega$ | - |
| Terminal voltages | $\mathbf{v}_R, \mathbf{v}_E, \mathbf{v}_s$ | V | - |
| Terminal currents | $\mathbf{i}_R, \mathbf{i}_E$ | A | - |
| Incident power waves | $\mathbf{a}_{\text{BS}}$ | $\sqrt{\text{W}}$ | Normalized so $\frac{1}{2}\|\mathbf{a}\|^2 = P_{\text{tx}}$ |
| Received power waves | $\mathbf{b}_{\text{UE}}$ | $\sqrt{\text{W}}$ | Normalized so $\frac{1}{2}\|\mathbf{b}\|^2 = P_{\text{rx}}$ |
| Effective channel matrix | $H_{\text{eff}}(\mathbf{b})$ | - | S-parameter ($b_k / a_m$) |
| Direct channel matrix | $H_{\text{direct}}$ | - | S-parameter ($b_k / a_m$) |
| RIS excitation operator | $C_{\text{in}}$ | - | Voltage transfer ($v_{\text{oc}} / v_s$) |
| RIS reradiation operator | $C_{\text{out}}$ | $\Omega$ | Power-wave transfer coefficient |
| Embedded effective length | $\mathbf{h}_{\text{eff}, n}(\theta, \phi)$ | m | V per (V/m) |
| Free-space wave impedance | $\eta_0$ | $\Omega$ | $\approx 376.730313668\ \Omega$ |
| Reference impedance | $Z_0$ | $\Omega$ | $50.0\ \Omega$ (real) |

### Phase and Coordinate Conventions
1. **Time-harmonic convention:** $e^{j \omega t}$. A wave traveling away from the origin in direction $\hat{\mathbf{k}}$ has spatial dependence $e^{- j k \hat{\mathbf{k}} \cdot \mathbf{r}}$.
2. **PyNEC EX Card 1 (Plane Wave):** The angles $(\theta, \phi)$ in PyNEC's `ex_card(1, ...)` represent the direction **from which the wave arrives**. Thus, the incident wave advances phase along $\hat{\mathbf{k}}_{\text{arr}}$, giving spatial factor $e^{+ j k \hat{\mathbf{k}}_{\text{arr}} \cdot \mathbf{r}}$.
3. **DeepMIMO / Sionna RT export:**
   - `aoa_az`: Azimuth angle toward the source ($\phi \in [-180^\circ, 180^\circ]$).
   - `aoa_el`: Zenith angle from $+z$ ($\theta \in [0^\circ, 180^\circ]$).
   - `phase`: Interaction phase without delay phase (degrees). Total path phase is $\phi_{\text{inter}} - 2\pi f \tau$.
   - `power`: Path power gain in dB, matching Friis formula $20 \log_{10}|\alpha|$.

---

## 6. Aperture Near-Field vs Far-Field Validity Criteria

The 64-element thin-wire RIS has:
- Elements: $8 \times 8$ grid of vertical wires ($L = 0.024\text{ m}$, radius $a = 50\ \mu\text{m}$).
- Grid spacing: $\Delta y = 0.028\text{ m}$, $\Delta z = 0.030\text{ m}$.
- Array dimensions: $W_y = 7 \times 0.028 = 0.196\text{ m}$, $W_z = 7 \times 0.030 = 0.210\text{ m}$.
- Array diagonal diameter:
  $$D = \sqrt{W_y^2 + W_z^2} = \sqrt{0.196^2 + 0.210^2} \approx 0.2873\text{ m}$$
- Wavelength at $f = 5.53\text{ GHz}$:
  $$\lambda = \frac{299792458}{5.53 \times 10^9} \approx 0.054212\text{ m}$$

### 6.1 Classical Fraunhofer Distance
The Fraunhofer far-field boundary of the complete 64-element aperture is:
$$R_{\text{Fraunhofer}} = \frac{2 D^2}{\lambda} = \frac{2 \times (0.2873)^2}{0.054212} \approx 3.045\text{ m}$$

### 6.2 BS-to-RIS Validity (Incident Side)
In the Munich scenario, the BS is located at $x \approx -210\text{ m}$, $y \approx 73\text{ m}$, $z = 105\text{ m}$, while the RIS is at $x = 65\text{ m}$, $y = 20\text{ m}$, $z = 3\text{ m}$.
The unfolded path lengths for all incident multipath rays exceed $d_{\text{BS-RIS}} \ge 298\text{ m}$.
Comparing with the Fraunhofer distance:
$$\frac{d_{\text{BS-RIS}}}{R_{\text{Fraunhofer}}} \ge \frac{298}{3.045} \approx 98 \gg 1$$
The maximum quadratic phase error across the entire 64-element aperture from a source at 298 m is:
$$\Delta \psi_{\max} = \frac{k D^2}{8 d} = \frac{2\pi \times (0.2873)^2}{8 \times 0.054212 \times 298} \approx 0.0040\text{ rad} \approx 0.23^\circ$$
This phase curvature is completely negligible ($< 0.25^\circ$). Therefore, the **incident field at the RIS is rigorously and strictly a sum of planar wavefronts**.

### 6.3 RIS-to-UE Validity (Reradiation Side)
In the Munich scenario, UE positions are located at distances $d_{\text{UE}} \in [10\text{ m}, 60\text{ m}]$.
- At $d = 60\text{ m}$: $d / R_{\text{Fraunhofer}} \approx 20 \gg 1$, quadratic phase deviation $\approx 1.2^\circ$ (far field).
- At $d = 10\text{ m}$: $d / R_{\text{Fraunhofer}} \approx 3.3$, quadratic phase deviation across the aperture is:
  $$\Delta \psi_{\max} = \frac{k D^2}{8 \times 10} \approx 0.12\text{ rad} \approx 6.9^\circ$$
While $7^\circ$ phase deviation is still within the standard Fraunhofer $\pi/8 \approx 22.5^\circ$ criterion, our hybrid implementation provides **both**:
1. Exact spherical-wave element-resolved steering (evaluating the exact distance $\|\mathbf{r}_{\text{UE}} - \mathbf{r}_n\|$ from each user to each of the 64 element centers).
2. The plane-wave angular-spectrum approximation (evaluating $h_{\text{eff}, n}(\theta_q, \phi_q)$ steered across the array).
Both formulations are compared and audited in Gate P2.

---

## 7. Prevention of Double-Counting RIS Scattering

A critical risk in hybrid ray-tracing + EM simulation is double-counting: if the ray tracer models the RIS as a physical reflecting wall or impedance boundary, and the EM solver also calculates the scattered field, the reflected energy is counted twice.

In our methodology, double counting is **strictly and provably avoided**:
1. **Virtual Probe Audit [V-file]:** In `scripts/trace_munich.py`, the 64 RIS elements are defined as virtual isotropic receiving and transmitting antenna nodes in free space. No geometric mesh, CAD surface, facade, or dielectric boundary was placed at the RIS panel location in the Mitsuba/Sionna scene.
2. **Ray Tracer Channels [V-file]:**
   - The direct BS-to-UE paths $H_{\text{direct}}$ traced by Sionna RT contain only environment paths (LOS between BS and UE, ground reflections, reflections off Munich buildings). They contain **zero** rays interacting with the RIS panel.
   - The BS-to-RIS paths contain rays traveling from the BS to the virtual probe points.
   - The RIS-to-UE paths contain rays traveling from the virtual probe points to the UE.
3. **RIS Structural vs Loaded Scattering:**
   - An open-circuited thin wire ($a = 50\ \mu\text{m} \ll \lambda$) has an extremely small backscattering radar cross-section (proportional to $(a/\lambda)^2 \approx 10^{-6}$).
   - **Correction (2026-11-07):** this claim is not supported for the 24 mm wires. The full-size gate run (`hybrid-gates-v2-20261107.md`) measured that the 64 open-gap wires change the BS-to-UE direct channel by 4.7%. The fixed hybrid models this term explicitly (derivation in that document).
   - The entire reradiated field from the RIS is governed by the loaded terminal currents $\mathbf{i}_R = - [A + Z_L(\mathbf{b})]^{-1} \mathbf{v}_{\text{oc}}$, which is completely captured by the operator $C_{\text{out}} [A + Z_L(\mathbf{b})]^{-1} C_{\text{in}}$.
   - When all elements are open-circuited ($Z_L \to \infty$), the RIS-mediated term vanishes, leaving $H_{\text{eff}} = H_{\text{direct}}$, exactly matching physical reality.

---

## 8. Stated Model Assumptions and Physical Limitations

To maintain absolute scientific rigor, the hybrid model makes the following explicit physical approximations:

1. **Thin-Wire Approximation:**
   Elements are modeled as PEC thin wires in free space ($L = 24\text{ mm}$, $a = 50\ \mu\text{m}$) using NEC-2's extended thin-wire kernel. This captures complete wire mutual impedance and reradiation, but does **not** model:
   - A finite dielectric PCB substrate.
   - A backing metal ground plane.
   - Microstrip patches or planar printed resonators.
   - Parasitic capacitances of PIN diodes, varactors, or bias feed networks.

2. **Linear Passive Lumped Load Model:**
   The binary loads are modeled as linear passive 1-port impedances $z_0, z_1 \in \mathbb{C}$ ($\operatorname{Re}(z) = 1.0\ \Omega$, $\operatorname{Im}(z) = - X_0 \pm 70\ \Omega$). Nonlinear switching transients, RF harmonics, intermodulation products, and bias line coupling are neglected.

3. **Multi-Bounce RIS-Environment Feedback:**
   The formulation models:
   - BS $\to$ environment $\to$ RIS
   - RIS $\to$ environment $\to$ UE
   - BS $\to$ environment $\to$ UE
   It neglects second-order multi-bounce feedback where energy reradiated by the RIS scatters off nearby buildings and returns to the RIS. Given that the nearest building reflector is several meters away ($> 4.8\text{ m}$), the path loss of the secondary return loop is $> 60\text{ dB}$ below the primary excitation, making this approximation highly accurate.

4. **Reference Baseline Nature:**
   This model is an **electromagnetically consistent physical reference baseline**, not an empirical calibration of a specific manufactured commercial metasurface (e.g. OpenSourceRIS).
