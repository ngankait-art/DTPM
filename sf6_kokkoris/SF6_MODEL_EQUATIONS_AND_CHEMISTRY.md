# SF6 Global Plasma Model — Complete Mathematical and Chemical Documentation

**Paper:** Kokkoris et al., J. Phys. D: Appl. Phys. 42 (2009) 055209  
**Implementation:** Python (NumPy + SciPy)

---

## 1. Model Overview

This is a zero-dimensional (global, volume-averaged) model for an SF6 inductively coupled plasma. It solves 18 coupled ordinary differential equations for 13 gas-phase species densities, 1 electron temperature, and 4 surface coverages. The model is time-integrated from pre-discharge initial conditions to steady state using the BDF (Backward Differentiation Formula) method.

---

## 2. Species

### 2.1 Gas-Phase Species (13)

| Index | Species | Type | Mass (amu) |
|-------|---------|------|------------|
| 0 | SF6 | Parent gas (neutral) | 146 |
| 1 | SF5 | Neutral radical | 127 |
| 2 | SF4 | Neutral radical | 108 |
| 3 | SF3 | Neutral radical | 89 |
| 4 | F2 | Neutral molecule | 38 |
| 5 | F | Neutral atom | 19 |
| 6 | SF5+ | Positive ion | 127 |
| 7 | SF4+ | Positive ion | 108 |
| 8 | SF3+ | Positive ion | 89 |
| 9 | F2+ | Positive ion | 38 |
| 10 | SF6- | Negative ion | 146 |
| 11 | F- | Negative ion | 19 |
| 12 | e- | Electron | 5.486e-4 |

### 2.2 Surface Species (4 coverage fractions)

| Symbol | Description |
|--------|-------------|
| theta_F | Fraction of wall surface covered by adsorbed F |
| theta_SF3 | Fraction covered by adsorbed SF3 |
| theta_SF4 | Fraction covered by adsorbed SF4 |
| theta_SF5 | Fraction covered by adsorbed SF5 |
| theta_bare | = 1 - theta_F - theta_SF3 - theta_SF4 - theta_SF5 |

---

## 3. Reactor Geometry

The reactor is modeled as a single equivalent cylinder combining the ICP tube and the diffusion chamber:

| Parameter | Symbol | Value | Formula |
|-----------|--------|-------|---------|
| Radius | R | 0.19 m | Paper Section 2 |
| Length | L | 0.38 m | 0.21 + 0.17 m |
| Volume | V | 0.04309 m³ | pi × R² × L |
| Axial wall area | A_ax | 0.2268 m² | 2 × pi × R² |
| Radial wall area | A_rad | 0.4536 m² | 2 × pi × R × L |
| Total wall area | A | 0.6804 m² | A_ax + A_rad |
| Diffusion length | Lambda | 0.0661 m | [(pi/L)² + (2.405/R)²]^(-1/2) |

---

## 4. Gas-Phase Reactions (50 Reactions)

### 4.1 Electron-Impact Rate Coefficient Formula

All electron-impact rate coefficients use the parametric form with Druyvesteyn EEDF parameters:

    k(Te) = exp(A + B × ln(Te) + C/Te + D/Te² + E/Te³)    [m³/s]

where Te is the electron temperature in eV.

### 4.2 Neutral Dissociations (G1–G7)

These reactions break SF6 and its fragments into smaller species by electron impact.

| Index | Reaction | Threshold (eV) | A | B | C | D | E |
|-------|----------|---------------|---|---|---|---|---|
| G1 | SF6 + e → SF5 + F + e | 9.6 | -29.35 | -0.2379 | -14.11 | -15.25 | -1.204 |
| G2 | SF6 + e → SF4 + 2F + e | 12.1 | -31.61 | -0.2592 | -10.00 | -31.24 | -0.7126 |
| G3 | SF6 + e → SF3 + 3F + e | 16.0 | -40.26 | 3.135 | 5.895 | -64.68 | 0.2607 |
| G4 | SF5 + e → SF4 + F + e | 9.6 | -29.36 | -0.2379 | -14.11 | -15.25 | -1.204 |
| G5 | SF4 + e → SF3 + F + e | 9.6 | -29.36 | -0.2379 | -14.11 | -15.25 | -1.204 |
| G6 | F2 + e → 2F + e | 3.16 | -31.44 | -0.6986 | -5.170 | -1.389 | -0.0650 |
| G7 | F2 + e → 2F + e | 4.34 | -33.44 | -0.2761 | -3.564 | -3.946 | -0.0393 |

G4 and G5 use the same cross section as G1 (assumption from paper).

### 4.3 Ionizations (G8–G16)

These reactions produce positive ions and free electrons.

| Index | Reaction | Threshold (eV) | A | B | C | D | E |
|-------|----------|---------------|---|---|---|---|---|
| G8 | SF6 + e → SF5+ + F + 2e | 15.5 | -33.66 | 1.212 | -4.594 | -56.66 | -0.3226 |
| G9 | SF6 + e → SF4+ + 2F + 2e | 18.5 | -37.14 | 1.515 | -4.829 | -80.42 | -0.7924 |
| G10 | SF6 + e → SF3+ + 3F + 2e | 20.0 | -36.82 | 1.740 | -0.1047 | -98.18 | 0.1060 |
| G11 | SF5 + e → SF5+ + 2e | 11.2 | -34.92 | 1.487 | -2.377 | -29.71 | -0.1449 |
| G12 | SF5 + e → SF4+ + F + 2e | 14.5 | -36.27 | 1.892 | -1.387 | -50.87 | -0.0758 |
| G13 | SF4 + e → SF4+ + 2e | 13.0 | -32.95 | 0.8763 | -10.19 | -31.21 | -3.989 |
| G14 | SF4 + e → SF3+ + F + 2e | 14.5 | -32.75 | 0.8222 | -10.82 | -40.59 | -4.274 |
| G15 | SF3 + e → SF3+ + 2e | 11.0 | -35.55 | 1.750 | -2.086 | -28.70 | -0.1357 |
| G16 | F2 + e → F2+ + 2e | 15.69 | -35.60 | 1.467 | -6.140 | -57.14 | -0.4860 |

### 4.4 Attachments (G17–G19)

These reactions produce negative ions by capturing an electron.

| Index | Reaction | Threshold (eV) | A | B | C | D | E |
|-------|----------|---------------|---|---|---|---|---|
| G17 | SF6 + e → SF5 + F- | 0.1 | -33.43 | -1.173 | -0.5614 | 0.1798 | -0.0145 |
| G18 | SF6 + e → SF6- | 0.0 | -33.46 | -1.500 | 0.0002 | -0.0023 | 0.0 |
| G19 | F2 + e → F + F- | 0.0 | -33.31 | -1.487 | -0.2795 | 0.0109 | -0.0004 |

### 4.5 Detachments (G20–G21)

These reactions destroy negative ions by liberating the captured electron. N represents any neutral species. The rate coefficient is constant (not electron-temperature dependent).

| Index | Reaction | k (m³/s) |
|-------|----------|----------|
| G20 | F- + N → F + N + e | exp(-44.39) = 4.77e-20 |
| G21 | SF6- + N → SF6 + N + e | exp(-44.98) = 2.58e-20 |

### 4.6 Momentum Transfer / Elastic Collisions (G22–G27)

These are elastic collisions where electrons lose a small fraction of their kinetic energy. They contribute to the power balance through the energy loss term 3(m_e/M) × Te per collision.

| Index | Reaction | A | B | C | D | E |
|-------|----------|---|---|---|---|---|
| G22 | SF6 + e → SF6 + e | -29.15 | 0.2126 | -1.455 | 0.2456 | -0.0141 |
| G23–G25 | SFx + e → SFx + e (x=3,4,5) | Same as G22 | | | | |
| G26 | F2 + e → F2 + e | -29.04 | -0.0987 | -0.4897 | -0.0319 | 0.0055 |
| G27 | F + e → F + e | Same as G26 | | | | |

### 4.7 Excitations (G28–G34)

Vibrational and electronic excitations that do not produce new species but consume electron energy.

| Index | Reaction | Threshold (eV) | Description |
|-------|----------|---------------|-------------|
| G28 | SF6 + e → SF6(v) + e | 0.09 | Total vibrational excitation of SF6 |
| G29 | F2 + e → F2(v) + e | 0.1108 | Vibrational excitation of F2 |
| G30 | F2 + e → F2(v) + e | 0.2188 | Vibrational excitation of F2 |
| G31 | F2 + e → F2(v) + e | 0.3237 | Vibrational excitation of F2 |
| G32 | F2 + e → F2(v) + e | 0.4205 | Vibrational excitation of F2 |
| G33 | F2 + e → F2* + e | 11.57 | Electronic excitation of F2 |
| G34 | F2 + e → F2* + e | 13.08 | Electronic excitation of F2 |

### 4.8 Neutral Recombinations (G35–G41)

These reactions reform larger molecules from fragments. G35–G37 use the **Troe fall-off formula** from Ryan & Plumb (1990), which makes them pressure-dependent. G38–G40 are constant. G41 is constant.

**Troe fall-off formula (for G35–G37):**

    k(p) = [k0 × [M] / (1 + k0 × [M] / kinf)] × Fc^{(1 + [log10(k0 × [M] / kinf)]²)^(-1)}

where [M] is the total gas number density (m⁻³ converted to cm⁻³ for the formula), and the parameters are:

| Index | Reaction | k0 (cm⁶/s) | kinf (cm³/s) | Fc | Regime at ~1 Pa |
|-------|----------|-----------|-------------|------|-----------------|
| G35 | F + SF5 → SF6 | 3.4e-23 | 1e-11 | 0.43 | High-pressure limit |
| G36 | F + SF4 → SF5 | 3.7e-28 | 5e-12 | 0.46 | Low-pressure limit |
| G37 | F + SF3 → SF4 | 2.8e-26 | 2e-11 | 0.47 | Fall-off region |

Constant rate coefficients:

| Index | Reaction | k (m³/s) |
|-------|----------|----------|
| G38 | F2 + SF5 → SF6 + F | exp(-46.41) = 6.82e-21 |
| G39 | F2 + SF4 → SF5 + F | exp(-46.41) = 6.82e-21 |
| G40 | F2 + SF3 → SF4 + F | exp(-46.41) = 6.82e-21 |
| G41 | SF5 + SF5 → SF6 + SF4 | exp(-41.50) = 9.48e-19 |

### 4.9 Ion–Ion Recombinations (G42–G49)

All positive–negative ion pairs recombine with the same rate coefficient.

| Index | Reaction | k (m³/s) |
|-------|----------|----------|
| G42–G49 | I+ + J- → I + J | exp(-29.93) = 1.10e-13 |

where I = SF5, SF4, SF3, F2 and J = SF6, F.

### 4.10 Ion–Molecule Reaction (G50)

| Index | Reaction | k (m³/s) |
|-------|----------|----------|
| G50 | SF6 + SF5+ → SF6 + SF3+ + F2 | exp(-39.65) = 5.42e-18 |

---

## 5. Surface Reactions (40 Reactions)

### 5.1 Adsorption/Sticking (S1–S4)

Neutral species adsorb on bare wall surface sites.

| Index | Reaction | Probability |
|-------|----------|-------------|
| S1 | F + bare → F(s) | 0.150 |
| S2 | SF3 + bare → SF3(s) | 0.080 |
| S3 | SF4 + bare → SF4(s) | 0.080 |
| S4 | SF5 + bare → SF5(s) | 0.080 |

### 5.2 Fluorination of Adsorbed SFx (S5–S7)

Gas-phase F atoms react with adsorbed SFx to build up larger species. Only SF6, being fully saturated, is released to the gas phase.

| Index | Reaction | Probability |
|-------|----------|-------------|
| S5 | F + SF3(s) → SF4(s) | 0.500 |
| S6 | F + SF4(s) → SF5(s) | 0.200 |
| S7 | F + SF5(s) → SF6(gas) | 0.025 |

### 5.3 Surface Recombinations (S8–S11)

Gas-phase species react with adsorbed F to produce gas-phase molecules.

| Index | Reaction | Probability |
|-------|----------|-------------|
| S8 | F + F(s) → F2(gas) + bare | 0.500 |
| S9 | SF3 + F(s) → SF4(gas) + bare | 1.000 |
| S10 | SF4 + F(s) → SF5(gas) + bare | 1.000 |
| S11 | SF5 + F(s) → SF6(gas) + bare | 1.000 |

### 5.4 Ion–Surface Interactions (S12–S31)

Positive ions arrive at the wall with the Bohm flux. Their fate depends on the surface coverage at the impact site.

**On bare surface (S12–S15):** Ions adsorb directly.

    SFx+ + bare → SFx(s)
    F2+ + bare → 2F(s)

**On F-covered surface (S16–S19):** Ions adsorb and sputter F back to gas phase.

    SFx+ + F(s) → SFx(s) + F(gas)
    F2+ + F(s) → 2F(s) + F(gas)

**On SFy-covered surface (S20–S31):** Ions adsorb and sputter SFy back to gas phase.

    SFx+ + SFy(s) → SFx(s) + SFy(gas)
    F2+ + SFx(s) → 2F(s) + SFx(gas)

The branching among these pathways is proportional to the surface coverages (theta_bare, theta_F, theta_SFx).

### 5.5 Deposition (S32–S40)

Gas-phase SFx radicals deposit on wall sites already occupied by SFy, forming a permanent fluoro-sulfur film P(s).

| Index | Reaction | Probability |
|-------|----------|-------------|
| S32–S40 | SFx + SFy(s) → SFx(s) + P(s) | 0.030 |

where x, y each range over {3, 4, 5}, giving 9 reactions.

---

## 6. Governing Equations

### 6.1 Neutral Species Mass Balance

For each neutral species i:

    dn_i/dt = (feed rate) - (pump rate) + Sum(gas-phase production) - Sum(gas-phase loss) + Sum(surface production) - Sum(surface loss)

**Feed term** (SF6 only):

    Feed = Q_SF6 / V

where Q_SF6 = 100 sccm converted to molecules/s.

**Pump term** (all neutrals):

    Pump = k_pump × n_i

    k_pump = Q_total / (n0 × V)

where n0 = p_OFF / (kB × T_gas) is the pre-discharge total density, and Q_total includes both SF6 and Ar flow.

**Example: SF6 balance**

    dn(SF6)/dt = Q_SF6/V - k_pump × n(SF6)
                + kG35 × n(F) × n(SF5)           [G35: F+SF5→SF6]
                + k_G38 × n(F2) × n(SF5)          [G38: F2+SF5→SF6+F]
                + k_G41 × n(SF5)²                  [G41: SF5+SF5→SF6+SF4]
                + RS7                               [S7: F+SF5(s)→SF6]
                + RS11                              [S11: SF5+F(s)→SF6]
                + k_detach × n(SF6-) × n_neutral    [G21: SF6- detachment]
                + k_ii × n(SF6-) × n_pos            [ion-ion recombination]
                - (k1+k2+k3+k8+k9+k10+k17+k18) × ne × n(SF6)   [all electron-impact losses]

### 6.2 Positive Ion Mass Balance

For each positive ion i:

    dn_i+/dt = Sum(ionization production) - (ion wall loss) - (ion-ion recombination) - (ion-molecule reactions) - (pump)

**Example: SF5+ balance**

    dn(SF5+)/dt = k8 × ne × n(SF6) + k11 × ne × n(SF5)     [ionization sources]
                - k_ii × n(SF5+) × n_neg                     [ion-ion recombination]
                - w_SF5+ × n(SF5+)                            [wall loss]
                - k_G50 × n(SF6) × n(SF5+)                   [ion-molecule G50]
                - k_pump × n(SF5+)

### 6.3 Negative Ion Mass Balance

Negative ions are NOT lost to walls (confined by sheath potential) and NOT pumped.

    dn(F-)/dt = k17 × ne × n(SF6) + k19 × ne × n(F2)     [attachment production]
              - k_detach × n(F-) × n_neutral                [detachment loss]
              - k_ii × n(F-) × n_pos                        [ion-ion recombination]

### 6.4 Electron Density (Charge Neutrality)

The electron density is not solved independently. It is enforced by charge neutrality at every timestep:

    ne = n(SF5+) + n(SF4+) + n(SF3+) + n(F2+) - n(SF6-) - n(F-)

    dne/dt = Sum(dn_i+/dt) - Sum(dn_j-/dt)

### 6.5 Electron Energy (Power) Balance

    d(3/2 × ne × Te)/dt = eta × P_abs / (V × e) - ne × P_coll - P_wall - P_pump

Using the product rule, this becomes an ODE for Te:

    dTe/dt = (2/(3×ne)) × [eta × P_abs/(V×e) - ne × P_coll - P_wall - P_pump] - (Te/ne) × dne/dt

**Power coupling efficiency:**

    eta = 0.50

This accounts for power losses in the matching network, sheaths, and electrode. The paper assumes eta = 1 but physically it is less.

**Collisional power loss (per electron):**

    P_coll = Sum_j [k_j(Te) × n_target_j × epsilon_j]

where for inelastic reactions, epsilon_j is the threshold energy (eV), and for elastic collisions:

    epsilon_elastic = 3 × (m_e / M_neutral) × Te

**Wall power loss:**

    P_wall = Sum_ions [Gamma_ion_wall × (0.5×Te + V_s)] + Gamma_ion_total × 2×Te

where the first term is the ion kinetic energy lost per ion reaching the wall, and the second is the electron energy carried to the wall by ambipolar coupling. The sheath voltage for each ion species is:

    V_s = (Te/2) × ln(M_ion / (2 × pi × m_e))

For SF5+ (127 amu): V_s ≈ 5.26 × Te.

**Pumping energy loss:**

    P_pump = k_pump × (3/2 × ne × Te + 3/2 × n_pos_total × Ti)

where Ti is the ion temperature from the Lee & Lieberman formula (Section 7.3).

### 6.6 Surface Coverage Balance

Each surface coverage evolves as:

    d(theta_X)/dt = [production flux - loss flux] × V / (A × n_sites)

where n_sites = 10¹⁹ m⁻² is the assumed surface site density.

**Example: theta_F balance**

    Production: F adsorption on bare (RS1), F2+ dissociative adsorption on bare
    Loss: F+F(s)→F2 (RS8), SFx+F(s)→SFx+1 (RS9–RS11), ion sputtering of F(s)

---

## 7. Transport Equations

### 7.1 Neutral Wall Loss — Chantry Formula

For a neutral species with total effective sticking probability s_eff:

    nu_wall = [1/nu_surface + 1/nu_diffusion]^(-1)

    nu_surface = [s_eff / (2 - s_eff)] × (v_th / 4) × (A / V)

    nu_diffusion = D / Lambda²

where:

    v_th = sqrt(8 × kB × T_gas / (pi × M))             [thermal velocity]
    D = (pi/8) × lambda_mfp × v_th                       [free diffusion coefficient]
    lambda_mfp = 1 / (n_gas × sigma_collision)            [mean free path]
    sigma_collision = pi × ((sigma_i + sigma_SF6) / 2)²   [hard-sphere cross section]

The total effective sticking s_eff is the sum over all competing surface reactions for that species. For example, for F atoms:

    s_eff(F) = s1 × theta_bare + s5 × theta_SF3 + s6 × theta_SF4 + s7 × theta_SF5 + s8 × theta_F

The individual reaction rates are then branched proportionally:

    Rate(S1) = nu_wall × [s1 × theta_bare / s_eff] × n(F)

### 7.2 Ion Wall Loss — Full Lee & Lieberman h-Factors

The ion wall loss frequency for species i is:

    w_i = u_B,i × (h_L × A_ax + h_R × A_rad) / V

where u_B,i is the Bohm velocity:

    u_B,i = sqrt(e × Te / M_i)

The h-factors include three physical effects: low-pressure free-streaming, intermediate-pressure collisional diffusion, and electronegative ion confinement.

**Axial h-factor (Lee & Lieberman 1995, Eq. A9):**

    h_L = [(1 + 3*alpha/gamma) / (1 + alpha)] × 0.86 / sqrt(3 + L/(2*lambda_i) + (0.86*L*uB/(pi*Da))²)

**Radial h-factor (Eq. A10):**

    h_R = [(1 + 3*alpha/gamma) / (1 + alpha)] × 0.80 / sqrt(4 + R/lambda_i + (0.80*R*uB/(2.405*J1(2.405)*Da))²)

where:

    alpha = n_neg / ne                      [electronegativity ratio]
    gamma = Te / Ti                          [temperature ratio]
    lambda_i = 1 / (n_gas × sigma_ion)      [ion mean free path]
    Da = e × Te / (M_ion × nu_i)            [ambipolar diffusion coefficient]
    nu_i = v_th,i / lambda_i                 [ion-neutral collision frequency]
    J1(2.405) = 0.5191                       [Bessel function value]

**Physical meaning of each term:**

- The prefactor (1 + 3*alpha/gamma)/(1 + alpha) accounts for negative ion confinement. At alpha = 0 (no negative ions) it equals 1. At alpha >> 1 (strongly electronegative) it approaches 3/gamma, which can be much less than 1 because gamma >> 1 for hot electrons and cold ions. This reduces the ion wall flux dramatically.

- The term 3 + L/(2*lambda_i) interpolates between free-streaming (low pressure, lambda >> L) and collisional transport (high pressure, lambda << L).

- The term (0.86*L*uB/(pi*Da))² adds the high-pressure ambipolar diffusion limitation.

### 7.3 Ion Temperature

From Lee & Lieberman 1995:

    Ti(eV) = (0.5 - T_gas/11605) / p_mTorr + T_gas/11605     [for p > 1 mTorr]
    Ti(eV) = 0.5                                                [for p <= 1 mTorr]

At the operating conditions (0.921 Pa = 6.9 mTorr, T_gas = 315 K): Ti ≈ 0.096 eV ≈ 1110 K.

### 7.4 Negative Ion Transport

Negative ions do not reach the walls. They are confined by the sheath potential barrier. Their only loss mechanisms are volume processes: detachment (G20–G21) and ion-ion recombination (G42–G49).

---

## 8. Operating Conditions

| Parameter | Value |
|-----------|-------|
| SF6 feed | 100 sccm |
| Ar feed | 10 sccm (actinometer, no reactions) |
| Gas temperature | 315 K |
| Power range | 100–3500 W |
| p_OFF range | 0.5–4.0 Pa |
| Pressure controller | OFF (constant outlet flow rate) |
| Power coupling efficiency | 0.50 |

**Ar treatment:** Ar is not included in the reaction set (no Ar reactions in Table 1). Ar contributes to the total neutral density for pressure and diffusion calculations. Its density is constant at n_Ar = (Q_Ar / Q_total) × p_OFF / (kB × T_gas).

**Constant flow rate mode:** The outlet volumetric flow rate is fixed at:

    S_p = Q_total × T_gas / (p_OFF × T0)

The pressure after plasma ignition is computed from the ideal gas law:

    p = (Sum of all neutral densities + n_Ar) × kB × T_gas

The pressure rise is:

    Delta_p = p - p_OFF

---

## 9. Solver Method

The system of 18 ODEs is integrated using scipy.integrate.solve_ivp with the BDF method (suitable for stiff systems).

**State vector (18 components):**

    y = [n_SF6, n_SF5, n_SF4, n_SF3, n_F2, n_F, n_SF5+, n_SF4+, n_SF3+, n_F2+, n_SF6-, n_F-, n_e, Te, theta_F, theta_SF3, theta_SF4, theta_SF5]

**Initial conditions:**

    n_SF6 = (Q_SF6/Q_total) × p_OFF / (kB × T_gas)    [pure SF6]
    All other species: 10^10 m^-3 (trace seed)
    Te = 5.0 eV
    All theta: 0.01

**Integration time:** 1.0 second (sufficient for all species to reach steady state).

**Tolerances:** rtol = 10⁻⁸, atol = 10⁻⁶, max_step = 10⁻³ s.

---

## 10. Key Physical Constants

| Constant | Symbol | Value |
|----------|--------|-------|
| Boltzmann constant | kB | 1.3807e-23 J/K |
| Elementary charge | e | 1.6022e-19 C |
| Electron mass | m_e | 9.1095e-31 kg |
| Atomic mass unit | amu | 1.6606e-27 kg |

---

## 11. References

1. Kokkoris et al., J. Phys. D: Appl. Phys. 42 (2009) 055209 — Target paper
2. Kokkoris et al., J. Phys. D: Appl. Phys. 41 (2008) 195211 — Power balance equations (Ref [36])
3. Lee & Lieberman, J. Vac. Sci. Technol. A 13 (1995) 368 — h-factors for electronegative plasmas (Ref [45])
4. Ryan & Plumb, Plasma Chem. Plasma Process. 10 (1990) 207 — Troe fall-off formula (Ref [25])
5. Ryan & Plumb, Plasma Chem. Plasma Process. 8 (1988) 281 — Experimental recombination rates
6. Chantry, J. Appl. Phys. 62 (1987) 1141 — Neutral wall loss formula
7. Christophorou & Olthoff, J. Phys. Chem. Ref. Data 29 (2000) 267 — SF6 cross sections (Ref [12])
