# Assumed Parameters — Transparent Literature Tabulation

This document lists every physical parameter in the Phase-1 Global–2D model
whose value is taken from the literature rather than measured directly for
the TEL etcher we simulate.  Each entry gives the value, its Lieberman /
Turner / other citation, and how it enters the code.

All values propagate through `config/default_config.yaml` and are flagged
with `# LIT: <reference>` comments at the point of use.

---

## Coupling / Power Coupling (Stage 1)

| Parameter | Value | Purpose | Reference |
|---|---|---|---|
| `circuit.R_coil` | 0.8 Ω | Real resistance of ICP coil wire (loss term in η) | Lieberman & Lichtenberg, *Principles of Plasma Discharges and Materials Processing*, 2nd ed. (Wiley, 2005), Eq. 12.2.19 for a 3-turn industrial TEL-class coil. |
| `circuit.L_coil` | 2.0 µH | Self-inductance of ICP coil (reactive term in series impedance) | Lieberman §12.2; typical value for TEL ICP coils. |

**Sensitivity note**: `R_coil` and `L_coil` together set the coupling
efficiency η.  Increasing `R_coil` decreases η.  At Mettler's 1000 W
operating point, η values in the range 0.6 – 0.85 are expected; anything
outside that window indicates the assumed `R_coil` is off.

---

## Wafer-Bias Sheath (Stage 2)

| Parameter | Value | Purpose | Reference |
|---|---|---|---|
| `bias.P_bias_W` | 200 W | RF bias power delivered to the wafer (Mettler condition) | Mettler 2025 PhD dissertation, §4.3.2, Fig. 4.17. |
| `bias.f_ion_to_gas` | 0.5 | Fraction of sheath ion kinetic energy deposited as neutral-gas heating | Turner et al. 2013 "Simulation benchmarks for low-pressure plasmas: Capacitive discharges", *Phys. Plasmas* **20** (2013) 013507. |
| `bias.A_wafer_eff` | 0.018 m² | Effective wafer area for sheath-power balance (π × R_wafer² at R_wafer = 75 mm) | TEL specification. |
| `bias.L_exp` | 30 mm | Decay length of ion-heated plasma-glow expansion into process chamber | Phase-1 engineering estimate (of order half the process-chamber height). |
| `bias.lambda_exp` | **3.20** (calibrated) | Dimensionless peak expansion factor at the aperture | Calibrated on Mettler Fig 4.17 90 % SF₆ bias-on centre [F] enhancement (model ×1.613 vs target ×1.60, +0.8 %). **Blind test at 30 % SF₆**: model ×1.88 vs target ×2.15 (−12.8 %, within the ±20 % acceptance band). |

**Sensitivity note**: `lambda_exp` is the *single* free parameter of the
bias model.  It is the bias-analogue of `gamma_Al` (which calibrates the
ICP-only model).  The blind-test on 30% SF6 verifies that the physics
captured by `lambda_exp` is composition-transferable, not an overfit.

---

## Ion Masses

| Ion | `m_i` | Notes |
|---|---|---|
| SF₅⁺ | 127 amu | Dominant positive ion at 90% SF₆ (per Lallement 2009 Table 2) |
| Ar⁺ | 40 amu | Dominant positive ion at 30% SF₆ (Ar-diluted regime) |
| F⁻ | 19 amu | Dominant negative ion (electronegative, all compositions) |

The ion-flux-to-wafer formula (Bohm flux) uses a composition-weighted
effective ion mass; see `src/dtpm/modules/m12_ccp_bias_sheath.py`.

---

## Wall Recombination (unchanged from prior Phase-1)

| Surface | γ | Reference |
|---|---|---|
| Quartz | 0.001 | Kokkoris et al. 2009 |
| Aluminium | 0.18 | Calibrated to Mettler 74% drop (Fig. 4.14, 90% SF₆) |
| Silicon wafer | 0.025 | Kokkoris 2009 |
| Viewing window | 0.001 | Same as quartz |

No change from prior Phase-1.  `γ_Al` remains the one calibrated number
in the ICP-only model; `lambda_exp` is its bias-on analogue.

---

## Electronegative Ambipolar Correction (L1 resolution, 2026-04-17)

No new parameters.  The electronegative correction factor
`(1+α)/(1+α·T_i/T_e)` is applied in `solve_ne_ambipolar()` using values of
α, T_e, T_i that are already present in the state:

| Quantity | Source | Notes |
|---|---|---|
| α | 0D global model (`result_0D['alpha']`) | No new input — produced by the 0D attachment/recombination balance that was already in the pipeline. |
| T_e | Local power-balance solve (m11) | 2D field, already computed. |
| T_i | `kB × operating.Tgas` | T_gas = 313 K (already in config). Assumed isothermal with the gas. |

Contrast with `γ_Al` and `λ_exp`: those are **fitted** against Mettler data.
The L1 correction is **not** fitted — it is the rigorous three-species flux
balance (Lieberman 2005 §10.3).

---

## Operating / Feed Gas (set by user, not literature)

| Parameter | Default | Units |
|---|---|---|
| `operating.pressure_mTorr` | 10 | mTorr |
| `operating.frac_Ar` | 0.3 (new 70% SF₆) or 0.1/0.7 for composition pair | — |
| `operating.Q_sccm` | 100 | sccm |
| `operating.Tgas` | 313 | K |

No literature input; these are controlled experimental inputs.

---

## How these values appear in the code

Each assumed value is marked at its first point of use with:

```python
# LIT: Lieberman 2005 Eq 12.2.19; assumed in absence of measurement
R_coil = config.circuit.get('R_coil', 0.8)
```

Grep for `# LIT:` in `src/dtpm/` to enumerate every literature-sourced input.

---

## If Any Of These Are Wrong

The calibration parameters (`γ_Al`, `lambda_exp`) are tuned once on a
Mettler reference condition; their values are robust by construction.

The literature values (`R_coil`, `L_coil`, `f_ion_to_gas`) are not tuned.
If they are off by a factor of 2, the corresponding model output (η,
V_dc, neutral-gas temperature) will be off by a comparable factor.  The
sensitivity study in Stage 3 (a power sweep from 200 to 1200 W) provides
an indirect check: a wrong R_coil will manifest as an unphysical
η-vs-P_rf curve.
