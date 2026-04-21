# Code Review "Ultrareview" — Pre-Stage-1 Audit

**Date**: 2026-04-17
**Scope**: `src/dtpm/` + `scripts/` + `config/default_config.yaml`
**Purpose**: Before rewriting m01/m10/m11 for self-consistent η, surface every *other* latent prescribed-input / rescaling pattern that could invalidate an absolute-magnitude benchmark (like the η tautology already did).

---

## Summary of Findings

| Severity | Issue | File:Line | Stage where fixed |
|---|---|---|---|
| CRITICAL | η tautology: `P_abs = eta_initial × P_rf`, E-field rescaled every iter | m11_plasma_chemistry.py:155, 166, 192 | Stage 1 |
| CRITICAL | `compute_E_scale_factor()` throws away FDTD absolute magnitude | m10_power_deposition.py:103–125 | Stage 1 |
| CRITICAL | m01 assumes whole ICP = matched 50 Ω load → I_peak = √(2P/Z) | m01_circuit.py:46–49 | Stage 1 (rewrite) |
| HIGH | 0D global model also uses `P_abs = P_rf × eta` (prescribed) | chemistry/global_model.py:85 | Documented, not fixed (see §B below) |
| HIGH | `eta_initial = 0.43` default plumbed through 5 files | several | Stage 1 (remove from new flow) |
| MEDIUM | `generate_report_figures.py` hard-codes `eta = P_abs / 700` at line 418 | generate_report_figures.py:418 | Stage 4 (fix when regenerating) |
| MEDIUM | `m06_fdtd_cylindrical.py:296`: `I_peak = state.get('I_peak', 5.0)` — 5.0 A fallback is arbitrary | m06_fdtd_cylindrical.py:296 | Stage 1 (drop default, require explicit I_peak) |
| LOW | `compute_external_circuit()` in m01 has R=50, L=1e-6, C=1e-12 hardcoded | m01_circuit.py:82–84 | Not in active pipeline; leave as-is |
| LOW | `fdtd_rf_cycles = 2` (50 ns at 40 MHz) may be short for absolute amplitude convergence | config/default_config.yaml | Stage 1 verification |

All other `.get('x', default)` patterns are **geometrical or operational parameters** (`R_icp`, `L_proc`, `pressure_mTorr`, `Tgas`, `gamma_Al`, `frac_Ar`, …) that the user sets deliberately; no hidden tautology there.

---

## A. The η Tautology (already documented)

See `validation/mettler_benchmark/EXECUTION_LOG.md` §"Findings" for the full
trace. Briefly: m11's Picard loop calls `compute_E_scale_factor()` to rescale
the FDTD E-field magnitude so that the Ohmic integral always equals
`eta_initial × P_rf`. Therefore `eta_computed == eta_initial` to machine
precision, and any "self-consistent η" claim in the prior report is false.
Stage 1 removes the rescale and replaces it with a transformer-coupling
iteration that lets η emerge.

---

## B. `m01_circuit.py` Assumes a Matched 50 Ω Load

**Current code** (m01_circuit.py:46–49):
```python
V_rms = np.sqrt(P * Z)
V_peak = V_rms * np.sqrt(2)
I_rms  = V_rms / Z
I_peak = I_rms * np.sqrt(2)
```

with `P = 700 W`, `Z = 50 Ω` (from config). This implicitly treats the entire
ICP coil + plasma as a single real 50 Ω resistor matched to the RF supply,
giving `I_peak = √(2 × 700 / 50) = 5.29 A`.

**Why this is wrong in principle**: an ICP coil has its own resistance
`R_coil ~ 1 Ω` and inductance `L_coil ~ 2 µH` (at 40 MHz, `X_L = ωL ≈ 500 Ω`).
The plasma couples through mutual inductance and presents a reflected
impedance `Z_reflected = ω²M² / Z_plasma`. The actual coil current is set by
Kirchhoff's voltage law across the RLC + transformer network, **not** by a
matched-load assumption.

**Why it produces a working simulation anyway**: because m11 subsequently
rescales the FDTD E-field to match `eta_initial × P_rf`, the I_peak value
from m01 doesn't matter — only its rough order of magnitude does. Both
errors cancel.

**What Stage 1 does**: rewrites `m01.compute_circuit_parameters()` to take
`R_coil`, `L_coil` (new config params) and iterate with m10 to find the
self-consistent `I_peak` that delivers `P_rf` through `(R_coil + R_plasma)`.
`Z = 50 Ω` (from the RF supply side) becomes a purely informational output,
not part of the current-computation path.

---

## C. 0D Global Model Prescribed η (acceptable)

`chemistry/global_model.py:85`:
```python
P_abs = P_rf * eta
```

The 0D model is volume-integrated; it has no spatial E-field, so there is
no first-principles way to compute η inside a 0D code. The 0D solver is
used in this project as a **reference / benchmark** for the 2D solver,
not as the primary result. The prescribed η there is acceptable **provided**:

1. We pass the 2D's self-consistent η into the 0D run for a like-for-like
   benchmark. This is already done via `eta=oper.get('eta_initial', 0.43)`
   in scripts/run_parameter_sweeps.py:86 — but the passed value is 0.43,
   not the 2D's computed η. **Fix in Stage 1**: pass `state['eta_computed']`
   after the 2D Picard converges.

2. The report clearly states the 0D η is a *prescribed input to the 0D*,
   not a validated output of the model.

Stage 4 adds that clarification to the report.

---

## D. Hard-Coded `eta = P_abs / 700` in Report Figures

`scripts/generate_report_figures.py:418`:
```python
eta = P_abs / 700 if P_abs > 0 else 0
```

This hard-codes `P_rf = 700 W`. At the new 1000 W operating point, every
figure in the existing pipeline would compute the wrong η.

**Fix (Stage 4)**: replace with `eta = P_abs / circuit['source_power']`.

---

## E. FDTD rf_cycles = 2 May Be Short

At 40 MHz, `T_rf = 25 ns`, so 2 cycles = 50 ns. For the current approach
(FDTD gives spatial *shape* only, magnitude is rescaled), 2 cycles is fine.

For the Stage-1 architecture (FDTD magnitude must be absolutely converged to
the steady-state value set by `I_peak`), 2 cycles may be insufficient.
Typical industrial practice is 5–10 cycles for amplitude convergence.

**Action (Stage 1)**: bump `fdtd_rf_cycles` from 2 to 5 in config (1.5x
runtime cost per Picard step). Verify by plotting E_theta(t) at a probe
point for the last 2 cycles and confirming amplitude stability to <1 %.

---

## F. m06 `I_peak` Default = 5.0 A

`m06_fdtd_cylindrical.py:296`:
```python
I_peak = state.get('I_peak', 5.0)
```

If m01 doesn't run first, m06 uses 5.0 A (coincidentally matches the 700 W /
50 Ω matched-load value). Any script that invokes m06 standalone with a
1000 W config will silently use 5.0 A — wrong.

**Fix (Stage 1)**: drop the default, raise a `KeyError` if `I_peak` is
missing from state. Forces explicit sourcing through the new m01.

---

## G. Parameters That Are Correctly Prescribed

The following `.get('x', default)` patterns are legitimate user inputs:

- `R_icp`, `R_proc`, `L_icp`, `L_proc`, `L_apt` — reactor geometry (TEL spec)
- `pressure_mTorr`, `Tgas`, `frac_Ar`, `Q_sccm` — operating conditions
- `gamma_quartz`, `gamma_Al`, `gamma_wafer`, `gamma_window` — wall chemistry (Kokkoris / Mettler calibration)
- `Nr`, `Nz`, `beta_r`, `beta_z` — mesh parameters
- `max_picard_iter`, `picard_tol`, `under_relax_*`, `inner_chem_iter`, `inner_chem_relax` — numerical iteration parameters
- `coil_r_position`, `num_coils`, `coil_start_z`, `coil_spacing` — coil geometry

No action needed on these; they're physics/geometry/numerics, not over-fit tunes.

---

## H. Bill of Health After Stage 1

After Stage 1 completes, the following must all be true:

- [ ] No `eta_initial` references in the active Phase-1 pipeline (config, src, or scripts called during a simulation).
- [ ] `compute_E_scale_factor` and the `target_P_abs` branch of `compute_power_deposition` are deleted from m10.
- [ ] `V_rms = √(P × Z)` formula is deleted from m01.
- [ ] m01 returns `I_peak` derived from `R_coil + R_plasma` series; m06 consumes it without fallback.
- [ ] `eta_computed` in state is derived purely from `R_plasma / (R_coil + R_plasma)` and **cannot** equal a prescribed number by construction.

A unit test (`tests/test_selfconsistent_eta.py`) asserts this with
`R_coil = 0` → η = 1 and `R_coil = ∞` → η = 0.

---

## I. Additional Parameters the User Approved (documented in `docs/ASSUMED_PARAMETERS.md`)

| Param | Value | Source |
|---|---|---|
| `R_coil` | 0.8 Ω | Lieberman 2005, Eq 12.2.19, typical TEL-class 3-turn coil |
| `L_coil` | 2.0 µH | Lieberman §12.2 |
| `f_ion_to_gas` | 0.5 | Turner 2013 |
| `A_wafer_eff` | 0.018 m² | TEL: R_wafer = 75 mm |
| `L_exp` | 30 mm | Phase-1 calibration (tunable) |
| `lambda_exp` | calibrated on Mettler 90% SF6 | Free parameter, Stage 2 |

These enter as config values in `default_config.yaml` and are called out
explicitly in the new §5 subsections of the report.

---

## Post-L1 Addendum (2026-04-17)

This ultrareview document is frozen as the pre-L1 audit.  The L1 limitation
(electronegative ambipolar diffusion coefficient) that was listed in the prior
revision's §5.5 is now also **resolved** — see `docs/L1_AUDIT.md` for the full
L1 resolution audit and `docs/report/main.tex §sec:electronegative-ambipolar`
for the report subsection.  The L1 fix is rigorous physics (no free parameters
added) and passes 7 unit tests in `tests/test_electronegative_ambipolar.py`.

Important new finding from L1: at Mettler's operating conditions the 0D model
returns α ≈ 0.02, giving a correction factor of ≈ 1.02.  The prior projection
that L1 would close the 5.8-pp F-drop gap was based on an over-estimate of α
and is therefore falsified by self-consistent physics.  L1 remains physically
essential (correction rises to ~2 at lower-power / higher-pressure conditions)
but does not resolve the current F-drop residual at Mettler's benchmark point.
