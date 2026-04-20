# Diagnostic D4 — tier-2 PINN Boltzmann rates in the Picard loop

**Date**: 2026-04-19
**Context**: Phase-1 v2 diagnostic sequence (`manifests/14_v2_execution_plan.md` §3a).
D1 (bias sheath toggle) and D3 (R_coil sweep) ruled out the bias/sheath module
and the EM coupling module as sources of the ~−56% absolute-magnitude gap
against Mettler Fig 4.17. D4 tests the last major electron-kinetics candidate:
replace the Maxwellian Arrhenius rate coefficients with BOLSIG+/PINN-trained
Boltzmann rates inside the Picard loop.

---

## What was wired up

### 1. `src/dtpm/chemistry/tier2_interface.py` (new)

A thin shim around `tier2_pinn.get_rates_pinn.get_rates_pinn` that

- auto-locates the phase-2 PINN package across sibling / DTPM-repo / env-var paths,
- caches model weights across Picard iterations so inference cost is a forward pass only,
- exposes `install_pinn()`, `refresh(E/N, x_Ar, p)`, `clear()`, and `apply_overrides(k)`,
- provides a helper `compute_eff_E_over_N()` that returns a scalar effective E/N
  volume-averaged over the active plasma region.

### 2. `src/dtpm/chemistry/sf6_rates.py` (edit)

The single `rates(Te)` tail now calls `tier2_interface.apply_overrides(k)`, which
is a no-op whenever the tier-2 cache is empty (tier-1 baseline). This keeps
every downstream caller (global_model, multispecies_transport, ambipolar_diffusion,
species_transport) unchanged — the switch is a single module-level flag.

### 3. `src/dtpm/modules/m11_plasma_chemistry.py` (edit)

- Reads `config.chemistry.use_boltzmann_rates` (default `False`)
- Calls `tier2.install_pinn()` once at the top of the Picard loop when enabled
- Calls `tier2.refresh(E/N, x_Ar, p_mTorr)` each outer iteration *after* the
  EM solve, using the volume-averaged E/N from `E_theta_rms(r,z)` and the total
  neutral density
- Writes a `tier2_boltzmann_cache` snapshot + `use_boltzmann_rates` flag into
  the returned state dict for downstream diagnostics
- Clears the cache at the end of the run so subsequent tier-1 runs in the same
  process are not polluted

### 4. `scripts/run_tier_comparison.py` (new)

Runs the tier-1 and tier-2 simulations **in parallel** at Mettler's 1000 W /
10 mTorr / 90% SF₆ / 200 W bias operating point using `multiprocessing.Pool(2)`.
Writes a `comparison.json` with the delta analysis.

---

## Result at Mettler 90% SF₆ bias-on (R_coil = 0.8 Ω, 200 W bias)

| Quantity | tier-1 Arrh | tier-2 PINN | Δ |
|---|---:|---:|---:|
| [F]_c at wafer centre (cm⁻³) | 1.659 × 10¹⁴ | 2.976 × 10¹³ | 0.179× |
| Centre-to-edge F-drop (%) | 66.81 | **71.59** | +4.78 pp |
| Residual vs Mettler (3.774 × 10¹⁴ cm⁻³) | −56.0 % | **−92.1 %** | −36.1 pp |
| η (coupling efficiency) | 0.948 | 0.956 | +0.008 |
| I_peak (A) | 11.38 | 10.51 | −0.87 |
| V_peak (V) | 175.7 | 190.3 | +14.6 |
| Wall-clock parallel (2 workers) | | | 73.2 s |

**Tier-2 PINN cache snapshot** (reported by m11):

- E/N (volume-averaged) = 157.1 Td
- Te_eff = 2.06 eV (vs tier-1 Te_avg ≈ 3.00 eV)
- k_iz = 1.22 × 10⁻²¹ m³/s (aggregate SF₆ ionisation)
- k_att = 2.63 × 10⁻¹⁵ m³/s (aggregate SF₆ attachment)
- k_diss = 3.58 × 10⁻¹⁸ m³/s (aggregate SF₆ dissociation)
- k_exc = 6.52 × 10⁻¹⁴ m³/s

---

## Interpretation

### 1. Tier-2 **worsens** absolute magnitude, not closes it.

The PINN gives a **5.6× lower** centre [F] than tier-1 (2.98 × 10¹³ vs 1.66 × 10¹⁴
cm⁻³). Against Mettler's 3.77 × 10¹⁴ cm⁻³ the residual goes from −56 % to
−92 %. This is the **opposite** of what the manifest-14 §3a hypothesis
predicted ("tier-2 closes the gap").

The root cause visible in the PINN cache: Te_eff = 2.06 eV (vs tier-1's 3.00 eV)
is physically reasonable for a realistic non-Maxwellian EEDF at E/N = 157 Td,
but it pushes the dissociation and ionisation rates down by 1–2 orders of
magnitude relative to Maxwellian Arrhenius at Te = 3 eV. The PINN is reporting
that the Maxwellian assumption was significantly *over*-estimating the high-energy
tail — which in turn was over-estimating electron-impact rates by a compensating
factor.

The combined conclusion is: the old tier-1 [F]_c = 1.66 × 10¹⁴ wasn't closer to
Mettler because the Maxwellian rates were right — it was closer because the
Maxwellian over-estimate happened to partially compensate for some other
under-prediction in the model (most likely neutral-transport / wall chemistry).

### 2. Tier-2 **improves** radial-shape agreement.

F-drop goes from 66.81 % to 71.59 % — closer to Mettler's 74.8 % for this
condition (90% SF₆ bias-on). This is consistent with the PINN's lower net
dissociation rate: slower F production in the source region allows more neutral
diffusion to equilibrate the radial profile, steepening the centre-to-edge
drop.

### 3. D4 does **not** close the Mettler absolute-magnitude gap.

Combined with D1 (bias toggle), D3 (R_coil sweep), and now D4 (tier-2 rates),
three of the four architectural knobs under Phase-1 control have been ruled
out as dominant sources of the residual:

| Diagnostic | Moves [F]_c at wafer? | Closes −56% gap? |
|---|---|---|
| D1 (bias sheath toggle) | ×1.6 for 90% SF₆ on/off (correct ratio) | No — gap is invariant under toggle |
| D3 (R_coil = {0.5, 4.0} Ω) | ±5 % | No — monotonically worsens with larger R_coil |
| **D4 (tier-1 vs tier-2)** | **×0.18 (drops by 82 %)** | **No — makes gap worse** |

The remaining candidates are:

- **Wall chemistry** (γ_Al = 0.18, possibly over-tuned against the old
  approximate Dₐ solver; and/or composition-insensitivity flagged as a known
  architectural limitation per the γ_Al decision)
- **Neutral-transport coefficients** (SF_x diffusion constants, SF_x wall
  recombination probabilities p_wallrec and s_SFx from Kokkoris et al.)
- **Actinometry calibration** (ruled out: direct read of Mettler Ch. 4.3.2
  confirms Fig 4.17 is probe-direct, not actinometry — no ×1.62 correction)

### 4. Interplay: tier-2 + R_coil?

Tier-2 gives η = 0.956 (slightly higher than tier-1's 0.948) because the lower
k_att reduces the plasma conductivity, slightly increasing the loading
efficiency. The I_peak and V_peak shift accordingly (lower I, higher V because
R_plasma increased from 14.6 Ω to 17.3 Ω). These second-order shifts are
within the D3 uncertainty band and do not change the qualitative picture.

---

## Recommendations

1. **Do not switch the production model to tier-2 rates** based on this single
   operating point. The radial-shape improvement is encouraging, but the 36 pp
   absolute-magnitude penalty is too large without a compensating fix
   elsewhere.
2. **Retain tier-2 as an opt-in config flag** (`config.chemistry.use_boltzmann_rates`).
   Keep it off for all report-grade runs until paired with a complementary fix
   from the remaining candidates below.
3. **Proceed to D5 (neutral-transport sensitivity sweep)** as the next
   diagnostic. The γ_Al slot freed by the D2 rejection is best reused here.
   Hypothesis: lowering SF_x wall recombination probabilities p_wallrec or
   SF_x diffusion coefficients D_SFx will raise the absolute [F]_c without
   affecting the radial shape.
4. **Report text** (if tier-2 needs citing): reference this memo, but keep
   §7.5 residuals as the tier-1 baseline.

---

## Open follow-ups

1. **Tier-2 cell-by-cell (full 2D E/N)** — the current implementation feeds a
   scalar volume-averaged E/N to the PINN. A per-cell evaluation would replace
   the scalar rates with an E/N(r,z)-dependent rate field; this is trivially
   batched (the PINN already accepts vectorised input) and would change the
   local dissociation-production profile. Deferred to a future D4b if the
   report goes past tier-1.
2. **m6_pinn.pt vs m5_surrogate.pt** — two PINN weight files exist. Current
   wiring uses the default (`m5_surrogate.pt`, supervised MLP). Swapping to
   the physics-informed variant (`m6_pinn.pt`) is a one-argument change via
   the `weights_path` parameter.
3. **GPU inference** — the PINN is 19k parameters; CPU-only inference is
   already <1 ms per Picard iteration, so GPU routing is not worth wiring for
   this project size.

---

## Files touched by D4

- **NEW**: `src/dtpm/chemistry/tier2_interface.py`
- **EDIT**: `src/dtpm/chemistry/sf6_rates.py` (tail-call to `apply_overrides`)
- **EDIT**: `src/dtpm/modules/m11_plasma_chemistry.py` (Picard-loop hook + config flag)
- **NEW**: `scripts/run_tier_comparison.py` (parallel tier-1/tier-2 runner)
- **DATA**: `results/tier_comparison/{tier1, tier2}/summary.json` + `comparison.json`

Also in this batch (independently useful):

- **EDIT**: `scripts/run_rcoil_sweep.py` — now uses `multiprocessing.Pool(8)` for the 8-point sweep (~2 min wall on an 8-core box vs ~10 min serial).
- **EDIT**: `scripts/run_mettler_composition_pair.py` — now uses `Pool(4)` for the 4 condition runs (~1 min wall vs ~5 min serial).
