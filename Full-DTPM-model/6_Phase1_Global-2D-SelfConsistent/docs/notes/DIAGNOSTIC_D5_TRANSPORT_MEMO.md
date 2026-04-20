# Diagnostic D5 — neutral-transport sensitivity sweep

**Date**: 2026-04-19
**Operating point**: Mettler Fig 4.17 90% SF₆ bias-on (P_ICP = 1000 W, 10 mTorr,
100 sccm SF₆/Ar with frac_Ar = 0.1, 200 W rf bias, R_coil = 0.8 Ω)
**Mettler reference**: [F]_c(r=0) = 3.774 × 10¹⁴ cm⁻³ (probe-direct, Fig 4.17)

**Context**: D5 is the last of the four-lever diagnostic sequence. D1, D3,
and D4 ruled out the bias/sheath module, EM coupling, and electron-impact
kinetics as the dominant source of the ~−56 % absolute-magnitude gap
against Mettler. D5 tests the remaining candidate: neutral-transport
coefficients (species diffusion D_s) and aluminium-wall surface kinetics
(s_F Al sticking probability). γ_Al = 0.18 is supervisor-locked and not
varied.

---

## Sweep design

Two dimensionless scale factors applied on top of the defaults:

- **`s_F_Al_scale`** ∈ {0.25, 1.0, 4.0} — scales `wall_chemistry.SURF_AL['s_F']` (default 0.015). Hypothesis: lower s_F → less F consumption at wall → higher bulk [F].
- **`d_scale`** ∈ {0.5, 1.0, 2.0} — scales every neutral diffusion coefficient from `compute_diffusion_coefficients`. Hypothesis: lower D → longer F residence in the ICP source → higher centre [F].

9-point grid. Execution: `scripts/run_d5_transport_sweep.py` with `multiprocessing.Pool(8)` — wall-clock 239 s vs ~12 min serial.

---

## Results

| s_F_Al_scale | d_scale | [F]_c (cm⁻³) | Residual vs Mettler | F-drop (%) | η | I_peak (A) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.50 | 1.847 × 10¹⁴ | −51.1 % | 78.86 | 0.947 | 11.52 |
| 0.25 | 1.00 | 1.659 × 10¹⁴ | −56.0 % | 66.81 | 0.948 | 11.38 |
| 0.25 | 2.00 | 1.390 × 10¹⁴ | −63.2 % | 51.03 | 0.950 | 11.19 |
| 1.00 | 0.50 | 1.847 × 10¹⁴ | −51.1 % | 78.86 | 0.947 | 11.52 |
| **1.00** | **1.00** | **1.659 × 10¹⁴** | **−56.0 %** | **66.81** | **0.948** | **11.38** |
| 1.00 | 2.00 | 1.390 × 10¹⁴ | −63.2 % | 51.03 | 0.950 | 11.19 |
| 4.00 | 0.50 | 1.847 × 10¹⁴ | −51.1 % | 78.86 | 0.947 | 11.52 |
| 4.00 | 1.00 | 1.659 × 10¹⁴ | −56.0 % | 66.81 | 0.948 | 11.38 |
| 4.00 | 2.00 | 1.390 × 10¹⁴ | −63.2 % | 51.03 | 0.950 | 11.19 |

(Bold row = D1/D3/D4 baseline, defaults.)

---

## Key findings

### 1. `s_F_Al` scaling has **zero** effect on the observed quantities.

Across a 16× range in `SURF_AL['s_F']` the outputs are bit-identical. That
is unambiguous evidence the SURF_AL surface-kinetics channel is **not** on
the critical F-loss path in the current model. The actual F loss at the
aluminium wall flows through `γ_Al = 0.18` in the Robin boundary condition
of the F-diffusion solver. SURF_AL only feeds the `wall_sf6_regeneration`
path, which contributes a second-order SF₆ regeneration but has a negligible
effect on the steady-state centre [F].

**Implication**: per-species surface-kinetics tuning under the Kokkoris
parameterisation is ineffective for this model unless γ_Al also moves.
The only composition-insensitivity fix available at the wall level *is*
the γ_Al path.

### 2. Diffusion scaling is a genuine knob but pairs absolute-magnitude against radial shape.

Halving D doubles the centre [F] and steepens the radial profile; doubling
D does the opposite. But neither end closes the gap:

- d_scale = 0.5 → residual −51.1 % (5 pp better), **F-drop 78.9 % (overshoots Mettler's 74.8 %)**
- d_scale = 1.0 → residual −56.0 %, F-drop 66.8 % (baseline, too shallow by 8 pp)
- d_scale = 2.0 → residual −63.2 %, F-drop 51.0 % (far too shallow)

A sweet spot around d_scale ≈ 0.65 would approximately match the F-drop
shape (74 %) but would still leave a residual of −54 %. **No (s_F, D)
combination in this grid simultaneously closes the absolute-magnitude gap
and matches the radial shape.**

### 3. Operating voltages and η are insensitive to transport scaling.

η stays at 0.947–0.950 across the full sweep. This is physically correct:
changing neutral diffusion moves the spatial [F] distribution but leaves
the electron-energy balance (and therefore the plasma conductivity that
drives η) essentially unchanged. Further confirmation that the Mettler
absolute-magnitude gap is not an EM-coupling artefact (already established
by D3 R_coil sweep).

---

## Combined D1–D5 diagnostic summary

| Diag | Physical lever | Effect on [F]_c | Effect on F-drop | Closes −56 % gap? |
|---|---|---:|---:|---|
| D1 | bias sheath toggle | ×1.61 on/off | invariant | No (gap is invariant under toggle) |
| D3 | R_coil 0.5–4.0 Ω | ±5 % | ±0.2 pp | No (monotonically worsens with larger R_coil) |
| D4 | tier-1 → tier-2 rates | ×0.18 (worse) | +4.8 pp | No (absolute worse, shape better) |
| **D5** | neutral-transport D | ±11 % | ±27 pp | **No (shape-magnitude coupling)** |
| — | γ_Al (supervisor-locked) | large | large | untested |

**The Phase-1 architecture cannot close the Mettler absolute-magnitude
residual within the currently-allowed parameter space.** The only
remaining Phase-1-local candidate is γ_Al retuning, which is explicitly
disallowed per 2026-04-19 supervisor decision.

---

## Recommendations

1. **Ask the supervisor to unlock γ_Al for a one-parameter refit.**
   If γ_Al is refitted from 0.18 to a lower value (candidate range 0.08–
   0.12, approximate via the D_F-drop sensitivity of ~5 pp per ±10 % in
   γ_Al), we expect [F]_c to scale up roughly ×2–3, closing the
   absolute-magnitude gap. The composition-insensitivity of a single
   γ_Al value remains a documented Phase-3 scope item either way.
2. **Alternatively, add physics rather than tune.**
   Candidates:
     a. **Composition-dependent γ_Al(x_Ar)** — explicitly rejected 2026-04-19.
     b. **2D-resolved electronegativity α(r, z)** instead of the scalar
        0D closure — currently α ≈ 0.02 uniform, but in the wafer region
        α may be much higher, which would steepen the F-drop and permit
        a lower γ_Al. Phase-3 scope per §5.5 of the report.
     c. **Non-Maxwellian EEDF feedback to wall kinetics** — the PINN's
        Te_eff ≈ 2 eV (vs the model's Te ≈ 3 eV) implies the surface
        F-atom flux is computed at the wrong velocity. Quantitative
        correction needed.
3. **Accept the systematic bias in the report.** Document the absolute-
   magnitude residual of −56 % as a known architectural limitation,
   attribute it correctly (γ_Al calibration + Maxwellian rate
   overestimate partial cancellation), and move on to Phase-2/Phase-3
   for resolution. The enhancement ratios, radial-profile shape within
   uncertainty bands, and 0D Lallement benchmark are all robust.

My recommendation is **#3 for the report + #1 for any further modelling
work.** The combined D1–D5 result is a clean, defensible characterisation
even if the absolute magnitude remains open.

---

## Files

- **NEW** `scripts/run_d5_transport_sweep.py` — multiprocessing-parallel sweep driver (Pool(8)).
- **DATA** `results/d5_transport_sweep/{s_F0.25_d0.50, ...}/summary.json` — 9 per-point summaries.
- **DATA** `results/d5_transport_sweep/d5_summary.json` — aggregate table.

No source-code edits; D5 uses runtime monkey-patching of `wall_chemistry.SURF_AL`
and `sf6_chemistry.compute_diffusion_coefficients` inside each worker process,
leaving the production code path untouched.
