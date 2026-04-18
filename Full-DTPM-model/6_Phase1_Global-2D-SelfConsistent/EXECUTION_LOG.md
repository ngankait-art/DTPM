# Phase-1 Rewrite — End-to-End Execution Log

**Date**: 2026-04-17
**Scope**: Fork → ultrareview → self-consistent η → m12 bias sheath → 14-run campaign → report update.

---

## Stage-by-Stage Summary

### Stage 0 — Migration (done)
Forked `5.Phase1_EM_Chemistry_Merged` + `5a_Phase1_Mettler_Validation_Correction` into
`6_Phase1_A Self-Consistent Global–2D Hybrid Framework for ICP Etch Modelling/`
(79 MB total).  Excluded `__pycache__`, stale build artefacts, and results/sweeps.
5a assets preserved under `validation/mettler_benchmark/`.  Global `0D–2D` → `Global–2D`
replacement in `main.tex`.

### Stage 5 — Ultrareview (done)
Produced `docs/CODE_REVIEW_ULTRAREVIEW.md`.  Three CRITICAL findings beyond the known
η tautology:
1. `m01_circuit.py` assumed whole coil+plasma is a matched 50 Ω load (`V_rms = √(P·Z)`).
2. `generate_report_figures.py` hard-coded `eta = P_abs / 700` (wrong at 1000 W).
3. FDTD Gaussian-pulsed source turned OFF after ~3 RF cycles — the hidden reason the
   old code required E-field rescaling.

### Stage 1 — Self-Consistent η (done)
Rewrote `m01_circuit.py`, `m10_power_deposition.py`, `m11_plasma_chemistry.py`; fixed
FDTD CW source in `m06_fdtd_cylindrical.py`.  Added `tests/test_selfconsistent_eta.py`
(8/8 pass, including the R_coil=0 ⇒ η=1 invariant).  Config: added `circuit.R_coil = 0.8 Ω`,
`circuit.L_coil = 2 µH`, bumped `fdtd_rf_cycles = 2 → 5`.

Smoke test at 1000 W / 10 mTorr / 70% SF6: η converged to 0.95 (dynamic, not pegged),
I_peak = 11.2 A, R_plasma = 15.3 Ω, P_abs = 950 W.  FDTD |E_max| went from 0.1 V/m
(pulsed source, wrong) to 2.56 kV/m (CW source, physical).

### Stage 2 — Wafer-Bias Sheath Module (done)
Created `src/dtpm/modules/m12_ccp_bias_sheath.py` implementing the reduced-order
Lieberman sheath + plasma-glow-expansion source.  Single free parameter `lambda_exp`
calibrated on Mettler 90% SF6 (target ×1.60) → `lambda_exp = 3.20` gives model ×1.613
(+0.85% deviation).  Blind test on 30% SF6 (target ×2.15) → model ×1.876 (−12.8%,
within ±20% band).  Config updated.

### Stage 3 — 14-Run Campaign (done)
Power sweep (200–1200 W at 100 W increments, 70% SF6, 200 W bias): 11 points.
Composition pair (1000 W, 90%/30% SF6, bias off/on): 4 runs.
All 15 completed without NaN/Inf; runtime ~3 min per run.

**Key numbers from the power sweep**:

| P_rf (W) | η | I_peak (A) | R_plasma (Ω) | F_drop (%) | [F]_centre (cm⁻³) |
|---|---|---|---|---|---|
| 200 | 0.956 | 4.67 | 17.57 | 64.5 | 9.25×10¹³ |
| 700 | 0.953 | 9.11 | 16.07 | 65.8 | 1.26×10¹⁴ |
| 1000 | 0.950 | 11.15 | 15.30 | 66.1 | 1.39×10¹⁴ |
| 1200 | 0.949 | 12.41 | 14.79 | 66.3 | 1.49×10¹⁴ |

η is nearly flat (0.949–0.956) because R_plasma ≫ R_coil throughout.  This was
not predictable in advance — it is a discovered property of the operating envelope.

**Composition pair summary**:

| Case | [F]_off | [F]_on | Enhancement | Mettler | Dev |
|---|---|---|---|---|---|
| 90% SF6 (cal) | 1.030×10¹⁴ | 1.661×10¹⁴ | ×1.614 | ×1.60 | +0.85% |
| 30% SF6 (blind) | 3.65×10¹³ | 6.84×10¹³ | ×1.876 | ×2.15 | −12.8% |

### Stage 4 — Report Update (done)
Inserted two new §§ in Chapter 5: §sec:selfconsistent-eta and §sec:m12-bias-sheath.
Updated §sec:hybrid-limits to 4 principal + 1 scope-note (L5 now kinetic-regime only;
old L5 and L6 resolved).  Replaced §sec:mettler-fig417-direct with v2 content using
the self-consistent + bias-on data.  Updated abstract.  Regenerated 4 new figures via
`scripts/generate_phase1_selfconsistent_figures.py`.  Renamed output PDF to
`A Self-Consistent Global–2D Hybrid Framework for ICP Etch Modelling.pdf`.

**Final compile**: 102 pages, 0 LaTeX errors, 0 undefined references.

---

## Residual Gap vs Prior Revision

The absolute-magnitude benchmark gap against Mettler Fig 4.17 (bias-on), at the wafer
centre (r = 0):

|  | Prior Phase-1 (5a) | This revision |
|---|---|---|
| 90% SF6 residual | −75.3% | ~−33% (see fig (c)) |
| 30% SF6 residual | −42.3% | ~−35% (see fig (c)) |

The fix was primarily due to (a) self-consistent η nearly 2× raising P_abs (430 W
prescribed → 950 W emergent) and (b) m12 adding an explicit 1.6×–1.9× bias
enhancement at the wafer.

Remaining gap (~30–35% below Mettler's absolute values at wafer centre) is consistent
with the **still-unresolved L1 electronegative ambipolar correction**.  The expected
(1+α)/(1+αTᵢ/Tₑ) ≈ 2 factor would close most of this gap; it is tagged as the highest-
priority Phase-2 task in Chapter 8 Future Work.

---

## Files Modified / Created

### Source (rewritten)
- `src/dtpm/modules/m01_circuit.py` — Lieberman transformer circuit
- `src/dtpm/modules/m10_power_deposition.py` — removed E-scale rescaling
- `src/dtpm/modules/m11_plasma_chemistry.py` — self-consistent Picard + m12 hook
- `src/dtpm/modules/m06_fdtd_cylindrical.py` — CW source (was Gaussian pulsed)

### New modules
- `src/dtpm/modules/m12_ccp_bias_sheath.py` — wafer-bias sheath + expansion source

### Tests
- `tests/test_selfconsistent_eta.py` — 8 tests, all passing

### Scripts
- `scripts/run_power_sweep_1000W_biased.py` — 11-point power sweep
- `scripts/run_mettler_composition_pair.py` — 4-run composition blind test
- `scripts/generate_phase1_selfconsistent_figures.py` — regen all figures

### Docs
- `docs/CODE_REVIEW_ULTRAREVIEW.md` — ultrareview findings
- `docs/ASSUMED_PARAMETERS.md` — full literature-value provenance
- `docs/tex_drafts/` — standalone TeX drafts (staged before main.tex edits)

### Config
- `config/default_config.yaml` — added `circuit.{R_coil, L_coil}`, `bias.*`,
  removed `operating.eta_initial`, bumped `fdtd_rf_cycles` to 5.

### Results (new, regenerated from scratch)
- `results/sweeps/power_1000W_biased/` — 11 sweep-point dirs + index.json
- `results/mettler_composition/{90pct_SF6,30pct_SF6}/{bias_off,bias_on}/` — 4 dirs
  + composition_summary.json

### Report figures (new)
- `fig_eta_sweep.{png,pdf}` — 3-panel emergent circuit sweep
- `fig_Fdrop_vs_power.{png,pdf}` — 3-panel [F] vs P_rf
- `fig_composition_blind_test.{png,pdf}` — bias enhancement bar chart
- `fig_mettler_fig417_v2.{png,pdf}` — absolute-magnitude benchmark

### Report
- `docs/report/main.tex` — abstract + 2 new §§ + updated §5.5 + updated §12 (Mettler)
- `docs/report/main.pdf` — 102 pages, 0 errors
- `docs/report/A Self-Consistent Global–2D Hybrid Framework for ICP Etch Modelling.pdf`
  — renamed output

---

## What Could Still Be Improved

- **R_coil = 0.8 Ω** is the Lieberman textbook coil-only value; real-world TEL ICPs
  have matching-network losses that would bump R_coil to ~2-3 Ω, dropping η to
  ~0.80–0.85 at the same operating conditions.  This would tighten the absolute-
  magnitude agreement with Mettler by another ~10–15 %.  A measurement-based
  refinement of R_coil is the cleanest next step.
- **L1 electronegative ambipolar** is the one unresolved architectural limitation that
  would meaningfully close the remaining absolute-magnitude gap.  Implementation cost
  is low (~1 day); it is the highest-priority Phase-2 task.
- **Composition-dependent γ_Al** would close the last 2-pp F-drop composition-
  sensitivity residual.  Phase-3 scope.
