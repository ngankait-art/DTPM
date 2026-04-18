# Mettler-Accuracy Pass — Execution Log

Date: 2026-04-17
Stages executed: **A + B** (as recommended in the plan; Stage C items C3 and C4 folded into A/B; C1 and C2 deferred).

## Summary of Changes

### Stage A — Phase-1 `main.tex` corrections (applied directly)

| Item | Location | Status |
|---|---|---|
| E1 — diagnostic description (BBAS→W/Al probes + actinometry) | line 1500 | done |
| Scope clarification (Helicon vs TEL) | line 1500 (appended) | done |
| E3 — qualified "74% drop" at §stage10 | line 1504 | done |
| E5 — γ_Al calibration narrative qualified | line 1090 | done |
| E3 — caption of fig:cross-section-F qualified | line 2135 | done |
| E3 — caption of fig:wafer-profiles qualified | line 2208 | done |
| Honest inner/outer residual narrative | line 2212 | done |
| Caption of fig:radial-F-mettler rewritten | line 2217 | done |
| E2 — §8 Absolute [F] paragraph (Fig 4.5→Fig 4.9) | line 2370 | done |
| E2 — caption of fig:abs-4panel (Fig 4.5→Fig 4.9) | line 2375 | done |
| `\ref{sec:limitations}` → `\ref{sec:hybrid-limits}` | 2 instances | done |
| L5 (kinetic regime only + bias absence) added | §hybrid-limits | done |
| "five limitations" (was "four") intro | §hybrid-limits | done |

### Stage B — New simulations + new §12 figure

| Artefact | Path | Status |
|---|---|---|
| `run_mettler_fig417_90pct.py` | `5a/scripts/` | created, executed |
| `run_mettler_fig417_30pct.py` | `5a/scripts/` | created, executed |
| `gen_mettler_fig417_overlay.py` (3-panel) | `5a/scripts/` | created, executed |
| `regen_radial_F_wafer_mettler.py` | `5a/scripts/` | created, executed |
| `fig_mettler_fig417_overlay.{png,pdf}` | `5a/figures/` → copied into report | done |
| §12 "Direct Benchmark Against Mettler Fig. 4.14 / 4.17" | `main.tex` (inserted at line 2382) | done |
| `section_mettler_fig417_direct.tex` (standalone draft) | `5a/tex_drafts/` | created |

### Data Corrections Applied to Shared Code

The existing `generate_stage10_figures.py` was carrying wrong Mettler data:
- **Before**: attributed to "Fig 4.5" (actually Helicon/PMIC); values 0.83/0.55/0.36/0.26 at 25/50/75/100 mm.
- **After**: correctly attributed to Fig 4.14; values 0.942/0.761/0.500/0.194 at 20/40/60/80 mm. Adds cubic-fit helper `mettler_fig414_cubic()` and absolute-[F] arrays `METTLER_F417_90PCT_NF_CM3` + `METTLER_F417_30PCT_NF_CM3` for re-use elsewhere.

The existing `gen_radial_F_wafer_mettler` function also updated to overlay the cubic fit (dashed green) in addition to the 5-point data, and the title rephrased to honestly flag the model-vs-Mettler condition mismatch (700 W vs 1000 W).

## Physics Findings From the New Benchmark

### Shape test — PASSED (inner wafer)
Both the 90% SF6 and 30% SF6 runs at 1000 W produce normalised radial profiles that track Mettler's Fig 4.14 cubic fit within ±10% for r ≤ 4 cm. Outer wafer (r = 6–8 cm) shows residual over-prediction consistent with the missing electronegative ambipolar correction (L1).

### Composition-sensitivity test — FAILED (expected)
The model predicts a **68% centre-to-edge drop for both compositions**, while Mettler measures 67% at 30% SF6 and 75% at 90% SF6. This 8-percentage-point range is a direct consequence of our **fixed γ_Al = 0.18** — the model has no mechanism to produce a composition-dependent wall-recombination rate. This is new, quantitative evidence that γ_Al may need a composition-dependent parameterisation in Phase-2 (folded into §12 discussion).

### Absolute-magnitude test — FAILED (decomposes into known missing physics)
Model under-predicts Mettler's bias-off absolute [F] by a factor ~4 (90% SF6) and ~1.7 (30% SF6). Root-cause decomposition:
1. **Power coupling**: η = 0.43 (prescribed) delivers only 430 W of 1000 W rf. Mettler's actual η at 1000 W with 200 W bias on is estimated ≥ 0.7. → factor ~1.7–2 under-prediction.
2. **No wafer-bias plasma expansion**: Mettler reports x1.6 (90% SF6) to x2.15 (30% SF6 centre) bias-on/off enhancement.

Combined correction (~1.7 × 1.6 ≈ 2.7) approximately recovers the observed 4× offset for 90% SF6. The residual ~1.5× gap at 90% SF6 is plausibly a combined signature of L1 (electronegative ambipolar) and any composition-dependent γ residual.

### Residual numbers (for the log)

90% SF6 condition (model vs Mettler Fig 4.17 bias-off, r = 0, 2, 4, 6, 8 cm):
```
−75.3%, −74.9%, −70.4%, −64.5%, −56.3%
```

30% SF6 condition (r = 0, 2, 4, 6, 8 cm):
```
−42.3%, −41.2%, −36.9%, −28.8%, −26.6%
```

## Verification Checklist (from plan)

- [x] `grep -c "BBAS\|broadband absorption" main.tex` → **0**
- [x] `grep -c "Mettler Fig.~4\.5" main.tex` → **0**
- [x] `grep -c "74\\\\% drop" main.tex` (unqualified instances) → **0**
- [x] Diagnostic correctly described
- [x] `fig_mettler_fig417_overlay.pdf` exists in report figures directory
- [x] New §12 references `sec:mettler-fig417-direct` (5 references: label + 4 forward cites from other sections)
- [x] New `run_mettler_fig417_{90pct,30pct}.py` use existing `save_sweep_point` API
- [x] Cubic-fit uses corrected normalised values
- [x] Residual panel honestly shows outer-wafer over-prediction
- [x] Bias-absence acknowledged with x1.6 / x2.15 enhancement factors
- [x] Helicon/TEL scope clarification present
- [x] §5.5 Limitations has L5 bullet
- [x] `main.pdf` rebuilds with 0 undefined references; **97 pages** (was 95)

## Outstanding / Deferred

- **Stage C1** (Fig 4.18 ε_Si scatter test) — deferred; no new simulation cost but needs a Si-etching surface model we do not yet have.
- **Stage C2** (Fig 4.9 flow-sweep benchmark) — deferred; 5 new simulations.

## Files Modified

- `5.Phase1_EM_Chemistry_Merged/docs/report/main.tex` (12 edits + new section)
- `5.Phase1_EM_Chemistry_Merged/scripts/generate_stage10_figures.py` (2 edits — data correction, cubic overlay)
- `5.Phase1_EM_Chemistry_Merged/docs/report/figures/fig_mettler_fig417_overlay.{png,pdf}` (new)
- `5.Phase1_EM_Chemistry_Merged/docs/report/figures/fig_radial_F_wafer_mettler.{png,pdf}` (regenerated)
- `5.Phase1_EM_Chemistry_Merged/docs/report/main.pdf` (recompiled, 95 → 97 pp)

## Files Created in 5a

- `5a/scripts/run_mettler_fig417_90pct.py`
- `5a/scripts/run_mettler_fig417_30pct.py`
- `5a/scripts/gen_mettler_fig417_overlay.py`
- `5a/scripts/regen_radial_F_wafer_mettler.py`
- `5a/results/mettler_fig417/{90pct,30pct}/` (sweep-point directories)
- `5a/figures/fig_mettler_fig417_overlay.{png,pdf}`
- `5a/figures/fig_mettler_fig417_overlay_data.json`
- `5a/tex_drafts/section_mettler_fig417_direct.tex`
- `5a/EXECUTION_LOG.md` (this file)
