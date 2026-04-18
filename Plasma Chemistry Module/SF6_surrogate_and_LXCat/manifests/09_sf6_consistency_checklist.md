# CONSISTENCY_CHECKLIST

Figure → script → data traceability audit for `SF6_tier3_repo_merged/`.

---

## Figure provenance matrix

| Figure | Generator script | Primary data dependency | Script exists | Data exists | Figure exists | Status |
|---|---|---|---|---|---|---|
| `fig01_local_vs_nonlocal_Te_kiz.png` | `scripts/generate_figures.py` → `fig01_local_vs_nonlocal()` | `outputs/phaseB_teacher_dataset_multicase.npz`, `outputs/phaseB_teacher_dataset_m2.npz`, `outputs/tier3_pinn_m3c.pt` | ✔ | ✔ | ✔ | **OK** |
| `fig02_pinn_vs_teacher_m3c.png` | `scripts/generate_figures.py` → `fig02_pinn_vs_teacher()` | `outputs/phaseB_teacher_dataset_m2.npz`, `outputs/tier3_pinn_m3c.pt` | ✔ | ✔ | ✔ | **OK** |
| `fig03_error_maps_m3c.png` | `scripts/generate_figures.py` → `fig03_error_maps()` | `outputs/phaseB_teacher_dataset_m2.npz`, `outputs/tier3_pinn_m3c.pt` | ✔ | ✔ | ✔ | **OK** |
| `fig04_lambda_generalization.png` | `scripts/generate_figures.py` → `fig04_lambda_generalization()` | `outputs/phaseB_teacher_dataset_m2.npz`, `outputs/tier3_pinn_m3c.pt` | ✔ | ✔ | ✔ | **OK** |
| `fig05_loss_curves_progression.png` | `scripts/generate_figures.py` → `fig05_loss_progression()` | `outputs/tier3_pinn_{m2,m3,m3b,m3c}.pt` (loss_history dicts) | ✔ | ✔ | ✔ | **OK** |
| `fig06_multicase_comparison.png` | `scripts/generate_figures.py` → `fig06_multicase_comparison()` | `outputs/phaseB_teacher_dataset_multicase.npz`, `outputs/tier3_pinn_multicase.pt` | ✔ | ✔ | ✔ | **OK** |
| `fig07_saturated_region_m3b_vs_m3c.png` | `scripts/generate_figures.py` → `fig07_saturated_region()` | `outputs/phaseB_teacher_dataset_m2.npz`, `outputs/tier3_pinn_{m3b,m3c}.pt` | ✔ | ✔ | ✔ | **OK** |
| `fig08_rotation_summary.png` | `scripts/analyze_rotation_and_variance.py` | `outputs/rotation_holdout{A..E}_seed0.json` | ✔ | ✔ | ✔ | **OK** |
| `fig09_xar_interpolation.png` | `scripts/analyze_rotation_and_variance.py` | `outputs/rotation_holdout{A..E}_seed0.json` + multicase v1 metrics | ✔ | ✔ | ✔ | **OK** |
| `fig10_seed_variance.png` | `scripts/analyze_rotation_and_variance.py` | `outputs/seed_variance_summary.json` | ✔ | ✔ | ✔ | **OK** |
| `fig11_validation_readiness.png` | `scripts/validation_prep.py` | `outputs/validation_prep_summary.json` | ✔ | ✔ | ✔ | **OK** |
| `fig12_power_conditioning.png` | `scripts/analyze_prf_and_seed_variance_v2.py` | `outputs/rotation_prf_holdout{A..E}_seed{0,1}.json`, `outputs/rotation_holdout{A..E}_seed{0,1,2}.json` | ✔ | ✔ | ✔ | **OK** |
| `fig13_seed_variance_all_holdouts.png` | `scripts/analyze_prf_and_seed_variance_v2.py` | `outputs/seed_variance_v2_summary.json` | ✔ | ✔ | ✔ | **OK** |
| `fig14_picmcc_comparison_A.png` | `scripts/compare_picmcc.py` | `outputs/picmcc_reference_TEST_FIXTURE.npz`, `outputs/phaseB_teacher_dataset_multicase_v3.npz`, `outputs/tier3_pinn_multicase.pt` | ✔ | ✔ | ✔ | **OK** (⚠ source is synthetic fixture) |
| `fig15_pressure_axis_closure.png` | `scripts/make_pressure_axis_figure.py` | `outputs/rotation_v{3,4}_table.md` (parsed) or underlying JSONs | ✔ | ✔ | ✔ | **OK** |

## Broken links found

**NONE.** All 15 figures trace back to existing scripts with existing data dependencies.

## Figure → report/slides cross-reference check

| Figure | Referenced in `technical_report.tex`? | Referenced in slides/README/status? |
|---|---|---|
| fig01 | ✔ | ✔ (README, status) |
| fig02 | ✔ | ✔ (README) |
| fig03 | ✔ | ✔ (README) |
| fig04 | ✔ | ✔ (README, status) |
| fig05 | ✔ | ✔ (README) |
| fig06 | ✔ | ✔ (README) |
| fig07 | ✔ | ✔ (README, status) |
| fig08 | ✔ | ✔ (README, status) |
| fig09 | ✔ | ✔ (README) |
| fig10 | ✔ | ✔ (README, status) |
| fig11 | ✔ | ✔ (status) |
| fig12 | ✔ | ✔ (README, status) |
| fig13 | ✔ | ✔ (README, status) |
| fig14 | ✔ | ✔ (README — with synthetic-source caveat) |
| fig15 | ✔ | ✔ (README, status) |

All 15 figures are referenced in at least one documentation file that correctly describes what they show.
