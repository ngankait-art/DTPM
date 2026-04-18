# Figure Provenance — 06b Report

Each figure file in `06b/figures/` is listed below, grouped by the codebase that produced it. This supports reproducibility: to regenerate a figure, find it in this table and invoke the named generator.

---

## New in 06b (produced by driver scripts in `code/driver/`)

| File | Driver | Output destination | Inputs |
|---|---|---|---|
| `tel_power_sweep_1000W_30pctAr.png` | `code/driver/run_tel_power_sweep.py` | saves to `figures/`, CSV to `output/` | Wall-chemistry 0D (`code/wall_chemistry/engine.py`) + Mettler Fig 4.17 CSV |
| `mettler_fig49_overlay.png` | `code/driver/run_mettler_fig49_overlay.py` | saves to `figures/`, CSV to `output/` | Lallement 0D (`code/lallement_sf6ar/sf6_global_model_final.py`) + Mettler Fig 4.9b CSVs |
| `mettler_fig417_overlay.png` | `code/driver/run_mettler_fig417_overlay.py` | saves to `figures/`, CSV to `output/` | Wall-chemistry 0D + Mettler Fig 4.17 CSVs |
| `fig710_radial_F_1000W_regenerated.png` | `code/driver/run_fig710_2d_regeneration.py` | saves to `figures/`, CSV to `output/` | `data/stage7_cached/{30,90}pct_SF6_bias_off/nF.npy` + Mettler Fig 4.14 + 4.17 CSVs |

Plus: `output/v3_anchor_points.csv` is produced by `code/driver/run_v3_anchors.py` (Phase A.1, no figure output — used in N2 text).

---

## Carried over from 06a with known generator

### Group K — Pure SF₆ / Kokkoris figures (Chapter 2)
**Generator**: `code/kokkoris_pure_sf6/plot_figures.py` (from `02_Kokkoris_SF6_Validation/code/`)

Figures:
- `Te_vs_power.png`, `ne_vs_power.png`, `alpha_vs_power.png`, `dp_vs_power.png`
- `Te_vs_poff.png`, `ne_vs_poff.png`, `alpha_vs_poff.png`, `dp_vs_poff.png`
- `nSF6_m3_vs_power.png`, `nF_m3_vs_power.png`, `nF2_m3_vs_power.png`, `nSF4_m3_vs_power.png`, `nSF5_m3_vs_power.png`, `nSF3_m3_vs_power.png`
- `nSF5p_m3_vs_power.png`, `nFm_m3_vs_power.png`, `nSF4p_m3_vs_power.png`, `nSF3p_m3_vs_power.png`, `nSF6m_m3_vs_power.png`
- All `*_vs_poff.png` equivalents
- `nF_with_exp_vs_power.png`, `nF_with_exp_vs_poff.png`
- `overview_neutrals_vs_power.png`, `overview_charged_vs_power.png`
- `overview_neutrals_vs_poff.png`, `overview_charged_vs_poff.png`

### Group L — Lallement SF₆/Ar figures (Chapter 3)
**Generators** (all in `code/lallement_sf6ar/`):
- `sf6_global_model_final.py` — base 0D solver
- `sf6_unified.py` — species and diagnostics plots
- `generate_overlays.py` — Lallement paper overlays
- `extended_analysis.py` — extended parameter studies
- `generate_csv_data.py` — CSV exports

Figures:
- `fig5a_ne_vs_power.png`, `fig5b_Te_vs_power.png`, `fig5c_ne_vs_Ar.png`, `fig5d_Te_vs_Ar.png`
- `fig7_alpha_vs_Ar.png`, `fig8_F_ne_vs_power.png`
- `fig5_overlay.png`, `fig7_overlay.png`, `fig8_overlay.png`
- `fig5_reproduction.png`, `fig7_reproduction.png`, `fig8_reproduction.png`
- `SF6_vs_Ar_species_vs_power.png`
- `Te_density_diagnostics.png`
- `species_overview.png`

### Group S — Kink locus & sensitivity figures (Chapter 6)
**Generator**: `code/kink_analysis/map_kink_locus.py` (from `04_Kink_Analysis_67pct_Ar/code/`)

Figures:
- `kink_locus_heatmap.png`, `kink_locus_curves.png`, `kink_locus_lines.png`

### Wall-chemistry figures (Chapter 4)
**Generator**: `code/wall_chemistry/main.py` (from `03_WallChemistry_Extension/code/`)

Figures:
- `fig1_alpha_chemistry_effect.png`
- `fig2_alpha_3pressures.png`, `fig2_dissociation.png`
- `fig3_dissociation_chemistry.png`
- `fig4_ne_F_vs_power.png`
- `fig5_geometry_effect.png`
- `fig6_AV_sensitivity.png`

---

## Carried over from 06a with UNKNOWN generator

These 11 figures are included verbatim in `06b/figures/` because their generator scripts are not located in any of the bundled codebases. See `COLLABORATOR_REQUESTS.md` for the itemised list to ask collaborators about.

| File | Appears in | Notes |
|---|---|---|
| `raw_vs_refined.png` | Appendix A / §2.7 | Digitisation refinement validation |
| `argon_density_vs_power.png` | §3.9 Extended Analysis | Ar ground / metastable |
| `ne_alpha_vs_Te.png` | §3.9 Extended Analysis | Coupled n_e–α |
| `fig3_plasma_params.png` | §4 wall chemistry | Plasma param evolution |
| `fig4_rate_balance.png` | §4 / §5 kink | Rate balance through kink |
| `fig5_feedback_trajectory.png` | §5 kink | Phase-space trajectory |
| `fig1_kink_overview.png` | §5 kink | Kink phenomenology |
| `rootcause_rate_decomposition.png` | §5.9 | Rate decomposition |
| `rootcause_probability_sensitivity.png` | §5.9 | Wall-probability sensitivity |
| `rootcause_Ar_ionisation_Ec.png` | §5.9 | Ar ionisation fraction + ε_c |
| `eta_sensitivity_TEL.png` | §6.2 | η sensitivity |

---

## Removed from 06b (superseded)

These 6 figures were in 06a but are NOT in 06b. They remain unchanged in `06a/figures/` as the audit baseline.

| File | Why removed |
|---|---|
| `mettler_benchmark_fig4p5.png` | Helicon/PMIC data mis-cited as TEL — replaced by `tel_power_sweep_1000W_30pctAr.png` |
| `fig7_tel_F_vs_power.png` | Same Helicon data reused — replaced by `tel_power_sweep_1000W_30pctAr.png` |
| `mettler_benchmark_fig4p9b.png` | Thin single-branch overlay — replaced by `mettler_fig49_overlay.png` (both branches) |
| `fig8_tel_F_vs_flow.png` | Same Fig 4.9 case reused — replaced by `mettler_fig49_overlay.png` |
| `mettler_benchmark_fig4p17.png` | Thin overlay — replaced by `mettler_fig417_overlay.png` (both compositions) |
| `fig9_tel_center_F.png` | Same Fig 4.17 case reused — replaced by `mettler_fig417_overlay.png` |
