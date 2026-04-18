# Collaborator Requests — Missing Figure Generators

This document enumerates **11 figures** that appear in the 06b report but whose generator source code is not bundled inside `06b/code/`. They are carried over as PNG files from 06a verbatim; to enable future regeneration.


Looking for the source codes of the kink-analysis, root-cause, and η-sensitivity plots which were generated as part of his March–April 2026 DTPM delivery.

---

## The 11 missing generators

For each figure, this table lists: the filename, the chapter and section it appears in, a brief description of what it shows, and the best guess for where the generator might live based on content and existing folder structure.

### Chapter 3 (§3.9 Extended Analysis) — 2 figures

| File | Content | Likely location of generator |
|---|---|---|
| `argon_density_vs_power.png` | Ar ground-state density and Ar* metastable density vs ICP power | Most likely an extension of `01_Lallement_SF6Ar_GasPhase_Model/code/extended_analysis.py` — check the file for a plot function we missed |
| `ne_alpha_vs_Te.png` | Coupled n_e–α dependence on T_e diagnostic | Probably a diagnostic plot in `extended_analysis.py` or `sf6_unified.py` |

### Chapter 4 (§4 wall chemistry) — 2 figures

| File | Content | Likely location |
|---|---|---|
| `fig3_plasma_params.png` | Plasma parameter evolution (n_e, T_e, α) with wall chemistry enabled | Expected in `03_WallChemistry_Extension/code/main.py` but not located in scan |
| `fig4_rate_balance.png` | Rate balance (production vs loss channels) across the kink | Likely in `main.py` or a kink-specific script |

### Chapter 5 (Kink / root-cause) — 5 figures

| File | Content | Likely location |
|---|---|---|
| `fig1_kink_overview.png` | Kink phenomenology overview panel | Expected in `04_Kink_Analysis_67pct_Ar/code/` — not located in scan (`map_kink_locus.py` has the locus heatmaps but not this overview) |
| `fig5_feedback_trajectory.png` | Phase-space feedback-loop trajectory visualisation | Expected in `04_Kink_Analysis_67pct_Ar/code/` |
| `rootcause_rate_decomposition.png` | Rate decomposition (electron-impact dissociation vs wall regeneration) through the kink | Expected co-located with `rootcause_diagnostic_data.csv` (which IS bundled in `data/`) but the plotting script is not |
| `rootcause_probability_sensitivity.png` | Surface-probability sensitivity (s_F, s_SFx, p_wallrec) perturbation test | Same — expected co-located with `rootcause_diagnostic_data.csv` |
| `rootcause_Ar_ionisation_Ec.png` | Ar ionisation fraction and collisional energy cost ε_c through the kink | Same |

### Chapter 6 (§6.2 η sensitivity) — 1 figure

| File | Content | Likely location |
|---|---|---|
| `eta_sensitivity_TEL.png` | η sensitivity curves (five values of η, [F] and n_e vs power) for the TEL geometry | Likely in `09_TEL_Simulation_Full_Package/` or `10_TEL_Simulation_Full_Package_04102026/` scripts |

### Appendix A / §2.7 — 1 figure

| File | Content | Likely location |
|---|---|---|
| `raw_vs_refined.png` | Comparison of raw WebPlotDigitizer output vs refined Kokkoris benchmark curves | Likely a post-processing script in `02_Kokkoris_SF6_Validation/code/` (e.g. `refine_data.py` is in that folder but doesn't match this figure's content) |

---

## Once you receive the code

Drop it into the appropriate subfolder under `06b/code/` (e.g. kink-related scripts into `code/kink_analysis/`, η-sensitivity into a new `code/eta_sensitivity/` folder), then update `FIGURE_PROVENANCE.md` to move the file from the UNKNOWN section to the known-generator section.
