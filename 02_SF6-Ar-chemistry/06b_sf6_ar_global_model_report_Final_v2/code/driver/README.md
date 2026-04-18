# `code/driver/` — Orchestration scripts for 06b

Each script in this folder produces one numerical artifact (CSV) and/or one figure (PNG) needed for the corrected 06b report. All scripts are self-contained with respect to the 06b folder: they use relative paths to `../{lallement_sf6ar,wall_chemistry}/` for code imports and `../../data/mettler/` for Mettler digitised CSVs.

## Which script produces what

| Script | Phase | Output figure | Output CSV | Inputs |
|---|---|---|---|---|
| `run_v3_anchors.py` | A.1 | — | `output/v3_anchor_points.csv` | Lallement 0D, Mettler Fig 4.9b CSVs |
| `run_mettler_fig49_overlay.py` | A.2 | `figures/mettler_fig49_overlay.png` | `output/mettler_fig49_overlay.csv` | Lallement 0D, Mettler Fig 4.9b CSVs |
| `run_mettler_fig417_overlay.py` | A.3 | `figures/mettler_fig417_overlay.png` | `output/mettler_fig417_overlay.csv` | Wall-chemistry 0D, Mettler Fig 4.17 CSVs |
| `run_tel_power_sweep.py` | A.4 | `figures/tel_power_sweep_1000W_30pctAr.png` | `output/tel_power_sweep_1000W_30pctAr.csv` | Wall-chemistry 0D, Mettler Fig 4.17 CSV |
| `run_fig710_2d_regeneration.py` | A.5 | `figures/fig710_radial_F_1000W_regenerated.png` | `output/fig710_radial_profile_1000W.csv` | Cached Stage-7 NPY arrays in `data/stage7_cached/`, Mettler Fig 4.14 + 4.17 CSVs |

## To regenerate all four new figures from scratch

```bash
cd 06b_sf6_ar_global_model_report[Final_v2]/code/driver/
python3 run_v3_anchors.py && \
python3 run_mettler_fig49_overlay.py && \
python3 run_mettler_fig417_overlay.py && \
python3 run_tel_power_sweep.py && \
python3 run_fig710_2d_regeneration.py
```

Total runtime: **~1 minute** (A.5 is near-instant since it loads cached arrays). All scripts save outputs into `06b/figures/` and `06b/output/` using relative paths.

## Dependencies

Standard Python 3.10+ scientific stack:
- `numpy`
- `scipy` (the Lallement / wall-chemistry solvers use `scipy.optimize.brentq`)
- `matplotlib`

No GPU, no `torch`, no `h5py`, no YAML parser required.

## Physics conventions

- Power `P_rf` in watts, pressure `p_mTorr` in milli-Torr, flow `Q_sccm` in standard cm³/min.
- `frac_Ar` is the Ar fraction by volume (0 = pure SF₆, 1 = pure Ar; Mettler Fig 4.17 "90% SF₆" ↔ `frac_Ar = 0.10`).
- [F] densities are m⁻³ throughout (matches Mettler's dissertation figures).
- Lallement geometry (bare 0D): R = 0.180 m, L = 0.175 m.
- TEL geometry (wall-chem): R = 0.105 m, L = 0.0535 m (processing region only).
- Bare Lallement uses η = 0.12; wall-chem extended uses η = 0.16 (these were the calibration values in the original 06a Chapter 3/4 work).

## Non-standalone caveat for A.5

`run_fig710_2d_regeneration.py` reads `nF.npy` from `06b/data/stage7_cached/30pct_SF6_bias_off/` and `90pct_SF6_bias_off/`. These NPY arrays were copied from the Stage-7 hybrid framework (folder `6_Phase1_A .../results/mettler_composition/`) at 06b-bootstrap time. If you need to regenerate the arrays themselves (not just the overlay figure), you must invoke the Stage-7 solver in folder 6 — the 2D solver is not bundled into `06b/code/tel_2d/` in runnable form. See `docs/FIGURE_PROVENANCE.md` for the context.

## Adding more figures later

1. Write a new `run_<purpose>.py` script in this folder.
2. Follow the `HERE = os.path.dirname(os.path.abspath(__file__))` / `REPO = os.path.abspath(os.path.join(HERE, '..', '..'))` pattern for portability.
3. Save figure to `{REPO}/figures/` and traceability CSV to `{REPO}/output/`.
4. Update this README and `docs/FIGURE_PROVENANCE.md`.
