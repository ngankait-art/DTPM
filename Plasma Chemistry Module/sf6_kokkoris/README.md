# SF6 Global Plasma Model

**Reproduction of:** Kokkoris et al., J. Phys. D: Appl. Phys. 42 (2009) 055209

---

## Quick Start — Reproduce All Results in 3 Commands

Open a terminal in VS Code (Ctrl+` or Terminal > New Terminal), then:

```bash
pip install numpy scipy matplotlib
python plot_results.py --outdir output
python plot_overlay.py
```

All results appear in `output/` (figures + CSV) and `output_overlay/` (benchmark overlays).

Expected runtime: ~5 minutes total.

---

## What Gets Generated

### CSV Data Files

| File | Rows | Description |
|------|------|-------------|
| `output/power_sweep_results.csv` | 19 | All species densities, Te, ne, alpha, pressure rise vs power |
| `output/pressure_sweep_results.csv` | 13 | Same quantities vs pre-discharge pressure |

Each CSV has 22 columns: Power, pOFF, Te, ne, alpha, dp, 6 neutral densities, 6 ion densities, 4 surface coverages.

### Model Figures (7 files in `output/`)

| File | Description |
|------|-------------|
| `pressure_rise_and_F_density_vs_power.png` | Pressure rise and F density vs power |
| `pressure_rise_and_F_density_vs_pOFF.png` | Pressure rise and F density vs pressure |
| `species_densities_vs_power.png` | All species vs power (2-panel) |
| `species_densities_vs_Te_power_sweep.png` | All species vs Te from power sweep |
| `species_densities_vs_Te_pressure_sweep.png` | All species vs Te from pressure sweep |
| `all_species_and_Te_vs_power.png` | 3-panel overview vs power |
| `all_species_and_Te_vs_pOFF.png` | 3-panel overview vs pressure |

### Benchmark Overlays (29 files in `output_overlay/`)

22 individual species overlays (one species per graph, paper vs model) plus 7 combined overlays.

---

## Run a Single Operating Point

```bash
python -c "
from sf6_ode_solver import run_to_steady_state
from sf6_global_model import NAMES, N_SP
r = run_to_steady_state(p_OFF=0.921, P_abs=2000, t_max=1.0, verbose=True)
for i in range(N_SP):
    print(f'  {NAMES[i]:6s} = {r[\"n\"][i]:.3e} m^-3')
print(f'  Te = {r[\"Te\"]:.3f} eV')
print(f'  dp = {r[\"dp\"]:.4f} Pa')
print(f'  alpha = {r[\"alpha\"]:.2f}')
"
```

## Run a Custom Sweep and Export CSV

```bash
python -c "
from plot_results import run_power_sweep, write_csv
results = run_power_sweep(p_OFF=2.0, powers=[500, 1000, 1500, 2000, 2500, 3000])
write_csv(results, 'my_custom_sweep.csv')
"
```

## Recompile LaTeX Report

```bash
pdflatex SF6_FINAL_REPORT.tex
pdflatex SF6_FINAL_REPORT.tex
```

---

## File Structure

```
sf6_model/
├── sf6_global_model.py          # Core model
├── sf6_ode_solver.py            # ODE solver
├── plot_results.py              # Generate figures + CSV
├── plot_overlay.py              # Generate benchmark overlays
├── paper_digitized.json         # Digitized paper data
├── README.md                    # This file
├── SF6_FINAL_REPORT.tex/.pdf    # Full report (15 pages)
├── SF6_model_documentation.tex/.pdf  # Equations document (9 pages)
├── HOW_TO_REPRODUCE_A_GLOBAL_PLASMA_MODEL.md
├── FIGURE_DESCRIPTIONS.md
├── SF6_MODEL_EQUATIONS_AND_CHEMISTRY.md
└── figs/                        # Pre-generated figures for LaTeX
```

## Requirements

Python 3.8+, NumPy, SciPy, Matplotlib. No network access needed.
