# How to Reproduce All Results

Step-by-step instructions for reproducing every figure, CSV data file, and the LaTeX report from scratch using a terminal (e.g., VS Code integrated terminal).

---

## Prerequisites

Install Python 3.8+ and the required packages:

```bash
pip install numpy scipy matplotlib
```

For recompiling the LaTeX report (optional):
```bash
# Ubuntu/Debian
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# macOS (with Homebrew)
brew install --cask mactex

# Or use Overleaf (upload sf6_model_report.tex and all figure PNGs)
```

---

## Step 1: Generate All Figures (required)

Open a terminal in the project directory and run:

```bash
# Generate the main reproduction figures (Fig 5, 7, 8, species overview)
python sf6_global_model_final.py
```

This produces:
- `fig5_reproduction.png` — ne and Te vs power and Ar fraction
- `fig7_reproduction.png` — Alpha vs Ar% at 3 pressures
- `fig8_reproduction.png` — [F] and ne vs power
- `species_overview.png` — Neutral densities and alpha vs power

Runtime: ~3–5 minutes.

---

## Step 2: Generate Overlay Figures with Digitized Paper Data

```bash
python generate_overlays.py
```

This produces:
- `fig5_overlay.png` — Same as Fig 5 but with paper data points overlaid
- `fig7_overlay.png` — Same as Fig 7 with paper data overlaid
- `fig8_overlay.png` — Same as Fig 8 with paper data overlaid

Runtime: ~3–5 minutes.

---

## Step 3: Generate Extended Analysis Figures

```bash
python extended_analysis.py
```

This produces:
- `argon_density_vs_power.png` — Ar ground-state and Ar* vs power
- `alpha_vs_power.png` — Alpha vs power at 4 Ar fractions
- `Te_density_diagnostics.png` — Species densities vs Te
- `ne_alpha_vs_Te.png` — ne and alpha vs Te
- `SF6_vs_Ar_species_vs_power.png` — SF6 vs Ar species comparison

Runtime: ~4–6 minutes.

---

## Step 4: Generate CSV Data Files

```bash
python generate_csv_data.py
```

This creates a `csv_data/` folder with 9 CSV files:

| File | Description | Rows |
|------|-------------|------|
| `power_sweep_0pct_Ar.csv` | All species vs power, pure SF6 | 37 |
| `power_sweep_20pct_Ar.csv` | All species vs power, 20% Ar | 37 |
| `power_sweep_50pct_Ar.csv` | All species vs power, 50% Ar | 37 |
| `power_sweep_80pct_Ar.csv` | All species vs power, 80% Ar | 37 |
| `ar_fraction_sweep_1500W.csv` | Key quantities vs Ar%, 1500W | 21 |
| `alpha_vs_Ar_5mTorr.csv` | Alpha vs Ar% at 5 mTorr | 17 |
| `alpha_vs_Ar_10mTorr.csv` | Alpha vs Ar% at 10 mTorr | 17 |
| `alpha_vs_Ar_20mTorr.csv` | Alpha vs Ar% at 20 mTorr | 17 |
| `summary_at_1500W_10mTorr.csv` | Summary table at reference conditions | 14 |

Runtime: ~4–6 minutes.

---

## Step 5: Compile the LaTeX Report (optional)

Copy all figure PNGs into the same directory as the .tex file, then:

```bash
# Copy figures next to the tex file
cp figures/*.png .
cp fig5_overlay.png fig7_overlay.png fig8_overlay.png .
cp argon_density_vs_power.png alpha_vs_power.png .
cp Te_density_diagnostics.png ne_alpha_vs_Te.png .
cp SF6_vs_Ar_species_vs_power.png species_overview.png .

# Compile (run twice for cross-references)
pdflatex sf6_model_report.tex
pdflatex sf6_model_report.tex
```

This produces `sf6_model_report.pdf` (21 pages).

---

## Quick One-Liner: Run Everything

```bash
python sf6_global_model_final.py && python generate_overlays.py && python extended_analysis.py && python generate_csv_data.py
```

---

## File Descriptions

### Python Scripts

| File | Purpose | Depends on |
|------|---------|------------|
| `sf6_global_model_final.py` | Core model solver + main figures | numpy, scipy, matplotlib |
| `generate_overlays.py` | Overlay plots with digitized paper data | sf6_global_model_final.py |
| `extended_analysis.py` | Ar density, alpha vs power, Te diagnostics | sf6_global_model_final.py |
| `generate_csv_data.py` | Export all numerical data as CSV | sf6_global_model_final.py |

### Documentation

| File | Contents |
|------|----------|
| `README.md` | Model overview, API usage, parameter table |
| `REPRODUCE.md` | This file — step-by-step reproduction instructions |
| `results_interpretation.md` | Physics validation against 6 literature sources |
| `WebPlotDigitizer_Guide.md` | How to digitize paper figures for overlays |
| `sf6_model_report.tex` | LaTeX source for the 21-page technical report |
| `sf6_model_report.pdf` | Compiled report (pre-built) |

### Output Figures (12 total)

| File | What it shows |
|------|--------------|
| `fig5_reproduction.png` | ne, Te vs power and Ar% (model only) |
| `fig5_overlay.png` | Same with digitized paper data overlaid |
| `fig7_reproduction.png` | Alpha vs Ar% at 5/10/20 mTorr (model only) |
| `fig7_overlay.png` | Same with digitized paper data overlaid |
| `fig8_reproduction.png` | [F] and ne vs power (model only) |
| `fig8_overlay.png` | Same with digitized paper data overlaid |
| `species_overview.png` | Neutral densities + alpha vs power |
| `argon_density_vs_power.png` | Ar and Ar* densities vs power |
| `alpha_vs_power.png` | Alpha vs power at 4 Ar fractions |
| `Te_density_diagnostics.png` | Species densities plotted against Te |
| `ne_alpha_vs_Te.png` | ne and alpha plotted against Te |
| `SF6_vs_Ar_species_vs_power.png` | Side-by-side SF6 vs Ar species |

### CSV Data (9 files in csv_data/)

Each CSV has a header row with column names and units.
All densities are in cm⁻³. Temperatures in eV. Power in Watts.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'sf6_global_model_final'"**
→ Make sure all .py files are in the same directory. The scripts import from `sf6_global_model_final.py`.

**Figures not appearing**
→ The scripts use `matplotlib.use('Agg')` (non-interactive backend) and save to the current directory. Check that you have write permissions.

**LaTeX compilation errors**
→ Ensure all 12 PNG figures are in the same directory as the .tex file. The report uses `\includegraphics{}` with bare filenames (no path prefix).

**Slow runtime**
→ Each script takes 3–6 minutes on a modern machine. The solver uses ~200 iterations per operating point with continuation. Total runtime for all 4 scripts: ~15–20 minutes.
