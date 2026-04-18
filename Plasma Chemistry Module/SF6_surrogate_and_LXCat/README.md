# Final Package: SF6 ICP Plasma Surrogate Modeling

## Deliverables

| File | Description |
|---|---|
| `report/main.tex` | Full paper-style LaTeX report (~30 pages, 28 figures) |
| `report/main.pdf` | Compiled report PDF |
| `slides/main.tex` | Beamer slide deck (Metropolis theme, 16:9) |
| `slides/main.pdf` | Compiled slides PDF |
| `speaker_notes/main.tex` | Standalone speaker notes document |
| `speaker_notes/main.pdf` | Compiled speaker notes PDF |

## Directory Structure

```
final_package/
├── report/
│   ├── main.tex          # Full manuscript
│   ├── refs.bib          # Bibliography
│   └── main.pdf          # Compiled PDF
├── slides/
│   ├── main.tex          # Beamer deck
│   └── main.pdf          # Compiled PDF
├── speaker_notes/
│   ├── main.tex          # Speaking script
│   └── main.pdf          # Compiled PDF
├── figures/              # Shared figure assets (36 .png files)
├── data_summary/
│   ├── final_metrics.json    # Machine-readable results
│   └── figure_inventory.md   # Figure audit trail
├── code_snapshot/        # Key scripts for reproducibility
└── README.md             # This file
```

## How to Compile

All three documents use figures from `../figures/` relative to their directory.

### Report
```bash
cd report/
tectonic main.tex          # or: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Slides
```bash
cd slides/
tectonic main.tex          # or: pdflatex main.tex
```

### Speaker Notes
```bash
cd speaker_notes/
tectonic main.tex          # or: pdflatex main.tex
```

## Key Results

- **Winning model**: surrogate_lxcat_v4_arch (E3 separate heads, 5-model ensemble)
- **nF RMSE**: 0.00154 (log10 scale) --- 47% better than legacy surrogate_v4
- **nSF6 RMSE**: 0.00128 (log10 scale)
- **Speedup**: >500x over FD solver
- **Key insight**: The LXCat accuracy gap was training recipe (bias initialization), not data difficulty

## Data Provenance

All results derive from:
- `active_projects/tel_model/results/` --- raw experimental outputs
- `active_projects/tel_model/src/` --- training and analysis scripts
- `active_projects/tel_model/results/pinn_dataset_lxcat_v3/` --- 221-case LXCat solver dataset
- `active_projects/tel_model/results/pinn_dataset_v4/` --- 221-case legacy solver dataset

## Assumptions and Caveats

- Solver is 2D axisymmetric (no 3D asymmetries)
- Steady-state only (no transient dynamics)
- Wall recombination coefficients from literature, not measured for this reactor
- F density has 36% mean relative error vs Mettler 2020 measurements
- LXCat Biagi cross sections overpredict Te by 58% vs Lallement 2009
- Surrogate trained on solver outputs, not experimental data
- Speedup numbers measured under CPU contention; solo runs would show higher speedup
