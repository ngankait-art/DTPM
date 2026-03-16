# SF₆/Ar 2D Axisymmetric ICP Plasma Simulator

**A hybrid 0D+2D fluid model for low-pressure SF₆/Ar inductively coupled plasma discharges, developed for the Digital Twin for Plasma Manufacturing (DTPM) project.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

---

## Abstract

This repository contains a multi-generation plasma simulation framework that models SF₆/Ar inductively coupled plasmas (ICPs) used for silicon etching.  The model evolves from a 0D global chemistry backbone (26 species, 54 reactions) through five spatial generations, culminating in a fully validated 2D axisymmetric solver that reproduces the 75% center-to-edge fluorine density drop measured experimentally by Mettler (2025).

**Key numbers:** 4,974 lines (unified) | 27 modules | 6 generations | 30–40 s per run | validated against Lallement (2009) and Mettler (2025)

---

## Scientific Scope

- **Plasma type:** Low-pressure (5–50 mTorr) SF₆/Ar ICP at 13.56 MHz
- **Application:** Silicon deep reactive ion etching (DRIE), Bosch process
- **Key output:** Radial fluorine atom density [F](r) at the wafer surface → etch rate and uniformity
- **Chemistry:** 26 species, 54 reactions including Troe fall-off, Penning ionization, dissociative attachment
- **Validation targets:** Electron density, electron temperature, electronegativity, and fluorine profile shape

---

## Model Generations

| Gen | Driver | Key advance | [F] drop | Status |
|-----|--------|-------------|----------|--------|
| 1 | `main_gen1.py` | 0D backbone + 2D diffusion eigenmode | 0% | Archival |
| 2 | `main_gen2.py` | Negative-ion transport, Lichtenberg stratification | 0% | Archival |
| 3 | `main_gen3.py` | Energy equation, self-consistent [F] | 3% | Archival |
| 4/4b | `main_gen4b.py` | Robin BCs, Hagelaar transport, wall-specific γ_F | 73% | Usable |
| **5** | **`main_gen5.py`** | **Self-consistent ne, sheath, gas T, multi-ion** | **75%** | **Recommended** |
| — | **`sf6_icp_unified.py`** | **All-in-one single file (4,974 lines)** | **75%** | **Recommended** |

---

## Quick Start

### Requirements
```bash
pip install numpy scipy matplotlib
```

### Run the unified code
```bash
# Default: Gen-5, pure SF6, 1500W, 10 mTorr
python code/unified/sf6_icp_unified.py

# Custom conditions
python code/unified/sf6_icp_unified.py --power 1000 --ar 0.3 --outdir my_output

# Parameter sweep
python code/unified/sf6_icp_unified.py --scan --outdir sweep_results

# Gen-4b instead
python code/unified/sf6_icp_unified.py --gen 4

# Skip plots (CSV only)
python code/unified/sf6_icp_unified.py --no-plot
```

### Output files per run
| File | Contents |
|------|----------|
| `summary.csv` | Scalar results: ne, Te, α, [F], etch rate |
| `radial_midplane.csv` | ne, Te, [F], nSF6, α vs r at z=L/2 |
| `radial_wafer.csv` | ne, Te, [F], etch_rate vs r at z=0 |
| `axial_axis.csv` | ne, Te, [F], P_ind vs z at r=0 |
| `fields_2d.csv` | Full 2D fields (r, z, ne, Te, [F], P_ind, α) |
| `wall_flux_wafer.csv` | Ion flux and energy at wafer |
| `profiles_2d.png` | 12-panel 2D contour plot |
| `F_radial.png` | Radial [F] + Mettler comparison |
| `etch_rate.png` | Si etch rate profile |
| `ion_wafer.png` | Ion flux + energy at wafer |

---

## Repository Structure

```
sf6_icp_2d_final/
├── README.md                          ← This file
├── LICENSE.txt                        ← MIT license
├── requirements.txt                   ← Python dependencies
├── CITATION.cff                       ← Citation metadata
├── .gitignore
│
├── code/
│   ├── generation_1/                  ← Gen-1: diffusion eigenmode
│   ├── generation_2/                  ← Gen-2: negative ions + neutrals
│   ├── generation_3/                  ← Gen-3: energy equation + F diffusion
│   ├── generation_4/                  ← Gen-4/4b: Robin BCs + Hagelaar
│   ├── generation_5/                  ← Gen-5: full physics
│   ├── unified/
│   │   └── sf6_icp_unified.py         ← ★ SINGLE-FILE SOLVER (recommended)
│   └── shared_modules/                ← Chemistry, mesh, solvers, transport
│
├── docs/
│   ├── final_report.pdf               ← 24-page LaTeX report
│   ├── final_report.tex               ← LaTeX source
│   └── figures/                       ← Figures for the report
│
├── figures/
│   ├── essential/                     ← 14 core figures (fig_01–fig_14)
│   └── validation/                    ← 2 Mettler comparison figures
│
├── data/
│   ├── generated_csv/                 ← Example CSV outputs
│   ├── validation_data/               ← Mettler digitized data (in code)
│   └── transport_data/                ← Ar Boltzmann table (.npz)
│
├── scripts/
│   ├── reproduce_essential_figures.sh ← Regenerate core figures
│   ├── reproduce_all_figures.sh       ← Full reproduction
│   └── build_report.sh               ← Compile LaTeX report
│
├── outputs/
│   └── example_outputs/               ← Pre-generated outputs (CSV + PNG)
│
├── manifests/
│   ├── figure_manifest.csv            ← Map of all figures
│   ├── results_summary.csv            ← Cross-generation validation
│   └── file_manifest.txt              ← Complete file listing
│
└── archive/
    └── legacy_materials/              ← Earlier report versions
```

---

## Reproducing Results

### Essential figures (4 runs, ~3 minutes)
```bash
bash scripts/reproduce_essential_figures.sh
```

### Full figure set including parameter sweeps
```bash
bash scripts/reproduce_all_figures.sh
```

### Compile the LaTeX report
```bash
bash scripts/build_report.sh
```

---

## Validation Summary

| Quantity | Model (Gen-5) | Experiment | Source | Agreement |
|----------|---------------|------------|--------|-----------|
| ⟨ne⟩ | 3.4×10⁹ cm⁻³ | ~6×10⁹ | Lallement 2009 | Factor 1.8 |
| ⟨Te⟩ | 2.99 eV | 3.0 eV | Lallement 2009 | < 1% |
| α | 64.6 | 30–40 | Lallement 2009 | Same order |
| **[F] profile drop** | **75%** | **75%** | **Mettler 2025** | **2% residual** |
| ne ∝ P | Linear | Linear | Lallement 2009 | Correct trend |
| Te ∝ 1/p | Confirmed | Confirmed | Lallement 2009 | Correct trend |

**Note:** The Mettler reactor (TEL, 2 MHz) differs from the model geometry (Lallement-type, 13.56 MHz). Profile shapes are compared; absolute values differ by ~12×.

---

## Limitations

- **γ_F is calibrated** (0.30 at sidewall) — not predicted from first principles
- **Geometry mismatch** with Mettler reactor — absolute [F] values differ
- **SF₆ rates use Arrhenius fits** — BOLSIG+ tables available only for Ar
- **No sheath resolution** — mesh Δr ≈ 2 mm vs λ_D ≈ 0.1 mm
- **Gas temperature** negligible at 10 mTorr (Tg = 300 K)

---

## Recommended Entry Point

**New users should start with the unified code:**
```bash
python code/unified/sf6_icp_unified.py --help
```

To understand the physics, read `docs/final_report.pdf` (24 pages, all equations and figures explained).

To understand the code evolution, browse `code/generation_*/README.md` from Gen-1 through Gen-5.

---

## References

1. Lallement et al., *Plasma Sources Sci. Technol.* **18**, 025001 (2009)
2. Lichtenberg et al., *J. Appl. Phys.* **75**, 2339 (1994)
3. Vahedi et al., *J. Appl. Phys.* **78**, 1446 (1995)
4. Lymberopoulos & Economou, *J. Appl. Phys.* **73**, 3668 (1993)
5. Hagelaar & Pitchford, *Plasma Sources Sci. Technol.* **14**, 722 (2005)
6. Mettler, PhD dissertation, UIUC (2025)
7. Phelps database, www.lxcat.net (Ar cross-sections)
8. Biagi database, www.lxcat.net (SF₆ cross-sections, Magboltz v10.6)
