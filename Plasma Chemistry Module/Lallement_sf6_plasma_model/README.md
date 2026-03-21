# SF₆/Ar Global Plasma Model

A zero-dimensional (global) model for SF₆/Ar inductively coupled plasma (ICP) discharges, reproducing and extending the results of Lallement et al., *Plasma Sources Sci. Technol.* **18**, 025001 (2009).

## Overview

This model solves the steady-state particle and power balance equations for a 26-species SF₆/Ar plasma including:
- 9 neutral species (SF₆, SF₅, SF₄, SF₃, SF₂, SF, S, F, F₂)
- 7 positive ions (SF₅⁺ through F⁺, plus Ar⁺)
- 7 negative ions (SF₆⁻ through F₂⁻)
- Ar ground state, Ar⁺, and Ar* metastable

Key physics:
- 54 gas-phase reactions (dissociation, ionization, attachment, neutral recombination)
- Troe fall-off pressure-dependent neutral recombination rates (Ryan & Plumb 1990)
- Penning ionization and Ar* quenching by molecular species
- Lee-Lieberman h-factors for electronegative plasmas
- Maxwellian EEDF

## Quick Start

```bash
pip install numpy scipy matplotlib
python sf6_global_model_final.py
```

This generates 4 figures reproducing the paper's Figs 5, 7, 8 and a species overview.

## Full Reproduction

See [REPRODUCE.md](REPRODUCE.md) for step-by-step instructions. Quick one-liner:

```bash
python sf6_global_model_final.py && python generate_overlays.py && python extended_analysis.py && python generate_csv_data.py && python mettler_benchmark.py
```

## File Structure

```
├── sf6_global_model_final.py       # Core model solver (self-contained)
├── generate_overlays.py            # Overlay plots with digitized Lallement data
├── extended_analysis.py            # Ar density, alpha vs power, Te diagnostics
├── generate_csv_data.py            # Export all results as CSV
├── mettler_benchmark.py            # Independent benchmarking vs Mettler (2025)
├── REPRODUCE.md                    # Step-by-step terminal instructions
├── README.md                       # This file
├── results_interpretation.md       # Physics validation against literature
├── WebPlotDigitizer_Guide.md       # How to digitize paper figures
├── sf6_model_report.tex            # 23-page LaTeX technical report
├── sf6_model_report.pdf            # Compiled report
├── figures/                        # All output figures (15 PNGs)
├── csv_data/                       # Numerical data for all plots (9 files)
├── lallement_digitized/            # WebPlotDigitizer data from Lallement (14 CSVs)
├── mettler_digitized/              # WebPlotDigitizer data from Mettler (6 CSVs)
└── paper_figures_for_digitization/ # Cropped Lallement figures for reference
```

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| R | 0.180 m | Reactor radius |
| L | 0.175 m | Reactor height |
| η | 0.12 | Power coupling efficiency |
| k_rec | 1.5×10⁻⁹ cm³/s | Ion-ion recombination |
| γ_F | 0.01 | F wall recombination probability |
| β_SFx | 0.02 | SFx wall sticking coefficient |
| T_gas | 300 K | Gas temperature |
| T_neg | 0.3 eV | Negative ion temperature |

## API Usage

```python
from sf6_global_model_final import solve_model, sweep_with_continuation

# Single point
result = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.0, eta=0.12)
print(f"ne = {result['ne']*1e-6:.2e} cm⁻³")
print(f"Te = {result['Te']:.2f} eV")
print(f"[F] = {result['n_F']*1e-6:.2e} cm⁻³")
print(f"alpha = {result['alpha']:.1f}")

# Parameter sweep with continuation
import numpy as np
results = sweep_with_continuation('P_rf', np.linspace(200, 2000, 37),
    dict(p_mTorr=10, frac_Ar=0.0, Q_sccm=40, eta=0.12))
```

## Results Summary

At 1500 W, 10 mTorr, pure SF₆:

| Quantity | Model | Lallement calc | Lallement exp |
|----------|-------|---------------|---------------|
| ne (cm⁻³) | 6.1×10⁹ | 6.1×10⁹ | 1.8×10⁹ |
| Te (eV) | 3.00 | 2.94 | 2.01 |
| [F] (cm⁻³) | 1.2×10¹⁴ | 9.7×10¹³ | 9.1×10¹³ |
| α | 35 | 40 | — |

## Known Limitations

1. **Alpha at >30% Ar**: The model's α drops more steeply with Ar dilution than the paper's, due to over-dissociation of SF₆ at intermediate Ar fractions. Root cause: missing wall-mediated recombination (SFx+F→SFx+1 on walls). See report Section 11 for details.

2. **Alpha at 5 mTorr**: Overpredicted by ~3-5× due to h-factor accuracy at very low pressure.

3. **Absolute ne**: Matches paper's experimental values but is ~2.5× below paper's calculated values due to different effective η.

## References

- Lallement et al., *PSST* **18**, 025001 (2009) — primary reference
- Kokkoris et al., *J. Phys. D* **42**, 055209 (2009) — surface chemistry
- Ryan & Plumb, *Plasma Chem. Plasma Process.* **10**, 207 (1990) — Troe rates
- Mettler, *Ph.D. dissertation*, UIUC (2025) — independent benchmarking
- Velazco & Setser, *J. Chem. Phys.* **69**, 4357 (1978) — Ar* quenching

## License

Research use. Please cite Lallement et al. (2009) and this repository.
