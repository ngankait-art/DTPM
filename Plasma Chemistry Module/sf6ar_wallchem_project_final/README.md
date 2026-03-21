# SF₆/Ar Global Model with Wall Surface Chemistry

Extension of the Lallement et al. (2009) SF₆/Ar ICP global model to include
Kokkoris-type heterogeneous wall surface reactions.

## Key Result

Adding wall-mediated SFx+F recombination reduces the α vs Ar% sum-of-squared-errors
by 26× (456 → 17.5), resolving the over-dissociation problem at intermediate Ar fractions.

| Ar% | Original α | Wall chem α | Paper α |
|-----|-----------|-------------|---------|
| 0%  | 35.2      | **40.2**    | 40.1    |
| 30% | 8.4       | **21.6**    | 24.9    |
| 50% | 2.4       | **13.0**    | 14.9    |
| 70% | 1.0       | **4.4**     | 2.7     |

## Quick Start

```bash
pip install numpy scipy matplotlib
python sf6_wallchem_model.py
python wallchem_benchmark.py
```

## Wall Chemistry Parameters

Kokkoris (2009) probabilities scaled by 0.007 for the Lallement reactor:
- s_F = 0.00105 (F adsorption)
- s_SFx = 0.00056 (SFx adsorption)
- p_wallrec = 0.007 (SFx + F(wall) → SFx+1)
- p_fluor = 0.025 (F + SF5(wall) → SF6)
- η = 0.16 (power coupling efficiency)

## Files

- `sf6_wallchem_model.py` — Model with wall_chem option
- `wallchem_benchmark.py` — Benchmarking vs Lallement digitized data
- `wallchem_report.pdf` — 7-page technical report
- `figures/` — 4 comparison figures
- `lallement_digitized/` — WebPlotDigitizer data (14 CSVs)

## References

- Lallement et al., PSST 18, 025001 (2009)
- Kokkoris et al., J. Phys. D 42, 055209 (2009)
