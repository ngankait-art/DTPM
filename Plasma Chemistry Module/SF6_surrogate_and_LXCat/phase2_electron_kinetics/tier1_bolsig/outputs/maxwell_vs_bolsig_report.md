# Tier 1: Maxwell vs BOLSIG+ rate comparison

## Reference operating point

- E/N = 50 Td (closest to nominal 50.0 Td)
- x_Ar = 0 (pure SF6)

| Channel | Maxwellian rate (m3/s) | Boltzmann rate (m3/s) | Ratio M/B | Within 20%? |
|---|---|---|---|---|
| `k_iz` | 1.080e-21 | 0.000e+00 | nan | NO |
| `k_att` | 6.031e-15 | 7.391e-15 | 0.816 | yes |
| `k_diss` | 8.148e-14 | 7.824e-14 | 1.04 | yes |
| `k_el` | 9.878e-14 | 1.018e-13 | 0.97 | yes |

## Decision (workplan §3.4)

**At least one dominant rate differs by more than 20%** at the reference point. The Maxwellian assumption introduces significant error; Tier 2 surrogate is essential, and the production DTPM model should use BOLSIG+/Tier 2 rates rather than Arrhenius forms.

## Full pure-SF6 sweep

The plot at `plots/ratio_vs_EN.png` shows the Maxwell/Boltzmann ratio for each channel across the full E/N range. The ratio is approximately unity for low-energy-dominated channels (attachment, elastic momentum transfer) and departs strongly from unity for tail-dominated channels (ionisation, dissociative excitation) where the Maxwellian over-populates the high-energy tail that drives those rates.