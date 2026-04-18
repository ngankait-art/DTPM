# Quantitative Literature Validation

## LEGACY solver

### F density vs power (Mettler et al., J. Vac. Sci. Technol. A 38, 023011 (2020))

| Power (W) | Sim (m^-3) | Exp (m^-3) | Rel Error |
|---|---|---|---|
| 300 | 2.78e+19 | 1.50e+19 | 85.6% |
| 500 | 3.70e+19 | 3.00e+19 | 23.4% |
| 700 | 4.33e+19 | 5.00e+19 | 13.4% |
| 900 | 4.80e+19 | 6.50e+19 | 26.2% |
| 1100 | 5.16e+19 | 7.50e+19 | 31.2% |

**Mean relative error: 36.0%**

### Te vs power (Lallement et al., J. Phys. D 42, 015203 (2009))

| Power (W) | Sim (eV) | Exp (eV) | Rel Error |
|---|---|---|---|
| 200 | 2.94 | 3.20 | 8.2% |
| 400 | 2.94 | 3.00 | 2.1% |
| 600 | 2.94 | 2.90 | 1.2% |
| 800 | 2.94 | 2.80 | 4.9% |
| 1000 | 2.94 | 2.70 | 8.7% |

**RMSE: 0.17 eV, Mean relative error: 5.0%**

## LXCAT solver

### F density vs power (Mettler et al., J. Vac. Sci. Technol. A 38, 023011 (2020))

| Power (W) | Sim (m^-3) | Exp (m^-3) | Rel Error |
|---|---|---|---|
| 300 | 2.81e+19 | 1.50e+19 | 87.6% |
| 500 | 3.76e+19 | 3.00e+19 | 25.3% |
| 700 | 4.41e+19 | 5.00e+19 | 11.8% |
| 900 | 4.89e+19 | 6.50e+19 | 24.7% |
| 1100 | 5.27e+19 | 7.50e+19 | 29.8% |

**Mean relative error: 35.9%**

### Te vs power (Lallement et al., J. Phys. D 42, 015203 (2009))

| Power (W) | Sim (eV) | Exp (eV) | Rel Error |
|---|---|---|---|
| 200 | 4.59 | 3.20 | 43.6% |
| 400 | 4.59 | 3.00 | 53.1% |
| 600 | 4.59 | 2.90 | 58.4% |
| 800 | 4.59 | 2.80 | 64.1% |
| 1000 | 4.59 | 2.70 | 70.1% |

**RMSE: 1.68 eV, Mean relative error: 57.9%**

