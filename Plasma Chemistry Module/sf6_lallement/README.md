# SF₆/Ar Global Plasma Model

**Reproduction of:** Lallement et al., *Plasma Sources Sci. Technol.* **18**, 025001 (2009)

A zero-dimensional (global) steady-state model for low-pressure SF₆/Ar inductively coupled plasmas (ICP), implemented in Python.

---

## What This Code Does

The model computes steady-state electron density, electron temperature, electronegativity (negative-ion-to-electron density ratio), and neutral species densities in an SF₆/Ar plasma as functions of RF power, gas pressure, and Ar dilution fraction. It includes:

- 26 plasma species (9 neutrals, 7 positive ions, 7 negative ions, Ar, Ar⁺, Ar*)
- 54 gas-phase reactions with Maxwellian rate coefficients
- Penning ionization (Ar* + SF₆ → SF₅⁺ + F + Ar + e)
- Ar* quenching by molecular species
- Pressure-dependent neutral recombination via Troe fall-off formalism
- Electronegative ion transport (Lee & Lieberman h-factors)

---

## Requirements

**Python 3.8+** with the following packages:

```
numpy
scipy
matplotlib
```

No other dependencies are needed. The code uses only standard scientific Python libraries.

### Installing dependencies

```bash
# Using pip
pip install numpy scipy matplotlib

# Or using conda
conda install numpy scipy matplotlib
```

---

## Quick Start

### Run the full model with default parameters

```bash
python sf6_global_model_final.py
```

This will:
1. Compute a single reference point (1500 W, 10 mTorr, pure SF₆)
2. Sweep RF power from 900–1700 W
3. Sweep Ar fraction from 0–100%
4. Compute α vs Ar% at three pressures (5, 10, 20 mTorr)
5. Generate four PNG figure files in the current directory

Output figures:
- `fig5_reproduction.png` — nₑ and Tₑ vs power and Ar fraction
- `fig7_reproduction.png` — electronegativity α vs Ar fraction at 3 pressures
- `fig8_reproduction.png` — [F] and nₑ vs power
- `species_overview.png` — neutral densities and α vs power

Runtime: approximately 2–5 minutes depending on hardware.

---

## Using the Model Programmatically

```python
from sf6_global_model_final import solve_model

# Single operating point
result = solve_model(
    P_rf=1500,       # RF power in watts
    p_mTorr=10,      # Pressure in mTorr
    frac_Ar=0.0,     # Ar fraction (0 = pure SF6, 1 = pure Ar)
    Q_sccm=40,       # Gas flow rate in sccm
    eta=0.12,        # Power coupling efficiency
    gamma_F=0.01,    # F atom wall recombination probability
    beta_SFx=0.02,   # SFx radical wall sticking probability
)

# Access results
print(f"Electron density:    {result['ne']:.2e} m⁻³")
print(f"Electron temperature:{result['Te']:.2f} eV")
print(f"Electronegativity:   {result['alpha']:.1f}")
print(f"[F]:                 {result['n_F']:.2e} m⁻³")
print(f"SF6 dissoc. frac:    {result['dissoc_frac']*100:.0f}%")
```

### Parameter Sweeps with Continuation

For sweeping a parameter across a range, use the continuation-based sweep function which seeds each point with the previous solution:

```python
from sf6_global_model_final import sweep_with_continuation
import numpy as np

# Sweep Ar fraction from 0 to 80%
results = sweep_with_continuation(
    param_name='frac_Ar',
    values=np.linspace(0, 0.8, 17),
    base_kwargs=dict(P_rf=1500, p_mTorr=10, Q_sccm=40, eta=0.12),
    verbose=True
)

# Each element in results is a dict with all plasma quantities
for r in results:
    print(f"Ar={r['frac_Ar']*100:.0f}%: ne={r['ne']*1e-6:.2e} cm⁻³, α={r['alpha']:.1f}")
```

---

## Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `P_rf` | P_rf | 1500 W | RF coupled power |
| `p_mTorr` | p | 10 mTorr | Gas pressure |
| `frac_Ar` | — | 0.0 | Ar fraction in feed (0–1) |
| `Q_sccm` | Q | 40 sccm | Total gas flow rate |
| `eta` | η | 0.12 | Power coupling efficiency |
| `gamma_F` | γ_F | 0.01 | F wall recombination probability |
| `beta_SFx` | β_SFx | 0.02 | SFx wall sticking probability |
| `Tgas` | T_gas | 300 K | Gas temperature |
| `T_neg` | T₋ | 0.3 eV | Negative ion temperature |

### Reactor Geometry

The default reactor is a cylinder with R = 0.180 m, L = 0.175 m, matching the Lallement et al. ICP chamber. To change geometry, modify the `Reactor` class instantiation in `solve_model()`.

---

## Output Dictionary

The `solve_model()` function returns a dictionary with the following keys:

| Key | Units | Description |
|-----|-------|-------------|
| `Te` | eV | Electron temperature |
| `ne` | m⁻³ | Electron density |
| `alpha` | — | Electronegativity (n₋/nₑ) |
| `n_SF6` | m⁻³ | SF₆ density |
| `n_SF5` | m⁻³ | SF₅ density |
| `n_SF4` | m⁻³ | SF₄ density |
| `n_SF3` | m⁻³ | SF₃ density |
| `n_SF2` | m⁻³ | SF₂ density |
| `n_SF` | m⁻³ | SF density |
| `n_S` | m⁻³ | S density |
| `n_F` | m⁻³ | F atom density |
| `n_F2` | m⁻³ | F₂ density |
| `Ec` | eV | Collisional energy loss per ion pair |
| `eps_T` | eV | Total energy cost per ion pair |
| `dissoc_frac` | — | SF₆ dissociation fraction (0–1) |
| `converged` | bool | Whether the solver converged |
| `iter` | — | Number of iterations used |
| `nArm` | m⁻³ | Ar metastable density |
| `R_Penning` | m⁻³s⁻¹ | Penning ionization rate |
| `ns` | dict | All neutral densities as a sub-dict |

---

## File Structure

```
.
├── sf6_global_model_final.py   # Main model code (single file, self-contained)
├── sf6_model_report.pdf        # Technical report (LaTeX-compiled)
├── sf6_model_report.tex        # LaTeX source for the report
├── README.md                   # This file
├── fig5_reproduction.png       # ne and Te vs power and Ar fraction
├── fig7_reproduction.png       # Alpha vs Ar fraction at 3 pressures
├── fig8_reproduction.png       # [F] and ne vs power
└── species_overview.png        # Neutral densities and alpha vs power
```

---

## Reproducing the Figures

To regenerate all figures from scratch:

```bash
python sf6_global_model_final.py
```

Figures are saved in the current working directory. To change the output path, modify the `plt.savefig()` calls in the `if __name__ == '__main__'` block.

### Custom Figure Generation

```python
import matplotlib.pyplot as plt
import numpy as np
from sf6_global_model_final import solve_model, sweep_with_continuation

# Example: alpha vs pressure at fixed 1500W, 50% Ar
pressures = np.linspace(3, 30, 20)
results = sweep_with_continuation(
    'p_mTorr', pressures,
    dict(P_rf=1500, frac_Ar=0.5, eta=0.12)
)

plt.figure()
plt.plot(pressures, [r['alpha'] for r in results], 'o-')
plt.xlabel('Pressure (mTorr)')
plt.ylabel('α = n₋/nₑ')
plt.title('Electronegativity vs Pressure at 50% Ar')
plt.savefig('alpha_vs_pressure.png', dpi=150)
```

---

## Numerical Details

### Solver Convergence

The solver uses a hybrid iterative scheme:
1. Sequential substitution for neutral densities (inner loop, 20 iterations)
2. Brent root-finding for Tₑ from the particle balance
3. Power balance for nₑ
4. Quadratic formula for α
5. Geometric relaxation (w = 0.08) on all outer variables

Typical convergence requires 80–200 outer iterations. The solver declares convergence when relative changes in Tₑ, nₑ, and α fall below 5×10⁻⁵, 5×10⁻⁴, and 10⁻³ respectively.

### Known Numerical Issues

- At very low pressure (< 3 mTorr), the solver may converge slowly or find a spurious electropositive solution. Use continuation from a higher-pressure starting point.
- At very high Ar fraction (> 95%) with low pressure (< 5 mTorr), the α value may oscillate. Reduce the relaxation weight if this occurs.
- Pure Ar (frac_Ar = 1.0) is a limiting case with α = 0 by definition.

---

## Physical Assumptions and Limitations

1. **0D model**: No spatial gradients. Results represent volume-averaged quantities.
2. **Steady state**: No time dependence. Not suitable for pulsed plasmas.
3. **Maxwellian EEDF**: Rate coefficients assume a Maxwellian electron energy distribution. The Lallement paper also uses Maxwellian; a Druyvesteyn EEDF would give ~0.8 eV lower Tₑ.
4. **No wafer surface chemistry**: The model does not include Si etching reactions.
5. **No polymer/deposit chemistry**: S₂F₁₀ and fluoro-sulfur film formation are not included.
6. **Simplified ion transport**: Uses Lee & Lieberman h-factors rather than solving the full Boltzmann equation for ion transport.

---

## Citation

If you use this code, please cite:

- L. Lallement, A. Rhallabi, C. Cardinaud, M.C. Peignon-Fernandez, and L.L. Alves, *Plasma Sources Sci. Technol.* **18**, 025001 (2009). [Original paper]
- K.R. Ryan and I.C. Plumb, *Plasma Chem. Plasma Process.* **10**, 207 (1990). [Neutral recombination rates]
- G. Kokkoris, A. Panagiotopoulos, A. Goodyear, M. Cooke, and E. Gogolides, *J. Phys. D: Appl. Phys.* **42**, 055209 (2009). [SF₆ surface kinetics model]

---

## License

This code is provided for academic and research use. The reaction rate data are from published literature sources cited above.
