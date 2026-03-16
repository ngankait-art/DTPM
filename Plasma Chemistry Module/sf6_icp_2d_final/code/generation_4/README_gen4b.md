# Generation 4b — Final Model (DT-Ready)

**This is the recommended generation for all production use.**

## What it adds over Gen-4

- **Wall-specific Robin BCs**: different γ_F at each wall surface
  - Sidewall (r=R): γ_F = 0.30 (anodized Al, highest recombination)
  - Wafer (z=0): γ_F = 0.02 (Si surface, moderate)
  - Window (z=L): γ_F = 0.001 (quartz, very low)
- **Self-consistent SF6 depletion**: SF6 solved by diffusion, no full anchor;
  partial floor at 80% of 0D prevents runaway depletion
- **Source-weighted ne profile**: ne shape from spatially varying ionization
  rate nu_iz(Te(r,z)) instead of uniform-source eigenmode.
  Peak/avg increases from 2.5 to 3.5.
- **73% center-to-edge [F] drop**, matching Mettler's measured 75%

## How to run

```bash
# PRIMARY VALIDATION CASE: pure SF6, 1500W, 10 mTorr
python main_v4b.py --power 1500 --pressure 10 --ar 0.0 --gamma-wall 0.30

# Mettler comparison conditions: 70/30 SF6/Ar, 1000W
python main_v4b.py --power 1000 --pressure 10 --ar 0.3 --gamma-wall 0.30

# Lower wall recombination (cleaner walls)
python main_v4b.py --power 1500 --pressure 10 --ar 0.0 --gamma-wall 0.10

# Pure Ar
python main_v4b.py --power 1000 --pressure 10 --ar 1.0
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--power` | 1500 | RF power (W) |
| `--pressure` | 10 | Pressure (mTorr) |
| `--ar` | 0.0 | Argon fraction |
| `--gamma-wall` | 0.10 | F recombination at sidewall (r=R) |
| `--gamma-wafer` | 0.02 | F recombination at wafer (z=0) |
| `--iter` | 100 | Max iterations |
| `--nr/--nz` | 30/40 | Grid cells |

## Output files

- `outputs_v4b/v4b_results.png` — 6-panel figure:
  - [F] 2D contour, SF6 depletion, Te field
  - Radial [F] profile, Mettler comparison, summary

## Key results (γ_wall = 0.30, pure SF6, 1500W)

| Quantity | Gen-4b | 0D ref | Mettler expt |
|----------|--------|--------|--------------|
| ⟨ne⟩ | 6.27×10⁹ | 6.27×10⁹ | not measured |
| ⟨Te⟩ | 2.99 eV (1.7–5.5) | 2.99 | not measured |
| α | 35.2 | 35.2 | not measured |
| **[F] center-to-edge drop** | **73%** | — | **75%** |
| SF6 depletion (center-edge) | 31% | 0% (uniform) | not measured |
| ne peak/avg | 3.5 | 2.5 (eigenmode) | not measured |

## γ_F calibration table

Results from γ_F sweep at 1500W, 10 mTorr, pure SF6:

| γ_F sidewall | [F] drop | Match |
|-------------|----------|-------|
| 0.05 | 36% | — |
| 0.10 | 51% | — |
| 0.15 | 60% | — |
| 0.20 | 66% | — |
| 0.30 | **73%** | **Mettler (75%)** |
| 0.50 | 80% | — |

## Etch rate prediction

After running, compute Si etch rate:

```python
from postprocess.etch_rate import etch_rate_profile
r_cm, R_etch = etch_rate_profile(result['nF'], result['mesh'])
# R_etch is in nm/s at each radial position
```

## Physics of the 73% drop

Three mechanisms combine:
1. **Sidewall Robin BC** (γ=0.30): F diffuses to r=R and recombines → steep radial gradient
2. **SF6 depletion**: SF6 is 15% at center vs 21% at edge → F source shifted to intermediate r
3. **Source-weighted ne**: ne peaks where Te is high (near coil) → ne peak/avg = 3.5

## Dependencies

- Python 3.8+, numpy, scipy, matplotlib
- `sf6_unified.py` (0D solver) must be on the Python path
  - Place in project root, OR
  - Set: `sys.path.insert(0, '/path/to/0d/solver')`
  - The path is configured at line ~80 of main_v4b.py
