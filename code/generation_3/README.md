# Generation 3 — Self-Consistent Energy Equation + F Diffusion

## What it adds over Gen-2

- **2D electron energy equation**: solves D_ε∇²ε̄ + P_ind/(ne·e) - ν_ε·ε̄ = 0
  with Neumann BCs, replacing the empirical Te ∝ P^0.15 approximation
- **SF6 diffusion transport**: D_SF6·∇²nSF6 + S_feed - ν_loss·nSF6 = 0 (Neumann BCs)
- **F diffusion transport**: D_F·∇²nF + R_F - ν_loss·nF = 0 (Neumann BCs + volumetric kw)
- **[F] is SELF-CONSISTENT** — no 0D anchor. Converges to 96% of 0D value independently
- **Hagelaar energy diffusivity**: D_ε reduced from naive (5/3)De approximation
- **Stronger EM coupling**: 5 sub-iterations, 80% weight, every 3 outer iterations

## How to run

```bash
# Pure SF6, 1500W, 10 mTorr (primary test case)
python main_v3.py --power 1500 --pressure 10 --ar 0.0

# Pure Ar
python main_v3.py --power 1000 --pressure 10 --ar 1.0

# 70/30 SF6/Ar (Mettler comparison conditions)
python main_v3.py --power 1000 --pressure 10 --ar 0.3
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--power` | 1500 | RF power (W) |
| `--pressure` | 10 | Pressure (mTorr) |
| `--ar` | 0.0 | Argon fraction 0-1 |
| `--iter` | 120 | Max iterations |
| `--nr` | 30 | Radial grid cells |
| `--nz` | 40 | Axial grid cells |
| `--no-plot` | False | Skip figure generation |

## Output files

- `outputs_v3/v3_profiles.png` — 12-panel 2D contour plot
- `outputs_v3/v3_F_radial.png` — Radial [F] at wafer vs midplane

## Key results (pure SF6, 1500W, 10 mTorr)

| Quantity | Gen-3 | 0D reference |
|----------|-------|--------------|
| ⟨ne⟩ | 6.27×10⁹ cm⁻³ | 6.27×10⁹ |
| ⟨Te⟩ | 2.99 eV (0.8–9.4) | 2.99 |
| α | 35.2 | 35.2 |
| ⟨[F]⟩ | 1.15×10¹⁴ cm⁻³ | 1.19×10¹⁴ |
| 2D/0D ratio | 0.96 (no anchor) | — |

## Limitations

- Te range (0.8–9.4 eV) is wider than physically expected (1.5–5 eV)
  because D_ε is still approximate
- [F] profile is nearly flat (3% center-to-edge drop) because Neumann BCs
  with volumetric wall loss do not create surface gradients
- SF6 still anchored to 0D average
