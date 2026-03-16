# Generation 1 — Hybrid 0D+2D Diffusion Eigenmode

## What it does
Couples the validated 0D global chemistry solver (26 species, 54 reactions)
with a 2D ambipolar diffusion profile solver. The 0D solver provides ne, Te,
alpha, [F], SF6; the 2D solver adds spatial profiles from the fundamental
diffusion eigenmode.

## How to run

```bash
# Pure Ar, 1000W, 10 mTorr
python main.py --power 1000 --pressure 10 --ar 1.0 --nr 40 --nz 50

# Pure SF6, 1500W, 10 mTorr
python main.py --power 1500 --pressure 10 --ar 0.0

# 50/50 mix
python main.py --power 1000 --pressure 10 --ar 0.5
```

## Options
- `--power`    RF power in Watts (default 1000)
- `--pressure` Pressure in mTorr (default 10)
- `--ar`       Argon fraction 0-1 (default 1.0)
- `--nr/--nz`  Grid cells (default 40x50)
- `--iter`     Max iterations (default 200)
- `--no-plot`  Skip plot generation

## Output
- `outputs/2d_profiles.png` — 9-panel 2D contour + line profiles

## Limitations
- Te from empirical P^0.15 perturbation (not energy equation)
- [F] spatially uniform (0D anchor)
- alpha from 0D average only
- ne profile is the fundamental eigenmode (peak/avg = 2.5 always)
