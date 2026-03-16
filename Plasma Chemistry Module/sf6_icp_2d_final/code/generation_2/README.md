# Generation 2 — Negative Ion Transport + 9-Species Neutrals

## What it adds over Gen-1
- Spatially resolved n_neg(r,z) from Neumann-BC diffusion (trapped ions)
- Full 9-species neutral chemistry (SF6→SF5→...→F) at each grid point
- Iterated EM-plasma feedback (3 sub-iterations per update)
- Alpha(r,z) showing Lichtenberg stratification (core ~14, edge >1000)
- Volume-averaged anchoring to 0D backbone prevents drift

## How to run

```bash
# Pure SF6 (primary validation case)
python main_v2.py --power 1500 --pressure 10 --ar 0.0

# Pure Ar
python main_v2.py --power 1000 --pressure 10 --ar 1.0

# 50/50 mix
python main_v2.py --power 1000 --pressure 10 --ar 0.5
```

## Options
Same as Gen-1 plus all output in `outputs_v2/`

## Key results
- ne = 6.27e9 cm-3, alpha = 35.2, [F] = 1.19e14 (all match 0D exactly)
- n_neg range: 1.5-3.4 × 10^17 m-3 (spatial structure from diffusion)
- alpha range: 14-1000 (Lichtenberg electronegative stratification)

## Known bug fixed in this generation
Sign error in _solve_diffusion_neumann: `spsolve(A, rhs)` → `spsolve(A, -rhs)`
