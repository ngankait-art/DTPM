# Generation 5 — Full Physics (Final)

**Status: CURRENT — recommended for all use.**

## What it adds over Gen-4b

1. **Self-consistent ne magnitude** — ne solved from local power balance (not 0D anchor)
2. **BOLSIG+ table infrastructure** — transport from Boltzmann-solver with LXCat cross-sections
3. **Analytic sheath model** — V_s, E_ion, ion flux at all walls
4. **Gas temperature solver** — 2D neutral energy equation
5. **Multi-ion species** — SF5+ (90%), SF3+ (10%), Ar+ tracked separately
6. **Enhanced DT API** — `run_dt_scan()` with etch rate output

## How to run

```bash
# Option A: Using the unified code (RECOMMENDED, no path setup needed):
python code/unified/sf6_icp_unified.py --gen 5 --power 1500 --ar 0.0 --outdir outputs/gen5

# Option B: Using the generation driver with run.sh (sets up PYTHONPATH):
cd code/generation_5
bash run.sh --power 1500 --ar 0.0 --gamma-wall 0.30
```

## Key results (1500W, 10 mTorr, pure SF6)

| Quantity | Gen-5 | 0D ref | Mettler |
|----------|-------|--------|---------|
| ne | 3.4×10⁹ cm⁻³ | 6.3×10⁹ | — |
| Te | 2.99 eV (1.8–4.5) | 2.99 | — |
| α | 64.6 | 35.2 | — |
| [F] drop | **75%** | — | **75%** |
| Mi_eff | 123 AMU | 127 | — |

## Limitations

- SF6 transport uses Arrhenius fits (Boltzmann solver only for Ar)
- Gas heating negligible at 10 mTorr (Tg stays at 300 K)
- ne magnitude 55% of 0D value (spatial losses higher than 0D h-factor estimate)
