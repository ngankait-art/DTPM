# Generation 4 — Robin BCs + Hagelaar Transport

## What it adds over Gen-3

- **Robin BCs for F atoms**: D_F·∂nF/∂n = γ_F·(v_th/4)·nF at walls,
  replacing Neumann + volumetric kw. Creates physical surface gradient.
- **Robin BCs for SF6**: surface sticking β_SF6 at walls
- **Hagelaar-corrected D_ε**: uses Ramsauer-corrected transport coefficients
  from hagelaar_transport.py. For Ar, D_e is 2× higher than Einstein relation;
  D_ε/D_e ≈ 2.6–3.0 instead of naive 5/3 = 1.67
- **Blanc's law mixing** for Ar/SF6 transport coefficients
- **Parameter scan API**: `run_scan()` for power × pressure × composition sweeps

## How to run

```bash
# Pure SF6 with calibrated gamma_F = 0.01
python main_v4.py --power 1500 --pressure 10 --ar 0.0 --gamma-F 0.01

# Higher gamma_F for steeper [F] gradient
python main_v4.py --power 1500 --pressure 10 --ar 0.0 --gamma-F 0.10

# Pure Ar
python main_v4.py --power 1000 --pressure 10 --ar 1.0

# Parameter scan mode
python main_v4.py --scan
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--power` | 1500 | RF power (W) |
| `--pressure` | 10 | Pressure (mTorr) |
| `--ar` | 0.0 | Argon fraction |
| `--gamma-F` | 0.10 | F wall recombination probability |
| `--beta-SF6` | 0.005 | SF6 wall sticking probability |
| `--iter` | 100 | Max iterations |
| `--nr/--nz` | 30/40 | Grid cells |
| `--scan` | False | Run parameter scan instead of single case |
| `--no-plot` | False | Skip plots |

## Output files

- `outputs_v4/v4_profiles.png` — 12-panel contour plot
- `outputs_v4/v4_mettler.png` — Mettler comparison (normalized + absolute)

## Key results (gamma_F = 0.01, pure SF6)

| Quantity | Gen-4 | 0D reference |
|----------|-------|--------------|
| ⟨ne⟩ | 6.27×10⁹ cm⁻³ | 6.27×10⁹ |
| ⟨Te⟩ | 2.99 eV (1.6–5.6) | 2.99 |
| ⟨[F]⟩ | 1.24×10¹⁴ | 1.19×10¹⁴ |
| [F] drop | 2% (γ_F = 0.01) | — |

## Parameter scan API

```python
from main_v4 import run_scan
results = run_scan(
    powers=[500, 1000, 1500, 2000],
    pressures=[10],
    compositions=[0.0, 0.3, 0.5, 0.7, 1.0],
    gamma_F=0.10
)
for r in results:
    if r['converged']:
        print(f"P={r['P_rf']}W: ne={r['ne_avg']*1e-6:.1e}, [F]drop={r['F_drop_pct']:.0f}%")
```

## Limitations

- Same Robin coefficient at all walls (wafer = sidewall = window)
- SF6 still anchored to 0D average
- ne profile from uniform-source eigenmode (peak/avg = 2.5)
