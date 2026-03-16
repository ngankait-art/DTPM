# Mettler Validation Script

## What it does

Runs Gen-3 at Mettler's experimental conditions (1000W, 10 mTorr, 70/30 SF6/Ar)
and generates a side-by-side comparison of the model's radial [F](r) profile
against Mettler's Fig 4.14 cubic fit and Fig 4.17 absolute data.

## How to run

```bash
python validate_mettler.py
```

No options — it runs at the fixed Mettler conditions.

## Output

- `outputs_v3/mettler_validation.png` — Two-panel figure:
  - Left: Normalized [F](r) comparison with Mettler cubic fit
  - Right: Absolute [F] comparison with Mettler Fig 4.17 data

## Mettler experimental conditions

- Reactor: proprietary TEL ICP etcher (NOT Lallement geometry)
- ICP frequency: 2 MHz (NOT 13.56 MHz)
- 1000W ICP, 10 mTorr, 70/30 SF6/Ar, 200W wafer bias
- Measurement: W/Al radical probes, 1 cm above wafer
- Cubic fit: nF_norm = 1.01 - 0.0185·r² + 7.14e-4·r³ (R² = 0.997)

## Important caveat

The model and experiment use **different reactors**. Compare **profile shapes**
(normalized), not absolute values. The model's [F] is ~3× lower than Mettler's
due to different geometry, frequency, and power coupling.
