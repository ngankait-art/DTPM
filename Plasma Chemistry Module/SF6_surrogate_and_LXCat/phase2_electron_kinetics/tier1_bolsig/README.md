# Tier 1 — BOLSIG+ lookup and Maxwellian comparison

**Workplan §3.** Build a BOLSIG+ lookup table on the (E/N, x_Ar) grid, compare the resulting aggregated rate coefficients against the Maxwellian Arrhenius rates currently used in DTPM, and decide whether the Maxwellian assumption is adequate at the reference operating point.

## Inputs

- `../data/raw/bolsig_data.h5` — 28×6 grid in (E/N, x_Ar), 600-point EEDFs, 53 rate-coefficient channels, Maxwellian reference rates, transport coefficients. Real BOLSIG+ two-term Boltzmann output.
- `../data/raw/SF6_biagi_lxcat.txt` — LXCat Biagi cross-section file (50 SF₆ channels).

## Scripts

- `generate_lookup_tables.py` — reads the HDF5, writes CSV projections and a plain-text summary.
- `compare_maxwellian_vs_bolsig.py` — runs the 20% decision gate at the reference E/N = 50 Td and writes a markdown report.

## Outputs

- `outputs/lookup_summary.txt` — human-readable summary of the lookup contents.
- `outputs/rates_{boltzmann,maxwell}_pure_SF6.csv` — pure-SF₆ rate coefficients vs E/N.
- `outputs/transport_pure_SF6.csv` — Te_eff and mean energy vs E/N.
- `outputs/maxwell_vs_bolsig_report.md` — **decision gate report** (workplan §3.4).
- `plots/rate_ratio_SF6.pdf` — per-channel Maxwell/Boltzmann rate ratios.
- `plots/ratio_vs_EN.png` — ratio vs E/N on log-log axes with reference line.
- `plots/eedf_{50,100}Td.pdf` — example EEDFs at two operating points.

## Decision

**Gate met** at the reference point (E/N = 50 Td, pure SF₆): attachment 0.82, dissociation-like 1.04, elastic 0.97 — all within the 20% band. Ionisation is below the numerical floor at this point in both references.

## Run

```bash
python generate_lookup_tables.py
python compare_maxwellian_vs_bolsig.py
```
