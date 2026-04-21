# 6 · Phase-1 Global–2D Self-Consistent Framework for ICP Etch Modelling

Snapshot of the Phase-1 DTPM framework after three architectural resolutions:

1. **Self-consistent ICP coupling efficiency η** (Lieberman 2005 §12 transformer circuit) — `src/dtpm/modules/m01_circuit.py` + rewrite of the Picard loop in `m11_plasma_chemistry.py` + CW FDTD source in `m06_fdtd_cylindrical.py`.
2. **Capacitive wafer-bias sheath module** (Lieberman §11 reduced-order) — `src/dtpm/modules/m12_ccp_bias_sheath.py`, with a single calibrated `λ_exp` that passes a 30%-SF6 composition blind test within ±13% of Mettler's target.
3. **Electronegative ambipolar diffusion correction** (Lieberman §10.3, rigorous physics — not a calibration) — `src/dtpm/solvers/ambipolar_diffusion.py`, with α threaded from the 0D global model; unit tests enforce the α=0 bit-for-bit electropositive limit.

Previously three "prescribed" knobs (η, no bias sheath, electropositive ambipolar) that made prior benchmarks questionable; now all three are resolved.  The remaining residual between the model and Mettler's 2025 TEL absolute [F] dataset is attributed to the composition-insensitive wall-recombination calibration `γ_Al = 0.18` (Phase-3 scope).

Further detail lives in the top-level report: `docs/report/A Self-Consistent Global–2D Hybrid Framework for ICP Etch Modelling.pdf` (103 pages).

---

## Repository layout

```
6_Phase1_Global-2D-SelfConsistent/
├─ src/dtpm/              # Python package (solvers, modules, chemistry, utils)
├─ config/                # default_config.yaml (operating conditions, coil R/L, bias)
├─ scripts/               # Reproduction scripts (see "Reproducing the results")
├─ tests/                 # Unit tests (15 pass)
├─ docs/
│   ├─ ASSUMED_PARAMETERS.md      # Every literature-sourced input with provenance
│   ├─ CODE_REVIEW_ULTRAREVIEW.md # Pre-rewrite audit of prescribed-input patterns
│   ├─ L1_AUDIT.md                # L1-resolution codebase sweep
│   ├─ L1_FINAL_SUMMARY.md        # One-page close-out
│   └─ report/                    # LaTeX source, PDF, figures
├─ results/               # JSON summaries + raw .npy fields from the campaign
├─ validation/            # Mettler benchmarking traceability (from the 5a work)
└─ animations/            # Optional visualisations
```

---

## Quick start

```bash
cd 6_Phase1_Global-2D-SelfConsistent
python -m pytest tests/ -v            # should show 15 passed
python scripts/run_simulation.py --config config/default_config.yaml
```

Expect the simulation to converge in ~55 s on one CPU core; prints η, I_peak, R_plasma, F_drop, and nF_centre at each Picard iteration.

---

## Reproducing the campaign (15 runs, ~60 min compute)

```bash
# Power sweep (11 runs at 200–1200 W, 70% SF6, 200 W bias)
python scripts/run_power_sweep_1000W_biased.py

# Composition pair (90%/30% SF6, bias-off vs bias-on — 4 runs)
python scripts/run_mettler_composition_pair.py

# Regenerate all four report figures from the saved results
python scripts/generate_phase1_selfconsistent_figures.py

# Rebuild the report (3 pdflatex passes)
cd docs/report && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

Expected outputs: matches `results/sweeps/power_1000W_biased/index.json` and `results/mettler_composition/composition_summary.json` within numerical tolerance.

---

## Literature inputs (no fitted parameters here)

| Parameter | Value | Reference |
|---|---|---|
| `circuit.R_coil` | 0.8 Ω | Lieberman 2005 Eq 12.2.19 |
| `circuit.L_coil` | 2.0 µH | Lieberman 2005 §12.2 |
| `bias.f_ion_to_gas` | 0.5 | Turner et al. 2013 |
| `bias.A_wafer_eff` | 0.01767 m² | TEL geometry (R_wafer = 75 mm) |
| `operating.Tgas` | 313 K | TEL baseline |

Full provenance in `docs/ASSUMED_PARAMETERS.md`.

## Fitted parameters (two calibrations only, both Mettler-transparent)

| Parameter | Value | Calibration target |
|---|---|---|
| `wall_chemistry.gamma_Al` | 0.18 | Mettler 74 % F-drop (90 % SF₆, bias-off) |
| `bias.lambda_exp` | 3.20 | Mettler ×1.60 bias-on [F] enhancement (90 % SF₆); blind-tested to ±13 % on 30 % SF₆ |

Every other knob in the code is either first-principles or an experimental operating condition.

---

## Key validated results (Mettler 2025 reference dataset)

At 1000 W / 10 mTorr / 70 % SF₆ / 200 W bias:

| Quantity | Model | Mettler |
|---|---|---|
| η | 0.950 (emergent) | n/a (not measured) |
| I_peak | 11.2 A | n/a |
| R_plasma | 15.3 Ω | n/a |
| F-drop 90 % SF₆ bias-on | 66.8 % | 74 % |
| F-drop 30 % SF₆ bias-on | 66.5 % | 67 % |
| Bias ×1.60 enhancement (90 % SF₆) | ×1.610 | ×1.60 (+0.6 %) |
| Bias enhancement (30 % SF₆ blind) | ×1.873 | ×2.15 (−12.9 %) |

Absolute-magnitude residual at wafer centre: ~−33 % (90 % SF₆), ~−35 % (30 % SF₆); shape residual inside r ≤ 4 cm is within ±10 %.  Source of residual: composition-insensitive γ_Al (Phase-3 scope).

---

## Unit tests

```
tests/test_selfconsistent_eta.py          8 tests (η ∈ [0,1], circuit fixed-point)
tests/test_electronegative_ambipolar.py   7 tests (α=0 limit, factor identities)
```

Both are intended as first-line regression guards — any rewrite that breaks these is catching a real physics error.

---

## Physics findings that survive review

1. **η emerges at 0.95 in the Lieberman regime**, not the 0.43 the prior work prescribed (not predictable *a priori*; only emerges because η is now a computed observable).  η stays narrowly in 0.949–0.956 across 200–1200 W because R_plasma ≫ R_coil throughout.
2. **At Mettler's operating point α = 0.02** from the 0D model — two orders of magnitude below the textbook "typical electronegative" estimate.  The electronegative ambipolar correction factor is therefore only 1.02 at this operating point, and L1 correctly does *not* close the 7-pp F-drop residual that the prior projection speculated it would.  This is falsified-by-self-consistent-physics evidence, documented in `docs/L1_FINAL_SUMMARY.md`.
3. **The residual F-drop gap is composition-sensitivity, not transport** — Mettler sees 67 %–75 % across compositions; the model is pegged at ~66 % because γ_Al does not vary with composition.

---

## Contact / provenance

Snapshot taken from the working tree at
`Steps/6_Phase1_A Self-Consistent Global–2D Hybrid Framework for ICP Etch Modelling/`.
See `EXECUTION_LOG.md` for the stage-by-stage trace and `docs/L1_FINAL_SUMMARY.md` for the one-page close-out.
