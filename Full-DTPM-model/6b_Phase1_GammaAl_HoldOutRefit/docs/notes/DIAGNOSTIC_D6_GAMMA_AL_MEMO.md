# DIAGNOSTIC D6 — gamma_Al Hold-Out Refit

**Decision:** **FAIL** — gamma_Al is NOT the closing lever for the Mettler
Fig 4.17 −56 % wafer-centre [F] residual.

**Reason:** At the fit-point-optimal gamma_Al* = 0.020, the 30 %-SF6
bias-off held-out condition has |residual_F| = 30.0 %, which exceeds the
FAIL threshold of 15 %. More critically, the residual *changes sign across
composition*: −21.7 % at 90 % SF6 bias-on and +30.0 % at 30 % SF6 bias-off
at the same gamma_Al. No scalar gamma_Al can satisfy both ends of the
composition axis simultaneously. This is composition-dependent structural
behaviour, not a single miscalibrated loss term.

---

## 1. Protocol (executed as planned)

- **Fit point**: `90pct_SF6_bias_on` (Mettler [F]_c = 3.774e14 cm⁻³,
  F-drop 74.8 %).
- **Held-out**: `90pct_SF6_bias_off`, `30pct_SF6_bias_on`,
  `30pct_SF6_bias_off`.
- **Grid**: `gamma_Al ∈ {0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.25, 0.35}`.
- **Locked knobs** (guard against drift): R_coil = 0.8, lambda_exp = 3.20,
  D_scale = 1.0, s_F_Al_scale = 1.0, `use_boltzmann_rates=False`,
  `use_2d_alpha=off`.
- **Literature bounds**: `[0.01, 0.40]` (Gray 1993, Ullal 2002, Kokkoris 2009).
- **Tolerances**: magnitude ±10 % on [F]_c, shape ±5 pp on F-drop.

### Decision gate (encoded in `scripts/analyze_gamma_al_refit.py`)

- **PASS** ⇔ all 3 held-out points meet `|residual_F| ≤ 10 %`
  AND `|residual_drop| ≤ 5 pp` AND `gamma_Al*` ∈ [0.01, 0.40].
- **PARTIAL** ⇔ magnitude tolerance met on all 3 held-out AND F-drop
  tolerance fails on ≥ 1.
- **FAIL** ⇔ otherwise (magnitude > 15 % on any held-out, or bounds violated).

## 2. Stage A0 prerequisite: BC-map floor-annulus fix

Before the sweep, [src/dtpm/core/geometry.py](../../src/dtpm/core/geometry.py)
`build_geometry_mask()` was patched to split the `j=0` bottom boundary
into `BC_WAFER` (r ≤ R_wafer = 0.075 m, silicon) and `BC_AL_TOP`
(R_wafer < r ≤ R_proc, aluminium pedestal-top / chamber-floor annulus).
The prior 6a code labelled the entire `j=0` row as `BC_WAFER`, under-
counting the Al F-sink by an annulus of area π(0.105² − 0.075²) ≈
0.0170 m² — 96 % of the wafer area.

Regression at gamma_Al = 0.18 with the fixed BC map:

| Condition | Old (broken) [F]_c | Fixed-BC [F]_c | Shift |
|---|---|---|---|
| 90 % SF₆ bias-off | 1.031e14 | 9.762e13 | −5.3 % |
| 90 % SF₆ bias-on  | 1.659e14 | 1.586e14 | −4.4 % |
| 30 % SF₆ bias-off | 3.648e13 | 3.426e13 | −6.1 % |
| 30 % SF₆ bias-on  | 6.834e13 | 6.432e13 | −5.9 % |

All four shift DOWN by 4–6 % (more Al surface area → more F loss). Residuals
vs Mettler worsen from {−56, −56, −47, −41 %} to {−58, −58, −50, −44 %}.
This is the "fixed-baseline" against which the refit was performed.

The fix is verified by 5 unit tests in
[tests/test_bc_map.py](../../tests/test_bc_map.py); all 20 tests in the
suite pass.

## 3. Stage A results — hold-out refit at gamma_Al* = 0.020

Full decision table (analyzer output):

| Condition | gamma_Al | Model [F]_c (cm⁻³) | Mettler [F]_c (cm⁻³) | res_F | Model F-drop | Mettler F-drop | res_drop |
|---|---|---|---|---|---|---|---|
| **FIT** `90pct_SF6_bias_on` | **0.020** | 2.956e+14 | 3.774e+14 | **−21.7 %** | 29.15 % | 74.80 % | **−45.65 pp** |
| HELD-OUT `90pct_SF6_bias_off` | 0.020 | 2.191e+14 | 2.327e+14 | −5.8 % | 28.09 % | 74.50 % | −46.41 pp |
| HELD-OUT `30pct_SF6_bias_on`  | 0.020 | 1.381e+14 | 1.297e+14 | +6.5 % | 26.51 % | 75.00 % | −48.49 pp |
| HELD-OUT `30pct_SF6_bias_off` | 0.020 | 8.005e+13 | 6.160e+13 | **+30.0 %** | 28.17 % | 67.20 % | −39.03 pp |

**Observations that make the FAIL robust:**

1. **Fit point itself fails the tolerance**: even at the minimum-residual
   gamma_Al (0.020), the fit-point [F]_c is 22 % below Mettler. The model
   has a structural deficit at the fit point that no gamma_Al in the
   physical window closes.
2. **Residual changes sign across composition**: at gamma_Al = 0.020 the
   90 % SF₆ branch under-predicts (−21.7 % to −5.8 %) while the 30 % SF₆
   bias-off over-predicts (+30.0 %). A scalar gamma_Al moves all four
   conditions in the same direction; the opposite signs rule out a scalar
   fix.
3. **F-drop is globally wrong at low gamma_Al**: at gamma_Al = 0.020, model
   F-drop is ~28 % across all four conditions vs Mettler 67–75 %. To recover
   F-drop you need gamma_Al ≈ 0.14–0.18, at which point magnitude residuals
   blow up to ~−55 % to −58 %.

### Full gamma_Al sweep table (all 36 points)

See [`results/gamma_al_sweep/gamma_al_sweep_summary.json`](../../results/gamma_al_sweep/gamma_al_sweep_summary.json)
and the four-panel plot [`fig_gamma_al_scan.pdf`](../../results/gamma_al_sweep/fig_gamma_al_scan.pdf).

Summary by condition (residual_F_pct at each gamma_Al):

| gamma_Al | 90%on | 90%off | 30%on | 30%off |
|---|---|---|---|---|
| 0.020 | −21.7 | −5.8 | +6.5 | +30.0 |
| 0.040 | −36.5 | −27.5 | −18.5 | −1.5 |
| 0.060 | −43.9 | −38.1 | −30.0 | −15.9 |
| 0.080 | −48.5 | −44.5 | −36.4 | −24.7 |
| 0.100 | −51.5 | −48.8 | −40.9 | −31.2 |
| 0.140 | −55.5 | −54.5 | −46.7 | −39.1 |
| 0.180 | −58.0 | −58.0 | −50.4 | −44.4 |
| 0.250 | −60.7 | −62.0 | −54.6 | −49.7 |
| 0.350 | −63.1 | −65.6 | −58.3 | −54.4 |

At NO value of gamma_Al in [0.01, 0.40] are all four residuals simultaneously
within ±15 %. The 30 % SF₆ bias-off row crosses 0 between gamma_Al = 0.040
(−1.5 %) and 0.020 (+30 %), while the 90 % SF₆ bias-on row never reaches
the ±10 % band for any gamma_Al ≤ 0.35.

## 4. Stage A' — gamma_quartz cross-check

Results from `scripts/run_gamma_quartz_crosscheck.py` (90 % SF₆ bias-on,
gamma_Al = 0.18, single condition):

| gamma_quartz | [F]_c (cm⁻³) | Δ vs Kokkoris default | F-drop (%) |
|---|---|---|---|
| 0.0001 (10× lower than Kokkoris) | 1.606e14 | +1.3 % | 76.31 |
| 0.0010 (Kokkoris, default)       | 1.586e14 | baseline | 76.27 |
| 0.0050 (5× higher)               | 1.509e14 | −4.9 % | 76.10 |
| 0.0200 (20× higher, unphysical for clean quartz) | 1.257e14 | −20.7 % | 75.48 |

The analyzer script reports a "27.71 % spread", but that is the full-range
spread including the γ_quartz = 0.02 endpoint which is unphysical for clean
quartz (Kokkoris 2009 measures 0.001; values near 0.02 apply to passivated
quartz or SiC surfaces — not the TEL clean-quartz tube the user confirmed).

**Within the physically plausible window** [0.0001, 0.005] (Kokkoris ±5×),
the [F]_c spread is 6.4 %. Even the best case (γ_quartz = 0.0001, ideal
clean quartz) only moves the fit-point [F]_c from 1.586e14 to 1.606e14 —
1.3 % closer to Mettler's 3.774e14.

**γ_quartz has sensitivity but not leverage**: it cannot rescue the −56 %
gap. The γ_Al single-knob FAIL verdict stands.

## 5. Scientific interpretation — why single-knob refit cannot close the gap

The Mettler residual is **composition-dependent in structure**, not a
scalar offset. Three independent signals:

1. **Sign-flip across composition** (§3 observation #2): residual at
   gamma_Al = 0.020 is −21.7 % at 90 % SF₆ bias-on and +30.0 % at 30 %
   SF₆ bias-off. This is incompatible with *any* globally-applied wall-loss
   coefficient.
2. **F-drop vs [F]_c tension** (§3 observation #3): F-drop requires
   gamma_Al ≈ 0.14–0.18 (consistent with Kokkoris); [F]_c magnitude
   requires gamma_Al ≤ 0.02. The two requirements are mutually exclusive
   for a scalar γ_Al.
3. **gamma_quartz has no closing leverage** (§4): confirms the deficit is
   not a mis-assigned loss on the quartz surface either.

The most likely physical origin (from the D5 memo's forward-looking
remarks) is **composition-dependent wall surface chemistry**: the γ_F
effective probability on aluminium depends on the surface coverage of
fluorinated species (AlFx), which itself depends on the local [F]/[SFx]
ratio. At 90 % SF₆ the surface saturates quickly and γ_F drops; at 30 %
SF₆ / 70 % Ar the surface is less saturated and γ_F stays high.

Another candidate is **non-Maxwellian EEDF**: at high SF₆ the attachment
tail is depleted and the effective e-impact dissociation rate is higher
than the Maxwellian-averaged value used here.

## 6. Recommendation — Stage C (separate thread)

Per the plan, Stage C is explicitly deferred. Two paths are worth pursuing
in a future thread, each requiring its own protocol:

1. **Physics-grounded γ_Al(θ_AlFx, T_wall) surface-chemistry model**. Adds
   one surface-balance equation for AlFx coverage; γ_F(θ) falls out of
   Langmuir kinetics. This introduces 2 physics parameters (an adsorption
   rate constant and a reaction probability) but *no empirical fit* to the
   4 Mettler points.
2. **Extended composition sweep (> 10 points)** paired with an empirical
   γ_Al(x_Ar) curve. Requires data we do not yet have; acquiring it is a
   Phase-3 deliverable.

**Do NOT** fit a γ_Al(x_Ar) polynomial to the 4 Mettler points in this
thread — it would be overfitting and indefensible, as the plan warned.

## 7. Files created / modified in 6b

| File | Change |
|---|---|
| `src/dtpm/core/geometry.py` | BC-map floor-annulus fix (R_wafer split at j=0) |
| `config/default_config.yaml` | Added `tel_geometry.R_wafer: 0.075` |
| `scripts/run_parameter_sweeps.py` | Pass `R_wafer` to `build_geometry_mask` |
| `scripts/run_simulation.py` | Pass `R_wafer` to `build_geometry_mask` |
| `tests/test_bc_map.py` | **NEW** — 5 tests for the BC split |
| `scripts/run_gamma_al_sweep.py` | **NEW** — 36-run γ_Al hold-out sweep |
| `scripts/analyze_gamma_al_refit.py` | **NEW** — decision-gate analyzer |
| `scripts/run_gamma_quartz_crosscheck.py` | **NEW** — 4-run γ_quartz cross-check |
| `results/gamma_al_sweep/` | 36 `summary.json` + `gamma_al_sweep_summary.json` + `decision.json` + `fig_gamma_al_scan.pdf` |
| `results/gamma_quartz_crosscheck/` | 4 `summary.json` + `gamma_quartz_crosscheck_summary.json` |
| `results/mettler_composition/composition_summary_6a_oldBCmap.json` | Preserved old baseline |
| `results/mettler_composition/composition_summary.json` | New baseline (fixed BC map) |

**Not modified** (decision gate did not authorise these):
- `docs/report/main.tex` — no §5.1 or §7.5 edits (gated on PASS).
- `config/default_config.yaml:gamma_Al` — unchanged at 0.18 (gated on PASS).

## 8. Chamber-material audit (reference, from the user's photos + CAD)

| Region | Material | BC label | γ |
|---|---|---|---|
| ICP source tube sidewall | Quartz (CAD annotation "blue = quartz pipe") | `BC_QUARTZ` | 0.001 (Kokkoris) |
| Processing-region sidewall | Bare / anodised aluminium | `BC_AL_SIDE` | 0.18 (refit target — falsified) |
| Aperture-plate underside | Aluminium | `BC_AL_TOP` | 0.18 |
| Shoulder (aperture transition) | Aluminium | `BC_SHOULDER` | 0.18 |
| Wafer | Silicon (Ø 150 mm) | `BC_WAFER` | 0.025 |
| Floor annulus (R_wafer < r ≤ R_proc) | Aluminium | `BC_AL_TOP` (after A0 fix) | 0.18 |

User confirmed the reactor is the same TEL ICP-etcher spec as Mettler's
UIUC chamber, so γ_Al calibrated against her F-drop would transfer
directly — **if γ_Al were the correct closing lever**. The hold-out
protocol now shows it is not.

## 9. Reproducibility

```bash
# From Steps/6b_Phase1_GammaAl_HoldOutRefit/:
PYTHONPATH=src python -m pytest tests/                              # 20 pass
PYTHONPATH=src python scripts/run_mettler_composition_pair.py       # fixed-baseline (~2 min)
PYTHONPATH=src python scripts/run_gamma_al_sweep.py                 # 36-run sweep (~12 min)
PYTHONPATH=src python scripts/run_gamma_quartz_crosscheck.py        # 4-run crosscheck (~2 min)
PYTHONPATH=src python scripts/analyze_gamma_al_refit.py             # emit decision
```

All random seeds are fixed and the Picard loop is deterministic, so the
FAIL verdict is reproducible.
