# L1 Resolution — Codebase Audit

**Date**: 2026-04-17
**Scope**: Verification that the electronegative ambipolar diffusion correction
(L1) is applied consistently across every place it is physically relevant.

## Physics summary

Replacement performed:
```
D_a^ep  =  D_i * (1 + T_e / T_i)                                 [pre-L1]
D_a^en  =  D_i * (1 + T_e / T_i) * (1 + alpha) / (1 + alpha * T_i / T_e)   [post-L1]
```

The correction factor `(1+α)/(1+α·T_i/T_e)` reduces exactly to 1 when α = 0, so
the electropositive form is the α → 0 limit of the electronegative form.  No
free parameters are introduced — this is **rigorous physics, not a calibration**.

## Files modified

| File | Change | Lines |
|---|---|---|
| `src/dtpm/solvers/ambipolar_diffusion.py` | Replaced per-cell D_a loop with vectorised D_a^en = D_a^ep × correction. Added `alpha` argument to `solve_ne_ambipolar()`. Updated module docstring. | ~25 lines |
| `src/dtpm/modules/m11_plasma_chemistry.py` | Thread `alpha_2D = result_0D['alpha']` into the Picard-loop call to `solve_ne_ambipolar(..., alpha=alpha_2D)`. | 7 lines |
| `tests/test_electronegative_ambipolar.py` | **NEW** — 7 invariant tests for the correction factor and backwards compatibility. | 101 lines |
| `docs/report/main.tex` | Added §5.x "Electronegative Ambipolar Diffusion Correction" (58 lines); removed old L1 from §5.5 and renumbered L2–L5 → L1–L4; updated abstract to list three resolved limitations; updated Fig 7.10 caption; updated §12 benchmark discussion of the residual gap. | ~80 lines |
| `docs/CODE_REVIEW_ULTRAREVIEW.md` | Footer note that L1 is now resolved. | 2 lines |

Total: 1 new unit-test file, 4 modified source/report files.

## Files NOT modified (deliberately)

| File | Reason |
|---|---|
| `src/dtpm/solvers/species_transport.py` | `build_diffusion_matrix()` is generic; it accepts any `D_field` as input and correctly propagates the new `D_a^en` through the existing finite-difference stencil. No change needed. |
| `src/dtpm/solvers/multispecies_transport.py` | Neutral-species transport does not use the ambipolar coefficient.  Line 223 computes a local α post-hoc for diagnostics; not in the transport path. |
| `src/dtpm/modules/m06_fdtd_cylindrical.py` | Electromagnetic solver, independent of ambipolar transport. |
| `src/dtpm/modules/m10_power_deposition.py` | Ohmic integral, independent of ambipolar transport. |
| `src/dtpm/modules/m12_ccp_bias_sheath.py` | Sheath power-balance, independent of ambipolar transport. |
| `src/dtpm/chemistry/global_model.py` | The 0D model is the PRODUCER of α; no change in how α is computed. |

## Grep audits performed

```bash
grep -rnE '\(1\.?\s*\+\s*T[_]?e\s*/\s*T[_]?i\)' src/
grep -rnE 'D_a\s*=\s*D_i\s*\*' src/
grep -rnE 'Te\s*/\s*Ti' src/
```

Only hits in `src/dtpm/solvers/ambipolar_diffusion.py` (updated) and its own
module docstring comments (updated).  No stale electropositive-only formulas
remain in the production code path.

## Test results

```
tests/test_electronegative_ambipolar.py::TestCorrectionFactor::test_electropositive_limit PASSED
tests/test_electronegative_ambipolar.py::TestCorrectionFactor::test_isothermal_degenerate PASSED
tests/test_electronegative_ambipolar.py::TestCorrectionFactor::test_lieberman_worked_case PASSED
tests/test_electronegative_ambipolar.py::TestCorrectionFactor::test_typical_icp_conditions PASSED
tests/test_electronegative_ambipolar.py::TestCorrectionFactor::test_monotonic_in_alpha PASSED
tests/test_electronegative_ambipolar.py::TestBackwardsCompat::test_alpha_zero_equals_electropositive PASSED
tests/test_electronegative_ambipolar.py::TestBackwardsCompat::test_alpha_positive_increases_D_a PASSED
tests/test_selfconsistent_eta.py::*                                                             8 PASSED
================================================================= 15 passed in 0.20s =========
```

## Key physics finding

**The 0D global model returns α = 0.02 at Mettler's canonical operating
point** (1000 W, 10 mTorr, 90% SF₆, 200 W bias).  This is two orders of
magnitude below the "typical electronegative" α ≈ 1.0–1.5 that was cited as a
generic textbook estimate in the prior revision's L1 projection.  Consequence:
the correction factor at this operating point is 1.019, not ≈ 2; the L1 fix
changes the predicted F-drop by less than 1 percentage point.

The prior claim that "L1 is the single largest source of the 5.8-pp residual"
is therefore **falsified** by self-consistent physics.  The residual gap
between the model's 66% F-drop and Mettler's 74% is NOT closable by L1 at this
operating point.  It is most plausibly attributable to the
composition-insensitive `γ_Al = 0.18` calibration, which was itself fitted to
the approximate (pre-L1) transport model.

The L1 machinery remains a required component of the framework because the
correction factor rises substantially at lower-power / higher-pressure
conditions where α becomes ~1.

## Physical correctness, not performance

This fix is guaranteed to be rigorous physics by the following argument:
1. The Lieberman 2005 §10.3 expression is the textbook derivation for
   three-species quasi-neutral steady-state flux balance.
2. Taking α → 0 recovers the two-species electropositive limit exactly
   (unit test `test_alpha_zero_equals_electropositive` verifies this).
3. Taking T_i → T_e reduces the correction factor to unity (isothermal
   degenerate case, unit test `test_isothermal_degenerate`).
4. The α, T_e, T_i inputs are all computed from independent physics solves
   that do not themselves involve the ambipolar coefficient — no circular
   dependency.

This is **not** a calibration.  A calibration would be introducing a free
parameter (like `γ_Al` or `λ_exp`) tuned to match an experimental data point.
The L1 fix introduces no such parameter.
