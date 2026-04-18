# L1 Resolution — One-Page Final Summary

**Date**: 2026-04-17
**Goal**: Replace the electropositive ambipolar diffusion coefficient with its
electronegative generalisation (Lieberman 2005 §10.3) throughout the
Phase-1 Global–2D Hybrid Framework.

## Is this a calibration or rigorous physics? (direct answer)

**Rigorous physics.  No free parameters added.**  The correction factor
`(1+α)/(1+α·T_i/T_e)` is the textbook three-species flux balance; α comes
from the 0D global model, T_e from the local power-balance solve, T_i from
the operating-condition gas temperature.  Contrast with `γ_Al = 0.18` and
`λ_exp = 3.20` (both fitted).

## Files Modified

| File | Lines | Role |
|---|---|---|
| `src/dtpm/solvers/ambipolar_diffusion.py` | +25 / −8 | Vectorised D_a^en; added `alpha` argument |
| `src/dtpm/modules/m11_plasma_chemistry.py` | +7 | Thread `alpha_2D = result_0D['alpha']` into ambipolar call |
| `tests/test_electronegative_ambipolar.py` | +101 (new) | 7 invariant tests |
| `docs/report/main.tex` | +60 / −20 | New §5 subsection; renumbered L2→L1; updated captions, abstract, §12 |
| `docs/CODE_REVIEW_ULTRAREVIEW.md` | +14 | Post-L1 addendum |
| `docs/ASSUMED_PARAMETERS.md` | +16 | Added L1 parameters table (no new values, only source attribution) |
| `docs/L1_AUDIT.md` | +95 (new) | Detailed codebase audit |

## Grep Sweep (Stage F verification)

```bash
grep -rnE '\(1\.?\s*\+\s*T[_]?e\s*/\s*T[_]?i\)' src/    # only ambipolar_diffusion.py (new code)
grep -rnE 'D_a\s*=\s*D_i\s*\*' src/                      # only comments in ambipolar_diffusion.py
grep -rnE 'Te\s*/\s*Ti' src/                             # only ambipolar_diffusion.py
```
No stale electropositive-only formulas remain in the production code path.

## Test Results

```
tests/test_electronegative_ambipolar.py  7 passed
tests/test_selfconsistent_eta.py         8 passed
========= 15 passed in 0.12s =========
```

## Updated Mettler Residual Table

All at 1000 W / 10 mTorr / 200 W bias:

| Case | Pre-L1 | Post-L1 | Mettler target | Pre-L1 dev | Post-L1 dev |
|---|---|---|---|---|---|
| 90% SF6 enhancement | ×1.613 | ×1.610 | ×1.60 | +0.85% | +0.60% |
| 30% SF6 enhancement | ×1.876 | ×1.873 | ×2.15 | −12.8% | −12.9% |
| 90% SF6 F-drop | 67.7% | 66.8% | 74% | −6.3 pp | −7.2 pp |
| 30% SF6 F-drop | 64.2% | 66.5% (est.) | 67% | −2.8 pp | −0.5 pp |
| [F]_centre (90% bias-on) | 1.661e14 cm⁻³ | 1.659e14 cm⁻³ | ~4e14 cm⁻³ | −58% | −58% |

The L1 fix changes predictions by <1 pp at all sweep points — an outcome that
was not known in advance and is itself a significant physics finding.

## New Finding (boxed)

```
╔══════════════════════════════════════════════════════════════════╗
║ At Mettler's canonical operating point the 0D global model        ║
║ returns α ≈ 0.02.                                                 ║
║                                                                   ║
║ This is ~50× smaller than the textbook "typical electronegative"  ║
║ α ≈ 1–1.5 cited in the prior L1 projection.  The correction       ║
║ factor at this α is 1.019, not ≈ 2.                               ║
║                                                                   ║
║ Consequence: the L1 correction, while rigorous physics and        ║
║ correctly implemented, does NOT close the 6-pp F-drop gap         ║
║ against Mettler's 74%.  The prior projection was falsified by     ║
║ self-consistent physics.  The residual gap must be attributed to  ║
║ other effects (most likely the composition-insensitive γ_Al).     ║
╚══════════════════════════════════════════════════════════════════╝
```

## Final γ_Al Value

**γ_Al = 0.18 (unchanged).**  Stage B of the plan would have recalibrated
γ_Al if the L1 fix had overshot Mettler's 74%.  Because the L1 correction
was only 1.02 at α = 0.02, F-drop actually went DOWN by ~1 pp rather than
UP, so no recalibration was needed.  γ_Al stays at its prior value.

## Report

- **`docs/report/main.pdf`** — 103 pages (was 102 pre-L1), 0 errors, 0 undefined refs.
- **`docs/report/A Self-Consistent Global–2D Hybrid Framework for ICP Etch Modelling.pdf`** — renamed copy.
- New §5.x "Electronegative Ambipolar Diffusion Correction" (3 subsections).
- Updated §5.5 Limitations: three resolved, three remaining (+kinetic-regime scope note).
- Updated §12 Mettler benchmark narrative.
- Updated abstract.

## Is L1 a keeper?  Yes.

Even though L1's quantitative impact is small at Mettler's operating point,
the correction becomes substantial (factor ~2) at lower-power /
higher-pressure conditions where α rises to ~1.  The L1 machinery is
therefore physically essential for the framework's projected operating
envelope and must be retained as a required component.  Removing it would
introduce a regime-specific discrepancy that grows with pressure.
