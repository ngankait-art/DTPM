# Mettler-Benchmarking Mistakes Shared With Collaborator

Summary of mistakes uncovered while reconciling a Phase-1 DTPM plasma model against
Jeremy Mettler's UIUC 2025 PhD dissertation.  Cross-checked independently by Claude
Code (text-only code/report audit) and ChatGPT (visual PDF-to-PDF comparison).

---

## Citation / Provenance Mistakes

- **Wrong diagnostic attributed to Mettler.** The report said "BBAS + OES".
  Mettler's actual primary diagnostic is **W/Al non-equilibrium radical probes**
  (etching-based thermal response) cross-calibrated against Ar/SF6 actinometry.
  BBAS is not used in the dissertation.
- **"Mettler Fig 4.5" cited as a TEL power sweep.**
  Fig 4.5 is actually his **Helicon / PMIC** source data (Ch. 4.1), NOT the TEL
  etcher.  The correct TEL absolute-[F] figure is **Fig 4.9** (SF6-flow sweep at
  fixed power/pressure).  Mettler does not report a direct TEL power sweep.
- **Helicon/PMIC data and TEL data conflated.**
  Mettler's Chapter 4.1 is probe-material screening in the Plasma-Materials
  Interaction Chamber (Helicon source), not the TEL etcher.  Only Ch. 4.2–4.3
  (Fig 4.6–4.19) are TEL data and thus appropriate for benchmarking a TEL ICP
  model.  Treating all of Ch. 4 as TEL data is a scope error.

## Numeric Mistakes

- **"74% centre-to-edge [F] drop" used as a universal Mettler number.**
  Mettler's drop is **67%–75% depending on SF6/Ar composition** (Fig 4.17).
  The 74% specifically applies to **90% SF6, bias-off, 1000 W, 10 mTorr**
  (Fig 4.14 and Fig 4.17 top panel).  Used without qualification, the number
  suggests composition-independent agreement the model does not actually
  demonstrate.
- **Digitised cubic-fit points were loose.**
  The original digitisation at (25, 50, 75, 100) mm with values (0.83, 0.55,
  0.36, 0.26) is incorrect.  The correct evaluation of Mettler's published
  cubic fit `y(r) = 1.01032 − 0.01847 r² + 7.139×10⁻⁴ r³` (r in cm) at
  (0, 2, 4, 6, 8) cm is **(1.010, 0.942, 0.761, 0.500, 0.194)**.  Old values
  deviate by up to 26% at the outer radii.

## Operating-Condition Mismatch Mistakes

- **Model run at the wrong operating point for the claimed benchmark.**
  Mettler's radial [F] benchmark (Fig 4.14 / 4.17) is at **1000 W ICP +
  200 W rf wafer bias**.  Models run at **700 W, no bias** (a design-baseline
  choice) were being compared head-to-head with Mettler's 1000 W + bias
  dataset.  This is an apples-to-oranges comparison regardless of model
  quality.
- **"Pure SF6" mismatch.**
  Mettler Fig 4.14 / 4.17 is **70 sccm SF6 / 30 sccm Ar** (Fig 4.14) or
  **30%/90% SF6 + Ar balance** (Fig 4.17); it is NEVER pure SF6.  Running
  the model in pure SF6 and comparing to these figures is a composition
  mismatch that manifests as a 2× density error by itself.
- **Wafer-bias effect silently absorbed into other residuals.**
  Mettler's Fig 4.17 bias-on curve is ×1.6 (90% SF6) to ×2.15 (30% SF6
  centre) above the bias-off curve.  A model with no bias-sheath module
  compared to Mettler's bias-on data is effectively missing half the
  absolute density by construction.

## Architectural (Code-Level) Mistakes

- **Power-coupling efficiency η prescribed, not computed.**
  The chain was: `P_abs = eta_initial × P_rf`, then the FDTD E-field was
  rescaled every Picard iteration so the Ohmic integral always matched
  that target.  `eta_computed` was therefore **numerically identical** to
  `eta_initial` on every iteration.  Any "self-consistent η" claim in a
  report built on this code is false; and any absolute-magnitude
  benchmark that relies on η (which every benchmark does) is meaningless.
  This is the single most important mistake to look for in your code.
- **Matched-50 Ω-load coil-current formula.**
  `V_rms = √(P × Z)` with `Z = 50 Ω` implicitly treats the entire
  coil+plasma system as a matched 50 Ω resistor.  Real coil-plasma
  impedance is `R_coil + R_plasma_reflected` (typically ~16 Ω here).
  The formula happens to give a plausible-looking `I_peak ≈ 5 A` at
  700 W / 50 Ω but this is coincidence — the real physics is a
  transformer-coupled RLC circuit.
- **FDTD source was Gaussian-pulsed, not CW.**
  This is the subtlest and most damaging bug.  A Gaussian envelope
  centred at `1.5 × T_rf` with width `0.5 × T_rf` turns the source OFF
  after ~3 RF cycles.  The RMS accumulation over the "last RF cycle"
  then captures a FREELY DECAYING cavity mode, not the driven
  steady-state amplitude.  Result: FDTD E-field magnitudes are ~1000×
  too small for CW ICP operation.  This is the **hidden mechanism that
  forced the η rescale**: without the rescale, the FDTD simply gave
  wrong absolute fields.  Fix: linear ramp-up over 2 RF cycles, then
  constant-amplitude CW drive for the remaining cycles.
- **`eta = P_abs / 700` hardcoded in report-figure scripts.**
  When you later run at 1000 W, the displayed η silently drops by 43%
  because of this literal constant.  Easy to miss in downstream figures.

## Narrative Mistakes

- **"Within 5–8%" claim on the radial profile.**
  The residuals actually show +8% inside (ok) but +22–27% on the outer
  half of the wafer.  The 5–8% number averages only over the inner
  points and misrepresents the outer-wafer over-prediction that is the
  actual interesting physics signal (a signature of the missing
  electronegative-ambipolar correction).
- **Missing explicit Helicon-vs-TEL clarifying sentence.**
  Any report that cites "Mettler 2025" without explicitly scoping which
  figures apply to the TEL etcher invites the Helicon/TEL conflation
  error later on.

---

## Recommended Quick Audit Checklist for Your Code

1. `grep -n 'eta_initial\|eta.*=.*0\.[0-9]\|P_abs.*=.*eta' src/` — find
   every place η is pegged or P_abs is prescribed.  Any `P_abs = eta × P_rf`
   followed by an `E_scale = sqrt(target / raw)` is the tautology pattern.
2. Search FDTD source injection for `gauss`, `exp(-((n` — a Gaussian
   envelope on the coil source is fine only for transient-resonator
   studies, not for CW ICP steady state.
3. Confirm your reported η is actually changing across Picard
   iterations by logging it each step.  If it is pegged to three
   decimal places at the config value, the η is prescribed.
4. Grep the report source for `74\%` — every hit should be qualified
   with the SF6/Ar composition and bias state.
5. Grep for `Fig.*4\.5\b` in any TEL benchmarking context — replace with
   Fig 4.9 (correct TEL reference) or drop.
6. Verify all digitised Mettler points against the published cubic fit
   directly: evaluate `1.01032 − 0.01847·r² + 7.139×10⁻⁴·r³` at the
   claimed radii and reconcile.

---

## References

- Mettler, J. D. *Spatially Resolved Probes for the Measurement of Fluorine Radicals.*
  PhD Dissertation, UIUC, 2025 (105 pp).
- Lieberman & Lichtenberg, *Principles of Plasma Discharges and Materials Processing*,
  2nd ed., Wiley 2005 — Chapter 12 for the coupled-circuit model used to replace the
  prescribed η.
- Turner et al., *Phys. Plasmas* 20, 013507 (2013) — `f_ion_to_gas ≈ 0.5` for the
  wafer-bias sheath.
