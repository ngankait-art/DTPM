# Mettler-citation corrections — audit trail

Supervisor-supplied list of citation/provenance/numeric/operating-condition/narrative mistakes
made in prior iterations of the Phase-1 self-consistent Global–2D report and the SF6/Ar zero-
dimensional report. Preserved here as the evidence trail for why the v2 reports use their
current wording. Also drives the workspace-wide audit in manifest 13 (sibling document).

**Status**: All 10 items are addressed in the two v2 reports the supervisor shared on
2026-04-18. Older reports in `archive_versions/` may still contain uncorrected versions;
those are historical and are not being revised. Files under `active_projects/` and
`final_package/` should be checked and fixed if any surviving violations are found.

---

## Citation / Provenance Mistakes

- **Wrong diagnostic attributed to Mettler.** The report said "BBAS + OES".
  Mettler's actual primary diagnostic is **W/Al non-equilibrium radical probes**
  (etching-based thermal response) cross-calibrated against Ar/SF6 actinometry.
  BBAS is not used in the dissertation.
- **"Mettler Fig 4.5" cited as a TEL power sweep.**
  Fig 4.5 is actually his **Helicon / PMIC** source data (Ch. 4.1), NOT the TEL
  etcher. The correct TEL absolute-[F] figure is **Fig 4.9** (SF6-flow sweep at
  fixed power/pressure). Mettler does not report a direct TEL power sweep.
- **Helicon/PMIC data and TEL data conflated.**
  Mettler's Chapter 4.1 is probe-material screening in the Plasma-Materials
  Interaction Chamber (Helicon source), not the TEL etcher. Only Ch. 4.2–4.3
  (Fig 4.6–4.19) are TEL data and thus appropriate for benchmarking a TEL ICP
  model. Treating all of Ch. 4 as TEL data is a scope error.

## Numeric Mistakes

- **"74% centre-to-edge [F] drop" used as a universal Mettler number.**
  Mettler's drop is **67%–75% depending on SF6/Ar composition** (Fig 4.17).
  The 74% specifically applies to **90% SF6, bias-off, 1000 W, 10 mTorr**
  (Fig 4.14 and Fig 4.17 top panel). Used without qualification, the number
  suggests composition-independent agreement the model does not actually
  demonstrate.
- **Digitised cubic-fit points were loose.**
  The original digitisation at (25, 50, 75, 100) mm with values (0.83, 0.55,
  0.36, 0.26) is incorrect. The correct evaluation of Mettler's published
  cubic fit `y(r) = 1.01032 − 0.01847 r² + 7.139×10⁻⁴ r³` (r in cm) at
  (0, 2, 4, 6, 8) cm is **(1.010, 0.942, 0.761, 0.500, 0.194)**. Old values
  deviate by up to 26% at the outer radii.

## Operating-Condition Mismatch Mistakes

- **Model run at the wrong operating point for the claimed benchmark.**
  Mettler's radial [F] benchmark (Fig 4.14 / 4.17) is at **1000 W ICP +
  200 W rf wafer bias**. Models run at **700 W, no bias** (a design-baseline
  choice) were being compared head-to-head with Mettler's 1000 W + bias
  dataset. This is an apples-to-oranges comparison regardless of model
  quality.
- **"Pure SF6" mismatch.**
  Mettler Fig 4.14 / 4.17 is **70 sccm SF6 / 30 sccm Ar** (Fig 4.14) or
  **30%/90% SF6 + Ar balance** (Fig 4.17); it is NEVER pure SF6. Running
  the model in pure SF6 and comparing to these figures is a composition
  mismatch that manifests as a 2× density error by itself.
- **Wafer-bias effect silently absorbed into other residuals.**
  Mettler's Fig 4.17 bias-on curve is ×1.6 (90% SF6) to ×2.15 (30% SF6
  centre) above the bias-off curve. A model with no bias-sheath module
  compared to Mettler's bias-on data is effectively missing half the
  absolute density by construction.

## Narrative Mistakes

- **"Within 5–8%" claim on the radial profile.**
  The residuals actually show +8% inside (ok) but +22–27% on the outer
  half of the wafer. The 5–8% number averages only over the inner
  points and misrepresents the outer-wafer over-prediction that is the
  actual interesting physics signal (a signature of the missing
  electronegative-ambipolar correction).
- **Missing explicit Helicon-vs-TEL clarifying sentence.**
  Any report that cites "Mettler 2025" without explicitly scoping which
  figures apply to the TEL etcher invites the Helicon/TEL conflation
  error later on.

---

## How the v2 reports fix each item

Spot-checked against the supervisor's updated Phase-1 and SF6/Ar reports
(shared 2026-04-18):

| # | Mistake | v2 report fix (verified) |
|---|---|---|
| 1 | BBAS+OES diagnostic | Phase-1 §4 cites "W/Al non-equilibrium radical probes cross-calibrated against Ar/SF6 actinometry" explicitly |
| 2 | Fig 4.5 as TEL power sweep | Phase-1 Fig 3.9 caption: "Mettler does not report a direct TEL power sweep; his absolute-[F] TEL data (Fig 4.9) is a SF6-flow sweep at fixed power (600 W/20 mTorr and 700 W/40 mTorr)" |
| 3 | Helicon/PMIC vs TEL conflation | Phase-1 §3.11 opens: "Mettler's thesis reports data from *two distinct experimental chambers*: a Plasma–Materials Interaction Chamber (PMIC) helicon source... and a TEL research-scale ICP etcher... Only the TEL dataset is relevant..." |
| 4 | "74%" universal | SF6/Ar v2 Fig 7.1 caption explicitly flags Fig 4.17 composition range (67% at 30:70 SF6:Ar, 76% at 90:10) vs Fig 4.14's 74% cubic-fit value (bias on, 70:30) |
| 5 | Loose cubic-fit digitisation | v2 re-digitisation matches the published cubic at (0,2,4,6,8) cm to <1% |
| 6 | 700W no-bias vs 1000W+bias | Phase-1 §7.5 uses 1000 W / 10 mTorr / 70% SF6 / 200 W bias as the canonical benchmark point with the m12 bias-sheath module active |
| 7 | Pure SF6 mismatch | Phase-1 Fig 7.22 runs 30% and 90% SF6 blind-test at Mettler's actual composition |
| 8 | Bias silently absorbed | Phase-1 Fig 7.24 explicitly plots the bias-on / bias-off ratio and validates the m12 calibration against ×1.6 (90% SF6) and ×2.15 (30% SF6) |
| 9 | "Within 5-8%" radial | Phase-1 Fig 7.31(a) explicitly reports +8% inner / +22-27% outer residuals |
| 10 | Missing PMIC vs TEL clarifier | Addressed in items 2 and 3 above |

All 10 items resolved in the v2 reports. Surviving violations in `active_projects/` or
`final_package/` (if any) will be enumerated in manifest 13 after the workspace audit.
