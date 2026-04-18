# CHANGES FROM V1 (06a) TO V2 (06b)

**Scope**: This document enumerates every textual and figure change applied to `main.tex` and `figures/` between the 06a baseline and the 06b corrected version. The only intended differences are those listed here; anything else in the diff is an accidental edit that should be reverted.

**Driving documents** (all in `docs/` of this folder):
- `METTLER_CORRECTIONS.md` — corrections C1–C8
- `METTLER_gap_report_CHATGPT_FEEDBACK.md` — validation points A1–A4, B1–B5, C1–C5, D1–D4
- `METTLER_TEL_DATA_DIGITISED.md` — hand-digitised Mettler numerical reference tables

---

## Textual corrections (N1–N14)

### N1 — §3 Mettler Benchmarking figure replacement (was fig 4.5)

**Driving issue**: E2 — Mettler Fig 4.5 is PMIC Helicon data, not TEL. Citing it as a TEL benchmark is factually wrong.

**Old** (06a line 1305–1307): `\includegraphics{mettler_benchmark_fig4p5.png}` captioned "F density vs ICP source power at 50 mTorr, 50% Ar."

**New** (06b): `\includegraphics{tel_power_sweep_1000W_30pctAr.png}` with a caption that (a) names Mettler Fig 4.17 30%-SF₆ bias-off as the reference marker, (b) explicitly notes the 06a report's Fig 4.5 mis-attribution, (c) discloses the volume-average vs centre comparison with the ~1.62× actinometry factor.

### N2 — §3 V3 anchor numbers (was vague "within 20%")

**Driving issue**: V3 — the report lacked explicit [F]_ICP anchor values at Mettler Fig 4.9 conditions.

**Old** (06a line 1324): "the model gives [F] = 7.4×10¹³ cm⁻³ at 30% SF₆ vs Mettler's 6.2×10¹³ cm⁻³, within 20%."

**New** (06b): Full paragraph quoting both Fig 4.9 anchor points: 600 W/20 mTorr/90 sccm SF₆ (model 3.4×10¹⁹ m⁻³ vs Mettler 9.5×10²⁰ m⁻³) and 700 W/40 mTorr/90 sccm SF₆ (model 3.9×10¹⁹ m⁻³ vs Mettler 2.0×10²¹ m⁻³), traceable to `output/v3_anchor_points.csv`. Gap explained via η uncertainty.

### N3 — §3 actinometry-to-centre ratio paragraph (NEW)

**Driving issue**: V4 — the ~1.62× actinometry-to-centre factor from Mettler Eq 4.2 was unstated.

**New** (06b): Dedicated paragraph citing Mettler Eq 4.2 with the n₀ = 2.5×10²⁰ vs n_act = 1.54×10²⁰ m⁻³ comparison.

### N4 — §3 design-operating-point disclosure (NEW)

**Driving issue**: E4 — report conflated the 1500 W pure SF₆ Lallement design point with Mettler's 1000 W / 70:30 SF₆:Ar benchmark.

**New** (06b): Paragraph distinguishing the two baselines and stating that comparisons are made at Mettler's conditions.

### N5 — §3 diagnostic description (was "diagnostic techniques")

**Driving issue**: E1 — Mettler's primary diagnostic is W/Al non-equilibrium radical probes cross-calibrated against actinometry, NOT BBAS or actinometry alone.

**Old** (06a line 1301): "…used a completely different reactor (TEL etcher) and diagnostic techniques."

**New** (06b): Paragraph explicitly naming "W/Al non-equilibrium radical probes (a reactive tungsten pellet and an inert aluminium pellet, each instrumented with a thermocouple) cross-calibrated against Ar/SF₆ optical emission actinometry in the ICP region" and noting "Neither broadband absorption spectroscopy (BBAS) nor actinometry alone is the primary measurement."

### N6 — §3 Helicon vs TEL distinction (NEW)

**Driving issue**: E6, C5 — Mettler's thesis has TWO reactors; only TEL (Ch 4.2–4.3) is used here. 06a conflated them.

**New** (06b): Paragraph stating "Mettler's thesis reports data from two distinct experimental chambers: a Plasma-Materials Interaction Chamber (PMIC) helicon source used for probe-material screening (Mettler Ch. 4.1, Figs 4.1–4.5), and a TEL research-scale ICP etcher used for all spatially resolved radical-density data (Ch. 4.2–4.3, Figs 4.6–4.19). Only the TEL dataset is relevant to the SF₆/Ar global model validated here; Helicon/PMIC data are not used as benchmarks in this report."

### N7 — §4 TEL Case 1 figure (was mis-attributed Helicon Fig 4.5)

**Driving issue**: E2 repeat — second instance of the same Helicon mis-attribution in the wall-chemistry chapter.

**Old** (06a line 1545): `\includegraphics{fig7_tel_F_vs_power.png}` with "\SI{50}{\milli\torr}, 50\% Ar" caption.

**New** (06b): `\includegraphics{tel_power_sweep_1000W_30pctAr.png}` (same figure as §3 N1) with caption explaining the withdrawal of the original figure.

### N8 — §4 TEL Case 2/3 captions (added Fig 4.9 / 4.17 names + V3 numbers + 67–76% range)

**Driving issue**: C2, C3 — captions did not cite Mettler's figure numbers or state the condition specificity.

**Old** (06a lines 1558, 1565): "TEL Case~2: [F] vs SF₆ flow rate at 20 and 40 mTorr." / "TEL Case~3: Center-of-wafer fluorine density at 10 mTorr, 1000 W, bias off."

**New** (06b): Case 2 cites Mettler Fig 4.9 explicitly with the V3 anchor numbers and the 1.1×10²¹ m⁻³ kinetic threshold. Case 3 cites Mettler Fig 4.17 with the 67%–76% range and explicit rejection of "74% = Fig 4.17 bias-off" conflation. Figure files swapped to `mettler_fig49_overlay.png` and `mettler_fig417_overlay.png`.

### N9 — Chapter 7 γ_Al = 0.18 calibration narrative

**Driving issue**: C7, E5 — "calibrated to reproduce Mettler's 74% drop" without qualification misleads the reader.

**Old** (06a line 2048): "The model predicts a 74\% centre-to-edge F density drop at the wafer surface, matching the Mettler measurement."

**New** (06b): Full sentence linking γ_Al = 0.18 to the Fig 4.17 90:10 SF₆:Ar bias-off branch specifically, stating the 10 pp shortfall at 30%-SF₆ from the Stage-7 run, and referring to the F4 deferred composition sweep.

### N10 — Stage 7 "matches Mettler" claim

**Driving issue**: E5 second location — the Stage 7 description overstated the match.

**Old** (06a line 2030): "Validation against Mettler yields a 74% centre-to-edge F density drop, matching the experimental value."

**New** (06b): Honest description of the Stage-7 result (~58% drop at r=0–8 cm for both compositions), matches Mettler's 30%-SF₆ (67%) within the range but underpredicts the 90%-SF₆ (76%) and Fig 4.14 cubic-fit (74%, bias-on) by ~10–18 pp. Inserts the new `fig710_radial_F_1000W_regenerated.png` figure (Fig 7.10) showing both compositions vs Mettler's three reference curves.

### N11 — Chapter 7 Limitations bullet 1 (74% → 67–76% range)

**Driving issue**: E3 — "74%" as a universal Mettler number conceals composition/bias dependence.

**Old** (06a line 2001): "Mettler measured a 74% centre-to-edge fluorine density drop at the wafer surface—a spatial effect entirely outside the scope of a volume-averaged model."

**New** (06b): Bullet now states "a 67%–76% centre-to-edge [F] drop across the compositions in Fig 4.17 (67% at 30:70 SF₆:Ar bias-off, 76% at 90:10 SF₆:Ar bias-off); his Fig 4.14 cubic-fit value of 74% applies specifically to 70:30 SF₆:Ar with 200 W bias on."

### N12 — Chapter 7 Limitations NEW bullet L5 (kinetic threshold)

**Driving issue**: E7, C8 — the kinetic-vs-diffusion-limited transition threshold was unstated.

**New** (06b): New bullet L5 citing Mettler's ~1.1×10²¹ m⁻³ threshold (Fig 4.9 + Table 4.4), confirming all Phase-1 results are in the kinetic regime where 0D comparisons are valid.

### N13 — Chapter 7 Limitations NEW bullet L6 (gas temperature)

**Driving issue**: V6 — Mettler's 300 °C ICP gas temperature vs our 300 K model assumption was unstated.

**New** (06b): New bullet L6 citing Mettler p. 64 (~573 K from inert-Al probe) and noting the 3-body recombination rate sensitivity.

### N14 — Chapter 7 new §Deferred Mettler Validations (F1–F7, V5, Figs 4.12/4.16)

**Driving issue**: F1–F7 and V5 from `METTLER_VALIDATION_POINTS.md` plus the ChatGPT-flagged underused Figs 4.12 and 4.16.

**New** (06b): Entire new section before Roadmap Summary enumerating 8 deferred items with brief explanation and link to the capability extension required for each.

---

## Figure changes (F1–F7)

| # | Action | File | Was/Is |
|---|---|---|---|
| F1 | REMOVED | `mettler_benchmark_fig4p5.png` | was: Helicon Fig 4.5; now: replaced by `tel_power_sweep_1000W_30pctAr.png` |
| F2 | REMOVED | `fig7_tel_F_vs_power.png` | was: "TEL Case 1" mis-citing Helicon; now: same replacement |
| F3 | REMOVED | `mettler_benchmark_fig4p9b.png` | was: thin overlay; now: `mettler_fig49_overlay.png` with both branches + threshold line |
| F4 | REMOVED | `mettler_benchmark_fig4p17.png` | was: thin overlay; now: `mettler_fig417_overlay.png` with both compositions + bar chart |
| F5 | REMOVED | `fig8_tel_F_vs_flow.png` | was: "TEL Case 2"; now: same `mettler_fig49_overlay.png` |
| F6 | REMOVED | `fig9_tel_center_F.png` | was: "TEL Case 3"; now: `mettler_fig417_overlay.png` |
| F7 | ADDED | `tel_power_sweep_1000W_30pctAr.png` | NEW: Phase A.4 driver output (wall-chem 0D, TEL geometry, 1000 W condition) |
| F8 | ADDED | `mettler_fig49_overlay.png` | NEW: Phase A.2, SF₆-flow sweep vs Mettler Fig 4.9 two branches |
| F9 | ADDED | `mettler_fig417_overlay.png` | NEW: Phase A.3, centre [F] vs Mettler Fig 4.17 two compositions |
| F10 | ADDED | `fig710_radial_F_1000W_regenerated.png` | NEW: Phase A.5, Stage-7 2D radial profile vs Mettler Fig 4.14 cubic fit + Fig 4.17 data |

Net delta: **6 removed + 4 added = 81 figures → 79 figures**. (All removed figures remain in the 06a folder untouched as an audit baseline.)

---

## Values NOT changed (everything else)

Every other numerical value in the report is carried over verbatim: all figure captions for Chapters 2, 5, 6, 8, and Appendix A; every governing equation in Chapters 2–6; every table (gas-phase species, reaction mechanism, Troe parameters, surface reactions, Kokkoris parameters, kink locus); every α, n_e, T_e, ε_c value quoted outside the Mettler section; every citation that is not a Mettler citation; the abstract; the table of contents structure; the entire front matter.

The single exception is the cross-reference `\ref{sec:sensitivity_eta}` (which I initially mis-typed in the new §3 text) corrected to `\ref{sec:sens_eta}` to match the actual label in §6.2.
