# Mettler Benchmarking Corrections — 06a Consolidated SF6/Ar Global Model Report

**Applies to**: `06a_sf6_ar_global_model_report[Final]/main.tex`

**Purpose**: This document records factual errors and operational-condition
mismatches found in how the 06a consolidated report cites Jeremy D. Mettler's
2025 UIUC dissertation. Use this in a separate Claude Code thread (opened on
the `06a_sf6_ar_global_model_report[Final]/` directory) to drive corrections.

**Source of truth**: `Literature/Mettler Dissertation.pdf` (105 pp, UIUC 2025).

**Companion documents (in the Phase-1 thread)**:
- `Steps/5.Phase1_EM_Chemistry_Merged/docs/METTLER_VALIDATION_POINTS.md` — full list of what Mettler measured, current-vs-future validation split.
- `Steps/5.Phase1_EM_Chemistry_Merged/docs/METTLER_CORRECTIONS.md` — same style of corrections but for the Phase-1 report thread.

---

## Key facts about Mettler's TEL dataset (the only part that benchmarks SF6/Ar models)

| Fact | Value | Source |
|---|---|---|
| Diagnostic | W/Al non-equilibrium radical probes + Ar/SF6 actinometry | Mettler Ch 3.2, 4.3 |
| Reactor | TEL research-scale ICP etcher (Tokyo Electron) | §3.1.2 |
| Radial [F] benchmark conditions | 1000 W ICP, 10 mTorr, 70 sccm SF6 / 30 sccm Ar, 200 W rf bias | Fig 4.14 / 4.17 |
| Normalised radial [F] cubic fit | y(r) = 1.01032 − 0.01847 r² + 7.139×10⁻⁴ r³ (r in cm) | Fig 4.14, R² = 0.997 |
| Centre-to-edge [F] drop range | 67%–75% depending on SF6/Ar composition | Fig 4.17 |
| 74% specific condition | 90% SF6 (90 sccm SF6 / 10 sccm Ar), bias off | Fig 4.17 top panel |
| ICP-region [F] (absolute) | ~1.0×10²¹ /m³ @ 600 W/20 mTorr/90 sccm; ~2.0×10²¹ /m³ @ 700 W/40 mTorr/90 sccm | Fig 4.9 |
| Kinetic → diffusion-limited threshold | ~1.1×10²¹ /m³ | Fig 4.9, Table 4.4 |
| F gas temperature (ICP region) | ~300 °C (573 K) | p. 64 |
| Si etch probability (spatially resolved) | ε_Si ≈ 0.025–0.04, flux-independent | Fig 4.18 |
| Actinometry-vs-local underestimation | actinometry returns ~0.6× the centre density | Eq 4.2, Fig 4.14 |

---

## What to check in the 06a report

Open `06a_sf6_ar_global_model_report[Final]/main.tex` and grep for "Mettler". For each hit, verify against the list below.

### C1. Diagnostic description
- **Wrong if the text says**: "broadband absorption spectroscopy", "BBAS", "optical emission spectroscopy", "OES" (alone, as the primary measurement).
- **Correct statement**: "W/Al non-equilibrium radical probes cross-calibrated against Ar/SF6 actinometry in the ICP region". Actinometry *alone* is also wrong — Mettler's novel contribution is the probe technique; actinometry is only the cross-calibration reference.

### C2. Centre-to-edge [F] drop
- **Wrong if the text says**: "74% drop" without qualification, as a universal Mettler number.
- **Correct statement**: "Mettler reports a centre-to-edge drop of 67%–75% depending on SF6/Ar composition (Fig 4.17). The 74% value is specifically the 90% SF6 / bias-off case at 1000 W, 10 mTorr."

### C3. Operating conditions of the radial benchmark
- **Wrong if the text says**: "Mettler measured the radial profile at 700 W" or "at no bias" or "in pure SF6".
- **Correct statement**: "Mettler's radial [F] profile (Fig 4.14, Fig 4.17) was measured at P_ICP = 1000 W with 200 W rf wafer bias, in a 70 sccm SF6 / 30 sccm Ar mixture at 10 mTorr."

### C4. "Fig 4.5" is NOT the TEL power sweep
- **Wrong if the text cites**: "Mettler Fig 4.5" as a TEL benchmark.
- **Correct statement**: Fig 4.5 is Mettler's Helicon / PMIC W-Al probe calibration curve (not TEL). The correct TEL reference for absolute [F] is **Fig 4.9** (SF6 flow sweep at fixed power/pressure). Mettler does **not** report a direct TEL power sweep.

### C5. Helicon vs TEL
- **Wrong if the text**: treats all Mettler figures as TEL.
- **Correct statement**: Mettler's Ch. 4.1 (Fig 4.1–4.5, Table 4.2) is Helicon/PMIC data for probe screening only. Only Ch. 4.2–4.3 (Fig 4.6–4.19) are TEL measurements and thus appropriate for benchmarking an SF6/Ar ICP model.

### C6. Wafer bias effect (if the report claims a "no-bias" condition)
- If the 06a report runs a no-bias simulation and compares to a Mettler "bias-on" figure (Fig 4.14 or Fig 4.17 top-bottom panels), the bias introduces a ×1.6 enhancement at 90% SF6, and up to ×2.15 at 30% SF6 (centre). Either (a) include this correction factor explicitly, or (b) compare only to Mettler's bias-off curves (Fig 4.17, which shows both).

### C7. γ (recombination coefficient) calibration target
- **Wrong if the text says**: "calibrated to reproduce Mettler's 74% drop" without qualification.
- **Correct statement**: "γ was calibrated to the 90% SF6 / bias-off case of Mettler Fig 4.17. Composition-dependent validation (vs the 30% SF6 branch) is a separate open test."

### C8. Kinetic regime statement (optional addition)
- Missing in most reports: Mettler's data has a diffusion-limited threshold at ~1.1×10²¹ /m³ (his Fig 4.9, Table 4.4). An SF6/Ar global model whose predicted [F] stays below this is safely in the kinetic regime; one that exceeds it needs surface-boundary-layer treatment.

---

## Additional benchmarks that should be present (if currently missing)

The following TEL datasets from Mettler are valuable for an SF6/Ar global model and should be considered for inclusion:

1. **Fig 4.9 ICP-region [F] vs SF6 flow** (at 600 W/20 mTorr and 700 W/40 mTorr). Direct test of 0D SF6 dissociation chemistry at volume-averaged densities.
2. **Fig 4.17 composition sweep** — 30% vs 90% SF6 — tests the electronegative/electropositive balance sensitivity of the model.
3. **Fig 4.14 cubic-fit overlay** — if the 06a report includes any radial profile output (e.g. from a follow-up 2D solve), overlay Mettler's cubic fit directly.
4. **Actinometry / local-density ratio ~1.6** — useful to cross-check any 2D model that outputs both volume-averaged and centre values.

---

## How to use this file in a separate Claude Code thread

```
cd "Steps/2.plasma-Radical-Models/2.SF6_Plasma_Model/1.DTPM_Zachariah_reports_March-2026/DTPM_Complete_Delivery/06a_sf6_ar_global_model_report[Final]"
claude
```

Then paste or reference:
> Read `METTLER_CORRECTIONS.md` (in this directory). Apply the corrections to `main.tex` and produce a diff-style summary of what changed. Do not re-run any simulations — citation and caption corrections only.
