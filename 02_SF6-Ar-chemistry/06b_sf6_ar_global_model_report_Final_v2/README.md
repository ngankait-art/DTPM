# 06b — Zero-Dimensional SF₆/Ar Global Model Report (Corrected v2)

This folder is a **self-contained, standalone** replacement for `06a_sf6_ar_global_model_report[Final]/`, incorporating the Mettler benchmarking corrections catalogued in three source-of-truth markdown files (in `docs/`):

- `METTLER_CORRECTIONS.md` — corrections C1–C8.
- `METTLER_gap_report_CHATGPT_FEEDBACK.md` — independent external review.
- `METTLER_TEL_DATA_DIGITISED.md` — hand-digitised Mettler numerical tables.

## What's here

```
06b_sf6_ar_global_model_report[Final_v2]/
├── README.md                       ← you are here
├── main.tex                        ← LaTeX source (2 267 lines, 14 edits from 06a)
├── references.bib                  ← unchanged from 06a
├── main.pdf                        ← latest compiled output (78 pages, 8.37 MB)
├── Zero-Dimensional…_v1.pdf        ← ORIGINAL 06a PDF (audit baseline)
├── Zero-Dimensional…_v2.pdf        ← corrected PDF
├── figures/                        ← 79 PNGs (75 carried over + 4 new, 6 removed)
├── code/
│   ├── kokkoris_pure_sf6/          ← Chapter 2 figures
│   ├── lallement_sf6ar/            ← Chapter 3 figures (bare Lallement 0D)
│   ├── wall_chemistry/             ← Chapter 4 figures (γ_Al, θ_F, Eley–Rideal)
│   ├── kink_analysis/              ← Chapters 5–6 figures
│   ├── tel_2d/                     ← Stage-10 TEL 2D package (reference only)
│   └── driver/                     ← NEW orchestration scripts for 06b
├── data/
│   ├── mettler/                    ← 21 digitised Mettler CSV files
│   ├── stage7_cached/              ← cached Stage-7 2D NPY arrays (for Fig 7.10)
│   └── rootcause_diagnostic_data.csv
├── docs/
│   ├── METTLER_CORRECTIONS.md      ← C1–C8 source of truth
│   ├── METTLER_TEL_DATA_DIGITISED.md
│   ├── METTLER_gap_report_CHATGPT_FEEDBACK.md
│   ├── CHANGES_FROM_V1.md          ← what changed 06a → 06b (N1–N14, F1–F6)
│   ├── FIGURE_PROVENANCE.md        ← which generator produced each figure
│   └── COLLABORATOR_REQUESTS.md    ← figures whose source code is missing
└── output/                         ← simulation traceability CSVs
```

## How to rebuild the PDF from source

```bash
cd 06b_sf6_ar_global_model_report[Final_v2]
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
# optional: cp main.pdf "Zero-Dimensional Global Modelling of SF6:Ar ICP Plasma_v2.pdf"
```

Expected result: 78 pages, ~8.4 MB, zero undefined references.

## How to reproduce the four new figures

Requires Python 3 with `numpy`, `scipy`, `matplotlib` (standard scientific packages — no GPU, no torch).

```bash
cd 06b_sf6_ar_global_model_report[Final_v2]/code/driver
python3 run_v3_anchors.py                   # Phase A.1 → output/v3_anchor_points.csv
python3 run_mettler_fig49_overlay.py        # Phase A.2 → figures/mettler_fig49_overlay.png
python3 run_mettler_fig417_overlay.py       # Phase A.3 → figures/mettler_fig417_overlay.png
python3 run_tel_power_sweep.py              # Phase A.4 → figures/tel_power_sweep_1000W_30pctAr.png
python3 run_fig710_2d_regeneration.py       # Phase A.5 → figures/fig710_radial_F_1000W_regenerated.png
```

Each script takes <1 minute (A.5 is near-instant — it uses cached stage-7 NPY arrays).

## Key differences from 06a

**Text**: 14 enumerated edits (N1–N14) — see `docs/CHANGES_FROM_V1.md` for the full list with before/after.
**Figures**: 6 removed (3 superseded Mettler overlays + 3 "TEL Case" figures mis-citing Helicon Fig 4.5), 4 new (Mettler Fig 4.9 overlay, Fig 4.17 overlay, TEL power sweep replacement, Fig 7.10 regenerated from Stage-7 2D).

## What this report does NOT fix

Six correction items (F1–F6, V5) flagged in the audit were deferred to Phase 2 because they require capabilities outside the 0D scope (bias sheath model, 2D ambipolar correction, near-wall boundary layer, composition sweep, E/H transition, spatial wall temperature). See Chapter 7 §Deferred Mettler Validations for the full list.

**Eleven figures** in this report were carried over verbatim from 06a because their generator scripts are not yet located. See `docs/COLLABORATOR_REQUESTS.md` for the list to ask collaborators about.

## Authoring

- Original 06a report: generated in March–April 2026 from the DTPM delivery package.
- 06b corrections pass: April 2026, driven by three audit markdown files (`docs/METTLER_*.md`).
- Simulation code: copied verbatim from the DTPM_Complete_Delivery subfolders, no modifications beyond the driver adapters in `code/driver/`.
