# Supplementary: identifiability / M7 side-study

**Author:** Zachariah Ngan, Illinois Plasma Institute
**Status:** Exploratory side-study. Not the Phase 2 deliverable.

---

## What this directory is

This directory holds an analytical side-study on **two-anchor etch-rate calibration identifiability and EEDF sensitivity** for an SF₆ ICP, built during an earlier session of work on this project. It contains:

- `manuscript_or_note.tex` — the original manuscript written during that side-study (the identifiability paper)
- `sections/` — the section files that manuscript imports
- `references.bib` — its bibliography
- `figures/` — the seven figures used by that manuscript (fig1 pipeline schematic, fig2 rate ratios, fig3 etch comparison, fig4 ν comparison, fig5 β validation, fig6 prediction residual, fig7 observability sweep)
- `memos/` — supporting notes on the pressure correction and the administrative bundle of that earlier work

## Why it is in `supplementary/`

**This work is not the Phase 2 deliverable.** It addresses a different question from the one Muhammad's April 2026 workplan asks. It was built in an earlier session before the supervisor's workplan was loaded into context, and at the time the scope was inferred from earlier project knowledge rather than read directly from the workplan document.

The Phase 2 workplan asks: *"Build a three-tier pipeline that replaces the Maxwellian EEDF assumption in the DTPM reduced-order model with a Boltzmann-derived kinetics layer, and determine how much replacing the assumption changes the fluorine profile prediction."*

This side-study asks something different: *"Given two literature etch-rate anchors, which physical parameters of a reduced-order SF₆ etch model are identifiable from the anchors alone, and which are not?"*

The two questions are not the same. The Phase 2 answer is a production-ready electron-kinetics module with a specific `get_rates_pinn()` API. The side-study answer is a closed-form identifiability theorem about a particular calibration configuration. They use some of the same building blocks (BOLSIG+ rate tables, LXCat cross sections, the Mansano-1999 and Panduranga-2019 literature anchors) but they are not substitutes for each other.

## What the side-study actually shows

For readers who want to engage with the side-study on its own terms, the main internally-consistent results are:

1. **The closed-form identifiable invariant.** Two etch-rate anchors at different pressures constrain only the product
   ```
   𝓘 = β · k_diss · n_e(P_H)
   ```
   where β is the fitted SF₆ depletion timescale, k_diss is the dominant dissociation rate coefficient, and n_e(P_H) is the electron density at the high-pressure anchor. The three factors are not individually identifiable from the two anchors alone; only their product is. This is a genuine mathematical result about that calibration scheme and it holds independently of whether the underlying EEDF is Maxwellian or Boltzmann-derived.

2. **The rate-source-swap prediction test.** The invariant 𝓘 transforms cleanly when the rate coefficients are computed from a Boltzmann-derived EEDF rather than a Maxwellian one, because both factor changes (in k_diss and in the implied n_e) cancel in the product. This means the etch-rate prediction at the intermediate-pressure validation point is approximately source-invariant, which is the sense in which the side-study's "Maxwell-vs-Boltzmann sensitivity" result exists.

3. **The Ar-dilution observability sweep.** Adding a third calibration anchor at a nonzero Ar fraction breaks the degeneracy and restores individual identifiability of β and k_diss — a straightforward observability result from the sweep calculation.

None of these results is wrong. They are just not the Phase 2 question.

## Relationship to the primary Phase 2 deliverable

The primary Phase 2 deliverable lives at the top level of this repository:

- `tier1_bolsig/` — BOLSIG+ lookup tables and Maxwellian comparison
- `tier2_pinn/` — the M5 supervised surrogate and the `get_rates_pinn()` production API
- `tier3_picmcc/` — the MCC collision core and the three workplan-specified validation cases (A/B/C)
- `report/technical_report.pdf` — the Tier-framed technical report Muhammad should read

The M7 side-study in this directory **shares some figures with the primary deliverable** (specifically `fig2_rate_ratios.pdf`, which is a re-export of the Tier 1 rate-ratio comparison) but its manuscript structure is organized around the identifiability theorem rather than the three-tier pipeline. **If you came to this repository to evaluate the Phase 2 workplan execution, do not read this directory first. Read `report/technical_report.pdf` at the top level.**

## If someone cites the side-study

The side-study's identifiability result is publishable on its own as a short analytical note. If it is going to be published, it should be published **as a methodological note separate from the Phase 2 deliverable**, not bundled with it. The two audiences are different: Phase 2 is for the internal DTPM integration pipeline, the side-study is for the plasma-model-calibration community. Muhammad should decide whether the side-study is worth carrying forward; if it is, it should be reframed as its own standalone paper rather than as a section of the Phase 2 report.

## Build instructions

The side-study manuscript in this directory is preserved as it was written — it has not been edited since being moved here. If you want to compile it, you will need to:

1. `cd supplementary/identifiability_M7`
2. `pdflatex manuscript_or_note.tex`
3. `bibtex manuscript_or_note`
4. `pdflatex manuscript_or_note.tex` (twice more for cross-references)

The figures in `figures/` are the same as the Phase 2 paper stub figures at `paper/figures/`.

---

**Last updated:** April 2026 (Phase 2 final reframing pass)
**Contact:** Zachariah Ngan, Illinois Plasma Institute, zngan@illinois.edu
