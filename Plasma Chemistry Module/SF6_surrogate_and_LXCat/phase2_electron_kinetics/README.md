# Phase 2 Electron Kinetics Pipeline

**Replacing the Maxwellian EEDF assumption in the SF₆ ICP reduced-order model.**

**Author:** Zachariah Ngan, Illinois Plasma Institute (`zngan@illinois.edu`)
**Status:** Phase 2 deliverable against the April 2026 supervisor workplan.

---

## What this repository is

This is the Phase 2 electron-kinetics deliverable for the Illinois Plasma Institute SF₆ ICP modelling project. Phase 1 (the DTPM two-dimensional axisymmetric fluid model of the TEL etcher) currently evaluates every electron-impact rate coefficient from a single-temperature Arrhenius form `k_j(Te) = A_j * exp(-E_j / Te)`, which implicitly assumes a Maxwellian EEDF. Phase 2 replaces that assumption with a Boltzmann-derived kinetics layer and asks whether the correction moves the fluorine profile prediction enough to justify the infrastructure change.

The supervisor's workplan (`PHASE2_ELECTRON_KINETICS_WORKPLAN.md`, April 2026) specifies a **three-tier programme** executed in order, with a decision gate between tiers. This repository is the direct implementation of that workplan:

| Tier | What it builds | Decision it enables | Directory |
|---|---|---|---|
| **Tier 1** | BOLSIG⁺ lookup table (168 points × 53 channels) | Does Maxwellian matter? (20% test at ref) | `tier1_bolsig/` |
| **Tier 2** | Supervised MLP surrogate + `get_rates_pinn()` API | Fast differentiable production kinetics | `tier2_pinn/` |
| **Tier 3** | Monte Carlo collision core + 3 workplan cases | Reusable physics layer for spatial PIC-MCC | `tier3_picmcc/` |

**If you are Muhammad (or any supervisor) reading this repository for the first time**, open `report/technical_report.pdf` first. That is the primary deliverable. The rest of this README exists to explain the layout so that the reader can navigate to the underlying code and data.

---

## Project Scope and Status

**This section is the single most important thing to read before evaluating the repository.** It states the gap between the full workplan and what has actually been implemented, so that nothing below is misread.

| Tier | Workplan requirement | Status | Notes |
|---|---|---|---|
| **Tier 1** | BOLSIG⁺ lookup on (E/N, x_Ar) grid + Maxwell vs Boltzmann comparison + 20% decision gate | ✅ **Complete** | 168-point grid, 53 channels, gate met at reference point |
| **Tier 2** | PINN / supervised surrogate trained on Tier 1, < 10% rate-coefficient error, `get_rates_pinn()` production API | ✅ **Complete** | 19,397-parameter MLP; median 0.41% error on Te_eff, 1.49% on k_att; API live and callable |
| **Tier 3** | MCC collision module **coupled to the existing Boris pusher** (M07/M08), spatial PIC-MCC runs at three operating points, non-local transport answer | ⚠️ **Partial — 0D cross-check only** | MCC collision core built and validated; three workplan cases executed **as 0D runs, not spatial PIC-MCC**; coupling to the Boris pusher not yet done |

### What Tier 3 delivers in the current package

- A working 0D Vahedi–Surendra null-collision MCC module (`tier3_picmcc/mcc_module.py`) that reads real LXCat Biagi SF₆ cross sections and runs macro-electrons under a prescribed uniform field.
- Three workplan-specified cases (A: 700 W / 10 mTorr / pure SF₆; B: 700 W / 5 mTorr / pure SF₆; C: 700 W / 10 mTorr / 50% Ar) executed at a representative E/N per case, with Te_eff agreement within 15% of BOLSIG⁺ at all three points and correct physical trends reproduced.
- A validated, reusable collision physics layer ready for the spatial Boris pusher to consume.

### What Tier 3 does NOT deliver in the current package

- **No spatial PIC-MCC run.** The MCC core is not coupled to the existing Boris pusher at `1.E-field-map/.../7.DTPM_Project_ICP/` modules M07/M08.
- **No spatially-resolved k_j(r, z) maps.**
- **No direct answer to the workplan §5.5 non-local transport question.** A 0D MCC cannot answer that by construction.
- **No DTPM Picard-loop integration of `get_rates_pinn()`.** The Tier 2 surrogate is validated against BOLSIG⁺ in isolation but has not yet been wired into the Stage 10 DTPM loop for a fluorine-profile re-validation.

### Why the Tier 3 scope is bounded at 0D

The short version, in one sentence: **the current deliverable is the kinetics-isolation step (Phase 2a); the spatial integration step (Phase 2b) is the explicit next package and needs its own time budget.**

The longer version:

1. **The spatial Boris pusher lives in a separate code base.** Modules M07 and M08 sit at `1.E-field-map/script_v2/2.Extended-Aspects/7.DTPM_Project_ICP/` in the project tree, not in this Phase 2 repository. Coupling the MCC core to the pusher requires modifications to the pusher's time-step loop and EM-field coupling interface, which belong in that upstream repository rather than in this standalone package.
2. **A working collision core is a prerequisite, not a shortcut.** Landing the collision physics as a standalone module first, validating it against BOLSIG⁺ at three operating points, and confirming the expected Te agreement and physical trends is a direct prerequisite for the spatial coupling. Without this step, any disagreement in the eventual spatial run would conflate collision-core bugs with real non-local effects.
3. **Isolating electron kinetics from spatial transport is necessary for diagnosis.** The workplan §5.5 question is specifically about the *difference* between a spatial MCC run and a cell-by-cell Tier 2 surrogate lookup. To attribute that difference cleanly to non-local transport, Tiers 1 and 2 and a 0D Tier 3 cross-check all need to agree with each other first. The current deliverable lands exactly that combination of checks.
4. **Computational and engineering cost.** A single self-consistent spatial PIC-MCC run takes hours to days of wall clock; the three workplan cases plus the coupling software-engineering work is a multi-week package. Splitting the Phase 2 work so that Phase 2a can be delivered, reviewed, and gated before Phase 2b begins is the same separation of concerns a PIC-MCC paper author would use when introducing a new collision module for an unfamiliar gas.

**The full reconciliation** (including the scientific value the current deliverable has independently of Phase 2b, the complete list of what remains, and the concrete next-step plan) lives at `report/motive_reconciliation.pdf`. Read that document if you want the detailed story; read the table above if you just want the status at a glance.

---

## Decision gates — at a glance

- **Tier 1 (workplan §3.4):** at E/N = 50 Td, pure SF₆, the Maxwell/Boltzmann ratios are 0.82 (k_att), 1.04 (k_diss-like), 0.97 (k_el) — all within the 20% band. Ionisation is below the numerical floor at this point (both in Maxwell and in Boltzmann) because the 15.7 eV threshold is far above the EEDF bulk.  **Gate: met.**
- **Tier 2 (workplan §4.5):** 19,397-parameter supervised MLP validated against full 168-point BOLSIG⁺ grid. Te_eff median error 0.41%, k_att median error 1.49%. Both comfortably under the 10% acceptance gate.  **Gate: met.**
- **Tier 3 (workplan §5.5):** Null-collision MCC core built against real LXCat Biagi cross sections. Three workplan cases (A/B/C) executed **as 0D cross-checks** (not spatial PIC-MCC); Te_eff within 15% of BOLSIG⁺ at all three operating points. **Gate: met for the 0D cross-check. The non-local transport question requires the spatial integration step, which is Phase 2b and has not yet been executed.**

---

## Repository layout

```
phase2_electron_kinetics/
├── README.md                    <-- this file
├── LICENSE                      <-- MIT
├── requirements.txt             <-- Python dependencies
├── .gitignore
│
├── tier1_bolsig/                <-- Tier 1: BOLSIG+ lookup + Maxwellian comparison
│   ├── generate_lookup_tables.py
│   ├── compare_maxwellian_vs_bolsig.py
│   ├── plots/                   real PDF/PNG plots from real data
│   └── outputs/                 CSV projections + decision-gate markdown report
│
├── tier2_pinn/                  <-- Tier 2: supervised MLP surrogate + production API
│   ├── get_rates_pinn.py        <-- THE workplan §4.4 production API
│   ├── evaluate_surrogate.py    validates surrogate against Tier 1 grid
│   ├── models/mlp.py            architecture definition + checkpoint loader
│   ├── weights/                 real pre-trained checkpoints (m5, m6)
│   ├── evaluation/              training diagnostics + residual CSV
│   └── outputs/                 residual plot
│
├── tier3_picmcc/                <-- Tier 3: MCC collision core + 3 workplan cases
│   ├── mcc_module.py            Vahedi-Surendra null-collision core
│   ├── lxcat_parser.py          minimal LXCat reader
│   ├── run_case_A.py            700 W / 10 mTorr / pure SF6
│   ├── run_case_B.py            700 W / 5 mTorr / pure SF6
│   ├── run_case_C.py            700 W / 10 mTorr / 50% Ar
│   ├── analyze_cases.py         cross-case comparison
│   ├── results/                 JSON result files + summary.md
│   └── outputs/                 diagnostic PNGs
│
├── data/
│   ├── raw/
│   │   ├── bolsig_data.h5       real BOLSIG+ output, 1.4 MB
│   │   └── SF6_biagi_lxcat.txt  real LXCat cross sections, 3158 lines
│   └── processed/
│
├── report/                      <-- PRIMARY DELIVERABLE: technical report
│   ├── technical_report.tex
│   ├── technical_report.pdf     15 pages, Tier 1/2/3 structure
│   ├── references.bib
│   └── figures/                 staged from tier1/tier2/tier3
│
├── paper/                       <-- short redirection stub, not the primary deliverable
│   ├── manuscript.tex
│   ├── manuscript.pdf           2 pages, explains where to look
│   └── figures/
│
├── slides/                      <-- supervisor presentation deck
│   ├── slides.tex
│   ├── slides.pdf               13 frames, Tier 1/2/3 narrative
│   └── speaker_notes.md         slide-by-slide talking points
│
├── supplementary/
│   ├── SI.tex, SI.pdf           supplementary information
│   ├── identifiability_M7/      <-- exploratory side-study, NOT Phase 2 deliverable
│   │   ├── README.md            reframing note explaining what this is
│   │   ├── manuscript_or_note.tex
│   │   ├── sections/
│   │   ├── references.bib
│   │   ├── figures/
│   │   └── memos/
│   └── derivations/
│
└── outputs/
    ├── figures_300dpi/          high-DPI exports (if regenerated)
    ├── logs/
    └── final_zip/
```

---

## How to reproduce the results

All three tiers run on a plain laptop with Python ≥ 3.10 and the packages in `requirements.txt`. No GPU, no BOLSIG⁺ binary, no external infrastructure.

### Install

```bash
pip install -r requirements.txt
```

### Run Tier 1

```bash
python tier1_bolsig/generate_lookup_tables.py
python tier1_bolsig/compare_maxwellian_vs_bolsig.py
```

Outputs:
- `tier1_bolsig/outputs/lookup_summary.txt` — what's in the HDF5
- `tier1_bolsig/outputs/maxwell_vs_bolsig_report.md` — **decision gate report**
- `tier1_bolsig/outputs/rates_{boltzmann,maxwell}_pure_SF6.csv` — rate tables
- `tier1_bolsig/plots/ratio_vs_EN.png` — visual comparison

### Run Tier 2

```bash
python -m tier2_pinn.get_rates_pinn       # runnable example: 6-point table
python tier2_pinn/evaluate_surrogate.py   # validate against Tier 1 grid
```

The first command prints a small table of Te_eff and rate coefficients. The second runs the full 168-point validation and produces:
- `tier2_pinn/evaluation/surrogate_vs_bolsig.csv` — per-point residuals
- `tier2_pinn/evaluation/surrogate_error_summary.md` — **acceptance gate report**
- `tier2_pinn/outputs/surrogate_error_plot.png` — visual

### Run Tier 3

```bash
python tier3_picmcc/run_case_A.py
python tier3_picmcc/run_case_B.py
python tier3_picmcc/run_case_C.py
python tier3_picmcc/analyze_cases.py
```

Each case runner takes ~10 s on a laptop. The analyzer produces:
- `tier3_picmcc/results/tier3_summary.md` — **cross-case decision report**
- `tier3_picmcc/outputs/case_comparison.png` — MCC vs BOLSIG⁺ bar chart
- `tier3_picmcc/outputs/case_{A,B,C}_mcc.png` — per-case diagnostic plots

### Compile LaTeX documents

```bash
cd report && pdflatex technical_report.tex && pdflatex technical_report.tex
cd ../slides && pdflatex slides.tex
cd ../paper && pdflatex manuscript.tex
cd ../supplementary && pdflatex SI.tex
```

---

## The `get_rates_pinn()` production API

This is the single most important deliverable for DTPM integration. It is the exact function signature Muhammad's workplan §4.4 asks for:

```python
from tier2_pinn.get_rates_pinn import get_rates_pinn
import numpy as np

E_over_N = np.array([10, 30, 50, 100, 300])   # in Td
rates = get_rates_pinn(E_over_N, x_Ar=0.0, pressure_mTorr=10.0)

rates['Te_eff']   # (5,) array, effective electron temperature in eV
rates['k_iz']     # (5,) array, total ionisation rate (m^3/s)
rates['k_att']    # (5,) array, total attachment rate (m^3/s)
rates['k_diss']   # (5,) array, aggregated dissociation rate (m^3/s)
rates['k_exc']    # (5,) array, aggregated excitation rate (m^3/s)
```

The model is cached at module level, so DTPM can call this function in a tight inner loop without paying the checkpoint-load cost on each call. The backend defaults to the supervised MLP checkpoint at `tier2_pinn/weights/m5_surrogate.pt`; pass `weights_path=...` to swap in the physics-informed M6 PINN.

**Drop-in replacement** for the current Arrhenius evaluation block in the Stage 10 DTPM Picard loop.

---

## Known caveats

### Tier 1
- Single pressure (10 mTorr). Pressure dependence assumed weak at the DTPM operating regime per workplan §4.6, but not verified over the full 5–30 mTorr range.
- Ionisation rates are below the numerical floor (10⁻²² m³/s) for E/N ≲ 150 Td; this is a property of BOLSIG⁺ at these conditions, not a code bug.

### Tier 2
- Pressure is accepted in the `get_rates_pinn()` signature for API stability but not currently used by the surrogate (consistent with workplan §4.6 permission to drop pressure as an input dimension).
- The p99 relative errors on k_iz and k_diss are dominated by low-E/N floor points where the ground truth is zero; the production-regime residuals (30 ≤ E/N ≤ 300 Td) are all under 10%.

### Tier 3
- **0D cross-check only.** The MCC module here runs in a uniform prescribed field with no spatial grid, no Poisson solve, and no Boris pusher coupling. It is the reusable physics layer, not the full spatial PIC-MCC run.
- Attachment rates are systematically lower than BOLSIG⁺ (by a factor of 5–10) because the 0D runs have finite duration and have not yet reached the attachment-drained steady state BOLSIG⁺ assumes. This does not affect the Te_eff agreement, which is the primary metric.
- No secondary electrons in ionisation channels (acceptable at the sub-percent ionisation fractions of interest; required for full self-consistent PIC-MCC).
- **The non-local transport question of workplan §5.5 cannot be answered by this 0D cross-check.** That is the stated next step: couple `tier3_picmcc/mcc_module.py` to the existing Boris pusher at `1.E-field-map/script_v2/2.Extended-Aspects/7.DTPM_Project_ICP/` (modules M07/M08) and re-run Cases A/B/C with spatial resolution.

### Supplementary M7 side-study
- The identifiability work at `supplementary/identifiability_M7/` is an earlier analytical side-study that addresses a different question from Phase 2. It is **not** the Phase 2 deliverable. See the README in that directory for the full reframing explanation.

---

## Recommended next steps (workplan end-of-Phase-2 outcomes)

1. **Spatial PIC-MCC integration.** Couple `tier3_picmcc/mcc_module.py` to the Boris pusher at modules M07/M08. Re-run Cases A/B/C with spatial resolution. Answers the non-local transport question.
2. **DTPM Picard-loop integration of `get_rates_pinn()`.** Replace the Arrhenius block in the Stage 10 DTPM Picard loop with a batch call to `get_rates_pinn()`. Re-validate the fluorine profile against Mettler 2025. If the centre-to-edge drop changes by less than 2 pp, the Arrhenius baseline is adequate; otherwise Tier 2 becomes production.
3. **Tier 1 pressure extension.** Repeat the BOLSIG⁺ batch at 5 and 20 mTorr. Retrain the Tier 2 surrogate with pressure as an input dimension.
4. **Secondary electrons in Tier 3 MCC.** Add Opal-Peterson-Beaty energy sharing to the ionisation channel in `mcc_module.py`. Required for self-consistent spatial PIC-MCC, not for the 0D cross-check.

---

## License

MIT. See `LICENSE`.

## Citation

If you use this work, cite it as:

> Ngan, Z. *Phase 2 Electron Kinetics Pipeline: replacing the Maxwellian assumption in the SF₆ ICP reduced-order model.* Technical report, Illinois Plasma Institute, 2026.
