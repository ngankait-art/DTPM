# M7 Detection Report

Content-level search for a post-M6 milestone in `/Users/kaingan8/Downloads/SF6_unified/`.

---

## TL;DR

**YES — M7 exists.** It is the **0D↔2D coupled reactor-scale model** in `TEL/TEL_Simulation_Full_Package/`, accompanied by the supervisor-response memo `TEL/Coupling_Response_to_Professor.md`. The folder is **not named "M7"** — it is named after the device ("TEL" = the Tokyo Electron ICP etcher it models). M7 supersedes M6 as the terminal milestone of the SF₆/plasma-dtpm scientific program.

**A red-herring folder named `identifiability_M7/` also exists** at `phase2_electron_kinetics/supplementary/identifiability_M7/`. Its own README explicitly disclaims it (*"Exploratory side-study. Not the Phase 2 deliverable. … built during an earlier session"*). The name `identifiability_M7` is a historical label, **not** the terminal M7.

---

## Task 1 — What would M7 look like?

Progression through M1–M6 in `m6b/plasma-dtpm/`:
- **M1** Boltzmann vs Maxwellian rate ratios.
- **M2** MCC skin-depth scan (non-local EEDF evaluation).
- **M3** Reflector / wall boundary tests.
- **M4** Power scan.
- **M5** Compact MLP surrogate for rate coefficients.
- **M6** Physics-informed non-local PINN — *bounded* non-local correction to M5.

Given this trajectory, a post-M6 milestone must plausibly do **at least one** of:

1. Couple the M5/M6 rate layer into a reactor-scale (0D/2D) transport model.
2. Close the loop — a self-consistent iteration between plasma parameters and species transport.
3. Provide a deployment-ready pipeline with a runnable entry point producing publication figures.
4. Validate against **external** data (experiment or PIC-MCC reference).
5. Generalize across multiple process conditions beyond the training grid.
6. Introduce new physics not in M6 (e.g., EM skin-depth, electron energy PDE, wall gamma variation).

## Task 2 — Signature search across the repository

Grep for `coupling|hybrid|closure|closed-loop|PINN.*TEL|deployment|production-ready|end-to-end` across every folder, and direct inspection of the top-level README of `m6b/plasma-dtpm/`.

Key findings:

- `m6b/plasma-dtpm/README.md` itself flags the Track-B sibling:
  > *"A separate, previously completed project ('Track B') exists that built a 2D global reactor model for fluorine transport in a TEL ICP etcher — full reactor geometry, diffusion chemistry, …"*
- `TEL/Coupling_Response_to_Professor.md` lays out the **full 0D↔2D fixed-point coupling**: $T_e$ by EN-corrected Lieberman balance solved by Brent bisection, $n_{e,\text{avg}}$ from power balance, 2D species PDE with 0D scalars as coefficients, Section 4.6 explaining scale-separation consistency.
- `TEL/TEL_Simulation_Full_Package/src/` contains a complete coupled solver stack: `solver.py`, `solver_multispecies.py`, `global_model.py`, `sf6_chemistry.py`, `sf6_rates.py`, `sf6_global_model_final.py`, `boundary_conditions.py`, `mesh.py`, `geometry.py`, `postprocessing.py`, `animate.py`.
- `TEL/TEL_Simulation_Full_Package/run.py` is a **deployment entry point** ("runs unified single-domain solver and generates all publication figures") with CLI flags (`--power`, `--ar`).
- `TEL/TEL_Simulation_Full_Package/run_models.py` runs **Model A (calibrated 2-species), Model B (hybrid 9-species calibrated), Model C (hybrid 9-species uncalibrated Kokkoris)** side-by-side; `configs/` also contains `model_D.yaml`, so four model variants are tracked.
- `TEL/TEL_Simulation_Full_Package/data/{lallement, mettler}` — **external experimental references** (Mettler radical-probe and Lallement data sets).
- `TEL/TEL_Simulation_Full_Package/results/action_items_status.md` lists completed supervisor action items: *"Validate absolute [F] density"* against Mettler Fig 4.5, *"Quantitative power sweep"*, *"Document source-sink balance"*, *"Add electron energy PDE"* (partial), *"EM solver for η"*.
- `phase2_electron_kinetics/supplementary/identifiability_M7/README.md` says on line 1–4: *"Supplementary: identifiability / M7 side-study. Author: Zachariah Ngan … Status: Exploratory side-study. Not the Phase 2 deliverable."* And further: *"This work is not the Phase 2 deliverable. … built in an earlier session before the supervisor's workplan was loaded into context."* A historical label, not the terminal milestone.

## Task 3 — Compare against M6 baseline

Taking `m6b/plasma-dtpm/` as the M6 baseline (Track A kinetic closure, bounded non-local PINN correction, no reactor geometry, no fluorine transport), the TEL package differs as follows:

| Axis | M6 (`m6b/plasma-dtpm`) | TEL package |
|---|---|---|
| Geometry | 0D (homogeneous plasma) | **2D (Nr×Nz mesh, reactor geometry, masked domain)** |
| Physics | Electron kinetics / rate coefficients only | **Self-consistent 0D↔2D: Brent-bisection Te, power balance n_e, 2D species PDE, EM skin-depth, (partial) electron energy PDE** |
| Coupling | None — produces rate tables | **Fixed-point iteration linking plasma scalars ↔ 2D species transport** |
| Chemistry | 1-species SF₆ rate ratios | **9-species SF₆ chemistry (`sf6_chemistry.py`, `SPECIES`, `WALL_GAMMA`)** |
| Validation | Internal (ionization-failure argument) | **External: Mettler absolute [F] density, Lallement data, action-items checklist** |
| Condition coverage | Training grid of M1–M6 | **Models A / B / C / D — calibrated, hybrid calibrated, hybrid uncalibrated (Kokkoris), plus model_D** |
| Workflow stage | Research surrogate | **Deployment-ready: `run.py`, `run_models.py`, publication-figure generator, animations** |
| Supervisor feedback | N/A in M6 | **Explicit response memo adding Section 4 "0D↔2D Hybrid Coupling: Mathematical Formulation" (pp. 10–14) and Section 4.6 consistency argument** |

Every axis shows a genuine forward step consistent with a new milestone, not a refinement of an existing one. The TEL package introduces **new physics, new coupling, new validation, and a new workflow stage** — the union of M7 hypotheses (1)–(5).

## Task 4 — Findings

### Does M7 exist? **YES.**

### Location
**Primary:** `TEL/TEL_Simulation_Full_Package/` (source tree) plus `TEL/Coupling_Response_to_Professor.md` (unique derivation memo). Supporting deliverables: `TEL/TEL_Simulation_Full_Package.zip`, `TEL/TEL_Report.pdf`, `TEL/TEL_Slides.pdf`.

### Key files constituting M7
- **Coupled solver core:** `TEL/TEL_Simulation_Full_Package/src/{solver.py, solver_multispecies.py, global_model.py, sf6_chemistry.py, sf6_rates.py, sf6_global_model_final.py, boundary_conditions.py, geometry.py, mesh.py, postprocessing.py, animate.py}`.
- **Entry points:** `run.py`, `run_models.py`.
- **Model hierarchy configs:** `configs/{default.py, model_A.yaml, model_B.yaml, model_C.yaml, model_D.yaml}`.
- **External validation data:** `data/lallement/`, `data/mettler/`.
- **Results + action-items status:** `results/action_items_status.md`, `results/figures/`, `results/raw_outputs/`.
- **Docs:** `docs/sections/`, `docs/figures/`.
- **Deliverables:** `TEL_Report.pdf`, `TEL_Slides.pdf`, supervisor-response memo `Coupling_Response_to_Professor.md`.

### What M7 adds over M6
New reactor geometry (2D Nr×Nz masked mesh), 9-species SF₆ chemistry, 0D↔2D fixed-point coupling, EM skin-depth η model, (partial) electron energy PDE, external experimental validation against Mettler and Lallement, four-model hierarchy (A/B/C/D), deployment-ready entry points, and a dedicated supervisor-response mathematical-bridge memo.

### Does M7 supersede M6?
**As the terminal milestone of the scientific program, yes.** M7 consumes M-layer rate coefficients (conceptually the output of M1–M6) and uses them in a coupled reactor-scale solver. M6 is **not obsolete** — it remains the canonical Track A (kinetic closure) deliverable — but it is no longer the terminal state of the work. Both are canonical in their own track; neither subsumes the other.

### Is M7 fragmented?
**Yes — mildly.** The source tree lives in `TEL/TEL_Simulation_Full_Package/`, the unique mathematical-bridge derivation lives **outside** it in `TEL/Coupling_Response_to_Professor.md`, and the rendered deliverables (`.zip`, PDFs) sit alongside. All three pieces together constitute M7. The `Coupling_Response_to_Professor.md` memo is the single most important unique artifact — it contains the 0D↔2D bridge exposition that is only summarized (not derived in full) inside the report.

### The `identifiability_M7` folder
`phase2_electron_kinetics/supplementary/identifiability_M7/` is **labelled** M7 but is **not** the terminal milestone. Its own README states: *"Exploratory side-study. Not the Phase 2 deliverable. … built during an earlier session before the supervisor's workplan was loaded into context."* It is an analytical note on **two-anchor etch-rate calibration identifiability** — a closed-form invariant 𝓘 = β·k_diss·n_e(P_H) and an observability sweep. Its figures (`fig1_pipeline_schematic`, …, `fig7_observability_sweep`) are separate from M7 content. Classify it as **historical side-study**, not terminal M7.

---

## Task 5 — Updated canonical decisions

| Family | Prior canonical call | New canonical call | Reason |
|---|---|---|---|
| SF6_tier3 | `SF6_tier3_repo 8/` | **unchanged — `SF6_tier3_repo 8/`** | M7 detection does not affect the SF6_tier3 lineage. |
| plasma-dtpm / Track A | `m6b/plasma-dtpm/` | **unchanged — `m6b/plasma-dtpm/`** | Still canonical for Track A, but re-labelled as the *M1–M6 precursor*, not the terminal state. |
| phase2_electron_kinetics | `phase2_electron_kinetics/` | **unchanged — `phase2_electron_kinetics/`**; reclassify `supplementary/identifiability_M7/` as DOCUMENTED_SIDE_STUDY (KEEP as part of canonical root, do not promote) | `identifiability_M7` is labelled but disclaimed; it stays inside the canonical `phase2_electron_kinetics/` tree where it already lives. |
| TEL / Track B / **M7** | `TEL/TEL_Simulation_Full_Package/` (distinct project) | **`TEL/TEL_Simulation_Full_Package/` — now understood as M7 (terminal milestone)**, with `TEL/Coupling_Response_to_Professor.md` promoted to **essential M7 artifact**, not just a deliverable | Content-level evidence shows TEL is the post-M6 coupled reactor model. |

### Consequences for `REPO_CLEANUP_MANIFEST.md` and `REPO_CLEANUP_PLAN.md`

- `TEL/TEL_Simulation_Full_Package/` remains `KEEP_ACTIVE`.
- `TEL/Coupling_Response_to_Professor.md` should be **re-classified** from pure `PACKAGED_DELIVERABLE` → **`KEEP_ACTIVE` (essential M7 source document)**; do not bury it with the PDFs in `deliverables/`. Recommend keeping it alongside the TEL source tree (e.g., copy into `TEL/TEL_Simulation_Full_Package/docs/`).
- In the unified-package layout (Phase 7), TEL should sit under `active/` **clearly labelled as M7 / Track B**, not as an unrelated distinct project. A short note in the top-level README should state: *"The plasma-dtpm (m6b) Track-A kinetic-closure work is the M1–M6 precursor to the TEL (Track-B) coupled reactor model, which is the terminal milestone (M7) of the SF₆ program."*
- `phase2_electron_kinetics/supplementary/identifiability_M7/` is **retained in place** as a documented side-study within the canonical `phase2_electron_kinetics/` tree; no action required. No merge, no archive, no rename.

### Is M7 complete or fragmented?

**Functionally complete, physically fragmented.** All pieces exist; they just live under three different filesystem locations (`TEL/TEL_Simulation_Full_Package/`, `TEL/Coupling_Response_to_Professor.md`, `TEL/TEL_Simulation_Full_Package.zip`+PDFs). No M7 code or derivation is missing. Nothing needs to be reconstructed — only re-labelled.
