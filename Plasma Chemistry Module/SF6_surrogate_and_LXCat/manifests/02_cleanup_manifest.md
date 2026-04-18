# REPO_CLEANUP_MANIFEST

Classification grounded in the content-level forensic audit in [REPO_FORENSIC_AUDIT.md](REPO_FORENSIC_AUDIT.md) **and** the M7 detection pass in [M7_DETECTION_REPORT.md](M7_DETECTION_REPORT.md). **No files have been moved, renamed, merged, or deleted.**

> **M7 note.** Content analysis identified `TEL/TEL_Simulation_Full_Package/` (together with the supervisor-response memo `TEL/Coupling_Response_to_Professor.md`) as the terminal milestone (M7) of the SF₆/plasma-dtpm program — a 0D↔2D coupled reactor-scale model that consumes the M1–M6 rate layer. The folder named `phase2_electron_kinetics/supplementary/identifiability_M7/` is a **labelled but disclaimed** earlier side-study; it is **not** the terminal M7.

---

## Summary

- **Total families identified:** 4 (SF6_tier3, plasma-dtpm, phase2_electron_kinetics, TEL).
- **Canonical active per family (content-based):**
  - SF6_tier3 → `SF6_tier3_repo 8/`
  - plasma-dtpm / Track A (M1–M6) → `m6b/plasma-dtpm/`
  - phase2_electron_kinetics → `phase2_electron_kinetics/` (retains `supplementary/identifiability_M7/` as a documented side-study)
  - TEL / Track B / **M7 (terminal milestone)** → `TEL/TEL_Simulation_Full_Package/` + `TEL/Coupling_Response_to_Professor.md`
- **Folders requiring human judgment:** none block automation, but two folders hold **unique narrative docs** that should be preserved as selective-merge archives before their container is dropped:
  - `SF6_tier3_repo 7/` — v4-round docs not in `8` (per-round project_status, future_work, slides, report).
  - top-level `plasma-dtpm/` — Track-A-only docs not in `m6b/plasma-dtpm/` (README, report.tex/pdf, slides.tex/pdf).

---

## Per-folder classification (content-justified)

| Path | Family | Family type | Planned status | Recommended action | Rationale (content) |
|---|---|---|---|---|---|
| `SF6_tier3_repo 8/` | SF6_tier3 | VERSIONED_LINEAGE | **KEEP_ACTIVE** | Promote to canonical SF6 root | Final-synchronized docs (`project_status.md` header: *"final synchronized state"*), full T3 code (`regenerate_teacher_v{3,4}.py`, `train_rotation_{prf,v3,v4}.py`), full T3 outputs (picmcc comparison, v3/v4 rotation holdouts, seed_variance_v2), no pycache. README 565 lines, report 1232 lines. |
| `SF6_tier3_repo 7/` | SF6_tier3 | VERSIONED_LINEAGE | **MERGE_SELECTIVELY** → ARCHIVE_VERSIONED | Extract 6 unique doc files into `archive/SF6_tier3/v4_round_docs/`, archive the rest | `tier3_pinn/` code and `outputs/` are identical to `8`; only docs differ. Unique files: `README.md`, `docs/project_status.md` (*"Snapshot after the v4 roadmap round"*), `docs/future_work_completed.md`, `report/technical_report.{tex,pdf}`, `slides/tier3_presentation.{tex,pdf}`, `slides/speaker_notes.md`. |
| `SF6_tier3_repo 6/` | SF6_tier3 | VERSIONED_LINEAGE | **DELETE_SAFE** | Delete after confirming `diff -rq` against `8` (excluding `__pycache__`) is empty | Byte-identical to `8` outside of six stale `__pycache__` trees. No unique content whatsoever. |
| `SF6_tier3_repo 5/` | SF6_tier3 | VERSIONED_LINEAGE | ARCHIVE_VERSIONED | Archive as "T2 / rotation-v1 milestone" | Adds `train_rotation.py`, `regenerate_teacher_v2.py`, `fluid_loop_mock.py`, rotation holdouts A–E seed0 (+C seeds), `seed_variance_summary.json`, `validation_prep_*` — all strictly subsumed by `8`'s v2+v3+v4 supersets. No unique assets. |
| `SF6_tier3_repo 4/` | SF6_tier3 | VERSIONED_LINEAGE | ARCHIVE_VERSIONED | Archive (compress) | T1 snapshot: figures added, pre-rotation. Fully subsumed. |
| `SF6_tier3_repo 3/` | SF6_tier3 | VERSIONED_LINEAGE | ARCHIVE_VERSIONED | Archive (compress) | T1 — interchangeable with `2`, `4`. |
| `SF6_tier3_repo 2/` | SF6_tier3 | VERSIONED_LINEAGE | ARCHIVE_VERSIONED | Archive (compress) | T1 — interchangeable with `3`, `4`. |
| `SF6_tier3_repo/` | SF6_tier3 | VERSIONED_LINEAGE | ARCHIVE_VERSIONED | Archive (compress) | T0: no `figures/`, pre-rotation code, pre-v2 outputs. Oldest complete snapshot. |
| `m6b/plasma-dtpm/` | plasma-dtpm | VERSIONED_LINEAGE | **KEEP_ACTIVE** | Promote to canonical plasma-dtpm root | Strict superset of top-level `plasma-dtpm/`; adds `src/m6_pinn/`, `results/m6_pinn/`, `figures/m6/`, `examples/m6_smoke_test.py`, `scripts/run_m6.sh`, M1–M6 README. |
| `m6b/plasma-dtpm.zip` | plasma-dtpm | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/plasma-dtpm/m6b/` | M6 distribution bundle. |
| `m6b/report.pdf` | plasma-dtpm | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/plasma-dtpm/m6b/` | M6 rendered report. |
| `m6b/slides.pdf` | plasma-dtpm | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/plasma-dtpm/m6b/` | M6 rendered slides. |
| top-level `plasma-dtpm/` | plasma-dtpm | VERSIONED_LINEAGE | **MERGE_SELECTIVELY** → ARCHIVE_VERSIONED | Copy `README.md`, `docs/report.{tex,pdf}`, `docs/slides.{tex,pdf}` to `archive/plasma-dtpm/track_A_only_docs/`, then drop the rest | Same refactored `src/` layout as `m6b` but scope-limited to Track A / M1–M5. Unique narrative only; no unique code vs `m6b/plasma-dtpm`. |
| `M1to5/plasma-dtpm/` | plasma-dtpm | VERSIONED_LINEAGE | ARCHIVE_VERSIONED | Archive as "pre-refactor M1–M5 snapshot" | Older flat src layout (`src/m1/ m2/ m3/ m4/ m5_surrogate/`), single-file per-phenomenon figures, no `data/`, no `pyproject.toml`, no `Makefile`. Structurally superseded; preserve as lineage. |
| `M1to5/plasma-dtpm.zip` | plasma-dtpm | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/plasma-dtpm/M1to5/` | Pre-refactor distribution bundle. |
| `M1to5/report.pdf` | plasma-dtpm | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/plasma-dtpm/M1to5/` | Pre-refactor rendered report. |
| `M1to5/slides.pdf` | plasma-dtpm | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/plasma-dtpm/M1to5/` | Pre-refactor rendered slides. |
| `phase2_electron_kinetics/` | phase2 | DISTINCT_PROJECT | **KEEP_ACTIVE** | Keep as canonical phase2 root | Full project tree (tier1/tier2/tier3, supplementary, paper, report, slides). Original name. |
| `phase2_electron_kinetics 2/` | phase2 | DUPLICATE_COPY | **DELETE_SAFE** | Delete | `diff -rq` vs canonical is empty (ignoring `.DS_Store`). Zero unique content. |
| `TEL/TEL_Simulation_Full_Package/` | TEL / **M7 (Track B)** | DISTINCT_PROJECT — terminal milestone | **KEEP_ACTIVE** | Keep as canonical TEL/M7 root | **Terminal milestone (M7).** 0D↔2D coupled reactor: `src/{solver, solver_multispecies, global_model, sf6_chemistry, sf6_rates, sf6_global_model_final, boundary_conditions, mesh, geometry, postprocessing, animate}.py`; entry points `run.py`, `run_models.py`; four-model hierarchy via `configs/model_{A,B,C,D}.yaml`; external validation data `data/{lallement, mettler}`; `results/action_items_status.md` checklist. |
| `TEL/Coupling_Response_to_Professor.md` | TEL / **M7** | ESSENTIAL M7 SOURCE (reclassified) | **KEEP_ACTIVE** | Keep with TEL source (recommend copying into `TEL/TEL_Simulation_Full_Package/docs/`); do **not** bury in `deliverables/` | Unique 0D↔2D fixed-point coupling derivation (Brent-bisection Te, power-balance n_e, 2D species PDE, Section 4.6 consistency argument). Only place this derivation exists at this level of detail. **Reclassified from PACKAGED_DELIVERABLE to KEEP_ACTIVE based on M7 audit.** |
| `TEL/TEL_Simulation_Full_Package.zip` | TEL / M7 | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/TEL/` | Zipped export of the M7 source. |
| `TEL/TEL_Report.pdf` | TEL / M7 | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/TEL/` | Rendered M7 report. |
| `TEL/TEL_Slides.pdf` | TEL / M7 | PACKAGED_DELIVERABLE | PACKAGED_DELIVERABLE | Move to `deliverables/TEL/` | Rendered M7 slides. |
| `phase2_electron_kinetics/supplementary/identifiability_M7/` | phase2 side-study | DOCUMENTED_SIDE_STUDY | **KEEP_ACTIVE (in place)** | No action — retain inside canonical `phase2_electron_kinetics/` tree | Labelled "M7" but explicitly disclaimed by its own README as an earlier exploratory side-study on two-anchor etch-rate identifiability. Not the terminal M7; not redundant; no merge, archive, or rename required. |
| `__pycache__/` (6 dirs under `SF6_tier3_repo 6/`) + any stray `*.pyc` | — | JUNK_OR_CACHE | **DELETE_SAFE** | Remove in Phase 1 (whole-repo sweep) | Regeneratable Python bytecode. |
| `.DS_Store` (top-level and scattered) | — | JUNK_OR_CACHE | DELETE_SAFE | Remove in Phase 1 | macOS metadata, not content. |

No folder in this repository matches **EXTERNAL_REFERENCE** or **UNCERTAIN_REVIEW_NEEDED** after the content audit.
