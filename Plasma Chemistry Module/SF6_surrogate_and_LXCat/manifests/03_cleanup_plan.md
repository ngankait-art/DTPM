# REPO_CLEANUP_PLAN

Staged cleanup plan for `/Users/kaingan8/Downloads/SF6_unified/`, grounded in the content-level audit.
Pairs with [REPO_FORENSIC_AUDIT.md](REPO_FORENSIC_AUDIT.md) and [REPO_CLEANUP_MANIFEST.md](REPO_CLEANUP_MANIFEST.md). **No file operations have occurred yet.**

Canonical active roots (fixed by content audit — do not re-debate during execution):
- SF6_tier3 → `SF6_tier3_repo 8/`
- plasma-dtpm / Track A (M1–M6) → `m6b/plasma-dtpm/`
- phase2_electron_kinetics → `phase2_electron_kinetics/`
- TEL / Track B / **M7 (terminal milestone)** → `TEL/TEL_Simulation_Full_Package/` + `TEL/Coupling_Response_to_Professor.md` (see [M7_DETECTION_REPORT.md](M7_DETECTION_REPORT.md))

**M7 note:** `TEL/Coupling_Response_to_Professor.md` is reclassified from `PACKAGED_DELIVERABLE` to **essential M7 source** — Phase 5 must **not** move it into `deliverables/`. Instead, Phase 5 should copy it into `TEL/TEL_Simulation_Full_Package/docs/` (or leave it adjacent) so the 0D↔2D derivation travels with the M7 source tree. `phase2_electron_kinetics/supplementary/identifiability_M7/` is a labelled-but-disclaimed side-study — leave in place, do not rename or archive.

---

## Phase 1 — Safe junk removal
- **Acts on manifest status:** `DELETE_SAFE` (junk subset only).
- **What:** Sweep the whole tree and delete every `__pycache__/` directory, every `*.pyc`, and every `.DS_Store`. Concretely this removes six `__pycache__` dirs under `SF6_tier3_repo 6/`.
- **Why:** Regeneratable, untracked noise. Removing it makes the Phase 2 duplicate check and the Phase 3 archival bundles clean.
- **Do NOT:** touch any `.py`, config, dataset, checkpoint, figure, or `.git/`/`.claude/` file. Do **not** yet delete `SF6_tier3_repo 6/`, `phase2_electron_kinetics 2/`, or anything else flagged DELETE_SAFE for duplication reasons — those are Phase 4.

## Phase 2 — Extract unique narrative artifacts (selective merge)
- **Acts on manifest status:** `MERGE_SELECTIVELY` (rows in the manifest that flag unique doc variants).
- **What:** Create `archive/SF6_tier3/v4_round_docs/` and copy **from `SF6_tier3_repo 7/`**:
  - `README.md`
  - `docs/project_status.md`, `docs/future_work_completed.md`
  - `report/technical_report.tex`, `report/technical_report.pdf`
  - `slides/tier3_presentation.tex`, `slides/tier3_presentation.pdf`, `slides/speaker_notes.md`
  Create `archive/plasma-dtpm/track_A_only_docs/` and copy **from top-level `plasma-dtpm/`**:
  - `README.md`
  - `docs/report.tex`, `docs/report.pdf`
  - `docs/slides.tex`, `docs/slides.pdf`
- **Why:** These two folders are the **only** containers holding unique narrative that will not exist anywhere else after cleanup. Everything else inside them is redundant with the canonical root.
- **Do NOT:** copy entire trees — that re-introduces the redundancy we're trying to remove. Do **not** modify the files during copy. Do **not** yet delete the source folders — that happens in Phase 3/4 after the archive is in place.

## Phase 3 — Archive older snapshots (no unique content loss)
- **Acts on manifest status:** `ARCHIVE_VERSIONED`.
- **What:** Create `archive/SF6_tier3/` and move:
  - `SF6_tier3_repo` → `archive/SF6_tier3/T0_pre_figures/` (and tar)
  - `SF6_tier3_repo 2..4` → `archive/SF6_tier3/T1_figures_added_{2,3,4}/` (and tar)
  - `SF6_tier3_repo 5` → `archive/SF6_tier3/T2_rotation_v1/` (tar)
  - `SF6_tier3_repo 7` → `archive/SF6_tier3/T3_v4_round/` (tar; its unique docs are already in Phase 2's archive so this is belt-and-braces)
  Create `archive/plasma-dtpm/` and move:
  - `M1to5/plasma-dtpm/` → `archive/plasma-dtpm/pre_refactor_M1to5/` (tar)
- **Why:** Preserves lineage in a single dedicated location and compresses it, making the active workspace navigable without destroying provenance.
- **Do NOT:** archive `SF6_tier3_repo 6/` here — it carries no unique content and goes to Phase 4 (delete). Do **not** archive top-level `plasma-dtpm/` as-is — the Phase 2 doc archive is the only part worth keeping; the rest is redundant. Do **not** re-tar `SF6_tier3_repo 8`, `m6b/`, `phase2_electron_kinetics`, or `TEL/` — those are canonical/active.

## Phase 4 — Remove confirmed redundant trees
- **Acts on manifest status:** `DELETE_SAFE` (duplicate-copy subset).
- **What:** Delete:
  - `SF6_tier3_repo 6/` (byte-identical to `SF6_tier3_repo 8/` outside of Phase-1-cleared `__pycache__`).
  - `phase2_electron_kinetics 2/` (byte-identical to `phase2_electron_kinetics/`).
  - top-level `plasma-dtpm/` **only after** Phase 2 copied its 5 unique doc files into the archive.
- **Why:** These folders carry zero unique content once Phase 2 has harvested the doc variants.
- **Do NOT:** delete before Phase 2 completes and the archives are verified readable. Do **not** delete `SF6_tier3_repo 7/` here — it is moved to `archive/` in Phase 3, not deleted.

## Phase 5 — Separate packaged deliverables
- **Acts on manifest status:** `PACKAGED_DELIVERABLE`.
- **What:** Create `deliverables/` and move in, grouped by family:
  - `TEL/TEL_Simulation_Full_Package.zip`, `TEL/TEL_Report.pdf`, `TEL/TEL_Slides.pdf` → `deliverables/TEL/`
  - `TEL/Coupling_Response_to_Professor.md` → **copy** into `TEL/TEL_Simulation_Full_Package/docs/coupling_response.md` (essential M7 source, not a deliverable)
  - `m6b/plasma-dtpm.zip`, `m6b/report.pdf`, `m6b/slides.pdf` → `deliverables/plasma-dtpm/m6b/`
  - `M1to5/plasma-dtpm.zip`, `M1to5/report.pdf`, `M1to5/slides.pdf` → `deliverables/plasma-dtpm/M1to5/`
- **Why:** Separates rendered distribution artifacts from source trees and from the archive (which is historical source, not deliverables).
- **Do NOT:** unzip any archive, regenerate any PDF, or discard the Coupling response memo — it contains unique mathematical content (0D↔2D fixed-point coupling).

## Phase 6 — Selective merge: nothing further expected
- **Acts on manifest status:** `MERGE_SELECTIVELY` (follow-ups only).
- **What:** Verify, by re-reading the content audit, that no additional selective merges are required beyond the two executed in Phase 2. If during execution anyone discovers a previously-missed unique asset (e.g., a one-off script in an archived snapshot), port it into the canonical root with a one-line manifest note.
- **Why:** The audit currently shows only two selective-merge sources; this phase is a backstop.
- **Do NOT:** bulk-rsync any archived snapshot → canonical root; that would overwrite newer canonical content with stale versions. Only ever port named files.

## Phase 7 — Create unified academic package
- **Acts on manifest status:** `KEEP_ACTIVE` (layout reorganization).
- **What:** Restructure the repo top level into:
  ```
  SF6_unified/
    active/
      SF6_tier3/               # from SF6_tier3_repo 8
      plasma-dtpm/             # from m6b/plasma-dtpm  -- Track A, M1–M6 precursor
      phase2_electron_kinetics/  # retains supplementary/identifiability_M7/ side-study in place
      TEL_M7/                  # from TEL_Simulation_Full_Package -- Track B, terminal milestone M7
    archive/                   # Phases 2 + 3
    deliverables/              # Phase 5
  ```
  The top-level README must state: *"The plasma-dtpm (m6b) Track-A kinetic-closure work is the M1–M6 precursor to the TEL (Track-B) coupled reactor model, which is the terminal milestone (M7) of the SF₆ program."*
  Fix any cross-project references encountered during the move.
- **Why:** Single predictable entry point per project; clean separation of active vs historical vs rendered.
- **Do NOT:** start this before Phases 1–5 are complete — reorganizing mid-cleanup scrambles diffs and invalidates path-based checks used in earlier phases.

## Phase 8 — Final documentation cleanup
- **Acts on manifest status:** `KEEP_ACTIVE` (docs); housekeeping.
- **What:** Write a top-level `README.md` summarizing each active project, pointing to `deliverables/` and noting archive provenance (which snapshot each archived folder came from). Decide whether to keep `REPO_FORENSIC_AUDIT.md`, `REPO_CLEANUP_MANIFEST.md`, and `REPO_CLEANUP_PLAN.md` at root or move them under `docs/`.
- **Why:** Makes the cleaned layout self-documenting for later reviewers and collaborators.
- **Do NOT:** rewrite per-project READMEs — each canonical project already has its own documentation, and editing them would churn the canonical trees unnecessarily.
