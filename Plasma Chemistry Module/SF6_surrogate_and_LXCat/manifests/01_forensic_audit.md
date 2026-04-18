# REPO_FORENSIC_AUDIT

Content-first audit of `/Users/kaingan8/Downloads/SF6_unified/`.
Decisions are based on actual file contents (code modules, outputs/checkpoints, figures, README and report/slides/docs narrative), **not** folder names. **No files have been moved, renamed, merged, or deleted in this pass.**

---

## How evidence was gathered

- `ls` of every candidate root, `src/`, `tier3_pinn/`, `outputs/`, `docs/`, `figures/`, `results/`, `examples/`, `scripts/` where applicable.
- `diff -rq` across all suspected siblings inside each family (ignoring `.DS_Store` and `__pycache__`).
- `wc -l` on README, project_status, future_work, technical_report.tex, tier3_presentation.tex to gauge narrative maturity.
- Direct reads of README and project_status headers to confirm which milestone/round each copy describes.

---

## Family A — SF6_tier3 (9 candidate roots)

**Candidates:** `SF6_tier3_repo/`, `SF6_tier3_repo 2/`, `SF6_tier3_repo 3/`, `SF6_tier3_repo 4/`, `SF6_tier3_repo 5/`, `SF6_tier3_repo 6/`, `SF6_tier3_repo 7/`, `SF6_tier3_repo 8/`.

### Evidence landscape

Comparing `tier3_pinn/` code modules and `outputs/` contents yields four distinct content tiers, not nine:

| Tier | Members | Distinctive content |
|---|---|---|
| **T0 (pre-figures)** | `SF6_tier3_repo/` | No `figures/`. Only `tier3_pinn` through m3c + multicase. Outputs: m1/m2/m3/m3b/m3c + multicase only. Docs: status/repo_map/reproducibility. |
| **T1 (figures added)** | `SF6_tier3_repo 2`, `3`, `4` | Adds `figures/` (report figures). Same tier3_pinn code and outputs as T0. |
| **T2 (v2 / rotation v1)** | `SF6_tier3_repo 5` | Adds `train_rotation.py`, `regenerate_teacher_v2.py`, `fluid_loop_mock.py`, `phaseB_teacher_dataset_multicase_v2.npz`, rotation holdouts A–E seed0 (+C seed1/2), `seed_variance_summary.json`, `validation_prep_*`, docs/`future_work_completed.md`, docs/`picmcc_comparison_interface.md`. |
| **T3 (v3+v4 / final)** | `SF6_tier3_repo 6`, `7`, `8` | Adds `regenerate_teacher_v3.py`, `regenerate_teacher_v4.py`, `train_rotation_prf.py`, `train_rotation_v3.py`, `train_rotation_v4.py`, rotation prf/v3/v4 holdouts (incl. cases F/G/H), `seed_variance_v2_summary.json`, `picmcc_comparison_A.{json,md}`, `picmcc_reference_TEST_FIXTURE.npz`. All three **share identical** `tier3_pinn/` and `outputs/` trees. |

### Distinguishing the three T3 copies (the real decision)

`diff -rq "SF6_tier3_repo 7" "SF6_tier3_repo 8"` (ignoring `.DS_Store` / `__pycache__`) shows only these files differ:

```
.claude/settings.local.json
README.md
docs/future_work_completed.md
docs/project_status.md
report/technical_report.{tex,pdf}
slides/tier3_presentation.{tex,pdf}
slides/speaker_notes.md
```

`diff -rq "SF6_tier3_repo 6" "SF6_tier3_repo 8"` (same filters) shows **zero differences** — they are byte-identical outside of `__pycache__`.

Narrative content confirms:

- `SF6_tier3_repo 7/docs/project_status.md` → `"Snapshot after the v4 roadmap round"` (per-round delta view; `project_status.md` 127 lines, `future_work_completed.md` 228 lines, `tier3_presentation.tex` 502 lines, `technical_report.tex` 1174 lines).
- `SF6_tier3_repo 8/docs/project_status.md` → `"(final synchronized state) … agrees numerically and in wording with report … slides"` (77-line status, 158-line future_work, 385-line slides, 1232-line report — *polished, not per-round*).
- `SF6_tier3_repo 6` matches `8` byte-for-byte but carries six `__pycache__` dirs.

So "`7` before `8`" is genuinely a newer polish pass that **replaced** a per-round narrative with a final synchronized narrative (shorter status, richer report).

### Signs of newer scientific state in `SF6_tier3_repo 8`
- `README.md` 565 lines (vs 557 in `7`).
- Final-synchronized status doc explicitly cross-references the `.tex` files.
- Technical report grew from 1174 → 1232 lines; slides grew from 385 → 502 in `7` then *were rewritten* in `8` (385 lines again, but replacing the v4-only framing with the full-story framing — confirmed by status header change).

### Signs of outdated state
- `SF6_tier3_repo` (no suffix) lacks `figures/` and any of the v2+ milestones — strictly superseded.
- `SF6_tier3_repo 2..4` are interchangeable T1 snapshots.
- `SF6_tier3_repo 5` is a T2 snapshot, superseded by T3.
- `SF6_tier3_repo 6` carries `__pycache__` residue, otherwise identical to `8`.
- `SF6_tier3_repo 7` carries the per-round (v4) narrative explicitly superseded by `8`'s final narrative.

### Canonical decision
**`SF6_tier3_repo 8/`** — final synchronized docs on top of the full v4 code/output set.
**Confidence: HIGH.** Merge risk: LOW (no unique code/outputs exist outside `8`; only doc variants).
**Selective merge needed:** YES — `SF6_tier3_repo 7`'s per-round narrative (`project_status.md`, `future_work_completed.md`, `speaker_notes.md`, larger report/slides `.tex`) is *unique history* not present in `8`. Preserve as `archive/SF6_tier3/v4_round_docs/` (verbatim copy of those 6 files) so the "what was added in each round" story is not lost.

### Per-root call
| Root | What it is | Canonical? | Unique assets? | Redundant? | Later status |
|---|---|---|---|---|---|
| `SF6_tier3_repo 8` | T3 final synchronized | **Yes** | — | no | KEEP_ACTIVE |
| `SF6_tier3_repo 7` | T3 v4-round docs variant | no | **Yes — round-by-round docs** | code/outputs are redundant | MERGE_SELECTIVELY (docs only), else ARCHIVE_VERSIONED |
| `SF6_tier3_repo 6` | T3 byte-identical to 8 + pycache | no | no | yes | DELETE_SAFE (after stripping pycache would equal `8`) |
| `SF6_tier3_repo 5` | T2 intermediate (rotation v1 only) | no | not vs `8` | yes, but is a real milestone | ARCHIVE_VERSIONED |
| `SF6_tier3_repo 4` | T1 | no | no | yes | ARCHIVE_VERSIONED (compress) |
| `SF6_tier3_repo 3` | T1 | no | no | yes | ARCHIVE_VERSIONED (compress) |
| `SF6_tier3_repo 2` | T1 | no | no | yes | ARCHIVE_VERSIONED (compress) |
| `SF6_tier3_repo` | T0 | no | no | yes | ARCHIVE_VERSIONED (compress) |

Naming vs content: **naming was partially misleading** — `6` looks intermediate but is in fact a polluted copy of `8`'s final state. `7` looks like a predecessor of `8` and that is correct at the narrative level, but code/outputs are identical.

---

## Family B — plasma-dtpm (3 candidate roots)

**Candidates:** top-level `plasma-dtpm/`, `M1to5/plasma-dtpm/`, `m6b/plasma-dtpm/`.

### Evidence landscape
- `m6b/plasma-dtpm/src/` = `{m1_local_analysis, m2_nonlocal_model, m3_boundary_tests, m4_power_scaling, m5_surrogate, m6_pinn}`. Has `figures/m6`, `results/m6_pinn`, `examples/m6_smoke_test.py`, `scripts/run_m6.sh`, `Makefile`, `data/`, `pyproject.toml`.
- Top-level `plasma-dtpm/src/` = `{m1_local_analysis, m2_nonlocal_model, m3_boundary_tests, m4_power_scaling, m5_surrogate}` — **identical refactored module layout as `m6b` minus `m6_pinn`**. Has `figures/{Te_scaling, eedf, energy_balance, m5, rate_ratios}`, `results/{m1..m5}`, Makefile, data, pyproject. README explicitly says *"Track A only … M1–M5"*.
- `M1to5/plasma-dtpm/src/` = `{m1, m2, m3, m4, m5_surrogate}` — **older flat naming scheme**. Figures are single-file PDFs (`m1_boltz_vs_maxwell.pdf` etc.) instead of per-phenomenon subdirs. No `data/`, no `pyproject.toml`, no `Makefile`. README framing is "Plasma Digital Twin Project: SF₆ kinetics investigation" (a whole-project framing, not yet Track A/B split).

### Content comparison — top-level `plasma-dtpm` vs `m6b/plasma-dtpm`

`diff -rq` (filters out `.DS_Store`) shows the top-level version differs from `m6b` only in:
- `README.md` (Track A-only vs M1–M6 framing),
- `docs/report.{tex,pdf}` and `docs/slides.{tex,pdf}` (Track-A-only deliverables),
- **missing** `examples/m6_smoke_test.py`, `figures/m6`, `results/m6_pinn`, `scripts/run_m6.sh`, `src/m6_pinn`.

### Signs of newer scientific state
- `m6b/plasma-dtpm` adds an entire sixth milestone (PINN-based non-local correction), including `src/m6_pinn/`, `results/m6_pinn/`, figures/m6, Makefile target `run_m6.sh`, and smoke test — a clear forward step beyond M5.

### Signs of outdated state
- `M1to5/plasma-dtpm` uses a pre-refactor src layout (`m1/`, `m2/`, …) that does not appear in either newer copy; its figures are un-organised single files. It predates the refactor to `m{N}_descriptive_name/` subpackages and predates the addition of `pyproject.toml`, `Makefile`, and `data/`.
- Top-level `plasma-dtpm` carries a Track-A-only report/slides that were *superseded* by the M1–M6 report in `m6b`. It has no unique code vs `m6b`.

### Canonical decision
**`m6b/plasma-dtpm/`** — strict superset of top-level `plasma-dtpm`, plus all M6 content.
**Confidence: HIGH.** Merge risk: LOW.
**Selective merge needed:** POSSIBLY — the Track-A-only `docs/report.tex` and `docs/slides.tex` in top-level `plasma-dtpm/` are a distinct deliverable variant (M1–M5 scope). If the Track-A story must be preserved as its own artifact, copy those four doc files to `archive/plasma-dtpm/track_A_only_docs/`. Otherwise they are redundant.

### Per-root call
| Root | What it is | Canonical? | Unique assets? | Redundant? | Later status |
|---|---|---|---|---|---|
| `m6b/plasma-dtpm` | M1–M6 full project | **Yes** | — | no | KEEP_ACTIVE |
| top-level `plasma-dtpm` | Refactored M1–M5 with Track-A docs | no | **Yes — Track-A-only report/slides** | code is redundant | MERGE_SELECTIVELY (docs only) → ARCHIVE_VERSIONED |
| `M1to5/plasma-dtpm` | Pre-refactor M1–M5 | no | **Yes — pre-refactor src layout + per-milestone single-file figures** | superseded structurally, but is the only record of the earlier layout | ARCHIVE_VERSIONED |

Naming vs content: **naming was moderately misleading** — the *top-level* `plasma-dtpm/` looks like it should be the canonical root but is actually a milestone-scoped (Track A) sibling, not the latest. `M1to5/` is correctly labelled.

---

## Family C — phase2_electron_kinetics (2 candidate roots)

**Candidates:** `phase2_electron_kinetics/`, `phase2_electron_kinetics 2/`.

### Evidence landscape
`diff -rq` ignoring `.DS_Store` produces **zero** differences. The two trees are byte-identical.

### Canonical decision
**`phase2_electron_kinetics/`** (original, unsuffixed name).
**Confidence: HIGH.** Merge risk: LOW. Selective merge needed: NO.

### Per-root call
| Root | What it is | Canonical? | Unique? | Redundant? | Later status |
|---|---|---|---|---|---|
| `phase2_electron_kinetics` | canonical | **Yes** | — | no | KEEP_ACTIVE |
| `phase2_electron_kinetics 2` | Finder-style duplicate | no | no | **yes (identical)** | DELETE_SAFE |

---

## Family D — TEL

**Candidates:** `TEL/TEL_Simulation_Full_Package/` (source), `TEL/TEL_Simulation_Full_Package.zip` (bundle), `TEL/TEL_Report.pdf`, `TEL/TEL_Slides.pdf`, `TEL/Coupling_Response_to_Professor.md`.

### Evidence landscape
- `TEL/TEL_Simulation_Full_Package/` has `configs/{default.py, model_A.yaml, model_B.yaml, model_C.yaml, model_D.yaml}`, `data/{lallement, mettler}`, `results/{action_items_status.md, figures, raw_outputs}`, plus `run.py`, `run_models.py`, `animations/`, `docs/`, `presentation/`. Full working project.
- `TEL/Coupling_Response_to_Professor.md` is a response to supervisor feedback explicitly describing the 0D↔2D fixed-point coupling added to a later version of the report — a genuinely unique artifact not present in the source tree's own docs.
- The `.zip`, `TEL_Report.pdf`, `TEL_Slides.pdf` are rendered deliverables sitting next to the source tree (not inside it).

### Canonical decision
**`TEL/TEL_Simulation_Full_Package/`** as the canonical source; the zip and PDFs are packaged deliverables; `Coupling_Response_to_Professor.md` is a deliverable memo.
**Confidence: HIGH.** Merge risk: LOW. Selective merge: no (but the memo must be preserved; it contains unique mathematical exposition).

### Per-root call
| Artifact | What it is | Canonical? | Later status |
|---|---|---|---|
| `TEL/TEL_Simulation_Full_Package/` | Source tree (run.py, configs, data, results) | **Yes** | KEEP_ACTIVE |
| `TEL/TEL_Simulation_Full_Package.zip` | Distribution bundle | no | PACKAGED_DELIVERABLE |
| `TEL/TEL_Report.pdf` | Rendered report | no | PACKAGED_DELIVERABLE |
| `TEL/TEL_Slides.pdf` | Rendered slides | no | PACKAGED_DELIVERABLE |
| `TEL/Coupling_Response_to_Professor.md` | Unique supervisor-response memo | no (not code) | PACKAGED_DELIVERABLE — preserve; unique narrative |

No external/reference project is present in any family.

---

## Misleading-naming summary

1. **`SF6_tier3_repo 6` vs `8`** — looks like `6 < 8`, but contents are byte-identical; `6` is simply `8` + stale `__pycache__`.
2. **`SF6_tier3_repo 7` vs `8`** — here naming happens to match reality (`7 < 8`), but the *only* thing that actually progressed is docs; all code/outputs are identical. Content, not the number, decides what to keep.
3. **Top-level `plasma-dtpm/`** — appears to be the "main" project by virtue of being at the root, but is actually a Track-A-only milestone-scoped sibling, **not** the canonical version.

## Folders that contain unique material and require selective merge later

- **`SF6_tier3_repo 7/`** — unique per-round docs (`docs/project_status.md`, `docs/future_work_completed.md`, `slides/speaker_notes.md`, `report/technical_report.tex+pdf`, `slides/tier3_presentation.tex+pdf`). Code/outputs are redundant with `8`.
- **Top-level `plasma-dtpm/`** — unique Track-A-only docs (`README.md`, `docs/report.tex+pdf`, `docs/slides.tex+pdf`). Code is redundant with `m6b/plasma-dtpm/`.

## Folders safe to archive later (no unique content loss)

- `SF6_tier3_repo`, `SF6_tier3_repo 2`, `SF6_tier3_repo 3`, `SF6_tier3_repo 4` — superseded T0/T1 snapshots.
- `SF6_tier3_repo 5` — T2 milestone, genuinely a stepping stone but no unique assets beyond what `8` contains (`8` strictly supersets T2 in both code and outputs).
- `M1to5/plasma-dtpm/` — unique pre-refactor layout; archive as-is.

## Folders still uncertain after content review

- **None.** All prior uncertainties (`SF6_tier3_repo 7` vs `8`, top-level `plasma-dtpm` vs `m6b/plasma-dtpm`) are now resolved with HIGH confidence. The only remaining human judgment is **policy**: whether to keep the unique older *narrative* documents in `SF6_tier3_repo 7` and top-level `plasma-dtpm/` (recommended, via selective-merge archive), or drop them to save space.
