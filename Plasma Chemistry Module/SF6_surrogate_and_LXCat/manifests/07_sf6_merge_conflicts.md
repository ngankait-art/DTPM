# MERGE_CONFLICTS

No unresolved merge conflicts.

## Why there are none

The verification step (documented in [MERGE_DECISIONS.md](MERGE_DECISIONS.md)) established that `SF6_tier3_repo 8` is a strict content superset of `SF6_tier3_repo 5`: every file that exists in repo 5 either also exists bit-identically in repo 8 or exists in an evolved (longer / later / regenerated) form in repo 8. The only "only in repo 5" file was `.DS_Store` (macOS junk). There was therefore no file to merge forward from repo 5 that was not already present or superseded in repo 8.

## Dual-version preservations

None required. No file was preserved in two versions inside the merged repo. If a future pass uncovers a specific repo-5 artifact that the current canonical state depends on, it can be sourced from `SF6_tier3_repo 5/` (retained untouched at the workspace root) and added here with a follow-up entry.

## Items that could become conflicts in a later pass (flagged, not yet acted upon)

- **`SF6_tier3_repo 7/`** holds the "v4 roadmap round" narrative in `docs/project_status.md`, `docs/future_work_completed.md`, `slides/speaker_notes.md`, and a variant `report/technical_report.tex`. Those texts describe the state of the project one round before the "final synchronized state" that the merged repo ships. They are **not** conflicts — the merged repo deliberately ships the final-synchronized narrative — but if a reader later asks for the per-round write-up, repo 7's docs are the only source. The forensic-audit plan already routes them into `archive/SF6_tier3/v4_round_docs/` rather than into the merged repo.
- **`outputs/validation_prep_report.md`** is shorter in repo 8 (50 lines) than in repo 5 (86 lines). This is a regeneration difference, not a content loss: the shorter file is emitted by the newer `scripts/validation_prep.py` and is paired with the exhaustive `docs/project_status.md` status table in repo 8. If a future reviewer wants the longer prose form, it can be recovered from repo 5 without disturbing the merged state.
