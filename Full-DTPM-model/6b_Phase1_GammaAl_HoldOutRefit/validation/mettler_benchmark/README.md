# 5a — Phase-1 Mettler Validation (Working Directory)

**Purpose**: Staging area for all Mettler-validation artefacts that feed the
Phase-1 DTPM report (`Steps/5.Phase1_EM_Chemistry_Merged/docs/report/main.pdf`)
and the 06a consolidated SF6/Ar global-model report.

This directory was created because the Mettler benchmarking work cuts across
both report threads and needs a single home for (i) the reference PDF, (ii)
digitised data points from Mettler's key TEL figures, (iii) the ChatGPT/Claude-
web prompt used for visual cross-checking, and (iv) the consolidated
corrections notes.

## Contents

- `README.md` — this file.
- `CHATGPT_PROMPT.md` — prompt to paste into a Claude.ai web or ChatGPT project
  with both `Mettler Dissertation.pdf` and `main.pdf` uploaded.
- `METTLER_TEL_DATA_DIGITISED.md` — canonical digitised values from Mettler
  Fig 4.9, 4.14, 4.17 in a format usable by `generate_stage10_figures.py`.

## Reference files (not copied — symlinked or cited by path)

- Mettler PDF: `Literature/Mettler Dissertation.pdf`
- Phase-1 report: `Steps/5.Phase1_EM_Chemistry_Merged/docs/report/main.pdf`
- Phase-1 corrections: `Steps/5.Phase1_EM_Chemistry_Merged/docs/METTLER_CORRECTIONS.md`
- Phase-1 validation points (current + future): `Steps/5.Phase1_EM_Chemistry_Merged/docs/METTLER_VALIDATION_POINTS.md`
- 06a corrections: `06a_sf6_ar_global_model_report[Final]/METTLER_CORRECTIONS.md`
- Plan file (Phase-1 execution): `~/.claude/plans/mutable-swimming-chipmunk.md`

## Workflow

1. Upload `Mettler Dissertation.pdf` and the current `main.pdf` to a ChatGPT
   Project (or Claude.ai).
2. Paste `CHATGPT_PROMPT.md` contents.
3. Capture the AI response verbatim as `CHATGPT_FEEDBACK.md` (to be added
   manually after the web session completes).
4. Return to Claude Code and run the Phase-1 plan
   (`~/.claude/plans/mutable-swimming-chipmunk.md`) after merging any new
   insights from the web session into `METTLER_CORRECTIONS.md`.
