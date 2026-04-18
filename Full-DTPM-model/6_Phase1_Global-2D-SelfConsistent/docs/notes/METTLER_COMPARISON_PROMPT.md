# ChatGPT / Claude Web — Mettler Comparison Prompt

This prompt is designed for a **new Claude.ai web or ChatGPT Project** where you can upload both PDFs side-by-side. The goal is a visual, figure-by-figure confirmation of the gaps and corrections already identified via text analysis in Claude Code.

## Files to upload

1. `Mettler Dissertation.pdf` (UIUC PhD, J.D. Mettler 2025, 105 pp)
   - Path in this workspace: `Literature/Mettler Dissertation.pdf`
2. `A Self-Consistent 0D–2D Hybrid Framework for ICP Etch Modelling.pdf` — our Phase-1 DTPM report (95 pp, 55 figures)
   - Path: `Steps/5.Phase1_EM_Chemistry_Merged/docs/report/A Self-Consistent 0D–2D Hybrid Framework for ICP Etch Modelling.pdf`

---

## Prompt to paste

```
I am validating a Phase-1 plasma simulation report ("A Self-Consistent 0D–2D Hybrid Framework for ICP Etch Modelling.pdf", 95 pages, DTPM
model) against an experimental dissertation ("Mettler Dissertation.pdf", 105
pages, UIUC 2025). Both PDFs are attached.

CONTEXT:
- Mettler's dissertation covers TWO reactors: (i) a Helicon / PMIC source used
  only for probe-material screening in Ch. 4.1, and (ii) a TEL ICP etcher used
  in Ch. 4.2-4.3 for the spatially resolved F-density measurements that are
  the benchmark for my model.
- My Phase-1 model targets the TEL geometry, SF6/Ar chemistry at 10 mTorr /
  ~700-1000 W.
- I already found three classes of issues via text analysis: (1) factual
  citation errors (e.g., I say "BBAS+OES" but Mettler used "W/Al radical
  probes + actinometry"), (2) operating-condition mismatch (Mettler's radial
  profile is at 1000 W + bias; my baseline is 700 W no-bias), and (3) missing
  comparisons.

WHAT I NEED FROM YOU:
1. **Figure-by-figure cross-check.** For each of my report's figures that
   cite Mettler (search my A Self-Consistent 0D–2D Hybrid Framework for ICP Etch Modelling.pdf for "Mettler" — should be ~10 occurrences
   including captions), open the corresponding Mettler figure and tell me:
   (a) Do the operating conditions in my caption match the Mettler figure's
       actual conditions (power, pressure, gas mixture, bias)?
   (b) Do the axis scales match? If Mettler reports values in #/m^3, is my
       plot in cm^-3 with the right factor of 1e6?
   (c) Is my overlay of Mettler data points plausibly correct, or does it look
       like I have digitised the wrong curve?

2. **Visual gap audit.** Look at Mettler's TEL figures specifically:
   Fig 4.9 (ICP absolute [F] vs SF6 flow), Fig 4.13 (bias effect on Si etch
   rate), Fig 4.14 (normalised radial [F] + cubic fit), Fig 4.17 (absolute
   radial [F] + Si etch rate at 30% and 90% SF6, bias on/off), Fig 4.18 (Si
   etch probability vs F flux). For each, tell me whether my report contains
   a direct like-for-like comparison and — if not — what a one-sentence
   summary of that missing comparison would look like.

3. **Numerical spot-check.** Mettler's Fig 4.14 gives a cubic-fit analytic
   formula: y(r) = 1.01032 - 0.01847*r^2 + 7.139e-4*r^3 with r in cm. Evaluate
   this at r = {0, 2, 4, 6, 8} cm and confirm the five values (approximately
   1.00, 0.95, 0.81, 0.53, 0.25). Then look at the radial profile I show in
   my A Self-Consistent 0D–2D Hybrid Framework for ICP Etch Modelling.pdf and estimate visually whether my curve passes through all five
   Mettler points within ±10%.

4. **Flag anything I missed.** If you see any other Mettler figure I should
   be comparing against (e.g., Fig 4.12 radial etch probability, Fig 4.16
   Arrhenius of W etch) and am currently ignoring, list those too.

Please structure your answer as:
  Section A: Factual citation errors confirmed / disputed
  Section B: Operating-condition mismatches confirmed / disputed
  Section C: Missing benchmarks — priority-ordered list
  Section D: Anything else unexpected
```

---

## Expected output

- A structured gap report (sections A–D) that you can paste back into Claude Code as **additional input** before executing Stages A + B of the Mettler-accuracy plan (`~/.claude/plans/mutable-swimming-chipmunk.md`).
- Any *new* corrections that ChatGPT / Claude-web catches but my text-only analysis missed can be added to the `METTLER_CORRECTIONS.md` and `METTLER_VALIDATION_POINTS.md` files before execution.

---

## Suggested follow-up prompts (iterative)

After the first round, useful follow-ups:

**Follow-up 1 — Digitise Fig 4.17 points for me:**
```
From Mettler Fig 4.17 (top panel, wafer bias off), digitise the 90% SF6
absolute F density points at radial positions (0, 2, 4, 6, 8) cm and return
them as a Python list of (r_cm, density_per_m3) tuples. Do the same for the
30% SF6 curve.
```

**Follow-up 2 — Cubic fit extrapolation:**
```
Mettler's cubic fit only covers 0-8 cm, but my wafer radius is 10 cm.
Evaluate the cubic at r = 10 cm. Does it still give a physically reasonable
(positive) value? If not, what extrapolation would you use instead?
```

**Follow-up 3 — Bias effect on absolute density:**
```
Compare Mettler Fig 4.17 top and bottom panels side-by-side. At r = 0, what
is the quantitative factor by which bias-on increases [F] compared to
bias-off, for 90% SF6? For 30% SF6? This tells me how much "free" density
my bias-less simulation is missing.
```
