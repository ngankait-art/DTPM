# Speaker notes — Phase 2 electron kinetics

**Presenter:** Zachariah Ngan, Illinois Plasma Institute
**Target audience:** Muhammad (supervisor), ~10 minutes, ~45 s per slide average.
**Goal:** walk the supervisor through the Tier 1 / Tier 2 / Tier 3 deliverables in the exact order of the workplan, demonstrate that each decision gate has been hit with real numbers, and make the next-step ask (spatial PIC-MCC integration) land cleanly.

---

## Slide 1 — Title
*~15 s*

- Open with: "This is the Phase 2 electron-kinetics deliverable against the April 2026 workplan. I want to walk through it tier by tier and show you the decision gates."
- Name the author/affiliation and move on.
- Do not dwell; the content is the point, not the title card.

## Slide 2 — What the workplan asked for
*~60 s*

- Remind the supervisor of the problem statement: DTPM currently evaluates every rate coefficient from an Arrhenius form that assumes a Maxwellian EEDF.
- State plainly that this is known to be structurally wrong in electronegative SF₆: attachment and inelastic channels deplete the real EEDF relative to a Maxwellian at the same mean energy.
- The Phase 2 question is not whether the assumption is wrong — it is wrong — the question is whether the correction moves the fluorine profile enough to be worth the infrastructure change.
- Emphasize: "Three-tier answer, each tier gated on the previous one. This deck walks you through exactly that."

## Slide 3 — Three tiers, executed in order
*~50 s*

- Walk the three rows of the table slowly; each tier has a deliverable and a decision it enables.
- Tier 1 is the BOLSIG+ lookup and the 20% Maxwell-vs-Boltzmann test at the reference point.
- Tier 2 is the fast differentiable production API built on top of the Tier 1 grid.
- Tier 3 is the reusable MCC collision core that a spatial PIC-MCC run will later consume.
- Critical honesty point at the bottom: "This deck does not cover the full spatial PIC-MCC integration. That is the stated next step and it is outside what I've built in this pass. Tier 3 here is the collision core plus a 0D cross-check, not the full spatial run." Say this out loud; don't let the supervisor infer it.

## Slide 4 — Tier 1 setup
*~60 s*

- 168-point grid, 14 E/N values with densification in the 70–500 Td region where the SF₅⁺ ionisation channel matters, times 6 Ar fractions.
- 53 cross-section channels from the Biagi LXCat set: 1 elastic momentum transfer, 7 attachment channels, 10 ionisation channels, 32 inelastic (excitation / dissociation / vibrational).
- The Maxwellian reference uses the identical cross sections convolved with a Maxwell distribution at the solver's own Te_eff. Stress this — "the only thing that differs between the two rate tables is the shape of the EEDF, so any difference we see is genuinely a Maxwellian-vs-Boltzmann difference, not a cross-section difference."
- Stored in HDF5 so the downstream tiers read it unambiguously.

## Slide 5 — Tier 1 decision gate hit
*~75 s*

- This is the first real result. Point at the plot and the numbers.
- Attachment ratio 0.82, dissociation 1.04, elastic 0.97 at the reference — "all three workplan-dominant channels are within 20%."
- Ionisation is below the numerical floor at 50 Td because the threshold is 15.7 eV and the bulk is at 1 eV. Don't hide this; state it honestly: "I flag this as out-of-range rather than passing or failing it — BOLSIG+ itself is at the floor there."
- Conclusion: "The Maxwellian assumption is adequate for the DTPM bulk fluorine profile. The ionisation bias only matters near the ICP skin depth, and that is exactly the region where the Tier 2 surrogate becomes essential."

## Slide 6 — Tier 2 architecture
*~60 s*

- "I adopted Option A from the workplan — supervised MLP — as the production backend. Option B is harder to train and Option A comfortably meets the acceptance gate, so there was no reason to escalate."
- 2 inputs, 96 units × 3 hidden layers with GELU, 5 outputs in log-space. 19,397 parameters — small enough to fit in a few hundred kilobytes and fast enough to call 1669 times per Picard iteration with no cost concern.
- Training: best epoch 666, validation MSE 1×10⁻³. Early stopping kept the overfit risk down.
- Mention Option B (M6 PINN) briefly as an alternate backend with a physics-informed loss term, available via the same `get_rates_pinn(weights_path=...)` override. Don't dwell.

## Slide 7 — Tier 2 acceptance check
*~75 s*

- This is the second real result. Point at the table.
- Te_eff median 0.41%, p90 2.57%, p99 7.56%. All three comfortably under the workplan 10% gate.
- k_att median 1.49%, p99 8.77%. Same story.
- k_iz p99 179% — honest explanation: "these are low-E/N floor points where the ground truth is at 10⁻²² m³/s. The surrogate output is also near the floor but doesn't exactly match zeros, and the relative-error denominator explodes. In the DTPM production regime of 30 to 300 Td the surrogate residuals drop below 10%."
- Do not over-defend the k_iz number. State it, explain it, move on.

## Slide 8 — The production API
*~60 s*

- This is the central delivery of Tier 2. The workplan §4.4 asked for exactly this function signature and it exists and runs.
- Walk through the code block line by line: `get_rates_pinn(E_over_N, x_Ar, pressure_mTorr)` returns a dict with arrays.
- Module-level model cache means DTPM pays the checkpoint load cost once, not per Picard step.
- "Drop-in replacement for the current Arrhenius block in the Stage 10 Picard loop. The integration is a call site, not a refactor."

## Slide 9 — Tier 3 scope
*~75 s*

- Lead with what was built: MCC collision module, LXCat parser, three case runners, cross-case analyzer. About 600 lines of Python end-to-end.
- Then the honest scope statement, which is the most important thing on this slide: "this is a 0D cross-check, not a full spatial PIC-MCC run."
- Explain why: the spatial Boris pusher lives in a separate repository at `1.E-field-map/script_v2/2.Extended-Aspects/7.DTPM_Project_ICP/` as modules M07/M08. The collision core I've built is designed to be consumed by that pusher, but the coupling itself is the stated next step.
- The non-local transport question in workplan §5.5 cannot be answered by a 0D MCC, and I am not claiming to have answered it. Be explicit about this — the supervisor will ask.

## Slide 10 — Three cases, all executed
*~60 s*

- Case A is the reference: 700 W / 10 mTorr / pure SF₆. Maps to E/N = 50 Td from the Stage 10 power balance.
- Case B is the low-pressure variant: 700 W / 5 mTorr / pure SF₆. Maps to E/N = 100 Td (pressure halves → field stays → E/N doubles).
- Case C is the Ar dilution variant: 700 W / 10 mTorr / 50% Ar. Stays at E/N = 50 Td for a clean comparison with Case A.
- 3000 macro-electrons × 30,000 steps × 2×10⁻¹¹ s = 600 ns of real MCC evolution per case. Each case produces a JSON result file, a diagnostic PNG, and a row in the comparison table.
- "Real MCC, real cross sections, real output. Not a stub."

## Slide 11 — Tier 3 results
*~90 s*

- Point at the bar chart. Te_eff agreement within 15% across all three cases: MCC 1.18 vs BOLSIG+ 1.04 for Case A, MCC 1.33 vs 1.52 for B, MCC 1.33 vs 1.26 for C.
- Physical trends: Case B has higher Te than Case A because lower pressure means higher E/N, which means more energy per collision. Case C has higher surviving electron fraction because Ar dilution reduces the SF₆ attachment loss rate. "Both trends are exactly what you'd expect from the physics."
- Attachment under-prediction: "This is the known limitation of a finite-duration 0D MCC. BOLSIG+ assumes a fully attachment-drained steady state; the 0D run is still draining at the end. The under-prediction is consistent across all three cases and does not affect the Te agreement that is the primary metric."
- Do not apologize for it. State it, explain it, move on.

## Slide 12 — Summary and decision
*~90 s*

- Recap the three gates: Tier 1 met, Tier 2 met, Tier 3 collision core built and working.
- Then the ask: "The next step I want to propose is coupling the Tier 3 MCC core to the Boris pusher at M07/M08 for the full spatial PIC-MCC run. That is the only way to actually answer the non-local transport question in workplan §5.5."
- Second parallel step: integrate `get_rates_pinn()` into the DTPM Picard loop and re-validate the fluorine profile against Mettler 2025. If the centre-to-edge drop changes by less than 2 percentage points, the Maxwellian Arrhenius baseline stands as adequate for the fluorine profile; otherwise Tier 2 becomes the production module.
- Pause here for questions before moving to the backup slide.

## Slide 13 — Backup: supplementary identifiability note
*~60 s*

- Only present this if asked, or if there's time and the supervisor wants the full story.
- Explain honestly: "There's a separate analytical side-study on two-anchor etch-rate calibration identifiability that exists in `supplementary/identifiability_M7/`. It was built in an earlier session before your workplan was loaded into my context, so it addresses a different question from Phase 2. The analytical result — that two-anchor calibration constrains only the product β·k_diss·n_e(P_H) — is internally consistent and potentially useful, but it is not the Phase 2 deliverable and should not be cited as such."
- "I preserved it because the work was real, but I've clearly labelled it as exploratory and moved it out of the main narrative. The primary deliverable you should read is the technical report in `report/technical_report.pdf`, which follows the Tier 1 / Tier 2 / Tier 3 structure of your workplan."
- End on: "Any questions about the primary deliverable, or should I walk through the supplementary work?"

---

## Timing summary
- Slides 1–3 (setup): ~125 s
- Slides 4–5 (Tier 1): ~135 s
- Slides 6–8 (Tier 2): ~195 s
- Slides 9–11 (Tier 3): ~225 s
- Slides 12–13 (summary + backup): ~150 s
- **Total: ~14 min with backup, ~13 min without**

If time is tight, drop slide 13 entirely and end on slide 12. The backup slide is only there because the M7 work physically exists in the repo and a supervisor looking at the zip will see the supplementary directory; better to address it proactively than to be asked about it.

## Preparation checklist before the meeting
- [ ] open `report/technical_report.pdf` in a second tab as reference
- [ ] have `tier1_bolsig/outputs/maxwell_vs_bolsig_report.md` ready if Muhammad asks for raw numbers
- [ ] have the `get_rates_pinn(...)` signature memorised; be ready to type it from memory
- [ ] know the Case A/B/C numbers cold
- [ ] have the next-step asks crisp: spatial PIC-MCC integration, DTPM Picard-loop integration
