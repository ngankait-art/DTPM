# REPO_COMPLETENESS_AUDIT

Deep technical audit of `SF6_tier3_repo 8/` (canonical SF6 Tier-3 PINN repo) and `TEL/TEL_Simulation_Full_Package/` (M7 coupled reactor), based only on file contents, not filenames or intent.

---

## 1. Pipeline diagram (what actually runs, end-to-end)

```
data/transport_pure_SF6.csv                       (Tier 1: BOLSIG+ Te↔E/N table, read-only)
            │
            ▼
tier2_pinn/get_rates_pinn.py  +  M5 weights       (Tier 2 local-rate surrogate, read-only)
            │
            ▼
phaseA_local_field_map.py   ── writes ──▶  outputs/phaseA_local_field.{npz,png}
   • builds TEL mesh via geom/{mesh,geometry}.py
   • places a prescribed exponential skin P_abs(r,z)         ← Assumption A1
   • frozen n_e = 1e18 everywhere                            ← Assumption A2
   • inverts P_abs → E/N per cell, calls get_rates_pinn
            │
            ▼
phaseB_teacher_generator.py + regenerate_teacher_{v2,v3,v4}.py
                          ── writes ──▶  outputs/phaseB_teacher_dataset_multicase{,_v2,_v3,_v4}.npz
   • algebraic local energy balance per cell → Te_local
   • Gaussian blur kernel (width λ_nl) → Te_nonlocal
   • recomputes rates on both fields
   • No PDE, no diffusion term, no self-consistent n_e
            │
            ▼
tier3_pinn/train_{m2,m3,m3b,m3c,multicase,rotation,rotation_prf,rotation_v3,rotation_v4}.py
                          ── writes ──▶  outputs/tier3_pinn_*.pt + logs + loss/pred PNGs
   • trains NonlocalCorrectionNet (6/8/9-channel variants)
   • masked loss: inside × (sat_flag == 0) + saturated-cell supervision + λ=0 identity
   • outputs: Δ(Te, log10 k_iz, log10 k_att)
            │
            ├─▶ scripts/generate_figures.py  ── writes ──▶  figures/fig01–fig15
            ├─▶ scripts/analyze_rotation_and_variance.py ── writes ──▶ outputs/rotation_*_table.{md,json}
            ├─▶ scripts/compare_picmcc.py  + scripts/make_picmcc_test_fixture.py
            │      ── synthetic fixture labelled SYNTHETIC_TEST_FIXTURE_NOT_A_PIC_MCC_RUN
            │      ── writes ──▶ outputs/picmcc_comparison_A.{md,json}
            ├─▶ scripts/validation_prep.py ── writes ──▶ outputs/validation_prep_report.md
            └─▶ scripts/integration_smoke_test.py (exits 0/1 only; writes nothing)
                          │
                          ▼
          tier3_pinn/integration_scaffold.py  ┐
          tier3_pinn/fluid_loop_mock.py       ├── scaffold + mocks; no plasma physics on fluid side
                          │                   ┘
                          ▼
          ┌──────── downstream DTPM / fluid solver ────────┐
          │  NOT PRESENT IN THIS REPO                      │
          │  mock_compute_local_rates = identity           │
          │  mock_advance_species_transport = identity     │
          │  + tiny perturbation                           │
          └────────────────────────────────────────────────┘

parallel: TEL/TEL_Simulation_Full_Package/  — 0D↔2D coupled reactor (src/sf6_chemistry.py, sf6_rates.py, solver_multispecies.py)
                          │
                          └── imports zero code from tier3_pinn; the PINN is NOT wired into TEL.
```

**True entry points (end-to-end that run):**
1. `phaseA_local_field_map.py` (Phase A reference map)
2. `phaseB_teacher_generator.py` + `regenerate_teacher_v{2,3,4}.py` (teacher datasets)
3. `python -m tier3_pinn.train_{multicase|rotation_v4|rotation_prf}` (PINN training)
4. `scripts/integration_smoke_test.py` (API smoke test with mocks)
5. `scripts/generate_figures.py` (all publication figures)
6. Inside TEL: `TEL_Simulation_Full_Package/run.py` and `run_models.py` (standalone 2D reactor; does not use the PINN).

**What does NOT run end-to-end:** any closed loop that goes *PINN → transport PDE → density fields → experimental comparison*. The path is broken at `fluid_loop_mock`.

---

## 2. PINN / kinetics status — **✔ COMPLETE**

Evidence:
- Architecture present: `tier3_pinn/model.py` defines `NonlocalCorrectionNet` (U-Net-ish, mask-aware first layer, GELU, residual ΔTe). In-channels configurable (6 / 8 / 9).
- Trainers present and self-consistent: `train_m2/m3/m3b/m3c/multicase/rotation/rotation_prf/rotation_v3/rotation_v4.py`, each with masked power-balance residual, saturated-cell supervision, identity-at-λ=0 constraint. GPU-aware (CUDA → MPS → CPU) in the v3/v4/prf trainers per status doc.
- Checkpoints saved and present: `tier3_pinn_m{1,2,3,3b,3c}.pt`, `tier3_pinn_multicase.pt`, `rotation_holdout{A..E}_seed{0,1,2}.pt`, `rotation_prf_holdout{A..E}_seed{0,1}.pt`, `rotation_v3_holdout{B,F}_seed{0,1}.pt`, `rotation_v4_holdout{A,B,G,H}_seed{0,1}.pt`. With paired JSON metric files.
- Reproducibility stats recorded: `outputs/seed_variance_summary.json`, `outputs/seed_variance_v2_summary.json`, `outputs/rotation_{,prf_,v3_,v4_}table.{md,json}`.
- Self-consistency checks pass per `validation_prep_report.md`: identity at λ=0 (max |pred| ≈ 3e-4 eV), smooth λ generalisation (err/std plateau 0.23–0.27 for λ ≥ 10 mm), 5-way hold-out rotation with ≥2 seeds, pressure bridge (v3) and pressure-axis closure (v4, hold-out B Δlog k_iz: 1.282 → 1.250 → 1.114).

**Classification: COMPLETE.** Predictions are reproducible from code, outputs are saved, internal validation is rigorous.

---

## 3. Non-local correction status — **✔ COMPLETE (on the stated scope)**

Evidence:
- Produces spatially resolved corrections on the full 2D TEL mesh: `predict_nonlocal_correction` returns `Te_nonlocal`, `k_iz_nonlocal`, `k_att_nonlocal`, and the three Δ fields (`integration_scaffold.py`, smoke-test shape/finite assertions).
- ΔTe, Δlog₁₀ k_iz, Δlog₁₀ k_att are the model's output channels (dataset/model headers both confirm). Re-exponentiation to k_iz/k_att happens inside `integration_scaffold`.
- Masking: first conv is mask-aware (`inside * x`), loss weight is `inside & (sat_flag == 0)`, smoke test runs `_check_finite(out, inside)` across all three fields. No NaN in masked region in the smoke test.
- Full vs half domain: raw data is half-domain (r ≥ 0). `scripts/generate_figures.py:119–123` has an explicit mirror helper ("*Mirroring helper — converts half-domain (r>=0) to full-domain display*"); `scripts/compare_picmcc.py:217` comments "*Figure — full mirrored TEL geometry, matching fig01–fig07 convention*". Display convention is consistently full-domain across fig01–fig15.

**Caveat (within scope):** the correction operator is a learned *Gaussian-blur surrogate on frozen neutrals and frozen n_e*. Phase A and Phase B explicitly freeze `n_e = 1e18` and neutrals; there is no self-consistent density update. Correct for a rate-coefficient correction; not a plasma simulator.

---

## 4. Reactor integration status — **B) SCAFFOLD ONLY**

Evidence:
- `tier3_pinn/integration_scaffold.py:3–6`: *"A minimal, honest scaffold … **This module does NOT couple to a real DTPM solver.** The DTPM code base is not part of this repository."*
- `tier3_pinn/fluid_loop_mock.py:3–14`: *"**This file does NOT simulate a plasma.** The 'fluid step' is a deliberately trivial operator (identity + tiny perturbation) whose only purpose is to drive the loop so we can show the call pattern and verify the Tier-3 correction API is wired correctly. A real DTPM coupling replaces `mock_compute_local_rates` and `mock_advance_species_transport` with actual solver calls."*
- `scripts/integration_smoke_test.py` exercises the public API but only checks shapes and finite-ness — *"Exits 0 on success, 1 on failure. … No outputs are written."* It is a CI smoke test, not a physics check.
- Convergence: `integrate_with_fluid_step` is a **single** Picard iteration. The driver loop inside `fluid_loop_mock.__main__` runs the loop but with an **identity fluid step** — convergence is trivial by construction.
- TEL coupling check: `grep` for `tier3_pinn`, `NonlocalCorrectionNet`, `load_tier3_model` across `TEL/TEL_Simulation_Full_Package/` returns **zero hits**. TEL computes its own rates via `src/sf6_rates.py` and `src/sf6_chemistry.py`; the PINN is **not** plugged into TEL. The two subsystems are disconnected.

**Closed-loop solver?** No.
**Kinetics actually used inside a transport loop?** No. Mocks only.
**Convergence implemented?** Only for the trivial identity operator.
**Executed anywhere against a real transport solver?** No.

**Classification: B — SCAFFOLD ONLY.**

---

## 5. Validation status — **⚠ PREP ONLY (plus internal self-consistency)**

Evidence:
- `outputs/validation_prep_report.md` Section 3 classifies every external dataset: `lallement_fig5b_Te` → **BLOCKED** (geometry mismatch, PINN TEL-only); `lallement_fig7_alpha_10mTorr` → **BLOCKED** (requires charge-balance PDE); `mettler_fig414_F_normalized` → **PARTIAL** (needs fluid step, scaffolded not implemented); `mettler_fig4p17_biasoff_density` → **BLOCKED**; `mettler_fig418_actinometry` → **OUT_OF_SCOPE**.
- Section 5 (Honest scope statement): *"A Tier 3–only quantitative comparison against experimental F density or etch-rate data is therefore not physically justified in isolation, and none is produced here."*
- PIC-MCC comparison: `outputs/picmcc_comparison_A.md` lists numeric metrics, but the PIC-MCC source field reads `SYNTHETIC_TEST_FIXTURE_NOT_A_PIC_MCC_RUN`. `scripts/make_picmcc_test_fixture.py:73–75` explicitly labels it synthetic; the script prints *"WARNING: this file is a synthetic test fixture, not real PIC-MCC data."*
- What is validated: internal self-consistency (rotation hold-outs, λ generalisation, seed variance, λ=0 identity). These are real, but not external validation.

**Classification: ⚠ PREP ONLY** (no real experimental or PIC-MCC validation; internal consistency is rigorous).

---

## 6. Figure audit

15 figures present under `SF6_tier3_repo 8/figures/`:

| Figure | What it shows (per status/report) | Domain | Data source in outputs | Judgment |
|---|---|---|---|---|
| fig01 | local vs non-local Te, k_iz | full mirrored | phaseB teacher | consistent |
| fig02 | PINN vs teacher M3c | full mirrored | teacher + tier3_pinn_m3c.pt | consistent |
| fig03 | error maps M3c | full mirrored | teacher + m3c | consistent |
| fig04 | λ generalisation | — | log files | consistent (plateau confirmed in status) |
| fig05 | loss curves progression | — | per-model `_log.md` | consistent |
| fig06 | multicase comparison | full mirrored | multicase ckpt | consistent |
| fig07 | saturated region m3b vs m3c | full mirrored | m3b, m3c | consistent; ~20× headline |
| fig08 | rotation summary | — | rotation_table.json | consistent |
| fig09 | x_Ar interpolation | full mirrored | multicase | consistent |
| fig10 | seed variance (hold-out C, n=3) | — | seed_variance_summary.json | consistent |
| fig11 | validation readiness | — | validation_prep_report.md | a *status chart*, not a validation |
| fig12 | power conditioning (9-ch vs 8-ch) | — | rotation_prf_table | consistent; negative result (overlap within σ) |
| fig13 | seed variance all hold-outs | — | seed_variance_v2_summary.json | consistent |
| fig14 | PIC-MCC comparison A | full mirrored | **synthetic fixture** | numerically consistent but source is synthetic |
| fig15 | pressure-axis closure (v2/v3/v4) | — | rotation_v{2,3,4} tables | consistent |

`scripts/generate_figures.py` centralises the half→full mirroring; no evidence of centerline artifacts in the code path (mirror is a clean flip with axis shared, not duplicated). No unused figures — each is referenced by the report, slides, or a status table.

**Caveat:** fig14 visually looks like a PIC-MCC comparison; the underlying source is `SYNTHETIC_TEST_FIXTURE_NOT_A_PIC_MCC_RUN`. The markdown report discloses this; the PNG alone does not.

---

## 7. Documentation audit

- `docs/project_status.md` (SF6_tier3_repo 8) is internally consistent with code and outputs: every claim references a file path; headline numbers (ΔTe err/std 0.23, M3c saturated improvement ~20×, hold-out B Δlog k_iz 1.282 → 1.114, hold-out C 0.589 ± 0.058) match the linked JSON/MD sources.
- The "Externally blocked" section names exact unblockers: real DTPM solver, PIC-MCC reference NPZ matching schema §6, third training power level.
- The "Not claimed (by design)" section is explicit: *"No experimental validation against real plasma-diagnostic data. No PIC-MCC validation; only harness verification against a labelled synthetic fixture. No DTPM-coupled species densities, etch rates, or F-density profiles."*
- `report/technical_report.tex` / `slides/tier3_presentation.tex` / `slides/speaker_notes.md`: the status doc asserts they agree numerically and in wording; the file sizes (1232 / 385 / 158 lines) line up with that role.
- **Inflated claims:** none found in `SF6_tier3_repo 8`'s own docs — the project self-describes as scaffolded at the integration boundary. A risk emerges only if *an outside summary* conflates "Tier 3 PINN complete" with "closed coupled solver complete".
- **Contradictions:** none internal to SF6_tier3_repo 8. Separately: `TEL/Coupling_Response_to_Professor.md` describes a 0D↔2D coupling that is real (in TEL) but is **not connected** to the Tier-3 PINN. A document summarising both could misleadingly imply integration.
- **Outdated numbers in older snapshots:** `SF6_tier3_repo 7/docs/project_status.md` frames itself as *"Snapshot after the v4 roadmap round"* (per-round narrative), not the final synchronized state — already captured in the forensic audit.

---

## 8. FINAL CLASSIFICATION

**B) FUNCTIONALLY COMPLETE (kinetics done, integration partial).**

---

## 9. Justification

### Definitively DONE
- 2D TEL mesh construction and masked domain (geom/*).
- Tier 2 local-rate surrogate integration (read-only import).
- Phase A local-field reference map (npz + figure + log).
- Phase B teacher dataset generator (single-case + v1/v2/v3/v4 multicase), schemas documented.
- Tier 3 PINN architecture (6/8/9-channel), training stack, 5-way rotation, seed-variance breadth, P_rf conditioning, pressure-bridge (v3) and pressure-axis closure (v4).
- Checkpoints and paired JSON metrics for every trained model.
- Public integration API (`load_tier3_model`, `predict_nonlocal_correction`, `apply_nonlocal_rate_correction`, `integrate_with_fluid_step`) with CI-style smoke test.
- Full figure set (fig01–fig15) with centralised mirror helper.
- Publication-grade internal docs: technical report, slides, speaker notes, reproducibility notes, interface spec for PIC-MCC comparison.
- TEL package: complete 0D↔2D coupled reactor solver with its own rate layer, four model variants (A/B/C/D), entry points `run.py` / `run_models.py`, external reference data sets, action-items checklist, supervisor-response derivation memo.

### PARTIAL
- Reactor integration on the Tier-3 side: scaffold + mock callbacks only. `fluid_loop_mock` uses identity operators. `integration_smoke_test` checks shapes/finite values, not physics.
- Electron energy PDE inside TEL: `solver.py` has `TELSolverWithEnergy` but `results/action_items_status.md` says it converges to unphysical T_e ≈ 0.5 eV; a parameterised T_e profile is used instead.
- External validation: `validation_prep` inventories datasets and computes only dimensionless/consistent partials; no absolute-quantity comparison exists.

### MISSING
- A real DTPM / fluid species-transport solver wired into `integrate_with_fluid_step` (both callbacks are mocks).
- A real PIC-MCC reference dataset matching the `docs/picmcc_comparison_interface.md` §6 schema. Currently only a synthetic fixture.
- A bridge between `tier3_pinn` and `TEL/TEL_Simulation_Full_Package/` — TEL imports none of the PINN modules; the PINN-corrected rates are not consumed by the 2D reactor solver.
- A third training power level (status doc's stated unblocker for continuous P_rf conditioning).
- Quantitative experimental validation against Mettler / Lallement / etch-rate data.

### What would lift this to A (FULLY COMPLETE SYSTEM)
1. Replace both callbacks in `integrate_with_fluid_step` with a real species-transport solver (either the TEL multispecies solver, re-entered as a callback, or an external DTPM solver).
2. Wire `tier3_pinn.predict_nonlocal_correction` into TEL's `sf6_rates.py` / `sf6_chemistry.py` rate evaluation.
3. Ingest a real PIC-MCC NPZ (not the synthetic fixture) and rerun `compare_picmcc.py`.
4. Close the TEL electron-energy PDE convergence issue (calibrate κ_e, fix wall BC) so the "partially complete" Action 4 becomes complete.
5. Produce at least one absolute-quantity comparison against an external experiment that the validation_prep report currently marks BLOCKED/PARTIAL.

---

## 10. Reviewer risks if presented as "complete"

A rigorous reviewer will challenge the following, in descending order of blast radius:

1. **"You claim a coupled Tier-3 PINN system, but both fluid callbacks are mocks."** The source says so explicitly (`fluid_loop_mock.py:3–14`, `integration_scaffold.py:3–6`). Any framing that elides "mock" is overstatement.
2. **"Your PIC-MCC comparison uses synthetic data."** `outputs/picmcc_comparison_A.md` names the source `SYNTHETIC_TEST_FIXTURE_NOT_A_PIC_MCC_RUN`. Presenting fig14 as PIC-MCC validation without that qualifier is misleading.
3. **"Your reactor-scale M7 deliverable (TEL) does not use the M6 PINN."** `grep` confirms TEL imports nothing from `tier3_pinn`. The "0D↔2D coupling closes the loop for the plasma-dtpm program" narrative is only true if the PINN is actually consumed by the reactor solver, which it is not.
4. **"Your electron energy PDE does not converge to physical values."** TEL `action_items_status.md` states κ_e and wall BC issues drive T_e → 0.5 eV; a parameterised profile is used instead. The paper should disclose this.
5. **"Your frozen-n_e assumption makes Phase A/B non-self-consistent."** `phaseA_local_field_map.py` header (Assumption A2) and `phaseB_teacher_generator.py` ("*No self-consistent n_e update (Phase C / future work)*") both flag this; any claim of self-consistency would contradict the code.
6. **"Your 'power conditioning' result is a negative result."** The v3 round confirms 8-ch and 9-ch overlap within seed σ when only {500, 700} W are in training. The status doc correctly labels this as a negative result; a slide deck that implies otherwise invites pushback.
7. **"Your validation is mostly internal consistency, not experimental."** Every external dataset in `validation_prep_report.md` is BLOCKED, PARTIAL, or OUT_OF_SCOPE.
8. **"Your mirrored-domain display could hide centerline inconsistencies."** The mirror helper in `generate_figures.py` looks clean, but a reviewer may ask for a raw half-domain version of fig01/fig02/fig03 to verify no interpolation artifact at r = 0.
9. **"Seed-variance n=2 on most rotations is thin."** Only hold-out C has n=3; most v4 hold-outs are n=2. The status doc discloses this; reviewers may still push for n ≥ 5 on the headline hold-outs.

**Internal honesty note:** the repository's own documentation (`validation_prep_report.md`, `project_status.md` "Not claimed" section, the explicit mock/synthetic labels in code) already pre-empts every one of these challenges. The misrepresentation risk does **not** come from the code or docs — it would come only from a downstream summary that skips the qualifiers baked into the source.
