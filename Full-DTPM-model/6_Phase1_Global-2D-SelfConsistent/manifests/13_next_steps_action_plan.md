# Next-steps action plan — electron-treatment upgrade

Date: 2026-04-19
Scope: Post-supervisor-meeting synthesis of the current scientific direction, current-state
audit, electron-model upgrade roadmap, discrepancy analysis, publication storyline, and
prioritised task list. Based on two parallel deep-dive audits of the `sf6_2d_icp/`,
`tel_model/`, and `phase2_electron_kinetics/` codebases completed on 2026-04-19.

Supervisor constraints in force:
- No PINNs as the main path — documented negative result
- No 3D
- No transient / E-H mode modelling
- Stay in the current 2D steady-state hybrid framework and deepen the physics

---

## 1. Current-state assessment

### 1.1 Codebase divergence (critical finding)

Two parallel 2D codebases exist and they have **diverged in scope**:

- [active_projects/sf6_2d_icp/](../active_projects/sf6_2d_icp/) — the **canonical physics solver**
  that matches the supervisor's v2 Phase-1 report. Contains
  `shared_modules/solvers/em_solver.py` (FDTD), `sheath_model.py` (Lieberman Bohm + m12 bias
  sheath), and the self-consistent η coil-plasma transformer. This is the codebase the
  publication is about.
- [active_projects/tel_model/](../active_projects/tel_model/) — a parallel pipeline that
  prototypes the same reactor but currently uses a **prescribed Bessel-cosine ne profile**, a
  **phenomenological Gaussian power-deposition ansatz**, and **Arrhenius-Maxwellian rates**.
  It also hosts the PINN negative-result branch and the supervised spatial surrogate. The
  Lieberman η-coupling and m12 bias-sheath modules from the v2 report are **absent from
  tel_model's production run path** (`run.py`, `solver.py`). tel_model's own `em_solver.py`
  and `sheath_model.py` analogues exist but are dormant.

**Consequence for the upgrade**: physics improvements should land in `sf6_2d_icp/`. tel_model
should be scoped as the ML/surrogate accelerator and gradually frozen unless it catches up on
physics.

### 1.2 Electron treatment — what's there today

| Aspect | Current state | Where |
|---|---|---|
| Te | Uniform 0D in default mode; optional 2D energy PDE with κ_e·∇²Te + P/(3neTe/2) − L_e = 0, Te=1 eV Dirichlet walls | `tel_model/src/solver.py:493-587`, `TELSolverWithEnergy` |
| EEDF | **Maxwellian everywhere** — both Arrhenius production rates and the "LXCat" path (which integrates σ(E) against a Maxwellian EEDF, not a solved Boltzmann EEDF) | `sf6_rates.py:63-139`, `lxcat_rates.py:31-67` |
| ne | **Prescribed** as Bessel-cosine eigenmode in the ICP region with exponential decay below the aperture; amplitude set by 0D volume-average power balance. No continuity PDE solved. | `solver.py:281-349` `_compute_ne()` |
| Power deposition | **Phenomenological Gaussian** in (r/R, z/L). FDTD not called in production. | `solver.py:503-534` `_compute_power_deposition()` |
| 0D↔2D coupling | **One-way Picard**: Te and ne push from 0D into 2D species transport; volume-averaged species NOT fed back to 0D | `solver.py:589-690` `solve()` |
| LXCat machinery | Present but **dormant** (imported only by `generate_lxcat_v3.py`, not by `run.py`); produces Maxwellian-averaged rates from Biagi cross-sections, toggled via `rate_mode='legacy'|'lxcat'` | `solver.py:79-87`, `lxcat_rate_provider.py` |

### 1.3 What changed after EM-coupling was introduced (v2)

The v2 Phase-1 report introduced:

1. Lieberman circuit-closure η (~0.95 at Mettler's 1000 W / 10 mTorr / 70% SF6, replacing the
   prescribed η=0.43),
2. m12 bias-sheath expansion source (calibrated λ_exp=3.20, blind-tested on 30% SF6 within
   ±20%),
3. electronegative ambipolar diffusion correction (rigorous but quantitatively ~2% at
   Mettler's α≈0.02).

**These landed in `sf6_2d_icp/`**. Their interpretation shifts the story from "we guess η and
the bias" to "η is an observable"; absolute-[F] magnitude gap closed from 2–4× to within
±30%.

---

## 2. Main scientific conclusions from the meeting

1. **The next scientific step is electron physics, not PINNs.** Physics-informed neural
   networks for the 2D PDE system are documented as a negative result
   (`tel_model/PURE_PINN_NEGATIVE_RESULT.md`) — Arrhenius stiffness + second-derivative
   autograd instability + multi-scale loss competition. Not worth more attention.
2. **ML work is supplementary, not a paper.** The 42-case EnsemblePredictor at 14.7% / 11.9%
   median error and 31× speedup is a useful accelerator, repositionable as (a) a
   supplementary section, or (b) a short companion ML paper. Not the main scientific
   contribution.
3. **Stay 2D, stay steady-state.** No 3D. No transient / E-H mode dynamics. Invest in physics
   depth within the current framework.
4. **Model-vs-measurement is the target, not 0D-vs-2D framing.** The remaining Mettler
   residuals (+8% inner, +22–27% outer; composition insensitivity; γ_Al calibrated only at
   90% SF6) are what the electron upgrade must move.

---

## 3. Electron-model upgrade roadmap

The phase-2 machinery already exists. It's **not wired in**. The upgrade is largely
integration, not green-field.

### Phase A — Minimum viable upgrade (1 week)

**Goal**: Replace the Arrhenius-Maxwellian rate coefficients in the `sf6_2d_icp` Picard loop
with Boltzmann-derived rates from the Tier-2 MLP surrogate. Toggleable behind a config flag
so the old path stays comparable.

- Import `phase2_electron_kinetics/tier2_pinn/get_rates_pinn.py`. API is
  `get_rates_pinn(E_over_N, x_Ar, pressure_mTorr) → {Te_eff, k_iz, k_att, k_diss, k_exc}`,
  returns numpy arrays, 19 397 parameters, validation error 0.41% (Te_eff) / 1.49% (k_att) /
  10.8% (k_iz near floor) on the 168-point grid.
- Two coupling moves required in `sf6_2d_icp`:
  1. At each Picard iteration, compute a **local reduced field E/N** from the FDTD-solved Eθ
     and the neutral density `n_0 = p/(kB·Tg)`.
  2. Call `get_rates_pinn(E/N, x_Ar, p)` per cell (batchable — one call per mesh) and use the
     returned k_iz, k_att, k_diss in place of Arrhenius.
- **Acceptance gate**: at Mettler's 1000 W / 10 mTorr / 90% SF6 / bias-off, compare the
  centre-to-edge [F] drop with Boltzmann rates vs Arrhenius rates. If Δ > 2 pp, Tier-2
  becomes the production path. The Phase-2 report (Tier-1 §3.4) already met the
  Maxwell-vs-Boltzmann gate at E/N=50 Td (≤20% per-channel deviation); moving to
  spatially-varying E/N amplifies or damps this.
- No solver dependencies: HDF5 pre-shipped, no BOLSIG+ binary, no GPU
  (`phase2_electron_kinetics/README.md` §Dependencies).

### Phase B — Stronger physics upgrade (2–3 weeks)

**Goal**: Replace the prescribed Bessel-cosine ne profile with a **solved 2D electron
continuity equation** with ambipolar diffusion and self-consistent ionisation source
S_iz(r, z) = n_e · ∑_j k_iz,j · n_j. Simultaneously replace the local-algebraic Te inversion
with a **2D electron energy transport PDE** that includes thermal conduction κ_e ∇²Te.

- Both equations are already partially implemented in `tel_model/src/solver.py`
  (`TELSolverWithEnergy`) — **port the energy PDE into `sf6_2d_icp/` and add a proper
  ambipolar continuity PDE** on the masked domain.
- Coupling schema: Picard iteration with Te(r,z) and ne(r,z) both evolving per iteration,
  Bohm BCs at solid walls for ne, mixed BCs for Te (Dirichlet T_wall ≈ 1 eV at walls in
  electron-temperature sense, or Neumann).
- **Acceptance gate**: α(r, z) emerges spatially varying (not the uniform α≈0.02 from 0D).
  The outer-wafer over-prediction (+22–27%) should contract — because α grows toward the edge
  where ne drops, the ambipolar correction bites there.
- This addresses Phase-1 report limitations L1 (electron-energy conduction) AND F2
  (electronegative ambipolar radial broadening) simultaneously.

### Phase C — Optional later refinement (4+ weeks, after Phases A/B settle)

**Goal**: Couple the Tier-3 MCC core (`tier3_picmcc/mcc_module.py`, Vahedi-Surendra
null-collision, 0D-validated within 15% of BOLSIG+) to the spatial Boris pusher for a
**one-shot spatial PIC-MCC validation** at one operating point. Not production — a
ground-truth check on whether non-local electron transport matters at Mettler's condition.

- This is the only place PINN/ML-flavoured work naturally reappears: the Tier-2 surrogate is
  torch-based and differentiable, so it could plug into a gradient-based Picard scheme if
  desired, but this is future research not present scope.

**Ranked priority**: Phase A first (~1 week, highest impact per hour). Phase B next
(2–3 weeks, most likely to close the outer-wafer residual). Phase C deferred.

---

## 4. Model-vs-measurement discrepancy analysis

At Mettler's canonical 1000 W / 10 mTorr / 90% SF6 / bias-off point (v2 Fig 7.31):

| Discrepancy | Region | Most likely driver | Electron-physics cause? | Fix |
|---|---|---|---|---|
| +8% over-prediction | r ≤ 2.5 cm | Rate-coefficient calibration (Arrhenius vs Boltzmann) at Mettler's E/N; or minor γ_Al sensitivity | **Partially** — Maxwellian assumption | Phase A |
| +22 to +27% over-prediction | r ≥ 5 cm | Prescribed ne profile doesn't feed back ionisation; uniform α instead of spatially-varying α(r,z); Arrhenius rates invalid at the lower E/N near the wafer edge | **Yes, dominantly** | Phase B + Phase A |
| −12.8% under-prediction (30% SF6 blind) | Centre | γ_Al = 0.18 is composition-independent in the model; Mettler's data implies γ_Al(x_Ar) | No — wall chemistry | F4 composition sweep (separate track) |
| Absolute magnitude factor (historical 2–4×) | Global | Prescribed η, absent bias sheath — **already resolved in v2** | No | Done |
| 200 W underprediction at 200 W ICP | Global | E-to-H mode transition | No — transient | Deferred per meeting |

### Ranked highest-value corrections

1. **Spatial ambipolar α(r,z) from a solved ne PDE** (Phase B) — most likely single fix for
   the outer-wafer residual.
2. **Boltzmann rates via Tier-2 surrogate** (Phase A) — cheap integration, moves all channels
   consistently; primarily a check that the 20% gate holds spatially.
3. **Composition sweep of γ_Al** (F4 rerun) — resolves the 30%/90% SF6 transferability gap.
   This is a chemistry/wall recalibration, independent of the electron upgrade.

Items that can be de-prioritised because they don't move Mettler residuals:

- E-H mode transition (L2), single wall temperature (L3), diffusion-limited surface regime
  (L4) — Mettler's [F] is safely in the kinetic regime (his Fig 4.9 / Table 4.4 threshold at
  ~1.1×10²¹ m⁻³).

---

## 5. Publication storyline

**Paper title (working)**:
*"Self-consistent two-dimensional hybrid ICP model with electromagnetic and wafer-bias
coupling for SF6/Ar etch reactors"*

### Narrative arc

1. **Motivation** — spatially resolved fluorine prediction is the central modelling problem
   for etch uniformity; existing frameworks either lack spatial information (0D global) or
   lack tractable parametric use (full 3D multiphysics); neither is deployable as an
   industrial digital-twin kernel.
2. **Framework** — masked-domain 2D hybrid on the TEL geometry: FDTD for Eθ(r,z), Lieberman
   coil-plasma transformer for η, m12 capacitive bias sheath, 54-reaction Lallement
   chemistry, Kokkoris-type wall surface chemistry. Three prescribed parameters of prior 2D
   approaches eliminated.
3. **Validation** — benchmark against 0D Lallement (internal consistency) and against
   Mettler's TEL W/Al-probe data (external cross-reactor benchmark). η(P) emerges flat at
   0.95 across 200–1200 W (loading-limited regime). Bias-sheath blind-test on 30% SF6 passes
   within ±20%.
4. **Residual physics limits** — enumerate L1–L4 and F1–F6. Show the +8% / +22–27% radial
   residual structure. **Point at Boltzmann rates and spatially-resolved α as the two
   remaining physics gaps**. Include the Maxwell-vs-Boltzmann Phase-2 comparison at the
   Mettler point (this is the NEW figure from Phase A).
5. **Computational cost** — ~55 s per converged Picard iteration on one CPU core; suitable
   kernel for parametric sweeps and surrogate training.

### Figures

| Figure | Source | Status |
|---|---|---|
| Reactor geometry (T-shape, masked domain) | Phase-1 v2 Fig 4.1 | ready |
| FDTD Eθ(r,z), \|B\|(r,z) | v2 Fig 6.1/6.2 | ready |
| Power deposition P(r,z) | v2 Fig 6.3 | ready |
| Converged 2D species fields (9-species panels) | v2 Figs 7.5–7.6 | ready |
| Wafer-level radial [F] vs Mettler | v2 Fig 7.10 | ready |
| Emergent η(P) across power sweep | v2 Fig 7.23 | ready |
| Bias-sheath composition blind test | v2 Fig 7.24 | ready |
| Residual vs radius + ratio vs power | v2 Fig 7.31 | ready |
| **Maxwell-vs-Boltzmann rate comparison at Mettler's E/N** | NEW — from Phase A | to generate |
| **Spatially-resolved α(r,z)** | NEW — from Phase B (if done in time) | optional |
| γ_Al composition sweep | F4 rerun | to generate |

**Figures that need rerun**: γ_Al composition sweep at (30%, 70%, 90%) SF6 to quantify
composition-dependence of the wall calibration. Nothing else needs rerunning.

### ML / surrogate positioning

Two options, companion paper preferred:

- **Option A — supplementary section** in the main paper. 2 pages: EnsemblePredictor as
  parametric accelerator, 14.7% / 11.9% median error, 31× speedup, with the PINN negative
  result explicitly acknowledged.
- **Option B — separate short companion paper** titled e.g. *"Spatial-field surrogate for
  fluid-plasma simulation: supervised MLP ensemble with physics regularisation"*. Cleaner
  because the ML story (supervised surrogate + PINN negative result + UQ ensemble) is a
  self-contained methodological contribution.

---

## 6. Prioritised action list

### 6.1 Immediate next tasks (this week)

| # | Task | Priority | Dependency | Output |
|---|---|---|---|---|
| 1 | Write a 100-line bridge script `sf6_2d_icp/code/boltzmann_rates.py` that imports `get_rates_pinn` from Phase-2 and exposes the same signature as the current Arrhenius block | HIGH | none | `boltzmann_rates.py`, 1 unit test |
| 2 | Wire Tier-2 surrogate into the `sf6_2d_icp` Picard loop behind a `use_boltzmann_rates=False` default flag | HIGH | #1 | diff against current solver |
| 3 | Run Mettler benchmark (1000 W / 10 mTorr / 90% SF6 / bias-off) twice — Arrhenius and Boltzmann — generate the Maxwell-vs-Boltzmann panel | HIGH | #2 | new figure, 1-page memo |
| 4 | Resolve the η canonical value (0.12 vs 0.16) the supervisor flagged | HIGH | — | 1-line answer in project README |

### 6.2 Code tasks (week 2+)

| # | Task | Priority | Dependency | Output |
|---|---|---|---|---|
| 5 | Port `TELSolverWithEnergy` (2D electron energy PDE with κ_e) from `tel_model/src/solver.py` into `sf6_2d_icp/` | HIGH | #3 | new solver module |
| 6 | Add an ambipolar ne continuity PDE to `sf6_2d_icp/` replacing the Bessel-cosine ansatz | HIGH | #5 | new solver module |
| 7 | Freeze `tel_model/` prescribed-profile path; add a redirect README pointing to `sf6_2d_icp/` for physics work | MEDIUM | #6 | README update |
| 8 | Produce a unified `requirements.txt` spanning `sf6_2d_icp/`, `sf6_ar_wallchem/`, `phase2_electron_kinetics/` | LOW | — | single file |

### 6.3 Physics / modelling tasks

| # | Task | Priority | Dependency | Output |
|---|---|---|---|---|
| 9 | γ_Al composition sweep at 30%/70%/90% SF6 using the v2 `sf6_2d_icp` framework | HIGH | #3 | new figure, closes F4 |
| 10 | Decide whether Tier-2 surrogate supersedes Arrhenius as production (Phase-2 gate: Δ centre-to-edge > 2 pp) | HIGH | #3 | 1-paragraph decision memo |
| 11 | Spatially-resolved α(r, z) diagnostic from the ne-PDE solve; quantify outer-wafer improvement | MEDIUM | #6 | new panel |

### 6.4 Validation tasks

| # | Task | Priority | Dependency | Output |
|---|---|---|---|---|
| 12 | Re-run Mettler Fig 4.17 validation (30% and 90% SF6, bias-on and bias-off) with Phase-A Boltzmann rates | HIGH | #3 | updated v2 Fig 7.22 |
| 13 | Rerun the 0D Lallement internal-consistency benchmark with Boltzmann rates | MEDIUM | #3 | updated v2 Fig 7.25–7.26 |

### 6.5 Paper / report tasks

| # | Task | Priority | Dependency | Output |
|---|---|---|---|---|
| 14 | Draft paper outline (5–6 pages, journal-ready structure) based on the storyline in §5 | HIGH | — | outline |
| 15 | Fold Phase-A results + Phase-2 technical content into Ch 8 (Future Work → partially realised) | HIGH | #3, #10 | updated report section |
| 16 | Write a 2-page supplementary OR companion paper stub on the EnsemblePredictor surrogate | MEDIUM | — | draft |
| 17 | Move MISTAKES_Phase1_located.md into `manifests/` as audit trail (DONE in this session) | DONE | — | — |

### 6.6 Optional later tasks

| # | Task | Priority | Dependency | Output |
|---|---|---|---|---|
| 18 | Phase C — spatial PIC-MCC one-shot validation at Mettler's point | LOW | #6 | confirmation memo (not production) |
| 19 | Extend Phase-2 BOLSIG+ grid to include SF5, SF4 fragments as they become non-negligible (>30% dissociation) | LOW | #10 outcome | extended lookup table |
| 20 | Decide canonical codebase: keep both `tel_model/` and `sf6_2d_icp/` or deprecate one | LOW | #7 | 1-page decision doc |

---

## 7. Next 1–2 weeks — specific plan

### Week 1 (immediate)

- **Day 1**: Tasks #1 + #4. Write the bridge script. Get supervisor's η decision. Run the
  unit test that `get_rates_pinn(E/N=50 Td, x_Ar=0)` returns Te_eff within 1% of Tier-1's
  tabulated value.
- **Day 2**: Task #2. Wire Tier-2 into the Picard loop in `sf6_2d_icp`. Test that the
  Arrhenius default path still converges unchanged.
- **Day 3**: Task #3. Run both Mettler benchmarks. Produce the side-by-side
  Maxwell-vs-Boltzmann figure. Quantify the Δ in centre-to-edge [F] drop.
- **Day 4**: Task #10. Write the 1-paragraph decision memo based on Phase-A result. If
  Δ > 2 pp → Tier-2 becomes production; if not → keep Arrhenius and document Phase-A as a
  check.
- **Day 5**: Task #9 start. Run γ_Al sweep at 30%/70%/90% SF6 (3 runs × 30 min each). Plot
  the result.

**Week-1 deliverables**:

- Working Tier-2 → sf6_2d_icp integration behind a flag
- Maxwell-vs-Boltzmann comparison figure at Mettler's point
- γ_Al composition sweep figure
- Decision memo on whether Phase A supersedes Arrhenius
- η canonicalisation (0.12 vs 0.16) landed in the project README

### Week 2 (physics upgrade)

- **Days 1–2**: Task #5. Port `TELSolverWithEnergy` energy PDE into `sf6_2d_icp`. Replicate
  Te(r,z) solution at Mettler's reference point and verify volume-averaged Te matches the 0D
  result within 5%.
- **Days 3–4**: Task #6. Add the ambipolar ne continuity PDE. Replace the Bessel-cosine
  ansatz. Re-solve Mettler's case and compare to the prescribed-profile result.
- **Day 5**: Task #11. Extract spatially-resolved α(r,z) from the converged ne solution.
  Quantify the outer-wafer residual change; update Fig 7.31.

**Week-2 deliverables**:

- Fully self-consistent 2D ne and Te in `sf6_2d_icp` (Phase B minimum)
- Updated residual analysis
- A decision on whether Phase B closes the +22–27% outer-wafer gap or whether further work
  is needed

After week 2, the paper draft (Task #14) becomes the primary output, with Phase A + Phase B
results as the new scientific content.

---

## Key blockers / decisions needed from supervisor

1. **η canonical value**: 0.12 (Lallement v1) vs 0.16 (March kink figures) vs the emergent
   η ≈ 0.95 from the v2 Lieberman circuit. Likely answer: the emergent value is canonical
   going forward; earlier prescribed values are historical. Confirmation needed.
2. **Codebase strategy**: consolidate on `sf6_2d_icp/` for physics and freeze tel_model at
   its current state as the ML-surrogate branch, OR merge. Recommended: consolidate on
   sf6_2d_icp.
3. **Companion-paper decision**: is the EnsemblePredictor a supplementary section of the
   main paper, or a separate short methods paper? Recommended: separate, because the ML
   story (supervised surrogate + PINN negative result + UQ ensemble) is a self-contained
   methodological contribution.
