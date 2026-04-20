# Phase-1 v2 execution plan — 5-step supervisor direction

Date: 2026-04-19
Scope: Concrete action plan for the supervisor's 5-step direction on the v2
self-consistent framework just pulled from
`ngankait-art/DTPM @ feat/phase1-global-2d-and-sf6ar-chemistry` →
`active_projects/phase1_self_consistent/`.

This supersedes manifest 13's earlier high-level roadmap; that doc was
produced before the v2 code was available locally. Manifest 13 remains
useful as the current-state audit and publication storyline.

Supervisor's 5-step direction:

1. Review the v2 report + codebase
2. Investigate the delta vs Mettler; find easy-reduction opportunities
   without biasing the model
3. Deploy the Surrogate ML approach and LxCat ML on the v2 code (wraps
   Phase 1)
4. Expand the model to full Boltzmann transport / PIC-MCC for electrons
   (predict EEDF from simulation)
5. Use PINNs / ML surrogate to accelerate the EEDF part only

Constraints in force (unchanged from manifest 13):

- No PINNs as the main path — negative result stands for full-PDE PINN
- No 3D
- No transient / E-H mode modelling
- Stay in the current 2D steady-state framework

---

## Step 1 — Complete

v2 codebase is local at `active_projects/phase1_self_consistent/` (72 MB,
15/15 unit tests pass). Report, EXECUTION_LOG, and config all read.
Summary of what's there:

### Three architectural upgrades implemented in v2

1. **Self-consistent η** via Lieberman transformer circuit
   (`src/dtpm/modules/m01_circuit.py`). η is no longer prescribed — it
   emerges as `R_plasma / (R_coil + R_plasma)`. At Mettler's
   1000 W / 10 mTorr / 70% SF6, η converges to ~0.95 (loading-limited).
2. **m12 capacitive wafer-bias sheath**
   (`src/dtpm/modules/m12_ccp_bias_sheath.py`). Single calibration
   `λ_exp = 3.20` matches 90% SF6 bias-on enhancement ×1.614 vs target
   ×1.60 (+0.85%); blind-tested on 30% SF6 at ×1.876 vs target ×2.15
   (−12.8%, within ±20%).
3. **Electronegative ambipolar diffusion correction**
   (`src/dtpm/solvers/ambipolar_diffusion.py`, Lieberman §10.3). α is
   threaded uniformly from the 0D global model into the 2D solver.

### Quantified Mettler delta

| Metric | Model | Mettler | Delta | Tolerance |
|---|---|---|---|---|
| Centre-to-edge [F] drop | 68.2% | 74% (Fig 4.14 cubic) | **5.8 pp low** | 8 pp (abstract) |
| Radial [F] residual, r ≤ 4 cm | — | — | ±10% | ±10% |
| Radial [F] residual, r ≥ 5 cm | — | — | **+15 to +27%** over-prediction | — |
| [F] at wafer centre, 90% SF6 bias-on | 1.66×10¹⁴ cm⁻³ | — | **−33%** vs Mettler | — |
| [F] at wafer centre, 30% SF6 bias-on | 0.68×10¹⁴ cm⁻³ | — | **−35%** vs Mettler | — |
| Bias enhancement, 90% SF6 | ×1.614 | ×1.60 | +0.85% (calibration) | — |
| Bias enhancement, 30% SF6 (blind) | ×1.876 | ×2.15 | −12.8% | ±20% |
| Mesh-discretisation contribution | — | — | 1.8 pp | — |

### Three residual-shape signatures

1. **Outer-wafer over-prediction (+15–27% at r ≥ 5 cm)** — structural, not
   noise.
2. **Absolute-magnitude under-prediction (−33 to −35%)** — both
   compositions similar; NOT composition-dependent.
3. **Composition-insensitive F-drop** — model gives 68% for both 30% and
   90% SF6; Mettler gives 67% and 75%. Direct fingerprint of fixed
   γ_Al = 0.18.

### Critical finding — L1 (electronegative ambipolar) was NOT the gap closer

Report §5.4.2 (lines 1752–1760): at Mettler's point, ne ≈ 10¹⁹ m⁻³ gives
α ≈ 0.02 (two orders of magnitude below the textbook 1–1.5 estimate).
The (1+α)/(1+αTᵢ/Tₑ) correction factor is only ~1.019, producing < 1 pp
change in the F-drop.

The EXECUTION_LOG (Apr 17) had predicted L1 would close ~6 pp of the gap;
the v2 simulation (Apr 18) falsified that expectation. L1 remains
physically essential in different regimes (higher p, lower P → α ≳ 1) but
is quantitatively small at Mettler's operating point.

This matters for step 2: **the remaining 30% absolute-magnitude gap and
the +15–27% outer-wafer residual cannot be attributed to L1.** The easy
knobs are elsewhere.

### Codebase state — three big findings

1. **Boltzmann PINN already exists** in the v2 code:
   `src/dtpm/modules/m09_boltzmann.py` + `src/dtpm/solvers/boltzmann_pinn.py`
   implement the two-term Boltzmann equation in energy space
   (Kim 2023 / Kawaguchi 2022 / Hagelaar & Pitchford 2005) with
   inference / bolos / train modes. **Supervisor's step 5 is ~70%
   implemented**; needs wiring into the Picard loop and a trained
   weights checkpoint.
2. **γ_Al is configurable** at `src/dtpm/core/config.py:94` and used via
   `src/dtpm/chemistry/wall_chemistry.py:99 get_gamma_map()`. Composition
   sweep is a YAML-only change.
3. **v2 runs locally.** `python3 -m pytest tests/` passes 15/15 in 0.08 s.

### Mettler citations

Greped the v2 `main.tex` for the 4 key mistake patterns (`Fig 4.5`,
`BBAS`, `pure SF6`, unqualified `74%`). **Zero hits.** Confirms all 10
documented mistakes from manifest 12 are fixed in the v2 report.

---

## Step 2 — Diagnostic plan (investigate the delta without biasing)

Goal: reduce the three residual-shape signatures without introducing knobs
that overfit a single data point.

### Ranked diagnostic tests

| # | Test | What it diagnoses | Runtime | Biases? |
|---|---|---|---|---|
| **D1** | Toggle m12 bias sheath off at Mettler 1000 W / 10 mTorr / 90% SF6 | Isolates how much of the absolute-magnitude residual is m12 calibration vs underlying physics. (Note: the prior "−33%" residual was a bias-on-model vs bias-off-Mettler comparison; the apples-to-apples residual at 90% SF6 bias-on is **−56.0%** against the Fig 4.17 digitised reference [F]_c = 3.774 × 10¹⁴ cm⁻³.) | 2 runs × 3 min | No — config flag |
| ~~D2~~ | ~~γ_Al composition-dependent fit~~ | **REJECTED by supervisor 2026-04-19**: γ_Al shall remain single-valued; do not make it species- or composition-dependent. | — | — |
| **D3** | **R_coil discovery sweep** — exact value is unknown per supervisor. Sweep R_coil ∈ {0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0} Ω and look at where the Mettler absolute-magnitude residual is minimised (sign-of-effect TBD by the sweep itself). | Identifies the operating R_coil consistent with measurement; quantifies how much of the absolute-magnitude gap is attributable to coil-loss uncertainty. **[EXECUTED 2026-04-19]**: η spans 0.79–0.97 across the sweep but [F]_c spans only 1.53–1.67 × 10¹⁴ cm⁻³ (±5%) and F-drop stays at 66.5–66.8% (±0.15%). Residual vs Mettler worsens monotonically with R_coil (best: R = 0.5 Ω → −55.8%; worst: R = 4.0 Ω → −59.4%). R_coil is NOT the lever that closes the Mettler gap. See `phase1_self_consistent/docs/notes/DIAGNOSTIC_SWEEPS_D1_D3_MEMO.md` for full table. | 8 runs × 3 min | No — supervisor asked for this sweep explicitly |
| **D4** | **Wire Tier-2 `get_rates_pinn` into Picard loop** at Mettler's point; compare Maxwell-Arrhenius vs Boltzmann rates side-by-side | Directly tests whether outer-wafer over-prediction is from Maxwellian rates at low E/N. If yes, this is both diagnosis AND fix. | ~1 day code + 2 runs × 3 min | No — physics-derived rate model replacing a simpler one |

**Recommendation**: Execute **D1 and D3 in parallel** (~30 min total; D2
rejected per supervisor — γ_Al stays single-valued). Then commit to
**D4** the following day. D4 is the code-wiring work that also lands
step 3a, so it has double value.

### What NOT to do (to avoid biasing)

- Do not tune λ_exp away from 3.20 on anything other than the original
  90% SF6 centre-point (that is the definition of its calibration).
- Do not tune multiple parameters on the same data point. Any single
  change must leave every other calibration invariant.
- **Do not make γ_Al species- or composition-dependent** (supervisor
  2026-04-19). γ_Al shall remain a single value; the composition-
  insensitivity fingerprint in the residual is accepted as a known
  limitation of the wall-chemistry layer, not a knob.

---

## Step 3 — Deployment plan (Phase 1 wrap)

### 3a. LxCat ML (Tier-2 `get_rates_pinn`) into v2 Picard loop

**Interface** (from `phase2_electron_kinetics/tier2_pinn/get_rates_pinn.py`):

```
get_rates_pinn(E_over_N, x_Ar, pressure_mTorr)
    -> {"Te_eff": ..., "k_iz": ..., "k_att": ...,
        "k_diss": ..., "k_exc": ...}
```

19,397-parameter MLP, validated to 0.41–1.49% median error on Tier-1's
BOLSIG+ grid. No BOLSIG+ binary required — HDF5 lookup pre-shipped.

**Wiring points in v2**:

- `src/dtpm/modules/m11_plasma_chemistry.py` — the Picard loop. Add a
  `use_boltzmann_rates: bool` config flag defaulting to `False` so the
  Arrhenius baseline stays comparable.
- `src/dtpm/chemistry/sf6_rates.py` — replace the Arrhenius k_iz, k_att,
  k_diss calls with a call into `get_rates_pinn(E/N(r,z), x_Ar, p)`.
- Compute local E/N from the FDTD Eθ field (already available at m06
  output) and the neutral density `n_0 = p / (kB · Tg)`.

**Acceptance gate**:

- If the centre-to-edge F drop changes by > 2 pp at Mettler's reference
  point → Boltzmann rates become production.
- If < 2 pp → fall back to Arrhenius and document Phase-A as validated.

### 3b. Spatial surrogate EnsemblePredictor on v2 outputs

**Source**: `active_projects/tel_model/src/spatial_surrogate_v2.py`
(42-case ensemble, 3-member UQ, 14.7% / 11.9% median error on nF / nSF6,
31× speedup, MC dropout).

**Integration path**:

- Re-train on v2 outputs. The tel_model surrogate was trained on the
  older prescribed-profile solver's outputs; those are not physically
  consistent with the canonical v2 framework. Re-training is essential.
- Keep the same architecture (3-model ensemble + MC dropout) and the same
  inputs `(r, z, P_rf, p_mTorr, frac_Ar)`.
- Expand the training grid if needed: manifest 13 Task #9 lists a
  2400-point sweep as the full target.

**Use**: parametric accelerator for the paper's sensitivity analysis; a
tool for rapid design-space exploration. **NOT** a replacement for the FD
solver — the FD solver remains the physics engine.

**Deliverable**: a `sf6_2d_icp_surrogate` package importable from anywhere,
with
`predict(r, z, P_rf, p_mTorr, frac_Ar) -> (fields, uncertainties)`.

---

## Step 4 — Full Boltzmann transport / PIC-MCC for electrons (post Phase 1)

**Target**: spatially-resolved EEDF f(ε; r, z) instead of Te(r,z) with a
Maxwellian assumption.

**Path of least resistance**: extend
`src/dtpm/modules/m09_boltzmann.py` from its current 0D (single E/N) mode
into a **per-cell** mode. The Picard loop calls the Boltzmann solver at
each cell with that cell's local E/N and composition, producing a
spatially-varying EEDF. This is a two-term spatial Boltzmann treatment;
not full PIC-MCC.

Full PIC-MCC (Tier-3 core in `phase2_electron_kinetics/tier3_picmcc/`,
Vahedi-Surendra null-collision, 0D-validated within 15% of BOLSIG+) is
deferred to a **one-shot validation run** at one operating point, as in
manifest 13 Phase C.

**Scope not yet decided**: whether to implement this per-cell Boltzmann
call tightly coupled to every Picard iteration (expensive), or as a
lookup-table approach that pre-computes the EEDF on a (E/N, x_Ar) grid
and interpolates (cheap). The Tier-1 BOLSIG+ lookup already does the
latter; per-cell live solve is the former. Recommendation: start with the
Tier-2 surrogate (step 3a), which effectively implements the cheap
lookup path.

---

## Step 5 — PINN / ML for EEDF-only acceleration (post Phase 1)

**Status**: `boltzmann_pinn.py` already implements a PINN for the
two-term Boltzmann equation in energy space. The scope reduction from
"full 2D PINN" to "Boltzmann-only PINN" is exactly what the supervisor
wants: the PINN solves the local EEDF at a given E/N, not the full
coupled fluid PDE system.

**Critical: the PINN negative result (`tel_model/PURE_PINN_NEGATIVE_RESULT.md`)
does NOT apply here.** That negative result was about solving the full
2D species + energy PDE system with a PINN — stiff Arrhenius chemistry,
second-derivative autograd instability, multi-scale loss competition.
Boltzmann-equation-only PINN is well-documented in the literature
(Kim 2023, Kawaguchi 2022) and the v2 code already has the
infrastructure.

**Next work**:

- Train the PINN on the Tier-1 BOLSIG+ grid (168 points × 53 channels).
  Training infrastructure exists (`BoltzmannPINNTrainer` class); needs a
  weights-checkpoint file.
- Benchmark inference speed vs bolos two-term solver. Expected ~100×
  speedup.
- Plug into the per-cell Boltzmann call from step 4.

---

## Immediate recommendation

Start **today** with D1, D2, D3 (three config-only sweeps, ~30 min total
wall-clock, fully parallelisable). While those run, read
`src/dtpm/modules/m11_plasma_chemistry.py` in detail to locate the exact
insertion point for D4.

By end-of-day tomorrow, have D4 working on a single Mettler point and
know whether Boltzmann rates shift the outer-wafer residual. That single
result determines whether Phase 1 wraps (step 3 complete) or whether the
residual is driven by something other than rate-coefficient fidelity.

## Supervisor decisions (2026-04-19)

1. ~~γ_Al composition-dependent fit~~ — **REJECTED**. γ_Al stays single-
   valued. The composition-insensitivity residual is a documented
   limitation, not a tunable.
2. **R_coil canonical value** — **SWEEP COMPLETED 2026-04-19**. Supervisor
   confirmed exact value is unknown; D3 executed the discovery sweep
   across R_coil ∈ {0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0} Ω. Finding:
   R_coil is a clean power-efficiency knob (η = 0.79–0.97) but does not
   meaningfully move [F]_c at the wafer (±5%) or the radial F-drop
   (±0.15%). Retain R_coil = 0.8 Ω as nominal; quote {0.5, 4.0} Ω as
   systematic uncertainty band in §5.5.1 of the report.
3. **Spatial surrogate (step 3b)** — still open. Recommend deferring
   re-training until after step 4 / 5 lands the canonical electron
   treatment (one training run, not two).

## Deliverable to supervisor after D1 + D3

Updated memo documenting:
- D1 (bias-off vs bias-on at Mettler's 90% SF6 centre) contribution to
  the absolute-magnitude residual
- D3 (R_coil sweep) — chosen canonical R_coil and the corresponding
  operating voltages (Vrms, Vpeak, Ipeak) and F-drop values at each sweep
  point
- All Mettler citations aligned with `manifests/12_mettler_citation_corrections.md`

## D3 radial-shape-invariance finding — scope implications for D4/D5

**Finding (2026-04-19)**: The centre-to-edge F-drop is invariant under
R_coil (66.5–66.8% across an 8× range in coil loss). This means the
radial profile shape is **not** set by the inductive-coupling physics;
it is set by neutral-transport + wall-loss balance. R_coil moves the
absolute [F]_c by ±5% without changing the shape.

Implications for the remaining diagnostic work:

- **D4 (tier-2 PINN Boltzmann rates)** will change the k_diss, k_iz,
  k_att values and therefore the absolute [F]_c magnitude, but is
  **unlikely to move the radial shape** — the shape comes from
  neutral-transport physics the PINN does not touch. Expectation for
  D4: it closes some or all of the ~55% absolute-magnitude gap but
  leaves the F-drop near 67%.
- **D5 (previously reserved for γ_Al composition fit, now freed since
  that was rejected)** should be re-scoped to a **neutral-transport
  sensitivity sweep** — vary the effective neutral diffusion
  coefficient and/or the SF_x wall-recombination coefficients at
  fixed γ_Al to see whether the radial shape is accessible through
  transport physics. This is the remaining lever if we want to match
  Mettler's 74% centre-to-edge drop exactly (model currently predicts
  67%, off by 7 percentage points — not a crisis but the main
  narrative gap in the radial fit).

Reserve this scope change for supervisor sign-off before committing
D5 to code.
