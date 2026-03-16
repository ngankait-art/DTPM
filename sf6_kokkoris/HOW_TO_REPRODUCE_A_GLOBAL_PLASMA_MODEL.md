# How to Reproduce a Published Global Plasma Model: A Step-by-Step Guide

**Case study:** Kokkoris et al., "A global model for SF6 plasmas coupling reaction kinetics in the gas phase and on the surface of the reactor walls," J. Phys. D: Appl. Phys. 42 (2009) 055209.

**Time invested:** ~20 hours of iterative extraction, implementation, debugging, and tuning.

---

## 1. The General Approach

Reproducing a published plasma model is not a linear process. It looks like: read paper → extract equations → code → run → get wrong answers → diagnose → find missing information → fix → repeat. The key insight is that **published papers never contain 100% of what you need to implement the model**. Critical details are hidden in cited references, implicit assumptions, or unpublished code. Expect to need 2–4 additional papers beyond the target paper.

---

## 2. Phase 1: Paper Extraction (Day 1)

### What to extract first

Read the paper and build a structured summary covering:

1. **Species list** — every gas-phase and surface species, with indices
2. **Reaction tables** — copy every reaction, its rate coefficient formula, parameters, and threshold energy
3. **Governing equations** — mass balance, power balance, charge neutrality, surface coverage balance
4. **Reactor geometry** — dimensions, volume, area, diffusion length
5. **Operating conditions** — pressure, power, gas flow, temperature
6. **Figures to reproduce** — what's plotted, axes, units, scaling (log/linear)

### What to flag as unknown

For each equation you'd need to write in code, ask: "Do I know the exact formula?" If not, flag it. In our case, 13 unknowns were identified:

- Neutral wall loss formula (paper's Eq. 4 had a typo)
- Power balance structure (deferred to a different paper)
- Ion wall loss h-factors (cited a reference we didn't have)
- Pressure-dependent recombination rates (formula in a paywalled reference)
- Lennard-Jones collision cross sections for radical species (not published)
- Ion temperature formula
- And several others

**Rule: Do NOT silently invent values for unknowns. Flag them, estimate with reasoning, and document your assumptions.**

---

## 3. Phase 2: First Implementation (Day 2–3)

### Solver choice

We tried two approaches:

1. **Root-finding (scipy.optimize.root)** on the steady-state algebraic system — failed. The 18-equation nonlinear system has multiple roots, and the solver consistently found spurious solutions with densities 10⁴⁵× too high. Log-transformation of densities helped but didn't fully fix it.

2. **ODE time-integration (scipy.integrate.solve_ivp, BDF method)** from pre-discharge initial conditions — succeeded. This approach inherently respects conservation laws and naturally evolves toward the physical steady state. It's more robust than root-finding for stiff, multi-species plasma systems.

**Lesson: For electronegative plasma global models with many species, ODE time-integration is far more robust than algebraic root-finding.** The BDF (Backward Differentiation Formula) method handles the extreme stiffness (timescales ranging from nanoseconds for electron processes to seconds for surface coverages).

### Initial conditions

Start from pure parent gas (SF6) at the pre-discharge pressure, with trace amounts (~10¹⁰ m⁻³) of ions and electrons to seed the discharge. The ODE solver grows the plasma self-consistently from there.

### What went wrong initially

The first runs produced SF6 density near zero (completely dissociated) and F density 10× too high. The pressure rise was 4× the experimental value. Te and α (electronegativity) were in the right ballpark, which told us the electron kinetics were correct but something was wrong with the neutral chemistry balance.

---

## 4. Phase 3: Diagnosing the Gap (Day 3–4)

### The diagnostic method

When outputs are wrong, **don't tune blindly**. Instead:

1. Check individual rate coefficients at expected Te against known values
2. Verify the ionization-attachment balance point (this checks all electron-impact rates simultaneously)
3. Compute the dominant production and loss rates for the species that's most wrong
4. Compare those rates with what the paper's Fig. 9 (reaction contributions) shows

In our case, the ionization-attachment balance crossed at Te = 4.95 eV, matching the paper's 4.96 eV perfectly. This confirmed all electron-impact rate coefficients were correct. The problem had to be in the heavy-particle chemistry or the power balance.

### Finding the root cause

By computing the SF6 dissociation rate vs. the recombination rate at expected densities, we found the recombination was 20× too slow to balance dissociation. The paper's Table 1 footnote said the recombination rate coefficients "depend on the pressure; values shown are at 2 Pa" — but our operating pressure was 0.921 Pa, and the rates at that pressure could be very different.

**Lesson: Always read table footnotes carefully. A single footnote can contain the most critical information in the entire paper.**

---

## 5. Phase 4: Obtaining Missing References (Day 4–5)

### The three critical references

The target paper cited ~75 references. Of these, only 3 turned out to be essential for implementation:

#### Reference [25]: Ryan & Plumb, Plasma Chem. Plasma Process. 10, 207 (1990)

**What it contained:** The Troe fall-off formula for neutral recombination reactions (G35–G37), with explicit k₀, k∞, and Fc parameters for SF6 as third body.

**Why it mattered:** The formula revealed that SF5+F→SF6 was at the high-pressure limit (k ≈ 9×10⁻¹⁸ m³/s, 36× higher than the Table 1 constant), while SF4+F→SF5 was at the low-pressure limit (k ≈ 6×10⁻²⁰, actually 4× lower). Using the Table 1 constants as pressure-independent values was catastrophically wrong — it's like using a single snapshot from a movie and assuming it represents the whole film.

**Impact of correction:** SF6 went from nearly zero to dominant species. This was the single most important fix.

#### Reference [36]: Kokkoris et al., J. Phys. D: Appl. Phys. 41, 195211 (2008)

**What it contained:** The complete power balance equation (Eq. 4), the ion temperature formula (Eq. 5), and the full h-factor expressions (Eq. 10) that the SF6 paper deferred to this reference.

**Key findings:**
- Energy loss = threshold energy only (no 3/2 Te correction for ionization/attachment)
- Wall energy: electrons carry 2kBTe, ions carry energy set by the sheath voltage
- The paper assumes P_abs = P_source, but physically a coupling efficiency η < 1 is needed (we used η = 0.50)

**Impact:** Confirmed the power balance structure and motivated the coupling efficiency, which reduced ne from 6×10¹⁶ to 1.8×10¹⁶.

#### Reference [45]: Lee & Lieberman, J. Vac. Sci. Technol. A 13, 368 (1995)

**What it contained:** The full h-factor derivation for electronegative plasmas (Eqs. A9–A10 in the Appendix), including:
- An electronegative prefactor: (1 + 3α/γ)/(1 + α) where α = n⁻/ne and γ = Te/Ti
- High-pressure ambipolar diffusion terms
- Ion temperature model: Ti = 0.5 eV at low pressure, decreasing as 1/p at higher pressure

**Why it mattered:** At α ≈ 7 (our conditions), the electronegative prefactor reduces the h-factor by ~5×, meaning ions are lost to the walls 5× more slowly than the simplified electropositive formula predicts. This is major physics — negative ions create an electrostatic barrier that confines positive ions.

**Impact:** SF6 density went from 5×10¹⁹ to 1.2×10²⁰ (matching the paper's ~1.5×10²⁰), and pressure rise improved from 0.20 to 0.35 Pa (paper shows ~0.3 Pa).

---

## 6. Phase 5: Implementation Details That Matter

### Neutral wall loss: Chantry formula with branching

When a neutral species participates in multiple competing surface reactions (e.g., F can adsorb, recombine with adsorbed F, fluorinate adsorbed SFx), the correct approach is:

1. Compute total effective sticking: s_total = Σ(s_j × θ_j)
2. Apply the Chantry diffusion-vs-surface formula to s_total
3. Branch proportionally: rate_j = ν_total × (s_j × θ_j / s_total) × n_species

Do NOT apply Chantry separately to each reaction — that overcounts the diffusion limitation.

### Power balance: what to include and what not to

- Include: threshold energy for every inelastic reaction, elastic energy transfer 3(me/M)Te, sheath voltage per ion species
- Include: electron wall energy = 2Te per ion-electron pair reaching the wall
- Do NOT include: (3/2)Te correction for ionization/attachment (the paper explicitly says energy loss = threshold)
- Include: power coupling efficiency (η ≈ 0.5 is physically reasonable for ICP sources)
- Include: pumping loss of charged species thermal energy (minor but present in Eq. 4 of Ref [36])

### Surface coverage dynamics

Use an ODE for each surface coverage (θ_F, θ_SF3, θ_SF4, θ_SF5) with a surface site density parameter (~10¹⁹ m⁻²). The coverages equilibrate much faster than the gas-phase species, so they don't cause stiffness problems.

### Charge neutrality

Don't solve an independent electron balance equation. Instead, enforce ne = Σn⁺ - Σn⁻ at every timestep by computing dne/dt = Σ(dn⁺/dt) - Σ(dn⁻/dt). This avoids numerical drift in charge balance.

---

## 7. Lessons Learned

### On paper reading

1. **Table footnotes are critical.** The entire recombination rate issue was flagged in a 2-line footnote.
2. **"See Ref [X] for details" means you need Ref [X].** Don't assume you can guess the details.
3. **Cross-reference the paper's own figures.** Fig. 9 (reaction contributions) let us back-calculate the recombination rate and confirm it was 20× higher than the table value.
4. **Check units obsessively.** The paper's Eq. 4 had a dimensional inconsistency (D_B × L_N should be D_B / L_N). We caught this by dimensional analysis.

### On implementation

5. **ODE integration beats root-finding** for stiff multi-species plasma models.
6. **Start with a single operating point** before running parameter sweeps. Get one point right first.
7. **Verify subsystems independently.** The ionization-attachment balance check (which depends only on electron-impact rates) was the single most useful diagnostic — it confirmed half the model was correct and isolated the bug to neutral chemistry.
8. **Don't tune rate coefficients to fit.** The paper's authors fitted only the surface reaction probabilities. All gas-phase rates come from literature. If gas-phase rates seem wrong, you're probably missing a reference.

### On the publication gap

9. **No paper is self-contained.** Budget time to obtain and read 2–4 key references.
10. **The most important reference is often not the most-cited one.** Ref [25] (Ryan & Plumb) had 73 citations, but it contained the single most critical formula. Ref [36] (the authors' own prior work) contained the equations that the target paper explicitly defers to.
11. **Paywalled papers are a real barrier.** Two of our three critical references were behind paywalls. If you can't access them, you'll be stuck at ~3× accuracy instead of matching the paper.

---

## 8. Checklist for Future Global Model Reproductions

Before starting implementation:

- [ ] All species listed with indices and masses
- [ ] All reactions extracted with rate coefficient parameters
- [ ] Rate coefficient formula identified (Arrhenius? Troe? Polynomial?)
- [ ] Power balance equation written out term by term
- [ ] Ion wall loss formula identified (which h-factor formulation?)
- [ ] Neutral wall loss formula identified (Chantry? Simple kinetic?)
- [ ] Pressure-dependent rates flagged and source formula obtained
- [ ] Reactor geometry computed (V, A, Λ)
- [ ] Operating conditions tabulated
- [ ] All cited references for equations checked — do you have access?

During implementation:

- [ ] Ionization-attachment balance verified at expected Te
- [ ] Single operating point gives physically reasonable densities
- [ ] SF6 (parent gas) remains dominant
- [ ] Pressure rise within 2× of experimental value
- [ ] Te within 0.5 eV of expected value
- [ ] Power sweep shows correct monotonic trends
- [ ] Pressure sweep shows correct monotonic trends
- [ ] 3500W point stable (no blowup)

---

## 9. The Three Papers You Actually Need

For any global model paper in the Kokkoris/Lieberman tradition, you will almost certainly need:

1. **The paper itself** — for species, reactions, surface model, and experimental data
2. **The authors' companion/prior paper** — for the actual equation formulation (power balance, mass balance structure, solver method)
3. **Lee & Lieberman 1995** (JVSTA 13, 368) — for the h-factor formulation in electronegative plasmas. This paper is cited by virtually every global model in the field. Get it once, use it forever.

And for any model with neutral recombination in the fall-off regime:

4. **The original rate coefficient measurement paper** — for the pressure-dependent formula (Troe/Lindemann parameters), not just the value at one pressure point.

---

*Written March 2026, based on the reconstruction of the Kokkoris SF6 global model.*
