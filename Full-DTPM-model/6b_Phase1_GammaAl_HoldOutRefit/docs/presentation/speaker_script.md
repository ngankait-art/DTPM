# Phase 1 Presentation — Speaker Script

## Slide 1: Title
[~30 sec]
Good morning/afternoon. I'm Muhammad Abdelghany, and this is joint work with Zachariah Ngan at the Illinois Plasma Institute. Today I'll present our Digital Twin Plasma Model — specifically the Phase 1 milestone where we achieved self-consistent coupling between electromagnetic field computation and SF6 plasma chemistry in a TEL ICP etcher. The key result: we predict a 78% fluorine centre-to-edge drop from first-principles physics, within 4 percentage points of the experimental measurement of 74%.

## Slide 2: Outline
[~15 sec]
The talk is in four parts: first, the electromagnetic pipeline that computes fields from RF circuit parameters to particle kinetics; second, the SF6/Ar plasma chemistry validated in our Stage 10 model; third, the Phase 1 integration that couples EM with chemistry self-consistently; and finally, the roadmap for future phases.

## Slide 3: ICP Etchers in Semiconductor Fabrication
[~2 min]
ICP reactors dominate anisotropic etching at advanced nodes below 10 nm. The TEL ICP etcher we model has a 6-inch wafer stage with dual RF — a 40 MHz high-frequency source at 700 watts for plasma generation, and a 13 MHz low-frequency bias for ion energy control. The ICP source is a quartz tube with radius 38 mm surrounded by a multi-turn coil.

The central challenge is predicting etch uniformity — how uniform is the radical flux arriving at the wafer? Mettler's 2025 dissertation at UIUC measured a 74% centre-to-edge drop in fluorine atom density using radical probes inserted into the processing chamber. Our goal is to predict this drop from physics, replacing the prescribed plasma parameters that previous models relied on.

## Slide 4: DTPM Pipeline Architecture
[~2 min]
The DTPM is organized as a modular pipeline of 18 modules. Each module is a pure function that takes a shared state dictionary and a configuration object, performs its computation, and returns results that get merged into the state. This design makes every intermediate result accessible for analysis and ML training.

The existing EM pipeline — modules M01 through M08 — handles RF circuit analysis, electrostatics, magnetostatics, FDTD electromagnetic wave propagation, particle-in-cell kinetics, and energy analysis. These were validated in prior work.

For Phase 1, we added three new modules shown in red: M06c, a cylindrical FDTD solver; M10, which computes power deposition from the EM fields; and M11, the Picard coupling loop that iterates between EM-derived quantities and the plasma chemistry until convergence. The dashed arrow shows the feedback: chemistry results update the plasma conductivity, which feeds back into the power calculation.

## Slide 5: M01 — RF Circuit Analysis
[~1 min]
M01 is intentionally simple — it converts engineering parameters into physical quantities. For 700 watts into 50 ohms at 40 MHz, we get a peak voltage of 264.6 volts and peak current of 5.29 amperes. These feed directly into the magnetostatic and FDTD solvers downstream.

## Slide 6: M02 — Electrostatic Field Solver
[~1.5 min]
M02 solves the Laplace equation for the electrostatic potential using Gauss-Seidel SOR with Red-Black ordering. The coil cross-sections are held at the peak voltage as Dirichlet boundary conditions, and the reactor walls are grounded. The electric field is then computed from central differences. The Red-Black ordering enables vectorisation in numpy, giving about a 50x speedup over the naive scalar implementation.

## Slide 7: M04 — Magnetostatic Field
[~1.5 min]
For the magnetostatic field from the coil, we use the exact analytical solution via complete elliptic integrals of the first and second kind. This is more accurate than the Biot-Savart segment approach because there is no discretisation error from approximating the circular coil as a polygon. The key parameter k-squared involves the geometric relationship between the field point and the coil position. The total field is obtained by superposition over all N coil turns.

## Slide 8: M05-M06 — FDTD EM Simulation
[~2 min]
For the time-dependent electromagnetic fields, we solve Maxwell's curl equations in 2D TM mode using the Yee grid finite-difference time-domain method. The Yee grid staggers the E and H fields by half a cell in both space and time, which naturally satisfies the divergence-free conditions. The CFL stability condition bounds the time step, and we use Mur's first-order absorbing boundary conditions at the domain edges to prevent spurious reflections.

## Slide 9: M07 — PIC Simulation
[~2 min]
The particle-in-cell module uses the Boris leap-frog algorithm, which is the gold standard for particle pushing in electromagnetic fields. The Boris algorithm has two key properties: it preserves the particle energy exactly in a pure magnetic field — we measured conservation errors of order 10 to the minus 15 — and it is second-order accurate in time. The charge deposition and field gathering use Cloud-in-Cell bilinear weighting, which reduces the noise compared to nearest-grid-point assignment.

## Slide 10: 0D Global Model
[~2 min]
Moving to the chemistry side. The 0D global model solves the particle and power balance to determine the electron temperature Te and density ne. The particle balance equates ionization to attachment plus wall loss, and is solved for Te using Brent's method. The power balance then gives ne. The electronegativity alpha — the ratio of negative ion to electron density — comes from a quadratic equation involving the attachment-to-recombination balance. For pure SF6 at 700 watts and 10 milliTorr, we get Te = 2.3 eV, ne = 9 times 10 to the 17 per cubic meter, and alpha = 1.5.

## Slide 11: 54-Reaction Mechanism
[~1.5 min]
The chemistry uses the Lallement 2009 reaction set with over 50 reactions organized into dissociation, ionization, attachment, recombination, and Penning channels. A critical detail is the Troe fall-off formula for the neutral recombination reactions — at 10 milliTorr, these reactions are 5 to 180 times slower than the high-pressure limit, which fundamentally changes the SF6 regeneration rate.

## Slide 12: Wall Surface Chemistry
[~1.5 min]
The wall chemistry follows Kokkoris 2009 with a five-channel mechanism. The critical parameter is gamma_Al — the fluorine recombination probability on aluminium surfaces — which is 0.18 and was calibrated once in Stage 10 to match the Mettler 74% drop. This is the single calibrated parameter in the entire model. The differential recombination between quartz walls at 0.001 and aluminium walls at 0.18 is the dominant driver of the centre-to-edge fluorine drop.

## Slide 13: 2D Species Transport
[~1.5 min]
The 2D transport solver discretises the diffusion-reaction equation in cylindrical coordinates on the masked T-shaped domain. The TEL reactor has an ICP source 38 mm in radius connected to a processing chamber 105 mm in radius through a 2 mm aperture. On a 50 by 80 mesh, 1669 of 4000 cells are active. The sparse matrix assembly uses Robin boundary conditions at each material interface, and the system is solved directly with SuperLU.

## Slide 14: Stage 10 Validation
[~1 min]
The Stage 10 model — using prescribed eta of 0.43 and a Bessel-cosine ne profile — matches the Mettler experimental data at 74% fluorine drop. This is our baseline. The question for Phase 1 is: can we replace these prescriptions with self-consistent physics and still get the right answer?

## Slide 15: Cylindrical FDTD (M06c)
[~2 min]
To answer that, we developed M06c — a cylindrical TE-mode FDTD solver. Unlike the Cartesian TM mode in M05-M06, the ICP drives an azimuthal electric field E-theta through the time-varying magnetic flux from the coil. The Maxwell equations in cylindrical coordinates include the critical 1-over-r term in the H_z update, which requires L'Hopital's rule at the axis. We also model the 3 mm quartz wall as a dielectric region with epsilon_r = 3.8. The source is normalised as a current density — I-peak divided by the cell area — with a Gaussian-modulated sinusoidal envelope. Full vectorisation with numpy gives a 46x speedup, bringing the 50-by-80 grid FDTD from 23 minutes down to 30 seconds.

## Slide 16: FDTD Results
[~1 min]
The FDTD at 40 MHz produces peak E-theta of 292 V/m instantaneous and 456 V/m RMS. The field is concentrated at the skin depth near the quartz wall, with clear hot spots at each of the 6 coil positions. The exponential decay from the wall into the bulk plasma is clearly visible.

## Slide 17: Power Deposition
[~1.5 min]
Module M10 computes the power deposition as half-sigma times E-squared. The plasma conductivity depends on ne and the momentum-transfer collision frequency. Since the FDTD gives the correct spatial shape but not the absolute magnitude — because we don't self-consistently model the coil-plasma impedance — we normalise the E-field so that the total absorbed power matches eta times P_rf. This scale factor is recomputed each Picard iteration.

## Slide 18: Ionization-Source Diffusion
[~2 min]
This is the key physics innovation. Instead of prescribing ne as a Bessel-cosine eigenmode, we solve the steady-state electron continuity equation with an explicit ionization source from the FDTD power profile. The source S_iz equals P divided by epsilon_T times e — this tells us where new electrons are being created. The crucial point is that electrons are PRODUCED at the skin depth, which is near the quartz wall, but they DIFFUSE INWARD via ambipolar transport. The resulting ne profile is centre-peaked — like the Bessel eigenmode — even though the source is wall-peaked. This is a single sparse linear solve, not an iterative eigenvalue problem, which makes it both faster and more physically correct.

## Slide 19: Picard Coupling Loop
[~1.5 min]
Module M11 orchestrates the coupling through Picard iteration. Each iteration: compute power deposition from the FDTD shape, solve for Te from local power balance, solve for ne from the ambipolar PDE, then run the inner chemistry loop for F and SF6 transport. Under-relaxation with w = 0.3 ensures stability. The loop converges when the relative change in ne drops below 2%.

## Slide 20: Cross-Section Results
[~1 min]
Here is the full-reactor cross-section showing the fluorine density on a log scale. The T-shaped geometry is clearly visible. F is highest in the ICP source where ne and Te drive dissociation, drops sharply through the aperture, and shows the characteristic centre-to-edge variation in the processing region that we're trying to predict.

## Slide 21: Wafer Profiles
[~1.5 min]
The wafer-level radial profiles are our primary validation diagnostic. Panel (a) shows the absolute fluorine density dropping from 5.5 times 10 to the 13 at the centre to about 1.2 times 10 to the 13 at the edge — a 78% drop. Panel (b) shows the normalised profile with the 74% Mettler level marked in green. The curve crosses the Mettler line at about r = 80 mm. Panel (c) confirms the ne profile peaks at r = 20 mm in the ICP source, and panel (d) shows SF6 increasing from centre to edge as it is consumed more where ne is higher.

## Slide 22: Results Summary
[~1 min]
The summary table compares Phase 1 with Stage 10 and experiment. Our self-consistent model predicts 77.8% fluorine drop versus the experimental 74% — within 4 percentage points, with zero calibration beyond the gamma_Al that was already set in Stage 10. The coupling efficiency eta converges to 0.43, matching the literature estimate. The total computation time is 112 seconds.

## Slide 23: Discussion
[~2 min]
The 3.8 percentage point gap between 77.8 and 74 is not a code bug — it reflects the physics difference between our self-consistent ne profile and the prescribed Bessel-cosine. Three physics improvements could close this gap naturally, without any recalibration: first, an electron energy transport PDE that would smooth Te toward the axis; second, re-running the FDTD self-consistently with the updated plasma conductivity; and third, EN-corrected h-factors for the ion wall loss in the electronegative SF6 plasma. All three are physics additions planned for Phase 2.

## Slide 24: Future Roadmap
[~1.5 min]
Phase 2 will address electron kinetics through a tiered approach: BOLSIG+ tables for offline rate computation, a PINN-based Boltzmann solver for ML acceleration, and PIC-MCC for one-time validation. Phase 3 will train ML surrogates for real-time prediction from parameter sweeps. Phase 4 migrates the remaining modules M09 through M18 for ion transport, sheath modelling, and etch profile prediction.

## Slide 25: Summary
[~1 min]
To summarise: we built a modular EM pipeline from RF circuit to energy analysis, integrated it with validated SF6 chemistry on the TEL reactor geometry, and achieved self-consistent coupling through Picard iteration. The key result is a 77.8% fluorine drop predicted from first principles, within 4 percentage points of the Mettler experimental value of 74%. The code runs in under 2 minutes on a standard laptop. Thank you.

## Slide 26: References
[~15 sec]
Here are the key references. I'm happy to take questions.
