# Speaker notes — slide-by-slide

## Slide 1 — Title
- Scope: Track A (kinetics). Track B (reactor) is separate.
- Every number traces to shipped code/outputs.

## Slide 2 — Problem
- Attachment resonance at 0–1 eV: electron sink at any bulk Te
- Ionisation threshold at 15.8 eV: demands hot tail
- Closure cliff at ~3 eV: below = no discharge
- Key question: which physics layer closes the gap?

## Slide 3 — Hypothesis elimination
- M1: Boltzmann k_iz = 0.24× Maxwellian at 1000 Td. Below floor at <150 Td. Maxwellian **overestimates**.
- M2: Te ceiling 2.48 eV. Gap = 0.52 eV. No redistribution helps.
- M3: Reflector → 49% fewer wall events, only 3.7% Te gain. Inelastic dominates.
- M4: 2× power → Te 1.85→3.74, ionisation 0→35 events. **Power is the answer.**

## Slide 4 — Tiers
- Data flows downward only. No tier modifies upstream.
- Decision gates prevent unnecessary work.

## Slide 5 — Rate ratios
- Median k_att ratio: 1.11. k_iz at 1000 Td: 0.24.
- Below 150 Td (operating point): k_iz below numerical floor.
- The Maxwellian tail is heavier → overestimates ionisation.

## Slide 6 — M5 surrogate
- 3×96 GELU MLP. 19,397 params.
- Te: 0.44%, k_att: 1.7%, k_iz: 8.4%. All <10%.
- Production API: `get_rates_pinn(E/N, x_Ar, p)`.

## Slide 7 — Non-local PINN
- Phase A: cell-by-cell Tier 2 lookup (local baseline)
- Phase B: Gaussian kernel teacher (non-local targets)
- PINN: ~72k params, learns ΔTe, Δlog k_iz, Δlog k_att
- Full mirrored TEL domain shown

## Slide 8 — Multicase
- 8 operating points: 3–20 mTorr, 0–50% Ar
- Case C held out (different x_Ar)
- Err/std ≈ 0.23

## Slide 9 — Pressure closure
- Hold-out B = hardest (5 mTorr extrapolation)
- v2→v3→v4: 1.282 → 1.250 → 1.114 (13% improvement)
- Std: 0.034 → 0.008 (more reproducible)

## Slide 10 — Pure PINN vs hybrid
- Pure PINN: correct autograd, correct BCs, correct residuals. Diverged.
- Root cause: stiff chemistry, 2nd derivatives, multi-scale losses.
- Data-only fitting works (23,000× convergence) → fields are representable.
- Pivot: supervised on FD outputs + light physics regularisation.
- This is the correct research conclusion.

## Slide 11 — Surrogate validation
- 33 held-out cases. RMSE 0.003 both species.
- Error bars = ensemble + MC dropout uncertainty.

## Slide 12 — Evolution
- v1→v4: 44× RMSE improvement. 19%→0.24% median error.
- Key: targeted enrichment around failure corners (133 new cases).
- v3→v4: 87% RMSE improvement.

## Slide 13 — Uncertainty
- Ensemble (5) + MC dropout (20 passes). Post-hoc calibration.
- 68% coverage at target. Cal scales: ~0.3 (predicted uncertainty was 3× too wide, corrected).

## Slide 14 — Validation
- Be direct about what is NOT done. Frame as scope, not defect.
- The kinetic closure is complete. Downstream coupling is next phase.

## Slide 15 — Future
- #1 priority: real DTPM solver callbacks.
- All else is refinement.

## Slide 16 — Takeaway
- Read as written. Pause after "power-limited."
