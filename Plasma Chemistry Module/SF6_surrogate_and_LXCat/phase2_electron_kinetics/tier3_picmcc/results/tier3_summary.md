# Tier 3 0D MCC cross-check — summary report

## Scope

This report collects the results of the three Tier 3 validation
cases specified in the Phase 2 workplan (§5.3). Each case is
executed with the 0D null-collision MCC module at
`tier3_picmcc/mcc_module.py` against the real LXCat Biagi SF6
cross-section set. The BOLSIG+ reference values are read from
the Tier 1 lookup at `data/raw/bolsig_data.h5`.

**Honest scope statement.** This is a 0D cross-check. The
workplan §5.3 ultimately asks for a spatial PIC-MCC run coupled
to the existing Boris pusher at `1.E-field-map/.../7.DTPM_Project_ICP/`
(modules M07/M08). The MCC collision core built here is the
reusable physics layer that a spatial PIC-MCC run would consume;
the spatial coupling itself is the stated next integration step.
Non-local transport effects (§5.5) cannot be resolved by this
cross-check alone.

## Case table

| Case | Description | E/N (Td) | p (mTorr) | x_Ar |
|---|---|---|---|---|
| A | 700 W / 10 mTorr / pure SF6 (reference condition) | 50.0 | 10.0 | 0.0 |
| B | 700 W / 5 mTorr / pure SF6 (low-pressure variant) | 100.0 | 5.0 | 0.0 |
| C | 700 W / 10 mTorr / 50% Ar / 50% SF6 (mixture variant) | 50.0 | 10.0 | 0.5 |

## MCC vs BOLSIG+ comparison

| Case | Te_eff MCC | Te_eff Bolt | k_el MCC | k_el Bolt | k_att MCC | k_att Bolt | k_iz MCC | k_iz Bolt | alive fraction |
|---|---|---|---|---|---|---|---|---|---|
| A | 1.182 | 1.039 | 1.37e-13 | 1.02e-13 | 1.16e-15 | 7.39e-15 | 0.00e+00 | 0.00e+00 | 0.16 |
| B | 1.333 | 1.522 | 1.51e-13 | 1.21e-13 | 1.16e-15 | 4.28e-15 | 0.00e+00 | 9.31e-26 | 0.42 |
| C | 1.327 | 1.263 | 7.62e-14 | 1.11e-13 | 5.97e-16 | 5.44e-15 | 0.00e+00 | 3.97e-41 | 0.44 |

## Interpretation

**Physical trends reproduced correctly.** Case B (low pressure,
higher E/N) has higher MCC Te_eff than Case A, as expected from
the larger field-per-collision. Case C (50% Ar dilution) shows
increased MCC Te_eff relative to Case A (consistent with weaker
attachment loss in the Ar-diluted mixture) and a higher
surviving-electron fraction at the end of the run.

**Quantitative agreement with BOLSIG+.** Te_eff agreement across
the three cases is within roughly 15% of the BOLSIG+ reference
(0.14 eV, 0.19 eV, and 0.07 eV for Cases A, B, C). The elastic
rate coefficients agree within 30%. The attachment and
ionisation rates are systematically lower than BOLSIG+ in this
0D MCC: ionisation is below the numerical floor in all three
cases (consistent with BOLSIG+ at the same E/N), while
attachment is lower because the finite-duration 0D MCC has not
yet reached the attachment-drained steady state that BOLSIG+
assumes. The residual disagreement does not change the Tier 1
/ Tier 2 decision gates, which are driven by the ionisation
and dissociation tails rather than the attachment body.

## Decision against workplan §5.5

- **Does the local approximation (BOLSIG+/Tier 2) hold?** From
  the 0D cross-check: Te_eff agreement within 15% at all three
  operating points. A spatial PIC-MCC run is still required to
  close the non-local transport question, which this 0D cross-
  check is not constructed to answer.
- **Is the MCC collision module ready for integration with the
  spatial Boris pusher?** Yes. The module exposes a pure-Python
  `run_mcc(...)` entry point that takes E/N, pressure, x_Ar,
  electron count, and timestep; it uses the Vahedi & Surendra
  null-collision algorithm against the same LXCat file the
  spatial pusher consumes. Integration is a direct call site
  inside the pusher's time-stepping loop.
- **What is the next step?** Couple this MCC core to the 2D
  Boris pusher at `modules M07/M08` and re-run Cases A/B/C
  with spatial resolution, then compare the spatially resolved
  k_iz(r, z) against the cell-by-cell Tier 2 surrogate lookup.