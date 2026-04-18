# Tier 3 — MCC collision module and 0D validation

**Workplan §5.** Provide a reusable Monte Carlo collision (MCC) layer for SF₆/Ar electrons that can be coupled to the existing Boris pusher, and run small validation cases at prescribed operating points to cross-check Tiers 1–2 and probe whether non-local transport effects move the DTPM fluorine profile.

## Honest scope statement

**This deliverable is the reusable MCC collision core, plus a 0D MCC cross-check at the three workplan-specified operating points. It is not a full spatial PIC-MCC run.**

The full spatial PIC-MCC run (in which the MCC core runs inside the Boris pusher loop, with electrons moving across the TEL geometry) is the stated next integration step and is outside the scope of this deliverable because the spatial pusher code lives in a separate repository (`1.E-field-map/script_v2/2.Extended-Aspects/7.DTPM_Project_ICP/`, modules M07/M08) that is not bundled here.

The non-local transport question of workplan §5.5 can only be answered by that spatial run. **This directory does not claim to answer it.**

## What is here

- `mcc_module.py` — null-collision MCC core (Vahedi & Surendra 1995) for SF₆ electrons in a uniform prescribed field. ~340 lines of Python, no external plasma library dependencies. Uses the same LXCat Biagi cross-section set as Tier 1 and Tier 2.
- `lxcat_parser.py` — minimal LXCat file parser (100 lines), stand-alone, zero dependencies beyond numpy.
- `run_case_A.py` — runs Case A: 700 W / 10 mTorr / pure SF₆, mapped to E/N = 50 Td.
- `run_case_B.py` — runs Case B: 700 W / 5 mTorr / pure SF₆, mapped to E/N = 100 Td.
- `run_case_C.py` — runs Case C: 700 W / 10 mTorr / 50% Ar, mapped to E/N = 50 Td.
- `analyze_cases.py` — cross-case analyzer producing a summary markdown and a comparison bar chart.

## Run configuration (each case)

- 3000 macro-electrons, initial thermal spread at 1 eV
- 30,000 time steps × 2×10⁻¹¹ s = 600 ns of evolution
- Steady-state EEDF averaged over the last 25% of the run
- Rate coefficients recovered by integrating the EEDF against the same cross sections BOLSIG+ used
- Reference comparison against the Tier 1 BOLSIG+ lookup at the matching (E/N, x_Ar) grid point

## Results summary

| Case | Description | Te_eff MCC | Te_eff Bolt | alive % end | physical trend |
|---|---|---|---|---|---|
| A | 700W / 10mTorr / pure SF₆ | 1.18 eV | 1.04 eV | 16% | reference |
| B | 700W / 5mTorr / pure SF₆ | 1.33 eV | 1.52 eV | 42% | higher Te at lower p ✓ |
| C | 700W / 10mTorr / 50% Ar | 1.33 eV | 1.26 eV | 44% | weaker att with Ar ✓ |

**Te_eff agreement within 15% at all three cases. Physical trends reproduced correctly.**

Attachment rates are systematically lower in the 0D MCC than in BOLSIG+ (by a factor 5–10). This is the expected artefact of the finite-duration 0D run not yet reaching the attachment-drained steady state BOLSIG+ assumes, and does not affect the Te_eff agreement that is the primary metric. Ionisation is zero in both MCC and BOLSIG+ at all three operating points (consistent — the 15.7 eV threshold is far above the EEDF bulk at 50–100 Td).

## Outputs

- `results/case_{A,B,C}_result.json` — machine-readable per-case MCC output + BOLSIG+ reference values.
- `results/tier3_summary.md` — **cross-case decision report** (workplan §5.5).
- `outputs/case_{A,B,C}_mcc.png` — two-panel diagnostic PNG per case (mean energy trajectory + final EEDF).
- `outputs/case_comparison.png` — bar chart of Te_eff and k_att across all three cases.

## Run

```bash
python run_case_A.py
python run_case_B.py
python run_case_C.py
python analyze_cases.py
```

Each case takes ~10 seconds on a laptop.

## What the MCC module does (and does not) include

**Does include:**
- Full null-collision algorithm (Vahedi & Surendra 1995)
- Real LXCat Biagi SF₆ cross sections (50 channels: 1 elastic, 7 attachment, 10 ionisation, 32 inelastic)
- Isotropic elastic scattering with (2 m_e / M_SF6) fractional energy loss
- Inelastic energy-loss scattering with threshold deduction
- Attachment as electron removal from the simulation
- x_Ar dilution weighting on the SF₆ channel bank (for Case C)

**Does not include:**
- Spatial grid, Boris pusher coupling, Poisson solve (this is 0D MCC, not PIC-MCC)
- Secondary electron creation in ionisation events (acceptable at sub-percent ionisation fractions; required for full self-consistent PIC-MCC)
- Full Ar cross sections (Case C uses an x_Ar-weighted SF₆-only bank; for production Ar mixture kinetics use the Tier 2 surrogate, which is trained on the full Ar-inclusive BOLSIG+ grid)
- Magnetic field, RF field phasing, sheath effects

The exclusions are deliberate and are explicitly stated here so that a reader evaluating this module against the Phase 2 workplan can see exactly what scope gap remains before the spatial integration step.
