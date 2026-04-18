# Mettler (UIUC 2025) Validation Points for Phase-1 DTPM

**Source**: Jeremy D. Mettler, *"Spatially Resolved Probes for the Measurement of Fluorine Radicals"*, Ph.D. Dissertation, University of Illinois at Urbana-Champaign, 2025 (105 pages).

**Scope note**: Mettler used **two different reactors**:
- **PMIC / Helicon source** (CPMI) — probe-material screening and calibration experiments (Ch. 4.1). *Not applicable* to our TEL model.
- **TEL research-scale etcher (Tokyo Electron)** — all spatially resolved [F] data and the only experimental reference for this Phase-1 work (Ch. 4.2–4.3).

**Diagnostic**: W/Al non-equilibrium radical probes (reactive W pellet + inert Al pellet thermocouples) cross-calibrated against Ar/SF6 actinometry in the ICP region. *Not* BBAS and *not* OES alone.

---

## Part 1 — Validation Points We Can Compare Against NOW

### V1. Normalised radial [F] profile at the wafer (Mettler Fig 4.14, p. 70)
- **Conditions**: P_ICP = 1000 W, 70 sccm SF6 / 30 sccm Ar, p_total = 10 mTorr, 200 W rf wafer bias.
- **Data**: 5 radial points (0, 2, 4, 6, 8 cm) with normalised F density 1.00, 0.95, 0.81, 0.53, 0.25.
- **Analytic cubic fit**: `y(r) = 1.01032 − 0.01847 r² + 7.139×10⁻⁴ r³` with r in cm (R² = 0.997).
- **Centre-to-edge drop**: 74% (for this specific 70% SF6 / 30% Ar / bias-on condition).
- **How to compare**: run simulation at 1000 W / 10 mTorr / frac_Ar = 0.3, overlay normalised radial profile and the cubic fit on the same axes.

### V2. Absolute radial [F] + Si etch-rate profiles (Mettler Fig 4.17, p. 75)
- **Conditions**: P_ICP = 1000 W, 100 sccm total, 10 mTorr, with and without 200 W rf bias, at **two SF6/Ar ratios**: 30% SF6 and 90% SF6.
- **Data (approx. digitised)**:
  - 90% SF6, bias off: [F]_centre ≈ 2.5×10²⁰ /m³, [F]_edge ≈ 0.6×10²⁰ /m³ (75% drop).
  - 90% SF6, bias on: [F]_centre ≈ 4×10²⁰ /m³ (×1.6 enhancement from bias).
  - 30% SF6, bias off: [F]_centre ≈ 0.6×10²⁰ /m³, [F]_edge ≈ 0.2×10²⁰ /m³ (67% drop).
  - 30% SF6, bias on: ×2.15 at centre, ×1.6 at edge.
- **Composition range**: 67%–75% centre-to-edge drop (*not a single 74% number*).
- **How to compare**: overlay our 1000 W / 10 mTorr predictions for `frac_Ar = 0.1` (90% SF6) and `frac_Ar = 0.7` (30% SF6) at bias-off.

### V3. Absolute [F] in ICP region vs SF6 flow (Mettler Fig 4.9, p. 61)
- **Conditions**: 20 mTorr / 600 W and 40 mTorr / 700 W, SF6 flow 10–90 sccm, total flow 100 sccm.
- **Data at 90 sccm SF6**:
  - 20 mTorr / 600 W: [F]_ICP ≈ 1.0×10²¹ /m³ = 1.0×10¹⁵ cm⁻³.
  - 40 mTorr / 700 W: [F]_ICP ≈ 2.0×10²¹ /m³ = 2.0×10¹⁵ cm⁻³.
- **Kinetic-to-diffusion-limited transition**: above [F] ≈ 1.1×10²¹ /m³ the tungsten probe etch rate saturates (Fig 4.10).
- **How to compare**: sweep SF6 flow at 700 W / 40 mTorr; extract volume-averaged [F] in the ICP region.

### V4. Spectrometer / actinometry underestimation factor (Mettler Eq 4.2, p. 68; Fig 4.14)
- **Claim**: actinometry (line-integrated over chamber diameter) underestimates the centre [F] density by ~60% (n_0 = 2.5×10²⁰ vs n_act = 1.54×10²⁰ /m³ at Mettler's Fig 4.14 conditions).
- **Ratio**: n_centre / n_act ≈ 1.6.
- **How to compare**: compute our ratio n(r=0, z=wafer) / volume-averaged n_F and compare to 1.6. Requires no new runs; derive from any existing sweep point.

### V5. Si etch probability independence of F flux (Mettler Fig 4.18, p. 77)
- **Claim**: using locally resolved [F] densities (not actinometry), Si etch probability collapses to ε_Si ≈ 0.025–0.04, roughly independent of F flux. Actinometry-derived values scatter from 0.05 to 0.10.
- **How to compare**: **no direct prediction possible** without a Si-etching surface model, but we can cite as *support* for the quality of Mettler's probe data.

### V6. F gas temperature ≈ 300 °C in the ICP region (Mettler p. 64)
- **Claim**: from thermal response of the Al (inert) probe near the end of the ICP tube.
- **How to compare**: our model assumes T_gas ≈ 400 K ≈ 127 °C uniformly. Mettler's 300 °C in the source is significantly higher — should at minimum be cited as context for a *future* gas-energy-equation extension (Phase 2).

### V7. W etch kinetics (secondary, chemistry-independent)
- **Claim**: Mettler Fig 4.16 gives E_a = 0.068 eV for W etching in SF6 at T_W = 100–250 °C; Fig 4.15 gives ε_W ≈ 0.03 at T_W = 300 °C (matches Rosner et al.).
- **How to compare**: not a direct chemistry benchmark; we do not etch W. Relevant only as context for the probe calibration pipeline.

---

## Part 2 — Validation Points Achievable in FUTURE WORK Only

Deferred because they require extensions not present in Phase-1.

### F1. Effect of rf wafer bias on [F] (Mettler Fig 4.13 and 4.17 bias-on curves)
- **Observation**: 200 W rf bias increases [F] by ×1.6 (90% SF6) to ×2.15 (30% SF6) at centre.
- **Mechanism**: expanded plasma volume in the process chamber increases local electron-impact dissociation.
- **Missing capability**: our ICP-only EM solver has no wafer-bias sheath model. Requires a capacitive-sheath / ion-bombardment coupling — **Phase-2 scope**.

### F2. Electronegative-ambipolar radial broadening
- **Observation**: Mettler's 74% radial drop at 90% SF6 is significantly steeper than the 68% we predict. The gap attributable to the electronegative ambipolar correction factor `(1 + α) / (1 + αT_i/T_e)` omitted from our transport solver (see Limitation L1 in §5.5).
- **Missing capability**: implement α-dependent D_a in `m09_transport_2d.py`. Scheduled for **Phase-2**.

### F3. Diffusion-limited surface regime
- **Observation**: Mettler Fig 4.9 shows etch-rate saturation at [F] > 1.1×10²¹ /m³, indicative of a boundary-layer diffusion limit.
- **Missing capability**: we do not compute near-surface WF6/SiF4 transport. Requires a 1D boundary-layer model coupled to the wall. **Phase-2 / Phase-3**.

### F4. Composition sweep validation (Mettler 30% vs 90% SF6)
- **Status**: partially achievable now (V2). A complete sweep of frac_Ar from 0 to 0.9 would characterise how the chemistry transitions from SF6-dominated electronegative to Ar-dominated electropositive.
- **Missing capability**: a full composition sweep has not been run; requires ~10 additional sweep points. *Upgradeable to "now" if desired* — marked here because it is not in the current report.

### F5. E-to-H mode transition at low power
- **Observation**: Mettler does not explicitly comment on E/H but the 200 W / 50 mTorr probe traces (Fig 4.4, helicon) show very different F production at low power — consistent with being below the H-mode threshold.
- **Missing capability**: our Picard iteration converges to a single H-mode branch; no E-to-H model.

### F6. Multiple wafer-region materials + spatial wall temperature
- **Observation**: Mettler's TEL etcher has stainless-steel chamber walls, aluminum pumping port, and a silicon wafer — plus thermal gradients from the heated pumping region.
- **Missing capability**: our wall model uses one γ per material but a single wall temperature. A spatially varying T_wall would refine the F+F+wall recombination rate.

### F7. Cl radical chemistry (Mettler §4.3.3)
- **Observation**: Mettler tested Mo/Ti probes in Cl2/Ar plasmas and found TiO2 layer breakthrough.
- **Relevance**: *none* to Phase-1 SF6/Ar work. Would only matter if DTPM is extended to Cl2-based chemistries in Phase-3+.

---

## Part 3 — Current Phase-1 Errors to Correct (cross-reference)

See `/Users/muabdelghany/muhammad/workspace/scientific/research/PhD-2024/1. ICP Project/Steps/5.Phase1_EM_Chemistry_Merged/docs/METTLER_CORRECTIONS.md` for the list of factual errors in the current `main.tex` that must be fixed before the next revision.

---

## Part 4 — Canonical Mettler TEL Operating Conditions (Quick Reference)

| Purpose | Mettler source | Power | Pressure | SF6:Ar | Bias |
|---|---|---|---|---|---|
| Radial [F] norm | Fig 4.14 | 1000 W | 10 mTorr | 70:30 | 200 W |
| Radial [F] abs (90% SF6) | Fig 4.17 | 1000 W | 10 mTorr | 90:10 | 0 / 200 W |
| Radial [F] abs (30% SF6) | Fig 4.17 | 1000 W | 10 mTorr | 30:70 | 0 / 200 W |
| ICP-region flow sweep | Fig 4.9 (LO p) | 600 W | 20 mTorr | 10–90 sccm SF6 / balance Ar | off |
| ICP-region flow sweep | Fig 4.9 (HI p) | 700 W | 40 mTorr | 10–90 sccm SF6 / balance Ar | off |

Use these conditions — not "700 W / 10 mTorr / no bias" which is our invented baseline — whenever performing a direct one-to-one comparison with Mettler data.
