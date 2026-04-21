# Mettler Benchmarking Corrections — Phase-1 Report

**Applies to**: `Steps/5.Phase1_EM_Chemistry_Merged/docs/report/main.tex`

**Source of truth**: `Literature/Mettler Dissertation.pdf` (J.D. Mettler, UIUC 2025, 105 pp.).

These are the factual errors and operational-condition mismatches found in the current report after reading Mettler's Chapter 4 (TEL etcher data) end-to-end. Each entry gives location, problem, and proposed correction.

---

## E1. Diagnostic method is mis-described

**Location**: [main.tex:1500]
**Current text**:
> "…Mettler, who performed spatially resolved fluorine radical density measurements in a TEL ICP etcher using broadband absorption spectroscopy (BBAS) and optical emission spectroscopy (OES)."

**Problem**: Mettler did **not** use BBAS. His primary diagnostic was a pair of non-equilibrium **W/Al radical probes** (reactive tungsten + inert aluminum thermocouple-tipped pellets) cross-calibrated against **Ar/SF6 actinometry** in the ICP region.

**Correction**:
> "…Mettler, who performed spatially resolved fluorine radical density measurements in a TEL ICP etcher using W/Al non-equilibrium radical probes (etching-based thermal response) cross-calibrated against Ar/SF6 actinometry in the ICP region."

---

## E2. "Mettler Fig 4.5" reference is to a different reactor

**Location**: [main.tex:2370–2375] (caption of our Fig 7.28 and surrounding paragraph)
**Current text**:
> "…the four-panel absolute-[F] validation against the Mettler Fig. 4.5 dataset…"

**Problem**: Mettler's Fig 4.5 is his **PMIC (helicon source)** data — W/Al probe response + actinometry vs power at 50 mTorr. It is *not* the TEL etcher. Our Phase-1 reactor is a TEL ICP, so this comparison is inappropriate.

**Correction**: Mettler does not report a pure-power sweep in the TEL. The closest TEL benchmark is **Fig 4.9** (ICP-region [F] vs SF6 flow rate at 20 mTorr/600 W and 40 mTorr/700 W). Change caption to:
> "…the four-panel absolute-[F] validation against Mettler's TEL ICP-region data (Fig 4.9, 700 W / 40 mTorr branch). Note that Mettler's TEL dataset is a SF6-flow sweep at fixed power, not a power sweep; our power sweep is plotted on a comparable ordinate for qualitative trend comparison."

---

## E3. "74% drop" is oversold as a single number

**Locations**: [main.tex:1504], [main.tex:2135 caption], [main.tex:2208 caption], [main.tex:2212], [main.tex:2217 caption]
**Current text**: "Mettler reported a 74% drop in F atom density…"

**Problem**: Mettler's Fig 4.17 shows the drop ranges **67%–75%** depending on SF6/Ar composition and wafer bias. The 74% specifically corresponds to the **90% SF6, bias-off** condition at 1000 W / 10 mTorr. Our current report implies the 74% is universal and applies at 700 W.

**Correction**: Replace each instance with:
> "Mettler reported a centre-to-edge [F] drop ranging from 67% to 75%, depending on SF6/Ar mixture ratio. The 74% value corresponds specifically to the 70 sccm SF6 / 30 sccm Ar, 1000 W ICP, 10 mTorr, 200 W rf-bias condition (Mettler Fig 4.14 / 4.17)."

---

## E4. Operating-condition mismatch between our baseline and Mettler's baseline

**Location**: everywhere we state "700 W, 10 mTorr" as the validation target.

**Problem**: Mettler's spatially resolved data (the only data with radial [F] profiles) were taken at:
- **P_ICP = 1000 W** (not 700 W)
- **10 mTorr** ✓
- **70 sccm SF6 / 30 sccm Ar** (not pure or near-pure SF6)
- **200 W rf wafer bias applied** (we have no bias model)

**Correction**: Either (a) rerun the baseline at Mettler's actual conditions (1000 W / 70% SF6) and treat that as the validation case, or (b) clearly state that the 700 W / 10 mTorr / SF6-dominated case is our **design operating point**, while noting that the Mettler benchmark is at elevated power and 70% SF6. The present report reads as though 700 W matches Mettler's measurement conditions, which is false.

---

## E5. γ_Al calibration narrative

**Location**: [main.tex:1090]
**Current text**: "γ_Al = 0.18 … fitted to reproduce the experimentally observed 74% F density drop."

**Problem**: The 74% target is specific to the Fig 4.14 (90% SF6) condition. As stated, the calibration locks the model to a single composition.

**Correction**: Clarify that γ_Al = 0.18 was calibrated against the 90%-SF6 branch of Fig 4.14 / 4.17. Once the composition sweep validation (F4 in the validation-points doc) is run, cross-check whether γ_Al = 0.18 also reproduces the 30% SF6 branch's 67% drop, or whether a composition-dependent γ would be needed.

---

## E6. Implicit conflation of Helicon and TEL data

**Location**: throughout the report wherever "Mettler" is cited.

**Problem**: Mettler's thesis has two distinct experimental chambers:
- **Ch. 4.1 — Helicon / PMIC** (probe screening, Fig 4.1–4.5)
- **Ch. 4.2–4.3 — TEL etcher** (all TEL data, Fig 4.6–4.19)

We cite Mettler as if all figures are from the TEL. Specifically the "Fig 4.5" citation (E2 above) actually reaches into the Helicon data.

**Correction**: Add one clarifying sentence near first Mettler citation:
> "Mettler's thesis contains both Plasma-Materials Interaction Chamber / Helicon-source experiments (Ch. 4.1, used primarily for probe-material screening) and TEL ICP-etcher data (Ch. 4.2–4.3). Only the TEL data (Fig 4.6–4.19) are used for validation in the present work."

---

## E7. Missing citation to the TEL-specific kinetic regime bound

**Location**: our Limitations §5.5

**Problem**: Mettler identifies a kinetic-to-diffusion-limited transition at [F] ≳ 1.1×10²¹ /m³ = 1.1×10¹⁵ cm⁻³ (Fig 4.9 etch-rate saturation, Table 4.4 diffusion-limit estimate). Our model stays well below this value (a few ×10¹⁴ cm⁻³), which is **fortunate for validity** but currently unremarked.

**Correction**: Add to §5.5:
> "(L5) Kinetic regime only. All Phase-1 simulations produce centre-of-wafer [F] below Mettler's ~1.1×10²¹ /m³ diffusion-limited threshold (his Fig 4.9 and Table 4.4), so our chemistry predictions compare cleanly to his kinetic-regime probe measurements. Extension to higher densities (e.g. higher power or SF6-rich NF3 replacements) would require coupling a near-surface reaction-product transport equation."

---

## Summary Fix Order (priority descending)

1. **E1** (diagnostic) — single sentence, no ambiguity → easy fix now.
2. **E2** (Fig 4.5 → Fig 4.9) — re-phrase captions of Fig 7.28; no re-run needed.
3. **E3** (74% range) — string replace across 5 locations.
4. **E6** (Helicon vs TEL scope) — add one sentence.
5. **E5** (γ_Al calibration scope) — one sentence clarification.
6. **E7** (L5 kinetic regime) — one paragraph in Limitations.
7. **E4** (1000 W rerun) — requires new simulation + new figure; deferred to Stage B of the gap-analysis plan.

See also: `METTLER_VALIDATION_POINTS.md` (comprehensive validation targets) and the Phase-1 plan file `~/.claude/plans/mutable-swimming-chipmunk.md`.
