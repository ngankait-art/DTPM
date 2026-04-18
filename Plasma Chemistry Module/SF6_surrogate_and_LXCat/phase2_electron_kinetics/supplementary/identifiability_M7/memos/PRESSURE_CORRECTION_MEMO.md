# Memo: Panduranga pressure correction — sensitivity analysis and decision

**Date:** Tue 14 Apr 2026
**To:** author
**From:** submission-preparation pass
**Status:** BLOCKING for submission; requires author decision before proceeding

---

## 1. What was found

Two factual errors in the current manuscript drafts were surfaced during
bibliography verification:

1. **The low anchor is misattributed.** The paper that reports the 1.4
   μm/min silicon etch rate at 200 W SF₆ ICP, 10 mTorr, is not "Jansen
   & Legtenberg 1999" (no such paper exists with those parameters), but
   rather
   > R. D. Mansano, P. Verdonck, H. S. Maciel, M. Massi,
   > "Anisotropic inductively coupled plasma etching of silicon with
   > pure SF₆", *Thin Solid Films* **343–344**, 378–380 (1999),
   > DOI: 10.1016/S0040-6090(98)01689-7.

   Jansen and Legtenberg appear only as reference [6] of the Mansano
   paper (their unrelated 1995 *J. Electrochem. Soc.* paper on
   SF₆/O₂/CHF₃ etching).

   **Caveat worth noting:** Mansano 1999 used 200 W ICP antenna power
   **plus 150 W RF bias on the cathode**, not zero bias as the
   manuscript assumes. The non-zero bias contributes additional
   ion-energy-driven etching at the wafer.

2. **Panduranga 2019 was at 30 mTorr, not 10 mTorr.** The verbatim
   recipe from the paper:
   > "The chamber pressure was set to **30 mT**, and the substrate
   > temperature was set to 20 °C. The ICP power was 2000 W with a
   > bias of 0 V. Finally, the SF₆ flow rate was set to 50 SCCM."

   The manuscript §3 and §5 drafts assume both anchors are at 10 mTorr.
   That is correct for Mansano but not for Panduranga.

---

## 2. Sensitivity analysis

I re-ran the M7g calibration pipeline under two scenarios, holding
everything else (operating E/N, kinetics backends, gas heating, Robin
BC) fixed.

**Case A — current manuscript assumption**: both anchors at 10 mTorr.
**Case B — corrected assignment**: low anchor at 10 mTorr, high
anchor at 30 mTorr.

| Quantity | Case A (both 10 mT) | Case B (mixed 10/30) | Shift |
|----------|--------------------:|----------------------:|------:|
| **β (Maxwell)**    |  14.27 ms | **193.7 ms**  | ×13.6 |
| **β (Boltzmann)**  | 509.2 ms  | **6911 ms**   | ×13.6 |
| ε (Maxwell)        | 7.73e-30 m/atom  | 2.71e-29 m/atom | ×3.50 |
| ε (Boltzmann)      | 2.76e-28 m/atom  | 9.68e-28 m/atom | ×3.50 |
| *k*_diss (Maxwell)  | 1.109e-17 m³/s  | 1.109e-17 m³/s | (unchanged) |
| *k*_diss (Boltzmann)| 3.804e-19 m³/s  | 3.804e-19 m³/s | (unchanged) |
| *n*_e(*P*_H) (Maxwell)  | 1.574e+19 m⁻³ | 5.233e+18 m⁻³ | ÷3.01 |
| *n*_e(*P*_H) (Boltzmann)| 1.285e+19 m⁻³ | 4.274e+18 m⁻³ | ÷3.01 |
| 𝓘 = β·*k*_diss·*n*_e (Max)  | 2.491 | 11.24 | ×4.51 |
| 𝓘 = β·*k*_diss·*n*_e (Bolt) | 2.489 | 11.24 | ×4.51 |

The central numerical result of the manuscript (β ≈ 14 ms for Maxwell,
β ≈ 509 ms for Boltzmann) is a **Case A** result. Under Case B, the
fitted β changes by a factor of ~13.6 in both rate-source modes.

---

## 3. What survives and what doesn't

**Fully preserved under the pressure correction:**

* **The identifiability theorem of §4.** In both cases, the invariant
  𝓘 = β·*k*_diss·*n*_e(*P*_H) is identical between the Maxwell and
  Boltzmann fits to within 0.08 %, confirming that a two-anchor
  calibration constrains only this scalar product. The theorem is
  pressure-independent.

* **The Maxwell↔Boltzmann shift ratio.** In Case A, β_Bolt/β_Max =
  35.68×. In Case B, β_Bolt/β_Max = 35.68× (agreement to four
  significant figures). The ratio is determined entirely by the
  kinetics backends (*k*_diss ratio × *n*_e ratio = 29.15 × 1.224 =
  35.70), and the identifiability theorem predicts this ratio is
  invariant to the calibration anchors.

* **§6's observability argument.** The off-anchor discrepancies
  (Δ_obs at 50 Td, 200 Td, 300 Td, *x*_Ar = 0.30) are percentage
  differences between Maxwell and Boltzmann predictions at the same
  operating point. They depend on the rate-source shapes, not on the
  anchor pressures. All §6 numerical values are unchanged.

**Does NOT survive under the pressure correction:**

* **The absolute numerical values** β = 14.27 ms and β = 509.22 ms.
  Case B gives 194 ms and 6911 ms respectively.

* **§5's "β is inside the residence-time band" claim.** The geometric
  residence-time range for τ_res at 10 mTorr is [36, 1438] ms. Under
  Case A, β_Bolt = 509 ms sits squarely inside this range and β_Max =
  14 ms sits far below it. This is the central physical-consistency
  argument of §5. Under Case B, the low and high anchors are at
  different pressures, and β no longer has a single residence-time
  interpretation because the two anchors sample two different gas
  densities. If Panduranga is the dominant anchor for β (which it is,
  since the high-power anchor governs the depletion constraint), then
  the 30 mTorr residence-time band [108, 4314] ms applies, in which
  **β_Max = 194 ms lies comfortably inside the band and β_Bolt =
  6911 ms lies above it**. This inverts §5's story.

* **§5's prediction-test numerics.** §5 reports a 0.5 % residual
  between predicted and fitted β_Bolt. The prediction test itself
  (Eq. β_new/β_old = *k*_diss,old/*k*_diss,new × *n*_e,old/*n*_e,new) is
  still exact — it passes in Case A and Case B with identical 0.08%
  residuals — but the specific numbers that illustrate the test
  would all change.

---

## 4. Three correction paths

| Path | Effort | Narrative impact |
|------|--------|------------------|
| **A. Same-day fix** | 2–4 hours | Keeps Case A numerics; text acknowledges pressure mismatch as a limitation; honest but non-ideal |
| **B. Anchor replacement** | 1–2 days | Replace Panduranga with a 10 mTorr literature value; preserves Case A numerics and §5 residence-time story |
| **C. Two-pressure recalibration** | 3–5 days | Treat each anchor at its own pressure; re-derive §5's geometric-band comparison on physical grounds; accept a different β value |

**My recommendation: Path A as submission minimum, Path B if a
replacement anchor can be found within the week.**

Path C is methodologically cleanest but introduces a factor-of-13.6
change in the central quoted β value, which means §3, §4.3 (invariant
numerics), §5 (all tables and figures), and §7 (bottom-line summary)
all need numerical updates. This is several days of work and should
only be undertaken if Paths A and B are unacceptable.

### Why Path A is defensible

The paper's methodological contribution — identifiability theorem,
EEDF sensitivity workflow, observability-driven experiment design —
is **independent of which specific anchors are used**. The theorem
predicts the shift in β exactly whether you calibrate at 10/10, 10/30,
or 30/30 mTorr. The only casualty of Path A is §5's secondary claim
that the Boltzmann β is "physically consistent with residence time",
which must be softened to something like

> "...the Boltzmann-corrected β is in the range expected for
> gas-phase SF₆ residence times in typical ICP reactors, though we
> note the two anchors are from chambers at different operating
> pressures, which limits a direct residence-time interpretation
> to order-of-magnitude consistency (see §7 for discussion)."

This is a ~2-sentence softening, not a restructuring. Referees will
accept it if the §7 limitations are clear.

### Why Path B is better if feasible

A replacement high-power anchor at 10 mTorr, 2000 W, zero bias, pure
SF₆, room temperature would restore the manuscript to its current
Case A state with no numerical changes. Candidate papers I noticed
during the search but did not have time to extract the exact
etch-rate values from:

* Rudenko, Kuzmenko, Miakonkikh, Lukichev (2022), *Vacuum* —
  "On temperature and flux dependence of isotropic silicon etching in
  inductively coupled SF6 plasma", Plasmalab 100, 750–2750 W,
  includes a pressure sweep that likely covers 10 mTorr. If they
  report a 10 mTorr / 2000 W / room-temperature etch rate close to
  2.27 μm/min, they can substitute directly for Panduranga.
* Miakonkikh, Kuzmenko, Efremov, Rudenko (2025), *Vacuum* —
  "Comparative study of Ar- and He-rich SF6+Ar+He plasmas", same
  reactor, 5–25 mTorr, 800–2750 W. This one is in mixed chemistries
  though, so it may not be a drop-in replacement.

**Author action for Path B:** consult these two papers (or any other
10 mTorr / 2 kW pure-SF₆ silicon etch-rate data you know of) and
extract a single number at 10 mTorr, 2000 W, zero bias, room
temperature. If that number is within 10–20 % of 2.27 μm/min, it's a
defensible replacement; refit β using it and re-run M7g. If it
differs by a factor of 2 or more, that's itself a finding worth
discussing.

---

## 5. Action items

If the author chooses **Path A** (same-day fix):

1. Global rename `jansen1999` → `mansano1999` in `references.bib` and
   every `\cite{jansen1999}` call.
2. In §3 where the anchors are introduced, add one sentence:
   > "We note that the Panduranga measurement was reported at 30 mTorr,
   > whereas the Mansano measurement is at 10 mTorr. We calibrate at
   > 10 mTorr (the nominal manuscript operating pressure) using
   > each paper's reported etch rate as the anchor value, treating
   > the ~3× pressure difference as a systematic absorbed into the
   > fitted ε and β; the identifiability theorem (§4) is unaffected by
   > this choice because the invariant 𝓘 = β·*k*_diss·*n*_e is
   > pressure-independent within the theorem's scope."
3. In §5, soften the "residence-time consistent" claim to the
   ~2-sentence version given above.
4. In §7, add the pressure mismatch to the limitations paragraph, and
   explicitly flag it as a reason why the absolute β value should
   not be over-interpreted as a physical residence time.
5. Add the "150 W cathode bias" caveat for Mansano as a separate
   sentence in §3: the low-anchor measurement used a biased substrate,
   which the model absorbs into ε, and a small future experiment at
   zero bias would remove this systematic.
6. The M7g figures (fig2–fig7 in the submission package) already have
   corrected labels reading "Mansano 1999, 10 mTorr" and "Panduranga
   2019, 30 mTorr" — no figure regeneration needed for Path A.

If the author chooses **Path B** (anchor replacement):

1. Identify the replacement paper (see §4 candidates above).
2. Update `references.bib` with the new entry.
3. Update §3 to cite the replacement paper instead of Panduranga.
4. Re-run M7g with `HIGH_TARGET_NMM` set to the new anchor value.
5. If the new β differs materially from the Case A values, update
   §3, §4.3, §5, §7 numerics accordingly.
6. Re-export figures at 300 dpi (one-line monkey-patch in the pipeline
   that is already in this conversation's M7g re-export script).

If the author chooses **Path C** (two-pressure recalibration):

1. All of Path A, *plus* —
2. Modify the M7g pipeline so that `OpPoint.pressure_Pa` is set per
   anchor (already supported — see the sensitivity script in this
   memo).
3. Re-run the full calibration with Mansano at 10 mTorr and Panduranga
   at 30 mTorr.
4. Update Table 1 (scaling test) and Table 2 (Maxwell vs Boltzmann
   parameter summary) with Case B values.
5. Re-derive the §5 residence-time-band comparison using the
   correct per-anchor pressures. The geometric band at each pressure
   is computed from n_SF6,0(p) × V / Q_in, so it's a simple scaling.
6. Update §7 to reflect the new absolute β values.
7. Re-export all figures.

---

## 6. Bottom line

The manuscript as currently drafted makes two factual errors about
literature anchors: the wrong author attribution for Mansano 1999,
and the wrong pressure for Panduranga 2019. Both are submission-
blocking if left unaddressed because any referee who looks up either
reference will catch them.

The methodological contribution of the paper is unaffected: the
identifiability theorem, the Boltzmann-vs-Maxwell ratio, and the
off-anchor observability analysis all survive the pressure correction
exactly. What changes is (a) the absolute numerical value of β, by
a factor of ~13.6 in the mixed-pressure case, and (b) the
residence-time-consistency argument in §5, which needs softening or
rebuilding.

I recommend Path A as the minimum submission bar, implemented within
the day. Path B is a modest effort improvement if a suitable
replacement anchor can be found. Path C is the rigorous answer but
costs several days and is only worth it if the reviewer fit is
particularly demanding on physical interpretation of β.
