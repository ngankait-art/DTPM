# Mettler validation gap report for Phase-1 DTPM plasma simulation report

## Section A: Factual citation errors confirmed / disputed

1. **Confirmed: the report misstates Mettler’s diagnostic for the TEL benchmark.**  
Your report says Mettler’s TEL measurements were done with **“BBAS and OES”** in §4.10, but the dissertation describes the TEL fluorine-density work using **W/Al radical probes**, with actinometry used for calibration / comparison, and the key TEL spatial figures (4.14, 4.17, 4.18) are explicitly probe-based.

2. **Confirmed: the report appears to collapse two different Mettler TEL benchmarks into one narrative.**  
In the report, Fig. 7.10 is presented as a **700 W, 10 mTorr** wafer radial benchmark with five Mettler points, while Mettler’s key absolute radial wafer comparison in Fig. 4.17 is at **1000 W ICP, 10 mTorr**, with **100 sccm total flow**, varying SF6/Ar ratio, and optional **200 W wafer bias**. So the report is mixing a real Mettler radial-profile benchmark with a different operating point than the dissertation’s main absolute radial wafer dataset.

3. **Confirmed: “Mettler 74% drop” is being used too generically.**  
The report repeatedly uses a single **74%** reference as if it were the generic TEL benchmark line, but the dissertation’s TEL comparisons are condition-specific: some are ICP-center actinometry / probe comparisons, some are wafer-radial radical-probe profiles, some include bias-on/off silicon etch rate overlays. Using a single 74% line in sensitivity plots and summary tables is acceptable only if you clearly say it comes from the **normalized wafer-radial profile fit**, not from all TEL datasets.

4. **Disputed / weaker than the original text-only flag: the unit-conversion issue is not obviously wrong from the report pages inspected.**  
The report’s relevant fluorine plots are labeled in **cm^-3**, while the dissertation captions and text commonly state **#/m^3**. That is a place where an error could easily occur, but from the visible report pages alone there is **not yet a smoking-gun factor-of-10^6 mistake**. This still merits a direct point-by-point overlay on the plotted values.

## Section B: Operating-condition mismatches confirmed / disputed

1. **Confirmed mismatch: report Fig. 7.10 vs Mettler TEL radial wafer data.**  
Report Fig. 7.10 is captioned **“Normalised radial [F] profile at the wafer (700 W, 10 mTorr)”** and says it overlays the five-point Mettler radical-probe data. But Mettler’s explicit radial wafer figure with absolute [F] and Si etch rate correlation is Fig. 4.17 at **1000 W ICP, 10 mTorr, 100 sccm total flow**, with **bias on/off** cases. Unless you intentionally extracted only the normalized shape from a different condition and state that explicitly, the current caption is not like-for-like.

2. **Confirmed mismatch: report baseline is pure SF6, while the key Mettler TEL figures are SF6/Ar mixture studies.**  
Your report’s reference operating point is repeatedly described as **700 W, 10 mTorr, pure SF6**. Mettler’s TEL figures 4.9 and 4.17 are explicitly at **100 sccm total flow with varying SF6/Ar ratio (balance Ar)**. That is a material mismatch.

3. **Confirmed mismatch: the report’s initial-validation narrative uses 700 W + prescribed η = 0.43, but Mettler’s key wafer-radial benchmark for direct spatial comparison is 1000 W and condition-specific.**  
The report’s §4.10 says the initial formulation benchmark target is **10 mTorr, 700 W, η = 0.43** and a 74% drop. That does not line up cleanly with the dissertation’s Fig. 4.17 wafer-radial dataset.

4. **Likely mismatch: bias handling.**  
Mettler’s Fig. 4.13 is specifically about **bias-on vs bias-off** effects on tungsten etch rate, and Fig. 4.17 carries **bias-on/off** silicon etch profiles. The report’s Mettler comparison does not appear to distinguish bias state in the radial wafer benchmark and instead uses a single 74% reference line. That is not like-for-like if the dissertation figure being borrowed includes both bias branches.

5. **Axis-scale / wrong-curve plausibility check.**  
There is **not firm evidence** that the plotted Mettler overlay in report Fig. 7.10 comes from the wrong curve entirely. The report itself says the five-point normalized profile is matched within **5–8%**, and its residual discussion says the inner wafer is within **±10%** while the edge is overpredicted by **15–27%**. That pattern is plausible for using the correct normalized radial dataset but the wrong operating-condition context. Current judgment: **probably the right normalized shape, probably the wrong caption / provenance / condition framing.**

## Section C: Missing benchmarks — priority-ordered list

1. **Highest priority missing benchmark: Mettler Fig. 4.17 (absolute radial [F] + Si etch rate, 30% and 90% SF6, bias on/off).**  
The report has a normalized wafer-radial [F] comparison, but it does **not** show a direct like-for-like absolute radial comparison against Mettler’s Fig. 4.17, nor the paired silicon etch-rate profile.  
**One-sentence missing-comparison summary:** “At 1000 W, 10 mTorr, 100 sccm total flow, the model should be compared directly against Mettler’s absolute wafer-radial fluorine-density profiles for 30% and 90% SF6, with separate bias-on/off branches, and against the matched silicon etch-rate profiles to test whether the predicted [F](r) reproduces the observed etch nonuniformity.”

2. **Next: Mettler Fig. 4.14 (normalized radial [F] + cubic fit).**  
The report effectively uses this as a validation motif, but it does not explicitly quote the dissertation’s cubic-fit function or show the actual five-point experimental markers against that fit.  
**One-sentence missing-comparison summary:** “The report should include a direct overlay of the model’s normalized wafer-radial [F] profile against Mettler’s Fig. 4.14 five-point probe data and its cubic fit, reporting pointwise percent errors at the measured radii.”

3. **Next: Mettler Fig. 4.18 (Si etch probability vs F flux).**  
The report currently stops at fluorine-density benchmarking; it does not propagate the model into the downstream kinetics observable that Mettler used to interpret silicon etching.  
**One-sentence missing-comparison summary:** “Using the modeled fluorine flux at the wafer, the report should compare predicted silicon etch probability–vs–F-flux trends against Mettler’s Fig. 4.18 to test whether the model reproduces the dissertation’s conclusion that spatially resolved probe-based probabilities are lower and more consistent than actinometry-based values.”

4. **Next: Mettler Fig. 4.9 (absolute [F] / etch rate vs SF6 flow).**  
The report has an absolute-[F]-vs-power benchmark, but not the TEL **flow-sweep** benchmark.  
**One-sentence missing-comparison summary:** “The model should be run over SF6 flow rate at fixed total flow and compared against Mettler’s Fig. 4.9 ICP-center fluorine-density and probe-etch-rate curves at 20 and 40 mTorr to test whether it captures the onset of high-density saturation / transport limitation.”

5. **Lower, but still useful: Mettler Fig. 4.13 (bias effect on etch rate).**  
The current report does not isolate the effect of wafer bias in the benchmark story.  
**One-sentence missing-comparison summary:** “A separate comparison should test whether adding wafer bias changes the modeled radical-driven etch observable in the same qualitative direction as Mettler’s Fig. 4.13, where bias increases measured tungsten etch rate across the temperature range.”

## Section D: Anything else unexpected

1. **The numerical spot-check values are not the same as the rough approximate set quoted in the request.**  
Evaluating  
`y(r) = 1.01032 - 0.01847*r^2 + 7.139e-4*r^3`  
gives approximately:
- `r = 0` -> **1.0103**
- `r = 2` -> **0.9422**
- `r = 4` -> **0.7605**
- `r = 6` -> **0.4996**
- `r = 8` -> **0.1938**

So the quoted “about” values are directionally right but noticeably loose at 4, 6, and 8 cm.

2. **Visual pass/fail on the report radial curve:**  
Based on the report’s own stated residuals, the curve likely passes the **inner** Mettler points within about ±10%, but **not all five points**. The report itself says the outer-half wafer points are overpredicted by **15–27%**. So the answer for the ±10% criterion is **no**.

3. **Additional Mettler figures probably underused:**
- **Fig. 4.12** is worth including because it anchors the tungsten etch-probability calibration chain that underlies the radical-probe-to-density conversion.
- **Fig. 4.16** is also useful because the Arrhenius activation energy gives a kinetic sanity check on the W/F chemistry used to infer density from probe response.

4. **Most important practical takeaway before Stages A + B:**  
The biggest issue is **not** that the report’s normalized radial profile is wildly wrong; it is that the benchmark provenance is muddled. The report seems to have borrowed a real Mettler TEL radial shape, but described it with the wrong **diagnostic**, wrong **operating point**, and too-generic **74%** shorthand. That is the first thing to correct in `METTLER_CORRECTIONS.md`.
