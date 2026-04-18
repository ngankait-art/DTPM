# LXCat vs Legacy Dataset Diagnosis

## 1. Target Dynamic Range

| Metric | Legacy | LXCat | Ratio |
|---|---|---|---|
| log10(nF) range | 2.351 | 2.335 | 0.99x |
| log10(nF) std | 0.5402 | 0.5354 | 0.99x |
| log10(nSF6) range | 2.284 | 2.287 | 1.00x |
| log10(nSF6) std | 0.3499 | 0.3484 | 1.00x |
| Te range (eV) | 1.53 | 1.53 | 1.00x |
| Te std (eV) | 0.224 | 0.240 | 1.07x |
| ne log-range | 3.40 | 3.66 | 1.08x |

## 2. Spatial Gradient Sharpness (P90)

| Gradient | Legacy P90 | LXCat P90 | Ratio |
|---|---|---|---|
| lnF radial | 10.251 | 10.194 | 0.99x |
| lnF axial | 9.604 | 9.527 | 0.99x |
| lnSF6 radial | 3.537 | 3.455 | --- |
| lnSF6 axial | 6.147 | 6.020 | --- |

## 3. Cross-Correlations (mean across cases)

| Correlation | Legacy | LXCat | Shift |
|---|---|---|---|
| Te vs lnF | 0.774 | 0.772 | -0.002 |
| Te vs lnSF6 | -0.420 | -0.389 | +0.030 |
| ne vs lnF | 0.571 | 0.573 | +0.001 |
| ne vs lnSF6 | -0.802 | -0.807 | --- |

## 4. Regime Separation (F-ratio = between-var / within-var)

| Grouping | Legacy nF F-ratio | LXCat nF F-ratio |
|---|---|---|
| By power | 0.563 | 0.560 |
| By pressure | 1.740 | 1.727 |
| By Ar fraction | 0.028 | 0.027 |

## 5. Per-Case Variance

| Metric | Legacy | LXCat |
|---|---|---|
| lnF spatial range (mean) | 1.462 | 1.447 |
| lnSF6 spatial range (mean) | 0.402 | 0.390 |
| lnF case-mean std | 0.1626 | 0.1624 |
| lnSF6 case-mean std | 0.3161 | 0.3167 |

## 6. Architectural Implications

Based on the diagnosis above, the key differences that affect learnability:

1. **Dynamic range**: How much narrower/wider are the LXCat targets?
2. **Gradient sharpness**: Are LXCat spatial profiles sharper (harder to represent)?
3. **Cross-correlations**: Does LXCat create stronger coupling (harder nonlinearity)?
4. **Regime structure**: Does LXCat create clearer regime separation (warranting gating)?
5. **Per-case variance**: Is the inter-case variability (signal the surrogate must capture) smaller in LXCat (lower SNR)?
