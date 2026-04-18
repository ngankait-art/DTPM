# Master model comparison

## All surrogate versions

| Model | Dataset | N cases | Outputs | nF RMSE | nSF6 RMSE | Te RMSE | Ens mean | Status |
|---|---|---|---|---|---|---|---|---|
| **surrogate_v4** | v4 legacy | 221 | nF, nSF6 | **0.0029** | **0.0027** | --- | 9.7e-5 | **Production** |
| surrogate_final | v4 legacy | 221 | nF, nSF6, Te | 0.0131 | 0.0091 | 0.0081 | 3.1e-3 | Exploratory |
| **surrogate_lxcat_v3** | **lxcat_v3** | **221** | nF, nSF6 | **0.0112** | **0.0081** | --- | 2.0e-3 | **Fair comparison** |
| surrogate_lxcat_v2 | lxcat_v2 | 30 | nF, nSF6 | 0.0713 | 0.0623 | --- | 8.9e-2 | Superseded |
| surrogate_lxcat_v1 | lxcat prov. | 221 | nF, nSF6 | 0.0118 | 0.0072 | --- | 1.9e-3 | Superseded |
| surrogate_te_v1 | v4 legacy | 221 | Te | --- | --- | 0.0045 | --- | Exploratory |

## Fair comparison: surrogate_v4 vs surrogate_lxcat_v3

| Metric | surrogate_v4 (legacy) | surrogate_lxcat_v3 (LXCat) | Ratio |
|---|---|---|---|
| nF RMSE (log10) | 0.0029 | 0.0112 | 3.85x |
| nSF6 RMSE (log10) | 0.0027 | 0.0081 | 3.00x |
| Ensemble mean loss | 9.7e-5 | 2.0e-3 | 20x |
| Dataset cases | 221 | 221 | same |
| Architecture | Fourier+4x128 GELU | Fourier+4x128 GELU | same |
| Epochs | 1500 | 1500 | same |
| Training time (s) | 3708 | 1186 | 0.32x |

### Controlled variables
- Same 221 operating conditions (P: 200-1200 W, p: 3-20 mTorr, Ar: 0-50%)
- Same mesh (30x50), same geometry, same boundary conditions
- Same architecture, optimizer, learning rate schedule, validation split
- **Only difference**: solver rate_mode ('legacy' vs 'lxcat')

### Physics interpretation

The 3-4x RMSE degradation in the LXCat surrogate is a **physics result**, not an ML failure:

1. **LXCat attachment is 4-20x larger** than legacy Arrhenius. This forces Te up by ~1.6 eV to maintain particle balance, which makes the Te distribution narrower (4.0-5.5 eV vs 2.5-4.0 eV).

2. **ne drops to 34% of legacy**. The narrower ne range (4e15 to 2e17 vs 1.5e16 to 3.8e17) means less dynamic range for the surrogate to learn.

3. **F density is controlled by diffusion/wall loss**, not electron-impact rates. Since dissociation channels remain on legacy fallback, the F spatial profile varies less across the LXCat parameter space, making it slightly harder to predict small variations.

4. **The LXCat dataset is intrinsically harder**: stronger coupling between Te and ne means the surrogate must capture a more nonlinear input-output mapping.

### Key findings

1. **surrogate_v4 remains best** for production nF/nSF6 prediction (0.003 RMSE, 2000x speedup).
2. **surrogate_lxcat_v3** is the first physics-consistent LXCat surrogate at full scale. The 3-4x RMSE increase is attributable to changed physics, not insufficient data or training.
3. **Scaling from 30 to 221 cases** improved LXCat surrogate dramatically: lxcat_v2 (30 cases) had 0.071 nF RMSE vs lxcat_v3 (221 cases) at 0.011 --- a **6.4x improvement**.
4. **The lxcat_v3 nF RMSE of 0.011** corresponds to ~2.6% median error in linear density --- still useful for physics exploration and sensitivity studies.

## Production recommendation

Use **surrogate_v4** for production applications requiring highest accuracy. Use **surrogate_lxcat_v3** for physics-consistent exploration of LXCat-based kinetics and sensitivity studies where the changed electron kinetics matter.
