# ML Baseline Comparison

Same LXCat dataset (pinn_dataset_lxcat_v3, 221 cases), same val split.

| Method | N train | nF RMSE | nSF6 RMSE | Train time |
|---|---|---|---|---|
| Polynomial regression (degree 3) + Ridge | 120,696 | 0.05100 | 0.03912 | 0.4s |
| Polynomial regression (degree 4) + Ridge | 120,696 | 0.03941 | 0.01959 | 4.2s |
| Gaussian Process (RBF kernel) | 3,000 | 0.00629 | 0.00225 | 8049.2s |
| surrogate_lxcat_v3 (5-ens, no reg) | 120,696 | 0.01120 | 0.00810 | ~1200s |
| surrogate_v4 (5-ens, legacy, phys reg) | 120,696 | 0.00290 | 0.00270 | ~3700s |
