# Ablation Study: Training Recipe Components

Dataset: pinn_dataset_lxcat_v3 (221 LXCat cases)
Architecture: Fourier + 4x128 GELU MLP (same as v4)
3 seeds per experiment

| Experiment | Bias Init | Phys Reg | Epochs | nF RMSE | nSF6 RMSE | Time |
|---|---|---|---|---|---|---|
| none | N | N | 1500 | 0.01830+/-0.00282 | 0.01557+/-0.00275 | 18708s |
| bias_only | Y | N | 1500 | 0.00314+/-0.00043 | 0.00352+/-0.00093 | 34334s |
| reg_only | N | Y | 1500 | 0.01888+/-0.00338 | 0.01541+/-0.00257 | 12392s |
| epochs_only | N | N | 2000 | 0.01610+/-0.00293 | 0.01340+/-0.00121 | 7178s |
| all_three | Y | Y | 2000 | 0.00385+/-0.00071 | 0.00355+/-0.00089 | 5765s |

## Relative improvement vs baseline (no recipe)

- **bias_only**: +82.8% nF improvement
- **reg_only**: -3.2% nF improvement
- **epochs_only**: +12.0% nF improvement
- **all_three**: +79.0% nF improvement
