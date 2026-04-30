# Ablation Study: Training Recipe Components

Dataset: ml_dataset / lxcat
Architecture: Fourier + 4x128 GELU MLP
3 seeds per experiment

| Experiment | Bias Init | Phys Reg | Epochs | nF RMSE | nSF6 RMSE | Time |
|---|---|---|---|---|---|---|
| reg_only | N | Y | 1500 | 0.01948+/-0.00199 | 0.01940+/-0.00306 | 31479s |
| all_three | Y | Y | 2000 | 0.00557+/-0.00051 | 0.00297+/-0.00051 | 40693s |

## Relative improvement vs baseline (none)

- **all_three**: +71.4% nF improvement


Wall-clock: 40703.8 s
