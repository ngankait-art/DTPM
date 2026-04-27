# Ablation Study: Training Recipe Components

Dataset: ml_dataset / legacy
Architecture: Fourier + 4x128 GELU MLP
3 seeds per experiment

| Experiment | Bias Init | Phys Reg | Epochs | nF RMSE | nSF6 RMSE | Time |
|---|---|---|---|---|---|---|
| none | N | N | 1500 | 0.02183+/-0.00073 | 0.02108+/-0.00289 | 59263s |
| bias_only | Y | N | 1500 | 0.00601+/-0.00065 | 0.00566+/-0.00063 | 59253s |
| reg_only | N | Y | 1500 | 0.02217+/-0.00142 | 0.02131+/-0.00284 | 99962s |
| epochs_only | N | N | 2000 | 0.02052+/-0.00031 | 0.02008+/-0.00189 | 81152s |
| all_three | Y | Y | 2000 | 0.00612+/-0.00051 | 0.00597+/-0.00043 | 109903s |

## Relative improvement vs baseline (none)

- **bias_only**: +72.5% nF improvement
- **reg_only**: -1.6% nF improvement
- **epochs_only**: +6.0% nF improvement
- **all_three**: +72.0% nF improvement


Wall-clock: 109907.8 s
