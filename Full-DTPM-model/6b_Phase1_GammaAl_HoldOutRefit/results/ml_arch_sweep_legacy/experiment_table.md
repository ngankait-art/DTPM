# Architecture Sweep — Recovered Results (legacy)

Recovered from `_run.log` after partial-completion crash.

| Experiment | Epochs | Phys Reg | nF RMSE | nSF6 RMSE | Wall |
|---|---|---|---|---|---|
| E0_baseline | 1500 | N | 0.02227+/-0.00202 | 0.02180+/-0.00398 | 2.6 h |
| E1_v4_recipe | 2000 | Y | 0.00606+/-0.00043 | 0.00614+/-0.00042 | 2.2 h |
| E2_wider_deeper | 2000 | Y | 0.00626+/-0.00039 | 0.00594+/-0.00050 | 15.3 h |
| E3_separate_heads | 2000 | Y | 0.00542+/-0.00069 | 0.00476+/-0.00031 | 1.5 h |
| E4_residual | 2000 | Y | 0.00530+/-0.00094 | 0.00503+/-0.00054 | 4.9 h |
| E5_enhanced_features | 2000 | Y | 0.01363+/-0.00161 | 0.00953+/-0.00059 | 1.2 h |

**Winner**: `E4_residual` (nF RMSE = 0.00530)

E6_huber_loss skipped — see note in JSON.
