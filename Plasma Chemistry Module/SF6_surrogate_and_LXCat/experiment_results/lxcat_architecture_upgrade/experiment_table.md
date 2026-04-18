# LXCat Architecture Upgrade — Experiment Results

## Reference baselines
- **surrogate_v4** (legacy): nF RMSE = 0.0029, nSF6 RMSE = 0.0027
- **surrogate_lxcat_v3** (LXCat baseline): nF RMSE = 0.0112, nSF6 RMSE = 0.0081

## Experiment table

| Experiment | Params | Epochs | Phys Reg | nF RMSE (mean+/-std) | nSF6 RMSE (mean+/-std) | nF gap to v4 | nF improv vs v3 |
|---|---|---|---|---|---|---|---|
| E0_baseline | 82,818 | 1500 | N | 0.01901+/-0.00393 | 0.01580+/-0.00236 | 6.6x | -69.7% |
| E1_v4_recipe | 82,818 | 2000 | Y | 0.00377+/-0.00056 | 0.00338+/-0.00108 | 1.3x | +66.3% |
| E2_wider_deeper | 395,522 | 2000 | Y | 0.00390+/-0.00099 | 0.00319+/-0.00070 | 1.3x | +65.2% |
| E3_separate_heads | 82,690 | 2000 | Y | 0.00210+/-0.00015 | 0.00155+/-0.00062 | 0.7x | +81.3% |
| E4_residual | 214,914 | 2000 | Y | 0.00244+/-0.00066 | 0.00210+/-0.00073 | 0.8x | +78.2% |
| E5_enhanced_features | 82,818 | 2000 | Y | 0.00957+/-0.00180 | 0.00606+/-0.00138 | 3.3x | +14.5% |
| E6_huber_loss | 82,818 | 2000 | Y | 0.00394+/-0.00083 | 0.00324+/-0.00088 | 1.4x | +64.8% |

## Best experiment: E3_separate_heads
- nF RMSE: 0.00210 (vs v3 baseline 0.0112)
- nSF6 RMSE: 0.00155 (vs v3 baseline 0.0081)
- Improvement over v3: 81.3% nF, 80.8% nSF6
- Remaining gap to v4: 0.7x nF, 0.6x nSF6
