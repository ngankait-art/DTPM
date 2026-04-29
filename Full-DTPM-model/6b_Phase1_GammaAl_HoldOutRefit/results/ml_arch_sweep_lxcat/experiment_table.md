# LXCat Architecture Upgrade — Experiment Results

## Reference baselines
- **surrogate_v4** (legacy): nF RMSE = 0.0029, nSF6 RMSE = 0.0027
- **surrogate_lxcat_v3** (LXCat baseline): nF RMSE = 0.0112, nSF6 RMSE = 0.0081

## Experiment table

| Experiment | Params | Epochs | Phys Reg | nF RMSE (mean+/-std) | nSF6 RMSE (mean+/-std) | nF gap to v4 | nF improv vs v3 |
|---|---|---|---|---|---|---|---|
| E0_baseline | 82,818 | 1500 | N | 0.02022+/-0.00275 | 0.01891+/-0.00314 | 7.0x | -80.5% |
| E1_v4_recipe | 82,818 | 2000 | Y | 0.00570+/-0.00074 | 0.00286+/-0.00043 | 2.0x | +49.1% |
| E2_wider_deeper | 395,522 | 2000 | Y | 0.00573+/-0.00102 | 0.00254+/-0.00032 | 2.0x | +48.8% |
| E3_separate_heads | 82,690 | 2000 | Y | 0.00410+/-0.00021 | 0.00168+/-0.00013 | 1.4x | +63.4% |
| E4_residual | 214,914 | 2000 | Y | 0.00449+/-0.00028 | 0.00179+/-0.00023 | 1.5x | +59.9% |
| E5_enhanced_features | 82,818 | 2000 | Y | 0.01060+/-0.00118 | 0.00827+/-0.00020 | 3.7x | +5.3% |
| E6_huber_loss | 82,818 | 2000 | Y | 0.00572+/-0.00042 | 0.00311+/-0.00074 | 2.0x | +49.0% |

## Best experiment: E3_separate_heads
- nF RMSE: 0.00410 (vs v3 baseline 0.0112)
- nSF6 RMSE: 0.00168 (vs v3 baseline 0.0081)
- Improvement over v3: 63.4% nF, 79.3% nSF6
- Remaining gap to v4: 1.4x nF, 0.6x nSF6
