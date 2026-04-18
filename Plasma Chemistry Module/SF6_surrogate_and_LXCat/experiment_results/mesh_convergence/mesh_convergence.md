# Mesh Convergence Study

Reference case: 700W, 10mTorr, pure SF6

## LEGACY mode

| Mesh | Cells | Active | nF_avg | nSF6_avg | ne_avg | F_drop | Time |
|---|---|---|---|---|---|---|---|
| coarse (20x30) | 600 | 240 | 1.325e+20 | 1.568e+20 | 1.026e+17 | 72.9% | 2.6s |
| current (30x50) | 1500 | 642 | 1.240e+20 | 1.663e+20 | 9.026e+16 | 71.7% | 8.1s |
| fine (50x80) | 4000 | 1669 | 1.263e+20 | 1.625e+20 | 9.490e+16 | 73.7% | 24.2s |

**Convergence (current vs fine):**
- nF relative error: 0.0186 (1.86%)
- nSF6 relative error: 0.0230 (2.30%)
- ne relative error: 0.0489 (4.89%)
- F_drop relative error: 0.0267 (2.67%)

## LXCAT mode

| Mesh | Cells | Active | nF_avg | nSF6_avg | ne_avg | F_drop | Time |
|---|---|---|---|---|---|---|---|
| coarse (20x30) | 600 | 240 | 1.329e+20 | 1.584e+20 | 3.241e+16 | 72.8% | 8.0s |
| current (30x50) | 1500 | 642 | 1.240e+20 | 1.681e+20 | 2.844e+16 | 71.6% | 13.2s |
| fine (50x80) | 4000 | 1669 | 1.264e+20 | 1.643e+20 | 2.993e+16 | 73.5% | 32.4s |

**Convergence (current vs fine):**
- nF relative error: 0.0196 (1.96%)
- nSF6 relative error: 0.0234 (2.34%)
- ne relative error: 0.0497 (4.97%)
- F_drop relative error: 0.0268 (2.68%)

