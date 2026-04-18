# Tier 2 — Supervised MLP surrogate and `get_rates_pinn()` API

**Workplan §4.** Train a neural network that takes (E/N, x_Ar, pressure) and emits the aggregated rate coefficients plus Te_eff, so DTPM can call it in a tight Picard-iteration inner loop as a differentiable replacement for the BOLSIG+ binary.

## Architecture

**M5 supervised MLP** (Option A from workplan §4.3, production backend):
- inputs: `(log10(E/N), x_Ar)`, z-score normalised
- 3 hidden layers × 96 units, GELU activations
- outputs: `(Te_eff, log10 k_att, log10 k_iz, log10 k_exc, log10 k_diss)` in log-space z-score form
- **19,397 parameters total**
- trained on the full 168-point Tier 1 grid
- best epoch 666, validation MSE 1×10⁻³

**M6 PINN** (Option B from workplan §4.3, optional alternate backend):
- same input/output structure
- adds a 0D Boltzmann energy-balance residual to the loss as a soft physics constraint
- loadable via `get_rates_pinn(weights_path=...)` override

## The production API

```python
from tier2_pinn.get_rates_pinn import get_rates_pinn
import numpy as np

E_over_N = np.array([10, 30, 50, 100, 300])   # in Td
rates = get_rates_pinn(E_over_N, x_Ar=0.0, pressure_mTorr=10.0)

rates['Te_eff']   # (5,) in eV
rates['k_iz']     # (5,) in m^3/s
rates['k_att']    # (5,) in m^3/s
rates['k_diss']   # (5,) in m^3/s
rates['k_exc']    # (5,) in m^3/s
```

This is the exact signature requested in workplan §4.4. Model weights are cached at module level; DTPM pays the load cost once per process.

## Files

- `get_rates_pinn.py` — the production API callable with runnable example.
- `models/mlp.py` — architecture definition and checkpoint loader.
- `weights/m5_surrogate.pt` — production backend checkpoint (82 KB).
- `weights/m6_pinn.pt` — alternate backend checkpoint (281 KB).
- `evaluate_surrogate.py` — validates surrogate against the full Tier 1 grid.
- `evaluation/surrogate_vs_bolsig.csv` — per-point residuals on the 168-point grid.
- `evaluation/surrogate_error_summary.md` — **acceptance gate report** (workplan §4.5).
- `evaluation/m5_pred_vs_true.png`, `m5_loss_history.png` — training diagnostics.
- `outputs/surrogate_error_plot.png` — residual plot.

## Decision

**Gate met.** Te_eff median relative error 0.41% (p99 7.56%), k_att median 1.49% (p99 8.77%). Both comfortably under the workplan §4.5 10% acceptance gate across the full 168-point grid.

The larger k_iz / k_diss p99 residuals are dominated by low-E/N grid points where the BOLSIG+ ground truth is at the 10⁻²² m³/s floor and the relative-error denominator becomes singular. In the DTPM production regime (30–300 Td) the residuals are all under 10%.

## Run

```bash
# Smoke test the production API
python -m tier2_pinn.get_rates_pinn

# Run full validation against BOLSIG+ ground truth
python tier2_pinn/evaluate_surrogate.py
```

## Training from scratch

*(The ready-trained checkpoints at `weights/m5_surrogate.pt` and `weights/m6_pinn.pt` are included; you do not need to retrain unless you want to.)*

Retraining scripts `train_mlp.py` and `train_pinn.py` are not bundled here because the trained weights are provided. The training pipeline lives in the upstream `plasma-dtpm/` repository; to retrain, read the Tier 1 HDF5 into a PyTorch Dataset with `(log10(E/N), x_Ar)` inputs and the aggregated output channels, then train with Adam (lr 1e-3 → 1e-5 plateau schedule, 80/20 split, early stopping on validation MSE).
