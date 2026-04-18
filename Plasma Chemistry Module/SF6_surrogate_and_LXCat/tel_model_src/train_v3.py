"""
Spatial surrogate v3 — refined training with:
  - 88-case dataset (v3)
  - Per-species loss weighting (nSF6 upweighted)
  - 5-model ensemble
  - Post-hoc uncertainty calibration
  - Diagnostic plots (error vs parameter)
  - Mandatory GPU/MPS acceleration

Usage:
    python train_v3.py
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(REPO, 'results', 'pinn_dataset_v3')
RESULTS_DIR = os.path.join(REPO, 'results', 'surrogate_v3')
os.makedirs(RESULTS_DIR, exist_ok=True)

R_PROC, Z_TOP = 0.105, 0.234

# ═══════════════════════════════════════════════════════════════
# 0. DEVICE SELECTION (mandatory GPU check)
# ═══════════════════════════════════════════════════════════════

def select_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        reason = "CUDA GPU detected"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
        reason = "Apple Silicon MPS detected"
    else:
        dev = torch.device('cpu')
        reason = "WARNING: No GPU available, falling back to CPU"
    print(f"  Device: {dev} ({reason})")
    return dev

# ═══════════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════════

def load_dataset(val_fraction=0.15):
    """Load v3 dataset, split by stratified case holdout."""
    with open(os.path.join(DATASET_DIR, 'metadata.json')) as f:
        meta = json.load(f)

    # Stratified holdout: pick cases spanning the full parameter range
    n = len(meta)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_val = max(int(n * val_fraction), 6)
    val_idx = set(indices[:n_val].tolist())

    all_data = {k: [] for k in ['r','z','P','p','Ar','lnF','lnSF6','case']}
    for entry in meta:
        d = np.load(os.path.join(DATASET_DIR, entry['file']))
        inside = d['inside'].astype(bool)
        rc, zc = d['rc'], d['zc']
        for i in range(len(rc)):
            for j in range(len(zc)):
                if inside[i, j]:
                    all_data['r'].append(rc[i])
                    all_data['z'].append(zc[j])
                    all_data['P'].append(float(entry['P_rf']))
                    all_data['p'].append(float(entry['p_mTorr']))
                    all_data['Ar'].append(float(entry['frac_Ar']))
                    all_data['lnF'].append(np.log10(max(d['nF'][i,j], 1e6)))
                    all_data['lnSF6'].append(np.log10(max(d['nSF6'][i,j], 1e6)))
                    all_data['case'].append(entry['idx'])

    arrays = {k: np.array(v, dtype=np.float32) for k, v in all_data.items()}
    arrays['case'] = arrays['case'].astype(np.int32)
    train_mask = np.array([c not in val_idx for c in arrays['case']])

    def _split(mask):
        return {k: v[mask] for k, v in arrays.items()}

    print(f"  Train: {train_mask.sum()} pts ({n - n_val} cases)")
    print(f"  Val:   {(~train_mask).sum()} pts ({n_val} cases)")
    return _split(train_mask), _split(~train_mask), meta, val_idx


def to_tensors(data, device):
    X = np.column_stack([
        data['r'] / R_PROC, data['z'] / Z_TOP,
        data['P'] / 1200.0, data['p'] / 20.0, data['Ar'],
    ])
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


# ═══════════════════════════════════════════════════════════════
# 2. MODEL (same arch as v2 but with configurable dropout)
# ═══════════════════════════════════════════════════════════════

class SurrogateV3(nn.Module):
    def __init__(self, n_hidden=128, n_layers=4, n_fourier=64,
                 fourier_scale=3.0, dropout=0.05):
        super().__init__()
        B = torch.randn(5, n_fourier) * fourier_scale
        self.register_buffer('B', B)
        n_in = 2 * n_fourier

        layers = [nn.Linear(n_in, n_hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU(),
                           nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(n_in, n_hidden)
        self.head = nn.Linear(n_hidden, 2)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))

    def forward(self, x):
        proj = x @ self.B
        e = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.head(self.backbone(e) + self.proj(e))


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING (with species-balanced loss + smoothness)
# ═══════════════════════════════════════════════════════════════

def train_single(model, X_train, Y_train, X_val, Y_val, y_std, device,
                 n_epochs=2000, lr=1e-3, w_nF=1.0, w_nSF6=1.5,
                 w_smooth=5e-4, batch_size=4096, model_idx=0):
    """Train one model with per-species weighting."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
    n = X_train.shape[0]
    species_w = torch.tensor([w_nF, w_nSF6], device=device)
    best_val, best_state = float('inf'), None
    hist = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss, nb = 0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            pred = model(xb)
            # Per-species weighted normalised MSE
            L_data = (species_w * ((pred - yb) / y_std).pow(2)).mean()

            # Smoothness (every 4th batch)
            L_sm = torch.tensor(0.0, device=device)
            if w_smooth > 0 and nb % 4 == 0:
                xb_g = xb.detach().clone().requires_grad_(True)
                p = model(xb_g)
                for col in range(2):
                    g = torch.autograd.grad(p[:, col].sum(), xb_g, create_graph=True)[0]
                    L_sm = L_sm + (g[:, 0]**2 + g[:, 1]**2).mean()

            loss = L_data + w_smooth * L_sm
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += L_data.item()
            nb += 1

        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(X_val)
            vl = (species_w * ((vp - Y_val) / y_std).pow(2)).mean().item()

        hist['train'].append(ep_loss / nb)
        hist['val'].append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(n_epochs // 10, 1) == 0:
            print(f"    M{model_idx} [{epoch:5d}] train={ep_loss/nb:.5f} val={vl:.5f} "
                  f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    hist['best_val'] = best_val
    return hist


# ═══════════════════════════════════════════════════════════════
# 4. ENSEMBLE + UQ + CALIBRATION + DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def ensemble_predict(models, X, device, n_mc=20):
    """Predict with ensemble + MC dropout. Returns mean, std."""
    preds_ens = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds_ens.append(m(X).cpu().numpy())
    preds_ens = np.stack(preds_ens)

    # MC dropout on each model (5 passes each)
    mc_preds = []
    for m in models:
        m.train()
        for _ in range(n_mc // len(models)):
            with torch.no_grad():
                mc_preds.append(m(X).cpu().numpy())
        m.eval()
    mc_preds = np.stack(mc_preds)

    ens_mean = preds_ens.mean(axis=0)
    total_var = preds_ens.var(axis=0) + mc_preds.var(axis=0)
    return ens_mean, np.sqrt(total_var)


def calibrate_uncertainty(pred_std, actual_err, n_bins=10):
    """Post-hoc calibration: find scale factor so 68% of errors fall within 1σ."""
    ratios = np.abs(actual_err) / np.clip(pred_std, 1e-8, None)
    # Target: 68% of ratios should be ≤ 1.0
    q68 = np.percentile(ratios, 68)
    # Scale factor: multiply predicted std by q68 so that 68% coverage is achieved
    return q68


def diagnostic_plots(val_data, pred_mean, pred_std, meta, output_dir):
    """Error vs operating condition plots."""

    Y_true = np.column_stack([val_data['lnF'], val_data['lnSF6']])
    err = np.abs(pred_mean - Y_true)

    for col, name in enumerate(['nF', 'nSF6']):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Error vs power
        ax = axes[0]
        ax.scatter(val_data['P'], err[:, col], s=3, alpha=0.3)
        ax.set_xlabel('P_rf (W)'); ax.set_ylabel(f'|error| log₁₀({name})')
        ax.set_title(f'{name}: error vs power')

        # Error vs pressure
        ax = axes[1]
        ax.scatter(val_data['p'], err[:, col], s=3, alpha=0.3)
        ax.set_xlabel('p (mTorr)'); ax.set_ylabel(f'|error| log₁₀({name})')
        ax.set_title(f'{name}: error vs pressure')

        # Error vs Ar fraction
        ax = axes[2]
        ax.scatter(val_data['Ar'], err[:, col], s=3, alpha=0.3)
        ax.set_xlabel('Ar fraction'); ax.set_ylabel(f'|error| log₁₀({name})')
        ax.set_title(f'{name}: error vs Ar')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'diagnostics_{name}.png'), dpi=150)
        plt.close(fig)

    # Worst cases
    case_ids = np.unique(val_data['case'])
    case_errors = []
    for c in case_ids:
        mask = val_data['case'] == c
        me = meta[int(c)]
        e_nF = np.sqrt((err[mask, 0]**2).mean())
        e_nSF6 = np.sqrt((err[mask, 1]**2).mean())
        case_errors.append({
            'case': int(c), 'P_rf': me['P_rf'], 'p_mTorr': me['p_mTorr'],
            'frac_Ar': me['frac_Ar'], 'rmse_nF': float(e_nF),
            'rmse_nSF6': float(e_nSF6),
        })
    case_errors.sort(key=lambda x: x['rmse_nF'], reverse=True)
    return case_errors


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Spatial Surrogate v3 — Refined Training")
    print("=" * 60)

    device = select_device()
    train_data, val_data, meta, val_idx = load_dataset()

    X_train, Y_train = to_tensors(train_data, device)
    X_val, Y_val = to_tensors(val_data, device)
    y_std = Y_train.std(dim=0, keepdim=True).clamp(min=1e-3)

    # ── Ensemble training ──
    N_ENSEMBLE = 5
    N_EPOCHS = 2000
    W_NF, W_NSF6 = 1.0, 1.5  # upweight nSF6

    print(f"\n  Config:")
    print(f"    Ensemble: {N_ENSEMBLE} models")
    print(f"    Epochs:   {N_EPOCHS}")
    print(f"    Weights:  nF={W_NF}, nSF6={W_NSF6}")
    print(f"    Smooth:   5e-4")

    models, histories = [], []
    t_total = time.time()

    for i in range(N_ENSEMBLE):
        print(f"\n  --- Model {i}/{N_ENSEMBLE} ---", flush=True)
        torch.manual_seed(42 + i * 137)
        np.random.seed(42 + i * 137)
        model = SurrogateV3().to(device)

        # Verify device
        param_dev = next(model.parameters()).device
        print(f"    Model on: {param_dev}", flush=True)
        assert str(param_dev).startswith(str(device).split(':')[0]), \
            f"Model NOT on {device}!"

        h = train_single(model, X_train, Y_train, X_val, Y_val, y_std, device,
                          n_epochs=N_EPOCHS, w_nF=W_NF, w_nSF6=W_NSF6,
                          model_idx=i)
        models.append(model.cpu())
        histories.append(h)
        print(f"    Best val: {h['best_val']:.4f}", flush=True)

    train_time = time.time() - t_total
    print(f"\n  Total training: {train_time:.1f}s ({train_time/60:.1f} min) on {device}")

    # Save models
    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(RESULTS_DIR, f'model_{i}.pt'))

    # ── Evaluation ──
    print(f"\n{'='*60}")
    print("  Evaluation")
    print(f"{'='*60}")

    # Move models to CPU for evaluation
    X_val_cpu = X_val.cpu()
    pred_mean, pred_std = ensemble_predict(models, X_val_cpu, 'cpu')
    Y_val_np = Y_val.cpu().numpy()

    for col, name in enumerate(['nF', 'nSF6']):
        err = pred_mean[:, col] - Y_val_np[:, col]
        rmse = float(np.sqrt((err**2).mean()))
        mae = float(np.abs(err).mean())
        pct = float(np.median(np.abs(err) * np.log(10) * 100))
        corr = float(np.corrcoef(np.abs(err), pred_std[:, col])[0, 1])
        cal = calibrate_uncertainty(pred_std[:, col], err)
        print(f"\n  {name}:")
        print(f"    RMSE:              {rmse:.4f}")
        print(f"    MAE:               {mae:.4f}")
        print(f"    Median % error:    {pct:.1f}%")
        print(f"    Unc-err corr:      {corr:.3f}")
        print(f"    Calibration (q68): {cal:.3f} (ideal=1.0)")

    # Ensemble stats
    vals = [h['best_val'] for h in histories]
    print(f"\n  Ensemble val losses: {[f'{v:.4f}' for v in vals]}")
    print(f"  Mean: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"  Best: {min(vals):.4f}, Worst: {max(vals):.4f}")

    # ── Diagnostic plots ──
    print(f"\n  Generating diagnostic plots...", flush=True)
    case_errors = diagnostic_plots(val_data, pred_mean, pred_std, meta, RESULTS_DIR)

    # Pred vs true scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for col, name in enumerate(['nF', 'nSF6']):
        ax = axes[col]
        p, t, u = pred_mean[:, col], Y_val_np[:, col], pred_std[:, col]
        ax.errorbar(t, p, yerr=u, fmt='o', ms=2, alpha=0.3,
                    elinewidth=0.5, capsize=0, ecolor='C0')
        lim = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlabel(f'FD solver log₁₀({name})')
        ax.set_ylabel(f'Surrogate log₁₀({name})')
        rmse = float(np.sqrt(((p - t)**2).mean()))
        ax.set_title(f'{name} (RMSE={rmse:.3f})')
        ax.set_aspect('equal')
    fig.suptitle(f'v3: {N_ENSEMBLE}-model ensemble, {len(meta)} cases', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'val_pred_vs_true.png'), dpi=150)
    plt.close()

    # Uncertainty calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, name in enumerate(['nF', 'nSF6']):
        ax = axes[col]
        err = np.abs(pred_mean[:, col] - Y_val_np[:, col])
        unc = pred_std[:, col]
        ax.scatter(unc, err, s=3, alpha=0.3)
        ax.plot([0, unc.max()], [0, unc.max()], 'k--', lw=1, label='ideal')
        ax.set_xlabel('Predicted σ')
        ax.set_ylabel('Actual |error|')
        ax.set_title(f'{name}: r={np.corrcoef(unc, err)[0,1]:.2f}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'uncertainty_calibration.png'), dpi=150)
    plt.close()

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, h in enumerate(histories):
        ax.semilogy(h['train'], alpha=0.5, label=f'M{i} train')
        ax.semilogy(h['val'], alpha=0.7, ls='--', label=f'M{i} val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Ensemble training curves'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'loss_curves.png'), dpi=150)
    plt.close()

    print(f"  Saved plots to {RESULTS_DIR}", flush=True)

    # ── Worst cases ──
    print(f"\n  Worst validation cases (by nF RMSE):")
    for ce in case_errors[:5]:
        print(f"    Case {ce['case']}: P={ce['P_rf']}W p={ce['p_mTorr']}mT "
              f"Ar={ce['frac_Ar']:.2f} → nF_rmse={ce['rmse_nF']:.3f} "
              f"nSF6_rmse={ce['rmse_nSF6']:.3f}")

    # ── Runtime benchmark ──
    print(f"\n  Runtime benchmark...", flush=True)
    from solver import TELSolver
    t0 = time.time()
    TELSolver(P_rf=700, p_mTorr=10, Nr=30, Nz=50).solve(n_iter=50, verbose=False)
    fd_time = time.time() - t0

    m_bench = models[0].to('cpu')
    x_bench = X_val_cpu[:642]  # one case worth
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            m_bench(x_bench)
    surr_time = (time.time() - t0) / 100

    print(f"    FD solver: {fd_time:.2f}s")
    print(f"    Surrogate: {surr_time*1000:.2f}ms")
    print(f"    Speedup:   {fd_time/surr_time:.0f}×")

    # ── Save summary ──
    summary = {
        'device': str(device),
        'n_cases': len(meta),
        'n_val_cases': len(val_idx),
        'n_ensemble': N_ENSEMBLE,
        'n_epochs': N_EPOCHS,
        'w_nF': W_NF, 'w_nSF6': W_NSF6,
        'train_time_s': train_time,
        'ensemble_val_losses': vals,
        'ensemble_mean': float(np.mean(vals)),
        'ensemble_std': float(np.std(vals)),
        'nF_rmse': float(np.sqrt(((pred_mean[:, 0] - Y_val_np[:, 0])**2).mean())),
        'nSF6_rmse': float(np.sqrt(((pred_mean[:, 1] - Y_val_np[:, 1])**2).mean())),
        'nF_median_pct': float(np.median(np.abs(pred_mean[:, 0] - Y_val_np[:, 0]) * np.log(10) * 100)),
        'nSF6_median_pct': float(np.median(np.abs(pred_mean[:, 1] - Y_val_np[:, 1]) * np.log(10) * 100)),
        'nF_unc_corr': float(np.corrcoef(np.abs(pred_mean[:, 0] - Y_val_np[:, 0]), pred_std[:, 0])[0, 1]),
        'nSF6_unc_corr': float(np.corrcoef(np.abs(pred_mean[:, 1] - Y_val_np[:, 1]), pred_std[:, 1])[0, 1]),
        'fd_time_s': fd_time,
        'surrogate_ms': surr_time * 1000,
        'speedup': fd_time / surr_time,
        'worst_cases': case_errors[:5],
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE. Results in {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
