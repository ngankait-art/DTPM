"""
Spatial field surrogate v2 for the TEL ICP reactor.

Improvements over v1:
  - 42-case expanded dataset (5 powers × 6 pressures × Ar sweeps)
  - Physics regularisation: smoothness + mass-conservation soft penalty
  - Ensemble of 5 models for uncertainty quantification
  - Fast-evaluator API for integration into TEL workflow
  - Runtime comparison against the FD solver

Usage:
    python spatial_surrogate_v2.py --train
    python spatial_surrogate_v2.py --evaluate
    python spatial_surrogate_v2.py --benchmark
"""
from __future__ import annotations

import os, sys, json, time, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
DATASET_DIR = os.path.join(REPO, 'results', 'pinn_dataset_v2')
RESULTS_DIR = os.path.join(REPO, 'results', 'surrogate_v2')

R_PROC, Z_TOP = 0.105, 0.234

# ═══════════════════════════════════════════════════════════════
# 1. DATASET (same structure as v1, pointing to v2 data)
# ═══════════════════════════════════════════════════════════════

def load_dataset(dataset_dir: str, val_indices: Optional[List[int]] = None):
    """Load and split into train/val by case index."""
    with open(os.path.join(dataset_dir, 'metadata.json')) as f:
        meta = json.load(f)

    if val_indices is None:
        # Hold out ~15% of cases: pick cases spanning the parameter space
        n = len(meta)
        val_indices = list(range(n - 6, n))  # last 6 cases (Ar corner sweeps)

    all_data = {k: [] for k in ['r','z','P','p','Ar','lnF','lnSF6','case']}

    for entry in meta:
        d = np.load(os.path.join(dataset_dir, entry['file']))
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

    train_mask = np.array([c not in val_indices for c in arrays['case']])

    def _split(mask):
        return {k: v[mask] for k, v in arrays.items()}

    return _split(train_mask), _split(~train_mask), meta


def to_tensors(data: dict, device: torch.device):
    """Convert data dict to normalised (X, Y) tensors."""
    X = np.column_stack([
        data['r'] / R_PROC,
        data['z'] / Z_TOP,
        data['P'] / 1200.0,    # max power in grid
        data['p'] / 20.0,
        data['Ar'],
    ])
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


# ═══════════════════════════════════════════════════════════════
# 2. MODEL (same architecture, supports MC dropout)
# ═══════════════════════════════════════════════════════════════

class SpatialSurrogateV2(nn.Module):
    """(r, z, P, p, Ar) → (log10 nF, log10 nSF6) with optional dropout."""

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

    def _fourier(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        e = self._fourier(x)
        h = self.backbone(e) + self.proj(e)
        return self.head(h)

    def mc_predict(self, x, n_samples=30):
        """MC dropout prediction: returns (mean, std) over stochastic passes."""
        self.train()  # keep dropout active
        preds = torch.stack([self(x) for _ in range(n_samples)])
        self.eval()
        return preds.mean(dim=0), preds.std(dim=0)


# ═══════════════════════════════════════════════════════════════
# 3. PHYSICS REGULARISATION
# ═══════════════════════════════════════════════════════════════

def smoothness_penalty(model, X, device):
    """Penalise large spatial gradients ∂(log n)/∂r and ∂(log n)/∂z.

    Encourages the surrogate to produce physically smooth fields rather
    than overfitting to mesh artifacts. Uses autograd on the (r, z) inputs.
    """
    X = X.requires_grad_(True)
    pred = model(X)  # (N, 2)

    penalty = torch.tensor(0.0, device=device)
    for col in range(2):
        grads = torch.autograd.grad(pred[:, col].sum(), X,
                                    create_graph=True)[0]
        # Only penalise spatial dimensions (cols 0, 1 = r, z)
        dr = grads[:, 0]  # ∂(log n)/∂(r_norm)
        dz = grads[:, 1]  # ∂(log n)/∂(z_norm)
        penalty = penalty + (dr**2 + dz**2).mean()

    return penalty


def mass_conservation_penalty(model, data, device):
    """Soft penalty: total F production ≈ total F loss per case.

    For each training case, the volume-integrated fluorine should satisfy:
        ∫ S_F dV ≈ ∫ L_F·n_F dV + wall losses

    Since we don't have the full chemistry in the surrogate, we use a
    simpler proxy: the predicted log10(nF) field should integrate to a
    value consistent with the training data's integrated value.
    """
    X, Y = to_tensors(data, device)
    pred = model(X)

    # Group by case and compare volume-weighted mean
    case_ids = data['case']
    unique_cases = np.unique(case_ids)
    penalty = torch.tensor(0.0, device=device)

    for c in unique_cases[:10]:  # sample up to 10 cases per batch
        mask = torch.tensor(case_ids == c, device=device)
        pred_mean = pred[mask, 0].mean()
        true_mean = Y[mask, 0].mean()
        penalty = penalty + (pred_mean - true_mean)**2

    return penalty / min(len(unique_cases), 10)


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING (with regularisation)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    n_epochs: int = 3000
    lr: float = 1e-3
    batch_size: int = 4096
    device: str = 'cpu'
    # Regularisation weights
    w_smooth: float = 1e-3
    w_mass: float = 1e-2
    # Ensemble
    n_ensemble: int = 5


def train_single(model, train_data, val_data, cfg, model_idx=0, verbose=True):
    """Train one model. Returns history dict."""
    device = torch.device(cfg.device)
    model = model.to(device)

    X_train, Y_train = to_tensors(train_data, device)
    X_val, Y_val = to_tensors(val_data, device)
    y_std = Y_train.std(dim=0, keepdim=True).clamp(min=1e-3)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs,
                                                         eta_min=1e-6)

    n_train = X_train.shape[0]
    best_val, best_state = float('inf'), None
    history = {'train': [], 'val': [], 'smooth': [], 'mass': []}

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss, ep_smooth, ep_mass, nb = 0, 0, 0, 0

        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb, yb = X_train[idx], Y_train[idx]

            pred = model(xb)
            L_data = ((pred - yb) / y_std).pow(2).mean()

            # Physics regularisation (every 4th batch to save compute)
            L_smooth = torch.tensor(0.0, device=device)
            L_mass = torch.tensor(0.0, device=device)
            if nb % 4 == 0:
                L_smooth = smoothness_penalty(model, xb.detach().clone(), device)
                L_mass = mass_conservation_penalty(model, train_data, device)

            loss = L_data + cfg.w_smooth * L_smooth + cfg.w_mass * L_mass

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += L_data.item()
            ep_smooth += L_smooth.item()
            ep_mass += L_mass.item()
            nb += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = ((val_pred - Y_val) / y_std).pow(2).mean().item()

        history['train'].append(ep_loss / nb)
        history['val'].append(val_loss)
        history['smooth'].append(ep_smooth / max(nb // 4, 1))
        history['mass'].append(ep_mass / max(nb // 4, 1))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and epoch % max(cfg.n_epochs // 10, 1) == 0:
            lr = opt.param_groups[0]['lr']
            print(f"  M{model_idx} [{epoch:5d}] "
                  f"train={ep_loss/nb:.4f} val={val_loss:.4f} "
                  f"sm={ep_smooth/max(nb//4,1):.2e} mass={ep_mass/max(nb//4,1):.2e} "
                  f"lr={lr:.1e}")

    if best_state:
        model.load_state_dict(best_state)
    history['best_val'] = best_val
    return history


def train_ensemble(train_data, val_data, cfg, verbose=True):
    """Train an ensemble of models with different random seeds."""
    models, histories = [], []

    for i in range(cfg.n_ensemble):
        torch.manual_seed(42 + i * 137)
        np.random.seed(42 + i * 137)
        model = SpatialSurrogateV2()
        if verbose:
            print(f"\n--- Ensemble member {i}/{cfg.n_ensemble} ---")
        h = train_single(model, train_data, val_data, cfg, model_idx=i,
                          verbose=verbose)
        models.append(model)
        histories.append(h)
        if verbose:
            print(f"  Best val: {h['best_val']:.4f}")

    return models, histories


# ═══════════════════════════════════════════════════════════════
# 5. UNCERTAINTY QUANTIFICATION
# ═══════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """Fast evaluator with ensemble + MC dropout uncertainty."""

    def __init__(self, models: List[SpatialSurrogateV2], device='cpu'):
        self.models = [m.to(device).eval() for m in models]
        self.device = torch.device(device)

    def predict(self, r, z, P_rf, p_mTorr, frac_Ar):
        """Predict fields with uncertainty.

        All inputs can be scalars or arrays of length N.

        Returns dict with:
            nF_mean, nF_std     (m⁻³, linear scale)
            nSF6_mean, nSF6_std
            log_nF_mean, log_nF_std   (log10 scale)
            log_nSF6_mean, log_nSF6_std
        """
        r = np.atleast_1d(np.asarray(r, dtype=np.float32))
        z = np.atleast_1d(np.asarray(z, dtype=np.float32))
        P = np.broadcast_to(np.float32(P_rf), r.shape)
        p = np.broadcast_to(np.float32(p_mTorr), r.shape)
        Ar = np.broadcast_to(np.float32(frac_Ar), r.shape)

        X = torch.tensor(np.column_stack([
            r / R_PROC, z / Z_TOP, P / 1200.0, p / 20.0, Ar
        ]), dtype=torch.float32, device=self.device)

        # Ensemble predictions
        preds = []
        for m in self.models:
            with torch.no_grad():
                preds.append(m(X).cpu().numpy())
        preds = np.stack(preds)  # (n_ensemble, N, 2)

        # Also do MC dropout on the best model
        mc_preds = []
        best = self.models[0]
        best.train()
        for _ in range(20):
            with torch.no_grad():
                mc_preds.append(best(X).cpu().numpy())
        best.eval()
        mc_preds = np.stack(mc_preds)

        # Combine: total uncertainty = sqrt(ensemble_var + mc_var)
        ens_mean = preds.mean(axis=0)
        ens_var = preds.var(axis=0)
        mc_var = mc_preds.var(axis=0)
        total_std = np.sqrt(ens_var + mc_var)

        return {
            'log_nF_mean': ens_mean[:, 0],
            'log_nF_std': total_std[:, 0],
            'log_nSF6_mean': ens_mean[:, 1],
            'log_nSF6_std': total_std[:, 1],
            'nF_mean': 10**ens_mean[:, 0],
            'nF_std': 10**ens_mean[:, 0] * np.log(10) * total_std[:, 0],
            'nSF6_mean': 10**ens_mean[:, 1],
            'nSF6_std': 10**ens_mean[:, 1] * np.log(10) * total_std[:, 1],
        }

    def predict_field(self, mesh_rc, mesh_zc, inside, P_rf, p_mTorr, frac_Ar):
        """Predict 2D fields on a mesh (like the FD solver output)."""
        Nr, Nz = len(mesh_rc), len(mesh_zc)
        r_pts, z_pts, ij_map = [], [], []
        for i in range(Nr):
            for j in range(Nz):
                if inside[i, j]:
                    r_pts.append(mesh_rc[i])
                    z_pts.append(mesh_zc[j])
                    ij_map.append((i, j))

        result = self.predict(np.array(r_pts), np.array(z_pts),
                              P_rf, p_mTorr, frac_Ar)

        fields = {}
        for key in ['nF_mean', 'nF_std', 'nSF6_mean', 'nSF6_std']:
            f = np.full((Nr, Nz), np.nan)
            for k, (i, j) in enumerate(ij_map):
                f[i, j] = result[key][k]
            fields[key] = f

        return fields


# ═══════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════

def full_evaluation(predictor, val_data, meta, output_dir, device='cpu'):
    """Comprehensive evaluation with plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    X_val, Y_val = to_tensors(val_data, torch.device(device))
    r_val, z_val = val_data['r'], val_data['z']
    P_val, p_val, Ar_val = val_data['P'], val_data['p'], val_data['Ar']

    result = predictor.predict(r_val, z_val, P_val, p_val, Ar_val)
    Y_np = Y_val.cpu().numpy()

    # Metrics
    for col, name in enumerate(['nF', 'nSF6']):
        pred = result[f'log_{name}_mean']
        true = Y_np[:, col]
        unc = result[f'log_{name}_std']
        err = pred - true
        metrics = {
            'rmse': float(np.sqrt((err**2).mean())),
            'mae': float(np.abs(err).mean()),
            'median_pct': float(np.median(np.abs(err) * np.log(10) * 100)),
            'mean_uncertainty': float(unc.mean()),
            'correlation_err_unc': float(np.corrcoef(np.abs(err), unc)[0, 1]),
        }
        print(f"\n  {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    # --- Plot 1: pred vs true with uncertainty ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for col, name in enumerate(['nF', 'nSF6']):
        ax = axes[col]
        p = result[f'log_{name}_mean']
        t = Y_np[:, col]
        u = result[f'log_{name}_std']
        ax.errorbar(t, p, yerr=u, fmt='o', ms=2, alpha=0.3,
                    elinewidth=0.5, capsize=0, ecolor='C0')
        lim = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlabel(f'FD solver log₁₀({name})')
        ax.set_ylabel(f'Surrogate log₁₀({name})')
        rmse = float(np.sqrt(((p - t)**2).mean()))
        ax.set_title(f'Validation: {name} (RMSE={rmse:.3f})')
        ax.set_aspect('equal')
    fig.suptitle(f'Spatial Surrogate v2 — Held-out Validation '
                 f'({len(set(val_data["case"].tolist()))} cases)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'val_pred_vs_true.png'), dpi=150)
    plt.close(fig)
    print("  Saved val_pred_vs_true.png")

    # --- Plot 2: uncertainty vs error (calibration) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, name in enumerate(['nF', 'nSF6']):
        ax = axes[col]
        err = np.abs(result[f'log_{name}_mean'] - Y_np[:, col])
        unc = result[f'log_{name}_std']
        ax.scatter(unc, err, s=3, alpha=0.3)
        ax.plot([0, unc.max()], [0, unc.max()], 'k--', lw=1, label='ideal')
        ax.set_xlabel('Predicted uncertainty (log₁₀)')
        ax.set_ylabel('Actual |error| (log₁₀)')
        ax.set_title(f'{name}: uncertainty calibration')
        corr = np.corrcoef(unc, err)[0, 1]
        ax.text(0.05, 0.92, f'r={corr:.2f}', transform=ax.transAxes)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'uncertainty_calibration.png'), dpi=150)
    plt.close(fig)
    print("  Saved uncertainty_calibration.png")

    # --- Plot 3: wafer profile for a held-out case ---
    cases_in_val = sorted(set(val_data['case'].tolist()))
    if cases_in_val:
        case_id = int(cases_in_val[0])
        mask = val_data['case'] == case_id
        r_c = val_data['r'][mask]
        z_c = val_data['z'][mask]

        z_min = z_c.min()
        wafer = np.abs(z_c - z_min) < 1e-3
        if wafer.sum() > 2:
            r_w = r_c[wafer]
            order = np.argsort(r_w)
            r_w = r_w[order]
            nF_true = 10**val_data['lnF'][mask][wafer][order]

            res = predictor.predict(r_w, np.full_like(r_w, z_min),
                                     val_data['P'][mask][wafer][order][0],
                                     val_data['p'][mask][wafer][order][0],
                                     val_data['Ar'][mask][wafer][order][0])

            me = meta[case_id]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(r_w*1e3, nF_true, 'ko-', ms=4, label='FD solver')
            ax.plot(r_w*1e3, res['nF_mean'], 'r^--', ms=4, label='Surrogate')
            ax.fill_between(r_w*1e3,
                            res['nF_mean'] - 2*res['nF_std'],
                            res['nF_mean'] + 2*res['nF_std'],
                            alpha=0.2, color='red', label='±2σ')
            ax.set_xlabel('r (mm)')
            ax.set_ylabel('[F] (m⁻³)')
            ax.set_title(f'Wafer F profile — held-out '
                         f'(P={me["P_rf"]}W, p={me["p_mTorr"]}mT, '
                         f'Ar={me["frac_Ar"]:.0%})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'val_wafer_profile.png'), dpi=150)
            plt.close(fig)
            print("  Saved val_wafer_profile.png")


# ═══════════════════════════════════════════════════════════════
# 7. RUNTIME BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark(predictor, verbose=True):
    """Compare surrogate vs FD solver runtime."""
    sys.path.insert(0, os.path.join(REPO, 'src'))
    from solver import TELSolver

    # FD solver
    t0 = time.time()
    s = TELSolver(P_rf=700, p_mTorr=10, Nr=30, Nz=50)
    r_fd = s.solve(n_iter=50, w=0.12, verbose=False)
    fd_time = time.time() - t0

    # Surrogate — single point prediction
    t0 = time.time()
    for _ in range(100):
        predictor.predict(np.array([0.05]), np.array([0.1]),
                          700.0, 10.0, 0.0)
    surr_single = (time.time() - t0) / 100

    # Surrogate — full field prediction
    m = r_fd['mesh']
    t0 = time.time()
    for _ in range(10):
        predictor.predict_field(m.rc, m.zc, r_fd['inside'], 700.0, 10.0, 0.0)
    surr_field = (time.time() - t0) / 10

    results = {
        'fd_solver_s': fd_time,
        'surrogate_single_point_ms': surr_single * 1000,
        'surrogate_full_field_ms': surr_field * 1000,
        'speedup_vs_fd': fd_time / surr_field,
    }

    if verbose:
        print(f"\n  Runtime comparison:")
        print(f"    FD solver:           {fd_time:.2f} s")
        print(f"    Surrogate (1 point): {surr_single*1000:.2f} ms")
        print(f"    Surrogate (field):   {surr_field*1000:.1f} ms")
        print(f"    Speedup:             {fd_time/surr_field:.0f}×")

    return results


# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--ensemble', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.train:
        print("=" * 60)
        print("  TEL Spatial Surrogate v2 — Ensemble Training")
        print("=" * 60)

        train_data, val_data, meta = load_dataset(DATASET_DIR)
        print(f"  Train: {len(train_data['r'])} points, "
              f"{len(set(train_data['case'].tolist()))} cases")
        print(f"  Val:   {len(val_data['r'])} points, "
              f"{len(set(val_data['case'].tolist()))} cases")

        cfg = TrainConfig(n_epochs=args.epochs, n_ensemble=args.ensemble,
                          device=args.device)
        models, histories = train_ensemble(train_data, val_data, cfg)

        # Save ensemble
        for i, (m, h) in enumerate(zip(models, histories)):
            torch.save(m.state_dict(),
                       os.path.join(RESULTS_DIR, f'model_{i}.pt'))
        with open(os.path.join(RESULTS_DIR, 'train_history.json'), 'w') as f:
            json.dump([{k: v for k, v in h.items() if isinstance(v, (list, float))}
                       for h in histories], f)

        vals = [h['best_val'] for h in histories]
        print(f"\n  Ensemble val losses: {[f'{v:.4f}' for v in vals]}")
        print(f"  Mean: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    if args.evaluate or args.benchmark:
        # Load ensemble
        models = []
        for i in range(args.ensemble):
            p = os.path.join(RESULTS_DIR, f'model_{i}.pt')
            if os.path.exists(p):
                m = SpatialSurrogateV2()
                m.load_state_dict(torch.load(p, map_location='cpu',
                                             weights_only=True))
                models.append(m)
        if not models:
            print("No trained models found. Run --train first.")
            return

        predictor = EnsemblePredictor(models, args.device)
        _, val_data, meta = load_dataset(DATASET_DIR)

    if args.evaluate:
        print("=" * 60)
        print("  TEL Spatial Surrogate v2 — Evaluation")
        print("=" * 60)
        full_evaluation(predictor, val_data, meta, RESULTS_DIR, args.device)

    if args.benchmark:
        print("=" * 60)
        print("  Runtime Benchmark")
        print("=" * 60)
        bm = benchmark(predictor)
        with open(os.path.join(RESULTS_DIR, 'benchmark.json'), 'w') as f:
            json.dump(bm, f, indent=2)


if __name__ == '__main__':
    main()
