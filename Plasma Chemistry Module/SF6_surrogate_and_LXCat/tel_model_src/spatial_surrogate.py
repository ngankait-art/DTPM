"""
Spatial field surrogate for the TEL ICP reactor.

Learns n_F(r, z, P_rf, p_mTorr, frac_Ar) and n_SF6(...) from FD solver
outputs. This is the production hybrid model — see PINN_STRATEGY_DECISION.md.

Usage:
    # Generate dataset (if not already done):
    python spatial_surrogate.py --generate

    # Train:
    python spatial_surrogate.py --train

    # Evaluate:
    python spatial_surrogate.py --evaluate
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
DATASET_DIR = os.path.join(REPO, 'results', 'pinn_dataset')
RESULTS_DIR = os.path.join(REPO, 'results', 'surrogate')

# ═══════════════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SurrogateDataset:
    """Consolidated dataset from multiple FD solver runs."""
    r: np.ndarray         # (N_total,) radial coords
    z: np.ndarray         # (N_total,) axial coords
    P_rf: np.ndarray      # (N_total,) power
    p_mTorr: np.ndarray   # (N_total,) pressure
    frac_Ar: np.ndarray   # (N_total,) Ar fraction
    log_nF: np.ndarray    # (N_total,) log10(n_F)
    log_nSF6: np.ndarray  # (N_total,) log10(n_SF6)
    case_idx: np.ndarray  # (N_total,) which case each point came from
    n_cases: int
    metadata: list

    @staticmethod
    def load(dataset_dir: str, val_indices: Optional[List[int]] = None
             ) -> Tuple['SurrogateDataset', 'SurrogateDataset']:
        """Load and split dataset into train/val by case index.

        val_indices: which case indices to hold out. Default: last 2.
        """
        with open(os.path.join(dataset_dir, 'metadata.json')) as f:
            meta = json.load(f)

        if val_indices is None:
            val_indices = [len(meta) - 2, len(meta) - 1]

        all_r, all_z, all_P, all_p, all_Ar = [], [], [], [], []
        all_lnF, all_lnSF6, all_case = [], [], []

        for entry in meta:
            d = np.load(os.path.join(dataset_dir, entry['file']))
            inside = d['inside'].astype(bool)
            rc, zc = d['rc'], d['zc']
            nF, nSF6 = d['nF'], d['nSF6']

            for i in range(len(rc)):
                for j in range(len(zc)):
                    if inside[i, j]:
                        all_r.append(rc[i])
                        all_z.append(zc[j])
                        all_P.append(float(entry['P_rf']))
                        all_p.append(float(entry['p_mTorr']))
                        all_Ar.append(float(entry['frac_Ar']))
                        all_lnF.append(np.log10(max(nF[i, j], 1e6)))
                        all_lnSF6.append(np.log10(max(nSF6[i, j], 1e6)))
                        all_case.append(entry['idx'])

        r = np.array(all_r, dtype=np.float32)
        z = np.array(all_z, dtype=np.float32)
        P = np.array(all_P, dtype=np.float32)
        p = np.array(all_p, dtype=np.float32)
        Ar = np.array(all_Ar, dtype=np.float32)
        lnF = np.array(all_lnF, dtype=np.float32)
        lnSF6 = np.array(all_lnSF6, dtype=np.float32)
        case = np.array(all_case, dtype=np.int32)

        train_mask = np.array([c not in val_indices for c in case])
        val_mask = ~train_mask

        def _make(mask):
            return SurrogateDataset(
                r=r[mask], z=z[mask], P_rf=P[mask], p_mTorr=p[mask],
                frac_Ar=Ar[mask], log_nF=lnF[mask], log_nSF6=lnSF6[mask],
                case_idx=case[mask], n_cases=len(meta), metadata=meta)

        return _make(train_mask), _make(val_mask)


def _to_input_tensor(ds: SurrogateDataset, device: torch.device
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset to normalised input/target tensors."""
    R_PROC, Z_TOP = 0.105, 0.234
    X = np.column_stack([
        ds.r / R_PROC,
        ds.z / Z_TOP,
        ds.P_rf / 1000.0,
        ds.p_mTorr / 20.0,
        ds.frac_Ar,
    ])
    Y = np.column_stack([ds.log_nF, ds.log_nSF6])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


# ═══════════════════════════════════════════════════════════════════
# 2. MODEL
# ═══════════════════════════════════════════════════════════════════

class FourierFeatures(nn.Module):
    """Random Fourier embedding for continuous inputs."""
    def __init__(self, n_input: int = 5, n_features: int = 64,
                 scale: float = 3.0):
        super().__init__()
        B = torch.randn(n_input, n_features) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SpatialSurrogate(nn.Module):
    """Neural field surrogate: (r, z, P, p, Ar) → (log10 nF, log10 nSF6).

    Architecture: Fourier embedding + residual GELU MLP.
    """
    def __init__(self, n_hidden: int = 128, n_layers: int = 4,
                 n_fourier: int = 64, fourier_scale: float = 3.0):
        super().__init__()
        self.ff = FourierFeatures(5, n_fourier, fourier_scale)
        n_in = 2 * n_fourier

        layers = [nn.Linear(n_in, n_hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.GELU()])
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
        # Bias head to typical log10(n) values
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.ff(x)
        h = self.backbone(e) + self.proj(e)
        return self.head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    n_epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 2048
    scheduler_patience: int = 200
    device: str = 'cpu'


def train(model: SpatialSurrogate, train_ds: SurrogateDataset,
          val_ds: SurrogateDataset, cfg: TrainConfig,
          verbose: bool = True) -> Dict:
    """Train the spatial surrogate."""
    device = torch.device(cfg.device)
    model = model.to(device)

    X_train, Y_train = _to_input_tensor(train_ds, device)
    X_val, Y_val = _to_input_tensor(val_ds, device)

    # Compute target statistics for normalised loss
    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True).clamp(min=1e-3)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=cfg.scheduler_patience, factor=0.5, min_lr=1e-6)

    n_train = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')
    best_state = None

    if verbose:
        print(f"  Train: {n_train} points from {len(set(train_ds.case_idx.tolist()))} cases")
        print(f"  Val:   {X_val.shape[0]} points from {len(set(val_ds.case_idx.tolist()))} cases")
        print(f"  Params: {model.num_parameters():,}")
        print()

    t0 = time.time()

    for epoch in range(cfg.n_epochs):
        model.train()

        # Mini-batch SGD
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb, yb = X_train[idx], Y_train[idx]

            pred = model(xb)
            loss = ((pred - yb) / y_std).pow(2).mean()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = ((val_pred - Y_val) / y_std).pow(2).mean().item()

        scheduler.step(val_loss)
        lr = optimiser.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and epoch % max(cfg.n_epochs // 20, 1) == 0:
            print(f"  [{epoch:5d}/{cfg.n_epochs}] "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.1e}")

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n  Done ({elapsed:.1f}s). Best val loss: {best_val:.4f}")

    # Compute final metrics in physical units
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).cpu().numpy()
        val_pred = model(X_val).cpu().numpy()

    train_target = Y_train.cpu().numpy()
    val_target = Y_val.cpu().numpy()

    def _metrics(pred, target, name):
        err = pred - target
        mae = np.abs(err).mean(axis=0)
        rmse = np.sqrt((err**2).mean(axis=0))
        # Convert log10 error to percentage in linear space
        # |10^(pred) - 10^(true)| / 10^(true) ≈ |ln(10) * err| for small err
        pct = np.abs(err) * np.log(10) * 100  # approximate % error
        return {
            f'{name}_mae_log10_nF': float(mae[0]),
            f'{name}_mae_log10_nSF6': float(mae[1]),
            f'{name}_rmse_log10_nF': float(rmse[0]),
            f'{name}_rmse_log10_nSF6': float(rmse[1]),
            f'{name}_median_pct_nF': float(np.median(pct[:, 0])),
            f'{name}_median_pct_nSF6': float(np.median(pct[:, 1])),
        }

    metrics = {**_metrics(train_pred, train_target, 'train'),
               **_metrics(val_pred, val_target, 'val'),
               'best_val_loss': best_val,
               'elapsed_s': elapsed,
               'n_params': model.num_parameters()}

    history['metrics'] = metrics
    return history


# ═══════════════════════════════════════════════════════════════════
# 4. EVALUATION AND PLOTTING
# ═══════════════════════════════════════════════════════════════════

def evaluate_and_plot(model: SpatialSurrogate, val_ds: SurrogateDataset,
                      output_dir: str, device: str = 'cpu'):
    """Generate evaluation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)
    model = model.to(dev).eval()

    X_val, Y_val = _to_input_tensor(val_ds, dev)
    with torch.no_grad():
        pred = model(X_val).cpu().numpy()
    target = Y_val.cpu().numpy()

    # --- Plot 1: pred vs true scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, name in enumerate(['nF', 'nSF6']):
        ax = axes[col]
        p, t = pred[:, col], target[:, col]
        ax.scatter(t, p, s=4, alpha=0.4, edgecolors='none')
        lim = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlabel(f'FD solver log10({name})')
        ax.set_ylabel(f'Surrogate log10({name})')
        ax.set_title(f'Validation: {name}')
        ax.set_aspect('equal')
        rmse = float(np.sqrt(((p - t)**2).mean()))
        ax.text(0.05, 0.92, f'RMSE={rmse:.4f}',
                transform=ax.transAxes, fontsize=10)
    fig.suptitle('Spatial Surrogate — Held-out Validation', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'val_pred_vs_true.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved val_pred_vs_true.png")

    # --- Plot 2: radial profile at wafer (z=0) for a held-out case ---
    # Find a held-out case
    cases_in_val = sorted(set(val_ds.case_idx.tolist()))
    if len(cases_in_val) > 0:
        case_id = cases_in_val[0]
        mask = val_ds.case_idx == case_id
        r_case = val_ds.r[mask]
        z_case = val_ds.z[mask]
        lnF_true = val_ds.log_nF[mask]

        # Get wafer points (z ≈ 0)
        z_min = z_case.min()
        wafer = np.abs(z_case - z_min) < 1e-3
        if wafer.sum() > 2:
            r_w = r_case[wafer]
            lnF_true_w = lnF_true[wafer]
            order = np.argsort(r_w)
            r_w = r_w[order]
            lnF_true_w = lnF_true_w[order]

            # Predict
            X_w = np.column_stack([
                r_w / 0.105,
                np.full_like(r_w, z_min / 0.234),
                np.full_like(r_w, val_ds.P_rf[mask][wafer][order] / 1000),
                np.full_like(r_w, val_ds.p_mTorr[mask][wafer][order] / 20),
                np.full_like(r_w, val_ds.frac_Ar[mask][wafer][order]),
            ])
            with torch.no_grad():
                pred_w = model(torch.tensor(X_w, dtype=torch.float32,
                                            device=dev)).cpu().numpy()

            meta_entry = val_ds.metadata[case_id]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(r_w * 1e3, 10**lnF_true_w, 'ko-', label='FD solver', ms=4)
            ax.plot(r_w * 1e3, 10**pred_w[:, 0], 'r^--', label='Surrogate', ms=4)
            ax.set_xlabel('r (mm)')
            ax.set_ylabel('[F] (m⁻³)')
            ax.set_title(f'Wafer F profile — held-out case '
                         f'(P={meta_entry["P_rf"]}W, p={meta_entry["p_mTorr"]}mT, '
                         f'Ar={meta_entry["frac_Ar"]:.0%})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'val_wafer_profile.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved val_wafer_profile.png")

    # --- Plot 3: training loss curve ---
    hist_path = os.path.join(output_dir, 'history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(hist['train_loss'], label='Train', alpha=0.7)
        ax.semilogy(hist['val_loss'], label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalised MSE')
        ax.set_title('Spatial Surrogate Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved loss_curve.png")


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TEL Spatial Surrogate')
    parser.add_argument('--generate', action='store_true',
                        help='Generate training dataset from FD solver')
    parser.add_argument('--train', action='store_true',
                        help='Train the surrogate')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate and plot')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.generate:
        print("Dataset generation: use the script that produced results/pinn_dataset/")
        print("(Already done. See metadata.json for 12 cases.)")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.train:
        print("=" * 60)
        print("  TEL Spatial Surrogate — Training")
        print("=" * 60)

        train_ds, val_ds = SurrogateDataset.load(
            DATASET_DIR, val_indices=[10, 11])  # hold out cases 10, 11

        model = SpatialSurrogate()
        cfg = TrainConfig(n_epochs=args.epochs, device=args.device)

        history = train(model, train_ds, val_ds, cfg)

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': history['metrics'],
        }, os.path.join(RESULTS_DIR, 'surrogate.pt'))

        with open(os.path.join(RESULTS_DIR, 'history.json'), 'w') as f:
            json.dump({k: v for k, v in history.items() if k != 'metrics'},
                      f)

        with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
            json.dump(history['metrics'], f, indent=2)

        print(f"\n  Metrics:")
        for k, v in sorted(history['metrics'].items()):
            print(f"    {k}: {v}")

    if args.evaluate:
        print("=" * 60)
        print("  TEL Spatial Surrogate — Evaluation")
        print("=" * 60)

        _, val_ds = SurrogateDataset.load(
            DATASET_DIR, val_indices=[10, 11])

        model = SpatialSurrogate()
        ckpt = torch.load(os.path.join(RESULTS_DIR, 'surrogate.pt'),
                          map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        evaluate_and_plot(model, val_ds, RESULTS_DIR, args.device)


if __name__ == '__main__':
    main()
