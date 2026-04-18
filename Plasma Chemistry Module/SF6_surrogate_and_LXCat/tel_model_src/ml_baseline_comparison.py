"""
Simple ML baseline: Gaussian Process and polynomial regression on the same
LXCat dataset and split. Proves that the Fourier-feature MLP is meaningfully
better than off-the-shelf methods.
"""
import os, sys, json, time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

R_PROC, Z_TOP = 0.105, 0.234
DS_LXCAT = os.path.join(REPO, 'results', 'pinn_dataset_lxcat_v3')
OUT = os.path.join(REPO, 'results', 'ml_baseline_comparison')
os.makedirs(OUT, exist_ok=True)


def load_dataset(val_frac=0.15, subsample=5000):
    """Load dataset with optional subsampling for GP (which is O(n^3))."""
    with open(os.path.join(DS_LXCAT, 'metadata.json')) as f:
        meta = json.load(f)
    meta = [e for e in meta if 'error' not in e]
    n = len(meta)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = max(int(n * val_frac), 10)
    val_idx = set(perm[:n_val].tolist())

    data = {k: [] for k in ['r', 'z', 'P', 'p', 'Ar', 'lnF', 'lnSF6', 'case']}
    for entry in meta:
        fpath = os.path.join(DS_LXCAT, entry['file'])
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        inside = d['inside'].astype(bool)
        rc, zc = d['rc'], d['zc']
        for i in range(len(rc)):
            for j in range(len(zc)):
                if inside[i, j]:
                    data['r'].append(rc[i])
                    data['z'].append(zc[j])
                    data['P'].append(float(entry['P_rf']))
                    data['p'].append(float(entry['p_mTorr']))
                    data['Ar'].append(float(entry['frac_Ar']))
                    data['lnF'].append(np.log10(max(d['nF'][i, j], 1e6)))
                    data['lnSF6'].append(np.log10(max(d['nSF6'][i, j], 1e6)))
                    data['case'].append(entry['idx'])

    arrays = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    arrays['case'] = arrays['case'].astype(np.int32)
    mask = np.array([c not in val_idx for c in arrays['case']])

    X_all = np.column_stack([arrays['r'] / R_PROC, arrays['z'] / Z_TOP,
                             arrays['P'] / 1200, arrays['p'] / 20, arrays['Ar']])
    Y_all = np.column_stack([arrays['lnF'], arrays['lnSF6']])

    X_train, Y_train = X_all[mask], Y_all[mask]
    X_val, Y_val = X_all[~mask], Y_all[~mask]

    # Subsample training for GP
    if subsample and len(X_train) > subsample:
        idx = np.random.choice(len(X_train), subsample, replace=False)
        X_train_gp = X_train[idx]
        Y_train_gp = Y_train[idx]
    else:
        X_train_gp = X_train
        Y_train_gp = Y_train

    return X_train, Y_train, X_val, Y_val, X_train_gp, Y_train_gp


def evaluate(Y_pred, Y_true):
    metrics = {}
    for c, nm in enumerate(['nF', 'nSF6']):
        err = Y_pred[:, c] - Y_true[:, c]
        metrics[nm] = {
            'rmse': float(np.sqrt((err ** 2).mean())),
            'mae': float(np.abs(err).mean()),
            'max_err': float(np.abs(err).max()),
        }
    return metrics


def main():
    print("=== ML Baseline Comparison ===\n", flush=True)

    print("Loading dataset...", flush=True)
    X_train, Y_train, X_val, Y_val, X_train_gp, Y_train_gp = load_dataset(subsample=3000)
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, GP subset: {X_train_gp.shape[0]}", flush=True)

    results = {}

    # 1. Polynomial regression (degree 3)
    print("\nTraining polynomial regression (degree 3)...", flush=True)
    t0 = time.time()
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=3, interaction_only=False)),
        ('ridge', Ridge(alpha=1e-3)),
    ])
    poly.fit(X_train, Y_train)
    Y_pred_poly = poly.predict(X_val)
    dt = time.time() - t0
    metrics_poly = evaluate(Y_pred_poly, Y_val)
    results['polynomial_deg3'] = {
        'method': 'Polynomial regression (degree 3) + Ridge',
        'train_time_s': dt,
        'n_train': len(X_train),
        'metrics': metrics_poly,
    }
    print(f"  nF RMSE: {metrics_poly['nF']['rmse']:.5f}, nSF6 RMSE: {metrics_poly['nSF6']['rmse']:.5f} ({dt:.1f}s)", flush=True)

    # 2. Polynomial regression (degree 4)
    print("\nTraining polynomial regression (degree 4)...", flush=True)
    t0 = time.time()
    poly4 = Pipeline([
        ('poly', PolynomialFeatures(degree=4, interaction_only=False)),
        ('ridge', Ridge(alpha=1e-3)),
    ])
    poly4.fit(X_train, Y_train)
    Y_pred_poly4 = poly4.predict(X_val)
    dt = time.time() - t0
    metrics_poly4 = evaluate(Y_pred_poly4, Y_val)
    results['polynomial_deg4'] = {
        'method': 'Polynomial regression (degree 4) + Ridge',
        'train_time_s': dt,
        'n_train': len(X_train),
        'metrics': metrics_poly4,
    }
    print(f"  nF RMSE: {metrics_poly4['nF']['rmse']:.5f}, nSF6 RMSE: {metrics_poly4['nSF6']['rmse']:.5f} ({dt:.1f}s)", flush=True)

    # 3. Gaussian Process (on subset)
    print(f"\nTraining Gaussian Process (n={len(X_train_gp)})...", flush=True)
    t0 = time.time()
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(5)) + WhiteKernel(noise_level=1e-3)
    gp_nF = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    gp_nSF6 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    gp_nF.fit(X_train_gp, Y_train_gp[:, 0])
    gp_nSF6.fit(X_train_gp, Y_train_gp[:, 1])
    pred_nF = gp_nF.predict(X_val)
    pred_nSF6 = gp_nSF6.predict(X_val)
    Y_pred_gp = np.column_stack([pred_nF, pred_nSF6])
    dt = time.time() - t0
    metrics_gp = evaluate(Y_pred_gp, Y_val)
    results['gaussian_process'] = {
        'method': 'Gaussian Process (RBF kernel)',
        'train_time_s': dt,
        'n_train': len(X_train_gp),
        'note': f'Subsampled to {len(X_train_gp)} for O(n^3) feasibility',
        'metrics': metrics_gp,
    }
    print(f"  nF RMSE: {metrics_gp['nF']['rmse']:.5f}, nSF6 RMSE: {metrics_gp['nSF6']['rmse']:.5f} ({dt:.1f}s)", flush=True)

    # Reference numbers — only from completed, confirmed models
    results['references'] = {
        'surrogate_lxcat_v3': {'nF_rmse': 0.0112, 'nSF6_rmse': 0.0081, 'method': '5-model Fourier MLP ensemble (no phys reg)'},
        'surrogate_v4': {'nF_rmse': 0.0029, 'nSF6_rmse': 0.0027, 'method': '5-model Fourier MLP ensemble (phys reg, legacy data)'},
    }
    # Add architecture sweep winner if available
    sweep_path = os.path.join(REPO, 'results', 'lxcat_architecture_upgrade', 'experiment_table.json')
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep = json.load(f)
        exps = sweep.get('experiments', [])
        if exps:
            best = min(exps, key=lambda e: e['nF_rmse_mean'])
            results['references']['arch_sweep_winner'] = {
                'name': best['name'],
                'nF_rmse': best['nF_rmse_mean'],
                'nSF6_rmse': best['nSF6_rmse_mean'],
                'method': f"{best['name']} (single model, phys reg)",
            }

    with open(os.path.join(OUT, 'baseline_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown
    md = "# ML Baseline Comparison\n\n"
    md += "Same LXCat dataset (pinn_dataset_lxcat_v3, 221 cases), same val split.\n\n"
    md += "| Method | N train | nF RMSE | nSF6 RMSE | Train time |\n"
    md += "|---|---|---|---|---|\n"
    for name, r in results.items():
        if name == 'references':
            continue
        md += f"| {r['method']} | {r['n_train']:,} | {r['metrics']['nF']['rmse']:.5f} | {r['metrics']['nSF6']['rmse']:.5f} | {r['train_time_s']:.1f}s |\n"
    md += f"| surrogate_lxcat_v3 (5-ens, no reg) | 120,696 | 0.01120 | 0.00810 | ~1200s |\n"
    md += f"| surrogate_v4 (5-ens, legacy, phys reg) | 120,696 | 0.00290 | 0.00270 | ~3700s |\n"
    if 'arch_sweep_winner' in results.get('references', {}):
        w = results['references']['arch_sweep_winner']
        md += f"| **{w['name']}** (single, phys reg) | 120,696 | **{w['nF_rmse']:.5f}** | **{w['nSF6_rmse']:.5f}** | — |\n"

    with open(os.path.join(OUT, 'baseline_comparison.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
