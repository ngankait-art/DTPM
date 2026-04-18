"""
Train surrogate_lxcat_v3: 5-model ensemble on full 221-case LXCat dataset.
Then perform fair comparison against surrogate_v4.

Architecture and hyperparameters match surrogate_v4 exactly for a fair comparison.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ['MPLBACKEND'] = 'Agg'

R_PROC, Z_TOP = 0.105, 0.234

DS_LXCAT = os.path.join(REPO, 'results', 'pinn_dataset_lxcat_v3')
DS_V4 = os.path.join(REPO, 'results', 'pinn_dataset_v4')
RES_LXCAT = os.path.join(REPO, 'results', 'surrogate_lxcat_v3')
RES_V4 = os.path.join(REPO, 'results', 'surrogate_v4')


def select_device():
    if torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    return torch.device('cpu'), "CPU"


class SurrogateNet(nn.Module):
    def __init__(self, n_out=2, nh=128, nl=4, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, n_out)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


def load_dataset(dataset_dir, val_frac=0.15):
    with open(os.path.join(dataset_dir, 'metadata.json')) as f:
        meta = json.load(f)
    # Filter out failed cases
    meta = [e for e in meta if 'error' not in e]
    n = len(meta)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = max(int(n * val_frac), 10)
    val_idx = set(perm[:n_val].tolist())

    data = {k: [] for k in ['r', 'z', 'P', 'p', 'Ar', 'lnF', 'lnSF6', 'case']}
    for entry in meta:
        fpath = os.path.join(dataset_dir, entry['file'])
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
    split = lambda m: {k: v[m] for k, v in arrays.items()}
    return split(mask), split(~mask), meta, val_idx


def to_tensors(data, device):
    X = np.column_stack([
        data['r'] / R_PROC, data['z'] / Z_TOP,
        data['P'] / 1200, data['p'] / 20, data['Ar']
    ])
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


def train_ensemble(dataset_dir, results_dir, label, dev, n_ens=5, n_ep=1500):
    os.makedirs(results_dir, exist_ok=True)

    # Check for existing complete ensemble
    existing = [i for i in range(n_ens)
                if os.path.exists(os.path.join(results_dir, f'model_{i}.pt'))]
    if len(existing) == n_ens and os.path.exists(os.path.join(results_dir, 'summary.json')):
        print(f"  {label}: all {n_ens} models already complete. Skipping.", flush=True)
        return json.load(open(os.path.join(results_dir, 'summary.json')))

    train_d, val_d, meta, vi = load_dataset(dataset_dir)
    Xt, Yt = to_tensors(train_d, dev)
    Xv, Yv = to_tensors(val_d, dev)
    n_out = 2
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    sw = torch.tensor([1.0, 1.5], device=dev)

    print(f"  Train: {Xt.shape[0]} pts, Val: {Xv.shape[0]} pts, Outputs: {n_out}", flush=True)
    models, vals_list = [], []
    t0 = time.time()

    for i in range(n_ens):
        ckpt = os.path.join(results_dir, f'model_{i}.pt')
        if os.path.exists(ckpt):
            print(f"  === MODEL {i}/{n_ens} ALREADY EXISTS --- loading ===", flush=True)
            m = SurrogateNet(n_out=n_out).to(dev)
            m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
            m.eval()
            with torch.no_grad():
                vp = m(Xv)
                vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
            models.append(m.cpu())
            vals_list.append(vl)
            print(f"  === RESUMED MODEL {i}/{n_ens} | val = {vl:.6f} ===", flush=True)
            continue

        print(f"  === STARTING MODEL {i}/{n_ens} ===", flush=True)
        torch.manual_seed(42 + i * 137)
        np.random.seed(42 + i * 137)
        m = SurrogateNet(n_out=n_out).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
        n = Xt.shape[0]
        best_v, best_s = float('inf'), None

        for ep in range(n_ep):
            m.train()
            pm = torch.randperm(n, device=dev)
            el, nb = 0, 0
            for s in range(0, n, 4096):
                idx = pm[s:s + 4096]
                pred = m(Xt[idx])
                loss = (sw * ((pred - Yt[idx]) / ys).pow(2)).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                el += loss.item()
                nb += 1
            sch.step()
            m.eval()
            with torch.no_grad():
                vp = m(Xv)
                vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
            if vl < best_v:
                best_v = vl
                best_s = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            if ep % max(n_ep // 5, 1) == 0:
                print(f"    M{i} [{ep:5d}] t={el/nb:.5f} v={vl:.5f}", flush=True)

        if best_s:
            m.load_state_dict(best_s)
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, ckpt)
        models.append(m.cpu())
        vals_list.append(best_v)
        print(f"  === FINISHED MODEL {i}/{n_ens} | best val = {best_v:.6f} ===", flush=True)
        print(f"  Running ensemble: mean={np.mean(vals_list):.6f} std={np.std(vals_list):.6f}", flush=True)

    tt = time.time() - t0
    print(f"  === ALL MODELS COMPLETE ({tt:.0f}s) ===", flush=True)

    # Evaluate
    Xvc = Xv.cpu()
    Yv_np = Yv.cpu().numpy()
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(Xvc).numpy())
    preds_arr = np.stack(preds)
    pm_mean = preds_arr.mean(0)
    pm_std = preds_arr.std(0)

    metrics = {}
    names = ['nF', 'nSF6']
    for c, nm in enumerate(names):
        err = pm_mean[:, c] - Yv_np[:, c]
        metrics[nm] = {
            'rmse': float(np.sqrt((err ** 2).mean())),
            'mae': float(np.abs(err).mean()),
            'max_err': float(np.abs(err).max()),
            'ens_spread': float(pm_std[:, c].mean()),
        }

    summary = {
        'label': label, 'device': str(dev), 'n_cases': len(meta),
        'n_out': 2, 'outputs': names,
        'train_time_s': tt, 'ensemble_vals': [float(v) for v in vals_list],
        'ens_mean': float(np.mean(vals_list)), 'ens_std': float(np.std(vals_list)),
        'metrics': metrics,
        'rate_source': 'lxcat',
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def fair_comparison(lxcat_summary):
    """Compare surrogate_lxcat_v3 against surrogate_v4 fairly."""
    v4_summary_path = os.path.join(RES_V4, 'summary.json')
    if not os.path.exists(v4_summary_path):
        print("  WARNING: surrogate_v4 summary not found!", flush=True)
        return None

    v4 = json.load(open(v4_summary_path))

    # Extract metrics - v4 may have different structure
    def get_metric(summary, species, metric='rmse'):
        # Check under 'metrics' first, then top-level (v4 format)
        m = summary.get('metrics', {})
        if species in m and isinstance(m[species], dict):
            return m[species].get(metric, None)
        top = summary.get(species, {})
        if isinstance(top, dict):
            return top.get(metric, None)
        return None

    v4_nF = get_metric(v4, 'nF')
    v4_nSF6 = get_metric(v4, 'nSF6')
    lx_nF = lxcat_summary['metrics']['nF']['rmse']
    lx_nSF6 = lxcat_summary['metrics']['nSF6']['rmse']

    comparison = {
        'title': 'Fair comparison: surrogate_lxcat_v3 vs surrogate_v4',
        'conditions': {
            'both_221_cases': True,
            'same_architecture': 'Fourier + 4x128 GELU MLP',
            'same_hyperparameters': '1500 epochs, lr=1e-3, cosine schedule',
            'same_ensemble_size': 5,
            'same_val_split': '15% (seed=42)',
            'difference': 'rate_mode: legacy vs lxcat',
        },
        'surrogate_v4': {
            'dataset': 'pinn_dataset_v4 (legacy Arrhenius)',
            'n_cases': v4.get('n_cases', 221),
            'nF_rmse': v4_nF,
            'nSF6_rmse': v4_nSF6,
            'ens_mean': v4.get('ens_mean'),
        },
        'surrogate_lxcat_v3': {
            'dataset': 'pinn_dataset_lxcat_v3 (LXCat Biagi)',
            'n_cases': lxcat_summary['n_cases'],
            'nF_rmse': lx_nF,
            'nSF6_rmse': lx_nSF6,
            'ens_mean': lxcat_summary['ens_mean'],
        },
    }

    # Compute ratios
    if v4_nF and lx_nF:
        comparison['nF_ratio'] = lx_nF / v4_nF
    if v4_nSF6 and lx_nSF6:
        comparison['nSF6_ratio'] = lx_nSF6 / v4_nSF6

    # Interpretation
    if v4_nF and lx_nF:
        ratio = lx_nF / v4_nF
        if ratio < 1.5:
            comparison['interpretation'] = (
                'LXCat-backed surrogate achieves comparable accuracy to legacy. '
                'The physics change (stronger attachment, higher Te) does not '
                'significantly degrade learnability.'
            )
        elif ratio < 3.0:
            comparison['interpretation'] = (
                'LXCat-backed surrogate shows moderate accuracy degradation. '
                'The shifted Te/ne distributions from LXCat rates create a '
                'harder learning problem but the surrogate still captures the '
                'main physics.'
            )
        else:
            comparison['interpretation'] = (
                'LXCat-backed surrogate shows significant accuracy degradation. '
                'The strong physics shift (Te +56%, ne -69%) may require '
                'architecture tuning or more training data in the LXCat regime.'
            )

    outpath = os.path.join(REPO, 'results', 'fair_comparison_v4_vs_lxcat_v3.json')
    with open(outpath, 'w') as f:
        json.dump(comparison, f, indent=2)
    return comparison


def main():
    dev, dev_name = select_device()
    print(f"{'=' * 60}", flush=True)
    print(f"  surrogate_lxcat_v3 Training --- Device: {dev} ({dev_name})", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Phase 1: Train
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 1: Train 5-model ensemble on LXCat v3 dataset", flush=True)
    print(f"{'=' * 60}", flush=True)
    summary = train_ensemble(DS_LXCAT, RES_LXCAT, 'surrogate_lxcat_v3', dev)
    print(f"  Metrics: {json.dumps(summary['metrics'], indent=2)}", flush=True)

    # Phase 2: Fair comparison
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 2: Fair comparison vs surrogate_v4", flush=True)
    print(f"{'=' * 60}", flush=True)
    comp = fair_comparison(summary)
    if comp:
        print(f"\n{json.dumps(comp, indent=2)}", flush=True)

    # Phase 3: Update master comparison
    print(f"\n{'=' * 60}", flush=True)
    print(f"  PHASE 3: Update master comparison", flush=True)
    print(f"{'=' * 60}", flush=True)
    master_path = os.path.join(REPO, 'results', 'model_comparison_master_v3.json')
    v4_path = os.path.join(RES_V4, 'summary.json')
    v4 = json.load(open(v4_path)) if os.path.exists(v4_path) else {}
    lxv2_path = os.path.join(REPO, 'results', 'surrogate_lxcat_v2', 'summary.json')
    lxv2 = json.load(open(lxv2_path)) if os.path.exists(lxv2_path) else {}
    te_path = os.path.join(REPO, 'results', 'surrogate_te_v1', 'summary.json')
    te_v1 = json.load(open(te_path)) if os.path.exists(te_path) else {}

    def safe_get(d, *keys, default='N/A'):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    master = {
        'surrogate_v4': {
            'dataset': 'pinn_dataset_v4 (221 legacy cases)',
            'outputs': ['nF', 'nSF6'],
            'nF_rmse': safe_get(v4, 'nF', 'rmse') if isinstance(v4.get('nF'), dict) else safe_get(v4, 'metrics', 'nF', 'rmse'),
            'nSF6_rmse': safe_get(v4, 'nSF6', 'rmse') if isinstance(v4.get('nSF6'), dict) else safe_get(v4, 'metrics', 'nSF6', 'rmse'),
            'ens_mean': safe_get(v4, 'ens_mean'),
            'status': 'Production',
        },
        'surrogate_lxcat_v3': {
            'dataset': 'pinn_dataset_lxcat_v3 (221 LXCat cases)',
            'outputs': ['nF', 'nSF6'],
            'nF_rmse': summary['metrics']['nF']['rmse'],
            'nSF6_rmse': summary['metrics']['nSF6']['rmse'],
            'ens_mean': summary['ens_mean'],
            'status': 'Fair comparison',
        },
        'surrogate_lxcat_v2': {
            'dataset': 'pinn_dataset_lxcat_v2 (30 LXCat cases)',
            'outputs': ['nF', 'nSF6'],
            'nF_rmse': safe_get(lxv2, 'metrics', 'nF', 'rmse'),
            'nSF6_rmse': safe_get(lxv2, 'metrics', 'nSF6', 'rmse'),
            'ens_mean': safe_get(lxv2, 'ens_mean'),
            'status': 'Superseded by v3',
        },
        'surrogate_te_v1': {
            'dataset': 'pinn_dataset_v4 (221 legacy cases)',
            'outputs': ['Te'],
            'Te_rmse': safe_get(te_v1, 'metrics', 'Te', 'rmse'),
            'ens_mean': safe_get(te_v1, 'ens_mean'),
            'status': 'Exploratory',
        },
        'fair_comparison': comp,
    }

    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2)
    print(f"  Master comparison written to {master_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  ALL PHASES COMPLETE", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == '__main__':
    main()
