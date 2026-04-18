"""
Spatial error analysis: break down surrogate error by reactor region.
Regions: ICP source, aperture, processing, wafer vicinity.
Uses surrogate_v4 on legacy data and surrogate_lxcat_v3 on LXCat data.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ['MPLBACKEND'] = 'Agg'

R_PROC, Z_TOP = 0.105, 0.234
R_ICP, Z_APT_TOP, Z_APT_BOT = 0.038, 0.184, 0.182

OUT = os.path.join(REPO, 'results', 'spatial_error_analysis')
os.makedirs(OUT, exist_ok=True)


class SurrogateNet(torch.nn.Module):
    def __init__(self, n_out=2, nh=128, nl=4, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [torch.nn.Linear(ni, nh), torch.nn.GELU(), torch.nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [torch.nn.Linear(nh, nh), torch.nn.GELU(), torch.nn.Dropout(drop)]
        self.bb = torch.nn.Sequential(*layers)
        self.proj = torch.nn.Linear(ni, nh)
        self.head = torch.nn.Linear(nh, n_out)

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


def classify_region(r, z):
    """Classify a point into reactor region."""
    if z >= Z_APT_TOP and r <= R_ICP:
        return 'icp_source'
    elif Z_APT_BOT <= z < Z_APT_TOP and r <= R_ICP:
        return 'aperture'
    elif z < Z_APT_BOT and z > 0.01:
        return 'processing'
    elif z <= 0.01:
        return 'wafer_vicinity'
    else:
        return 'other'


def analyze_dataset(ds_dir, model_dir, label):
    """Load dataset, run surrogate, compute per-region errors."""
    with open(os.path.join(ds_dir, 'metadata.json')) as f:
        meta = json.load(f)
    meta = [e for e in meta if 'error' not in e]

    # Use same val split
    np.random.seed(42)
    n = len(meta)
    perm = np.random.permutation(n)
    n_val = max(int(n * 0.15), 10)
    val_idx = set(perm[:n_val].tolist())

    # Load ensemble
    models = []
    for i in range(5):
        ckpt = os.path.join(model_dir, f'model_{i}.pt')
        if not os.path.exists(ckpt):
            break
        m = SurrogateNet()
        m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        m.eval()
        models.append(m)

    if not models:
        return None

    # Collect per-region errors
    regions = {'icp_source': [], 'aperture': [], 'processing': [], 'wafer_vicinity': [], 'other': []}
    region_errors = {reg: {'nF': [], 'nSF6': []} for reg in regions}

    for entry in meta:
        if entry['idx'] not in val_idx:
            continue
        d = np.load(os.path.join(ds_dir, entry['file']))
        inside = d['inside'].astype(bool)
        rc, zc = d['rc'], d['zc']
        P, p, Ar = float(entry['P_rf']), float(entry['p_mTorr']), float(entry['frac_Ar'])

        for i in range(len(rc)):
            for j in range(len(zc)):
                if not inside[i, j]:
                    continue
                r, z = rc[i], zc[j]
                region = classify_region(r, z)
                true_nF = np.log10(max(d['nF'][i, j], 1e6))
                true_nSF6 = np.log10(max(d['nSF6'][i, j], 1e6))

                X = torch.tensor([[r / R_PROC, z / Z_TOP, P / 1200, p / 20, Ar]],
                                 dtype=torch.float32)
                preds = []
                for m in models:
                    with torch.no_grad():
                        preds.append(m(X).numpy()[0])
                pred_mean = np.mean(preds, axis=0)

                region_errors[region]['nF'].append(pred_mean[0] - true_nF)
                region_errors[region]['nSF6'].append(pred_mean[1] - true_nSF6)

    # Compute stats per region
    stats = {}
    for reg, errs in region_errors.items():
        if len(errs['nF']) == 0:
            continue
        nF_err = np.array(errs['nF'])
        nSF6_err = np.array(errs['nSF6'])
        stats[reg] = {
            'n_points': len(nF_err),
            'nF_rmse': float(np.sqrt((nF_err ** 2).mean())),
            'nF_mae': float(np.abs(nF_err).mean()),
            'nF_bias': float(nF_err.mean()),
            'nF_max_err': float(np.abs(nF_err).max()),
            'nSF6_rmse': float(np.sqrt((nSF6_err ** 2).mean())),
            'nSF6_mae': float(np.abs(nSF6_err).mean()),
            'nSF6_bias': float(nSF6_err.mean()),
            'nSF6_max_err': float(np.abs(nSF6_err).max()),
        }

    return {'label': label, 'n_models': len(models), 'regions': stats}


def main():
    print("=== Spatial Error Analysis ===\n", flush=True)

    DS_V4 = os.path.join(REPO, 'results', 'pinn_dataset_v4')
    DS_LX = os.path.join(REPO, 'results', 'pinn_dataset_lxcat_v3')
    M_V4 = os.path.join(REPO, 'results', 'surrogate_v4')
    M_LX = os.path.join(REPO, 'results', 'surrogate_lxcat_v3')

    print("Analyzing surrogate_v4 on legacy data...", flush=True)
    v4_stats = analyze_dataset(DS_V4, M_V4, 'surrogate_v4_legacy')

    print("Analyzing surrogate_lxcat_v3 on LXCat data...", flush=True)
    lx_stats = analyze_dataset(DS_LX, M_LX, 'surrogate_lxcat_v3')

    results = {'surrogate_v4': v4_stats, 'surrogate_lxcat_v3': lx_stats}

    with open(os.path.join(OUT, 'spatial_errors.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown
    md = "# Spatial Error Analysis by Reactor Region\n\n"

    for name, stats in results.items():
        if stats is None:
            continue
        md += f"## {stats['label']} ({stats['n_models']}-model ensemble)\n\n"
        md += "| Region | N pts | nF RMSE | nF MAE | nF bias | nF max | nSF6 RMSE | nSF6 max |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        for reg in ['icp_source', 'aperture', 'processing', 'wafer_vicinity']:
            if reg not in stats['regions']:
                continue
            s = stats['regions'][reg]
            md += f"| {reg} | {s['n_points']} | {s['nF_rmse']:.5f} | {s['nF_mae']:.5f} | "
            md += f"{s['nF_bias']:+.5f} | {s['nF_max_err']:.4f} | {s['nSF6_rmse']:.5f} | {s['nSF6_max_err']:.4f} |\n"
        md += "\n"

    with open(os.path.join(OUT, 'spatial_errors.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
