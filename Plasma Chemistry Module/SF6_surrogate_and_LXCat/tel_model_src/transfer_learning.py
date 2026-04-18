"""
Transfer learning: initialize from surrogate_v4 weights, fine-tune on LXCat data.
Tests whether legacy-learned spatial features transfer to the LXCat regime.

Uses same architecture as surrogate_v4 (Fourier + 4x128 GELU) and same val split.
3 seeds for stability comparison against training from scratch.
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
V4_DIR = os.path.join(REPO, 'results', 'surrogate_v4')
OUT = os.path.join(REPO, 'results', 'transfer_learning')
os.makedirs(OUT, exist_ok=True)


class SurrogateNet(nn.Module):
    def __init__(self, nh=128, nl=4, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


def reg_smoothness(model, xb, device):
    xg = xb.detach().clone().requires_grad_(True)
    pred = model(xg)
    L = torch.tensor(0.0, device=device)
    for c in range(2):
        g = torch.autograd.grad(pred[:, c].sum(), xg, create_graph=True)[0]
        L = L + (g[:, 0] ** 2 + g[:, 1] ** 2).mean()
    return L


def reg_bounded_density(pred, y_std):
    lo, hi = 17.0, 21.5
    below = torch.relu(lo - pred).pow(2).mean()
    above = torch.relu(pred - hi).pow(2).mean()
    return (below + above) / y_std.mean() ** 2


def reg_wafer_smoothness(model, data, device):
    r = torch.linspace(0.001, 0.95, 30, device=device)
    z = torch.full((30,), 0.01, device=device)
    idx = np.random.randint(0, len(data['P']), 1)[0]
    P = torch.full((30,), data['P'][idx] / 1200, device=device)
    p = torch.full((30,), data['p'][idx] / 20, device=device)
    Ar = torch.full((30,), data['Ar'][idx], device=device)
    x = torch.stack([r, z, P, p, Ar], dim=-1)
    pred = model(x)
    d2 = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    return d2.pow(2).mean()


def load_dataset(val_frac=0.15):
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
    split = lambda m: {k: v[m] for k, v in arrays.items()}
    return split(mask), split(~mask)


def to_tensors(data, device):
    X = np.column_stack([data['r'] / R_PROC, data['z'] / Z_TOP,
                         data['P'] / 1200, data['p'] / 20, data['Ar']])
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


def train_and_eval(model, Xt, Yt, Xv, Yv, train_data, dev, n_ep=2000, lr=1e-3, label=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
    sw = torch.tensor([1.0, 1.5], device=dev)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    n = Xt.shape[0]
    best_v, best_s = float('inf'), None

    for ep in range(n_ep):
        model.train()
        pm = torch.randperm(n, device=dev)
        el, nb = 0, 0
        for s in range(0, n, 4096):
            idx = pm[s:s + 4096]
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)
            Ld = (sw * ((pred - yb) / ys).pow(2)).mean()
            loss = Ld
            if nb % 4 == 0:
                loss = loss + 5e-4 * reg_smoothness(model, xb, dev)
                loss = loss + 1e-3 * reg_bounded_density(pred, ys)
            if nb % 8 == 0:
                loss = loss + 2e-4 * reg_wafer_smoothness(model, train_data, dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += Ld.item()
            nb += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv)
            vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
        if vl < best_v:
            best_v = vl
            best_s = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 400 == 0:
            print(f"    {label} [{ep:5d}] t={el / nb:.5f} v={vl:.5f}", flush=True)

    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).cpu().numpy()
    Yv_np = Yv.cpu().numpy()
    metrics = {}
    for c, nm in enumerate(['nF', 'nSF6']):
        err = pred[:, c] - Yv_np[:, c]
        metrics[nm] = {'rmse': float(np.sqrt((err ** 2).mean())),
                       'mae': float(np.abs(err).mean())}
    return metrics, best_v


def main():
    dev = torch.device('cpu')
    print(f"{'=' * 60}", flush=True)
    print(f"  Transfer Learning: v4 -> LXCat", flush=True)
    print(f"  Device: CPU", flush=True)
    print(f"{'=' * 60}", flush=True)

    train_d, val_d = load_dataset()
    Xt, Yt = to_tensors(train_d, dev)
    Xv, Yv = to_tensors(val_d, dev)
    print(f"  Train: {Xt.shape[0]}, Val: {Xv.shape[0]}", flush=True)

    seeds = [42, 179, 316]
    results = {'transfer': [], 'scratch': []}

    for i, seed in enumerate(seeds):
        # Transfer: load v4 weights, fine-tune on LXCat
        print(f"\n  --- Transfer run {i} (seed={seed}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        v4_ckpt = os.path.join(V4_DIR, f'model_{i}.pt')
        if not os.path.exists(v4_ckpt):
            print(f"  WARNING: {v4_ckpt} not found, skipping", flush=True)
            continue

        model_t = SurrogateNet().to(dev)
        model_t.load_state_dict(torch.load(v4_ckpt, map_location=dev, weights_only=True))
        t0 = time.time()
        # Fine-tune with lower LR for 1000 epochs (transfer doesn't need as many)
        metrics_t, val_t = train_and_eval(model_t, Xt, Yt, Xv, Yv, train_d, dev,
                                          n_ep=1000, lr=3e-4, label=f"transfer[{i}]")
        dt_t = time.time() - t0
        results['transfer'].append({
            'seed': seed, 'metrics': metrics_t, 'best_val': val_t, 'time_s': dt_t,
            'init': f'model_{i}.pt from surrogate_v4', 'epochs': 1000, 'lr': 3e-4,
        })
        print(f"  Transfer: nF={metrics_t['nF']['rmse']:.5f} nSF6={metrics_t['nSF6']['rmse']:.5f} ({dt_t:.0f}s)", flush=True)

        # Scratch: same architecture, random init, full training
        print(f"\n  --- Scratch run {i} (seed={seed}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_s = SurrogateNet().to(dev)
        with torch.no_grad():
            model_s.head.bias.copy_(torch.tensor([19.8, 20.0]))
        t0 = time.time()
        metrics_s, val_s = train_and_eval(model_s, Xt, Yt, Xv, Yv, train_d, dev,
                                          n_ep=2000, lr=1e-3, label=f"scratch[{i}]")
        dt_s = time.time() - t0
        results['scratch'].append({
            'seed': seed, 'metrics': metrics_s, 'best_val': val_s, 'time_s': dt_s,
            'init': 'random + bias init', 'epochs': 2000, 'lr': 1e-3,
        })
        print(f"  Scratch: nF={metrics_s['nF']['rmse']:.5f} nSF6={metrics_s['nSF6']['rmse']:.5f} ({dt_s:.0f}s)", flush=True)

    # Summary
    t_nF = [r['metrics']['nF']['rmse'] for r in results['transfer']]
    s_nF = [r['metrics']['nF']['rmse'] for r in results['scratch']]
    t_nSF6 = [r['metrics']['nSF6']['rmse'] for r in results['transfer']]
    s_nSF6 = [r['metrics']['nSF6']['rmse'] for r in results['scratch']]

    summary = {
        'transfer': {
            'nF_rmse_mean': float(np.mean(t_nF)) if t_nF else None,
            'nF_rmse_std': float(np.std(t_nF)) if t_nF else None,
            'nSF6_rmse_mean': float(np.mean(t_nSF6)) if t_nSF6 else None,
            'epochs': 1000, 'lr': 3e-4,
            'runs': results['transfer'],
        },
        'scratch': {
            'nF_rmse_mean': float(np.mean(s_nF)) if s_nF else None,
            'nF_rmse_std': float(np.std(s_nF)) if s_nF else None,
            'nSF6_rmse_mean': float(np.mean(s_nSF6)) if s_nSF6 else None,
            'epochs': 2000, 'lr': 1e-3,
            'runs': results['scratch'],
        },
        'transfer_benefit': {
            'nF_improvement': (float(np.mean(s_nF)) - float(np.mean(t_nF))) / float(np.mean(s_nF)) if s_nF and t_nF else None,
            'time_savings': 'Transfer uses 1000ep vs 2000ep (half the compute)',
        },
    }

    with open(os.path.join(OUT, 'transfer_learning.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    md = "# Transfer Learning: surrogate_v4 -> LXCat\n\n"
    md += "| Method | Init | Epochs | LR | nF RMSE | nSF6 RMSE |\n"
    md += "|---|---|---|---|---|---|\n"
    if t_nF:
        md += f"| Transfer | v4 weights | 1000 | 3e-4 | {np.mean(t_nF):.5f}+/-{np.std(t_nF):.5f} | {np.mean(t_nSF6):.5f}+/-{np.std(t_nSF6):.5f} |\n"
    if s_nF:
        md += f"| Scratch | random+bias | 2000 | 1e-3 | {np.mean(s_nF):.5f}+/-{np.std(s_nF):.5f} | {np.mean(s_nSF6):.5f}+/-{np.std(s_nSF6):.5f} |\n"

    with open(os.path.join(OUT, 'transfer_learning.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
