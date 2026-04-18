"""
Te auxiliary head: add Te prediction as a third output to see if the
auxiliary gradient signal improves nF/nSF6 accuracy.

Uses the same LXCat dataset (Te is available in the npz files).
3 seeds, compared against E1 (same arch, no Te head).
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
OUT = os.path.join(REPO, 'results', 'te_auxiliary_head')
os.makedirs(OUT, exist_ok=True)


class ThreeOutputNet(nn.Module):
    """Same as baseline but with 3 outputs: nF, nSF6, Te."""
    def __init__(self, nf=64, nh=128, nl=4, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, 3)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0, 4.5]))

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


def reg_smoothness(model, xb, device):
    xg = xb.detach().clone().requires_grad_(True)
    pred = model(xg)
    L = torch.tensor(0.0, device=device)
    for c in range(3):
        g = torch.autograd.grad(pred[:, c].sum(), xg, create_graph=True)[0]
        L = L + (g[:, 0] ** 2 + g[:, 1] ** 2).mean()
    return L


def reg_bounded_density(pred, y_std):
    lo, hi = 17.0, 21.5
    density_pred = pred[:, :2]
    return (torch.relu(lo - density_pred).pow(2).mean() +
            torch.relu(density_pred - hi).pow(2).mean()) / y_std[:2].mean() ** 2


def load_dataset(val_frac=0.15):
    with open(os.path.join(DS_LXCAT, 'metadata.json')) as f:
        meta = json.load(f)
    meta = [e for e in meta if 'error' not in e]
    n = len(meta)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = max(int(n * val_frac), 10)
    val_idx = set(perm[:n_val].tolist())

    data = {k: [] for k in ['r', 'z', 'P', 'p', 'Ar', 'lnF', 'lnSF6', 'Te', 'case']}
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
                    data['Te'].append(float(d['Te'][i, j]))
                    data['case'].append(entry['idx'])

    arrays = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    arrays['case'] = arrays['case'].astype(np.int32)
    mask = np.array([c not in val_idx for c in arrays['case']])
    split = lambda m: {k: v[m] for k, v in arrays.items()}
    return split(mask), split(~mask)


def to_tensors(data, device):
    X = np.column_stack([data['r'] / R_PROC, data['z'] / Z_TOP,
                         data['P'] / 1200, data['p'] / 20, data['Ar']])
    Y = np.column_stack([data['lnF'], data['lnSF6'], data['Te']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


def main():
    dev = torch.device('cpu')
    print(f"{'=' * 60}", flush=True)
    print(f"  Te Auxiliary Head Experiment", flush=True)
    print(f"{'=' * 60}", flush=True)

    train_d, val_d = load_dataset()
    Xt, Yt = to_tensors(train_d, dev)
    Xv, Yv = to_tensors(val_d, dev)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    # Weight: nF=1.0, nSF6=1.5, Te=0.5 (auxiliary, lower weight)
    sw = torch.tensor([1.0, 1.5, 0.5], device=dev)
    print(f"  Train: {Xt.shape[0]}, Val: {Xv.shape[0]}", flush=True)

    seeds = [42, 179, 316]
    results = []

    for i, seed in enumerate(seeds):
        print(f"\n  --- Run {i} (seed={seed}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ThreeOutputNet().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-6)
        n = Xt.shape[0]
        best_v, best_s = float('inf'), None

        t0 = time.time()
        for ep in range(2000):
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
                # Validate on nF+nSF6 only (same metric as 2-output models)
                vl = (sw[:2] * ((vp[:, :2] - Yv[:, :2]) / ys[:, :2]).pow(2)).mean().item()
            if vl < best_v:
                best_v = vl
                best_s = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ep % 400 == 0:
                print(f"    [{ep:5d}] t={el / nb:.5f} v={vl:.5f}", flush=True)
        dt = time.time() - t0

        if best_s:
            model.load_state_dict(best_s)
        model.eval()
        with torch.no_grad():
            pred = model(Xv).cpu().numpy()
        Yv_np = Yv.cpu().numpy()

        metrics = {}
        for c, nm in enumerate(['nF', 'nSF6', 'Te']):
            err = pred[:, c] - Yv_np[:, c]
            metrics[nm] = {
                'rmse': float(np.sqrt((err ** 2).mean())),
                'mae': float(np.abs(err).mean()),
            }
        results.append({'seed': seed, 'metrics': metrics, 'best_val': best_v, 'time_s': dt})
        print(f"  nF={metrics['nF']['rmse']:.5f} nSF6={metrics['nSF6']['rmse']:.5f} "
              f"Te={metrics['Te']['rmse']:.4f}eV ({dt:.0f}s)", flush=True)

    nF_vals = [r['metrics']['nF']['rmse'] for r in results]
    nSF6_vals = [r['metrics']['nSF6']['rmse'] for r in results]
    Te_vals = [r['metrics']['Te']['rmse'] for r in results]

    summary = {
        'method': '3-output (nF, nSF6, Te) with Te as auxiliary',
        'Te_weight': 0.5,
        'nF_rmse_mean': float(np.mean(nF_vals)),
        'nF_rmse_std': float(np.std(nF_vals)),
        'nSF6_rmse_mean': float(np.mean(nSF6_vals)),
        'Te_rmse_mean': float(np.mean(Te_vals)),
        'reference_E1_nF': 0.0038,
        'runs': results,
    }

    with open(os.path.join(OUT, 'te_auxiliary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    md = "# Te Auxiliary Head\n\n"
    md += f"| Output | RMSE (mean+/-std) | Reference (E1, 2-output) |\n"
    md += f"|---|---|---|\n"
    md += f"| nF | {np.mean(nF_vals):.5f}+/-{np.std(nF_vals):.5f} | 0.0038 |\n"
    md += f"| nSF6 | {np.mean(nSF6_vals):.5f}+/-{np.std(nSF6_vals):.5f} | 0.0034 |\n"
    md += f"| Te | {np.mean(Te_vals):.4f}+/-{np.std(Te_vals):.4f} eV | N/A |\n"

    with open(os.path.join(OUT, 'te_auxiliary.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
