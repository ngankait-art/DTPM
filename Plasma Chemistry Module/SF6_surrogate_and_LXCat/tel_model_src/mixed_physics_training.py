"""
Mixed-physics training: train on both legacy and LXCat datasets simultaneously
with a physics-label input feature (0=legacy, 1=lxcat).

One model that handles both regimes. Tests whether doubling the data
improves generalization.
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
DS_V4 = os.path.join(REPO, 'results', 'pinn_dataset_v4')
DS_LXCAT = os.path.join(REPO, 'results', 'pinn_dataset_lxcat_v3')
OUT = os.path.join(REPO, 'results', 'mixed_physics_training')
os.makedirs(OUT, exist_ok=True)


class MixedSurrogateNet(nn.Module):
    """6 inputs: r, z, P, p, Ar, physics_label."""
    def __init__(self, nf=64, nh=128, nl=4, fs=3.0, drop=0.05):
        super().__init__()
        n_in = 6  # extra input for physics label
        B = torch.randn(n_in, nf) * fs
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
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))

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
    return (torch.relu(lo - pred).pow(2).mean() + torch.relu(pred - hi).pow(2).mean()) / y_std.mean() ** 2


def load_mixed_dataset(val_frac=0.15):
    """Load both datasets with physics label."""
    all_data = {k: [] for k in ['r', 'z', 'P', 'p', 'Ar', 'label', 'lnF', 'lnSF6', 'case', 'source']}

    for ds_dir, physics_label, source in [(DS_V4, 0.0, 'legacy'), (DS_LXCAT, 1.0, 'lxcat')]:
        with open(os.path.join(ds_dir, 'metadata.json')) as f:
            meta = json.load(f)
        meta = [e for e in meta if 'error' not in e]
        for entry in meta:
            fpath = os.path.join(ds_dir, entry['file'])
            if not os.path.exists(fpath):
                continue
            d = np.load(fpath)
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
                        all_data['label'].append(physics_label)
                        all_data['lnF'].append(np.log10(max(d['nF'][i, j], 1e6)))
                        all_data['lnSF6'].append(np.log10(max(d['nSF6'][i, j], 1e6)))
                        all_data['case'].append(entry['idx'] + (1000 if source == 'lxcat' else 0))
                        all_data['source'].append(source)

    arrays = {k: np.array(v, dtype=np.float32) for k, v in all_data.items() if k != 'source'}
    sources = np.array(all_data['source'])
    arrays['case'] = arrays['case'].astype(np.int32)

    # Split: use same seed, hold out 15% of EACH dataset's cases
    np.random.seed(42)
    # Legacy cases
    legacy_cases = sorted(set(arrays['case'][sources == 'legacy'].tolist()))
    lxcat_cases = sorted(set(arrays['case'][sources == 'lxcat'].tolist()))
    n_val_leg = max(int(len(legacy_cases) * val_frac), 5)
    n_val_lx = max(int(len(lxcat_cases) * val_frac), 5)
    val_leg = set(np.random.choice(legacy_cases, n_val_leg, replace=False).tolist())
    val_lx = set(np.random.choice(lxcat_cases, n_val_lx, replace=False).tolist())
    val_cases = val_leg | val_lx

    mask = np.array([c not in val_cases for c in arrays['case']])
    split = lambda m: {k: v[m] for k, v in arrays.items()}

    # Also split val by source for separate evaluation
    val_data = split(~mask)
    val_sources = sources[~mask]

    return split(mask), val_data, val_sources


def to_tensors(data, device):
    X = np.column_stack([data['r'] / R_PROC, data['z'] / Z_TOP,
                         data['P'] / 1200, data['p'] / 20, data['Ar'],
                         data['label']])
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


def main():
    dev = torch.device('cpu')
    print(f"{'=' * 60}", flush=True)
    print(f"  Mixed-Physics Training", flush=True)
    print(f"{'=' * 60}", flush=True)

    train_d, val_d, val_sources = load_mixed_dataset()
    Xt, Yt = to_tensors(train_d, dev)
    Xv, Yv = to_tensors(val_d, dev)
    print(f"  Train: {Xt.shape[0]}, Val: {Xv.shape[0]}", flush=True)
    print(f"  Val legacy: {(val_sources == 'legacy').sum()}, Val lxcat: {(val_sources == 'lxcat').sum()}", flush=True)

    seeds = [42, 179, 316]
    results = []

    for i, seed in enumerate(seeds):
        print(f"\n  --- Run {i} (seed={seed}) ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MixedSurrogateNet().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-6)
        sw = torch.tensor([1.0, 1.5], device=dev)
        ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
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
                vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
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
            pred_all = model(Xv).cpu().numpy()
        Yv_np = Yv.cpu().numpy()

        # Evaluate separately by source
        metrics = {}
        for src in ['legacy', 'lxcat']:
            src_mask = val_sources == src
            pred_src = pred_all[src_mask]
            true_src = Yv_np[src_mask]
            m = {}
            for c, nm in enumerate(['nF', 'nSF6']):
                err = pred_src[:, c] - true_src[:, c]
                m[nm] = {'rmse': float(np.sqrt((err ** 2).mean())),
                         'mae': float(np.abs(err).mean())}
            metrics[src] = m

        results.append({
            'seed': seed, 'time_s': dt, 'best_val': best_v,
            'metrics_legacy': metrics['legacy'],
            'metrics_lxcat': metrics['lxcat'],
        })
        print(f"  Legacy: nF={metrics['legacy']['nF']['rmse']:.5f}  "
              f"LXCat: nF={metrics['lxcat']['nF']['rmse']:.5f}  ({dt:.0f}s)", flush=True)

    # Summary
    leg_nF = [r['metrics_legacy']['nF']['rmse'] for r in results]
    lx_nF = [r['metrics_lxcat']['nF']['rmse'] for r in results]

    summary = {
        'method': 'Mixed-physics training (legacy + LXCat, physics label input)',
        'n_runs': len(results),
        'legacy_nF_mean': float(np.mean(leg_nF)),
        'legacy_nF_std': float(np.std(leg_nF)),
        'lxcat_nF_mean': float(np.mean(lx_nF)),
        'lxcat_nF_std': float(np.std(lx_nF)),
        'reference': {
            'surrogate_v4_legacy_only': 0.0029,
            'E1_lxcat_only': 0.0038,
        },
        'runs': results,
    }

    with open(os.path.join(OUT, 'mixed_physics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    md = "# Mixed-Physics Training\n\n"
    md += "One model trained on both legacy and LXCat data with physics-label input.\n\n"
    md += "| Evaluation | nF RMSE (mean+/-std) | Reference (single-physics) |\n"
    md += "|---|---|---|\n"
    md += f"| Legacy val | {np.mean(leg_nF):.5f}+/-{np.std(leg_nF):.5f} | surrogate_v4: 0.0029 |\n"
    md += f"| LXCat val | {np.mean(lx_nF):.5f}+/-{np.std(lx_nF):.5f} | E1 v4-recipe: 0.0038 |\n"

    with open(os.path.join(OUT, 'mixed_physics.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
