"""
Final surrogate — 3-output model (nF, nSF6, Te) on v4 dataset.
Extends v4 from 2 outputs to 3 by adding electron temperature.
Trains on MPS/CUDA, 5-model ensemble, physics-regularised.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(REPO, 'results', 'pinn_dataset_v4')
RESULTS_DIR = os.path.join(REPO, 'results', 'surrogate_final')
os.makedirs(RESULTS_DIR, exist_ok=True)
R_PROC, Z_TOP = 0.105, 0.234

def select_device():
    if torch.cuda.is_available():
        d, r = torch.device('cuda'), "CUDA GPU"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        d, r = torch.device('mps'), "Apple MPS"
    else:
        d, r = torch.device('cpu'), "WARNING: CPU fallback"
    print(f"  Device: {d} ({r})", flush=True)
    return d

def load_dataset(val_fraction=0.15):
    with open(os.path.join(DATASET_DIR, 'metadata.json')) as f:
        meta = json.load(f)
    n = len(meta)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = max(int(n * val_fraction), 10)
    val_idx = set(perm[:n_val].tolist())

    data = {k: [] for k in ['r','z','P','p','Ar','lnF','lnSF6','lnTe','case']}
    for entry in meta:
        d = np.load(os.path.join(DATASET_DIR, entry['file']))
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
                    data['lnF'].append(np.log10(max(d['nF'][i,j], 1e6)))
                    data['lnSF6'].append(np.log10(max(d['nSF6'][i,j], 1e6)))
                    data['lnTe'].append(float(d['Te'][i,j]))  # Te in eV, not log
                    data['case'].append(entry['idx'])
    arrays = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    arrays['case'] = arrays['case'].astype(np.int32)
    mask = np.array([c not in val_idx for c in arrays['case']])
    split = lambda m: {k: v[m] for k, v in arrays.items()}
    return split(mask), split(~mask), meta, val_idx

def to_tensors(data, device):
    X = np.column_stack([data['r']/R_PROC, data['z']/Z_TOP,
                         data['P']/1200, data['p']/20, data['Ar']])
    Y = np.column_stack([data['lnF'], data['lnSF6'], data['lnTe']])
    return torch.tensor(X, dtype=torch.float32, device=device), \
           torch.tensor(Y, dtype=torch.float32, device=device)

class SurrogateFinal(nn.Module):
    """3-output: (log10 nF, log10 nSF6, Te_eV)."""
    def __init__(self, nh=128, nl=4, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
        self.register_buffer('B', B)
        ni = 2*nf
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl-1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, 3)  # 3 outputs now
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0, 2.7]))

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))

def train_single(model, Xt, Yt, Xv, Yv, ys, dev, ne=2000, lr=1e-3,
                 sw=None, ws=5e-4, bs=4096, mi=0):
    if sw is None:
        sw = torch.tensor([1.0, 1.5, 2.0], device=dev)  # upweight Te
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ne, eta_min=1e-6)
    n = Xt.shape[0]
    best_v, best_s = float('inf'), None
    hist = {'t': [], 'v': []}
    for ep in range(ne):
        model.train()
        pm = torch.randperm(n, device=dev)
        el, nb = 0, 0
        for s in range(0, n, bs):
            idx = pm[s:s+bs]
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)
            Ld = (sw * ((pred - yb)/ys).pow(2)).mean()
            # Smoothness every 4th batch
            Ls = torch.tensor(0., device=dev)
            if ws > 0 and nb % 4 == 0:
                xg = xb.detach().clone().requires_grad_(True)
                p2 = model(xg)
                for c in range(3):
                    g = torch.autograd.grad(p2[:,c].sum(), xg, create_graph=True)[0]
                    Ls = Ls + (g[:,0]**2 + g[:,1]**2).mean()
            loss = Ld + ws*Ls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += Ld.item(); nb += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv)
            vl = (sw * ((vp-Yv)/ys).pow(2)).mean().item()
        hist['t'].append(el/nb); hist['v'].append(vl)
        if vl < best_v:
            best_v = vl; best_s = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % max(ne//10,1) == 0:
            print(f"    M{mi} [{ep:5d}] t={el/nb:.5f} v={vl:.5f} lr={opt.param_groups[0]['lr']:.1e}", flush=True)
    if best_s: model.load_state_dict(best_s)
    hist['best_v'] = best_v
    return hist

def main():
    print("="*60)
    print("  Final Surrogate — 3-Output (nF, nSF6, Te)")
    print("="*60)
    dev = select_device()
    train_d, val_d, meta, vi = load_dataset()
    Xt, Yt = to_tensors(train_d, dev)
    Xv, Yv = to_tensors(val_d, dev)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    print(f"  Train: {Xt.shape[0]} pts, Val: {Xv.shape[0]} pts")
    print(f"  Outputs: 3 (log10 nF, log10 nSF6, Te eV)")

    NE = 5
    models, hists = [], []
    t0 = time.time()
    for i in range(NE):
        print(f"\n  === STARTING MODEL {i}/{NE} ===", flush=True)
        torch.manual_seed(42+i*137); np.random.seed(42+i*137)
        m = SurrogateFinal().to(dev)
        print(f"    Device: {next(m.parameters()).device}", flush=True)
        h = train_single(m, Xt, Yt, Xv, Yv, ys, dev, ne=2000, mi=i)
        models.append(m.cpu()); hists.append(h)
        print(f"  === FINISHED MODEL {i}/{NE} | best val = {h['best_v']:.6f} ===", flush=True)
    tt = time.time()-t0
    print(f"\n  === ALL MODELS COMPLETE === ({tt:.0f}s on {dev})", flush=True)

    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(RESULTS_DIR, f'model_{i}.pt'))

    # Evaluate
    print(f"\n{'='*60}\n  Evaluation\n{'='*60}")
    Xvc = Xv.cpu(); Yv_np = Yv.cpu().numpy()
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad(): preds.append(m(Xvc).numpy())
    preds = np.stack(preds)
    pm = preds.mean(0); ps = preds.std(0)

    metrics = {}
    for c, nm in enumerate(['nF', 'nSF6', 'Te']):
        err = pm[:,c] - Yv_np[:,c]
        rmse = float(np.sqrt((err**2).mean()))
        pct = float(np.median(np.abs(err) * (np.log(10)*100 if c < 2 else 100/max(Yv_np[:,c].mean(),1e-3))))
        corr = float(np.corrcoef(np.abs(err), ps[:,c])[0,1]) if ps[:,c].std() > 1e-10 else 0.0
        metrics[nm] = dict(rmse=rmse, pct=pct, unc_corr=corr)
        unit = 'log10' if c < 2 else 'eV'
        print(f"  {nm}: RMSE={rmse:.4f} ({unit}), median_err={pct:.2f}%, unc_corr={corr:.3f}")

    vals = [float(h['best_v']) for h in hists]
    print(f"\n  Ensemble: mean={np.mean(vals):.6f} ± {np.std(vals):.6f}")

    # Runtime
    from solver import TELSolver
    t0=time.time(); TELSolver(P_rf=700,p_mTorr=10,Nr=30,Nz=50).solve(n_iter=50,verbose=False)
    fd_t=time.time()-t0
    xb=Xvc[:642]
    t0=time.time()
    for _ in range(100):
        with torch.no_grad(): models[0](xb)
    st=(time.time()-t0)/100
    print(f"  FD: {fd_t:.2f}s, Surr: {st*1e3:.2f}ms, Speedup: {fd_t/st:.0f}×")

    # Plots
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for c,(ax,nm) in enumerate(zip(axes,['nF','nSF6','Te'])):
        p_,t_,u_=pm[:,c],Yv_np[:,c],ps[:,c]
        ax.errorbar(t_,p_,yerr=u_,fmt='o',ms=2,alpha=.3,elinewidth=.5,capsize=0)
        lm=[min(t_.min(),p_.min())-.05,max(t_.max(),p_.max())+.05]
        ax.plot(lm,lm,'k--',lw=1)
        unit='log₁₀' if c<2 else 'eV'
        ax.set_xlabel(f'FD {nm} ({unit})'); ax.set_ylabel(f'Surr {nm}')
        ax.set_title(f'{nm} RMSE={metrics[nm]["rmse"]:.4f}'); ax.set_aspect('equal')
    fig.suptitle('Final 3-output surrogate',fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR,'val_pred_vs_true.png'),dpi=150); plt.close()

    # Save summary
    summary = {
        'device': str(dev), 'n_cases': len(meta), 'n_val': len(vi),
        'n_ensemble': NE, 'n_outputs': 3, 'outputs': ['log10_nF','log10_nSF6','Te_eV'],
        'train_time_s': tt, 'ensemble_vals': vals,
        'ens_mean': float(np.mean(vals)), 'ens_std': float(np.std(vals)),
        'metrics': {k: {kk: float(vv) for kk,vv in v.items()} for k,v in metrics.items()},
        'fd_time_s': float(fd_t), 'surrogate_ms': float(st*1e3), 'speedup': float(fd_t/st),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE. Results in {RESULTS_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
