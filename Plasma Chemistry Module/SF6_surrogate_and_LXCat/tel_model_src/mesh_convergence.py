"""
Mesh convergence study: run reference case at 3 resolutions.
700W, 10mTorr, pure SF6 — both legacy and LXCat modes.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

from solver import TELSolver

OUT = os.path.join(REPO, 'results', 'mesh_convergence')
os.makedirs(OUT, exist_ok=True)

RESOLUTIONS = [
    ('coarse', 20, 30),
    ('current', 30, 50),
    ('fine', 50, 80),
]

CASES = [
    {'label': 'legacy', 'rate_mode': 'legacy'},
    {'label': 'lxcat', 'rate_mode': 'lxcat'},
]


def extract_metrics(result, solver):
    inside = solver.inside
    return {
        'nF_avg': float(np.mean(result['nF'][inside])),
        'nSF6_avg': float(np.mean(result['nSF6'][inside])),
        'Te_avg': float(np.mean(result['Te'][inside])),
        'ne_avg': float(result['ne_avg']),
        'F_drop_pct': float(result['F_drop_pct']),
        'nF_max': float(np.max(result['nF'][inside])),
        'nSF6_max': float(np.max(result['nSF6'][inside])),
    }


def main():
    results = {}

    for case in CASES:
        mode = case['rate_mode']
        results[mode] = {}

        for name, Nr, Nz in RESOLUTIONS:
            print(f"  {mode} / {name} ({Nr}x{Nz})...", flush=True, end=' ')
            t0 = time.time()
            solver = TELSolver(Nr=Nr, Nz=Nz, P_rf=700, p_mTorr=10,
                               frac_Ar=0.0, rate_mode=mode)
            res = solver.solve(n_iter=80, w=0.12, verbose=False)
            dt = time.time() - t0
            metrics = extract_metrics(res, solver)
            metrics['time_s'] = dt
            metrics['n_cells'] = Nr * Nz
            metrics['n_active'] = int(solver.n_active)
            results[mode][name] = metrics
            print(f"{dt:.1f}s  nF_avg={metrics['nF_avg']:.3e}  ne={metrics['ne_avg']:.3e}", flush=True)

    # Compute convergence errors (relative to fine)
    convergence = {}
    for mode in ['legacy', 'lxcat']:
        fine = results[mode]['fine']
        convergence[mode] = {}
        for name in ['coarse', 'current']:
            m = results[mode][name]
            convergence[mode][name] = {
                'nF_avg_rel_err': abs(m['nF_avg'] - fine['nF_avg']) / fine['nF_avg'],
                'nSF6_avg_rel_err': abs(m['nSF6_avg'] - fine['nSF6_avg']) / fine['nSF6_avg'],
                'ne_avg_rel_err': abs(m['ne_avg'] - fine['ne_avg']) / fine['ne_avg'],
                'F_drop_rel_err': abs(m['F_drop_pct'] - fine['F_drop_pct']) / max(fine['F_drop_pct'], 0.01),
            }

    output = {
        'reference_case': {'P_rf': 700, 'p_mTorr': 10, 'frac_Ar': 0.0},
        'resolutions': {name: {'Nr': Nr, 'Nz': Nz} for name, Nr, Nz in RESOLUTIONS},
        'results': results,
        'convergence_vs_fine': convergence,
    }

    with open(os.path.join(OUT, 'mesh_convergence.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Markdown summary
    md = "# Mesh Convergence Study\n\n"
    md += "Reference case: 700W, 10mTorr, pure SF6\n\n"

    for mode in ['legacy', 'lxcat']:
        md += f"## {mode.upper()} mode\n\n"
        md += "| Mesh | Cells | Active | nF_avg | nSF6_avg | ne_avg | F_drop | Time |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        for name, Nr, Nz in RESOLUTIONS:
            m = results[mode][name]
            md += f"| {name} ({Nr}x{Nz}) | {m['n_cells']} | {m['n_active']} | "
            md += f"{m['nF_avg']:.3e} | {m['nSF6_avg']:.3e} | {m['ne_avg']:.3e} | "
            md += f"{m['F_drop_pct']:.1f}% | {m['time_s']:.1f}s |\n"

        md += f"\n**Convergence (current vs fine):**\n"
        c = convergence[mode]['current']
        md += f"- nF relative error: {c['nF_avg_rel_err']:.4f} ({c['nF_avg_rel_err']*100:.2f}%)\n"
        md += f"- nSF6 relative error: {c['nSF6_avg_rel_err']:.4f} ({c['nSF6_avg_rel_err']*100:.2f}%)\n"
        md += f"- ne relative error: {c['ne_avg_rel_err']:.4f} ({c['ne_avg_rel_err']*100:.2f}%)\n"
        md += f"- F_drop relative error: {c['F_drop_rel_err']:.4f} ({c['F_drop_rel_err']*100:.2f}%)\n\n"

    with open(os.path.join(OUT, 'mesh_convergence.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
