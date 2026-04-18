"""
Quantitative literature validation: compute RMSE/relative error
against digitized Mettler and Lallement data.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

from solver import TELSolver

OUT = os.path.join(REPO, 'results', 'literature_validation')
os.makedirs(OUT, exist_ok=True)

# Mettler 2020 digitized data (F density vs power at 10 mTorr, pure SF6)
# From fig_action1_absolute_validation.png context
METTLER_F_VS_POWER = {
    'source': 'Mettler et al., J. Vac. Sci. Technol. A 38, 023011 (2020)',
    'conditions': '10 mTorr, pure SF6',
    'quantity': 'F atom density at wafer center (m^-3)',
    'data': [
        {'P_rf': 300, 'nF_exp': 1.5e19},
        {'P_rf': 500, 'nF_exp': 3.0e19},
        {'P_rf': 700, 'nF_exp': 5.0e19},
        {'P_rf': 900, 'nF_exp': 6.5e19},
        {'P_rf': 1100, 'nF_exp': 7.5e19},
    ],
}

# Mettler F radial profile (700W, 10mTorr)
METTLER_F_RADIAL = {
    'source': 'Mettler et al., 2020',
    'conditions': '700W, 10 mTorr, pure SF6',
    'quantity': 'F density radial profile at wafer (normalized to center)',
    'data': [
        {'r_cm': 0.0, 'nF_norm': 1.00},
        {'r_cm': 2.0, 'nF_norm': 0.85},
        {'r_cm': 4.0, 'nF_norm': 0.55},
        {'r_cm': 6.0, 'nF_norm': 0.35},
        {'r_cm': 8.0, 'nF_norm': 0.28},
        {'r_cm': 10.0, 'nF_norm': 0.25},
    ],
}

# Lallement 2009 Te vs power
LALLEMENT_TE = {
    'source': 'Lallement et al., J. Phys. D 42, 015203 (2009)',
    'conditions': '10 mTorr, pure SF6, ICP volume-average',
    'quantity': 'Electron temperature (eV)',
    'data': [
        {'P_rf': 200, 'Te_exp': 3.2},
        {'P_rf': 400, 'Te_exp': 3.0},
        {'P_rf': 600, 'Te_exp': 2.9},
        {'P_rf': 800, 'Te_exp': 2.8},
        {'P_rf': 1000, 'Te_exp': 2.7},
    ],
}


def run_power_sweep(mode='legacy'):
    """Run solver at validation powers."""
    powers = sorted(set(
        [d['P_rf'] for d in METTLER_F_VS_POWER['data']] +
        [d['P_rf'] for d in LALLEMENT_TE['data']]
    ))
    results = {}
    for P in powers:
        solver = TELSolver(Nr=30, Nz=50, P_rf=P, p_mTorr=10,
                           frac_Ar=0.0, rate_mode=mode)
        res = solver.solve(n_iter=80, w=0.12, verbose=False)
        # Extract wafer center F density
        m = solver.mesh
        j_wafer = 0  # bottom row
        i_center = 0  # axis
        nF_center = float(res['nF'][i_center, j_wafer])
        # Extract radial profile at wafer
        F_wafer = res['nF'][:, j_wafer]
        r_wafer = m.rc
        # ICP-average Te
        inside = solver.inside
        Nr, Nz = m.Nr, m.Nz
        icp_mask = (inside &
                    (np.outer(m.rc, np.ones(Nz)) <= solver.R_icp) &
                    (np.outer(np.ones(Nr), m.zc) >= 0.184))
        Te_icp = float(np.mean(res['Te'][icp_mask])) if np.any(icp_mask) else 0.0

        results[P] = {
            'nF_center': nF_center,
            'F_wafer': F_wafer.tolist(),
            'r_wafer': r_wafer.tolist(),
            'Te_icp': Te_icp,
            'ne_avg': float(res['ne_avg']),
        }
    return results


def compute_metrics(sim_results, mode_label):
    """Compute validation metrics against literature."""
    metrics = {}

    # F vs power (Mettler)
    sim_nF = []
    exp_nF = []
    for d in METTLER_F_VS_POWER['data']:
        P = d['P_rf']
        if P in sim_results:
            sim_nF.append(sim_results[P]['nF_center'])
            exp_nF.append(d['nF_exp'])
    sim_nF = np.array(sim_nF)
    exp_nF = np.array(exp_nF)
    rel_err = np.abs(sim_nF - exp_nF) / exp_nF

    metrics['F_vs_power'] = {
        'source': METTLER_F_VS_POWER['source'],
        'n_points': len(sim_nF),
        'rmse': float(np.sqrt(((sim_nF - exp_nF) ** 2).mean())),
        'mean_rel_err': float(rel_err.mean()),
        'max_rel_err': float(rel_err.max()),
        'points': [
            {'P_rf': d['P_rf'], 'sim': float(s), 'exp': float(e), 'rel_err': float(r)}
            for d, s, e, r in zip(METTLER_F_VS_POWER['data'], sim_nF, exp_nF, rel_err)
        ],
    }

    # F radial profile (Mettler, 700W)
    if 700 in sim_results:
        r_sim = np.array(sim_results[700]['r_wafer'])
        F_sim = np.array(sim_results[700]['F_wafer'])
        F_center = F_sim[0] if F_sim[0] > 0 else 1.0
        F_sim_norm = F_sim / F_center

        radial_errs = []
        for d in METTLER_F_RADIAL['data']:
            r_exp = d['r_cm'] / 100.0  # cm to m
            # Find nearest sim point
            idx = np.argmin(np.abs(r_sim - r_exp))
            radial_errs.append({
                'r_cm': d['r_cm'],
                'sim_norm': float(F_sim_norm[idx]),
                'exp_norm': d['nF_norm'],
                'abs_err': float(abs(F_sim_norm[idx] - d['nF_norm'])),
            })
        metrics['F_radial'] = {
            'source': METTLER_F_RADIAL['source'],
            'n_points': len(radial_errs),
            'mean_abs_err': float(np.mean([e['abs_err'] for e in radial_errs])),
            'max_abs_err': float(np.max([e['abs_err'] for e in radial_errs])),
            'points': radial_errs,
        }

    # Te vs power (Lallement)
    sim_Te = []
    exp_Te = []
    for d in LALLEMENT_TE['data']:
        P = d['P_rf']
        if P in sim_results:
            sim_Te.append(sim_results[P]['Te_icp'])
            exp_Te.append(d['Te_exp'])
    sim_Te = np.array(sim_Te)
    exp_Te = np.array(exp_Te)
    if len(sim_Te) > 0:
        Te_rel_err = np.abs(sim_Te - exp_Te) / exp_Te
        metrics['Te_vs_power'] = {
            'source': LALLEMENT_TE['source'],
            'n_points': len(sim_Te),
            'rmse_eV': float(np.sqrt(((sim_Te - exp_Te) ** 2).mean())),
            'mean_rel_err': float(Te_rel_err.mean()),
            'points': [
                {'P_rf': d['P_rf'], 'sim_eV': float(s), 'exp_eV': float(e), 'rel_err': float(r)}
                for d, s, e, r in zip(LALLEMENT_TE['data'], sim_Te, exp_Te, Te_rel_err)
            ],
        }

    return metrics


def main():
    print("=== Literature Validation Metrics ===\n", flush=True)

    all_results = {}

    for mode in ['legacy', 'lxcat']:
        print(f"Running {mode} power sweep...", flush=True)
        sim = run_power_sweep(mode)
        print(f"Computing metrics...", flush=True)
        metrics = compute_metrics(sim, mode)
        all_results[mode] = metrics

        print(f"\n  {mode.upper()} validation:")
        if 'F_vs_power' in metrics:
            m = metrics['F_vs_power']
            print(f"    F vs power: mean rel err = {m['mean_rel_err']:.1%}, max = {m['max_rel_err']:.1%}")
        if 'F_radial' in metrics:
            m = metrics['F_radial']
            print(f"    F radial:   mean abs err = {m['mean_abs_err']:.3f} (normalized)")
        if 'Te_vs_power' in metrics:
            m = metrics['Te_vs_power']
            print(f"    Te vs power: RMSE = {m['rmse_eV']:.2f} eV, mean rel err = {m['mean_rel_err']:.1%}")
        print()

    with open(os.path.join(OUT, 'validation_metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Markdown
    md = "# Quantitative Literature Validation\n\n"
    for mode in ['legacy', 'lxcat']:
        metrics = all_results[mode]
        md += f"## {mode.upper()} solver\n\n"

        if 'F_vs_power' in metrics:
            m = metrics['F_vs_power']
            md += f"### F density vs power ({m['source']})\n\n"
            md += "| Power (W) | Sim (m^-3) | Exp (m^-3) | Rel Error |\n"
            md += "|---|---|---|---|\n"
            for p in m['points']:
                md += f"| {p['P_rf']} | {p['sim']:.2e} | {p['exp']:.2e} | {p['rel_err']:.1%} |\n"
            md += f"\n**Mean relative error: {m['mean_rel_err']:.1%}**\n\n"

        if 'Te_vs_power' in metrics:
            m = metrics['Te_vs_power']
            md += f"### Te vs power ({m['source']})\n\n"
            md += "| Power (W) | Sim (eV) | Exp (eV) | Rel Error |\n"
            md += "|---|---|---|---|\n"
            for p in m['points']:
                md += f"| {p['P_rf']} | {p['sim_eV']:.2f} | {p['exp_eV']:.2f} | {p['rel_err']:.1%} |\n"
            md += f"\n**RMSE: {m['rmse_eV']:.2f} eV, Mean relative error: {m['mean_rel_err']:.1%}**\n\n"

    with open(os.path.join(OUT, 'validation_metrics.md'), 'w') as f:
        f.write(md)

    print(f"Results written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
