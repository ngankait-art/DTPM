#!/usr/bin/env python3
"""Generate the two figures referenced by §18 (Cluster Replication on Delta):

  - delta_arch_sweep_bar.pdf   bars of E0..E6 nF/nSF6 RMSE with seed-σ error bars
  - mac_vs_delta_ensemble.pdf  Mac (MPS) vs Delta (CUDA) ensemble RMSE comparison

Outputs land in ../../../Plasma Chemistry Module/SF6_surrogate_and_LXCat/figures/.
Run from the 6b root.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIXB_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO_ROOT = os.path.abspath(os.path.join(SIXB_ROOT, '..', '..'))
FIG_DIR = os.path.join(
    REPO_ROOT,
    'Plasma Chemistry Module', 'SF6_surrogate_and_LXCat', 'figures')

ARCH_SWEEP_JSON = os.path.join(
    SIXB_ROOT, 'results', 'ml_arch_sweep_lxcat', 'experiment_table.json')
ENSEMBLE_SUMMARY = os.path.join(
    SIXB_ROOT, 'results', 'ml_production_ensemble_lxcat', 'summary.json')

# Mac (MPS) numbers from main.tex / main.pdf §9 (the originally-reported Phase 1 results).
MAC_NF_RMSE = 0.00154
MAC_NSF6_RMSE = 0.00128


def fig_delta_arch_sweep_bar():
    with open(ARCH_SWEEP_JSON) as f:
        data = json.load(f)
    rows = data.get('experiments', data) if isinstance(data, dict) else data
    names = [r['name'] for r in rows]
    nF = np.array([r['nF_rmse_mean'] for r in rows])
    nF_std = np.array([r['nF_rmse_std'] for r in rows])
    nSF6 = np.array([r['nSF6_rmse_mean'] for r in rows])
    nSF6_std = np.array([r['nSF6_rmse_std'] for r in rows])
    short = [n.replace('E', 'E').split('_')[0] for n in names]

    winner_idx = int(np.argmin(nF))
    colors_nF = ['#1f77b4'] * len(names)
    colors_nF[winner_idx] = '#d62728'
    colors_nSF6 = ['#ff7f0e'] * len(names)
    colors_nSF6[winner_idx] = '#d62728'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    x = np.arange(len(names))

    ax1.bar(x, nF, yerr=nF_std, color=colors_nF, capsize=4, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=0)
    ax1.set_ylabel(r'$n_\mathrm{F}$ RMSE')
    ax1.set_title('Architecture sweep on Delta — nF')
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    ax2.bar(x, nSF6, yerr=nSF6_std, color=colors_nSF6, capsize=4, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=0)
    ax2.set_ylabel(r'$n_{\mathrm{SF}_6}$ RMSE')
    ax2.set_title('Architecture sweep on Delta — nSF6')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'delta_arch_sweep_bar.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def fig_mac_vs_delta_ensemble():
    with open(ENSEMBLE_SUMMARY) as f:
        ens = json.load(f)
    delta_nF = ens['metrics']['nF']['rmse']
    delta_nSF6 = ens['metrics']['nSF6']['rmse']

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    species = ['nF', 'nSF6']
    mac_vals = [MAC_NF_RMSE, MAC_NSF6_RMSE]
    delta_vals = [delta_nF, delta_nSF6]
    x = np.arange(len(species))
    w = 0.36

    ax.bar(x - w/2, mac_vals, w, label='Mac (MPS)', color='#7f7f7f', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, delta_vals, w, label='Delta (CUDA, A100)', color='#d62728', edgecolor='black', linewidth=0.5)

    for xi, mv, dv in zip(x, mac_vals, delta_vals):
        ax.text(xi - w/2, mv + 0.0001, f'{mv:.5f}', ha='center', va='bottom', fontsize=8)
        ax.text(xi + w/2, dv + 0.0001, f'{dv:.5f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([r'$n_\mathrm{F}$ RMSE', r'$n_{\mathrm{SF}_6}$ RMSE'])
    ax.set_ylabel('Test-set RMSE')
    ax.set_title('Production ensemble: Mac–MPS vs Delta–CUDA')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'mac_vs_delta_ensemble.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_delta_arch_sweep_bar()
    fig_mac_vs_delta_ensemble()
