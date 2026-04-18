#!/usr/bin/env python3
"""
Generate publication-quality figures comparing model results with
refined benchmark data from Kokkoris et al. (2009).

Usage: python plot_figures.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# Publication style
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2,
})

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
REF  = os.path.join(DATA, 'refined')
FIG  = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

def load_model(filename):
    path = os.path.join(DATA, filename)
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def col(rows, key):
    return [r[key] for r in rows]

def load_ref(filename):
    path = os.path.join(REF, filename)
    if not os.path.exists(path):
        return None, None
    x, y = [], []
    with open(path) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except:
                    pass
    return (np.array(x), np.array(y)) if x else (None, None)

def load_exp(filename):
    """Load experimental data points."""
    return load_ref(filename)

# =====================================================================
# Plotting functions
# =====================================================================
MODEL_STYLE = dict(color='#2166AC', marker='o', ms=4, lw=2, label='Model')
REF_STYLE   = dict(color='#1A1A1A', lw=2.5, ls='-', label='Kokkoris et al.')

# Distinct styles for experimental data sets: filled vs hollow markers, different colors
EXP_STYLES = [
    dict(color='#D62728', marker='s', ms=9, ls='none', mew=1.5, mfc='#D62728'),   # filled red square
    dict(color='#2CA02C', marker='s', ms=9, ls='none', mew=1.5, mfc='none'),       # hollow green square
    dict(color='#FF7F0E', marker='D', ms=8, ls='none', mew=1.5, mfc='#FF7F0E'),   # filled orange diamond
]

def make_plot(mx, my, ref_file, xlabel, ylabel, title, savename,
              logy=True, exp_files=None, xlim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pf = ax.semilogy if logy else ax.plot

    # Model
    pf(mx, my, **MODEL_STYLE, zorder=3)

    # Reference (Kokkoris model curve)
    rx, ry = load_ref(ref_file)
    if rx is not None:
        pf(rx, ry, **REF_STYLE, zorder=2)

    # Experimental data — each set gets a distinct style
    if exp_files:
        for i, (ef, elabel) in enumerate(exp_files):
            ex, ey = load_exp(ef)
            if ex is not None:
                style = EXP_STYLES[i % len(EXP_STYLES)]
                ax.plot(ex, ey, **style, label=elabel, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, savename))
    plt.close()
    print(f'  {savename}')

if __name__ == '__main__':
    # Load model data
    mp = load_model('model_power_sweep.csv')
    mq = load_model('model_pressure_sweep.csv')

    PW = col(mp, 'Power_W')
    PQ = col(mq, 'pOFF_Pa')

    print("Power sweep figures:")
    XL_P = 'Power (W)'
    XL_Q = r'$p_{\mathrm{OFF}}$ (Pa)'
    DENS = r'Number density (m$^{-3}$)'

    # --- Power sweep ---
    make_plot(PW, col(mp,'Te_eV'), 'kokkoris_Te_vs_power.csv',
              XL_P, r'$T_e$ (eV)', r'Electron temperature vs power',
              'Te_vs_power.png', logy=False, xlim=(0,3600))

    make_plot(PW, col(mp,'ne_m3'), 'kokkoris_ne_vs_power.csv',
              XL_P, DENS, r'Electron density vs power',
              'ne_vs_power.png', xlim=(0,3600))

    make_plot(PW, col(mp,'alpha'), 'kokkoris_alpha_vs_power.csv',
              XL_P, r'$\alpha = n^-/n_e$', r'Electronegativity vs power',
              'alpha_vs_power.png', logy=False, xlim=(0,3600))

    for mk, kf, lb in [
        ('nSF6_m3', 'kokkoris_nSF6_vs_power.csv', r'SF$_6$'),
        ('nF_m3',   'kokkoris_nF_vs_power.csv',   r'F'),
        ('nF2_m3',  'kokkoris_nF2_vs_power.csv',  r'F$_2$'),
        ('nSF4_m3', 'kokkoris_nSF4_vs_power.csv', r'SF$_4$'),
        ('nSF5_m3', 'kokkoris_nSF5_vs_power.csv', r'SF$_5$'),
        ('nSF3_m3', 'kokkoris_nSF3_vs_power.csv', r'SF$_3$'),
        ('nSF5p_m3','kokkoris_nSF5p_vs_power.csv',r'SF$_5^+$'),
        ('nSF4p_m3','kokkoris_nSF4p_vs_power.csv',r'SF$_4^+$'),
        ('nSF3p_m3','kokkoris_nSF3p_vs_power.csv',r'SF$_3^+$'),
        ('nFm_m3',  'kokkoris_nFm_vs_power.csv',  r'F$^-$'),
        ('nSF6m_m3','kokkoris_nSF6m_vs_power.csv',r'SF$_6^-$'),
    ]:
        make_plot(PW, col(mp, mk), kf, XL_P, DENS,
                  f'{lb} vs power', f'{mk}_vs_power.png', xlim=(0,3600))

    # dp vs power with experimental data
    make_plot(PW, col(mp,'dp_Pa'), 'kokkoris_dp_model_vs_power.csv',
              XL_P, r'Pressure rise (Pa)', r'Pressure rise vs power',
              'dp_vs_power.png', logy=False, xlim=(0,3600),
              exp_files=[('kokkoris_dp_exp_vs_power.csv', 'Experiment')])

    # F with experimental data
    make_plot(PW, col(mp,'nF_m3'), 'kokkoris_nF_model_vs_power.csv',
              XL_P, DENS, r'F density vs power (with experiment)',
              'nF_with_exp_vs_power.png', xlim=(0,3600),
              exp_files=[('kokkoris_nF_exp_056_vs_power.csv', r'Exp. $K_F$=0.56'),
                         ('kokkoris_nF_exp_076_vs_power.csv', r'Exp. $K_F$=0.76')])

    # --- Pressure sweep ---
    print("\nPressure sweep figures:")
    make_plot(PQ, col(mq,'Te_eV'), 'kokkoris_Te_vs_poff.csv',
              XL_Q, r'$T_e$ (eV)', r'Electron temperature vs $p_{\mathrm{OFF}}$',
              'Te_vs_poff.png', logy=False)

    make_plot(PQ, col(mq,'ne_m3'), 'kokkoris_ne_vs_poff.csv',
              XL_Q, DENS, r'Electron density vs $p_{\mathrm{OFF}}$',
              'ne_vs_poff.png')

    make_plot(PQ, col(mq,'alpha'), 'kokkoris_alpha_vs_poff.csv',
              XL_Q, r'$\alpha = n^-/n_e$', r'Electronegativity vs $p_{\mathrm{OFF}}$',
              'alpha_vs_poff.png', logy=False)

    for mk, kf, lb in [
        ('nSF6_m3', 'kokkoris_nSF6_vs_poff.csv', r'SF$_6$'),
        ('nF_m3',   'kokkoris_nF_vs_poff.csv',   r'F'),
        ('nF2_m3',  'kokkoris_nF2_vs_poff.csv',  r'F$_2$'),
        ('nSF4_m3', 'kokkoris_nSF4_vs_poff.csv', r'SF$_4$'),
        ('nSF5_m3', 'kokkoris_nSF5_vs_poff.csv', r'SF$_5$'),
        ('nSF3_m3', 'kokkoris_nSF3_vs_poff.csv', r'SF$_3$'),
        ('nSF5p_m3','kokkoris_nSF5p_vs_poff.csv',r'SF$_5^+$'),
        ('nSF4p_m3','kokkoris_nSF4p_vs_poff.csv',r'SF$_4^+$'),
        ('nSF3p_m3','kokkoris_nSF3p_vs_poff.csv',r'SF$_3^+$'),
        ('nFm_m3',  'kokkoris_nFm_vs_poff.csv',  r'F$^-$'),
        ('nSF6m_m3','kokkoris_nSF6m_vs_poff.csv',r'SF$_6^-$'),
    ]:
        make_plot(PQ, col(mq, mk), kf, XL_Q, DENS,
                  f'{lb} vs $p_{{\\mathrm{{OFF}}}}$', f'{mk}_vs_poff.png')

    make_plot(PQ, col(mq,'dp_Pa'), 'kokkoris_dp_model_vs_poff.csv',
              XL_Q, r'Pressure rise (Pa)', r'Pressure rise vs $p_{\mathrm{OFF}}$',
              'dp_vs_poff.png', logy=False,
              exp_files=[('kokkoris_dp_exp_vs_poff.csv', 'Experiment')])

    make_plot(PQ, col(mq,'nF_m3'), 'kokkoris_nF_model_vs_poff.csv',
              XL_Q, DENS, r'F density vs $p_{\mathrm{OFF}}$ (with experiment)',
              'nF_with_exp_vs_poff.png',
              exp_files=[('kokkoris_nF_exp_056_vs_poff.csv', r'Exp. $K_F$=0.56'),
                         ('kokkoris_nF_exp_076_vs_poff.csv', r'Exp. $K_F$=0.76')])

    # =================================================================
    # COMPREHENSIVE MULTI-SPECIES OVERVIEW FIGURES
    # =================================================================
    print("\nOverview figures:")

    NEUTRAL_SPECIES = [
        ('nSF6_m3', r'SF$_6$', '#1b9e77'),
        ('nF_m3',   r'F',      '#d95f02'),
        ('nF2_m3',  r'F$_2$',  '#e7298a'),
        ('nSF4_m3', r'SF$_4$', '#7570b3'),
        ('nSF5_m3', r'SF$_5$', '#66a61e'),
        ('nSF3_m3', r'SF$_3$', '#e6ab02'),
    ]
    CHARGED_SPECIES = [
        ('nSF5p_m3', r'SF$_5^+$', '#1b9e77'),
        ('nSF4p_m3', r'SF$_4^+$', '#d95f02'),
        ('nSF3p_m3', r'SF$_3^+$', '#e7298a'),
        ('nSF6m_m3', r'SF$_6^-$', '#7570b3'),
        ('nFm_m3',   r'F$^-$',    '#66a61e'),
        ('ne_m3',    r'e$^-$',    '#e6ab02'),
    ]

    def make_overview(model_rows, x_key, species_list, xlabel, title, savename, xlim=None):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = col(model_rows, x_key)
        for mk, lb, clr in species_list:
            ax.semilogy(x, col(model_rows, mk), '-', color=clr, lw=2, label=lb)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(r'Number density (m$^{-3}$)', fontsize=14)
        ax.set_title(title, fontsize=14)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, savename), dpi=200)
        plt.close()
        print(f'  {savename}')

    # 1. Neutral species vs power
    make_overview(mp, 'Power_W', NEUTRAL_SPECIES,
                  'Power (W)', 'Neutral species densities vs power',
                  'overview_neutrals_vs_power.png', xlim=(0, 3600))

    # 2. Charged species vs power
    make_overview(mp, 'Power_W', CHARGED_SPECIES,
                  'Power (W)', 'Charged species densities vs power',
                  'overview_charged_vs_power.png', xlim=(0, 3600))

    # 3. Neutral species vs pOFF
    make_overview(mq, 'pOFF_Pa', NEUTRAL_SPECIES,
                  r'$p_{\mathrm{OFF}}$ (Pa)', r'Neutral species densities vs $p_{\mathrm{OFF}}$',
                  'overview_neutrals_vs_poff.png')

    # 4. Charged species vs pOFF
    make_overview(mq, 'pOFF_Pa', CHARGED_SPECIES,
                  r'$p_{\mathrm{OFF}}$ (Pa)', r'Charged species densities vs $p_{\mathrm{OFF}}$',
                  'overview_charged_vs_poff.png')

    n_figs = len([f for f in os.listdir(FIG) if f.endswith('.png')])
    print(f"\nDone. {n_figs} figures in figures/")
