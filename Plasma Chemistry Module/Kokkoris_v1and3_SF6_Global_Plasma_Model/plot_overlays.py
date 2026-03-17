#!/usr/bin/env python3
"""
SF6 Global Plasma Model — Generate overlay plots (model vs Kokkoris).

Usage:
    python plot_overlays.py

Reads CSV results from output/ and Kokkoris data from data/kokkoris_extracted/.
Saves figures to output/figures/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os, sys

KDIR = os.path.join(os.path.dirname(__file__), 'data', 'kokkoris_extracted')
ODIR = os.path.join(os.path.dirname(__file__), 'output')
FDIR = os.path.join(ODIR, 'figures')
os.makedirs(FDIR, exist_ok=True)

def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def load_kok(fname):
    path = os.path.join(KDIR, fname)
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

def col(rows, k):
    return [r[k] for r in rows]

def make(v1x, v1y, v3x, v3y, kf, xl, yl, title, fname,
         logy=True, exp_f=None, extra_k=None, xlim=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    pf = ax.semilogy if logy else ax.plot

    if v1x is not None:
        pf(v1x, v1y, 's--', color='tab:gray', lw=1.5, ms=3, alpha=0.5,
           label='V1 (L&L, $\\eta$=0.50)', zorder=1)
    if v3x is not None:
        pf(v3x, v3y, 'o-', color='tab:blue', lw=2, ms=4,
           label='V3 (Kim, $\\eta$=0.70)', zorder=4)

    kx, ky = load_kok(kf)
    if kx is not None:
        pf(kx, ky, '-', color='black', lw=2.5, label='Kokkoris et al.', zorder=3)

    if extra_k:
        for ef, el, ec in extra_k:
            ex, ey = load_kok(ef)
            if ex is not None:
                pf(ex, ey, '-', color=ec, lw=1.5, label=el, zorder=2)

    if exp_f:
        for ef, el, em, ec, emfc in exp_f:
            ex, ey = load_kok(ef)
            if ex is not None:
                ax.plot(ex, ey, marker=em, ls='none', color=ec, ms=10, mew=1.5,
                        mfc=emfc, label=el, zorder=5)

    ax.set_xlabel(xl, fontsize=13)
    ax.set_ylabel(yl, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    plt.savefig(os.path.join(FDIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  {fname}')

if __name__ == '__main__':
    # Load model results
    v1p_path = os.path.join(ODIR, 'v1_power_sweep.csv')
    v1q_path = os.path.join(ODIR, 'v1_pressure_sweep.csv')
    v3p_path = os.path.join(ODIR, 'v3_power_sweep.csv')
    v3q_path = os.path.join(ODIR, 'v3_pressure_sweep.csv')

    v1p = load_csv(v1p_path) if os.path.exists(v1p_path) else None
    v1q = load_csv(v1q_path) if os.path.exists(v1q_path) else None
    v3p = load_csv(v3p_path) if os.path.exists(v3p_path) else None
    v3q = load_csv(v3q_path) if os.path.exists(v3q_path) else None

    def v1P(k): return col(v1p, k) if v1p else None
    def v1Q(k): return col(v1q, k) if v1q else None
    def v3P(k): return col(v3p, k) if v3p else None
    def v3Q(k): return col(v3q, k) if v3q else None

    # === POWER SWEEP ===
    print("Power sweep overlays:")
    species = [
        ('nSF6_m3', 'kokkoris_nSF6_vs_power.csv', 'SF$_6$'),
        ('nF_m3', 'kokkoris_nF_vs_power.csv', 'F'),
        ('nF2_m3', 'kokkoris_nF2_vs_power.csv', 'F$_2$'),
        ('nSF4_m3', 'kokkoris_nSF4_vs_power.csv', 'SF$_4$'),
        ('nSF5_m3', 'kokkoris_nSF5_vs_power.csv', 'SF$_5$'),
        ('nSF3_m3', 'kokkoris_nSF3_vs_power.csv', 'SF$_3$'),
        ('nSF5p_m3', 'kokkoris_nSF5p_vs_power.csv', 'SF$_5^+$'),
        ('nFm_m3', 'kokkoris_nFm_vs_power.csv', 'F$^-$'),
        ('ne_m3', 'kokkoris_ne_vs_power.csv', 'e$^-$'),
        ('nSF6m_m3', 'kokkoris_nSF6m_vs_power.csv', 'SF$_6^-$'),
    ]
    for mk, kf, lb in species:
        make(v1P('Power_W'), v1P(mk), v3P('Power_W'), v3P(mk), kf,
             'Power (W)', 'Density (m$^{-3}$)', f'{lb} vs Power',
             f'{mk}_vs_power.png', xlim=(0, 3600))

    make(v1P('Power_W'), v1P('Te_eV'), v3P('Power_W'), v3P('Te_eV'),
         'kokkoris_Te_vs_power.csv', 'Power (W)', '$T_e$ (eV)',
         '$T_e$ vs Power', 'Te_vs_power.png', logy=False, xlim=(0, 3600))
    make(v1P('Power_W'), v1P('alpha'), v3P('Power_W'), v3P('alpha'),
         'kokkoris_alpha_vs_power.csv', 'Power (W)', '$\\alpha$',
         '$\\alpha$ vs Power', 'alpha_vs_power.png', logy=False, xlim=(0, 3600))
    make(v1P('Power_W'), v1P('dp_Pa'), v3P('Power_W'), v3P('dp_Pa'),
         'kokkoris_dp_model_vs_power.csv', 'Power (W)', '$\\Delta p$ (Pa)',
         '$\\Delta p$ vs Power', 'dp_allconfig_vs_power.png', logy=False, xlim=(0, 3600),
         exp_f=[('kokkoris_dp_exp_vs_power.csv', 'Exp.', 's', 'black', 'black')],
         extra_k=[('kokkoris_dp_nodeposition_vs_power.csv', 'Kok: no dep.', 'tab:cyan'),
                  ('kokkoris_dp_nosurface_vs_power.csv', 'Kok: no surf.', 'tab:red')])

    # === PRESSURE SWEEP ===
    print("Pressure sweep overlays:")
    species_p = [
        ('nSF6_m3', 'kokkoris_nSF6_vs_poff.csv', 'SF$_6$'),
        ('nF_m3', 'kokkoris_nF_vs_poff.csv', 'F'),
        ('nF2_m3', 'kokkoris_nF2_vs_poff.csv', 'F$_2$'),
        ('nSF4_m3', 'kokkoris_nSF4_vs_poff.csv', 'SF$_4$'),
        ('nSF5_m3', 'kokkoris_nSF5_vs_poff.csv', 'SF$_5$'),
        ('nSF3_m3', 'kokkoris_nSF3_vs_poff.csv', 'SF$_3$'),
        ('nSF5p_m3', 'kokkoris_nSF5p_vs_poff.csv', 'SF$_5^+$'),
        ('nFm_m3', 'kokkoris_nFm_vs_poff.csv', 'F$^-$'),
        ('ne_m3', 'kokkoris_ne_vs_poff.csv', 'e$^-$'),
        ('nSF6m_m3', 'kokkoris_nSF6m_vs_poff.csv', 'SF$_6^-$'),
    ]
    for mk, kf, lb in species_p:
        make(v1Q('pOFF_Pa'), v1Q(mk), v3Q('pOFF_Pa'), v3Q(mk), kf,
             '$p_{OFF}$ (Pa)', 'Density (m$^{-3}$)', f'{lb} vs $p_{{OFF}}$',
             f'{mk}_vs_poff.png')

    make(v1Q('pOFF_Pa'), v1Q('Te_eV'), v3Q('pOFF_Pa'), v3Q('Te_eV'),
         'kokkoris_Te_vs_poff.csv', '$p_{OFF}$ (Pa)', '$T_e$ (eV)',
         '$T_e$ vs $p_{OFF}$', 'Te_vs_poff.png', logy=False)
    make(v1Q('pOFF_Pa'), v1Q('alpha'), v3Q('pOFF_Pa'), v3Q('alpha'),
         'kokkoris_alpha_vs_poff.csv', '$p_{OFF}$ (Pa)', '$\\alpha$',
         '$\\alpha$ vs $p_{OFF}$', 'alpha_vs_poff.png', logy=False)
    make(v1Q('pOFF_Pa'), v1Q('dp_Pa'), v3Q('pOFF_Pa'), v3Q('dp_Pa'),
         'kokkoris_dp_model_vs_poff.csv', '$p_{OFF}$ (Pa)', '$\\Delta p$ (Pa)',
         '$\\Delta p$ vs $p_{OFF}$', 'dp_allconfig_vs_poff.png', logy=False,
         exp_f=[('kokkoris_dp_exp_vs_poff.csv', 'Exp.', 's', 'black', 'black')],
         extra_k=[('kokkoris_dp_nodeposition_vs_poff.csv', 'Kok: no dep.', 'tab:cyan'),
                  ('kokkoris_dp_nosurface_vs_poff.csv', 'Kok: no surf.', 'tab:red')])
    make(v1Q('pOFF_Pa'), v1Q('nF_m3'), v3Q('pOFF_Pa'), v3Q('nF_m3'),
         'kokkoris_nF_model_vs_poff.csv', '$p_{OFF}$ (Pa)', 'F density (m$^{-3}$)',
         'F vs $p_{OFF}$', 'nF_allconfig_vs_poff.png', logy=True,
         exp_f=[('kokkoris_nF_exp_056_vs_poff.csv', 'Exp $K_F$=0.56', 's', 'black', 'black'),
                ('kokkoris_nF_exp_076_vs_poff.csv', 'Exp $K_F$=0.76', 'o', 'gray', 'none')],
         extra_k=[('kokkoris_nF_nodeposition_vs_poff.csv', 'Kok: no dep.', 'tab:cyan'),
                  ('kokkoris_nF_nosurface_vs_poff.csv', 'Kok: no surf.', 'tab:red')])

    nfigs = len([f for f in os.listdir(FDIR) if f.endswith('.png')])
    print(f"\nDone. {nfigs} figures in output/figures/")
