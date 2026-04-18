#!/usr/bin/env python3
"""
main.py — Wall Surface Chemistry Extension: Full Analysis Pipeline
===================================================================
Runs all simulation branches, benchmarking, and figure generation.

Branch A: Lallement geometry (gas-phase vs wall chemistry)
Branch B: TEL-informed geometry (gas-phase vs wall chemistry)

Usage: cd code/ && python main.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine import solve_model, sweep_with_continuation, Reactor

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'lines.linewidth': 2.0,
    'lines.markersize': 7, 'axes.grid': True, 'grid.alpha': 0.3,
})

FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG, exist_ok=True)

# ── Reactor geometries ──
LAL = Reactor(R=0.180, L=0.175)   # Lallement ICP
TEL = Reactor(R=0.105, L=0.0535)  # TEL processing region

# ── Wall chemistry parameters (Kokkoris × 0.007) ──
WC = {'s_F': 0.00105, 's_SFx': 0.00056, 'p_fluor': 0.025,
      'p_wallrec': 0.007, 'p_FF': 0.0035, 'p_F_SF3': 0.5, 'p_F_SF4': 0.2}

# ── Digitized paper data ──
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def load(fname):
    try:
        return np.genfromtxt(os.path.join(DATA, fname), delimiter=',')
    except:
        return None

def save_fig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {name}")


def main():
    print("=" * 70)
    print("WALL SURFACE CHEMISTRY EXTENSION — FULL ANALYSIS")
    print("=" * 70)

    # ══════════════════════════════════════════════════
    # GEOMETRY SUMMARY
    # ══════════════════════════════════════════════════
    print("\n▶ Reactor Geometries")
    for name, rx in [("Lallement", LAL), ("TEL", TEL)]:
        print(f"  {name:>10}: R={rx.R*100:.1f}cm L={rx.L*100:.1f}cm "
              f"V={rx.V*1e6:.0f}cm³ A={rx.A*1e4:.0f}cm² A/V={rx.A/rx.V:.1f}m⁻¹")
    
    kB = 1.381e-23
    for label, p_mT, Q, V in [
        ("Lallement 10mT", 10, 40, LAL.V),
        ("TEL 10mT", 10, 100, TEL.V),
        ("TEL 50mT", 50, 40, TEL.V),
    ]:
        p_Pa = p_mT * 0.13332
        Q_tp = Q * 1e-6/60 * 1.01325e5 * (300/273.15)
        tau = p_Pa * V / Q_tp
        ng = p_Pa / (kB * 300)
        print(f"  {label:>15}: tau={tau*1000:.1f}ms  ng={ng:.2e}m⁻³")

    # ══════════════════════════════════════════════════
    # BRANCH A: Lallement geometry sweeps
    # ══════════════════════════════════════════════════
    print("\n▶ Branch A: Lallement geometry")
    fracs = np.linspace(0, 0.8, 41)
    powers = np.linspace(200, 2000, 37)
    
    # A1: Gas-phase only
    print("  A1: Gas-phase, alpha vs Ar...")
    A1_alpha = sweep_with_continuation('frac_Ar', fracs,
        dict(P_rf=1500, p_mTorr=10, Q_sccm=40, eta=0.12, wall_chem=False, reactor=LAL),
        verbose=False)
    
    # A2: Wall chemistry
    print("  A2: Wall chemistry, alpha vs Ar...")
    A2_alpha = sweep_with_continuation('frac_Ar', fracs,
        dict(P_rf=1500, p_mTorr=10, Q_sccm=40, eta=0.16, wall_chem=WC, reactor=LAL),
        verbose=False)
    
    # A1/A2 power sweeps (pure SF6)
    print("  Power sweeps (pure SF6)...")
    A1_power = sweep_with_continuation('P_rf', powers,
        dict(p_mTorr=10, Q_sccm=40, frac_Ar=0, eta=0.12, wall_chem=False, reactor=LAL),
        verbose=False)
    A2_power = sweep_with_continuation('P_rf', powers,
        dict(p_mTorr=10, Q_sccm=40, frac_Ar=0, eta=0.16, wall_chem=WC, reactor=LAL),
        verbose=False)
    
    # A2 at 3 pressures
    print("  Wall chem at 3 pressures...")
    A2_alpha_p = {}
    for p in [5, 10, 20]:
        A2_alpha_p[p] = sweep_with_continuation('frac_Ar', fracs,
            dict(P_rf=1500, p_mTorr=p, Q_sccm=40, eta=0.16, wall_chem=WC, reactor=LAL),
            verbose=False)

    # ══════════════════════════════════════════════════
    # BRANCH B: TEL geometry sweeps
    # ══════════════════════════════════════════════════
    print("\n▶ Branch B: TEL geometry")
    
    # B1: Gas-phase with TEL geometry
    print("  B1: Gas-phase, alpha vs Ar...")
    B1_alpha = sweep_with_continuation('frac_Ar', fracs,
        dict(P_rf=1500, p_mTorr=10, Q_sccm=100, eta=0.12, wall_chem=False, reactor=TEL),
        verbose=False)
    
    # B2: Wall chemistry with TEL geometry
    print("  B2: Wall chemistry, alpha vs Ar...")
    B2_alpha = sweep_with_continuation('frac_Ar', fracs,
        dict(P_rf=1500, p_mTorr=10, Q_sccm=100, eta=0.16, wall_chem=WC, reactor=TEL),
        verbose=False)

    # TEL benchmarking sweeps
    print("  TEL [F] vs power (50mTorr, 50% Ar)...")
    B2_tel45 = sweep_with_continuation('P_rf', np.linspace(200, 3000, 29),
        dict(p_mTorr=50, frac_Ar=0.5, Q_sccm=40, eta=0.16, wall_chem=WC, reactor=TEL),
        verbose=False)
    
    print("  TEL [F] vs SF6 flow...")
    B2_tel49_20, B2_tel49_40 = [], []
    for sf6 in np.arange(10, 95, 10):
        fAr = 1.0 - sf6/100
        r20 = solve_model(P_rf=600, p_mTorr=20, frac_Ar=fAr, Q_sccm=100, eta=0.16,
                         wall_chem=WC, reactor=TEL)
        r20['sf6_sccm'] = sf6
        B2_tel49_20.append(r20)
        r40 = solve_model(P_rf=700, p_mTorr=40, frac_Ar=fAr, Q_sccm=100, eta=0.16,
                         wall_chem=WC, reactor=TEL)
        r40['sf6_sccm'] = sf6
        B2_tel49_40.append(r40)
    
    print("  TEL center [F] at 10mTorr...")
    r_30 = solve_model(P_rf=1000, p_mTorr=10, frac_Ar=0.7, Q_sccm=100, eta=0.16,
                      wall_chem=WC, reactor=TEL)
    r_90 = solve_model(P_rf=1000, p_mTorr=10, frac_Ar=0.1, Q_sccm=100, eta=0.16,
                      wall_chem=WC, reactor=TEL)
    print(f"    30% SF6: [F]={r_30['n_F']*1e-6:.2e} cm⁻³")
    print(f"    90% SF6: [F]={r_90['n_F']*1e-6:.2e} cm⁻³")

    # ══════════════════════════════════════════════════
    # SENSITIVITY: A/V ratio scan
    # ══════════════════════════════════════════════════
    print("\n▶ A/V sensitivity scan")
    av_results = []
    for scale in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # Scale reactor while keeping L/R ratio constant
        Rscale = LAL.R * np.sqrt(1.0/scale) if scale != 1.0 else LAL.R
        Lscale = LAL.L * np.sqrt(1.0/scale) if scale != 1.0 else LAL.L
        rx = Reactor(R=Rscale, L=Lscale)
        av_ratio = rx.A / rx.V
        
        r = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.3, Q_sccm=40, eta=0.16,
                       wall_chem=WC, reactor=rx)
        av_results.append((av_ratio, r['alpha'], r['dissoc_frac'], r['n_F']*1e-6))
    print(f"  {len(av_results)} points scanned")

    # ══════════════════════════════════════════════════
    # LOAD BENCHMARK DATA
    # ══════════════════════════════════════════════════
    fig7_5  = load('lallement_fig7_alpha_5mTorr.csv')
    fig7_10 = load('lallement_fig7_alpha_10mTorr.csv')
    fig7_20 = load('lallement_fig7_alpha_20mTorr.csv')
    fig5a_c = load('lallement_fig5a_ne_calc.csv')
    fig5a_e = load('lallement_fig5a_ne_exp.csv')
    fig8_Fc = load('lallement_fig8_F_calc.csv')
    fig8_Fe = load('lallement_fig8_F_exp.csv')
    met_45a = load('mettler_fig4p5_SF6_actinometry.csv')
    met_45r = load('mettler_fig4p5_SF6_radicalprobe.csv')
    met_49_20 = load('mettler_fig4p9b_20mTorr.csv')
    met_49_40 = load('mettler_fig4p9b_40mTorr.csv')
    met_417_30 = load('mettler_fig4p17_30pctSF6_biasoff_density.csv')
    met_417_90 = load('mettler_fig4p17_90pctSF6_biasoff_density.csv')

    # ══════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════
    print("\n▶ Generating figures")
    
    C_gp = '#4575b4'    # gas-phase blue
    C_wc = '#d73027'    # wall chem red
    C_tel = '#ff7f00'   # TEL orange
    C_twc = '#984ea3'   # TEL wall chem purple
    C_pc = '#2d004b'    # paper calc
    C_pe = '#7fbc41'    # paper exp

    # ── FIG 1: KEY RESULT — α vs Ar (chemistry effect) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([r['frac_Ar'] for r in A1_alpha], [r['alpha'] for r in A1_alpha],
            '--', color=C_gp, lw=2, label='Gas-phase only')
    ax.plot([r['frac_Ar'] for r in A2_alpha], [r['alpha'] for r in A2_alpha],
            '-', color=C_wc, lw=2.5, label='Wall surface chemistry')
    if fig7_10 is not None:
        ax.plot(fig7_10[:,0], fig7_10[:,1], 'o', color=C_pc, ms=11,
                markerfacecolor='white', markeredgewidth=2.5, label='Lallement (digitized)')
    ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])', fontsize=14)
    ax.set_ylabel(r'$\alpha = n_-/n_e$', fontsize=14)
    ax.set_title('Chemistry Effect: α vs Ar fraction — Lallement geometry, 1500 W, 10 mTorr')
    ax.legend(fontsize=11); ax.set_xlim(-0.02, 0.75); ax.set_ylim(0, 55)
    plt.tight_layout()
    save_fig(fig, 'fig1_alpha_chemistry_effect.png')

    # ── FIG 2: α at 3 pressures with wall chem ──
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors_p = {5: '#74add1', 10: '#d73027', 20: '#053061'}
    markers_p = {5: '^', 10: 'o', 20: 's'}
    paper_p = {5: fig7_5, 10: fig7_10, 20: fig7_20}
    for p in [5, 10, 20]:
        d = A2_alpha_p[p]
        ax.plot([r['frac_Ar'] for r in d], [r['alpha'] for r in d],
                '-', color=colors_p[p], lw=2.5, label=f'Wall chem {p} mTorr')
        if paper_p[p] is not None:
            ax.plot(paper_p[p][:,0], paper_p[p][:,1], markers_p[p], color=colors_p[p], ms=10,
                    markerfacecolor='white', markeredgewidth=2, label=f'Lallement {p} mTorr')
    ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])'); ax.set_ylabel(r'$\alpha$')
    ax.set_title('Wall chemistry model — 3 pressures'); ax.legend(ncol=2); ax.set_ylim(0, 100)
    plt.tight_layout()
    save_fig(fig, 'fig2_alpha_3pressures.png')

    # ── FIG 3: SF6 dissociation (chemistry effect) ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([r['frac_Ar'] for r in A1_alpha], [r['dissoc_frac']*100 for r in A1_alpha],
            '--', color=C_gp, lw=2, label='Gas-phase only')
    ax.plot([r['frac_Ar'] for r in A2_alpha], [r['dissoc_frac']*100 for r in A2_alpha],
            '-', color=C_wc, lw=2.5, label='Wall surface chemistry')
    ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])'); ax.set_ylabel('SF$_6$ dissociation (%)')
    ax.set_title('SF$_6$ dissociation — Lallement geometry, 1500 W, 10 mTorr')
    ax.legend(fontsize=11); ax.set_ylim(0, 100)
    plt.tight_layout()
    save_fig(fig, 'fig3_dissociation_chemistry.png')

    # ── FIG 4: ne and [F] vs power ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for res, label, ls, c in [(A1_power, 'Gas-phase', '--', C_gp), (A2_power, 'Wall chem', '-', C_wc)]:
        P = [r['P_rf'] for r in res]
        ax1.plot(P, [r['ne']*1e-6/1e9 for r in res], ls, color=c, lw=2, label=label)
        ax2.plot(P, [r['n_F']*1e-6/1e13 for r in res], ls, color=c, lw=2, label=label)
    if fig5a_c is not None:
        ax1.plot(fig5a_c[:,0], fig5a_c[:,1], 's', color=C_pc, ms=8, label='Lallement calc.')
    if fig5a_e is not None:
        ax1.plot(fig5a_e[:,0], fig5a_e[:,1], 'o', color=C_pe, ms=8, markerfacecolor='white', label='Lallement exp.')
    if fig8_Fc is not None:
        ax2.plot(fig8_Fc[:,0], fig8_Fc[:,1], 's', color=C_pc, ms=8, label='Lallement calc.')
    if fig8_Fe is not None:
        ax2.plot(fig8_Fe[:,0], fig8_Fe[:,1], 'o', color=C_pe, ms=8, markerfacecolor='white', label='Lallement exp.')
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel(r'$n_e$ ($10^9$ cm$^{-3}$)')
    ax1.set_title(r'$n_e$ vs power — pure SF$_6$'); ax1.legend()
    ax2.set_xlabel('Power (W)'); ax2.set_ylabel(r'[F] ($10^{13}$ cm$^{-3}$)')
    ax2.set_title(r'[F] vs power — pure SF$_6$'); ax2.legend()
    plt.tight_layout()
    save_fig(fig, 'fig4_ne_F_vs_power.png')

    # ── FIG 5: GEOMETRY EFFECT — Lallement vs TEL ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Alpha
    for res, label, ls, c in [
        (A2_alpha, 'Lallement + wall chem', '-', C_wc),
        (B2_alpha, 'TEL + wall chem', '-', C_twc),
        (A1_alpha, 'Lallement gas-phase', '--', C_gp),
    ]:
        ax1.plot([r['frac_Ar'] for r in res], [r['alpha'] for r in res],
                ls, color=c, lw=2, label=label)
    if fig7_10 is not None:
        ax1.plot(fig7_10[:,0], fig7_10[:,1], 'o', color=C_pc, ms=10,
                markerfacecolor='white', markeredgewidth=2, label='Lallement data')
    ax1.set_xlabel(r'Ar fraction'); ax1.set_ylabel(r'$\alpha$')
    ax1.set_title(r'(a) $\alpha$ — geometry comparison'); ax1.legend(fontsize=8)
    ax1.set_xlim(-0.02, 0.75); ax1.set_ylim(0, 55)
    
    # Dissociation
    for res, label, ls, c in [
        (A2_alpha, 'Lallement + wall chem', '-', C_wc),
        (B2_alpha, 'TEL + wall chem', '-', C_twc),
        (A1_alpha, 'Lallement gas-phase', '--', C_gp),
    ]:
        ax2.plot([r['frac_Ar'] for r in res], [r['dissoc_frac']*100 for r in res],
                ls, color=c, lw=2, label=label)
    ax2.set_xlabel(r'Ar fraction'); ax2.set_ylabel('SF$_6$ dissociation (%)')
    ax2.set_title('(b) Dissociation — geometry comparison'); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 100)
    plt.tight_layout()
    save_fig(fig, 'fig5_geometry_effect.png')

    # ── FIG 6: A/V SENSITIVITY ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    avs = [x[0] for x in av_results]
    ax1.plot(avs, [x[1] for x in av_results], 'o-', color=C_wc, ms=8, lw=2)
    ax1.axvline(LAL.A/LAL.V, ls='--', color=C_gp, alpha=0.7, label=f'Lallement A/V={LAL.A/LAL.V:.0f}')
    ax1.axvline(TEL.A/TEL.V, ls='--', color=C_twc, alpha=0.7, label=f'TEL A/V={TEL.A/TEL.V:.0f}')
    ax1.set_xlabel('A/V (m⁻¹)'); ax1.set_ylabel(r'$\alpha$ at 30% Ar')
    ax1.set_title('(a) α sensitivity to A/V ratio'); ax1.legend()
    
    ax2.plot(avs, [x[2]*100 for x in av_results], 'o-', color=C_wc, ms=8, lw=2)
    ax2.axvline(LAL.A/LAL.V, ls='--', color=C_gp, alpha=0.7, label='Lallement')
    ax2.axvline(TEL.A/TEL.V, ls='--', color=C_twc, alpha=0.7, label='TEL')
    ax2.set_xlabel('A/V (m⁻¹)'); ax2.set_ylabel('Dissociation (%)')
    ax2.set_title('(b) Dissociation sensitivity to A/V'); ax2.legend()
    plt.tight_layout()
    save_fig(fig, 'fig6_AV_sensitivity.png')

    # ── FIG 7: TEL [F] vs power ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy([r['P_rf'] for r in B2_tel45], [r['n_F']*1e-6 for r in B2_tel45],
                '-', color=C_twc, lw=2.5, label='Model (TEL geom + wall chem)')
    if met_45a is not None:
        ax.semilogy(met_45a[:,0], met_45a[:,1]*1e-6, 'o', color='#d62728', ms=9,
                    markerfacecolor='white', markeredgewidth=2, label='Mettler actinometry')
    if met_45r is not None:
        ax.semilogy(met_45r[:,0], met_45r[:,1]*1e-6, 's', color='#d62728', ms=9,
                    label='Mettler radical probe')
    ax.set_xlabel('ICP Power (W)'); ax.set_ylabel(r'[F] (cm$^{-3}$)')
    ax.set_title('TEL: [F] vs power — 50 mTorr, 50% Ar')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'fig7_tel_F_vs_power.png')

    # ── FIG 8: TEL [F] vs SF6 flow ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy([r['sf6_sccm'] for r in B2_tel49_20], [r['n_F']*1e-6 for r in B2_tel49_20],
                '-', color='#1f77b4', lw=2, label='Model 20 mTorr')
    ax.semilogy([r['sf6_sccm'] for r in B2_tel49_40], [r['n_F']*1e-6 for r in B2_tel49_40],
                '-', color='#d62728', lw=2, label='Model 40 mTorr')
    if met_49_20 is not None:
        ax.semilogy(met_49_20[:,0], met_49_20[:,1]*1e-6, 'o', color='#1f77b4', ms=9,
                    markerfacecolor='white', markeredgewidth=2, label='Mettler 20 mTorr')
    if met_49_40 is not None:
        ax.semilogy(met_49_40[:,0], met_49_40[:,1]*1e-6, 's', color='#d62728', ms=9,
                    markerfacecolor='white', markeredgewidth=2, label='Mettler 40 mTorr')
    ax.set_xlabel(r'SF$_6$ flow (sccm)'); ax.set_ylabel(r'[F] (cm$^{-3}$)')
    ax.set_title('TEL: [F] vs SF₆ flow — 600/700 W, 100 sccm total')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'fig8_tel_F_vs_flow.png')

    # ── FIG 9: TEL center [F] ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ['30% SF₆', '90% SF₆']
    model_vals = [r_30['n_F']*1e-6, r_90['n_F']*1e-6]
    met_vals = []
    for d in [met_417_30, met_417_90]:
        if d is not None:
            met_vals.append(d[0, 1] * 1e-6)  # center value, m-3 -> cm-3
        else:
            met_vals.append(0)
    x = [0, 1]
    ax.bar(x, model_vals, 0.35, color=C_twc, alpha=0.7, label='Model (TEL + wall chem)')
    ax.bar([xi+0.35 for xi in x], met_vals, 0.35, color='#d62728', alpha=0.7, label='Mettler probe (center)')
    ax.set_xticks([xi+0.175 for xi in x]); ax.set_xticklabels(labels)
    ax.set_ylabel(r'[F] (cm$^{-3}$)'); ax.set_title('TEL center [F] — 1000 W, 10 mTorr, bias off')
    ax.legend(); ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout()
    save_fig(fig, 'fig9_tel_center_F.png')

    # ══════════════════════════════════════════════════
    # SUMMARY TABLES
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Chemistry Effect (Lallement geometry, 10 mTorr, 1500 W) ---")
    print(f"{'Ar%':>5} {'Gas-phase α':>12} {'Wall chem α':>12} {'Paper α':>10} {'Improvement':>14}")
    paper_a = {0.0: 40.1, 0.3: 24.9, 0.5: 14.9, 0.7: 2.7}
    for fAr, pa in paper_a.items():
        r0 = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=fAr, Q_sccm=40, eta=0.12, wall_chem=False, reactor=LAL)
        rw = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=fAr, Q_sccm=40, eta=0.16, wall_chem=WC, reactor=LAL)
        e0 = abs(r0['alpha']-pa)/pa*100
        ew = abs(rw['alpha']-pa)/pa*100
        print(f"{fAr*100:>4.0f}% {r0['alpha']:>12.1f} {rw['alpha']:>12.1f} {pa:>10.1f} {e0:.0f}% -> {ew:.0f}%")

    print(f"\n--- Geometry Effect (wall chem, 30% Ar, 1500 W) ---")
    print(f"{'Geometry':>12} {'α':>8} {'Dissoc%':>8} {'[F] cm-3':>12} {'ne cm-3':>12}")
    for name, rx, Q in [('Lallement', LAL, 40), ('TEL', TEL, 100)]:
        r = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.3, Q_sccm=Q, eta=0.16, wall_chem=WC, reactor=rx)
        print(f"{name:>12} {r['alpha']:>8.1f} {r['dissoc_frac']*100:>7.0f}% {r['n_F']*1e-6:>12.2e} {r['ne']*1e-6:>12.2e}")

    print(f"\n--- A/V Sensitivity (wall chem, 30% Ar) ---")
    print(f"{'A/V (m-1)':>10} {'α':>8} {'Dissoc%':>8}")
    for av, a, d, f in av_results:
        tag = ""
        if abs(av - LAL.A/LAL.V) < 2: tag = " ← Lallement"
        if abs(av - TEL.A/TEL.V) < 5: tag = " ← TEL"
        print(f"{av:>10.1f} {a:>8.1f} {d*100:>7.0f}%{tag}")

    print(f"\n✓ All {len(os.listdir(FIG))} figures saved to figures/")
    print("✓ Pipeline complete.")


if __name__ == '__main__':
    main()
