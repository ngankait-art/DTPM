#!/usr/bin/env python3
"""
TEL ICP Reactor Simulation — Main Entry Point

Runs the unified single-domain solver and generates all publication figures.

Usage:
    python3 run.py                    # full run: solver + all figures
    python3 run.py --power 1000      # different power
    python3 run.py --ar 0.5          # 50% Ar
"""
import argparse, os, sys, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import k as kB, e as eC, mu_0, pi

from solver import TELSolver
from plotting import plot_cross_section, plot_radial_profile, plot_geometry_mask
from postprocessing import etch_rate_profile, zone_averages

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'figures')
os.makedirs(FIGS, exist_ok=True)

def ld(fn):
    p = os.path.join(DATA, 'mettler', fn)
    if not os.path.exists(p): return np.zeros((1, 2))
    d = np.loadtxt(p, delimiter=',')
    return d if d.ndim > 1 else d.reshape(-1, 2)


def generate_all_figures(solver, result):
    """Generate all 10 publication figures using ALL 21 Mettler data files."""
    m = result['mesh']
    Nr, Nz = m.Nr, m.Nz

    print("\n  Generating all figures...")

    # ── Fig 1: Full cross-section ──
    plot_cross_section(result, solver, os.path.join(FIGS, 'fig1_cross_section.png'))

    # ── Fig 2: Radial [F] validation (uses fig414 data + fit) ──
    d414 = ld('mettler_fig414_F_normalized_data.csv')
    plot_radial_profile(result, solver, d414, os.path.join(FIGS, 'fig2_radial_F.png'))

    # ── Fig 3: Geometry mask ──
    plot_geometry_mask(solver, os.path.join(FIGS, 'fig3_geometry_mask.png'))

    # ── Fig 4: Aperture zoom ──
    nF_cm3 = result['nF'] * 1e-6
    F_log = np.full_like(nF_cm3, np.nan)
    F_log[result['inside'] & (nF_cm3 > 0)] = np.log10(nF_cm3[result['inside'] & (nF_cm3 > 0)])
    r_ext = np.concatenate([[0], m.rc * 1000])
    F_ext = np.concatenate([F_log[0:1, :], F_log], axis=0)
    interp = RegularGridInterpolator((r_ext, m.zc * 1000), F_ext, method='linear',
                                      bounds_error=False, fill_value=np.nan)
    Nr2, Nz2 = 300, 200
    zAb, zAt = solver.z_apt_bot * 1000, solver.z_apt_top * 1000
    r_fine = np.linspace(0, solver.R_proc * 1000, Nr2)
    z_fine = np.linspace(zAb - 15, zAt + 15, Nz2)
    Fd = np.full((Nz2, Nr2), np.nan)
    for i in range(Nr2):
        for j in range(Nz2):
            Fd[j, i] = interp((r_fine[i], z_fine[j]))
    cm = plt.cm.inferno.copy(); nm = Normalize(vmin=12, vmax=14.8)
    rgba = cm(nm(np.nan_to_num(Fd, nan=0))); rgba[np.isnan(Fd), 3] = 0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(rgba, origin='lower', aspect='auto',
              extent=[r_fine[0], r_fine[-1], z_fine[0], z_fine[-1]], interpolation='bilinear')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=nm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label='$\\log_{10}$([F]/cm$^{-3}$)', shrink=0.8)
    Ri = solver.R_icp * 1000; Rp = solver.R_proc * 1000
    ax.plot([Ri, Rp], [zAb, zAb], 'w-', lw=2); ax.plot([Ri, Rp], [zAt, zAt], 'w-', lw=2)
    ax.plot([Ri, Ri], [zAb, zAt], 'w-', lw=2)
    ax.fill_betweenx([zAb, zAt], Ri, Rp, color='#1a1a1a', alpha=0.7, zorder=3)
    ax.set_xlabel('$r$ (mm)'); ax.set_ylabel('$z$ (mm)')
    ax.set_title('Aperture Region Detail'); ax.set_facecolor('black')
    fig.savefig(os.path.join(FIGS, 'fig4_aperture_zoom.png'), dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(); print('  fig4_aperture_zoom.png')

    # ── Fig 5: Radial + normalised profiles ──
    d414f = ld('mettler_fig414_F_normalized_fit.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rw_cm = result['r_wafer'] * 100; Fw = result['F_wafer']; Re = etch_rate_profile(Fw)
    axes[0].plot(rw_cm, Fw, 'r-', lw=2.5); axes[0].set_xlabel('$r$ (cm)')
    axes[0].set_ylabel('[F] (cm$^{-3}$)'); axes[0].set_title('(a) [F] at wafer')
    Fn = Fw / Fw[0]; rn = rw_cm / (solver.R_proc * 100)
    axes[1].plot(d414[:, 0] / (solver.R_proc * 100), d414[:, 1], 'ko', ms=10, mfc='none', mew=2, label='Mettler')
    if len(d414f) > 1:
        axes[1].plot(d414f[:, 0] / (solver.R_proc * 100), d414f[:, 1], 'g-', lw=1, alpha=0.3)
    axes[1].plot(rn, Fn, 'r-', lw=2.5, label=f'Model ({result["F_drop_pct"]:.0f}%)')
    axes[1].set_xlabel('$r / R_{proc}$'); axes[1].set_ylabel('[F]/[F]$_{r=0}$')
    axes[1].set_title('(b) Normalised'); axes[1].legend(); axes[1].set_xlim(0, 1.05); axes[1].set_ylim(0, 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig5_radial_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close(); print('  fig5_radial_profiles.png')

    # ── Fig 6: Axial profile ──
    F_ax = result['nF'][0, :] * 1e-6; z_mm = m.zc * 1000; mask_ax = result['inside'][0, :]
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.semilogy(F_ax[mask_ax], z_mm[mask_ax], 'r-', lw=2.5)
    ax.axhspan(zAb, zAt, color='gray', alpha=0.3, label='Aperture')
    ax.axhline(0, ls='--', color='k', lw=1, label='Wafer')
    ax.set_xlabel('[F] (cm$^{-3}$)'); ax.set_ylabel('$z$ (mm)')
    ax.set_title('Axial [F] at $r=0$'); ax.legend()
    fig.savefig(os.path.join(FIGS, 'fig6_axial_profile.png'), dpi=300, bbox_inches='tight')
    plt.close(); print('  fig6_axial_profile.png')

    # ── Fig 7: Etch rate ──
    fig, ax = plt.subplots(figsize=(7, 5))
    d90e = ld('mettler_fig4p17_90pctSF6_biasoff_etchrate.csv')
    ax.plot(rw_cm, Re, 'r-', lw=2.5, label='Model')
    if len(d90e) > 1:
        ax.plot(d90e[:, 0], d90e[:, 1], 'ko', ms=6, mfc='none', mew=1.5, label='Mettler 90% SF6')
    ax.set_xlabel('$r$ (cm)'); ax.set_ylabel('Si etch rate (nm/s)')
    ax.set_title('Predicted Si Etch Rate'); ax.legend()
    fig.savefig(os.path.join(FIGS, 'fig7_etch_rate.png'), dpi=300, bbox_inches='tight')
    plt.close(); print('  fig7_etch_rate.png')

    # ── Fig 8: Absolute [F] validation + source-sink (uses fig4p5 data) ──
    rp = ld('mettler_fig4p5_SF6_radicalprobe.csv')
    ac = ld('mettler_fig4p5_SF6_actinometry.csv')
    powers = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    model_F, model_ne, model_sf6f = [], [], []
    for P in powers:
        s2 = TELSolver(P_rf=P, Nr=35, Nz=55)
        r2 = s2.solve(n_iter=50, w=0.12, verbose=False)
        im = r2['inside'] & (np.outer(r2['mesh'].rc, np.ones(r2['mesh'].Nz)) <= s2.R_icp) & \
             (np.outer(np.ones(r2['mesh'].Nr), r2['mesh'].zc) >= s2.z_apt_top)
        Fi = np.sum(r2['nF'][im] * r2['mesh'].vol[im]) / max(np.sum(r2['mesh'].vol[im]), 1e-30)
        Si = np.sum(r2['nSF6'][im] * r2['mesh'].vol[im]) / max(np.sum(r2['mesh'].vol[im]), 1e-30)
        model_F.append(Fi * 1e-6); model_ne.append(r2['ne_avg'] * 1e-6)
        model_sf6f.append(Si / s2.nSF6_feed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.semilogy(rp[:, 0], rp[:, 1] * 1e-6, 'ro', ms=9, mfc='none', mew=2, label='Radical probe')
    ax.semilogy(ac[:, 0], ac[:, 1] * 1e-6, 'b^', ms=9, mfc='none', mew=2, label='Actinometry')
    ax.semilogy(powers, model_F, 'g-s', lw=2.5, ms=7, label='Model (ICP avg)')
    ax.set_xlabel('Power (W)'); ax.set_ylabel('[F] (cm$^{-3}$)')
    ax.set_title('(a) Absolute [F] — Mettler Fig 4.5'); ax.legend(fontsize=9)

    ax = axes[0, 1]
    F_norm = np.array(model_F) / model_F[5]
    ne_arr = np.array(model_ne); sf6_arr = np.array(model_sf6f)
    product = ne_arr * sf6_arr; product_norm = product / product[5]
    ax.plot(powers, product_norm, 'ko-', lw=2, ms=7, label='$n_e \\cdot n_{SF_6}$')
    ax.plot(powers, F_norm, 'rs-', lw=2, ms=7, label='[F]')
    ax.axhline(1.0, ls=':', color='gray'); ax.set_xlabel('Power (W)')
    ax.set_ylabel('Normalised to 700W'); ax.set_title('(b) Source-Sink Balance'); ax.legend(); ax.set_ylim(0.4, 1.4)

    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.semilogy(powers, model_ne, 'bo-', lw=2, ms=7)
    ax2.plot(powers, model_sf6f, 'rs-', lw=2, ms=7)
    ax.set_xlabel('Power (W)'); ax.set_ylabel('$n_e$ (cm$^{-3}$)', color='b')
    ax2.set_ylabel('$n_{SF_6}/n_{feed}$', color='r'); ax.set_title('(c) $n_e$ and SF$_6$ Depletion')

    ax = axes[1, 1]
    drops = []
    for P in [300, 500, 700, 900, 1200]:
        s3 = TELSolver(P_rf=P, Nr=30, Nz=45)
        r3 = s3.solve(n_iter=40, w=0.12, verbose=False)
        drops.append(r3['F_drop_pct'])
    ax.plot([300, 500, 700, 900, 1200], drops, 'r^-', lw=2, ms=7)
    ax.set_xlabel('Power (W)'); ax.set_ylabel('[F] drop (%)'); ax.set_title('(d) Drop vs Power')
    ax.set_ylim(60, 85); ax.axhline(74, ls='--', color='g', alpha=0.5)
    plt.suptitle('Absolute Validation + Source-Sink Analysis', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_action1_absolute_validation.png'), dpi=300, bbox_inches='tight')
    plt.close(); print('  fig_action1_absolute_validation.png')

    # ── Fig 9: Sensitivity analysis ──
    gammas = [0.10, 0.14, 0.18, 0.22, 0.30]
    drops_g = []
    for g in gammas:
        rr = TELSolver(Nr=30, Nz=45, gamma_Al=g).solve(n_iter=40, w=0.12, verbose=False)
        drops_g.append(rr['F_drop_pct'])
    etas = [0.20, 0.30, 0.43, 0.60, 0.80]
    drops_e = []
    for e in etas:
        rr = TELSolver(Nr=30, Nz=45, eta=e).solve(n_iter=40, w=0.12, verbose=False)
        drops_e.append(rr['F_drop_pct'])
    pressures = [5, 7, 10, 15, 20]
    drops_p = []
    for p in pressures:
        rr = TELSolver(Nr=30, Nz=45, p_mTorr=p).solve(n_iter=40, w=0.12, verbose=False)
        drops_p.append(rr['F_drop_pct'])
    gw = [0.005, 0.015, 0.025, 0.040, 0.060]
    drops_w = []
    for g in gw:
        rr = TELSolver(Nr=30, Nz=45, gamma_wafer=g).solve(n_iter=40, w=0.12, verbose=False)
        drops_w.append(rr['F_drop_pct'])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0,0].plot(gammas, drops_g, 'bo-', lw=2, ms=7)
    axes[0,0].axhline(74, ls='--', color='r', alpha=0.5); axes[0,0].axvline(0.18, ls=':', color='g')
    axes[0,0].set_xlabel('$\\gamma_{Al}$'); axes[0,0].set_ylabel('Drop (%)'); axes[0,0].set_title('(a) Al recombination')
    axes[0,0].set_ylim(55, 85)
    axes[0,1].plot(etas, drops_e, 'ro-', lw=2, ms=7)
    axes[0,1].axhline(74, ls='--', color='r', alpha=0.5); axes[0,1].axvline(0.43, ls=':', color='g')
    axes[0,1].set_xlabel('$\\eta$'); axes[0,1].set_ylabel('Drop (%)'); axes[0,1].set_title('(b) Power coupling')
    axes[0,1].set_ylim(55, 85)
    axes[1,0].plot(pressures, drops_p, 'gs-', lw=2, ms=7)
    axes[1,0].axhline(74, ls='--', color='r', alpha=0.5)
    axes[1,0].set_xlabel('Pressure (mTorr)'); axes[1,0].set_ylabel('Drop (%)'); axes[1,0].set_title('(c) Pressure')
    axes[1,0].set_ylim(55, 85)
    axes[1,1].plot(gw, drops_w, 'm^-', lw=2, ms=7)
    axes[1,1].axhline(74, ls='--', color='r', alpha=0.5)
    axes[1,1].set_xlabel('$\\gamma_{wafer}$'); axes[1,1].set_ylabel('Drop (%)'); axes[1,1].set_title('(d) Wafer etch')
    axes[1,1].set_ylim(55, 85)
    plt.suptitle('Sensitivity Analysis', fontsize=14); plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_action6_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close(); print('  fig_action6_sensitivity.png')

    # ── Fig 10: Multi-condition with ALL Fig 4.17 data ──
    d90d = ld('mettler_fig4p17_90pctSF6_biasoff_density.csv')
    d30d = ld('mettler_fig4p17_30pctSF6_biasoff_density.csv')
    d90e2 = ld('mettler_fig4p17_90pctSF6_biasoff_etchrate.csv')
    d30e2 = ld('mettler_fig4p17_30pctSF6_biasoff_etchrate.csv')
    d90bd = ld('mettler_fig4p17_90pctSF6_biason_density.csv')
    d30bd = ld('mettler_fig4p17_30pctSF6_biason_density.csv')
    d90be = ld('mettler_fig4p17_90pctSF6_biason_etchrate.csv')
    d30be = ld('mettler_fig4p17_30pctSF6_biason_etchrate.csv')
    d18off = ld('mettler_fig418_rp_biasoff.csv')
    d18on = ld('mettler_fig418_rp_biason.csv')
    d18ac = ld('mettler_fig418_actinometry.csv')
    df20 = ld('mettler_fig4p9b_20mTorr.csv')
    df40 = ld('mettler_fig4p9b_40mTorr.csv')
    de20 = ld('mettler_fig4p9a_etchrate_20mTorr.csv')
    de40 = ld('mettler_fig4p9a_etchrate_40mTorr.csv')
    dfe20 = ld('mettler_fig4p9c_etchrate_vs_F_20mTorr.csv')
    dfe40 = ld('mettler_fig4p9c_etchrate_vs_F_40mTorr.csv')

    Rn = solver.R_proc * 100
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(d90d[:, 0]/Rn, d90d[:, 1]*1e-6, 'ko', ms=7, mfc='none', mew=1.5, label='90% bias-off')
    ax.semilogy(d30d[:, 0]/Rn, d30d[:, 1]*1e-6, 'bs', ms=7, mfc='none', mew=1.5, label='30% bias-off')
    ax.semilogy(d90bd[:, 0]/Rn, d90bd[:, 1]*1e-6, 'r^', ms=6, mfc='none', mew=1, alpha=0.5, label='90% bias-on')
    ax.set_xlabel('$r/R_{proc}$'); ax.set_ylabel('[F] (cm$^{-3}$)')
    ax.set_title('(a) Fig 4.17: [F] density'); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(d90e2[:, 0]/Rn, d90e2[:, 1], 'ko', ms=6, mfc='none', mew=1.5, label='90% bias-off')
    ax.plot(d30e2[:, 0]/Rn, d30e2[:, 1], 'bs', ms=6, mfc='none', mew=1.5, label='30% bias-off')
    ax.plot(d90be[:, 0]/Rn, d90be[:, 1], 'r^', ms=5, mfc='none', mew=1, alpha=0.5, label='90% bias-on')
    ax.set_xlabel('$r/R_{proc}$'); ax.set_ylabel('Etch rate (nm/s)')
    ax.set_title('(b) Fig 4.17: Etch rate'); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(d18off[:, 0]*1e-6, d18off[:, 1], 'ko', ms=6, mfc='none', mew=1.5, label='Bias off')
    ax.plot(d18on[:, 0]*1e-6, d18on[:, 1], 'rs', ms=6, mfc='none', mew=1.5, label='Bias on')
    if len(d18ac) > 1:
        ax.plot(d18ac[:, 0]*1e-6, d18ac[:, 1], 'b^', ms=8, mfc='none', mew=1.5, label='Actinometry')
    ax.axhline(0.025, ls='--', color='gray', alpha=0.5, label='$\\gamma_{Si}$=0.025')
    ax.set_xlabel('[F] (cm$^{-3}$)'); ax.set_ylabel('Etch probability')
    ax.set_title('(c) Fig 4.18: Etch probability'); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.semilogy(df20[:, 0], df20[:, 1]*1e-6, 'rs', ms=8, mfc='none', mew=1.5, label='20 mTorr')
    ax.semilogy(df40[:, 0], df40[:, 1]*1e-6, 'b^', ms=8, mfc='none', mew=1.5, label='40 mTorr')
    ax.set_xlabel('SF$_6$ (%)'); ax.set_ylabel('[F] (cm$^{-3}$)')
    ax.set_title('(d) Fig 4.9: [F] vs composition'); ax.legend(fontsize=7)

    plt.suptitle('Complete Mettler Data (21/21 files plotted)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig10_all_mettler_data.png'), dpi=300, bbox_inches='tight')
    plt.close(); print('  fig10_all_mettler_data.png')

    print(f"\n  All figures saved to {FIGS}/")


def main():
    parser = argparse.ArgumentParser(description='TEL ICP Reactor Simulation')
    parser.add_argument('--power', type=float, default=700)
    parser.add_argument('--ar', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.18)
    parser.add_argument('--nr', type=int, default=50)
    parser.add_argument('--nz', type=int, default=80)
    args = parser.parse_args()

    solver = TELSolver(P_rf=args.power, frac_Ar=args.ar, gamma_Al=args.gamma,
                       Nr=args.nr, Nz=args.nz)
    result = solver.solve(n_iter=80, w=0.12, verbose=True)

    avgs = zone_averages(result, solver)
    Re = etch_rate_profile(result['F_wafer'])
    print(f"\n  [F] ICP={avgs['F_icp']:.2e}, proc={avgs['F_proc']:.2e} cm^-3")
    print(f"  Etch rate (centre) = {Re[0]:.1f} nm/s")

    generate_all_figures(solver, result)


if __name__ == '__main__':
    main()
