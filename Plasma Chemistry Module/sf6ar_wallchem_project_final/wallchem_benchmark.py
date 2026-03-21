#!/usr/bin/env python3
"""
Wall Chemistry Model — Full Benchmarking
==========================================
Compares the original model (no wall chem) vs the new model (with Kokkoris-inspired
wall surface chemistry) against digitized Lallement (2009) and Mettler (2025) data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sf6_wallchem_model import solve_model, sweep_with_continuation

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'lines.linewidth': 2.2,
    'lines.markersize': 7, 'axes.grid': True, 'grid.alpha': 0.3,
})

C_orig  = '#4575b4'   # blue - original model
C_wall  = '#d73027'   # red - wall chem model
C_paper_c = '#2d004b' # dark purple - paper calc
C_paper_e = '#7fbc41' # green - paper exp

# Optimal wall chemistry parameters
WC_OPT = {'s_F': 0.00105, 's_SFx': 0.00056, 'p_fluor': 0.025,
           'p_wallrec': 0.007, 'p_FF': 0.0035, 'p_F_SF3': 0.5, 'p_F_SF4': 0.2}

# Load digitized Lallement data
DATA = '/mnt/user-data/outputs'
fig7_5mT  = np.genfromtxt(f'{DATA}/lallement_fig7_alpha_5mTorr.csv', delimiter=',')
fig7_10mT = np.genfromtxt(f'{DATA}/lallement_fig7_alpha_10mTorr.csv', delimiter=',')
fig7_20mT = np.genfromtxt(f'{DATA}/lallement_fig7_alpha_20mTorr.csv', delimiter=',')
fig5a_ne_calc = np.genfromtxt(f'{DATA}/lallement_fig5a_ne_calc.csv', delimiter=',')
fig5a_ne_exp  = np.genfromtxt(f'{DATA}/lallement_fig5a_ne_exp.csv', delimiter=',')
fig8_F_calc = np.genfromtxt(f'{DATA}/lallement_fig8_F_calc.csv', delimiter=',')
fig8_F_exp  = np.genfromtxt(f'{DATA}/lallement_fig8_F_exp.csv', delimiter=',')

OUT = '/mnt/user-data/outputs'

# ═══════════════════════════════════════════════════════════════
print("="*65)
print("Wall Chemistry Model — Full Benchmarking")
print("="*65)

# --- Alpha vs Ar at 10 mTorr: original vs wall chem ---
print("\n▶ Alpha vs Ar fraction (10 mTorr, 1500 W)")
fracs = np.linspace(0, 0.8, 41)
base_orig = dict(p_mTorr=10, Q_sccm=40, eta=0.12, P_rf=1500)
base_wall = dict(p_mTorr=10, Q_sccm=40, eta=0.16, P_rf=1500)

res_orig = sweep_with_continuation('frac_Ar', fracs, {**base_orig, 'wall_chem': False}, verbose=False)
res_wall = sweep_with_continuation('frac_Ar', fracs, {**base_wall, 'wall_chem': WC_OPT}, verbose=False)
print("  Done.")

# Also at 5 and 20 mTorr
print("▶ Alpha at 5 and 20 mTorr")
res_alpha = {}
for p in [5, 10, 20]:
    b = {**base_wall, 'p_mTorr': p, 'wall_chem': WC_OPT}
    res_alpha[p] = sweep_with_continuation('frac_Ar', fracs, b, verbose=False)
print("  Done.")

# --- Power sweep: both models ---
print("▶ Power sweeps")
powers = np.linspace(200, 2000, 51)
res_P_orig = sweep_with_continuation('P_rf', powers, {**base_orig, 'frac_Ar': 0.0, 'wall_chem': False}, verbose=False)
res_P_wall = sweep_with_continuation('P_rf', powers, {**base_wall, 'frac_Ar': 0.0, 'wall_chem': WC_OPT}, verbose=False)
print("  Done.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: KEY RESULT — Alpha vs Ar% comparison
# ═══════════════════════════════════════════════════════════════
print("\n▶ Generating figures...")

fig1, ax = plt.subplots(figsize=(10, 7))

fAr_o = [r['frac_Ar'] for r in res_orig]
fAr_w = [r['frac_Ar'] for r in res_wall]

ax.plot(fAr_o, [r['alpha'] for r in res_orig], '--', color=C_orig, lw=2, label='Original model (no wall chem)')
ax.plot(fAr_w, [r['alpha'] for r in res_wall], '-', color=C_wall, lw=2.5, label='Wall chemistry model')
ax.plot(fig7_10mT[:,0], fig7_10mT[:,1], 'o', color=C_paper_c, ms=11, markerfacecolor='white',
        markeredgewidth=2.5, label='Lallement calc. (digitized)')

ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])', fontsize=14)
ax.set_ylabel(r'$\alpha = n_-/n_e$', fontsize=14)
ax.set_title(r'Electronegativity vs Ar fraction — 1500 W, 10 mTorr' + '\n'
             'Effect of wall surface chemistry on SF$_6$ regeneration', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-0.02, 0.75); ax.set_ylim(0, 55)
plt.tight_layout()
fig1.savefig(f'{OUT}/wallchem_alpha_comparison.png', dpi=200, bbox_inches='tight')
print("  Saved wallchem_alpha_comparison.png")

# FIGURE 2: Alpha at 3 pressures with wall chem
fig2, ax = plt.subplots(figsize=(9, 6.5))
colors = {5: '#74add1', 10: '#d73027', 20: '#053061'}
paper_data = {5: fig7_5mT, 10: fig7_10mT, 20: fig7_20mT}
markers = {5: '^', 10: 'o', 20: 's'}

for p in [5, 10, 20]:
    fAr_m = [r['frac_Ar'] for r in res_alpha[p]]
    al_m = [r['alpha'] for r in res_alpha[p]]
    ax.plot(fAr_m, al_m, '-', color=colors[p], lw=2.5, label=f'Model {p} mTorr')
    d = paper_data[p]
    ax.plot(d[:,0], d[:,1], markers[p], color=colors[p], ms=10,
            markerfacecolor='white', markeredgewidth=2, label=f'Lallement {p} mTorr')

ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])', fontsize=13)
ax.set_ylabel(r'$\alpha = n_-/n_e$', fontsize=13)
ax.set_title('Wall chemistry model — α vs Ar fraction at 3 pressures', fontsize=13)
ax.legend(ncol=2, fontsize=9); ax.set_xlim(-0.02, 0.75); ax.set_ylim(0, 100)
plt.tight_layout()
fig2.savefig(f'{OUT}/wallchem_fig7_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved wallchem_fig7_overlay.png")

# FIGURE 3: ne and [F] vs power — wall chem vs original
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

P_o = [r['P_rf'] for r in res_P_orig]
P_w = [r['P_rf'] for r in res_P_wall]

ax1.plot(P_o, [r['ne']*1e-6/1e9 for r in res_P_orig], '--', color=C_orig, lw=2, label='Original')
ax1.plot(P_w, [r['ne']*1e-6/1e9 for r in res_P_wall], '-', color=C_wall, lw=2.5, label='Wall chem')
ax1.plot(fig5a_ne_calc[:,0], fig5a_ne_calc[:,1], 's', color=C_paper_c, ms=8,
         markerfacecolor=C_paper_c, label='Lallement calc.')
ax1.plot(fig5a_ne_exp[:,0], fig5a_ne_exp[:,1], 's', color=C_paper_e, ms=8,
         markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax1.set_xlabel('Coupled power (W)'); ax1.set_ylabel(r'$n_e$ ($10^9$ cm$^{-3}$)')
ax1.set_title(r'(a) $n_e$ vs power — pure SF$_6$, 10 mTorr'); ax1.legend()
ax1.set_xlim(0, 2100); ax1.set_ylim(0, 14)

ax2.plot(P_o, [r['n_F']*1e-6/1e13 for r in res_P_orig], '--', color=C_orig, lw=2, label='Original')
ax2.plot(P_w, [r['n_F']*1e-6/1e13 for r in res_P_wall], '-', color=C_wall, lw=2.5, label='Wall chem')
ax2.plot(fig8_F_calc[:,0], fig8_F_calc[:,1], 's', color=C_paper_c, ms=8,
         markerfacecolor=C_paper_c, label='Lallement calc.')
ax2.plot(fig8_F_exp[:,0], fig8_F_exp[:,1], 'o', color=C_paper_e, ms=8,
         markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax2.set_xlabel('Coupled power (W)'); ax2.set_ylabel(r'[F] ($10^{13}$ cm$^{-3}$)')
ax2.set_title(r'(b) [F] vs power — pure SF$_6$, 10 mTorr'); ax2.legend()
ax2.set_xlim(0, 2100); ax2.set_ylim(0, 25)

plt.tight_layout()
fig3.savefig(f'{OUT}/wallchem_ne_F_vs_power.png', dpi=200, bbox_inches='tight')
print("  Saved wallchem_ne_F_vs_power.png")

# FIGURE 4: SF6 dissociation comparison
fig4, ax = plt.subplots(figsize=(9, 6))
ax.plot(fAr_o, [r['dissoc_frac']*100 for r in res_orig], '--', color=C_orig, lw=2, label='Original')
ax.plot(fAr_w, [r['dissoc_frac']*100 for r in res_wall], '-', color=C_wall, lw=2.5, label='Wall chem')
ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])', fontsize=13)
ax.set_ylabel('SF$_6$ dissociation (%)', fontsize=13)
ax.set_title('SF$_6$ dissociation fraction — 1500 W, 10 mTorr', fontsize=13)
ax.legend(fontsize=11); ax.set_xlim(-0.02, 0.82); ax.set_ylim(0, 100)
plt.tight_layout()
fig4.savefig(f'{OUT}/wallchem_dissociation.png', dpi=200, bbox_inches='tight')
print("  Saved wallchem_dissociation.png")

# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("QUANTITATIVE COMPARISON AT 1500 W, 10 mTorr")
print("="*70)

print(f"\n{'Quantity':<20} {'Original':>12} {'Wall chem':>12} {'Paper calc':>12} {'Improvement':>14}")
print("-"*72)

ro = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.0, eta=0.12, wall_chem=False, Q_sccm=40)
rw = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.0, eta=0.16, wall_chem=WC_OPT, Q_sccm=40)

print(f"{'ne (cm⁻³)':<20} {ro['ne']*1e-6:>12.2e} {rw['ne']*1e-6:>12.2e} {'6.06e+09':>12} {'✓' if abs(rw['ne']*1e-6-6.06e9)/6.06e9 < 0.15 else '':>14}")
print(f"{'Te (eV)':<20} {ro['Te']:>12.2f} {rw['Te']:>12.2f} {'2.94':>12}")
print(f"{'[F] (cm⁻³)':<20} {ro['n_F']*1e-6:>12.2e} {rw['n_F']*1e-6:>12.2e} {'9.74e+13':>12}")
print(f"{'α (0% Ar)':<20} {ro['alpha']:>12.1f} {rw['alpha']:>12.1f} {'40.1':>12} {'✓ EXACT' if abs(rw['alpha']-40.1)<1 else ''}")
print(f"{'dissoc %':<20} {ro['dissoc_frac']*100:>11.0f}% {rw['dissoc_frac']*100:>11.0f}%")

for fAr, pa in [(0.3, 24.9), (0.5, 14.9), (0.7, 2.7)]:
    ro2 = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=fAr, eta=0.12, wall_chem=False, Q_sccm=40)
    rw2 = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=fAr, eta=0.16, wall_chem=WC_OPT, Q_sccm=40)
    err_o = abs(ro2['alpha']-pa)/pa*100
    err_w = abs(rw2['alpha']-pa)/pa*100
    print(f"{'α ('+str(int(fAr*100))+'% Ar)':<20} {ro2['alpha']:>12.1f} {rw2['alpha']:>12.1f} {pa:>12.1f} {err_o:.0f}%→{err_w:.0f}%")

print("\n✓ All benchmarking figures saved.")
