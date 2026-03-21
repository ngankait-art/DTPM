#!/usr/bin/env python3
"""
Publication-Quality Overlay Plots — WebPlotDigitizer Data
=========================================================
Uses digitized data from Lallement et al. PSST 18 (2009) 025001.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sf6_global_model_final import solve_model, sweep_with_continuation
except ImportError:
    from sf6_global_model_final import solve_model, sweep_with_continuation

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'figure.dpi': 150, 'lines.linewidth': 2.2,
    'lines.markersize': 7, 'axes.grid': True, 'grid.alpha': 0.3,
})

C_model = '#2166ac'
C_calc  = '#b2182b'
C_exp   = '#ef8a62'

# ═══════════════════════════════════════════════════════════════
# LOAD DIGITIZED DATA
# ═══════════════════════════════════════════════════════════════
print("Loading digitized Lallement data...")

DATA = '/mnt/user-data/outputs'

fig5a_ne_calc = np.genfromtxt(f'{DATA}/lallement_fig5a_ne_calc.csv', delimiter=',')
fig5a_ne_exp  = np.genfromtxt(f'{DATA}/lallement_fig5a_ne_exp.csv', delimiter=',')
fig5b_Te_calc = np.genfromtxt(f'{DATA}/lallement_fig5b_Te_calc.csv', delimiter=',')
fig5b_Te_exp  = np.genfromtxt(f'{DATA}/lallement_fig5b_Te_exp.csv', delimiter=',')
fig5c_ne_calc = np.genfromtxt(f'{DATA}/lallement_fig5c_ne_calc.csv', delimiter=',')
fig5c_ne_exp  = np.genfromtxt(f'{DATA}/lallement_fig5c_ne_exp.csv', delimiter=',')
fig5d_Te_calc = np.genfromtxt(f'{DATA}/lallement_fig5d_Te_calc.csv', delimiter=',')
fig5d_Te_exp  = np.genfromtxt(f'{DATA}/lallement_fig5d_Te_exp.csv', delimiter=',')
fig7_5mT  = np.genfromtxt(f'{DATA}/lallement_fig7_alpha_5mTorr.csv', delimiter=',')
fig7_10mT = np.genfromtxt(f'{DATA}/lallement_fig7_alpha_10mTorr.csv', delimiter=',')
fig7_20mT = np.genfromtxt(f'{DATA}/lallement_fig7_alpha_20mTorr.csv', delimiter=',')
fig8_F_calc  = np.genfromtxt(f'{DATA}/lallement_fig8_F_calc.csv', delimiter=',')
fig8_F_exp   = np.genfromtxt(f'{DATA}/lallement_fig8_F_exp.csv', delimiter=',')
fig8_ne_calc = np.genfromtxt(f'{DATA}/lallement_fig8_ne_calc.csv', delimiter=',')

print("  14 files loaded.")

# ═══════════════════════════════════════════════════════════════
# GENERATE MODEL DATA
# ═══════════════════════════════════════════════════════════════
print("Generating model data...")

base = dict(p_mTorr=10, Q_sccm=40, eta=0.12)

# Power sweep (pure SF6)
powers = np.linspace(200, 2000, 37)
res_P = sweep_with_continuation('P_rf', powers, {**base, 'frac_Ar': 0.0}, verbose=False)
print("  Power sweep done.")

# Ar fraction sweep (1500W)
fracs = np.linspace(0, 1.0, 21)
res_Ar = sweep_with_continuation('frac_Ar', fracs, {**base, 'P_rf': 1500}, verbose=False)
print("  Ar fraction sweep done.")

# Alpha vs Ar at 3 pressures
res_alpha = {}
for p in [5, 10, 20]:
    b = base.copy(); b['p_mTorr'] = p; b['P_rf'] = 1500
    res_alpha[p] = sweep_with_continuation('frac_Ar', np.linspace(0, 0.8, 17), b, verbose=False)
print("  Alpha sweeps done.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5 OVERLAY
# ═══════════════════════════════════════════════════════════════
print("Generating figures...")

fig5, axes = plt.subplots(2, 2, figsize=(13, 10))

# --- (a) ne vs Power ---
ax = axes[0,0]
P_mod = [r['P_rf'] for r in res_P]
ne_mod = [r['ne']*1e-6/1e9 for r in res_P]  # 10⁹ cm⁻³
ax.plot(P_mod, ne_mod, '-', color=C_model, lw=2.5, label='This model')
ax.plot(fig5a_ne_calc[:,0], fig5a_ne_calc[:,1], 's', color=C_calc, ms=9,
        markerfacecolor=C_calc, label='Lallement calc.')
ax.plot(fig5a_ne_exp[:,0], fig5a_ne_exp[:,1], 's', color=C_exp, ms=9,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel('Coupled power (W)')
ax.set_ylabel(r'$n_e$ ($10^9$ cm$^{-3}$)')
ax.set_title(r'(a) $n_e$ vs power — SF$_6$, 10 mTorr')
ax.legend(loc='upper left')
ax.set_xlim(0, 2100); ax.set_ylim(0, 12)

# --- (b) Te vs Power ---
ax = axes[0,1]
Te_mod = [r['Te'] for r in res_P]
ax.plot(P_mod, Te_mod, '-', color=C_model, lw=2.5, label='This model')
ax.plot(fig5b_Te_calc[:,0], fig5b_Te_calc[:,1], 's', color=C_calc, ms=9,
        markerfacecolor=C_calc, label='Lallement calc.')
ax.plot(fig5b_Te_exp[:,0], fig5b_Te_exp[:,1], 's', color=C_exp, ms=9,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel('Coupled power (W)')
ax.set_ylabel(r'$T_e$ (eV)')
ax.set_title(r'(b) $T_e$ vs power — SF$_6$, 10 mTorr')
ax.legend()
ax.set_xlim(0, 2100); ax.set_ylim(0, 5)

# --- (c) ne vs Ar% (log scale) ---
ax = axes[1,0]
fAr_mod = [r['frac_Ar'] for r in res_Ar]
ne_Ar_mod = [r['ne']*1e-6 for r in res_Ar]  # cm⁻³
ax.plot(fAr_mod, ne_Ar_mod, '-', color=C_model, lw=2.5, label='This model')
ax.plot(fig5c_ne_calc[:,0], fig5c_ne_calc[:,1], 's', color=C_calc, ms=9,
        markerfacecolor=C_calc, label='Lallement calc.')
ax.plot(fig5c_ne_exp[:,0], fig5c_ne_exp[:,1], 's', color=C_exp, ms=9,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])')
ax.set_ylabel(r'$n_e$ (cm$^{-3}$)')
ax.set_title(r'(c) $n_e$ vs Ar fraction — 1500 W, 10 mTorr')
ax.set_yscale('log'); ax.legend()
ax.set_xlim(-0.02, 1.05); ax.set_ylim(1e9, 1e12)

# --- (d) Te vs Ar% ---
ax = axes[1,1]
Te_Ar_mod = [r['Te'] for r in res_Ar]
fAr_mod_pct = [r['frac_Ar']*100 for r in res_Ar]
ax.plot(fAr_mod_pct, Te_Ar_mod, '-', color=C_model, lw=2.5, label='This model')
ax.plot(fig5d_Te_calc[:,0], fig5d_Te_calc[:,1], 's', color=C_calc, ms=9,
        markerfacecolor=C_calc, label='Lallement calc.')
ax.plot(fig5d_Te_exp[:,0], fig5d_Te_exp[:,1], 's', color=C_exp, ms=9,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar]) %')
ax.set_ylabel(r'$T_e$ (eV)')
ax.set_title(r'(d) $T_e$ vs Ar% — 1500 W, 10 mTorr')
ax.legend()
ax.set_xlim(-2, 105); ax.set_ylim(0, 5)

fig5.suptitle('Overlay with Lallement et al. PSST 18 (2009) — Figure 5\n(digitized via WebPlotDigitizer)',
              fontsize=13, y=1.02)
plt.tight_layout()
fig5.savefig('/mnt/user-data/outputs/fig5_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved fig5_overlay.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7 OVERLAY — Alpha vs Ar%
# ═══════════════════════════════════════════════════════════════

fig7, ax = plt.subplots(figsize=(9, 6.5))

model_colors = {5: '#74add1', 10: '#2166ac', 20: '#053061'}
paper_colors  = {5: '#fc8d59', 10: '#e34a33', 20: '#b30000'}
paper_markers = {5: '^', 10: 'o', 20: 's'}
paper_data    = {5: fig7_5mT, 10: fig7_10mT, 20: fig7_20mT}

for p in [5, 10, 20]:
    # Model
    fAr_m = [r['frac_Ar'] for r in res_alpha[p]]
    al_m = [r['alpha'] for r in res_alpha[p]]
    ax.plot(fAr_m, al_m, '-', color=model_colors[p], lw=2.5,
            label=f'Model {p} mTorr')
    # Paper
    d = paper_data[p]
    ax.plot(d[:,0], d[:,1], paper_markers[p], color=paper_colors[p], ms=10,
            markerfacecolor='white', markeredgewidth=2.0,
            label=f'Lallement {p} mTorr')

ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar])', fontsize=13)
ax.set_ylabel(r'$\alpha = n_-/n_e$', fontsize=13)
ax.set_title('Electronegativity vs Ar fraction — 1500 W\n'
             'Overlay with Lallement et al. (digitized)', fontsize=13)
ax.legend(ncol=2, loc='upper right', fontsize=9)
ax.set_xlim(-0.02, 0.85); ax.set_ylim(0, 100)

plt.tight_layout()
fig7.savefig('/mnt/user-data/outputs/fig7_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved fig7_overlay.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 8 OVERLAY — [F] and ne vs Power
# ═══════════════════════════════════════════════════════════════

fig8, ax1 = plt.subplots(figsize=(9, 6))

# Model [F]
F_mod = [r['n_F']*1e-6/1e13 for r in res_P]  # 10¹³ cm⁻³
ne_mod_8 = [r['ne']*1e-6/1e9 for r in res_P]  # 10⁹ cm⁻³

ax1.plot(P_mod, F_mod, '-', color='#d73027', lw=2.5, label='[F] — this model')
ax1.plot(fig8_F_calc[:,0], fig8_F_calc[:,1], 's', color='#d73027', ms=9,
         markerfacecolor='#d73027', label='[F] — Lallement calc.')
ax1.plot(fig8_F_exp[:,0], fig8_F_exp[:,1], 'o', color='#fc8d59', ms=8,
         markerfacecolor='white', markeredgewidth=1.5, label='[F] — Lallement exp.')
ax1.set_xlabel('Coupled power (W)', fontsize=13)
ax1.set_ylabel(r'[F] ($10^{13}$ cm$^{-3}$)', color='#d73027', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#d73027')
ax1.set_xlim(0, 2100); ax1.set_ylim(0, 16)

ax2 = ax1.twinx()
ax2.plot(P_mod, ne_mod_8, '-', color='#4575b4', lw=2.5, label=r'$n_e$ — this model')
ax2.plot(fig8_ne_calc[:,0], fig8_ne_calc[:,1], 's', color='#4575b4', ms=9,
         markerfacecolor='#4575b4', label=r'$n_e$ — Lallement calc.')
ax2.set_ylabel(r'$n_e$ ($10^{9}$ cm$^{-3}$)', color='#4575b4', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#4575b4')
ax2.set_ylim(0, 12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
ax1.set_title('[F] and $n_e$ vs power — SF$_6$, 10 mTorr\n'
              'Overlay with Lallement et al. (digitized)', fontsize=13)

plt.tight_layout()
fig8.savefig('/mnt/user-data/outputs/fig8_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved fig8_overlay.png")

# ═══════════════════════════════════════════════════════════════
# PRINT QUANTITATIVE COMPARISON
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("QUANTITATIVE COMPARISON AT KEY POINTS")
print("="*70)

# Find model values at 1500W
idx_1500 = np.argmin(np.abs(np.array(P_mod) - 1500))

print(f"\nAt 1500 W, 10 mTorr, pure SF6:")
print(f"  {'Quantity':<20} {'Model':>12} {'Lall. calc':>12} {'Lall. exp':>12} {'Model/calc':>12}")
print(f"  {'-'*68}")

# ne
ne_m = res_P[idx_1500]['ne']*1e-6/1e9
idx_lc = np.argmin(np.abs(fig5a_ne_calc[:,0] - 1500))
idx_le = np.argmin(np.abs(fig5a_ne_exp[:,0] - 1500))
print(f"  {'ne (10⁹ cm⁻³)':<20} {ne_m:>12.2f} {fig5a_ne_calc[idx_lc,1]:>12.2f} {fig5a_ne_exp[idx_le,1]:>12.2f} {ne_m/fig5a_ne_calc[idx_lc,1]:>12.2f}×")

# Te
Te_m = res_P[idx_1500]['Te']
idx_lc = np.argmin(np.abs(fig5b_Te_calc[:,0] - 1500))
idx_le = np.argmin(np.abs(fig5b_Te_exp[:,0] - 1500))
print(f"  {'Te (eV)':<20} {Te_m:>12.2f} {fig5b_Te_calc[idx_lc,1]:>12.2f} {fig5b_Te_exp[idx_le,1]:>12.2f} {Te_m/fig5b_Te_calc[idx_lc,1]:>12.2f}×")

# [F]
F_m = res_P[idx_1500]['n_F']*1e-6/1e13
idx_lc = np.argmin(np.abs(fig8_F_calc[:,0] - 1500))
idx_le = np.argmin(np.abs(fig8_F_exp[:,0] - 1500))
print(f"  {'[F] (10¹³ cm⁻³)':<20} {F_m:>12.2f} {fig8_F_calc[idx_lc,1]:>12.2f} {fig8_F_exp[idx_le,1]:>12.2f} {F_m/fig8_F_calc[idx_lc,1]:>12.2f}×")

# Alpha at key Ar fractions (10 mTorr)
print(f"\nAlpha at 10 mTorr, 1500 W:")
print(f"  {'Ar frac':<10} {'Model':>10} {'Lallement':>10} {'Ratio':>10}")
for fAr_target in [0.0, 0.3, 0.5, 0.7]:
    # Model
    idx_m = np.argmin(np.abs(np.array([r['frac_Ar'] for r in res_alpha[10]]) - fAr_target))
    al_m = res_alpha[10][idx_m]['alpha']
    # Paper
    idx_p = np.argmin(np.abs(fig7_10mT[:,0] - fAr_target))
    al_p = fig7_10mT[idx_p, 1]
    print(f"  {fAr_target:<10.1f} {al_m:>10.1f} {al_p:>10.1f} {al_m/al_p:>10.2f}×")

print("\n✓ All overlay figures saved.")
