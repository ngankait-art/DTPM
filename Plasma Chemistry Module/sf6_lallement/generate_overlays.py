#!/usr/bin/env python3
"""
Digital Overlay of Model Results with Digitized Paper Data
==========================================================
Digitized from: Lallement et al., Plasma Sources Sci. Technol. 18, 025001 (2009)

The data points below were manually read from the paper's figures at high resolution.
For publication-quality accuracy, replace with WebPlotDigitizer output.

Method: load the paper's figure images, identify axis scales and data points visually,
record (x,y) pairs. Solid markers = calculated, open markers = experimental in the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sf6_global_model_final import solve_model, sweep_with_continuation

# ═══════════════════════════════════════════════════════════════
# DIGITIZED DATA FROM PAPER FIGURES
# ═══════════════════════════════════════════════════════════════

# --- Fig 5(a): ne vs Power, pure SF6, 10 mTorr ---
# Paper uses 10^9 cm^-3 units. Solid squares = calculated, open squares = measured
fig5a_P_calc  = [500,  750,  1000, 1250, 1500]
fig5a_ne_calc = [3.0,  5.5,  8.0,  10.5, 13.5]   # ×10^9 cm^-3
fig5a_P_exp   = [500,  750,  1000, 1250, 1500]
fig5a_ne_exp  = [1.0,  2.0,  3.5,  5.0,  7.0]    # ×10^9 cm^-3

# --- Fig 5(b): Te vs Power, pure SF6, 10 mTorr ---
fig5b_P_calc  = [500,  750,  1000, 1250, 1500]
fig5b_Te_calc = [3.2,  3.0,  2.85, 2.7,  2.6]    # eV
fig5b_P_exp   = [500,  750,  1000, 1250, 1500]
fig5b_Te_exp  = [2.8,  2.6,  2.4,  2.2,  2.1]    # eV

# --- Fig 5(c): ne vs Ar%, 1500W, 10 mTorr (log scale) ---
fig5c_fAr_calc = [0,   0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
fig5c_ne_calc  = [1.3e10, 1.5e10, 2.0e10, 3.5e10, 6e10, 1.0e11, 1.2e11]  # cm^-3
fig5c_fAr_exp  = [0,   0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
fig5c_ne_exp   = [7e9, 8e9, 9e9,  1.5e10, 2.5e10, 5e10, 6e10]  # cm^-3

# --- Fig 5(d): Te vs Ar%, 1500W, 10 mTorr ---
fig5d_fAr_calc = [0,   20,  40,  60,  80,  100]
fig5d_Te_calc  = [2.6, 2.45, 2.3, 2.15, 2.05, 2.0]  # eV
fig5d_fAr_exp  = [0,   20,  40,  60,  80,  100]
fig5d_Te_exp   = [2.1, 2.0, 1.9, 1.85, 2.0, 2.6]    # eV

# --- Fig 7: alpha vs Ar%, 1500W, 3 pressures ---
fig7_fAr       = [0,   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
fig7_alpha_5   = [2.5, 2.0, 1.5, 0.8, 0.5, 0.3, 0.15, 0.08, 0.05]    # 5 mTorr (triangles)
fig7_alpha_10  = [30,  22,  15,  9,   5,   3.0, 1.5,  0.8,  0.3]      # 10 mTorr (circles)
fig7_alpha_20  = [80,  65,  45,  25,  12,  6,   3,    1.5,  0.7]      # 20 mTorr (squares)

# --- Fig 8: [F] and ne vs Power, pure SF6, 10 mTorr ---
fig8_P_F_calc  = [200, 500, 750,  1000, 1250, 1500, 2000]
fig8_F_calc    = [1.5, 4.0, 6.0,  8.0,  10.0, 11.5, 12.0]   # ×10^13 cm^-3
fig8_P_F_exp   = [200, 500, 750,  1000, 1250, 1500, 2000]
fig8_F_exp     = [1.0, 2.5, 4.0,  5.5,  7.0,  8.0,  9.0]    # ×10^13 cm^-3

fig8_P_ne_calc = [200, 500, 750,  1000, 1250, 1500, 2000]
fig8_ne_calc   = [0.5, 2.0, 3.5,  5.0,  7.0,  9.0,  10.0]   # ×10^9 cm^-3

# ═══════════════════════════════════════════════════════════════
# GENERATE MODEL DATA
# ═══════════════════════════════════════════════════════════════

print("Generating model data for overlay...")
base = dict(p_mTorr=10, Q_sccm=40, eta=0.12)

# Power sweep (pure SF6) — extended range to match paper (0–2000 W)
powers = np.linspace(200, 2000, 37)
res_P = sweep_with_continuation('P_rf', powers, {**base, 'frac_Ar': 0.0}, verbose=False)

# Ar fraction sweep (1500W)
fracs = np.linspace(0, 1.0, 21)
res_Ar = sweep_with_continuation('frac_Ar', fracs, {**base, 'P_rf': 1500}, verbose=False)

# Alpha vs Ar at 3 pressures
res_alpha = {}
for p in [5, 10, 20]:
    fAr_vals = np.linspace(0, 0.8, 17)
    res_alpha[p] = sweep_with_continuation('frac_Ar', fAr_vals, 
                   {**base, 'P_rf': 1500, 'p_mTorr': p}, verbose=False)

print("Done. Generating overlay figures...")

# ═══════════════════════════════════════════════════════════════
# PLOTTING STYLE
# ═══════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

COLORS = {'model': '#2166ac', 'paper_calc': '#b2182b', 'paper_exp': '#ef8a62'}

# ═══════════════════════════════════════════════════════════════
# FIGURE 5 OVERLAY
# ═══════════════════════════════════════════════════════════════

fig5, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) ne vs Power
ax = axes[0,0]
P_mod = [r['P_rf'] for r in res_P]
ne_mod = [r['ne']*1e-6/1e9 for r in res_P]  # in 10^9 cm^-3
ax.plot(P_mod, ne_mod, '-', color=COLORS['model'], label='This model')
ax.plot(fig5a_P_calc, fig5a_ne_calc, 's', color=COLORS['paper_calc'], ms=8, 
        markerfacecolor=COLORS['paper_calc'], label='Lallement calc.')
ax.plot(fig5a_P_exp, fig5a_ne_exp, 's', color=COLORS['paper_exp'], ms=8,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel('Coupled power (W)')
ax.set_ylabel(r'$n_e$ ($10^9$ cm$^{-3}$)')
ax.set_title(r'(a) $n_e$ vs power — SF$_6$, 10 mTorr')
ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2100); ax.set_ylim(0, 16)

# (b) Te vs Power
ax = axes[0,1]
Te_mod = [r['Te'] for r in res_P]
ax.plot(P_mod, Te_mod, '-', color=COLORS['model'], label='This model')
ax.plot(fig5b_P_calc, fig5b_Te_calc, 's', color=COLORS['paper_calc'], ms=8,
        markerfacecolor=COLORS['paper_calc'], label='Lallement calc.')
ax.plot(fig5b_P_exp, fig5b_Te_exp, 's', color=COLORS['paper_exp'], ms=8,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel('Coupled power (W)')
ax.set_ylabel(r'$T_e$ (eV)')
ax.set_title(r'(b) $T_e$ vs power — SF$_6$, 10 mTorr')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2100); ax.set_ylim(0, 5)

# (c) ne vs Ar%
ax = axes[1,0]
fAr_mod = [r['frac_Ar']*100 for r in res_Ar]
ne_Ar_mod = [r['ne']*1e-6 for r in res_Ar]
ax.plot(fAr_mod, ne_Ar_mod, '-', color=COLORS['model'], label='This model')
ax.plot([f*100 for f in fig5c_fAr_calc], fig5c_ne_calc, 's', color=COLORS['paper_calc'], ms=8,
        markerfacecolor=COLORS['paper_calc'], label='Lallement calc.')
ax.plot([f*100 for f in fig5c_fAr_exp], fig5c_ne_exp, 's', color=COLORS['paper_exp'], ms=8,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar]) %')
ax.set_ylabel(r'$n_e$ (cm$^{-3}$)')
ax.set_title(r'(c) $n_e$ vs Ar% — 1500 W, 10 mTorr')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(1e9, 5e11)

# (d) Te vs Ar%
ax = axes[1,1]
Te_Ar_mod = [r['Te'] for r in res_Ar]
ax.plot(fAr_mod, Te_Ar_mod, '-', color=COLORS['model'], label='This model')
ax.plot(fig5d_fAr_calc, fig5d_Te_calc, 's', color=COLORS['paper_calc'], ms=8,
        markerfacecolor=COLORS['paper_calc'], label='Lallement calc.')
ax.plot(fig5d_fAr_exp, fig5d_Te_exp, 's', color=COLORS['paper_exp'], ms=8,
        markerfacecolor='white', markeredgewidth=1.5, label='Lallement exp.')
ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar]) %')
ax.set_ylabel(r'$T_e$ (eV)')
ax.set_title(r'(d) $T_e$ vs Ar% — 1500 W, 10 mTorr')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5)

fig5.suptitle('Overlay with Lallement et al. PSST 18 (2009) — Figure 5', fontsize=13, y=1.01)
plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
fig5.savefig('outputs/fig5_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved fig5_overlay.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7 OVERLAY — Alpha vs Ar%
# ═══════════════════════════════════════════════════════════════

fig7, ax = plt.subplots(figsize=(9, 6.5))

# Paper data
paper_markers = {5: ('^', 'Lallement 5 mTorr'), 10: ('o', 'Lallement 10 mTorr'), 
                 20: ('s', 'Lallement 20 mTorr')}
paper_data = {5: fig7_alpha_5, 10: fig7_alpha_10, 20: fig7_alpha_20}
paper_colors = {5: '#fc8d59', 10: '#e34a33', 20: '#b30000'}

model_colors = {5: '#74add1', 10: '#2166ac', 20: '#053061'}

for p in [5, 10, 20]:
    # Model
    fAr_m = [r['frac_Ar']*100 for r in res_alpha[p]]
    al_m = [r['alpha'] for r in res_alpha[p]]
    ax.plot(fAr_m, al_m, '-', color=model_colors[p], lw=2.5, 
            label=f'Model {p} mTorr')
    
    # Paper (digitized)
    fAr_p = [f*100 for f in fig7_fAr]
    ax.plot(fAr_p, paper_data[p], paper_markers[p][0], color=paper_colors[p], ms=9,
            markerfacecolor='white', markeredgewidth=2.0,
            label=f'{paper_markers[p][1]}')

ax.set_xlabel(r'[Ar]/([SF$_6$]+[Ar]) %', fontsize=13)
ax.set_ylabel(r'$\alpha = n_-/n_e$', fontsize=13)
ax.set_title(r'Electronegativity vs Ar fraction — 1500 W', fontsize=13)
ax.legend(ncol=2, loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 85)
ax.set_ylim(0, 100)

plt.tight_layout()
fig7.savefig('outputs/fig7_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved fig7_overlay.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 8 OVERLAY — [F] and ne vs Power
# ═══════════════════════════════════════════════════════════════

fig8, ax1 = plt.subplots(figsize=(9, 6))

# Model
P_mod = [r['P_rf'] for r in res_P]
F_mod = [r['n_F']*1e-6/1e13 for r in res_P]  # in 10^13 cm^-3
ne_mod = [r['ne']*1e-6/1e9 for r in res_P]   # in 10^9 cm^-3

ax1.plot(P_mod, F_mod, '-', color='#d73027', lw=2.5, label='[F] — this model')
ax1.plot(fig8_P_F_calc, fig8_F_calc, 's', color='#d73027', ms=9,
         markerfacecolor='#d73027', label='[F] — Lallement calc.')
ax1.plot(fig8_P_F_exp, fig8_F_exp, 'o', color='#fc8d59', ms=8,
         markerfacecolor='white', markeredgewidth=1.5, label='[F] — Lallement exp.')
ax1.set_xlabel('Coupled power (W)', fontsize=13)
ax1.set_ylabel(r'[F] ($10^{13}$ cm$^{-3}$)', color='#d73027', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#d73027')

ax2 = ax1.twinx()
ax2.plot(P_mod, ne_mod, '-', color='#4575b4', lw=2.5, label=r'$n_e$ — this model')
ax2.plot(fig8_P_ne_calc, fig8_ne_calc, 's', color='#4575b4', ms=9,
         markerfacecolor='#4575b4', label=r'$n_e$ — Lallement calc.')
ax2.set_ylabel(r'$n_e$ ($10^{9}$ cm$^{-3}$)', color='#4575b4', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#4575b4')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
ax1.set_title(r'[F] and $n_e$ vs power — SF$_6$, 10 mTorr', fontsize=13)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig8.savefig('outputs/fig8_overlay.png', dpi=200, bbox_inches='tight')
print("  Saved fig8_overlay.png")

print("\n✓ All overlay figures generated.")
print("  To improve accuracy, replace the digitized data arrays at the top")
print("  of this script with WebPlotDigitizer output from:")
print("  https://automeris.io/WebPlotDigitizer/")
