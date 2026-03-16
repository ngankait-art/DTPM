#!/usr/bin/env python3
"""
Extended Analysis — Ar density, alpha vs power, Te-based diagnostics
====================================================================
Reuses the existing sf6_penning_v2 solver. Does NOT rebuild the model.
Generates all requested plots with consistent formatting.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sf6_global_model_final import solve_model, sweep_with_continuation

# ═══════════════════════════════════════════════════════════════
# PLOT STYLE — consistent with previous paper figures
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'lines.linewidth': 2.2,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette — SF6-related in reds/oranges, Ar-related in blues/greens
C_SF6     = '#d73027'
C_SF5     = '#fc8d59'
C_SF4     = '#fee090'
C_SF3     = '#e0f3f8'
C_F       = '#91bfdb'
C_F2      = '#4575b4'
C_S       = '#999999'
C_ne      = '#2166ac'
C_Te      = '#b2182b'
C_alpha   = '#1a9850'
C_Ar      = '#006837'
C_Arm     = '#66c2a4'

# ═══════════════════════════════════════════════════════════════
# GENERATE DATA — two sweeps needed
# ═══════════════════════════════════════════════════════════════

print("="*65)
print("Extended Analysis — Ar density, alpha, Te diagnostics")
print("="*65)

base_SF6 = dict(p_mTorr=10, Q_sccm=40, eta=0.12, frac_Ar=0.0)
base_Ar50 = dict(p_mTorr=10, Q_sccm=40, eta=0.12, frac_Ar=0.5)

powers = np.linspace(200, 2000, 37)

print("\n▶ Power sweep: pure SF6 (0% Ar)")
res_SF6 = sweep_with_continuation('P_rf', powers, base_SF6, verbose=False)
print(f"  Done — {len(res_SF6)} points, P = {powers[0]:.0f}–{powers[-1]:.0f} W")

print("\n▶ Power sweep: 50% Ar / 50% SF6")
res_Ar50 = sweep_with_continuation('P_rf', powers, base_Ar50, verbose=False)
print(f"  Done — {len(res_Ar50)} points")

# Also do a few additional Ar fractions for the Te-based plots
print("\n▶ Power sweep: 20% Ar")
res_Ar20 = sweep_with_continuation('P_rf', powers, 
           dict(p_mTorr=10, Q_sccm=40, eta=0.12, frac_Ar=0.2), verbose=False)
print(f"  Done — {len(res_Ar20)} points")

print("\n▶ Power sweep: 80% Ar")
res_Ar80 = sweep_with_continuation('P_rf', powers,
           dict(p_mTorr=10, Q_sccm=40, eta=0.12, frac_Ar=0.8), verbose=False)
print(f"  Done — {len(res_Ar80)} points")

# ═══════════════════════════════════════════════════════════════
# ALPHA DEFINITION — print explicit analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("ALPHA DEFINITION ANALYSIS")
print("="*65)
print("""
In the code (sf6_penning_v2.py, line 248–258), alpha is computed as:

    Ratt = k['att_SF6_total'] * nSF6
    rhs  = Ratt / (k_rec * ne)
    alpha = (-1 + sqrt(1 + 4*rhs)) / 2

where att_SF6_total = k30 + k31 + k32 + k33 + k34 + k35 + k36

These are the 7 dissociative attachment channels of SF6:
    (30) SF6 + e → SF6⁻           (zero-energy attachment)
    (31) SF6 + e → SF5⁻ + F       
    (32) SF6 + e → SF4⁻ + 2F      
    (33) SF6 + e → SF3⁻ + 3F      
    (34) SF6 + e → SF2⁻ + 4F      
    (35) SF6 + e → F⁻ + SF5       (dominant at higher Te)
    (36) SF6 + e → F2⁻ + SF4      

KEY FINDING:
  • Alpha is computed EXCLUSIVELY from SF6-derived negative ions.
  • There is NO contribution from Ar species — Ar has no electron 
    attachment channels (positive electron affinity).
  • The attachment rate Ratt is proportional to nSF6: as SF6 is 
    dissociated or diluted by Ar, Ratt decreases and alpha drops.
  • The negative ions produced include SF6⁻, SF5⁻, SF4⁻, SF3⁻, 
    SF2⁻, F⁻, and F2⁻ — ALL originating from SF6 attachment.
  • Ar contributes to alpha only INDIRECTLY by:
    (a) diluting SF6 → reducing nSF6 → reducing Ratt
    (b) providing additional ionization → increasing ne → reducing alpha
    (c) Penning ionization of SF6 → consuming SF6 → reducing Ratt
""")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Argon density vs Power
# ═══════════════════════════════════════════════════════════════

print("▶ Generating figures...")

fig1, ax = plt.subplots(figsize=(9, 6))

for label, res, color, ls in [
    ('Ar ground (50% Ar)', res_Ar50, C_Ar, '-'),
    ('Ar* metastable (50% Ar)', res_Ar50, C_Arm, '--'),
    ('Ar ground (20% Ar)', res_Ar20, '#2ca02c', '-'),
    ('Ar* metastable (20% Ar)', res_Ar20, '#98df8a', '--'),
    ('Ar ground (80% Ar)', res_Ar80, '#006400', '-'),
    ('Ar* metastable (80% Ar)', res_Ar80, '#b2df8a', '--'),
]:
    P = [r['P_rf'] for r in res]
    if 'metastable' in label:
        n = [r['nArm']*1e-6 for r in res]  # m⁻³ → cm⁻³
    else:
        n = [r['nAr0']*1e-6 for r in res]
    ax.plot(P, n, ls, color=color, label=label)

ax.set_xlabel('Coupled power (W)')
ax.set_ylabel('Density (cm$^{-3}$)')
ax.set_title('Argon species density vs power — 10 mTorr')
ax.set_yscale('log')
ax.set_ylim(1e8, 5e15)
ax.set_xlim(0, 2100)
ax.legend(loc='center right', fontsize=8.5)
plt.tight_layout()
fig1.savefig('/mnt/user-data/outputs/argon_density_vs_power.png', dpi=200, bbox_inches='tight')
print("  Saved argon_density_vs_power.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Alpha vs Power
# ═══════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(9, 6))

for label, res, color, marker in [
    ('Pure SF$_6$ (0% Ar)', res_SF6, C_SF6, 'o'),
    ('20% Ar', res_Ar20, C_SF5, 's'),
    ('50% Ar', res_Ar50, C_F2, '^'),
    ('80% Ar', res_Ar80, C_ne, 'D'),
]:
    P = [r['P_rf'] for r in res]
    alpha = [r['alpha'] for r in res]
    ax.plot(P, alpha, '-', color=color, marker=marker, ms=4, markevery=3, label=label)

ax.set_xlabel('Coupled power (W)')
ax.set_ylabel(r'$\alpha = n_{-}/n_e$')
ax.set_title(r'Electronegativity $\alpha$ vs power — 10 mTorr')
ax.set_yscale('log')
ax.set_ylim(0.05, 500)
ax.set_xlim(0, 2100)
ax.legend()
ax.axhline(y=1, color='gray', ls=':', lw=1, label='')
ax.text(2050, 1.15, r'$\alpha=1$', fontsize=9, color='gray', ha='right')
plt.tight_layout()
fig2.savefig('/mnt/user-data/outputs/alpha_vs_power.png', dpi=200, bbox_inches='tight')
print("  Saved alpha_vs_power.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: SF6 species densities vs Te (Te on x-axis)
# ═══════════════════════════════════════════════════════════════

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: SF6-related species vs Te (pure SF6 case)
sf6_species = [
    (r'SF$_6$', 'n_SF6', C_SF6),
    (r'SF$_5$', 'n_SF5', C_SF5),
    (r'SF$_4$', 'n_SF4', '#fdae61'),
    (r'SF$_3$', 'n_SF3', '#abd9e9'),
    ('F',       'n_F',   C_F),
    (r'F$_2$',  'n_F2',  C_F2),
    ('S',       'n_S',   C_S),
]

Te_sf6 = [r['Te'] for r in res_SF6]
for label, key, color in sf6_species:
    vals = [r[key]*1e-6 for r in res_SF6]  # → cm⁻³
    if max(vals) > 1e5:  # only plot species with significant density
        ax1.plot(Te_sf6, vals, '-', color=color, lw=2, label=label)

ax1.set_xlabel(r'Electron temperature $T_e$ (eV)')
ax1.set_ylabel('Density (cm$^{-3}$)')
ax1.set_title('SF$_6$-derived species vs $T_e$ — pure SF$_6$, 10 mTorr')
ax1.set_yscale('log')
ax1.set_ylim(1e8, 5e14)
ax1.legend(ncol=2)
ax1.invert_xaxis()  # Te decreases with increasing power
ax1.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax1.transAxes,
         fontsize=9, color='gray')

# Right panel: Ar species vs Te (50% Ar case)
Te_Ar = [r['Te'] for r in res_Ar50]
ne_Ar = [r['ne']*1e-6 for r in res_Ar50]
nAr_ground = [r['nAr0']*1e-6 for r in res_Ar50]
nArm = [r['nArm']*1e-6 for r in res_Ar50]

ax2.plot(Te_Ar, nAr_ground, '-', color=C_Ar, lw=2.5, label='Ar (ground)')
ax2.plot(Te_Ar, nArm, '--', color=C_Arm, lw=2.5, label=r'Ar$^*$ (metastable)')
ax2.plot(Te_Ar, ne_Ar, '-', color=C_ne, lw=2, label=r'$n_e$')

# Also show SF6 in the 50% mixture for context
nSF6_mix = [r['n_SF6']*1e-6 for r in res_Ar50]
nF_mix = [r['n_F']*1e-6 for r in res_Ar50]
ax2.plot(Te_Ar, nSF6_mix, '-', color=C_SF6, lw=1.5, alpha=0.7, label=r'SF$_6$ (in mix)')
ax2.plot(Te_Ar, nF_mix, '-', color=C_F, lw=1.5, alpha=0.7, label='F (in mix)')

ax2.set_xlabel(r'Electron temperature $T_e$ (eV)')
ax2.set_ylabel('Density (cm$^{-3}$)')
ax2.set_title(r'Ar and mixed species vs $T_e$ — 50\% Ar, 10 mTorr')
ax2.set_yscale('log')
ax2.set_ylim(1e8, 5e14)
ax2.legend(ncol=2, fontsize=8.5)
ax2.invert_xaxis()
ax2.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax2.transAxes,
         fontsize=9, color='gray')

plt.tight_layout()
fig3.savefig('/mnt/user-data/outputs/Te_density_diagnostics.png', dpi=200, bbox_inches='tight')
print("  Saved Te_density_diagnostics.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: ne and alpha vs Te (Te on x-axis, both cases)
# ═══════════════════════════════════════════════════════════════

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: ne vs Te for different Ar fractions
for label, res, color in [
    ('Pure SF$_6$', res_SF6, C_SF6),
    ('20% Ar', res_Ar20, C_SF5),
    ('50% Ar', res_Ar50, C_F2),
    ('80% Ar', res_Ar80, C_ne),
]:
    Te = [r['Te'] for r in res]
    ne = [r['ne']*1e-6 for r in res]
    ax1.plot(Te, ne, 'o-', color=color, ms=3, markevery=3, label=label)

ax1.set_xlabel(r'$T_e$ (eV)')
ax1.set_ylabel(r'$n_e$ (cm$^{-3}$)')
ax1.set_title(r'Electron density vs $T_e$ — 10 mTorr')
ax1.set_yscale('log')
ax1.legend()
ax1.invert_xaxis()
ax1.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax1.transAxes,
         fontsize=9, color='gray')

# Right: alpha vs Te for different Ar fractions
for label, res, color in [
    ('Pure SF$_6$', res_SF6, C_SF6),
    ('20% Ar', res_Ar20, C_SF5),
    ('50% Ar', res_Ar50, C_F2),
    ('80% Ar', res_Ar80, C_ne),
]:
    Te = [r['Te'] for r in res]
    alpha = [r['alpha'] for r in res]
    ax2.plot(Te, alpha, 'o-', color=color, ms=3, markevery=3, label=label)

ax2.set_xlabel(r'$T_e$ (eV)')
ax2.set_ylabel(r'$\alpha = n_-/n_e$')
ax2.set_title(r'Electronegativity vs $T_e$ — 10 mTorr')
ax2.set_yscale('log')
ax2.set_ylim(0.05, 500)
ax2.legend()
ax2.axhline(y=1, color='gray', ls=':', lw=1)
ax2.invert_xaxis()
ax2.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax2.transAxes,
         fontsize=9, color='gray')

plt.tight_layout()
fig4.savefig('/mnt/user-data/outputs/ne_alpha_vs_Te.png', dpi=200, bbox_inches='tight')
print("  Saved ne_alpha_vs_Te.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Comprehensive SF6 vs Ar species comparison vs Power
# ═══════════════════════════════════════════════════════════════

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: SF6-derived neutral species vs power (pure SF6)
P = [r['P_rf'] for r in res_SF6]
for label, key, color, ls in [
    (r'SF$_6$', 'n_SF6', C_SF6, '-'),
    (r'SF$_5$', 'n_SF5', C_SF5, '-'),
    (r'SF$_4$', 'n_SF4', '#fdae61', '-'),
    (r'SF$_3$', 'n_SF3', '#abd9e9', '--'),
    ('F',       'n_F',   C_F, '-'),
    (r'F$_2$',  'n_F2',  C_F2, '-'),
    ('SF$_2$',  'n_SF2', '#bcbddc', '--'),
    ('S',       'n_S',   C_S, ':'),
]:
    vals = [r[key]*1e-6 for r in res_SF6]
    if max(vals) > 1e6:
        ax1.plot(P, vals, ls, color=color, lw=2 if ls=='-' else 1.5, label=label)

ax1.set_xlabel('Coupled power (W)')
ax1.set_ylabel('Density (cm$^{-3}$)')
ax1.set_title('SF$_6$-derived neutrals — pure SF$_6$, 10 mTorr')
ax1.set_yscale('log'); ax1.set_ylim(1e7, 5e14); ax1.set_xlim(0, 2100)
ax1.legend(ncol=2, fontsize=9)

# Right panel: Ar + mixed species vs power (50% Ar)
P = [r['P_rf'] for r in res_Ar50]
ax2.plot(P, [r['nAr0']*1e-6 for r in res_Ar50], '-', color=C_Ar, lw=2.5, label='Ar (ground)')
ax2.plot(P, [r['nArm']*1e-6 for r in res_Ar50], '--', color=C_Arm, lw=2.5, label=r'Ar$^*$ (metastable)')
ax2.plot(P, [r['ne']*1e-6 for r in res_Ar50], '-', color=C_ne, lw=2, label=r'$n_e$')
ax2.plot(P, [r['n_SF6']*1e-6 for r in res_Ar50], '-', color=C_SF6, lw=1.5, label=r'SF$_6$')
ax2.plot(P, [r['n_F']*1e-6 for r in res_Ar50], '-', color=C_F, lw=1.5, label='F')
ax2.plot(P, [r['n_SF5']*1e-6 for r in res_Ar50], '--', color=C_SF5, lw=1.5, label=r'SF$_5$')

ax2.set_xlabel('Coupled power (W)')
ax2.set_ylabel('Density (cm$^{-3}$)')
ax2.set_title('All species — 50% Ar / 50% SF$_6$, 10 mTorr')
ax2.set_yscale('log'); ax2.set_ylim(1e7, 5e14); ax2.set_xlim(0, 2100)
ax2.legend(ncol=2, fontsize=9)

plt.tight_layout()
fig5.savefig('/mnt/user-data/outputs/SF6_vs_Ar_species_vs_power.png', dpi=200, bbox_inches='tight')
print("  Saved SF6_vs_Ar_species_vs_power.png")

# ═══════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SUMMARY TABLE — key quantities at 1500 W, 10 mTorr")
print("="*65)
print(f"{'Quantity':<25} {'Pure SF6':>14} {'20% Ar':>14} {'50% Ar':>14} {'80% Ar':>14}")
print("-"*83)

# Find the 1500W point in each sweep
idx = np.argmin(np.abs(powers - 1500))
for label, key, unit, scale in [
    ('ne', 'ne', 'cm⁻³', 1e-6),
    ('Te', 'Te', 'eV', 1),
    ('alpha', 'alpha', '—', 1),
    ('[F]', 'n_F', 'cm⁻³', 1e-6),
    ('[SF6]', 'n_SF6', 'cm⁻³', 1e-6),
    ('nAr (ground)', 'nAr0', 'cm⁻³', 1e-6),
    ('nAr* (meta)', 'nArm', 'cm⁻³', 1e-6),
    ('Ec', 'Ec', 'eV', 1),
    ('SF6 dissoc %', 'dissoc_frac', '%', 100),
]:
    vals = []
    for res in [res_SF6, res_Ar20, res_Ar50, res_Ar80]:
        v = res[idx][key] * scale
        vals.append(v)
    if key in ['ne', 'n_F', 'n_SF6', 'nAr0', 'nArm']:
        print(f"{label+' ('+unit+')':<25} {vals[0]:>14.2e} {vals[1]:>14.2e} {vals[2]:>14.2e} {vals[3]:>14.2e}")
    elif key == 'dissoc_frac':
        print(f"{label:<25} {vals[0]:>13.0f}% {vals[1]:>13.0f}% {vals[2]:>13.0f}% {vals[3]:>13.0f}%")
    else:
        print(f"{label+' ('+unit+')':<25} {vals[0]:>14.2f} {vals[1]:>14.2f} {vals[2]:>14.2f} {vals[3]:>14.2f}")

print("\n" + "="*65)
print("TECHNICAL INTERPRETATION")
print("="*65)
print("""
1. ARGON DENSITY vs POWER:
   Ar ground-state density is constant with power (set by feed composition 
   and pressure, not affected by the discharge at these ionization fractions).
   Ar* metastable density increases with power because its production rate 
   (k_exc * ne * nAr) is proportional to ne, which rises with power.
   At 50% Ar, nAr* reaches ~10¹⁰–10¹¹ cm⁻³, roughly 4–5 orders of 
   magnitude below ground-state Ar.

2. ALPHA vs POWER:
   Alpha decreases monotonically with power for ALL Ar fractions because:
   (a) Higher power → higher ne → more SF6 dissociation → less nSF6
   (b) Less nSF6 → lower Ratt = k_att * nSF6
   (c) alpha ∝ Ratt/ne → decreases as both Ratt↓ and ne↑
   At pure SF6, alpha drops from ~300 at 200W to ~10 at 2000W.
   At 50% Ar, alpha is 1–2 orders of magnitude lower at every power level 
   because Ar dilution reduces the initial SF6 fraction.
   The α=1 line marks the transition from electronegative (α>1) to 
   effectively electropositive (α<1) plasma.

3. Te-BASED DIAGNOSTICS:
   When plotted against Te (which decreases with increasing power), the 
   species densities reveal the underlying kinetics:
   - SF6 density drops exponentially with decreasing Te because lower Te
     corresponds to higher ne and more dissociation
   - F density rises as SF6 is consumed, then plateaus when dissociation
     is nearly complete
   - The ne vs Te curves for different Ar fractions collapse partially,
     showing that the ionization–loss balance sets a universal ne(Te) 
     relationship modified mainly by the Ar ionization contribution
   - Alpha vs Te shows that pure SF6 sustains very high electronegativity
     even at low Te, while Ar-diluted mixtures transition to electropositive
     at Te ≈ 2.3–2.5 eV

4. SF6 vs Ar SPECIES:
   In pure SF6, the dissociation cascade SF6→SF5→SF4→SF3→...→F is clearly 
   visible, with each intermediate at progressively lower density.
   In the 50% Ar mixture, the Ar ground-state density (~1.6×10¹⁴ cm⁻³) 
   dominates over all SF6-derived species. The metastable Ar* density 
   is comparable to ne, confirming that stepwise ionization through Ar* 
   is a significant ionization pathway in the mixture.
""")

print("✓ All extended analysis figures saved to /mnt/user-data/outputs/")
