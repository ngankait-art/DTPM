"""
SF6 Global Model — Digital Overlay Plots
Paper data (digitized) vs Python model on the same axes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, csv, os

OUT = 'output_overlay'
os.makedirs(OUT, exist_ok=True)

# Load digitized paper data
with open('paper_digitized.json') as f:
    paper = json.load(f)

# Load model CSV data
def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})
    return rows

model_power = load_csv('output/power_sweep_results.csv')
model_poff  = load_csv('output/pressure_sweep_results.csv')

def mP(key): return [r[key] for r in model_power]
def mQ(key): return [r[key] for r in model_poff]

# Style constants
PAPER_STYLE = dict(ls='--', lw=2.0, alpha=0.85)
MODEL_STYLE = dict(ls='-', lw=1.8, marker='o', ms=4)
EXP_STYLE   = dict(ls='none', ms=8, mew=1.5, alpha=0.9)

# =====================================================================
# OVERLAY 1: Pressure rise & F density vs Power
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Pressure rise
ax1.plot(paper['fig5a']['model_power'], paper['fig5a']['dp_model'],
         color='black', label='Kokkoris model', **PAPER_STYLE)
ax1.plot(paper['fig5a']['power'], paper['fig5a']['dp_exp'],
         marker='s', color='black', label='Kokkoris exp.', **EXP_STYLE)
ax1.plot(mP('Power_W'), mP('dp_Pa'),
         color='tab:blue', label='This work', **MODEL_STYLE)
ax1.set_xlabel('Power (W)', fontsize=12)
ax1.set_ylabel('Pressure rise (Pa)', fontsize=12)
ax1.set_title('Pressure rise vs Power')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 3600)

# F density
ax2.plot(paper['fig5b']['model_power'], paper['fig5b']['nF_model'],
         color='black', label='Kokkoris model', **PAPER_STYLE)
ax2.plot(paper['fig5b']['power'], paper['fig5b']['nF_056'],
         marker='s', color='black', label='Kokkoris exp. ($K_F$=0.56)', **EXP_STYLE)
ax2.plot(paper['fig5b']['power'], paper['fig5b']['nF_076'],
         marker='o', mfc='none', color='black', label='Kokkoris exp. ($K_F$=0.76)', **EXP_STYLE)
ax2.plot(mP('Power_W'), mP('nF_m3'),
         color='tab:red', label='This work', **MODEL_STYLE)
ax2.set_xlabel('Power (W)', fontsize=12)
ax2.set_ylabel('F density (m$^{-3}$)', fontsize=12)
ax2.set_title('F density vs Power')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 3600)

fig.suptitle('Overlay: Pressure Rise & F Density vs Power ($p_{OFF}$=0.921 Pa)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_dp_F_vs_power.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_dp_F_vs_power.png')

# =====================================================================
# OVERLAY 2: Pressure rise & F density vs pOFF
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(paper['fig6a']['model_poff'], paper['fig6a']['dp_model'],
         color='black', label='Kokkoris model', **PAPER_STYLE)
ax1.plot(paper['fig6a']['poff'], paper['fig6a']['dp_exp'],
         marker='s', color='black', label='Kokkoris exp.', **EXP_STYLE)
ax1.plot(mQ('pOFF_Pa'), mQ('dp_Pa'),
         color='tab:blue', label='This work', **MODEL_STYLE)
ax1.set_xlabel('$p_{OFF}$ (Pa)', fontsize=12)
ax1.set_ylabel('Pressure rise (Pa)', fontsize=12)
ax1.set_title('Pressure rise vs $p_{OFF}$')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

ax2.plot(paper['fig6b']['model_poff'], paper['fig6b']['nF_model'],
         color='black', label='Kokkoris model', **PAPER_STYLE)
ax2.plot(paper['fig6b']['poff'], paper['fig6b']['nF_056'],
         marker='s', color='black', label='Kokkoris exp. ($K_F$=0.56)', **EXP_STYLE)
ax2.plot(paper['fig6b']['poff'], paper['fig6b']['nF_076'],
         marker='o', mfc='none', color='black', label='Kokkoris exp. ($K_F$=0.76)', **EXP_STYLE)
ax2.plot(mQ('pOFF_Pa'), mQ('nF_m3'),
         color='tab:red', label='This work', **MODEL_STYLE)
ax2.set_xlabel('$p_{OFF}$ (Pa)', fontsize=12)
ax2.set_ylabel('F density (m$^{-3}$)', fontsize=12)
ax2.set_title('F density vs $p_{OFF}$')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

fig.suptitle('Overlay: Pressure Rise & F Density vs $p_{OFF}$ (P=2000 W)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_dp_F_vs_poff.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_dp_F_vs_poff.png')

# =====================================================================
# OVERLAY 3: Neutral species vs Power
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 7))

specs = [
    ('nSF6', 'nSF6_m3', 'SF$_6$', 'tab:blue'),
    ('nF',   'nF_m3',   'F',       'tab:brown'),
    ('nF2',  'nF2_m3',  'F$_2$',   'tab:purple'),
    ('nSF4', 'nSF4_m3', 'SF$_4$',  'tab:red'),
    ('nSF5', 'nSF5_m3', 'SF$_5$',  'tab:green'),
    ('nSF3', 'nSF3_m3', 'SF$_3$',  'tab:orange'),
]

for pk, mk, label, color in specs:
    ax.semilogy(paper['fig7']['power'], paper['fig7'][pk],
                color=color, ls='--', lw=2.0, alpha=0.7)
    ax.semilogy(mP('Power_W'), mP(mk),
                color=color, ls='-', marker='o', ms=3, lw=1.5,
                label=label)

# Add custom legend entries for line styles
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0],[0], color='gray', ls='--', lw=2.0))
labels.append('Kokkoris (dashed)')
handles.append(Line2D([0],[0], color='gray', ls='-', marker='o', ms=3, lw=1.5))
labels.append('This work (solid)')
ax.legend(handles, labels, fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.35))

ax.set_xlabel('Power (W)', fontsize=13)
ax.set_ylabel('Number density (m$^{-3}$)', fontsize=13)
ax.set_title('Overlay: Neutral Species vs Power ($p_{OFF}$=0.921 Pa)', fontsize=14)
ax.set_ylim(1e17, 3e20)
ax.set_xlim(0, 3600)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_neutrals_vs_power.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_neutrals_vs_power.png')

# =====================================================================
# OVERLAY 4: Charged species + electrons vs Power
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Paper
ax.semilogy(paper['fig7']['power'], paper['fig7']['nSF5p'],
            color='tab:blue', ls='--', lw=2.0, alpha=0.7)
ax.semilogy(paper['fig7']['power'], paper['fig7']['nFm'],
            color='tab:purple', ls='--', lw=2.0, alpha=0.7)
ax.semilogy(paper['fig7']['power'], paper['fig7']['ne'],
            color='black', ls='--', lw=2.0, alpha=0.7)

# Model
ax.semilogy(mP('Power_W'), mP('nSF5p_m3'),
            color='tab:blue', ls='-', marker='o', ms=3, lw=1.5, label='SF$_5^+$')
ax.semilogy(mP('Power_W'), mP('nFm_m3'),
            color='tab:purple', ls='-', marker='D', ms=3, lw=1.5, label='F$^-$')
ax.semilogy(mP('Power_W'), mP('ne_m3'),
            color='black', ls='-', marker='x', ms=4, lw=1.5, label='e$^-$')
ax.semilogy(mP('Power_W'), mP('nSF4p_m3'),
            color='tab:green', ls='-', marker='s', ms=3, lw=1.5, label='SF$_4^+$')
ax.semilogy(mP('Power_W'), mP('nSF3p_m3'),
            color='tab:red', ls='-', marker='^', ms=3, lw=1.5, label='SF$_3^+$')

handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0],[0], color='gray', ls='--', lw=2.0))
labels.append('Kokkoris (dashed)')
ax.legend(handles, labels, fontsize=10, loc='lower right')

ax.set_xlabel('Power (W)', fontsize=13)
ax.set_ylabel('Number density (m$^{-3}$)', fontsize=13)
ax.set_title('Overlay: Charged Species vs Power ($p_{OFF}$=0.921 Pa)', fontsize=14)
ax.set_xlim(0, 3600)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_charged_vs_power.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_charged_vs_power.png')

# =====================================================================
# OVERLAY 5: Alpha and Te vs Power
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# Alpha
ax1.plot(paper['fig7']['power'], paper['fig7']['alpha'],
         color='tab:blue', ls='--', lw=2.0, alpha=0.7, label='$\\alpha$ Kokkoris')
ax1.plot(mP('Power_W'), mP('alpha'),
         color='tab:blue', ls='-', marker='o', ms=4, lw=1.5, label='$\\alpha$ This work')
ax1.set_xlabel('Power (W)', fontsize=13)
ax1.set_ylabel('$\\alpha$ = n$^-$/n$_e$', fontsize=13, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xlim(0, 3600)

# Te on twin axis
ax2 = ax1.twinx()
ax2.plot(paper['fig7']['power'], paper['fig7']['Te'],
         color='tab:red', ls='--', lw=2.0, alpha=0.7, label='$T_e$ Kokkoris')
ax2.plot(mP('Power_W'), mP('Te_eV'),
         color='tab:red', ls='-', marker='s', ms=4, lw=1.5, label='$T_e$ This work')
ax2.set_ylabel('$T_e$ (eV)', fontsize=13, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=10, loc='center right')

ax1.set_title('Overlay: Electronegativity & $T_e$ vs Power ($p_{OFF}$=0.921 Pa)', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_alpha_Te_vs_power.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_alpha_Te_vs_power.png')

# =====================================================================
# OVERLAY 6: Neutral species vs pOFF
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 7))

for pk, mk, label, color in specs:
    ax.semilogy(paper['fig8']['poff'], paper['fig8'][pk],
                color=color, ls='--', lw=2.0, alpha=0.7)
    ax.semilogy(mQ('pOFF_Pa'), mQ(mk),
                color=color, ls='-', marker='o', ms=3, lw=1.5,
                label=label)

handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0],[0], color='gray', ls='--', lw=2.0))
labels.append('Kokkoris (dashed)')
handles.append(Line2D([0],[0], color='gray', ls='-', marker='o', ms=3, lw=1.5))
labels.append('This work (solid)')
ax.legend(handles, labels, fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.35))

ax.set_xlabel('$p_{OFF}$ (Pa)', fontsize=13)
ax.set_ylabel('Number density (m$^{-3}$)', fontsize=13)
ax.set_title('Overlay: Neutral Species vs $p_{OFF}$ (P=2000 W)', fontsize=14)
ax.set_ylim(1e17, 1e21)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_neutrals_vs_poff.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_neutrals_vs_poff.png')

# =====================================================================
# OVERLAY 7: Alpha and Te vs pOFF
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(paper['fig8']['poff'], paper['fig8']['alpha'],
         color='tab:blue', ls='--', lw=2.0, alpha=0.7, label='$\\alpha$ Kokkoris')
ax1.plot(mQ('pOFF_Pa'), mQ('alpha'),
         color='tab:blue', ls='-', marker='o', ms=4, lw=1.5, label='$\\alpha$ This work')
ax1.set_xlabel('$p_{OFF}$ (Pa)', fontsize=13)
ax1.set_ylabel('$\\alpha$ = n$^-$/n$_e$', fontsize=13, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(paper['fig8']['poff'], paper['fig8']['Te'],
         color='tab:red', ls='--', lw=2.0, alpha=0.7, label='$T_e$ Kokkoris')
ax2.plot(mQ('pOFF_Pa'), mQ('Te_eV'),
         color='tab:red', ls='-', marker='s', ms=4, lw=1.5, label='$T_e$ This work')
ax2.set_ylabel('$T_e$ (eV)', fontsize=13, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=10, loc='center left')

ax1.set_title('Overlay: Electronegativity & $T_e$ vs $p_{OFF}$ (P=2000 W)', fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay_alpha_Te_vs_poff.png'), dpi=200, bbox_inches='tight')
plt.close()
print('  Saved overlay_alpha_Te_vs_poff.png')

print(f'\nAll overlay plots saved to {OUT}/')
