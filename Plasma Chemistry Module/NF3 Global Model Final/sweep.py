"""
NF3/Ar Global Model - Parametric Sweeps and Comparison with Paper
=================================================================
Reproduces key figures from Huang et al. 2026 (PSST 35, 015019)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver import NF3GlobalModel, ReactorConfig, gas_temperature

# Style
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150
})

def run_sweep(param_name, param_values, base_config_kwargs):
    """Run model for a sweep of one parameter."""
    results = []
    for val in param_values:
        kwargs = base_config_kwargs.copy()
        if param_name == 'power':
            kwargs['P_abs'] = val
        elif param_name == 'pressure':
            kwargs['pressure_mTorr'] = val
        elif param_name == 'nf3_ratio':
            total_flow = kwargs.get('flow_Ar_sccm', 80) + kwargs.get('flow_NF3_sccm', 20)
            kwargs['flow_NF3_sccm'] = val / 100.0 * total_flow
            kwargs['flow_Ar_sccm'] = (1 - val / 100.0) * total_flow
        
        config = ReactorConfig(**kwargs)
        model = NF3GlobalModel(config)
        sol = model.solve(t_end=0.05, n_points=500)
        
        if sol.success:
            y_ss = sol.y[:, -1]
            Te = y_ss[model.idx_Te]
            Tg = gas_temperature(config.P_abs, config.pressure_mTorr)
            
            data = {'param': val, 'Te': Te, 'Tg': Tg}
            for sp in model.species_idx:
                data[sp] = max(y_ss[model.species_idx[sp]], 0)
            
            # Derived quantities
            n_neg = data.get('F-', 0) + data.get('F2-', 0)
            ne = data.get('e', 1e10)
            data['alpha'] = n_neg / max(ne, 1e6)
            
            results.append(data)
            print(f"  {param_name}={val:.1f}: ne={ne:.2e}, Te={Te:.2f} eV, "
                  f"nF={data.get('F', 0):.2e}")
    
    return results


def plot_results():
    """Generate comparison plots."""
    base = {
        'R': 0.21, 'L': 0.06, 'P_abs': 600.0,
        'pressure_mTorr': 30.0, 'flow_Ar_sccm': 80.0, 'flow_NF3_sccm': 20.0,
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('NF3/Ar ICP Global Model — Reproducing Huang et al. 2026', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # 1. Electron density vs Power (cf. Fig 10a)
    # =========================================================================
    print("\n=== Sweep: Power ===")
    powers = np.array([500, 600, 700, 800, 900, 1000])
    res_power = run_sweep('power', powers, base)
    
    ax = axes[0, 0]
    ne_vals = [r['e'] / 1e16 for r in res_power]
    ax.plot(powers, ne_vals, 'bo-', linewidth=2, markersize=6, label='Model (20% NF3)')
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Electron density (×10¹⁶ m⁻³)')
    ax.set_title('(a) Electron density vs Power\ncf. Fig. 10a')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 2. Electron density vs Pressure (cf. Fig 10b)
    # =========================================================================
    print("\n=== Sweep: Pressure ===")
    pressures = np.array([10, 20, 30, 50, 70, 100])
    res_press = run_sweep('pressure', pressures, base)
    
    ax = axes[0, 1]
    ne_vals = [r['e'] / 1e16 for r in res_press]
    ax.plot(pressures, ne_vals, 'rs-', linewidth=2, markersize=6, label='Model (20% NF3)')
    ax.set_xlabel('Pressure (mTorr)')
    ax.set_ylabel('Electron density (×10¹⁶ m⁻³)')
    ax.set_title('(b) Electron density vs Pressure\ncf. Fig. 10b')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 3. Neutral species vs Pressure (cf. Fig 7b)
    # =========================================================================
    ax = axes[0, 2]
    for sp, color, ls in [
        ('NF3', 'blue', '-'), ('NF2', 'green', '-'), ('NF', 'cyan', '--'),
        ('F2', 'orange', '-'), ('N2', 'purple', '--'), ('F', 'red', '-'),
        ('N', 'brown', '--'), ('Ar', 'gray', '-')
    ]:
        vals = [r.get(sp, 0) / 1e18 for r in res_press]
        if max(vals) > 0.001:
            ax.plot(pressures, vals, color=color, linestyle=ls, linewidth=1.5,
                    marker='o', markersize=4, label=sp)
    
    ax.set_xlabel('Pressure (mTorr)')
    ax.set_ylabel('Density (×10¹⁸ m⁻³)')
    ax.set_title('(c) Neutral densities vs Pressure\ncf. Fig. 7b')
    ax.legend(ncol=2, fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e4)
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 4. Electron temperature vs Pressure (cf. Fig 11b)
    # =========================================================================
    ax = axes[1, 0]
    Te_vals = [r['Te'] for r in res_press]
    Tg_vals = [r['Tg'] for r in res_press]
    
    ax.plot(pressures, Te_vals, 'bo-', linewidth=2, markersize=6, label='Te (eV)')
    ax2 = ax.twinx()
    ax2.plot(pressures, Tg_vals, 'r^--', linewidth=2, markersize=6, label='Tg (K)')
    ax2.set_ylabel('Gas Temperature (K)', color='red')
    
    ax.set_xlabel('Pressure (mTorr)')
    ax.set_ylabel('Electron Temperature (eV)', color='blue')
    ax.set_title('(d) Te and Tg vs Pressure\ncf. Fig. 11b')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 5. Electronegativity vs Pressure (cf. Fig 14)
    # =========================================================================
    ax = axes[1, 1]
    alpha_vals = [r['alpha'] for r in res_press]
    ax.plot(pressures, alpha_vals, 'g^-', linewidth=2, markersize=6, label='20% NF3')
    ax.set_xlabel('Pressure (mTorr)')
    ax.set_ylabel('Electronegativity (n⁻/ne)')
    ax.set_title('(e) Electronegativity vs Pressure\ncf. Fig. 14')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 6. NF3 dissociation rate vs Pressure (cf. Fig 5b)
    # =========================================================================
    ax = axes[1, 2]
    
    # Compute dissociation fractions
    dissoc_vals = []
    for r in res_press:
        n_NF3 = r.get('NF3', 0)
        n_Ar = r.get('Ar', 0)
        # Initial values from ideal gas law
        Tg = r['Tg']
        p_Pa = r['param'] * 0.1333
        n_total_0 = p_Pa / (1.381e-23 * 300.0)  # at room temp
        n_NF3_0 = 0.2 * n_total_0
        n_Ar_0 = 0.8 * n_total_0
        
        # Dissociation = 1 - (Ar0/Ar) * (NF3/NF3_0) corrected for gas heating
        if n_NF3_0 > 0 and n_Ar > 0:
            ratio = n_NF3 / (n_NF3 + sum(r.get(s, 0) for s in ['NF2', 'NF', 'F', 'F2', 'N', 'N2']) * 71/19)
            dissoc = (1.0 - n_NF3 / (n_total_0 * 0.2 * 300 / Tg)) * 100
            dissoc = max(0, min(100, dissoc))
        else:
            dissoc = 0
        dissoc_vals.append(dissoc)
    
    ax.plot(pressures, dissoc_vals, 'mo-', linewidth=2, markersize=6, label='20% NF3')
    ax.set_xlabel('Pressure (mTorr)')
    ax.set_ylabel('NF3 Dissociation (%)')
    ax.set_title('(f) NF3 Dissociation vs Pressure\ncf. Fig. 5b')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('/home/claude/nf3_model/nf3_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to nf3_results.png")
    
    return fig


if __name__ == '__main__':
    plot_results()
