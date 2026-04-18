#!/usr/bin/env python3
"""
TEL ICP Reactor — Model Hierarchy Runner
==========================================
Runs Models A, B, C side-by-side and generates comparison figures.

Model A: Calibrated 2-species (F + SF6), gamma_Al = 0.18
Model B: Hybrid 9-species (calibrated), gamma_Al = 0.18
Model C: Hybrid 9-species (uncalibrated), gamma_Al = 0.015 (Kokkoris)

Usage:
    python run_models.py              # Run all models
    python run_models.py --model A    # Run only Model A
    python run_models.py --model C    # Run only Model C
"""

import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.constants import k as kB, pi, atomic_mass as AMU
from scipy.special import j0

from solver import TELSolver
from sf6_chemistry import compute_rates, SPECIES, M, WALL_GAMMA

FIGS = 'results/figures'
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({'font.size':13,'font.family':'sans-serif','axes.labelsize':14,
    'axes.titlesize':14,'xtick.labelsize':12,'ytick.labelsize':12,'lines.linewidth':2.5,
    'axes.linewidth':1.2,'figure.facecolor':'white','savefig.facecolor':'white'})


def run_model_A(Nr=50, Nz=80, gamma_Al=0.18, verbose=True):
    """Model A: Calibrated 2-species (F + SF6)."""
    if verbose: print("\n" + "="*60 + "\n  MODEL A: Calibrated 2-species\n" + "="*60)
    s = TELSolver(Nr=Nr, Nz=Nz, gamma_Al=gamma_Al)
    r = s.solve(n_iter=70, w=0.12, verbose=verbose)
    r['model'] = 'A'
    r['gamma_Al'] = gamma_Al
    r['calibrated'] = True
    return r, s


def run_hybrid_multispecies(solver_result, solver_obj, verbose=True):
    """Compute 7 intermediate species from local 0D chemistry."""
    r = solver_result; s = solver_obj
    m = r['mesh']; Nr, Nz = m.Nr, m.Nz; ins = r['inside']
    
    Te0 = 2.7; ng = 0.1333*10/(kB*313)
    P_abs = 0.43*700; eps_T = (280+7.2*Te0)*1.602e-19
    k_iz = 1.2e-7*np.exp(-18.1/Te0)*1e-6
    V_icp = pi*s.R_icp**2*s.L_icp
    ne_avg = min(P_abs/(k_iz*ng*eps_T*V_icp+1e-30), 1e18)
    
    ne_field = np.zeros((Nr,Nz))
    for i in range(Nr):
        rv = m.rc[i]
        for j in range(Nz):
            z = m.zc[j]
            if ins[i,j]:
                if z >= s.z_apt_top and rv <= s.R_icp:
                    ne_field[i,j] = ne_avg*j0(2.405*rv/s.R_icp)*max(np.cos(pi*(z-s.z_apt_top)/(2*s.L_icp)),0.01)
                else: ne_field[i,j] = ne_avg*0.005
    
    ng_cm3 = ng*1e-6; k = compute_rates(Te0, ng_cm3)
    tau_R = 0.1333*10*np.sum(m.vol[ins])/(100*1.01325e5/60*1e-6)
    
    fields = {sp: np.zeros((Nr,Nz)) for sp in SPECIES}
    fields['SF6'] = r.get('nSF6', np.where(ins, ng*0.5, 0))
    fields['F'] = r['nF']
    
    for i in range(Nr):
        for j in range(Nz):
            if not ins[i,j]: continue
            ne_l = ne_field[i,j]; nSF6_l = fields['SF6'][i,j]; nF_l = fields['F'][i,j]
            if ne_l < 1e10 or nSF6_l < 1e10: continue
            fields['SF5'][i,j] = max(k['d1']*ne_l*nSF6_l/((k['d7']+k['iz25']+k['iz26'])*ne_l+k['nr42']*nF_l+1/tau_R+1e-30), 0)
            fields['SF4'][i,j] = max((k['d2']*ne_l*nSF6_l+k['d7']*ne_l*fields['SF5'][i,j]+k['nr45']*fields['SF5'][i,j]**2)/(k['d8']*ne_l+k['nr41']*nF_l+1/tau_R+1e-30), 0)
            fields['SF3'][i,j] = max((k['d3']*ne_l*nSF6_l+k['d8']*ne_l*fields['SF4'][i,j])/((k['d9']+k['iz27'])*ne_l+k['nr40']*nF_l+1/tau_R+1e-30), 0)
            fields['SF2'][i,j] = max((k['d4']*ne_l*nSF6_l+k['d9']*ne_l*fields['SF3'][i,j])/(k['d10']*ne_l+k['nr39']*nF_l+1/tau_R+1e-30), 0)
            fields['SF'][i,j]  = max((k['d5']*ne_l*nSF6_l+k['d10']*ne_l*fields['SF2'][i,j])/(k['d11']*ne_l+k['nr38']*nF_l+1/tau_R+1e-30), 0)
            fields['S'][i,j]   = max(k['d11']*ne_l*fields['SF'][i,j]/(k['iz29']*ne_l+k['nr37']*nF_l+1/tau_R+1e-30), 0)
            fields['F2'][i,j]  = max(ne_l*nSF6_l*(k['d4']+k['d5']+k['iz21']+k['iz22']+k['iz23'])/(k['d6']*ne_l+1/tau_R+1e-30), 0)
    
    r['fields'] = fields
    icp = ins & (np.outer(m.rc,np.ones(Nz))<=s.R_icp) & (np.outer(np.ones(Nr),m.zc)>=s.z_apt_top)
    proc = ins & (np.outer(np.ones(Nr),m.zc)<s.z_apt_bot)
    for sp in SPECIES:
        f = fields[sp]
        r[f'{sp}_icp'] = np.sum(f[icp]*m.vol[icp])/max(np.sum(m.vol[icp]),1e-30)
        r[f'{sp}_proc'] = np.sum(f[proc]*m.vol[proc])/max(np.sum(m.vol[proc]),1e-30)
    return r


def run_model_B(verbose=True):
    """Model B: Hybrid 9-species with calibrated gamma_Al."""
    if verbose: print("\n" + "="*60 + "\n  MODEL B: Hybrid multi-species (calibrated)\n" + "="*60)
    r, s = run_model_A(gamma_Al=0.18, verbose=verbose)
    r = run_hybrid_multispecies(r, s, verbose)
    r['model'] = 'B'; r['calibrated'] = True
    if verbose:
        print(f"  [F] drop = {r['F_drop_pct']:.1f}%")
        for sp in SPECIES:
            print(f"  [{sp}] ICP = {r[f'{sp}_icp']*1e-6:.2e} cm-3")
    return r, s


def run_model_C(verbose=True):
    """Model C: Hybrid 9-species with UNCALIBRATED gamma_Al."""
    if verbose: print("\n" + "="*60 + "\n  MODEL C: Uncalibrated prediction (Kokkoris literature)\n" + "="*60)
    r, s = run_model_A(gamma_Al=0.015, verbose=verbose)
    r = run_hybrid_multispecies(r, s, verbose)
    r['model'] = 'C'; r['calibrated'] = False
    if verbose:
        print(f"  [F] drop = {r['F_drop_pct']:.1f}%")
    return r, s


def plot_model_comparison(rA, rC, sA):
    """MANDATORY: Calibrated vs uncalibrated vs experiment."""
    m = rA['mesh']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) Absolute F density at wafer
    FA = rA['nF'][:, 0] * 1e-6; FC = rC['nF'][:, 0] * 1e-6
    ax1.plot(m.rc*1000, FA, 'r-', lw=3, label=f'Model A (calibrated, $\\gamma_{{Al}}$=0.18)')
    ax1.plot(m.rc*1000, FC, 'b--', lw=3, label=f'Model C (uncalib., $\\gamma_{{Al}}$=0.015)')
    ax1.set_xlabel('$r$ (mm)'); ax1.set_ylabel('[F] at wafer (cm$^{-3}$)')
    ax1.set_title('(a) Absolute [F] at Wafer', fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(alpha=0.2, ls=':')
    ax1.set_xlim(0, sA.R_proc*1000+5)
    
    # (b) Normalised profiles
    FA_n = FA / max(FA[0], 1e-6); FC_n = FC / max(FC[0], 1e-6)
    ax2.plot(m.rc*1000, FA_n, 'r-', lw=3, label=f'Model A: {rA["F_drop_pct"]:.0f}% drop')
    ax2.plot(m.rc*1000, FC_n, 'b--', lw=3, label=f'Model C: {rC["F_drop_pct"]:.0f}% drop')
    ax2.axhline(0.26, ls=':', color='#333', lw=1.5, label='Mettler 74% drop level')
    
    # Mettler data points (digitised from Fig 4.14)
    r_mettler = np.array([0, 25, 50, 75, 100]) * 1e-3 * 1000  # mm
    f_mettler = np.array([1.0, 0.90, 0.68, 0.42, 0.26])
    ax2.plot(r_mettler, f_mettler, 'ko', markersize=10, zorder=5, label='Mettler data')
    
    ax2.set_xlabel('$r$ (mm)'); ax2.set_ylabel('$[F] / [F]_{r=0}$')
    ax2.set_title('(b) Normalised Radial Profile', fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(alpha=0.2, ls=':')
    ax2.set_xlim(0, sA.R_proc*1000+5); ax2.set_ylim(0, 1.15)
    
    plt.suptitle('Model Comparison: Calibrated vs Uncalibrated vs Experiment', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{FIGS}/fig_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGS}/fig_model_comparison.png")


def plot_flux_budget(r, s):
    """Flux budget: source vs loss breakdown."""
    m = r['mesh']; Nr, Nz = m.Nr, m.Nz; ins = r['inside']
    nF = r['nF']; ng = 0.1333*10/(kB*313)
    
    Te0 = 2.7; ne_avg = 4e17
    k = compute_rates(Te0, ng*1e-6)
    tau_R = 0.1333*10*np.sum(m.vol[ins])/(100*1.01325e5/60*1e-6)
    nSF6 = r.get('nSF6', np.where(ins, ng*0.5, 0))
    
    # Compute ne field
    ne = np.zeros((Nr,Nz))
    for i in range(Nr):
        for j in range(Nz):
            if ins[i,j]:
                if m.zc[j] >= s.z_apt_top and m.rc[i] <= s.R_icp:
                    ne[i,j] = ne_avg*j0(2.405*m.rc[i]/s.R_icp)*max(np.cos(pi*(m.zc[j]-s.z_apt_top)/(2*s.L_icp)),0.01)
                else: ne[i,j] = ne_avg*0.005
    
    # Source: electron-impact dissociation
    R_prod = ne * nSF6 * (k['d1']+2*k['d2']+3*k['d3']+2*k['d4']+3*k['d5'])
    total_prod = np.sum(R_prod[ins] * m.vol[ins])
    
    # Losses by type
    icp_mask = ins & (np.outer(m.rc,np.ones(Nz))<=s.R_icp) & (np.outer(np.ones(Nr),m.zc)>=s.z_apt_top)
    proc_mask = ins & (np.outer(np.ones(Nr),m.zc)<s.z_apt_bot)
    
    prod_icp = np.sum(R_prod[icp_mask] * m.vol[icp_mask])
    prod_proc = np.sum(R_prod[proc_mask] * m.vol[proc_mask])
    pump_loss = np.sum((nF/tau_R)[ins] * m.vol[ins])
    
    # Wall losses (approximate from gamma * v_th/4 * n * A)
    v_th = np.sqrt(8*kB*313/(pi*19*AMU))
    gamma_Al = r.get('gamma_Al', 0.18)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Production\n(ICP)', 'Production\n(proc)', 'Pump\nloss', 'Al wall\nloss', 'Quartz\nloss', 'Wafer\nloss']
    values = [prod_icp, prod_proc, -pump_loss, 0, 0, 0]
    
    # Estimate wall losses from boundary cells
    for i in range(Nr):
        for j in range(Nz):
            if not ins[i,j]: continue
            bc = s.bc_type[i,j]
            from geometry import BC_AL_SIDE, BC_AL_TOP, BC_QUARTZ, BC_WAFER, BC_WINDOW
            n_local = nF[i,j]
            if bc in (BC_AL_SIDE, BC_AL_TOP):
                values[3] -= gamma_Al * v_th/4 * n_local * m.vol[i,j] / (0.5*m.dr[i])
            elif bc == BC_QUARTZ:
                values[4] -= 0.001 * v_th/4 * n_local * m.vol[i,j] / (0.5*m.dr[i])
            elif bc == BC_WAFER:
                values[5] -= 0.025 * v_th/4 * n_local * m.vol[i,j] / (0.5*m.dz[j])
    
    colors = ['#2ecc71','#27ae60','#e74c3c','#c0392b','#e67e22','#8e44ad']
    bars = ax.bar(categories, [abs(v) for v in values], color=colors, edgecolor='#333', zorder=3)
    
    for b, v in zip(bars, values):
        sign = '+' if v > 0 else '-'
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.05, f'{sign}{abs(v):.1e}',
                ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Rate (m$^{-3}$ s$^{-1}$ × volume)', fontsize=13)
    ax.set_title(f'Fluorine Flux Budget — Model {"A" if gamma_Al==0.18 else "C"}\n'
                 f'$\\gamma_{{Al}}$ = {gamma_Al}', fontsize=14, fontweight='bold')
    ax.set_yscale('log'); ax.grid(axis='y', alpha=0.2, ls=':')
    plt.tight_layout()
    fig.savefig(f'{FIGS}/fig_flux_budget.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGS}/fig_flux_budget.png")


def plot_sensitivity_map(verbose=True):
    """2D sensitivity map: gamma_Al vs pressure."""
    if verbose: print("  Computing sensitivity map (gamma_Al vs pressure)...")
    gammas = [0.02, 0.05, 0.10, 0.15, 0.18, 0.25, 0.30]
    pressures = [5, 10, 15, 20]
    drops = np.zeros((len(gammas), len(pressures)))
    
    for ig, g in enumerate(gammas):
        for ip, p in enumerate(pressures):
            s = TELSolver(Nr=30, Nz=45, gamma_Al=g, p_mTorr=p)
            r = s.solve(n_iter=50, w=0.12, verbose=False)
            drops[ig, ip] = r['F_drop_pct']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(drops, origin='lower', aspect='auto', cmap='RdYlBu_r',
                   extent=[pressures[0]-2.5, pressures[-1]+2.5, gammas[0]-0.01, gammas[-1]+0.01])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('[F] drop (%)', fontsize=13)
    
    # Add contour lines
    P, G = np.meshgrid(pressures, gammas)
    cs = ax.contour(P, G, drops, levels=[30, 50, 60, 70, 74, 80], colors='k', linewidths=1.5)
    ax.clabel(cs, fmt='%d%%', fontsize=10)
    
    ax.plot([10], [0.18], 'w*', markersize=15, markeredgecolor='k', zorder=5, label='Calibrated (Model A)')
    ax.plot([10], [0.015], 'ws', markersize=10, markeredgecolor='k', zorder=5, label='Kokkoris (Model C)')
    
    ax.set_xlabel('Pressure (mTorr)', fontsize=14)
    ax.set_ylabel('$\\gamma_{Al}$', fontsize=14)
    ax.set_title('Sensitivity Map: [F] Drop vs $\\gamma_{Al}$ and Pressure', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    fig.savefig(f'{FIGS}/fig_sensitivity_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGS}/fig_sensitivity_map.png")


def print_summary_table(rA, rB, rC):
    """Print the comparison summary table."""
    print("\n" + "="*75)
    print("  MODEL COMPARISON SUMMARY TABLE")
    print("="*75)
    print(f"{'':>25s} {'Model A':>14s} {'Model B':>14s} {'Model C':>14s} {'Mettler':>10s}")
    print(f"{'':>25s} {'(calib 2sp)':>14s} {'(calib 9sp)':>14s} {'(uncalib 9sp)':>14s}")
    print("-"*75)
    print(f"{'gamma_Al':>25s} {rA['gamma_Al']:14.3f} {rB['gamma_Al']:14.3f} {rC['gamma_Al']:14.3f} {'—':>10s}")
    print(f"{'Calibrated?':>25s} {'YES':>14s} {'YES':>14s} {'NO':>14s} {'—':>10s}")
    print(f"{'[F] drop (%)':>25s} {rA['F_drop_pct']:14.1f} {rB['F_drop_pct']:14.1f} {rC['F_drop_pct']:14.1f} {'74':>10s}")
    print(f"{'[F] ICP (cm-3)':>25s} {rA['nF'][rA['inside']].mean()*1e-6:14.2e}", end='')
    if 'F_icp' in rB: print(f" {rB['F_icp']*1e-6:14.2e}", end='')
    else: print(f" {'—':>14s}", end='')
    if 'F_icp' in rC: print(f" {rC['F_icp']*1e-6:14.2e}", end='')
    print(f" {'2.6e+14':>10s}")
    print(f"{'# species':>25s} {'2':>14s} {'9':>14s} {'9':>14s} {'—':>10s}")
    print("="*75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEL ICP Model Hierarchy Runner')
    parser.add_argument('--model', choices=['A','B','C','all'], default='all')
    parser.add_argument('--plots', action='store_true', default=True)
    args = parser.parse_args()
    
    t0 = time.time()
    results = {}
    
    if args.model in ('A', 'all'):
        rA, sA = run_model_A()
        results['A'] = (rA, sA)
    
    if args.model in ('B', 'all'):
        rB, sB = run_model_B()
        results['B'] = (rB, sB)
    
    if args.model in ('C', 'all'):
        rC, sC = run_model_C()
        results['C'] = (rC, sC)
    
    if args.model == 'all' and args.plots:
        rA, sA = results['A']; rB, _ = results['B']; rC, _ = results['C']
        
        print("\n" + "="*60 + "\n  GENERATING COMPARISON PLOTS\n" + "="*60)
        plot_model_comparison(rA, rC, sA)
        plot_flux_budget(rA, sA)
        plot_sensitivity_map()
        
        print_summary_table(rA, rB, rC)
    
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
