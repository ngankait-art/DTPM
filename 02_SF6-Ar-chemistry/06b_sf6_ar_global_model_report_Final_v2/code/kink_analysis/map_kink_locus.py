#!/usr/bin/env python3
"""
Kink Locus Mapping — SF6/Ar Wall Chemistry Model
==================================================
Maps the wall chemistry buffer exhaustion transition ("kink") across
operating conditions (Power, Pressure) in Ar fraction space.

Method:
  For each (P_rf, p_mTorr) pair:
    1. Sweep Ar fraction from 0% to 85% at 1% resolution
    2. Compute dα/d(Ar%) numerically (central differences)
    3. Find the Ar% where dα/d(Ar%) is most negative → kink location
    4. Record kink Ar%, α at kink, dα/d(Ar%) at kink, ne at kink

Output:
  - kink_locus_data.csv with all (P, p, Ar%_kink, α_kink, ...) data
  - kink_locus_heatmap.png — heatmap of kink Ar% in (P, p) space
  - kink_locus_3d.png — 3D surface of kink location
  - kink_alpha_at_kink.png — α value at the kink vs conditions

Usage:
  cd kink_locus/ && python map_kink_locus.py

Requires engine.py in the same directory or parent.
"""

import sys, os, csv, time
import numpy as np

# Import the solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from engine import solve_model, sweep_with_continuation, Reactor
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
    from engine import solve_model, sweep_with_continuation, Reactor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Operating condition grid
POWERS   = [500, 750, 1000, 1250, 1500, 1750, 2000]          # W
PRESSURES = [5, 7.5, 10, 15, 20, 30]                          # mTorr

# Ar fraction sweep resolution (1% steps for speed; 0.5% for publication)
AR_FRACS = np.arange(0.0, 0.86, 0.01)

# Reactor and wall chemistry (Lallement geometry, Kokkoris × 0.007)
REACTOR = Reactor(R=0.180, L=0.175)
WC = {'s_F': 0.00105, 's_SFx': 0.00056, 'p_fluor': 0.025,
      'p_wallrec': 0.007, 'p_FF': 0.0035, 'p_F_SF3': 0.5, 'p_F_SF4': 0.2}

# Plot style
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 9.5, 'figure.dpi': 150, 'lines.linewidth': 2.0,
    'axes.grid': True, 'grid.alpha': 0.3,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_kink(ar_fracs, alphas):
    """Detect the kink location from an α vs Ar% curve.
    
    The kink is defined as the Ar fraction where dα/d(Ar%) is most negative,
    excluding the endpoints and requiring α > 2 (to avoid the trivial EN→EP
    crossover at α ≈ 1).
    
    Returns
    -------
    dict with keys:
        ar_kink : float — Ar fraction at the kink (0–1)
        alpha_kink : float — α value at the kink
        dalpha_max : float — most negative dα/d(Ar%) value
        kink_width : float — width of the kink region (Ar% where |dα/d%| > 2× background)
        kink_detected : bool — whether a clear kink was found
    """
    if len(ar_fracs) < 5 or len(alphas) < 5:
        return {'ar_kink': np.nan, 'alpha_kink': np.nan, 'dalpha_max': np.nan,
                'kink_width': np.nan, 'kink_detected': False}
    
    ar_pct = np.array(ar_fracs) * 100  # convert to percentage
    alpha = np.array(alphas)
    
    # Central differences for dα/d(Ar%)
    dalpha = np.gradient(alpha, ar_pct)
    
    # Mask: only look where α > 2 (above trivial EN→EP) and away from endpoints
    mask = (alpha > 2.0) & (ar_pct > 10) & (ar_pct < 82)
    
    if not np.any(mask):
        # No kink in the electronegative regime — system is already EP
        return {'ar_kink': np.nan, 'alpha_kink': np.nan, 'dalpha_max': np.nan,
                'kink_width': np.nan, 'kink_detected': False}
    
    # Find the most negative derivative (steepest drop)
    dalpha_masked = dalpha.copy()
    dalpha_masked[~mask] = 0  # zero out regions outside mask
    
    idx_min = np.argmin(dalpha_masked)
    dalpha_min = dalpha[idx_min]
    
    # Background rate: median of dα/d(Ar%) in the smooth region (20–50% Ar)
    bg_mask = (ar_pct >= 20) & (ar_pct <= 50) & mask
    if np.any(bg_mask):
        bg_rate = np.median(dalpha[bg_mask])
    else:
        bg_rate = np.median(dalpha[mask])
    
    # Kink is "detected" if the steepest drop is at least 3× the background rate
    kink_detected = (dalpha_min < 0) and (abs(dalpha_min) > 3 * abs(bg_rate))
    
    # Kink width: range of Ar% where |dα/d%| exceeds 2× background
    if kink_detected and bg_rate != 0:
        threshold = 2 * abs(bg_rate)
        steep_mask = (abs(dalpha) > threshold) & mask
        if np.any(steep_mask):
            steep_pcts = ar_pct[steep_mask]
            kink_width = steep_pcts[-1] - steep_pcts[0]
        else:
            kink_width = 0.0
    else:
        kink_width = 0.0
    
    return {
        'ar_kink': ar_fracs[idx_min],
        'alpha_kink': alpha[idx_min],
        'dalpha_max': dalpha_min,
        'dalpha_bg': bg_rate,
        'kink_width': kink_width,
        'kink_detected': kink_detected,
    }


def run_locus_mapping():
    """Run the full (P, p) grid scan and detect kink at each point."""
    
    print("=" * 70)
    print("KINK LOCUS MAPPING")
    print(f"Grid: {len(POWERS)} powers × {len(PRESSURES)} pressures = {len(POWERS)*len(PRESSURES)} points")
    print(f"Ar sweep: {AR_FRACS[0]*100:.0f}%–{AR_FRACS[-1]*100:.0f}% in {len(AR_FRACS)} steps")
    print("=" * 70)
    
    results = []
    t0 = time.time()
    
    for ip, P in enumerate(POWERS):
        for jp, p in enumerate(PRESSURES):
            label = f"P={P:>5}W, p={p:>4}mTorr"
            print(f"\n▶ [{ip*len(PRESSURES)+jp+1}/{len(POWERS)*len(PRESSURES)}] {label}")
            
            # Run Ar fraction sweep with continuation
            base = dict(P_rf=P, p_mTorr=p, Q_sccm=40, eta=0.16,
                       wall_chem=WC, reactor=REACTOR)
            sweep = sweep_with_continuation('frac_Ar', AR_FRACS, base, verbose=False)
            
            # Extract α curve
            ar_vals = [r['frac_Ar'] for r in sweep]
            alpha_vals = [r['alpha'] for r in sweep]
            ne_vals = [r['ne'] for r in sweep]
            dissoc_vals = [r['dissoc_frac'] for r in sweep]
            
            # Detect kink
            kink = detect_kink(ar_vals, alpha_vals)
            
            # Get plasma params at kink location
            if kink['kink_detected'] and not np.isnan(kink['ar_kink']):
                idx_k = np.argmin(np.abs(np.array(ar_vals) - kink['ar_kink']))
                ne_kink = ne_vals[idx_k]
                dissoc_kink = dissoc_vals[idx_k]
                Te_kink = sweep[idx_k]['Te']
            else:
                ne_kink = np.nan
                dissoc_kink = np.nan
                Te_kink = np.nan
            
            row = {
                'P_rf_W': P,
                'p_mTorr': p,
                'ar_kink_pct': kink['ar_kink'] * 100 if not np.isnan(kink['ar_kink']) else np.nan,
                'alpha_kink': kink['alpha_kink'],
                'dalpha_max': kink['dalpha_max'],
                'dalpha_bg': kink.get('dalpha_bg', np.nan),
                'kink_width_pct': kink['kink_width'],
                'kink_detected': kink['kink_detected'],
                'ne_kink_cm3': ne_kink * 1e-6 if not np.isnan(ne_kink) else np.nan,
                'Te_kink_eV': Te_kink,
                'dissoc_kink_pct': dissoc_kink * 100 if not np.isnan(dissoc_kink) else np.nan,
                'alpha_curve': alpha_vals,  # store full curve for plotting
                'ar_curve': ar_vals,
            }
            results.append(row)
            
            status = "KINK" if kink['kink_detected'] else "no kink"
            ar_str = f"{kink['ar_kink']*100:.0f}%" if not np.isnan(kink['ar_kink']) else "—"
            a_str = f"{kink['alpha_kink']:.1f}" if not np.isnan(kink['alpha_kink']) else "—"
            print(f"  → {status}: Ar*={ar_str}, α={a_str}, "
                  f"dα/d%={kink['dalpha_max']:.2f}" if not np.isnan(kink['dalpha_max']) else "")
    
    elapsed = time.time() - t0
    print(f"\n✓ Scan complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    return results


def save_csv(results, fname):
    """Save results to CSV (excluding the full curve arrays)."""
    keys = ['P_rf_W', 'p_mTorr', 'ar_kink_pct', 'alpha_kink', 'dalpha_max',
            'dalpha_bg', 'kink_width_pct', 'kink_detected',
            'ne_kink_cm3', 'Te_kink_eV', 'dissoc_kink_pct']
    
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in keys}
            writer.writerow(row)
    print(f"  Saved {fname}")


def plot_heatmap(results, fname='kink_locus_heatmap.png'):
    """Heatmap of kink Ar% in (Power, Pressure) space."""
    
    # Build 2D grid
    P_unique = sorted(set(r['P_rf_W'] for r in results))
    p_unique = sorted(set(r['p_mTorr'] for r in results))
    
    Z = np.full((len(p_unique), len(P_unique)), np.nan)
    Z_alpha = np.full_like(Z, np.nan)
    
    for r in results:
        ip = P_unique.index(r['P_rf_W'])
        jp = p_unique.index(r['p_mTorr'])
        if r['kink_detected']:
            Z[jp, ip] = r['ar_kink_pct']
            Z_alpha[jp, ip] = r['alpha_kink']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Kink location heatmap
    im1 = ax1.imshow(Z, aspect='auto', origin='lower', cmap='RdYlBu_r',
                     extent=[P_unique[0], P_unique[-1], p_unique[0], p_unique[-1]],
                     interpolation='nearest', vmin=40, vmax=85)
    ax1.set_xlabel('ICP Power (W)')
    ax1.set_ylabel('Pressure (mTorr)')
    ax1.set_title('Kink Location: Ar% at Buffer Exhaustion')
    cb1 = plt.colorbar(im1, ax=ax1, label='Ar% at kink')
    
    # Annotate each cell
    for r in results:
        ip = P_unique.index(r['P_rf_W'])
        jp = p_unique.index(r['p_mTorr'])
        if r['kink_detected']:
            ax1.text(r['P_rf_W'], r['p_mTorr'], f"{r['ar_kink_pct']:.0f}%",
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if r['ar_kink_pct'] < 60 else 'black')
        else:
            ax1.text(r['P_rf_W'], r['p_mTorr'], '—',
                    ha='center', va='center', fontsize=8, color='gray')
    
    # α at kink heatmap
    im2 = ax2.imshow(Z_alpha, aspect='auto', origin='lower', cmap='viridis',
                     extent=[P_unique[0], P_unique[-1], p_unique[0], p_unique[-1]],
                     interpolation='nearest')
    ax2.set_xlabel('ICP Power (W)')
    ax2.set_ylabel('Pressure (mTorr)')
    ax2.set_title(r'$\alpha$ at Kink Location')
    cb2 = plt.colorbar(im2, ax=ax2, label=r'$\alpha$ at kink')
    
    for r in results:
        if r['kink_detected'] and not np.isnan(r['alpha_kink']):
            ax2.text(r['P_rf_W'], r['p_mTorr'], f"{r['alpha_kink']:.0f}",
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    plt.suptitle('Wall Chemistry Buffer Exhaustion Locus — Lallement Geometry', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_curves_panel(results, fname='kink_locus_curves.png'):
    """Panel of α vs Ar% curves at different pressures, one column per power."""
    
    P_subset = [750, 1500, 2000]
    p_all = sorted(set(r['p_mTorr'] for r in results))
    
    fig, axes = plt.subplots(1, len(P_subset), figsize=(6*len(P_subset), 5.5), sharey=True)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(p_all)))
    
    for j, P in enumerate(P_subset):
        ax = axes[j]
        for i, p in enumerate(p_all):
            match = [r for r in results if r['P_rf_W'] == P and r['p_mTorr'] == p]
            if match:
                r = match[0]
                ar_pct = [x*100 for x in r['ar_curve']]
                ax.plot(ar_pct, r['alpha_curve'], '-', color=colors[i], lw=1.8,
                        label=f'{p} mTorr')
                if r['kink_detected']:
                    ax.axvline(r['ar_kink_pct'], color=colors[i], ls=':', alpha=0.5, lw=1)
                    ax.plot(r['ar_kink_pct'], r['alpha_kink'], 'o', color=colors[i],
                            ms=8, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        
        ax.set_xlabel('Ar%')
        ax.set_title(f'{P} W')
        ax.set_xlim(0, 85)
        if j == 0:
            ax.set_ylabel(r'$\alpha = n_-/n_e$')
            ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle(r'$\alpha$ vs Ar fraction — kink locations marked (●)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_locus_line(results, fname='kink_locus_line.png'):
    """Plot kink Ar% vs Power at each pressure — the locus as parametric curves."""
    
    p_all = sorted(set(r['p_mTorr'] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(p_all)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, p in enumerate(p_all):
        rows = sorted([r for r in results if r['p_mTorr'] == p and r['kink_detected']],
                      key=lambda x: x['P_rf_W'])
        if len(rows) < 2:
            continue
        P_vals = [r['P_rf_W'] for r in rows]
        ar_vals = [r['ar_kink_pct'] for r in rows]
        alpha_vals = [r['alpha_kink'] for r in rows]
        
        ax1.plot(P_vals, ar_vals, 'o-', color=colors[i], ms=7, lw=2, label=f'{p} mTorr')
        ax2.plot(P_vals, alpha_vals, 'o-', color=colors[i], ms=7, lw=2, label=f'{p} mTorr')
    
    ax1.set_xlabel('ICP Power (W)')
    ax1.set_ylabel('Ar% at kink')
    ax1.set_title('Kink Location vs Power')
    ax1.legend()
    ax1.set_ylim(30, 90)
    
    ax2.set_xlabel('ICP Power (W)')
    ax2.set_ylabel(r'$\alpha$ at kink')
    ax2.set_title(r'$\alpha$ at Kink vs Power')
    ax2.legend()
    
    plt.suptitle('Wall Chemistry Buffer Exhaustion Locus', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def print_summary(results):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("KINK LOCUS SUMMARY")
    print("=" * 90)
    print(f"{'P (W)':>7} {'p (mT)':>7} {'Kink?':>6} {'Ar%':>6} {'α':>7} "
          f"{'dα/d%':>8} {'Width':>6} {'ne (cm⁻³)':>12} {'Te (eV)':>8} {'Dissoc%':>8}")
    print("-" * 90)
    
    for r in sorted(results, key=lambda x: (x['P_rf_W'], x['p_mTorr'])):
        det = "YES" if r['kink_detected'] else "no"
        ar = f"{r['ar_kink_pct']:.0f}" if not np.isnan(r.get('ar_kink_pct', np.nan)) else "—"
        al = f"{r['alpha_kink']:.1f}" if not np.isnan(r.get('alpha_kink', np.nan)) else "—"
        da = f"{r['dalpha_max']:.2f}" if not np.isnan(r.get('dalpha_max', np.nan)) else "—"
        wi = f"{r['kink_width_pct']:.0f}" if not np.isnan(r.get('kink_width_pct', np.nan)) else "—"
        ne = f"{r['ne_kink_cm3']:.1e}" if not np.isnan(r.get('ne_kink_cm3', np.nan)) else "—"
        te = f"{r['Te_kink_eV']:.2f}" if not np.isnan(r.get('Te_kink_eV', np.nan)) else "—"
        di = f"{r['dissoc_kink_pct']:.0f}" if not np.isnan(r.get('dissoc_kink_pct', np.nan)) else "—"
        
        print(f"{r['P_rf_W']:>7} {r['p_mTorr']:>7} {det:>6} {ar:>6} {al:>7} "
              f"{da:>8} {wi:>6} {ne:>12} {te:>8} {di:>8}")
    
    # Physical trends
    detected = [r for r in results if r['kink_detected']]
    if detected:
        print(f"\n--- Physical Trends ---")
        print(f"  Kink detected in {len(detected)}/{len(results)} conditions")
        ar_range = (min(r['ar_kink_pct'] for r in detected), max(r['ar_kink_pct'] for r in detected))
        print(f"  Kink Ar% range: {ar_range[0]:.0f}%–{ar_range[1]:.0f}%")
        
        # Power trend at 10 mTorr
        p10 = sorted([r for r in detected if r['p_mTorr'] == 10], key=lambda x: x['P_rf_W'])
        if len(p10) >= 2:
            print(f"  At 10 mTorr: kink moves from {p10[0]['ar_kink_pct']:.0f}% Ar "
                  f"({p10[0]['P_rf_W']}W) to {p10[-1]['ar_kink_pct']:.0f}% Ar ({p10[-1]['P_rf_W']}W)")
            trend = "lower" if p10[-1]['ar_kink_pct'] < p10[0]['ar_kink_pct'] else "higher"
            print(f"  → Higher power pushes the kink to {trend} Ar fractions")
        
        # Pressure trend at 1500 W
        P15 = sorted([r for r in detected if r['P_rf_W'] == 1500], key=lambda x: x['p_mTorr'])
        if len(P15) >= 2:
            print(f"  At 1500 W: kink moves from {P15[0]['ar_kink_pct']:.0f}% Ar "
                  f"({P15[0]['p_mTorr']}mT) to {P15[-1]['ar_kink_pct']:.0f}% Ar ({P15[-1]['p_mTorr']}mT)")
            trend = "higher" if P15[-1]['ar_kink_pct'] > P15[0]['ar_kink_pct'] else "lower"
            print(f"  → Higher pressure pushes the kink to {trend} Ar fractions")


def main():
    # Run the full scan
    results = run_locus_mapping()
    
    # Save data
    print("\n▶ Saving outputs...")
    save_csv(results, 'kink_locus_data.csv')
    
    # Generate figures
    print("\n▶ Generating figures...")
    plot_heatmap(results)
    plot_curves_panel(results)
    plot_locus_line(results)
    
    # Print summary
    print_summary(results)
    
    print(f"\n✓ All outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
