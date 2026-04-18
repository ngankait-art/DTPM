"""
Phase A: Comprehensive dataset diagnosis comparing LXCat vs legacy datasets.

Quantifies:
- Target dynamic range (nF, nSF6)
- Spatial gradient sharpness
- Variance across operating conditions
- Cross-correlations (Te-nF, Te-nSF6, ne-nF, ne-nSF6)
- Regime structure by pressure/power/Ar fraction
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))

DS_V4 = os.path.join(REPO, 'results', 'pinn_dataset_v4')
DS_LX = os.path.join(REPO, 'results', 'pinn_dataset_lxcat_v3')
OUT = os.path.join(REPO, 'results', 'lxcat_architecture_upgrade')


def load_all_cases(ds_dir):
    """Load all cases into structured arrays."""
    with open(os.path.join(ds_dir, 'metadata.json')) as f:
        meta = json.load(f)
    meta = [e for e in meta if 'error' not in e]

    cases = []
    for entry in meta:
        fpath = os.path.join(ds_dir, entry['file'])
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        inside = d['inside'].astype(bool)
        cases.append({
            'idx': entry['idx'],
            'P_rf': entry['P_rf'],
            'p_mTorr': entry['p_mTorr'],
            'frac_Ar': entry['frac_Ar'],
            'ne_avg': float(d['ne_avg']),
            'F_drop_pct': float(d['F_drop_pct']),
            'nF': d['nF'],
            'nSF6': d['nSF6'],
            'Te': d['Te'],
            'ne': d['ne'],
            'inside': inside,
            'rc': d['rc'],
            'zc': d['zc'],
        })
    return cases


def compute_stats(cases, label):
    """Compute comprehensive statistics for a dataset."""
    # Collect all interior values
    all_nF, all_nSF6, all_Te, all_ne = [], [], [], []
    all_lnF, all_lnSF6 = [], []

    # Per-case stats
    case_nF_range, case_nSF6_range = [], []
    case_Te_mean, case_ne_avg = [], []
    case_nF_mean, case_nSF6_mean = [], []
    case_P, case_p, case_Ar = [], [], []

    # Gradient stats
    grad_nF_r, grad_nF_z = [], []
    grad_nSF6_r, grad_nSF6_z = [], []

    # Cross-correlations per case
    corr_Te_nF, corr_Te_nSF6 = [], []
    corr_ne_nF, corr_ne_nSF6 = [], []

    for c in cases:
        inside = c['inside']
        nF = c['nF'][inside]
        nSF6 = c['nSF6'][inside]
        Te = c['Te'][inside]
        ne = c['ne'][inside]

        lnF = np.log10(np.maximum(nF, 1e6))
        lnSF6 = np.log10(np.maximum(nSF6, 1e6))

        all_nF.extend(nF.tolist())
        all_nSF6.extend(nSF6.tolist())
        all_Te.extend(Te.tolist())
        all_ne.extend(ne.tolist())
        all_lnF.extend(lnF.tolist())
        all_lnSF6.extend(lnSF6.tolist())

        case_nF_range.append(float(lnF.max() - lnF.min()))
        case_nSF6_range.append(float(lnSF6.max() - lnSF6.min()))
        case_Te_mean.append(float(Te.mean()))
        case_ne_avg.append(float(c['ne_avg']))
        case_nF_mean.append(float(lnF.mean()))
        case_nSF6_mean.append(float(lnSF6.mean()))
        case_P.append(c['P_rf'])
        case_p.append(c['p_mTorr'])
        case_Ar.append(c['frac_Ar'])

        # Spatial gradients (on the 2D grid)
        Nr, Nz = c['nF'].shape
        rc, zc = c['rc'], c['zc']
        dr = rc[1] - rc[0] if len(rc) > 1 else 1e-3
        dz = zc[1] - zc[0] if len(zc) > 1 else 1e-3

        lnF_2d = np.log10(np.maximum(c['nF'], 1e6))
        lnSF6_2d = np.log10(np.maximum(c['nSF6'], 1e6))

        for i in range(1, Nr - 1):
            for j in range(1, Nz - 1):
                if inside[i, j]:
                    grad_nF_r.append(abs(lnF_2d[i+1, j] - lnF_2d[i-1, j]) / (2 * dr))
                    grad_nF_z.append(abs(lnF_2d[i, j+1] - lnF_2d[i, j-1]) / (2 * dz))
                    grad_nSF6_r.append(abs(lnSF6_2d[i+1, j] - lnSF6_2d[i-1, j]) / (2 * dr))
                    grad_nSF6_z.append(abs(lnSF6_2d[i, j+1] - lnSF6_2d[i, j-1]) / (2 * dz))

        # Cross-correlations
        if len(nF) > 10:
            def safe_corr(a, b):
                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    return 0.0
                return float(np.corrcoef(a, b)[0, 1])

            corr_Te_nF.append(safe_corr(Te, lnF))
            corr_Te_nSF6.append(safe_corr(Te, lnSF6))
            corr_ne_nF.append(safe_corr(ne, lnF))
            corr_ne_nSF6.append(safe_corr(ne, lnSF6))

    all_nF = np.array(all_nF)
    all_nSF6 = np.array(all_nSF6)
    all_Te = np.array(all_Te)
    all_ne = np.array(all_ne)
    all_lnF = np.array(all_lnF)
    all_lnSF6 = np.array(all_lnSF6)

    # Regime analysis: bin by pressure, power, Ar
    def regime_variance(vals, bins_arr, bin_edges):
        """Compute between-bin and within-bin variance."""
        bin_means = []
        within_vars = []
        for i in range(len(bin_edges) - 1):
            mask = (bins_arr >= bin_edges[i]) & (bins_arr < bin_edges[i + 1])
            v = np.array(vals)[mask]
            if len(v) > 1:
                bin_means.append(float(v.mean()))
                within_vars.append(float(v.var()))
        between = float(np.var(bin_means)) if len(bin_means) > 1 else 0.0
        within = float(np.mean(within_vars)) if within_vars else 0.0
        return between, within

    P_edges = [0, 400, 700, 1000, 1300]
    p_edges = [0, 5, 10, 15, 25]
    Ar_edges = [0, 0.1, 0.25, 0.55]

    P_arr = np.array(case_P)
    p_arr = np.array(case_p)
    Ar_arr = np.array(case_Ar)
    nF_arr = np.array(case_nF_mean)
    nSF6_arr = np.array(case_nSF6_mean)

    regime_P_nF = regime_variance(nF_arr, P_arr, P_edges)
    regime_p_nF = regime_variance(nF_arr, p_arr, p_edges)
    regime_Ar_nF = regime_variance(nF_arr, Ar_arr, Ar_edges)
    regime_P_nSF6 = regime_variance(nSF6_arr, P_arr, P_edges)
    regime_p_nSF6 = regime_variance(nSF6_arr, p_arr, p_edges)
    regime_Ar_nSF6 = regime_variance(nSF6_arr, Ar_arr, Ar_edges)

    stats = {
        'label': label,
        'n_cases': len(cases),
        'target_range': {
            'lnF': {
                'min': float(all_lnF.min()),
                'max': float(all_lnF.max()),
                'range': float(all_lnF.max() - all_lnF.min()),
                'std': float(all_lnF.std()),
                'mean': float(all_lnF.mean()),
            },
            'lnSF6': {
                'min': float(all_lnSF6.min()),
                'max': float(all_lnSF6.max()),
                'range': float(all_lnSF6.max() - all_lnSF6.min()),
                'std': float(all_lnSF6.std()),
                'mean': float(all_lnSF6.mean()),
            },
        },
        'Te': {
            'min': float(all_Te.min()),
            'max': float(all_Te.max()),
            'range': float(all_Te.max() - all_Te.min()),
            'mean': float(all_Te.mean()),
            'std': float(all_Te.std()),
        },
        'ne': {
            'min': float(all_ne.min()),
            'max': float(all_ne.max()),
            'log_range': float(np.log10(max(all_ne.max(), 1)) - np.log10(max(all_ne.min(), 1))),
            'mean': float(all_ne.mean()),
        },
        'spatial_gradients': {
            'lnF_radial': {
                'mean': float(np.mean(grad_nF_r)),
                'p50': float(np.percentile(grad_nF_r, 50)),
                'p90': float(np.percentile(grad_nF_r, 90)),
                'p99': float(np.percentile(grad_nF_r, 99)),
            },
            'lnF_axial': {
                'mean': float(np.mean(grad_nF_z)),
                'p50': float(np.percentile(grad_nF_z, 50)),
                'p90': float(np.percentile(grad_nF_z, 90)),
                'p99': float(np.percentile(grad_nF_z, 99)),
            },
            'lnSF6_radial': {
                'mean': float(np.mean(grad_nSF6_r)),
                'p50': float(np.percentile(grad_nSF6_r, 50)),
                'p90': float(np.percentile(grad_nSF6_r, 90)),
                'p99': float(np.percentile(grad_nSF6_r, 99)),
            },
            'lnSF6_axial': {
                'mean': float(np.mean(grad_nSF6_z)),
                'p50': float(np.percentile(grad_nSF6_z, 50)),
                'p90': float(np.percentile(grad_nSF6_z, 90)),
                'p99': float(np.percentile(grad_nSF6_z, 99)),
            },
        },
        'per_case_variance': {
            'lnF_range_across_cases': {
                'mean': float(np.mean(case_nF_range)),
                'std': float(np.std(case_nF_range)),
                'min': float(np.min(case_nF_range)),
                'max': float(np.max(case_nF_range)),
            },
            'lnSF6_range_across_cases': {
                'mean': float(np.mean(case_nSF6_range)),
                'std': float(np.std(case_nSF6_range)),
                'min': float(np.min(case_nSF6_range)),
                'max': float(np.max(case_nSF6_range)),
            },
            'lnF_mean_across_cases': {
                'std': float(np.std(case_nF_mean)),
                'range': float(np.max(case_nF_mean) - np.min(case_nF_mean)),
            },
            'lnSF6_mean_across_cases': {
                'std': float(np.std(case_nSF6_mean)),
                'range': float(np.max(case_nSF6_mean) - np.min(case_nSF6_mean)),
            },
        },
        'cross_correlations': {
            'Te_lnF': {
                'mean': float(np.mean(corr_Te_nF)),
                'std': float(np.std(corr_Te_nF)),
                'min': float(np.min(corr_Te_nF)),
                'max': float(np.max(corr_Te_nF)),
            },
            'Te_lnSF6': {
                'mean': float(np.mean(corr_Te_nSF6)),
                'std': float(np.std(corr_Te_nSF6)),
                'min': float(np.min(corr_Te_nSF6)),
                'max': float(np.max(corr_Te_nSF6)),
            },
            'ne_lnF': {
                'mean': float(np.mean(corr_ne_nF)),
                'std': float(np.std(corr_ne_nF)),
                'min': float(np.min(corr_ne_nF)),
                'max': float(np.max(corr_ne_nF)),
            },
            'ne_lnSF6': {
                'mean': float(np.mean(corr_ne_nSF6)),
                'std': float(np.std(corr_ne_nSF6)),
                'min': float(np.min(corr_ne_nSF6)),
                'max': float(np.max(corr_ne_nSF6)),
            },
        },
        'regime_analysis': {
            'by_power': {
                'nF_between_var': regime_P_nF[0],
                'nF_within_var': regime_P_nF[1],
                'nF_F_ratio': regime_P_nF[0] / max(regime_P_nF[1], 1e-10),
                'nSF6_between_var': regime_P_nSF6[0],
                'nSF6_within_var': regime_P_nSF6[1],
                'nSF6_F_ratio': regime_P_nSF6[0] / max(regime_P_nSF6[1], 1e-10),
            },
            'by_pressure': {
                'nF_between_var': regime_p_nF[0],
                'nF_within_var': regime_p_nF[1],
                'nF_F_ratio': regime_p_nF[0] / max(regime_p_nF[1], 1e-10),
                'nSF6_between_var': regime_p_nSF6[0],
                'nSF6_within_var': regime_p_nSF6[1],
                'nSF6_F_ratio': regime_p_nSF6[0] / max(regime_p_nSF6[1], 1e-10),
            },
            'by_Ar_fraction': {
                'nF_between_var': regime_Ar_nF[0],
                'nF_within_var': regime_Ar_nF[1],
                'nF_F_ratio': regime_Ar_nF[0] / max(regime_Ar_nF[1], 1e-10),
                'nSF6_between_var': regime_Ar_nSF6[0],
                'nSF6_within_var': regime_Ar_nSF6[1],
                'nSF6_F_ratio': regime_Ar_nSF6[0] / max(regime_Ar_nSF6[1], 1e-10),
            },
        },
    }
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)

    print("Loading legacy dataset...", flush=True)
    legacy = load_all_cases(DS_V4)
    print(f"  {len(legacy)} cases", flush=True)

    print("Loading LXCat dataset...", flush=True)
    lxcat = load_all_cases(DS_LX)
    print(f"  {len(lxcat)} cases", flush=True)

    print("Computing legacy stats...", flush=True)
    leg_stats = compute_stats(legacy, 'legacy_v4')

    print("Computing LXCat stats...", flush=True)
    lx_stats = compute_stats(lxcat, 'lxcat_v3')

    # Comparative analysis
    comparison = {
        'legacy': leg_stats,
        'lxcat': lx_stats,
        'differences': {
            'lnF_range_ratio': lx_stats['target_range']['lnF']['range'] / max(leg_stats['target_range']['lnF']['range'], 1e-10),
            'lnSF6_range_ratio': lx_stats['target_range']['lnSF6']['range'] / max(leg_stats['target_range']['lnSF6']['range'], 1e-10),
            'lnF_std_ratio': lx_stats['target_range']['lnF']['std'] / max(leg_stats['target_range']['lnF']['std'], 1e-10),
            'lnSF6_std_ratio': lx_stats['target_range']['lnSF6']['std'] / max(leg_stats['target_range']['lnSF6']['std'], 1e-10),
            'Te_range_ratio': lx_stats['Te']['range'] / max(leg_stats['Te']['range'], 1e-10),
            'Te_std_ratio': lx_stats['Te']['std'] / max(leg_stats['Te']['std'], 1e-10),
            'ne_logrange_ratio': lx_stats['ne']['log_range'] / max(leg_stats['ne']['log_range'], 1e-10),
            'gradient_sharpness_ratio_nF_r': (
                lx_stats['spatial_gradients']['lnF_radial']['p90'] /
                max(leg_stats['spatial_gradients']['lnF_radial']['p90'], 1e-10)
            ),
            'gradient_sharpness_ratio_nF_z': (
                lx_stats['spatial_gradients']['lnF_axial']['p90'] /
                max(leg_stats['spatial_gradients']['lnF_axial']['p90'], 1e-10)
            ),
            'Te_nF_corr_shift': lx_stats['cross_correlations']['Te_lnF']['mean'] - leg_stats['cross_correlations']['Te_lnF']['mean'],
            'Te_nSF6_corr_shift': lx_stats['cross_correlations']['Te_lnSF6']['mean'] - leg_stats['cross_correlations']['Te_lnSF6']['mean'],
            'ne_nF_corr_shift': lx_stats['cross_correlations']['ne_lnF']['mean'] - leg_stats['cross_correlations']['ne_lnF']['mean'],
            'regime_pressure_F_ratio_change': (
                lx_stats['regime_analysis']['by_pressure']['nF_F_ratio'] /
                max(leg_stats['regime_analysis']['by_pressure']['nF_F_ratio'], 1e-10)
            ),
        },
    }

    with open(os.path.join(OUT, 'data_diagnosis.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Generate markdown report
    leg = leg_stats
    lx = lx_stats
    diff = comparison['differences']

    md = f"""# LXCat vs Legacy Dataset Diagnosis

## 1. Target Dynamic Range

| Metric | Legacy | LXCat | Ratio |
|---|---|---|---|
| log10(nF) range | {leg['target_range']['lnF']['range']:.3f} | {lx['target_range']['lnF']['range']:.3f} | {diff['lnF_range_ratio']:.2f}x |
| log10(nF) std | {leg['target_range']['lnF']['std']:.4f} | {lx['target_range']['lnF']['std']:.4f} | {diff['lnF_std_ratio']:.2f}x |
| log10(nSF6) range | {leg['target_range']['lnSF6']['range']:.3f} | {lx['target_range']['lnSF6']['range']:.3f} | {diff['lnSF6_range_ratio']:.2f}x |
| log10(nSF6) std | {leg['target_range']['lnSF6']['std']:.4f} | {lx['target_range']['lnSF6']['std']:.4f} | {diff['lnSF6_std_ratio']:.2f}x |
| Te range (eV) | {leg['Te']['range']:.2f} | {lx['Te']['range']:.2f} | {diff['Te_range_ratio']:.2f}x |
| Te std (eV) | {leg['Te']['std']:.3f} | {lx['Te']['std']:.3f} | {diff['Te_std_ratio']:.2f}x |
| ne log-range | {leg['ne']['log_range']:.2f} | {lx['ne']['log_range']:.2f} | {diff['ne_logrange_ratio']:.2f}x |

## 2. Spatial Gradient Sharpness (P90)

| Gradient | Legacy P90 | LXCat P90 | Ratio |
|---|---|---|---|
| lnF radial | {leg['spatial_gradients']['lnF_radial']['p90']:.3f} | {lx['spatial_gradients']['lnF_radial']['p90']:.3f} | {diff['gradient_sharpness_ratio_nF_r']:.2f}x |
| lnF axial | {leg['spatial_gradients']['lnF_axial']['p90']:.3f} | {lx['spatial_gradients']['lnF_axial']['p90']:.3f} | {diff['gradient_sharpness_ratio_nF_z']:.2f}x |
| lnSF6 radial | {leg['spatial_gradients']['lnSF6_radial']['p90']:.3f} | {lx['spatial_gradients']['lnSF6_radial']['p90']:.3f} | --- |
| lnSF6 axial | {leg['spatial_gradients']['lnSF6_axial']['p90']:.3f} | {lx['spatial_gradients']['lnSF6_axial']['p90']:.3f} | --- |

## 3. Cross-Correlations (mean across cases)

| Correlation | Legacy | LXCat | Shift |
|---|---|---|---|
| Te vs lnF | {leg['cross_correlations']['Te_lnF']['mean']:.3f} | {lx['cross_correlations']['Te_lnF']['mean']:.3f} | {diff['Te_nF_corr_shift']:+.3f} |
| Te vs lnSF6 | {leg['cross_correlations']['Te_lnSF6']['mean']:.3f} | {lx['cross_correlations']['Te_lnSF6']['mean']:.3f} | {diff['Te_nSF6_corr_shift']:+.3f} |
| ne vs lnF | {leg['cross_correlations']['ne_lnF']['mean']:.3f} | {lx['cross_correlations']['ne_lnF']['mean']:.3f} | {diff['ne_nF_corr_shift']:+.3f} |
| ne vs lnSF6 | {leg['cross_correlations']['ne_lnSF6']['mean']:.3f} | {lx['cross_correlations']['ne_lnSF6']['mean']:.3f} | --- |

## 4. Regime Separation (F-ratio = between-var / within-var)

| Grouping | Legacy nF F-ratio | LXCat nF F-ratio |
|---|---|---|
| By power | {leg['regime_analysis']['by_power']['nF_F_ratio']:.3f} | {lx['regime_analysis']['by_power']['nF_F_ratio']:.3f} |
| By pressure | {leg['regime_analysis']['by_pressure']['nF_F_ratio']:.3f} | {lx['regime_analysis']['by_pressure']['nF_F_ratio']:.3f} |
| By Ar fraction | {leg['regime_analysis']['by_Ar_fraction']['nF_F_ratio']:.3f} | {lx['regime_analysis']['by_Ar_fraction']['nF_F_ratio']:.3f} |

## 5. Per-Case Variance

| Metric | Legacy | LXCat |
|---|---|---|
| lnF spatial range (mean) | {leg['per_case_variance']['lnF_range_across_cases']['mean']:.3f} | {lx['per_case_variance']['lnF_range_across_cases']['mean']:.3f} |
| lnSF6 spatial range (mean) | {leg['per_case_variance']['lnSF6_range_across_cases']['mean']:.3f} | {lx['per_case_variance']['lnSF6_range_across_cases']['mean']:.3f} |
| lnF case-mean std | {leg['per_case_variance']['lnF_mean_across_cases']['std']:.4f} | {lx['per_case_variance']['lnF_mean_across_cases']['std']:.4f} |
| lnSF6 case-mean std | {leg['per_case_variance']['lnSF6_mean_across_cases']['std']:.4f} | {lx['per_case_variance']['lnSF6_mean_across_cases']['std']:.4f} |

## 6. Architectural Implications

Based on the diagnosis above, the key differences that affect learnability:

1. **Dynamic range**: How much narrower/wider are the LXCat targets?
2. **Gradient sharpness**: Are LXCat spatial profiles sharper (harder to represent)?
3. **Cross-correlations**: Does LXCat create stronger coupling (harder nonlinearity)?
4. **Regime structure**: Does LXCat create clearer regime separation (warranting gating)?
5. **Per-case variance**: Is the inter-case variability (signal the surrogate must capture) smaller in LXCat (lower SNR)?
"""

    with open(os.path.join(OUT, 'data_diagnosis.md'), 'w') as f:
        f.write(md)

    print(f"\nDiagnosis written to {OUT}/", flush=True)
    print(f"  data_diagnosis.json", flush=True)
    print(f"  data_diagnosis.md", flush=True)

    # Print key findings
    print(f"\n{'='*60}")
    print(f"  KEY FINDINGS")
    print(f"{'='*60}")
    print(f"  lnF range:  legacy={leg['target_range']['lnF']['range']:.3f}  lxcat={lx['target_range']['lnF']['range']:.3f}  ratio={diff['lnF_range_ratio']:.2f}x")
    print(f"  lnF std:    legacy={leg['target_range']['lnF']['std']:.4f}  lxcat={lx['target_range']['lnF']['std']:.4f}  ratio={diff['lnF_std_ratio']:.2f}x")
    print(f"  lnSF6 std:  legacy={leg['target_range']['lnSF6']['std']:.4f}  lxcat={lx['target_range']['lnSF6']['std']:.4f}  ratio={diff['lnSF6_std_ratio']:.2f}x")
    print(f"  Te range:   legacy={leg['Te']['range']:.2f}  lxcat={lx['Te']['range']:.2f}")
    print(f"  ne log-rng: legacy={leg['ne']['log_range']:.2f}  lxcat={lx['ne']['log_range']:.2f}")
    print(f"  Regime F (pressure/nF): legacy={leg['regime_analysis']['by_pressure']['nF_F_ratio']:.3f}  lxcat={lx['regime_analysis']['by_pressure']['nF_F_ratio']:.3f}")
    print(f"  Corr Te-nF: legacy={leg['cross_correlations']['Te_lnF']['mean']:.3f}  lxcat={lx['cross_correlations']['Te_lnF']['mean']:.3f}")
    return comparison


if __name__ == '__main__':
    main()
