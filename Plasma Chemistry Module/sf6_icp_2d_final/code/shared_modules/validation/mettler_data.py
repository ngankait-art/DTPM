"""
Mettler Dissertation Experimental Data — Digitized for Validation

Source: J.J.H. Mettler, PhD Dissertation, UIUC (2025)
"Spatially Resolved Probes for the Measurement of Fluorine Radicals"

CRITICAL NOTES:
- Reactor is a proprietary TEL ICP etcher (geometry not fully specified)
- ICP frequency is 2 MHz (NOT 13.56 MHz like the Lallement reactor)
- No ne, Te, or alpha data available (no Langmuir probe measurements)
- Validation is limited to radial [F] profiles and Si etch rates
- Focus on RELATIVE profile shapes (normalized), not absolute values,
  since reactor geometries differ
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4.14 — Normalized F density profile
# 1000W ICP (2 MHz), 10 mTorr, 70/30 SF6/Ar, 200W bias (40 MHz)
# Measured ~1 cm above wafer using W/Al radical probes
# ═══════════════════════════════════════════════════════════════════

fig414_r_cm    = np.array([0.0, 1.5, 3.5, 6.0, 8.0])
fig414_nF_norm = np.array([1.00, 0.97, 0.80, 0.50, 0.25])

# Cubic fit from figure inset: y = A + C*r² + D*r³ (B fixed at 0)
fig414_A = 1.01032
fig414_C = -0.01847
fig414_D = 7.13914e-4
fig414_R2 = 0.99703

def fig414_fit(r_cm):
    """Normalized F density from cubic fit in Figure 4.14.
    Valid for 0 ≤ r ≤ 8 cm."""
    return fig414_A + fig414_C * r_cm**2 + fig414_D * r_cm**3

# Absolute center density
nF_center_414 = 2.5e20   # m⁻³ (corrected via Eq 4.2)
nF_act_avg_414 = 1.54e20  # m⁻³ (actinometry line-average)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4.17 — Absolute F density + Si etch rate (4 conditions)
# All: 1000W ICP (2 MHz), 10 mTorr, 100 sccm total, ~1 cm above wafer
# ═══════════════════════════════════════════════════════════════════

r_417 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# Condition 1: 90% SF6 (90 sccm SF6, 10 sccm Ar), Bias OFF
nF_90off   = np.array([1.70, 1.60, 1.50, 1.35, 1.10, 0.85, 0.65, 0.50, 0.40]) * 1e20
etch_90off = np.array([22.0, 21.5, 20.0, 18.0, 15.0, 12.0, 10.0,  8.0,  6.0])

# Condition 2: 30% SF6 (30 sccm SF6, 70 sccm Ar), Bias OFF
nF_30off   = np.array([0.50, 0.48, 0.42, 0.35, 0.28, 0.22, 0.15, 0.12, 0.10]) * 1e20
etch_30off = np.array([ 5.5,  5.2,  4.8,  4.0,  3.5,  3.0,  2.5,  2.0,  1.5])

# Condition 3: 90% SF6, Bias ON (200W at 40 MHz)
nF_90on    = np.array([2.80, 2.70, 2.50, 2.20, 1.80, 1.40, 1.05, 0.75, 0.55]) * 1e20
etch_90on  = np.array([32.0, 31.0, 29.0, 26.0, 22.0, 18.0, 15.0, 12.0,  9.0])

# Condition 4: 30% SF6, Bias ON (200W at 40 MHz)
nF_30on    = np.array([1.05, 1.00, 0.90, 0.80, 0.60, 0.40, 0.28, 0.18, 0.12]) * 1e20
etch_30on  = np.array([12.0, 11.5, 10.5,  9.5,  7.5,  5.5,  4.0,  3.0,  2.0])


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4.9(b) — F density in ICP source region vs SF6 flow
# ═══════════════════════════════════════════════════════════════════

sf6_flow_sccm = np.array([10, 20, 30, 50, 70, 90])
nF_icp_20mTorr_600W = np.array([1.8e20, 4.0e20, 5.5e20, 8.5e20, 1.05e21, 1.2e21])
nF_icp_40mTorr_700W = np.array([3.5e20, 7.0e20, 1.1e21, 1.6e21, 2.0e21, 2.35e21])


# ═══════════════════════════════════════════════════════════════════
# REACTOR PARAMETERS (partial — proprietary TEL tool)
# ═══════════════════════════════════════════════════════════════════

ICP_freq_MHz = 2.0          # NOT 13.56 MHz
bias_freq_MHz = 40.0
ICP_to_wafer_cm = 10.0
probe_above_wafer_cm = 1.0
T_gas_ICP_K = 573.0         # ~300°C
T_gas_wafer_K = 298.0       # ~25°C
K_actinometry = 4.1
si_etch_prob_radical_probe = 0.025
si_etch_prob_actinometry = 0.08


if __name__ == '__main__':
    # Quick summary
    print("Mettler Validation Data Summary")
    print("="*50)
    print(f"\nFig 4.14: Normalized [F](r) at 1000W, 10mT, 70/30 SF6/Ar")
    print(f"  Center: {fig414_fit(0):.3f}, Edge(8cm): {fig414_fit(8):.3f}")
    print(f"  Drop: {(1-fig414_fit(8)/fig414_fit(0))*100:.0f}%")
    print(f"  Absolute center: {nF_center_414:.2e} m⁻³ = {nF_center_414*1e-6:.2e} cm⁻³")
    print(f"\nFig 4.17: Absolute [F](r=0)")
    print(f"  90% SF6, no bias: {nF_90off[0]:.2e} m⁻³")
    print(f"  30% SF6, no bias: {nF_30off[0]:.2e} m⁻³")
    print(f"  90% SF6, 200W bias: {nF_90on[0]:.2e} m⁻³")
    print(f"  30% SF6, 200W bias: {nF_30on[0]:.2e} m⁻³")
    print(f"\nNOTE: TEL reactor uses 2 MHz ICP, not 13.56 MHz")
    print(f"      Compare PROFILE SHAPES, not absolute values")
