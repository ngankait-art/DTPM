"""
tel_mapping.py — TEL Etcher reactor mapping and parameter calculations
======================================================================
Explicit calculations for mapping the Lallement-calibrated model to TEL geometry.
TEL geometry from supervisor spec sheet. Measurement context from Mettler (2025).
"""

import numpy as np
from transport import Reactor, LALLEMENT_REACTOR, residence_time, gas_density, kB

MTORR_TO_PA = 0.13332

# ============================================================
# TEL geometry from supervisor spec sheet
# ============================================================
TEL_SPEC = {
    'quartz_cylinder_radius': 0.038,        # m
    'icp_coil_radius': 0.0405,              # m
    'icp_coil_length': 0.088,               # m
    'coil_turns': 10,
    'hf_source_freq_MHz': 40,
    'hf_source_power_W': 700,
    'lf_source_freq_MHz': 13,
    'lf_source_power_W': 700,
    'assumed_coil_impedance_ohm': 50,
    'wafer_size_inch': 6,
    'processing_region_ID_m': 0.210,
    'processing_region_OD_m': 0.238,
    'processing_region_height_m': 0.0535,
    'icp_to_wafer_separation_m': 0.0975,
    'tmp_speed_Ar_Ls': 2500,
    # Flow-pressure anchor points
    'flow_pressure_anchors': [
        (200, 9),     # (sccm, mTorr)
        (1200, 50),   # (sccm, mTorr)
    ],
}

# Mettler measurement context
METTLER_CONTEXT = {
    'icp_source_freq_MHz': 2,
    'icp_power_range_kW': (0, 3),
    'base_pressure_Torr': 3e-6,
    'probe_icp_position': 'center of ICP source (axial)',
    'probe_wafer_position': '~1 cm above wafer, ~10 cm below ICP bottom',
    'probe_radial_travel_cm': 8,
    'radial_nonuniformity': 'density drops to ~25% of center at wafer edge',
    'actinometry_note': 'line-averaged (chord through center), not local measurement',
    'actinometry_probe_separation': 'measured separately due to limited ICP access',
    'transport_limit_m3': 1e21,  # probe saturates above this F density
    'bias_effect': 'expands plasma volume, generates F locally (not ion-enhanced etching)',
}


def tel_effective_geometry():
    """Compute effective 0D geometry approximation for TEL reactor.
    
    Uses simple cylindrical approximation of the processing region.
    
    Returns
    -------
    dict with geometry parameters and Reactor object
    """
    R_eff = TEL_SPEC['processing_region_ID_m'] / 2  # 0.105 m
    L_eff = TEL_SPEC['processing_region_height_m']   # 0.0535 m
    
    reactor = Reactor(R=R_eff, L=L_eff, name='TEL effective')
    
    wafer_radius = TEL_SPEC['wafer_size_inch'] * 0.0254 / 2  # 6" -> 0.0762 m
    A_wafer = np.pi * wafer_radius**2
    A_top = np.pi * R_eff**2
    A_side = 2 * np.pi * R_eff * L_eff
    
    return {
        'reactor': reactor,
        'R_eff': R_eff,
        'L_eff': L_eff,
        'V_eff': reactor.V,
        'A_total': reactor.A,
        'A_wafer': A_wafer,
        'A_top': A_top,
        'A_side': A_side,
        'Lambda': reactor.Lambda,
        'limitation': ('Collapsing a two-region system (ICP source + processing region) '
                      'into a single 0D zone cannot capture the ~10 cm transport gap '
                      'between the ICP source and wafer surface.'),
    }


def compute_residence_times():
    """Compute residence times for representative TEL and Lallement cases.
    
    Returns
    -------
    dict of dicts with (pressure_mTorr, flow_sccm) -> tau_R in seconds
    """
    tel_geom = tel_effective_geometry()
    tel_V = tel_geom['V_eff']
    lal_V = LALLEMENT_REACTOR.V
    
    cases = {
        'TEL': [
            (10, 100),   # Mettler Fig 4.17 conditions
            (20, 100),   # Mettler Fig 4.9b
            (40, 100),   # Mettler Fig 4.9b
            (50, 40),    # Mettler Fig 4.5
        ],
        'Lallement': [
            (10, 40),    # Lallement standard conditions
            (5, 40),
            (20, 40),
        ],
    }
    
    results = {}
    for reactor_name, case_list in cases.items():
        V = tel_V if reactor_name == 'TEL' else lal_V
        for p_mTorr, Q_sccm in case_list:
            p_Pa = p_mTorr * MTORR_TO_PA
            tau = residence_time(p_Pa, V, Q_sccm)
            key = f"{reactor_name}_{p_mTorr}mTorr_{Q_sccm}sccm"
            results[key] = {
                'reactor': reactor_name,
                'p_mTorr': p_mTorr,
                'Q_sccm': Q_sccm,
                'V_m3': V,
                'tau_R_s': tau,
                'tau_R_ms': tau * 1000,
            }
    
    return results


def tel_eta_analysis():
    """Analyze power coupling efficiency for TEL vs Lallement.
    
    Returns
    -------
    dict with eta estimates and justification
    """
    return {
        'eta_lallement': 0.12,
        'eta_lallement_note': ('Calibrated to reproduce ne at pure SF6, 10 mTorr. '
                              'Represents power coupling in Lallement ICP (13.56 MHz).'),
        'eta_tel_range': (0.05, 0.20),
        'eta_tel_note': ('TEL operates at 2 MHz with proprietary ICP design. '
                        'Lower frequency generally gives lower coupling efficiency. '
                        'No single eta_tel can be justified without independent ne measurement. '
                        'We bracket with sensitivity study rather than fitting.'),
        'eta_tel_anchor': 0.12,
        'anchor_note': ('Using eta=0.12 as baseline (same as Lallement) and noting '
                       'that any absolute density offset may be attributed to this uncertainty.'),
    }


def measurement_interpretation_notes():
    """Return detailed notes on TEL measurement interpretation.
    
    These are essential caveats for comparing a 0D global model against
    spatially resolved experimental data from a two-region reactor.
    """
    return {
        'actinometry': (
            'Actinometry measures a line-averaged optical emission intensity ratio. '
            'In the TEL, the spectrometer integrates over a chord through the chamber. '
            'The resulting density estimate is: n_act = (n_0/R) * integral(f(r)dr, 0, R) '
            'where f(r) is the normalized radial profile. '
            'This is NOT equivalent to the center-of-wafer value measured by the radical probe. '
            'Mettler (2025) found the radial density drops to ~25% of center at the wafer edge.'
        ),
        'radical_probe': (
            'W/Al radical probes measure local etch rate at a single point. '
            'The F density is inferred from the etch rate using calibrated tungsten etch kinetics. '
            'The measurement is position-dependent: ICP center gives ~10x higher density than '
            'above-wafer position due to the ~10 cm transport gap between ICP source and wafer.'
        ),
        'probe_perturbation': (
            'Actinometry and radical probe measurements were conducted separately for each '
            'condition due to limited access to the ICP region. The probe itself consumes '
            'fluorine radicals, so any reduction in radical density from probe presence is not '
            'reflected in the actinometry measurements.'
        ),
        'transport_limitation': (
            'The radical probe saturates above ~1e21 m^-3 F density due to transport limitations '
            '(the etch rate becomes diffusion-limited rather than kinetically limited). '
            'Data above this threshold should be interpreted with caution.'
        ),
        'bias_effect': (
            'Wafer stage bias (200 W rf) increases measured F density by 1.6-2.15x. '
            'This is NOT ion-enhanced etching; temperature dependence shows a linear offset '
            'consistent with expanded plasma volume generating additional F radicals locally '
            'in the wafer region.'
        ),
        'global_vs_local': (
            'A 0D global model predicts a volume-averaged density. Comparing this against '
            'center-of-wafer radical probe data overestimates model agreement at center and '
            'underestimates it at the edge. The comparison is inherently qualitative.'
        ),
    }


def print_tel_summary():
    """Print formatted summary of TEL mapping calculations."""
    geom = tel_effective_geometry()
    taus = compute_residence_times()
    eta = tel_eta_analysis()
    
    print("=" * 70)
    print("TEL REACTOR MAPPING SUMMARY")
    print("=" * 70)
    
    print(f"\n--- Effective Geometry ---")
    print(f"  R_eff = {geom['R_eff']*100:.1f} cm")
    print(f"  L_eff = {geom['L_eff']*100:.2f} cm")
    print(f"  V_eff = {geom['V_eff']*1e6:.1f} cm³ = {geom['V_eff']*1e3:.3f} L")
    print(f"  A_total = {geom['A_total']*1e4:.0f} cm²")
    print(f"  A_wafer = {geom['A_wafer']*1e4:.0f} cm²")
    print(f"  Lambda = {geom['Lambda']*100:.2f} cm")
    print(f"  Limitation: {geom['limitation']}")
    
    print(f"\n--- Lallement Reference ---")
    lal = LALLEMENT_REACTOR
    print(f"  R = {lal.R*100:.1f} cm, L = {lal.L*100:.1f} cm")
    print(f"  V = {lal.V*1e6:.0f} cm³ = {lal.V*1e3:.2f} L")
    print(f"  A = {lal.A*1e4:.0f} cm²")
    print(f"  V ratio (TEL/Lal) = {geom['V_eff']/lal.V:.3f}")
    
    print(f"\n--- Residence Times ---")
    for key, data in sorted(taus.items()):
        print(f"  {data['reactor']:>10} {data['p_mTorr']:>3} mTorr, {data['Q_sccm']:>4} sccm: "
              f"tau = {data['tau_R_ms']:.1f} ms")
    
    print(f"\n--- Power Coupling ---")
    print(f"  eta_Lallement = {eta['eta_lallement']}")
    print(f"  eta_TEL range = {eta['eta_tel_range']}")
    print(f"  Approach: {eta['eta_tel_note'][:80]}...")
    
    print(f"\n--- Neutral Density at 10 mTorr, 300 K ---")
    p_Pa = 10 * MTORR_TO_PA
    ng = gas_density(p_Pa)
    print(f"  n_g = {ng:.2e} m^-3 = {ng*1e-6:.2e} cm^-3")


if __name__ == '__main__':
    print_tel_summary()
