"""
Multi-ion species tracker for SF6/Ar ICP.

Resolves SF5+, SF3+, F+, Ar+ as separate species instead of a single
effective ion. Each has its own mass, mobility, and production/loss rates.

The ion composition affects:
  - Mean ion mass (affects Bohm velocity, ambipolar diffusion)
  - Ion flux composition at the wafer (affects etch selectivity)
  - Ion energy distribution (mass-dependent sheath acceleration)

Production channels (from sf6_rates.py):
  SF5+ from iz18 (SF6 + e → SF5+ + F + 2e)
  SF3+ from iz20 (SF6 + e → SF3+ + 3F + 2e) and iz_SF5 channels
  Ar+  from Ar_iz (direct) and Ar_iz_m (stepwise via Ar*)
  F+   from iz28 (F + e → F+ + 2e)

Loss: wall flux (Bohm) and ion-ion recombination with negative ions.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e

AMU = 1.66054e-27

ION_SPECIES = {
    'SF5+': {'mass_amu': 127.06, 'sigma_in': 5e-19},
    'SF3+': {'mass_amu': 89.06,  'sigma_in': 4.5e-19},
    'SF+':  {'mass_amu': 51.06,  'sigma_in': 4e-19},
    'F+':   {'mass_amu': 19.00,  'sigma_in': 3e-19},
    'Ar+':  {'mass_amu': 39.948, 'sigma_in': 5e-19},
}


def compute_ion_fractions(Te_eV, nSF6, nAr, ne, nArm, ng):
    """Compute the fractional composition of positive ions.
    
    Returns dict: {'SF5+': fraction, 'SF3+': fraction, ...}
    Fractions sum to 1.
    """
    from chemistry.sf6_rates import rates
    k = rates(Te_eV)
    
    # Production rates (per unit volume per second)
    R = {}
    
    # SF5+ from SF6
    R['SF5+'] = (k['iz18'] + k['iz19']) * ne * nSF6
    
    # SF3+ from SF6
    R['SF3+'] = (k['iz20'] + k['iz21']) * ne * nSF6
    
    # Other SF_x+ ions
    R['SF+'] = k.get('iz23', 0) * ne * nSF6
    
    # F+ from F atoms (small)
    R['F+'] = k.get('iz28', 0) * ne * 1e15  # rough n_F estimate
    
    # Ar+ from direct + stepwise
    R['Ar+'] = (k['Ar_iz'] * nAr + k['Ar_iz_m'] * nArm) * ne
    
    # Penning
    R['SF5+'] += k.get('Penn_SF6', 0) * nArm * nSF6
    
    total = sum(R.values())
    if total < 1e-30:
        if nAr > nSF6:
            return {'SF5+': 0, 'SF3+': 0, 'SF+': 0, 'F+': 0, 'Ar+': 1.0}
        else:
            return {'SF5+': 0.6, 'SF3+': 0.3, 'SF+': 0.05, 'F+': 0.0, 'Ar+': 0.05}
    
    fractions = {sp: R[sp] / total for sp in R}
    return fractions


def effective_ion_mass(fractions):
    """Compute the mean ion mass from composition [kg]."""
    M = sum(fractions[sp] * ION_SPECIES[sp]['mass_amu'] for sp in fractions if sp in ION_SPECIES)
    return M * AMU


def compute_ion_fractions_2d(mesh, ne, Te_field, nSF6, nAr, nArm, ng):
    """Compute ion composition at every grid point.
    
    Returns dict of arrays: {'SF5+': (Nr,Nz) array of fractions, ...}
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    fracs = {sp: np.zeros((Nr, Nz)) for sp in ION_SPECIES}
    M_eff = np.zeros((Nr, Nz))
    
    for i in range(Nr):
        for j in range(Nz):
            f = compute_ion_fractions(
                Te_field[i,j], nSF6[i,j], nAr, ne[i,j], nArm[i,j], ng)
            for sp in f:
                if sp in fracs:
                    fracs[sp][i,j] = f[sp]
            M_eff[i,j] = effective_ion_mass(f)
    
    return fracs, M_eff


def ion_flux_by_species(ne_edge, Te_eV, fractions, alpha=0.0, T_neg=0.3):
    """Ion flux at the wall, resolved by species [m^-2 s^-1].
    
    Each species has the same drift velocity (ambipolar, single fluid)
    but different mass → different energy.
    """
    from solvers.sheath_model import bohm_velocity, ion_energy_at_wall
    
    fluxes = {}
    energies = {}
    
    # Total flux uses effective mass
    M_eff = effective_ion_mass(fractions)
    uB = bohm_velocity(Te_eV, M_eff, alpha, T_neg=T_neg)
    total_flux = ne_edge * uB
    
    for sp in fractions:
        if sp in ION_SPECIES:
            Mi = ION_SPECIES[sp]['mass_amu'] * AMU
            fluxes[sp] = fractions[sp] * total_flux
            energies[sp] = ion_energy_at_wall(Te_eV, Mi)
    
    return fluxes, energies


if __name__ == '__main__':
    from scipy.constants import k as kB
    ng = 10 * 0.133322 / (kB * 300)
    
    print("Multi-Ion Species Analysis")
    print("=" * 50)
    
    for frac_Ar, label in [(0.0, 'Pure SF6'), (0.3, '70/30 SF6/Ar'), (1.0, 'Pure Ar')]:
        nSF6 = ng * (1 - frac_Ar)
        nAr = ng * frac_Ar
        ne = 6e15
        nArm = ne * 1.5 if frac_Ar > 0 else 0
        
        f = compute_ion_fractions(3.0, nSF6, nAr, ne, nArm, ng)
        M = effective_ion_mass(f)
        
        print(f"\n{label}:")
        for sp in sorted(f.keys(), key=lambda x: -f[x]):
            if f[sp] > 0.01:
                print(f"  {sp:>5s}: {f[sp]*100:5.1f}%")
        print(f"  M_eff = {M/AMU:.1f} AMU")
