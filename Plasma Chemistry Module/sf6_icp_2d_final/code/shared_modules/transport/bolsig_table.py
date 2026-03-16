"""
BOLSIG+ transport table infrastructure.

Provides a BOLSIGTable class that:
  1. Can load real BOLSIG+ output files (when available)
  2. Falls back to Hagelaar-corrected analytical fits
  3. Interpolates all transport coefficients and rate coefficients
     as functions of mean electron energy and gas composition

Usage:
    # With analytical fits (default)
    table = BOLSIGTable.from_analytical(x_Ar=0.3, x_SF6=0.7)
    
    # With real BOLSIG+ output
    table = BOLSIGTable.from_file('bolsig_Ar70_SF30.dat')
    
    # Get transport at a point
    mu_e, D_e, D_eps = table.get_transport(eps_bar=3.0, N=3.2e20)
    k_iz = table.get_rate('ionization', eps_bar=3.0)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import e as eC, m_e, k as kB


class BOLSIGTable:
    """Interpolation table for electron transport and rate coefficients."""
    
    def __init__(self):
        self._eps_grid = None
        self._interp = {}
        self._N_ref = None  # Reference gas density
        self.source = 'uninitialized'
    
    @classmethod
    def from_analytical(cls, x_Ar=0.0, x_SF6=1.0, N=3.22e20):
        """Create table from Hagelaar-corrected analytical fits.
        
        This is the fallback when real BOLSIG+ data is unavailable.
        """
        from transport.hagelaar_transport import transport_mixture
        
        table = cls()
        table._N_ref = N
        table.source = f'analytical (Hagelaar, {x_Ar*100:.0f}%Ar/{x_SF6*100:.0f}%SF6)'
        
        # Build table on a fine energy grid
        eps_grid = np.linspace(0.5, 20.0, 100)
        Te_grid = eps_grid / 1.5
        
        mu_e_vals = np.zeros_like(eps_grid)
        D_e_vals = np.zeros_like(eps_grid)
        mu_eps_vals = np.zeros_like(eps_grid)
        D_eps_vals = np.zeros_like(eps_grid)
        
        for k, Te in enumerate(Te_grid):
            r = transport_mixture(max(Te, 0.3), N, x_Ar, x_SF6)
            mu_e_vals[k] = r['mu_e']
            D_e_vals[k] = r['D_e']
            mu_eps_vals[k] = r['mu_eps']
            D_eps_vals[k] = r['D_eps']
        
        table._eps_grid = eps_grid
        kw = dict(kind='linear', bounds_error=False, fill_value='extrapolate')
        table._interp['mu_e'] = interp1d(eps_grid, mu_e_vals * N, **kw)
        table._interp['D_e'] = interp1d(eps_grid, D_e_vals * N, **kw)
        table._interp['mu_eps'] = interp1d(eps_grid, mu_eps_vals * N, **kw)
        table._interp['D_eps'] = interp1d(eps_grid, D_eps_vals * N, **kw)
        
        # Rate coefficients from Arrhenius fits (placeholder)
        from chemistry.sf6_rates import rates
        rate_names = ['Ar_iz', 'Ar_exc', 'Ar_iz_m', 'iz_SF6_total', 'att_SF6_total',
                      'd1', 'd2', 'd3', 'd4', 'd5']
        for name in rate_names:
            k_vals = np.array([rates(max(Te, 0.3)).get(name, 0) for Te in Te_grid])
            table._interp[f'k_{name}'] = interp1d(eps_grid, k_vals, **kw)
        
        return table
    
    @classmethod
    def from_file(cls, filepath):
        """Load from a BOLSIG+ output file.
        
        Expected format: whitespace-separated columns:
          E/N(Td)  eps_bar(eV)  mu_e*N  D_e*N  mu_eps*N  D_eps*N  k_iz  k_att ...
        
        First line: header with column names.
        """
        table = cls()
        table.source = f'BOLSIG+ file: {filepath}'
        
        data = np.loadtxt(filepath, skiprows=1)
        eps_grid = data[:, 1]
        table._eps_grid = eps_grid
        
        kw = dict(kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Standard BOLSIG+ columns
        col_map = {
            'mu_e': 2, 'D_e': 3, 'mu_eps': 4, 'D_eps': 5,
        }
        for name, col in col_map.items():
            if col < data.shape[1]:
                table._interp[name] = interp1d(eps_grid, data[:, col], **kw)
        
        # Rate coefficient columns (6 onwards)
        # Would need header parsing for specific names
        for col in range(6, min(data.shape[1], 20)):
            table._interp[f'k_col{col}'] = interp1d(eps_grid, data[:, col], **kw)
        
        return table
    
    def get_transport(self, eps_bar, N):
        """Get all transport coefficients at given mean energy.
        
        Parameters
        ----------
        eps_bar : float — mean electron energy [eV]
        N : float — gas number density [m^-3]
        
        Returns
        -------
        dict: mu_e, D_e, mu_eps, D_eps [SI units]
        """
        eps = np.clip(eps_bar, self._eps_grid[0], self._eps_grid[-1])
        return {
            'mu_e': float(self._interp['mu_e'](eps)) / N,
            'D_e': float(self._interp['D_e'](eps)) / N,
            'mu_eps': float(self._interp['mu_eps'](eps)) / N,
            'D_eps': float(self._interp['D_eps'](eps)) / N,
        }
    
    def get_rate(self, name, eps_bar):
        """Get a rate coefficient at given mean energy.
        
        Parameters
        ----------
        name : str — rate name (e.g., 'Ar_iz', 'att_SF6_total')
        eps_bar : float — mean electron energy [eV]
        
        Returns
        -------
        float — rate coefficient [m^3/s]
        """
        key = f'k_{name}'
        if key not in self._interp:
            raise KeyError(f"Rate '{name}' not in table. Available: "
                          f"{[k[2:] for k in self._interp if k.startswith('k_')]}")
        eps = np.clip(eps_bar, self._eps_grid[0], self._eps_grid[-1])
        return float(self._interp[key](eps))
    
    def get_all_rates(self, eps_bar):
        """Get all rate coefficients at given mean energy."""
        eps = np.clip(eps_bar, self._eps_grid[0], self._eps_grid[-1])
        return {k[2:]: float(v(eps)) for k, v in self._interp.items() if k.startswith('k_')}


if __name__ == '__main__':
    N = 10 * 0.133322 / (kB * 300)
    
    print("BOLSIG+ Table Infrastructure Test")
    print("=" * 50)
    
    for x_Ar, x_SF6, label in [(0.0, 1.0, 'Pure SF6'), (0.3, 0.7, '70/30'), (1.0, 0.0, 'Pure Ar')]:
        table = BOLSIGTable.from_analytical(x_Ar=x_Ar, x_SF6=x_SF6, N=N)
        print(f"\n{label} (source: {table.source}):")
        for eps in [3.0, 4.5, 7.5]:
            t = table.get_transport(eps, N)
            print(f"  eps={eps:.1f} eV: mu_e={t['mu_e']:.0f}, D_e={t['D_e']:.0f}, "
                  f"D_eps={t['D_eps']:.0f}, D_eps/D_e={t['D_eps']/t['D_e']:.2f}")
