"""
NF3/Ar ICP Global Model Solver
================================
Implements the 0D global model from:
  Huang et al. 2026, Plasma Sources Sci. Technol. 35, 015019

Solves time-dependent ODEs for species densities and electron energy
until steady state is reached.
"""

import numpy as np
from scipy.integrate import solve_ivp
from reactions import (REACTIONS, NEUTRAL_SPECIES, POSITIVE_IONS, NEGATIVE_IONS,
                       ALL_SPECIES, SPECIES_MASS, SURFACE_RECOMBINATION,
                       THREE_BODY_DATA)

# Physical constants
e_charge = 1.602e-19   # C
k_B = 1.381e-23        # J/K
m_e = 9.109e-31        # kg
epsilon_0 = 8.854e-12  # F/m
eV_to_J = 1.602e-19
eV_to_K = 11604.5

# =============================================================================
# Reactor geometry and operating conditions
# =============================================================================
class ReactorConfig:
    """ICP reactor configuration from Huang et al. 2026."""
    def __init__(self, R=0.21, L=0.06, P_abs=600.0, pressure_mTorr=30.0,
                 flow_Ar_sccm=80.0, flow_NF3_sccm=20.0, Twall=300.0):
        self.R = R              # Chamber radius [m]
        self.L = L              # Chamber height [m]
        self.P_abs = P_abs      # Absorbed power [W]
        self.pressure_mTorr = pressure_mTorr
        self.pressure_Pa = pressure_mTorr * 0.1333
        self.flow_Ar = flow_Ar_sccm    # sccm
        self.flow_NF3 = flow_NF3_sccm  # sccm
        self.flow_total = flow_Ar_sccm + flow_NF3_sccm
        self.Twall = Twall      # Wall temperature [K]
        
        # Derived geometry
        self.V = np.pi * R**2 * L                      # Volume [m^3]
        self.A = 2 * np.pi * R**2 + 2 * np.pi * R * L  # Total wall area [m^2]
        
        # Substrate area (radius 5 cm = 0.05 m from paper)
        self.R_sub = 0.05


# =============================================================================
# Gas temperature model (Thorsteinsson Eq. 8-9)
# =============================================================================
def gas_temperature(P_abs, pressure_mTorr):
    """Compute gas temperature from empirical formula.
    Tg = 300 + s(p) * log10(P_abs / 40)
    s(p) = 780*(1 - exp(-0.091*p)) + 250*exp(-0.337*p)
    where p is in mTorr and P_abs in W.
    """
    p = pressure_mTorr
    s = 780.0 * (1.0 - np.exp(-0.091 * p)) + 250.0 * np.exp(-0.337 * p)
    Tg = 300.0 + s * np.log10(max(P_abs / 40.0, 1.01))
    return max(Tg, 300.0)


# =============================================================================
# Ion/negative ion temperature (Eq. 24)
# =============================================================================
def ion_temperature(pressure_mTorr, Tg):
    """Ion temperature model from Lee & Lieberman."""
    if pressure_mTorr <= 1.0:
        return 0.5 * eV_to_K  # 0.5 eV in Kelvin
    else:
        return (0.5 * eV_to_K - Tg) / pressure_mTorr + Tg


# =============================================================================
# Diffusion and wall loss for neutrals (Eq. 11-14)
# =============================================================================
def neutral_wall_loss_rate(species, n_dict, config, Tg):
    """Compute wall loss rate for neutral species.
    kloss = [Lambda^2/D + 2V(2-gamma)/(A*v*gamma)]^{-1}
    Returns: loss rate coefficient [s^{-1}] to multiply by n_species
    """
    gamma = SURFACE_RECOMBINATION.get(species, 0)
    if gamma <= 0:
        return 0.0
    
    m = SPECIES_MASS.get(species, 40.0 * 1.6605e-27)
    T_K = Tg
    R, L = config.R, config.L
    
    # Thermal velocity (Eq. 14)
    v_th = np.sqrt(8.0 * k_B * T_K / (np.pi * m))
    
    # Diffusion length (Eq. 12)
    Lambda2 = 1.0 / ((np.pi / L)**2 + (2.405 / R)**2)
    
    # Mean free path: lambda = 1 / (sum_i n_i * sigma_i)
    # Use Lennard-Jones cross section ~ 3e-19 m^2 for neutral-neutral
    sigma_nn = 3e-19  # m^2
    n_total = sum(n_dict.get(s, 0) for s in NEUTRAL_SPECIES)
    n_total = max(n_total, 1e15)
    lambda_mfp = 1.0 / (n_total * sigma_nn)
    
    # Free diffusion coefficient (Eq. 13): D = kT*lambda/(m*v_th)
    D = k_B * T_K * lambda_mfp / (m * v_th) if v_th > 0 else 1.0
    
    # Wall loss rate
    term1 = Lambda2 / D
    term2 = 2.0 * config.V * (2.0 - gamma) / (config.A * v_th * gamma)
    
    k_loss = 1.0 / (term1 + term2)
    return k_loss


# =============================================================================
# Bohm velocity and effective area for ion wall loss (Eqs. 15-23)
# =============================================================================
def compute_ion_wall_loss(n_dict, Te, Tg, config):
    """Compute ion wall loss parameters.
    Returns dict of {ion_species: loss_rate [s^{-1}]} for each positive ion.
    """
    ne = n_dict.get('e', 1e14)
    n_neg = sum(n_dict.get(s, 0) for s in NEGATIVE_IONS)
    
    # Electronegativity
    alpha = n_neg / max(ne, 1e6)
    
    # Ion and negative ion temperatures
    T_ion_K = ion_temperature(config.pressure_mTorr, Tg)
    T_ion_eV = T_ion_K / eV_to_K
    T_neg_eV = T_ion_eV  # Same as positive ions per Eq. 24
    
    # Temperature ratios
    gamma_plus = Te / max(T_ion_eV, 0.01)
    gamma_minus = Te / max(T_neg_eV, 0.01)
    
    # Electronegativity at sheath boundary (Eq. 23, solved iteratively)
    alpha_s = alpha  # First approximation
    for _ in range(50):
        if gamma_minus <= 1.0001 or alpha < 1e-6:
            alpha_s = alpha
            break
        exponent = 0.5 * (1.0 + alpha_s) * (gamma_minus - 1.0) / max(1.0 + gamma_minus * alpha_s, 0.01)
        alpha_new = alpha / max(np.exp(exponent), 1e-30)
        alpha_new = max(alpha_new, 0.0)
        if abs(alpha_new - alpha_s) < 1e-6 * max(alpha_s, 1e-10):
            alpha_s = alpha_new
            break
        alpha_s = 0.7 * alpha_s + 0.3 * alpha_new  # Relaxed update
    alpha_s = max(alpha_s, 0.0)
    
    R, L = config.R, config.L
    
    # Mean free path for ions (ion-neutral collisions)
    # For Ar+ in Ar: charge-exchange cross section ~5e-19 m^2
    # For mixed gases, use similar value as estimate
    sigma_in = 5e-19  # m^2, ion-neutral collision cross section
    n_total = sum(n_dict.get(s, 0) for s in NEUTRAL_SPECIES)
    n_total = max(n_total, 1e15)
    lambda_i = 1.0 / (n_total * sigma_in)
    
    loss_rates = {}
    for ion in POSITIVE_IONS:
        m_ion = SPECIES_MASS.get(ion, 40.0 * 1.6605e-27)
        
        # Bohm velocity (Eq. 22)
        uB = np.sqrt(eV_to_J * Te / m_ion * (1.0 + alpha_s) / max(1.0 + gamma_minus * alpha_s, 0.01))
        
        # Free ion diffusion coefficient: D+ = kT+ * lambda_i / (m_ion * v_th_ion)
        # where v_th_ion = sqrt(8*kT+/(pi*m))
        v_th_ion = np.sqrt(8.0 * eV_to_J * T_ion_eV / (np.pi * m_ion))
        D_plus = eV_to_J * T_ion_eV * lambda_i / (m_ion * v_th_ion) if v_th_ion > 0 else 1.0
        
        # Ambipolar diffusion coefficient (Eq. 21)
        Da = D_plus * (1.0 + gamma_plus + gamma_plus * alpha_s) / max(1.0 + gamma_plus * alpha_s, 0.01)
        
        alpha_0 = 1.5 * alpha  # approx
        
        # eta factor
        eta = 2.0 * T_ion_eV / max(T_ion_eV + T_neg_eV, 0.001)
        
        # hL0 (Eq. 18) - axial edge-to-center density ratio
        term_hL = 3.0 + eta * L / (2.0 * max(lambda_i, 1e-6)) + (0.86 * L * uB / (np.pi * max(Da, 1e-6)))**2
        hL0 = 0.86 / (1.0 + alpha_0) * term_hL**(-0.5)
        
        # hR0 (Eq. 19) - radial edge-to-center density ratio
        chi01 = 2.405
        J1_chi01 = 0.5191  # J1(2.405)
        term_hR = 4.0 + eta * R / max(lambda_i, 1e-6) + (0.8 * R * uB / (chi01 * J1_chi01 * max(Da, 1e-6)))**2
        hR0 = 0.8 / (1.0 + alpha_0) * term_hR**(-0.5)
        
        # hc correction for electronegative plasma (Eq. 20)
        # hc ~ 1 / (gamma_minus^0.5 + gamma_plus^0.5 * (n*^0.5 * n+ / n-^1.5))
        # For highly electronegative plasmas, this correction is important
        if alpha > 0.1:
            # Estimate n* from Eq in paper: n* = 15*eta^2*v_neg / (56*k_rec*lambda_i)
            v_neg = np.sqrt(8.0 * eV_to_J * T_neg_eV / (np.pi * (19.0 * 1.6605e-27)))
            k_rec = 2e-13  # typical ion-ion recombination rate m^3/s
            n_star = 15.0 * eta**2 * v_neg / (56.0 * k_rec * max(lambda_i, 1e-6))
            
            n_plus = sum(n_dict.get(s, 0) for s in POSITIVE_IONS)
            n_minus = n_neg
            if n_minus > 0 and n_plus > 0:
                ratio = n_star**0.5 * n_plus / max(n_minus**1.5, 1e10)
                hc = 1.0 / (gamma_minus**0.5 + gamma_plus**0.5 * ratio)
            else:
                hc = 0.0
            hc = min(hc, 0.5)  # Physical bound
        else:
            hc = 0.0
        
        hL = np.sqrt(hL0**2 + hc**2)
        hR = np.sqrt(hR0**2 + hc**2)
        
        # Clamp h-factors to physical range
        hL = min(hL, 0.5)
        hR = min(hR, 0.5)
        
        # Effective area (Eq. 16)
        Aeff = 2.0 * np.pi * (R**2 * hL + R * L * hR)
        
        # Loss rate (Eq. 15)
        loss_rates[ion] = uB * Aeff / config.V
    
    return loss_rates, alpha, alpha_s


# =============================================================================
# Power balance energy losses (Eqs. 25-28)
# =============================================================================
def compute_wall_energy_loss(Te, alpha_s, Tg, config):
    """Compute energy lost per ion-electron pair at the wall.
    Returns (epsilon_e + epsilon_p + epsilon_s) in eV.
    """
    T_ion_K = ion_temperature(config.pressure_mTorr, Tg)
    T_neg_eV = T_ion_K / eV_to_K
    gamma_minus = Te / max(T_neg_eV, 0.01)
    
    # epsilon_e: average kinetic energy lost by electron (Eq. 26)
    eps_e = 2.0 * Te
    
    # epsilon_p: presheath drop (Eq. 27)
    eps_p = (1.0 + alpha_s) / (2.0 + 2.0 * alpha_s * gamma_minus) * Te
    
    # epsilon_s: sheath drop (Eq. 28)
    # Simplified version
    ve = np.sqrt(8.0 * eV_to_J * Te / (np.pi * m_e))
    # Average Bohm velocity (use Ar+ mass as representative)
    m_avg = 40.0 * 1.6605e-27
    uB_avg = np.sqrt(eV_to_J * Te / m_avg * (1 + alpha_s) / (1 + gamma_minus * alpha_s))
    
    v_neg = np.sqrt(8.0 * eV_to_J * T_neg_eV / (np.pi * (19.0 * 1.6605e-27)))
    
    ratio = 4.0 * uB_avg / ve * (1 + alpha_s) / (1 + alpha_s * (v_neg / ve)**2)
    eps_s = np.log(max(ratio, 1.01)) * Te
    
    return eps_e + eps_p + eps_s


# =============================================================================
# Pumping rates (Eq. 10)
# =============================================================================
def pumping_rate(species, n, config, n_total, Tg):
    """Compute pump-in and pump-out rates for a species.
    Returns dn/dt contribution [m^{-3} s^{-1}].
    """
    V = config.V
    
    # Pump in: only Ar and NF3
    Q_in = 0.0
    if species == 'Ar':
        Q_in = config.flow_Ar
    elif species == 'NF3':
        Q_in = config.flow_NF3
    
    pump_in = 4.48e17 * Q_in / V  # sccm to s^{-1}/m^3
    
    # Pump out: all neutral species and positive ions
    if species in NEUTRAL_SPECIES or species in POSITIVE_IONS:
        Q_out = config.flow_total  # sccm
        P_Torr = config.pressure_mTorr * 1e-3
        
        # ccc parameter: adjusted self-consistently
        # For now, estimate from ideal gas law
        # P*V = N*k*T => P = n_total * k_B * Tg (in Pa)
        P_Pa_ideal = n_total * k_B * Tg
        P_Pa_set = config.pressure_Pa
        ccc = P_Pa_ideal / max(P_Pa_set, 0.01)
        ccc = max(ccc, 0.1)
        
        pump_out = 1.27e-5 * n * Q_out / (ccc * P_Torr * V)
    else:
        pump_out = 0.0
    
    return pump_in - pump_out


# =============================================================================
# Main ODE system
# =============================================================================
class NF3GlobalModel:
    """0D global model for Ar/NF3 ICP."""
    
    def __init__(self, config):
        self.config = config
        self.species_idx = {s: i for i, s in enumerate(ALL_SPECIES)}
        self.n_species = len(ALL_SPECIES)
        # Index for electron energy: last variable
        self.idx_Te = self.n_species  
        
    def initial_conditions(self):
        """Set initial densities from feed gas composition."""
        cfg = self.config
        Tg = gas_temperature(cfg.P_abs, cfg.pressure_mTorr)
        
        # Total gas density from ideal gas law
        n_total = cfg.pressure_Pa / (k_B * Tg)
        
        # Initial composition: Ar and NF3 only
        frac_Ar = cfg.flow_Ar / cfg.flow_total
        frac_NF3 = cfg.flow_NF3 / cfg.flow_total
        
        y0 = np.zeros(self.n_species + 1)
        y0[self.species_idx['Ar']] = frac_Ar * n_total
        y0[self.species_idx['NF3']] = frac_NF3 * n_total
        
        # Seed electron density and ions
        ne0 = 1e15  # 10^15 m^{-3} seed
        y0[self.species_idx['e']] = ne0
        y0[self.species_idx['Ar+']] = ne0
        
        # Initial electron temperature (eV)
        y0[self.idx_Te] = 4.0
        
        return y0
    
    def rhs(self, t, y):
        """Right-hand side of the ODE system."""
        cfg = self.config
        
        # Unpack densities (enforce non-negative)
        n = {}
        for s, i in self.species_idx.items():
            n[s] = max(y[i], 0.0)
        
        Te = max(y[self.idx_Te], 0.1)
        
        # Gas temperature
        Tg = gas_temperature(cfg.P_abs, cfg.pressure_mTorr)
        Tng = Tg / 300.0
        
        # Electron density from charge neutrality (Eq. 6)
        n_pos_total = sum(n.get(s, 0) for s in POSITIVE_IONS)
        n_neg_total = sum(n.get(s, 0) for s in NEGATIVE_IONS)
        ne_neutrality = n_pos_total - n_neg_total
        ne = max(ne_neutrality, 1e10)
        n['e'] = ne
        
        # Total neutral density
        n_total = sum(n.get(s, 0) for s in NEUTRAL_SPECIES)
        n_total = max(n_total, 1e14)
        
        # Initialize dn/dt
        dndt = {s: 0.0 for s in ALL_SPECIES}
        
        # Power balance: energy loss rate
        power_loss = 0.0  # in eV * m^{-3} * s^{-1}
        
        # =====================================================================
        # Process all reactions
        # =====================================================================
        for rxn in REACTIONS:
            rid = rxn['id']
            rtype = rxn['type']
            
            # Compute rate coefficient
            k = rxn['rate_coeff'](Te, Tg, Tng) if callable(rxn['rate_coeff']) else rxn['rate_coeff']
            
            # Compute reaction rate
            reactants = rxn['reactants']
            
            if rtype == 'electron':
                # e + X -> products: rate = k * ne * n_X
                if len(reactants) == 2:
                    sp = reactants[1]  # non-electron reactant
                    if sp == 'e':
                        # e + e reactions (not present here)
                        continue
                    rate = k * ne * n.get(sp, 0)
                else:
                    continue
                    
            elif rtype in ('penning_diss', 'penning_ion'):
                r1, r2 = reactants[0], reactants[1]
                rate = k * n.get(r1, 0) * n.get(r2, 0)
                
            elif rtype in ('neutral', 'ion_neutral', 'ion_ion', 'detachment'):
                r1, r2 = reactants[0], reactants[1]
                rate = k * n.get(r1, 0) * n.get(r2, 0)
                
            elif rtype == 'high_temp':
                # X + M -> products: pseudo-first-order with M = total neutrals
                r1 = reactants[0]  # species being dissociated
                rate = k * n.get(r1, 0) * n_total
                
            elif rtype == 'three_body':
                # A + B + M -> products: rate = k * n_A * n_B * n_total
                r1, r2 = reactants[0], reactants[1]
                rate = k * n.get(r1, 0) * n.get(r2, 0) * n_total
                
            else:
                continue
            
            if rate <= 0 or not np.isfinite(rate):
                continue
            
            # Apply stoichiometry: consume reactants, produce products
            # Use Counter to properly handle duplicate species
            from collections import Counter
            
            react_counts = Counter(rxn['reactants'])
            prod_counts = Counter(rxn['products'])
            
            for sp, count in react_counts.items():
                if sp == 'M':
                    continue
                if sp == 'e':
                    continue  # electron consumption handled via power balance / charge neutrality
                if sp in dndt:
                    dndt[sp] -= count * rate
            
            for sp, count in prod_counts.items():
                if sp == 'M':
                    continue
                if sp == 'e':
                    continue  # electron production handled via charge neutrality
                if sp in dndt:
                    dndt[sp] += count * rate
            
            # Handle electron count changes
            n_e_reactants = rxn['reactants'].count('e')
            n_e_products = rxn['products'].count('e')
            delta_e = n_e_products - n_e_reactants
            # For reactions that produce/consume electrons
            # (handled via charge neutrality, but track for power balance)
            
            # Power balance: electron energy loss
            eloss = rxn.get('energy_loss', 0)
            if isinstance(eloss, str):
                # Elastic collision: energy loss = 3*Te*me/M per collision
                if 'elastic' in eloss:
                    sp = rxn['reactants'][1]
                    M = SPECIES_MASS.get(sp, 40 * 1.6605e-27)
                    eloss_val = 3.0 * Te * m_e / M
                    power_loss += rate * eloss_val
            elif eloss != 0:
                power_loss += rate * eloss
        
        # =====================================================================
        # Surface losses for neutrals
        # =====================================================================
        for sp, gamma in SURFACE_RECOMBINATION.items():
            if n.get(sp, 0) > 0:
                k_wall = neutral_wall_loss_rate(sp, n, cfg, Tg)
                wall_loss = k_wall * n[sp]
                dndt[sp] -= wall_loss
                
                # Surface recombination products
                if sp == 'F':
                    dndt['F2'] += 0.5 * wall_loss  # F + F -> F2
                elif sp == 'N':
                    dndt['N2'] += 0.5 * wall_loss  # N + N -> N2
                elif sp in ('Ar_1s5', 'Ar_4p'):
                    dndt['Ar'] += wall_loss  # De-excitation
        
        # =====================================================================
        # Ion wall losses
        # =====================================================================
        ion_loss_rates, alpha, alpha_s = compute_ion_wall_loss(n, Te, Tg, cfg)
        
        wall_energy_per_pair = compute_wall_energy_loss(Te, alpha_s, Tg, cfg)
        
        for ion, k_wall_ion in ion_loss_rates.items():
            ion_wall_loss = k_wall_ion * n.get(ion, 0)
            dndt[ion] -= ion_wall_loss
            
            # Neutralization products (simplified: ion -> corresponding neutral)
            neutral_product = ion.replace('+', '')
            if neutral_product == 'NF3':
                dndt['NF3'] += ion_wall_loss
            elif neutral_product == 'NF2':
                dndt['NF2'] += ion_wall_loss
            elif neutral_product == 'NF':
                dndt['NF'] += ion_wall_loss
            elif neutral_product == 'F2':
                dndt['F2'] += ion_wall_loss
            elif neutral_product == 'N2':
                dndt['N2'] += ion_wall_loss
            elif neutral_product == 'F':
                dndt['F'] += ion_wall_loss
            elif neutral_product == 'N':
                dndt['N'] += ion_wall_loss
            elif neutral_product == 'Ar':
                dndt['Ar'] += ion_wall_loss
            
            # Energy loss at wall
            power_loss += ion_wall_loss * wall_energy_per_pair
        
        # =====================================================================
        # Pumping
        # =====================================================================
        for sp in NEUTRAL_SPECIES:
            pump = pumping_rate(sp, n.get(sp, 0), cfg, n_total, Tg)
            dndt[sp] += pump
        
        # =====================================================================
        # Power balance (Eq. 25)
        # d/dt(3/2 * e * ne * Te) = P_abs/V - power_loss_collisions - power_loss_wall
        # =====================================================================
        # Convert: dTe/dt = (2/3) * 1/(e*ne) * [P_abs/V - e*ne*sum(nm*km*em) - wall_loss]
        power_input = cfg.P_abs / cfg.V  # W/m^3 = J/(s*m^3)
        power_input_eV = power_input / eV_to_J  # eV/(s*m^3)
        
        # power_loss is already in eV*m^{-3}*s^{-1}
        dTedt = (2.0 / 3.0) / max(ne, 1e10) * (power_input_eV - power_loss)
        
        # Limit Te rate of change for stability
        dTedt = np.clip(dTedt, -1e9, 1e9)
        
        # =====================================================================
        # Pack derivatives
        # =====================================================================
        dydt = np.zeros(self.n_species + 1)
        for s, i in self.species_idx.items():
            if s == 'e':
                # Electron density from charge neutrality
                dydt[i] = sum(dndt.get(sp, 0) for sp in POSITIVE_IONS) - \
                          sum(dndt.get(sp, 0) for sp in NEGATIVE_IONS)
            else:
                dydt[i] = dndt.get(s, 0)
        
        dydt[self.idx_Te] = dTedt
        
        return dydt
    
    def solve(self, t_end=0.05, n_points=2000, method='BDF'):
        """Solve the ODE system to steady state."""
        y0 = self.initial_conditions()
        
        t_span = (0, t_end)
        t_eval = np.linspace(0, t_end, n_points)
        
        print(f"Solving NF3/Ar global model...")
        print(f"  Conditions: {self.config.flow_Ar}/{self.config.flow_NF3} sccm Ar/NF3, "
              f"{self.config.pressure_mTorr} mTorr, {self.config.P_abs} W")
        print(f"  Chamber: R={self.config.R*100:.0f} cm, L={self.config.L*100:.0f} cm, "
              f"V={self.config.V*1e6:.1f} cm^3")
        
        Tg = gas_temperature(self.config.P_abs, self.config.pressure_mTorr)
        print(f"  Gas temperature: {Tg:.0f} K")
        
        sol = solve_ivp(
            self.rhs, t_span, y0,
            method=method,
            t_eval=t_eval,
            rtol=1e-6, atol=1e-6,
            max_step=t_end / 100
        )
        
        if sol.success:
            print(f"  Solution converged ({sol.nfev} function evaluations)")
        else:
            print(f"  WARNING: Solution failed: {sol.message}")
        
        return sol
    
    def print_results(self, sol):
        """Print steady-state results."""
        y_ss = sol.y[:, -1]
        
        Te = y_ss[self.idx_Te]
        Tg = gas_temperature(self.config.P_abs, self.config.pressure_mTorr)
        
        print(f"\n{'='*60}")
        print(f"STEADY-STATE RESULTS")
        print(f"{'='*60}")
        print(f"Electron temperature: {Te:.2f} eV")
        print(f"Gas temperature: {Tg:.0f} K")
        
        ne = max(y_ss[self.species_idx['e']], 0)
        print(f"Electron density: {ne:.3e} m^-3 ({ne/1e16:.2f} x10^16 m^-3)")
        
        # Neutral densities
        print(f"\nNeutral species densities [m^-3]:")
        n_NF3_init = self.initial_conditions()[self.species_idx['NF3']]
        for sp in NEUTRAL_SPECIES:
            n_val = max(y_ss[self.species_idx[sp]], 0)
            print(f"  {sp:8s}: {n_val:.3e}")
        
        # NF3 dissociation rate
        n_NF3 = max(y_ss[self.species_idx['NF3']], 0)
        n_Ar = max(y_ss[self.species_idx['Ar']], 0)
        n_Ar_0 = self.initial_conditions()[self.species_idx['Ar']]
        if n_Ar > 0 and n_Ar_0 > 0:
            dissoc = (1.0 - (n_Ar_0 / n_Ar) * (n_NF3 / n_NF3_init)) * 100 if n_NF3_init > 0 else 0
            print(f"\nNF3 dissociation: {dissoc:.1f}%")
        
        # Ion densities
        print(f"\nPositive ion densities [m^-3]:")
        for sp in POSITIVE_IONS:
            n_val = max(y_ss[self.species_idx[sp]], 0)
            if n_val > 1e10:
                print(f"  {sp:8s}: {n_val:.3e}")
        
        print(f"\nNegative ion densities [m^-3]:")
        for sp in NEGATIVE_IONS:
            n_val = max(y_ss[self.species_idx[sp]], 0)
            if n_val > 1e10:
                print(f"  {sp:8s}: {n_val:.3e}")
        
        # Electronegativity
        n_neg = sum(max(y_ss[self.species_idx[s]], 0) for s in NEGATIVE_IONS)
        alpha = n_neg / max(ne, 1e6)
        print(f"\nElectronegativity (n_neg/n_e): {alpha:.3f}")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    # Base case from Huang et al. 2026
    config = ReactorConfig(
        R=0.21,        # 21 cm radius
        L=0.06,        # 6 cm height
        P_abs=600.0,   # 600 W absorbed power
        pressure_mTorr=30.0,
        flow_Ar_sccm=80.0,
        flow_NF3_sccm=20.0,
        Twall=300.0
    )
    
    model = NF3GlobalModel(config)
    sol = model.solve(t_end=0.05, n_points=2000)
    
    if sol.success:
        model.print_results(sol)
