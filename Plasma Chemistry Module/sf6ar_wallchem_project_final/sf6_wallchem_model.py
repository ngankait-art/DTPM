#!/usr/bin/env python3
"""
SF6/Ar Global Model — With Penning Ionization
===============================================
Based on sf6_FINAL.py (working hybrid solver with continuation).

NEW PHYSICS ADDED:
  Ar* + SF6 → SF5+ + F + Ar + e   (Penning ionization)
  Ar* + SF6 → Ar + SF5 + F        (Penning dissociation, non-ionizing quenching)
  Ar* + SFx → Ar + SFx*           (quenching by SFx fragments)
  Ar* + F2  → Ar + 2F             (quenching/dissociation by F2)

Rate coefficients from:
  Velazco & Setser, J. Chem. Phys. 62 (1975) 1990
  Kolts & Setser, J. Chem. Phys. 68 (1978) 4848
  Typical values used in Ar/molecular gas global models:
    Gudmundsson & Thorsteinsson, PSST 16 (2007) 399 (O2/Ar analog)
    
Why this matters for alpha vs Ar%:
  1. Penning ionization provides ionization WITHOUT consuming electrons
     → produces SF5+ + e from Ar* + SF6 at rates proportional to nAr*·nSF6
  2. Quenching by SF6/SFx DESTROYS Ar* much faster than electron processes
     → reduces nArm by factor ~5-10 compared to pure Ar
     → this strongly reduces the effective Ar* stepwise ionization rate
  3. Net effect: alpha decreases MORE SLOWLY with Ar% because:
     - At moderate Ar%, Penning sustains some ionization (ne stays modest)
     - SF6 quenching prevents Ar* from accumulating (limits stepwise iz)
     - The EN→EP transition is gradual rather than abrupt
"""
import numpy as np
from scipy.optimize import brentq
from scipy.constants import e as eC, k as kB, m_e, pi
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6
M = {'S':32.06,'F':19.0,'F2':38.0,'SF':51.06,'SF2':70.06,'SF3':89.06,
     'SF4':108.06,'SF5':127.06,'SF6':146.06,'Ar':39.948}

class Reactor:
    def __init__(self, R=0.180, L=0.175):
        self.R, self.L = R, L
        self.V = pi*R**2*L; self.A = 2*pi*R**2 + 2*pi*R*L
        self.Lambda = 1.0/np.sqrt((pi/L)**2 + (2.405/R)**2)

def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate (Ryan & Plumb 1990, Eq from Troe 1977).
    Returns effective bimolecular rate in cm³/s."""
    if M_cm3 <= 0 or k0 <= 0: return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F

def rates(Te):
    Te = max(Te, 0.3); k = {}
    # SF6 dissociation
    k['d1']=1.5e-7*np.exp(-8.1/Te)*cm3; k['d2']=9e-9*np.exp(-13.4/Te)*cm3
    k['d3']=2.5e-8*np.exp(-33.5/Te)*cm3; k['d4']=2.3e-8*np.exp(-23.9/Te)*cm3
    k['d5']=1.5e-9*np.exp(-26.0/Te)*cm3; k['d6']=1.2e-8*np.exp(-5.8/Te)*cm3
    k['d7']=1.5e-7*np.exp(-9.0/Te)*cm3; k['d8']=6.2e-8*np.exp(-9.0/Te)*cm3
    k['d9']=8.6e-8*np.exp(-9.0/Te)*cm3; k['d10']=4.5e-8*np.exp(-9.0/Te)*cm3
    k['d11']=6.2e-8*np.exp(-9.0/Te)*cm3
    # Excitation / elastic
    k['vib_SF6']=7.9e-8*np.exp(-0.1*Te+0.002*Te**2)*cm3
    k['el_SF6']=2.8e-7*np.exp(-1.5/Te)*cm3
    k['exc_F']=9.2e-9*np.exp(-14.3/Te)*cm3; k['el_F']=1.1e-7*np.exp(-1.93/Te)*cm3
    k['vib_F2']=1.8e-10*Te**1.72*np.exp(-1.55/Te)*cm3; k['el_F2']=2.5e-7*np.exp(-0.48/Te)*cm3
    # Ionization from SF6
    k['iz18']=1.2e-7*np.exp(-18.1/Te)*cm3; k['iz19']=8.4e-9*np.exp(-19.9/Te)*cm3
    k['iz20']=3.2e-8*np.exp(-20.7/Te)*cm3; k['iz21']=7.6e-9*np.exp(-24.4/Te)*cm3
    k['iz22']=1.2e-8*np.exp(-26.0/Te)*cm3; k['iz23']=1.4e-8*np.exp(-39.9/Te)*cm3
    k['iz24']=1.2e-8*np.exp(-31.7/Te)*cm3
    # Ionization from fragments
    k['iz25']=1.0e-7*np.exp(-17.8/Te)*cm3; k['iz26']=9.4e-8*np.exp(-22.8/Te)*cm3
    k['iz27']=1.0e-7*np.exp(-18.9/Te)*cm3; k['iz28']=1.3e-8*np.exp(-16.5/Te)*cm3
    k['iz29']=1.6e-7*np.exp(-13.3/Te)*cm3
    # Attachment
    k['at30']=2.4e-10/Te**1.49*cm3; k['at31']=2.0e-11/Te**1.46*cm3
    k['at32']=3.9e-12*np.exp(0.45*Te-0.04*Te**2)*cm3; k['at33']=1.2e-13*np.exp(0.70*Te-0.05*Te**2)*cm3
    k['at34']=5.4e-15*np.exp(0.77*Te-0.05*Te**2)*cm3; k['at35']=3.4e-11*np.exp(0.46*Te-0.04*Te**2)*cm3
    k['at36']=2.2e-13*np.exp(0.71*Te-0.05*Te**2)*cm3
    # Neutral recombination — Troe fall-off (Ryan & Plumb 1990, Table III)
    # k = k0*[M]/(1+Pr) * Fc^(1/(1+log10(Pr)^2)), Pr=k0*[M]/kinf
    # k0 values for SF6 as third body
    # These are PRESSURE-DEPENDENT — must be evaluated at actual [M]
    k['nr42_k0']=3.4e-23; k['nr42_kinf']=1.0e-11; k['nr42_Fc']=0.43  # SF5+F→SF6
    k['nr41_k0']=3.7e-28; k['nr41_kinf']=5.0e-12; k['nr41_Fc']=0.46  # SF4+F→SF5
    k['nr40_k0']=2.8e-26; k['nr40_kinf']=2.0e-11; k['nr40_Fc']=0.47  # SF3+F→SF4
    k['nr39_k0']=1.7e-28; k['nr39_kinf']=2.0e-11; k['nr39_Fc']=0.56  # SF2+F→SF3
    k['nr38_k0']=1.0e-30; k['nr38_kinf']=2.0e-11; k['nr38_Fc']=0.67  # SF+F→SF2
    k['nr37_k0']=7.5e-33; k['nr37_kinf']=2.0e-11; k['nr37_Fc']=0.73  # S+F→SF
    # Disproportionation (pressure-independent)
    k['nr45']=2.5e-11*cm3; k['nr44']=2.5e-11*cm3; k['nr43']=2.5e-11*cm3
    k['rec']=1.5e-9*cm3
    # Ar electron-impact reactions (Table 4 of Lallement)
    k['Ar_iz']=1.2e-10*np.exp(-21.7/Te)*cm3      # (101) Ar + e → Ar+ + 2e
    k['Ar_exc']=4.2e-9*np.exp(-8.0/Te)*cm3        # (102) Ar + e → Ar* + e
    k['Ar_iz_m']=2.05e-7*np.exp(-4.95/Te)*cm3     # (103) Ar* + e → Ar+ + 2e
    k['Ar_q']=2.0e-7*cm3                           # (104) Ar* + e → Ar + e
    k['Ar_el']=max((-1.1e-8+3.9e-8*Te-1.3e-8*Te**2+2e-9*Te**3-1.4e-10*Te**4+3.9e-12*Te**5)*cm3,1e-20)
    
    # ─── NEW: Penning ionization and Ar* quenching by molecular species ───
    # Ar*(3P2) energy = 11.55 eV; SF5 IE = 10.16 eV → Penning ionization allowed
    # Velazco & Setser (1975): total quenching of Ar* by SF6 ≈ 5e-10 cm³/s
    # Penning ionization fraction ≈ 30-40% → k_Penning ≈ 2e-10 cm³/s
    k['Penn_SF6'] = 2.0e-10*cm3    # Ar* + SF6 → SF5+ + F + Ar + e
    k['qnch_SF6'] = 3.0e-10*cm3    # Ar* + SF6 → Ar + SF5 + F (non-ionizing dissociative quench)
    # Quenching by SFx fragments (estimated, smaller cross-section)
    k['qnch_SFx'] = 1.0e-10*cm3    # Ar* + SFx → Ar + SFx (quenching, each species)
    # Quenching by F2 (dissociative)
    k['qnch_F2']  = 5.0e-11*cm3    # Ar* + F2 → Ar + 2F
    # Quenching by F atoms (very small — no molecular modes to absorb energy)
    k['qnch_F']   = 5.0e-12*cm3    # Ar* + F → Ar + F (weak, radiative quenching)
    
    # Totals
    k['iz_SF6_total']=k['iz18']+k['iz19']+k['iz20']+k['iz21']+k['iz22']+k['iz23']+k['iz24']
    k['att_SF6_total']=k['at30']+k['at31']+k['at32']+k['at33']+k['at34']+k['at35']+k['at36']
    return k

def kw_n(g, Ma, Tg, ng, R): 
    if g<=0: return 0.
    Mk=Ma*AMU; v=np.sqrt(8*kB*Tg/(pi*Mk)); l=1/(ng*4e-19) if ng>0 else .1
    De=1/(1/max(kB*Tg*l/(Mk*v),1e-30)+1/max(v*l/3,1e-30))
    return 1/(R.Lambda**2/De+2*R.V*(2-g)/(R.A*v*g))

def kw_i(Te, Ma, al, Tn, R, ng):
    Mk=Ma*AMU; gam=Te/max(Tn,.01)
    uB=np.sqrt(eC*Te*(1+al)/(Mk*(1+gam*al)))
    sig=5e-19; l=1/(ng*sig) if ng>0 else .1
    Da=eC*Te/(Mk*np.sqrt(eC*Tn/Mk)/l)
    EN=(1+3*al/gam)/(1+al)
    hL=EN*.86/np.sqrt(3+R.L/(2*l)+(.86*R.L*uB/(pi*gam*Da))**2)
    hR=EN*.8/np.sqrt(4+R.R/l+(.8*R.R*uB/(2.405*.5191*2.405*gam*Da))**2)
    hL=np.clip(hL,1e-5,1); hR=np.clip(hR,1e-5,1)
    return 2*uB*(hL/R.L+hR/R.R)

def solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0., Q_sccm=40, Tgas=300, T_neg=0.3,
                gamma_F=0.01, beta_SFx=0.02, eta=0.12, k_rec_override=None,
                use_troe=True, wall_chem=False,
                init_Te=None, init_ne=None, init_alpha=None, init_ns=None):
    """Solve with Penning ionization, Ar* quenching, and optional Kokkoris wall chemistry.
    
    wall_chem: bool or dict. If True, uses default Kokkoris surface reaction probabilities.
        If dict, keys are: s_F (F sticking), s_SFx (SFx sticking), p_fluor (fluorination
        of adsorbed SF5 to SF6), p_wallrec (SFx + F(wall) → SFx+1), p_FF (F+F(wall)→F2).
    """
    R = Reactor(); P_abs = P_rf*eta
    p_Pa = p_mTorr*MTORR_TO_PA; ng0 = p_Pa/(kB*Tgas)
    nSF6_0 = ng0*(1-frac_Ar); nAr0 = ng0*frac_Ar
    Q_tp = Q_sccm*1e-6/60*1.01325e5*(Tgas/273.15)
    tau = p_Pa*R.V/Q_tp if Q_tp>0 else 1e10
    
    Te = init_Te if init_Te else 3.0
    ne = init_ne if init_ne else 5e15
    alpha = init_alpha if init_alpha else 5.0
    if init_ns:
        nSF6,nSF5,nSF4,nSF3,nSF2,nSF,nS,nF,nF2 = [init_ns.get(s,nSF6_0*0.01) for s in 
            ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']]
    else:
        nSF6=nSF6_0*0.4; nSF5=nSF6_0*0.02; nSF4=nSF6_0*0.01; nSF3=nSF6_0*0.005
        nSF2=nSF6_0*0.001; nSF=nSF6_0*0.0005; nS=nSF6_0*0.0001
        nF=nSF6_0*0.3; nF2=nSF6_0*0.01
    
    converged = False
    for outer in range(500):
        Te_old,ne_old,al_old = Te,ne,alpha
        k = rates(Te); ng = ng0
        if k_rec_override is not None:
            k['rec'] = k_rec_override * 1e-6  # cm³/s → m³/s
        
        # Compute pressure-dependent neutral recombination rates (Troe formula)
        ng_cm3 = ng * 1e-6  # convert m⁻³ to cm⁻³ for Troe formula
        if use_troe is True or use_troe == 1.0:
            k['nr42'] = troe_rate(k['nr42_k0'], k['nr42_kinf'], k['nr42_Fc'], ng_cm3) * cm3  # → m³/s
            k['nr41'] = troe_rate(k['nr41_k0'], k['nr41_kinf'], k['nr41_Fc'], ng_cm3) * cm3
            k['nr40'] = troe_rate(k['nr40_k0'], k['nr40_kinf'], k['nr40_Fc'], ng_cm3) * cm3
            k['nr39'] = troe_rate(k['nr39_k0'], k['nr39_kinf'], k['nr39_Fc'], ng_cm3) * cm3
            k['nr38'] = troe_rate(k['nr38_k0'], k['nr38_kinf'], k['nr38_Fc'], ng_cm3) * cm3
            k['nr37'] = troe_rate(k['nr37_k0'], k['nr37_kinf'], k['nr37_Fc'], ng_cm3) * cm3
        elif use_troe is False or use_troe == 0.0:
            # Use Lallement Table 3 values directly (effective at ~1 Torr)
            k['nr42'] = 1.0e-11 * cm3   # SF5+F→SF6
            k['nr41'] = 1.7e-12 * cm3   # SF4+F→SF5
            k['nr40'] = 1.6e-11 * cm3   # SF3+F→SF4
            k['nr39'] = 2.6e-12 * cm3   # SF2+F→SF3
            k['nr38'] = 2.9e-14 * cm3   # SF+F→SF2
            k['nr37'] = 2.0e-16 * cm3   # S+F→SF
        else:
            # Blend: use_troe is a float 0-1 (1=pure Troe, 0=pure Table3)
            # Geometric interpolation in log space
            import math
            f = float(use_troe)
            t3 = {'nr42': 1.0e-11, 'nr41': 1.7e-12, 'nr40': 1.6e-11,
                  'nr39': 2.6e-12, 'nr38': 2.9e-14, 'nr37': 2.0e-16}
            for ch in ['nr42','nr41','nr40','nr39','nr38','nr37']:
                k_troe = troe_rate(k[ch+'_k0'], k[ch+'_kinf'], k[ch+'_Fc'], ng_cm3)
                k[ch] = math.exp(f*math.log(k_troe) + (1-f)*math.log(t3[ch])) * cm3
        
        kwF = kw_n(gamma_F, M['F'], Tgas, ng, R)
        kwS = kw_n(beta_SFx, M['SF5'], Tgas, ng, R)
        
        # ── Wall surface chemistry (Kokkoris-inspired) ──
        # Surface coverages and wall-mediated recombination producing SF6, SFx+1, F2
        R_wall_SF6 = 0.0   # wall production rate of SF6 (m⁻³ s⁻¹)
        R_wall_SFx = {}    # wall production of SFx+1 from SFx + F(wall)
        R_wall_F2 = 0.0    # wall production of F2 from F + F(wall)
        F_wall_consumption = 0.0  # extra F consumption on walls beyond simple sticking
        
        if wall_chem:
            # Default Kokkoris probabilities (Table 2)
            wc = wall_chem if isinstance(wall_chem, dict) else {}
            s_F     = wc.get('s_F', 0.15)        # F adsorption probability
            s_SFx   = wc.get('s_SFx', 0.08)      # SFx adsorption probability
            p_fluor = wc.get('p_fluor', 0.025)    # F + SF5(wall) → SF6
            p_wr    = wc.get('p_wallrec', 1.0)    # SFx + F(wall) → SFx+1
            p_FF    = wc.get('p_FF', 0.5)         # F + F(wall) → F2
            p_F_SF3 = wc.get('p_F_SF3', 0.5)      # F + SF3(wall) → SF4(wall)
            p_F_SF4 = wc.get('p_F_SF4', 0.2)      # F + SF4(wall) → SF5(wall)
            
            # Thermal velocities (m/s)
            vth_F = np.sqrt(8*kB*Tgas/(pi*M['F']*AMU))
            vth_SF5 = np.sqrt(8*kB*Tgas/(pi*M['SF5']*AMU))
            vth_SF4 = np.sqrt(8*kB*Tgas/(pi*M['SF4']*AMU))
            vth_SF3 = np.sqrt(8*kB*Tgas/(pi*M['SF3']*AMU))
            
            AoV = R.A / R.V  # wall area / volume ratio
            
            # Impinging fluxes (m⁻² s⁻¹) = n * vth / 4
            flux_F = nF * vth_F / 4
            flux_SF5 = nSF5 * vth_SF5 / 4
            flux_SF4 = nSF4 * vth_SF4 / 4
            flux_SF3 = nSF3 * vth_SF3 / 4
            
            # Steady-state surface coverages (simplified: assume fast equilibration)
            # θ_F from balance: s_F * flux_F = θ_F * (p_FF*flux_F + p_wr*(flux_SF3+flux_SF4+flux_SF5))
            # → θ_F = s_F * flux_F / (p_FF*flux_F + p_wr*(flux_SF3+flux_SF4+flux_SF5) + 1e-30)
            denom_F = p_FF*flux_F + p_wr*(flux_SF3+flux_SF4+flux_SF5) + 1e-30
            theta_F = min(s_F * flux_F / denom_F, 0.5)
            
            # θ_SF5 from balance: s_SFx*flux_SF5 + p_F_SF4*theta_SF4*flux_F = θ_SF5*(p_fluor*flux_F)
            # θ_SF4: s_SFx*flux_SF4 + p_F_SF3*theta_SF3*flux_F = θ_SF4*(p_F_SF4*flux_F)
            # θ_SF3: s_SFx*flux_SF3 = θ_SF3*(p_F_SF3*flux_F)
            theta_SF3 = min(s_SFx * flux_SF3 / (p_F_SF3 * flux_F + 1e-30), 0.3)
            theta_SF4 = min((s_SFx*flux_SF4 + p_F_SF3*theta_SF3*flux_F) / (p_F_SF4*flux_F + 1e-30), 0.3)
            theta_SF5 = min((s_SFx*flux_SF5 + p_F_SF4*theta_SF4*flux_F) / (p_fluor*flux_F + 1e-30), 0.3)
            
            # Wall production rates (m⁻³ s⁻¹ = flux * probability * theta * A/V)
            # S7: F + SF5(wall) → SF6 — returns SF6 to gas phase!
            R_wall_SF6 = p_fluor * theta_SF5 * flux_F * AoV
            
            # S9-S11: SFx + F(wall) → SFx+1 — returns SFx+1 to gas phase
            R_wall_SFx['SF4'] = p_wr * theta_F * flux_SF3 * AoV  # SF3 + F(wall) → SF4
            R_wall_SFx['SF5'] = p_wr * theta_F * flux_SF4 * AoV  # SF4 + F(wall) → SF5  
            R_wall_SFx['SF6_from_SF5'] = p_wr * theta_F * flux_SF5 * AoV  # SF5 + F(wall) → SF6
            R_wall_SF6 += R_wall_SFx['SF6_from_SF5']  # total SF6 wall production
            
            # S8: F + F(wall) → F2 — produces F2, consumes F
            R_wall_F2 = p_FF * theta_F * flux_F * AoV
            
            # Extra F consumption from fluorination reactions
            F_wall_consumption = (p_fluor*theta_SF5 + p_F_SF4*theta_SF4 + p_F_SF3*theta_SF3) * flux_F * AoV
        
        k_e = k['d1']+k['d2']+k['d3']+k['d4']+k['d5']+k['iz_SF6_total']+k['att_SF6_total']
        
        # ── Ar* metastable density with FULL quenching ──
        # Production: k_Ar_exc * ne * nAr
        # Loss: (k_iz_m + k_q)*ne + k_Penn*nSF6 + k_qnch*nSF6 + k_qnch_SFx*(nSF5+nSF4+nSF3) 
        #        + k_qnch_F2*nF2 + k_qnch_F*nF + k_wall_diff
        k_Arm_wall = kw_n(1.0, M['Ar'], Tgas, ng, R)  # diffusion loss (gamma=1 for metastable)
        
        # Total Ar* quenching rate by heavy species (NEW)
        R_quench_heavy = (k['Penn_SF6']*nSF6 + k['qnch_SF6']*nSF6 + 
                         k['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                         k['qnch_F2']*nF2 + k['qnch_F']*nF)
        
        nArm = k['Ar_exc']*ne*nAr0 / ((k['Ar_iz_m']+k['Ar_q'])*ne + k_Arm_wall + R_quench_heavy + 1e-30)
        
        # Penning ionization rate (produces SF5+ + F + e, consumes SF6)
        R_Penning = k['Penn_SF6'] * nArm * nSF6  # m^-3 s^-1
        
        # Neutral sub-iteration with damping
        for sub in range(20):
            # SF6 balance: gas-phase sources + WALL PRODUCTION of SF6
            s6 = nSF6_0/tau + k['nr42']*nSF5*nF + k['nr45']*nSF5**2 + R_wall_SF6
            # Loss of SF6: electron-impact + Penning by Ar*
            k_SF6_loss = k_e*ne + k['Penn_SF6']*nArm + k['qnch_SF6']*nArm + 1/tau
            nSF6_n = s6/(k_SF6_loss+1e-30); nSF6_n = np.clip(nSF6_n, 1e10, nSF6_0)
            nSF6 = 0.3*nSF6 + 0.7*nSF6_n
            
            # SF5: non-ionizing quench produces neutral SF5; wall recombination of SF4+F(wall)→SF5
            nSF5_prod = k['d1']*ne*nSF6 + k['nr41']*nSF4*nF + k['qnch_SF6']*nArm*nSF6
            nSF5_prod += R_wall_SFx.get('SF5', 0.0)  # wall: SF4 + F(wall) → SF5
            nSF5 = nSF5_prod/((k['d7']+k['iz25']+k['iz26'])*ne+k['nr42']*nF+2*k['nr45']*nSF5+kwS+k['qnch_SFx']*nArm+1/tau+1e-30)
            
            nSF4_prod = k['d2']*ne*nSF6+k['d7']*ne*nSF5+k['nr45']*nSF5**2
            nSF4_prod += R_wall_SFx.get('SF4', 0.0)  # wall: SF3 + F(wall) → SF4
            nSF4 = nSF4_prod/(k['d8']*ne+k['nr41']*nF+kwS+k['qnch_SFx']*nArm+1/tau+1e-30)
            
            nSF3 = (k['d3']*ne*nSF6+k['d8']*ne*nSF4)/((k['d9']+k['iz27'])*ne+k['nr40']*nF+kwS+k['qnch_SFx']*nArm+1/tau+1e-30)
            nSF2 = (k['d4']*ne*nSF6+k['d9']*ne*nSF3)/(k['d10']*ne+k['nr39']*nF+1/tau+1e-30)
            nSF = (k['d5']*ne*nSF6+k['d10']*ne*nSF2)/(k['d11']*ne+k['nr38']*nF+1/tau+1e-30)
            nS = k['d11']*ne*nSF/(k['iz29']*ne+k['nr37']*nF+1/tau+1e-30)
            
            # F balance: includes F from Penning + wall F2 production consumes F
            RF = ne*nSF6*(k['d1']+2*k['d2']+3*k['d3']+2*k['d4']+3*k['d5']+k['iz18']+2*k['iz19']+3*k['iz20']+2*k['iz21']+3*k['iz22']+4*k['iz23']+k['at31'])
            RF += ne*(nSF5*(k['d7']+k['iz26'])+nSF4*k['d8']+nSF3*k['d9']+nSF2*k['d10']+nSF*k['d11'])
            RF += k['Penn_SF6']*nArm*nSF6
            RF += k['qnch_SF6']*nArm*nSF6
            RF += 2*k['qnch_F2']*nArm*nF2
            
            nF2_prod = ne*nSF6*(k['d4']+k['d5']+k['iz21']+k['iz22']+k['iz23'])+.5*kwF*nF
            nF2_prod += R_wall_F2  # wall: F + F(wall) → F2
            nF2 = nF2_prod/(k['d6']*ne+k['qnch_F2']*nArm+1/tau+1e-30)
            RF += 2*k['d6']*ne*nF2
            kFL = kwF+1/tau+k['iz28']*ne+k['nr42']*nSF5+k['nr41']*nSF4+k['nr40']*nSF3+k['nr39']*nSF2+k['nr38']*nSF+k['nr37']*nS
            kFL += F_wall_consumption/(nF+1e-30)  # extra F loss from wall fluorination
            nF_n = RF/(kFL+1e-30)
            nF = 0.3*nF + 0.7*min(nF_n, 6*nSF6_0)
            
            # Update nArm with new neutral densities
            R_quench_heavy = (k['Penn_SF6']*nSF6 + k['qnch_SF6']*nSF6 + 
                             k['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                             k['qnch_F2']*nF2 + k['qnch_F']*nF)
            nArm = k['Ar_exc']*ne*nAr0 / ((k['Ar_iz_m']+k['Ar_q'])*ne + k_Arm_wall + R_quench_heavy + 1e-30)
            R_Penning = k['Penn_SF6'] * nArm * nSF6
            
            # Update wall chemistry with new densities
            if wall_chem:
                flux_F = nF * vth_F / 4
                flux_SF5 = nSF5 * vth_SF5 / 4
                flux_SF4 = nSF4 * vth_SF4 / 4
                flux_SF3 = nSF3 * vth_SF3 / 4
                denom_F = p_FF*flux_F + p_wr*(flux_SF3+flux_SF4+flux_SF5) + 1e-30
                theta_F = min(s_F * flux_F / denom_F, 0.5)
                theta_SF3 = min(s_SFx * flux_SF3 / (p_F_SF3 * flux_F + 1e-30), 0.3)
                theta_SF4 = min((s_SFx*flux_SF4 + p_F_SF3*theta_SF3*flux_F) / (p_F_SF4*flux_F + 1e-30), 0.3)
                theta_SF5 = min((s_SFx*flux_SF5 + p_F_SF4*theta_SF4*flux_F) / (p_fluor*flux_F + 1e-30), 0.3)
                R_wall_SF6 = p_fluor * theta_SF5 * flux_F * AoV + p_wr * theta_F * flux_SF5 * AoV
                R_wall_SFx['SF4'] = p_wr * theta_F * flux_SF3 * AoV
                R_wall_SFx['SF5'] = p_wr * theta_F * flux_SF4 * AoV
                R_wall_F2 = p_FF * theta_F * flux_F * AoV
                F_wall_consumption = (p_fluor*theta_SF5 + p_F_SF4*theta_SF4 + p_F_SF3*theta_SF3) * flux_F * AoV
        
        ng = nSF6+nSF5+nSF4+nSF3+nSF2+nSF+nS+nF+nF2+nAr0; ng = max(ng, ng0*0.1)
        
        # ── Ionization/attachment ──
        # Total ionization now includes Penning as a source of SF5+ (and electron)
        Riz_electron = (k['iz_SF6_total']*nSF6+(k['iz25']+k['iz26'])*nSF5+k['iz27']*nSF3+
                       k['iz28']*nF+k['iz29']*nS+k['Ar_iz']*nAr0+k['Ar_iz_m']*nArm)
        # Penning ionization rate per unit volume (NOT proportional to ne!)
        R_Penn_vol = k['Penn_SF6']*nArm*nSF6  # m^-3 s^-1
        
        Ratt = k['att_SF6_total']*nSF6
        kwi = kw_i(Te, M['SF5'], alpha, T_neg, R, ng)
        
        # Alpha from quadratic: now with Penning as additional ionization source
        # Total ionization: Riz_electron * ne + R_Penn_vol
        # Total attachment: Ratt * ne
        # At steady state: alpha*(1+alpha) = Ratt / (k_rec * ne)
        # (Penning affects Te equation, not directly the alpha quadratic,
        #  because Penning produces BOTH an ion AND an electron)
        rhs_q = Ratt/(k['rec']*ne) if ne>0 else 0
        alpha_new = (-1+np.sqrt(1+4*rhs_q))/2 if rhs_q>0 else 0
        
        # ── Ec: collisional energy loss per electron-ion pair ──
        # CRITICAL FIX: Ar excitation that leads to stepwise ionization is NOT pure loss.
        # The 2-step path (e+Ar→Ar*+e, then e+Ar*→Ar++2e) costs 12+4.95≈17 eV total,
        # which is already counted: 12 eV in Ar_exc and 4.95 eV in Ar_iz_m.
        # But if we count all Ar excitation as loss AND count stepwise ionization separately,
        # we double-count the 12 eV for the fraction that goes through stepwise ionization.
        # 
        # Correct approach: subtract the Ar excitation energy that becomes stepwise ionization.
        # fraction_stepwise = k_Ar_iz_m * ne / (k_Ar_iz_m*ne + k_Ar_q*ne + k_wall + R_quench_heavy)
        # This is the fraction of Ar* that gets stepwise-ionized vs quenched/lost.
        
        R_quench_total = ((k['Ar_iz_m']+k['Ar_q'])*ne + k_Arm_wall + R_quench_heavy)
        frac_stepwise = k['Ar_iz_m']*ne / max(R_quench_total, 1e-30)
        frac_stepwise = min(frac_stepwise, 1.0)
        
        # Ar excitation energy: only count the WASTED fraction as pure loss
        # (the stepwise fraction's 12 eV is accounted for through the stepwise iz threshold)
        Ar_exc_loss = 12*k['Ar_exc']*nAr0 * (1.0 - frac_stepwise)
        # The stepwise ionization threshold (4.95 eV) is the ADDITIONAL cost beyond the 12 eV
        # already invested in creating Ar*. Total cost for stepwise path = 12+4.95=16.95 eV.
        # We count: 12*(1-f) as excitation loss + 4.95 in stepwise iz + 12*f implicitly through
        # the fact that we already counted the excitation energy for those Ar*.
        # Actually simpler: count 16.95*k_Ar_iz_m*nArm as the FULL cost of stepwise ionization
        # and subtract all 12*k_Ar_exc*nAr0 from the excitation loss.
        # But this overcounts because some Ar* are quenched.
        # The cleanest: Eloss = 12*(1-f)*k_exc*nAr + (12+4.95)*f*k_exc*nAr for Ar channels
        #             = 12*k_exc*nAr + 4.95*f*k_exc*nAr = original + 4.95*k_iz_m*nArm
        # And in Riz, stepwise is already counted. So Eloss/Riz just needs the Ar_exc unchanged
        # PLUS the stepwise iz energy (4.95 eV * rate), which should be in Riz denominator too.
        # 
        # Actually the real issue is: the stepwise ionization k_Ar_iz_m*nArm is in Riz,
        # so its threshold energy (4.95 eV) should be in Eloss. And the excitation energy
        # (12 eV) should be in Eloss weighted by the TOTAL excitation rate (not just stepwise).
        # This IS what we're currently doing. The Ec is correct as written.
        #
        # THE REAL PROBLEM: ne is too high because the Ar-dominated eps_T 
        # is artificially low when stepwise ionization is efficient.
        # At moderate Ar%, stepwise iz makes Riz large → ne = P/(Riz*eT*V) → ne is large.
        # The paper avoids this because they solve the full coupled ODE system with C++ Newton-Raphson,
        # which finds a self-consistent solution where Te is slightly higher and nArm is lower.
        #
        # Practical fix: ensure the Ar stepwise ionization energy cost is properly accounted.
        # Each stepwise event costs: 12 eV (excitation) + 4.95 eV (ionization of Ar*) = 16.95 eV
        # But in our Eloss, the 12 eV is counted per Ar excitation event (whether or not it leads to iz).
        # The Riz includes k_Ar_iz_m*nArm.
        # So Ec = Eloss/Riz correctly ATTRIBUTES the excitation cost.
        # If we use the effective cost for stepwise: Ec_stepwise ≈ (12/frac_stepwise + 4.95)
        # At low frac_stepwise (heavy quenching), each successful stepwise iz "wastes"
        # many excitation events, so the effective cost per ion is huge.
        
        Riz_total = Riz_electron  # per ne
        Eloss = ((16*k['iz18']+20*k['iz19']+20.5*k['iz20']+28*k['iz21']+37.5*k['iz22']+18*k['iz23']+29*k['iz24'])*nSF6+
                 (9.6*k['d1']+12.1*k['d2']+16*k['d3']+18.6*k['d4']+22.7*k['d5'])*nSF6+
                 .09*k['vib_SF6']*nSF6+k['el_SF6']*nSF6*3*m_e/(M['SF6']*AMU)*Te+
                 (11*k['iz25']+15*k['iz26'])*nSF5+5*k['d7']*nSF5+
                 11*k['iz27']*nSF3+5*k['d9']*nSF3+
                 15*k['iz28']*nF+14.4*k['exc_F']*nF+k['el_F']*nF*3*m_e/(M['F']*AMU)*Te+
                 3.2*k['d6']*nF2+.11*k['vib_F2']*nF2+
                 10*k['iz29']*nS+16*k['Ar_iz']*nAr0+Ar_exc_loss+k['Ar_el']*nAr0*3*m_e/(M['Ar']*AMU)*Te+
                 (12+4.95)*k['Ar_iz_m']*nArm)  # stepwise iz: full 2-step cost
        Ec = np.clip(Eloss/max(Riz_total,1e-30), 10, 2000)
        eiw = .5*Te*np.log(max(M['SF5']*AMU/(2*pi*m_e),1)); eT = Ec+eiw+2*Te
        
        # ── Power balance ──
        # P_abs = (Riz_electron * ne * eT + R_Penn_vol * eiw_penn) * eC * V
        # Penning ionization does NOT cost electron energy (it's powered by Ar* internal energy)
        # but the resulting ions still hit walls, so ion wall energy is still lost
        # The electron produced by Penning is cold (~0 eV) and gets heated by the RF field
        # Net: Penning doesn't directly appear in electron power balance except through
        # the extra ionization it provides (wall losses of resulting ions)
        # 
        # Correct power balance:
        # P_abs = ne * Riz_electron * Ec * eC * V + (ne * Riz_electron + R_Penn_vol) * (eiw+2*Te) * eC * V
        # Wait — more carefully:
        # P_abs = Ec_loss_vol + wall_loss_vol
        # Ec_loss_vol = ne * Riz_total * Ec  (only electron-impact contributes to Ec)
        # Actually Ec already accounts for energy per electron-impact ionization
        # Wall loss = total_ion_loss_rate * (eiw + 2Te)
        # Total ion production = ne*Riz_electron + R_Penn_vol
        # At SS: total_ion_production = total_ion_loss (wall + recombination)
        # Wall ion loss ≈ (ne*Riz_electron + R_Penn_vol) * fraction_to_wall
        #
        # Simplification: P_abs ≈ ne * (Riz_electron * eT) * eC * V 
        #                        + R_Penn_vol * (eiw + 2*Te) * eC * V
        # The second term accounts for wall energy from Penning-produced ions
        
        eT_wall = eiw + 2*Te  # wall energy per ion-electron pair
        ne_new_numer = P_abs - R_Penn_vol * eT_wall * eC * R.V  # subtract Penning contribution
        ne_new_numer = max(ne_new_numer, P_abs * 0.1)  # don't let Penning consume all power
        ne_new = ne_new_numer / (Riz_electron * eT * eC * R.V) if Riz_electron>0 and eT>0 else 1e15
        ne_new = np.clip(ne_new, 1e10, 1e19)
        
        # ── Te from particle balance ──
        # Total ionization (electron + Penning) = total loss (wall + recombination)
        # (Riz_electron(Te) * ne + R_Penn(Te)) = kw*(1+alpha)*ne + k_rec*alpha*ne*(1+alpha)*ne
        # But for the brentq, we need to include Penning:
        def Tf(T):
            kk=rates(T)
            Ri=kk['iz_SF6_total']*nSF6+(kk['iz25']+kk['iz26'])*nSF5+kk['iz27']*nSF3+kk['iz28']*nF+kk['iz29']*nS+kk['Ar_iz']*nAr0
            Ra=kk['att_SF6_total']*nSF6
            kw=kw_i(T, M['SF5'], alpha, T_neg, R, ng)
            
            # Ar* density at this Te (self-consistent with quenching)
            R_qh = (kk['Penn_SF6']*nSF6 + kk['qnch_SF6']*nSF6 + 
                    kk['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                    kk['qnch_F2']*nF2 + kk['qnch_F']*nF)
            nArm_t = kk['Ar_exc']*ne*nAr0 / ((kk['Ar_iz_m']+kk['Ar_q'])*ne + k_Arm_wall + R_qh + 1e-30)
            
            # Add stepwise ionization from Ar*
            Ri += kk['Ar_iz_m']*nArm_t
            
            # Penning ionization rate / ne (to normalize)
            R_Penn_per_ne = kk['Penn_SF6']*nArm_t*nSF6 / max(ne, 1e10)
            
            # Particle balance: Riz + R_Penn/ne = Ratt + kw*(1+alpha)
            return Ri + R_Penn_per_ne - Ra - kw*(1+alpha)
        
        try: Te_new = brentq(Tf, 0.5, 15, xtol=.005)
        except: Te_new = Te
        
        # Adaptive relaxation
        w = 0.08
        Te = Te + w*(Te_new-Te)
        ne = ne * (ne_new/ne)**w
        alpha = alpha * (max(alpha_new,0.001)/max(alpha,0.001))**w
        Te = np.clip(Te, 0.5, 15); ne = np.clip(ne, 1e10, 1e19)
        alpha = np.clip(alpha, 0, 5000)
        
        if outer > 50:
            dTe = abs(Te-Te_old)/(Te_old+.1)
            dne = abs(ne-ne_old)/(ne_old+1e10)
            dal = abs(alpha-al_old)/(al_old+.1)
            if dTe<5e-5 and dne<5e-4 and dal<1e-3:
                converged = True; break
    
    dissoc = 1-nSF6/max(nSF6_0,1) if nSF6_0>0 else 1
    return {'Te':Te,'ne':ne,'alpha':alpha,'n_SF6':nSF6,'n_SF5':nSF5,'n_SF4':nSF4,
            'n_SF3':nSF3,'n_SF2':nSF2,'n_SF':nSF,'n_S':nS,'n_F':nF,'n_F2':nF2,
            'Ec':Ec,'eps_T':eT,'dissoc_frac':dissoc,'converged':converged,'iter':outer,
            'nArm':nArm,'nAr0':nAr0,'R_Penning':R_Penning,'R_wall_SF6':R_wall_SF6,
            'ns':{'SF6':nSF6,'SF5':nSF5,'SF4':nSF4,'SF3':nSF3,'SF2':nSF2,'SF':nSF,'S':nS,'F':nF,'F2':nF2}}

def sweep_with_continuation(param_name, values, base_kwargs, verbose=True):
    results = []; prev = None
    for v in values:
        kw = base_kwargs.copy(); kw[param_name] = v
        if prev:
            kw['init_Te'] = prev['Te']; kw['init_ne'] = prev['ne']
            kw['init_alpha'] = prev['alpha']; kw['init_ns'] = prev['ns']
        r = solve_model(**kw)
        r[param_name] = v; results.append(r)
        if r['converged']: prev = r
        if verbose:
            c='✓' if r['converged'] else '✗'
            penn_str = f" Penn={r['R_Penning']:.1e}" if r.get('R_Penning',0)>0 else ""
            print(f"  {c} {param_name}={v}: ne={r['ne']*1e-6:.2e} Te={r['Te']:.2f} [F]={r['n_F']*1e-6:.2e} α={r['alpha']:.1f} SF6%={(1-r['dissoc_frac'])*100:.0f}%{penn_str}")
    return results

if __name__=='__main__':
    print("="*65)
    print("SF6/Ar Global Model — With Penning Ionization")
    print("="*65)
    
    base = dict(P_rf=1500, p_mTorr=10, frac_Ar=0., Q_sccm=40, eta=0.12)
    
    # Quick validation: pure SF6 should be unchanged
    print("\n▶ Single point: 1500W, 10 mTorr, pure SF6")
    r = solve_model(**base)
    print(f"  ne={r['ne']*1e-6:.2e} Te={r['Te']:.2f} α={r['alpha']:.1f} nArm={r['nArm']:.2e} R_Penn={r['R_Penning']:.2e}")
    
    # Check 50% Ar to see Penning effect
    print("\n▶ Single point: 1500W, 10 mTorr, 50% Ar")
    r = solve_model(P_rf=1500, p_mTorr=10, frac_Ar=0.5, eta=0.12)
    print(f"  ne={r['ne']*1e-6:.2e} Te={r['Te']:.2f} α={r['alpha']:.1f} nArm={r['nArm']:.2e} R_Penn={r['R_Penning']:.2e}")
    
    print("\n▶ Power sweep (200–2000 W)")
    res_P = sweep_with_continuation('P_rf', np.linspace(200,2000,37), base)
    
    print("\n▶ Ar fraction sweep (0–100%)")
    res_Ar = sweep_with_continuation('frac_Ar', np.linspace(0,1.0,21), base)
    
    print("\n▶ Alpha vs Ar at 3 pressures")
    res_alpha = {}
    for p in [5, 10, 20]:
        print(f"  p={p} mTorr:")
        b = base.copy(); b['p_mTorr'] = p
        res_alpha[p] = sweep_with_continuation('frac_Ar', np.linspace(0,0.8,17), b, verbose=False)
        rl = res_alpha[p]
        print(f"    α(0%)={rl[0]['alpha']:.1f}  α(40%)={rl[8]['alpha']:.1f}  α(80%)={rl[-1]['alpha']:.1f}")
    
    # ── Plots ──
    P_vals = [r['P_rf'] for r in res_P]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0,0].plot(P_vals, [r['ne']*1e-6 for r in res_P], 'bs-', lw=2, ms=4)
    axes[0,0].set_ylabel('$n_e$ (cm$^{-3}$)'); axes[0,0].set_xlabel('Power (W)')
    axes[0,0].set_title('(a) $n_e$ vs power — 10 mTorr'); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(P_vals, [r['Te'] for r in res_P], 'rs-', lw=2, ms=4)
    axes[0,1].set_ylabel('$T_e$ (eV)'); axes[0,1].set_xlabel('Power (W)')
    axes[0,1].set_title('(b) $T_e$ vs power'); axes[0,1].set_ylim(0,5); axes[0,1].grid(True, alpha=0.3)
    fAr = [r['frac_Ar']*100 for r in res_Ar]
    axes[1,0].plot(fAr, [r['ne']*1e-6 for r in res_Ar], 'bs-', lw=2, ms=4)
    axes[1,0].set_ylabel('$n_e$ (cm$^{-3}$)'); axes[1,0].set_xlabel('Ar%')
    axes[1,0].set_title('(c) $n_e$ vs Ar%'); axes[1,0].set_yscale('log'); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(fAr, [r['Te'] for r in res_Ar], 'rs-', lw=2, ms=4)
    axes[1,1].set_ylabel('$T_e$ (eV)'); axes[1,1].set_xlabel('Ar%')
    axes[1,1].set_title('(d) $T_e$ vs Ar%'); axes[1,1].set_ylim(0,5); axes[1,1].grid(True, alpha=0.3)
    fig.suptitle('Figure 5 — With Penning Ionization', fontsize=13)
    plt.tight_layout(); fig.savefig('/mnt/user-data/outputs/fig5_reproduction.png', dpi=150, bbox_inches='tight')
    
    fig7, ax = plt.subplots(figsize=(8,6))
    for p, rl in res_alpha.items():
        ax.plot([r['frac_Ar']*100 for r in rl], [r['alpha'] for r in rl], 'o-', lw=2, ms=5, label=f'{p} mTorr')
    ax.set_xlabel('Ar%'); ax.set_ylabel('α'); ax.set_title('Figure 7 — Electronegativity (1500 W)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig7.savefig('/mnt/user-data/outputs/fig7_reproduction.png', dpi=150, bbox_inches='tight')
    
    fig8, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(P_vals, [r['n_F']*1e-6 for r in res_P], 'ro-', lw=2, ms=5, label='[F]')
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel('[F] (cm$^{-3}$)', color='r')
    ax2 = ax1.twinx(); ax2.plot(P_vals, [r['ne']*1e-6 for r in res_P], 'bs-', lw=2, ms=5, label='$n_e$')
    ax2.set_ylabel('$n_e$ (cm$^{-3}$)', color='b')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax1.set_title('Figure 8 — [F] and $n_e$ vs power'); ax1.grid(True, alpha=0.3)
    plt.tight_layout(); fig8.savefig('/mnt/user-data/outputs/fig8_reproduction.png', dpi=150, bbox_inches='tight')
    
    fig_sp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for sp,key in [('SF$_6$','n_SF6'),('SF$_5$','n_SF5'),('SF$_4$','n_SF4'),('SF$_3$','n_SF3'),('F','n_F'),('F$_2$','n_F2'),('S','n_S')]:
        vals = [r[key]*1e-6 for r in res_P]
        if max(vals)>0: ax1.plot(P_vals, vals, 'o-', lw=1.5, ms=4, label=sp)
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel('Density (cm$^{-3}$)'); ax1.set_title('Neutral densities')
    ax1.legend(ncol=2); ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)
    ax2.plot(P_vals, [r['alpha'] for r in res_P], 'go-', lw=2, ms=5)
    ax2.set_xlabel('Power (W)'); ax2.set_ylabel('α'); ax2.set_title('α vs power'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); fig_sp.savefig('/mnt/user-data/outputs/species_overview.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ All figures saved")
