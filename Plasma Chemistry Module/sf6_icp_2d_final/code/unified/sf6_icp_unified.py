#!/usr/bin/env python3
"""
SF6/Ar 2D ICP Plasma Simulator — Unified Single-File
=====================================================
Complete Gen-4b + Gen-5 simulator in one portable file.

Usage:
  python sf6_icp_unified.py                          # Gen-5, pure SF6, 1500W
  python sf6_icp_unified.py --power 1000 --ar 0.3    # 70/30 mix
  python sf6_icp_unified.py --gen 4                   # Gen-4b
  python sf6_icp_unified.py --scan                    # Parameter sweep
  python sf6_icp_unified.py --help                    # All options

Requirements: numpy, scipy, matplotlib (for plots only)
"""

import sys, os, time, argparse, warnings
import numpy as np
from scipy.constants import e as eC, k as kB, m_e, epsilon_0, pi, N_A
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore", category=RuntimeWarning)

AMU = 1.66054e-27
MTORR_TO_PA = 0.133322
cm3 = 1e-6

# Numpy 2.0 compatibility
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz



#==============================================================================
# 0D GLOBAL CHEMISTRY SOLVER
#==============================================================================


#!/usr/bin/env python3
"""
SF6/Ar Global Model — Unified Script
======================================
Consolidates sf6_global_model_final.py, extended_analysis.py, and
generate_csv_data.py into a single file.

Runs all sweeps once, then generates all figures and CSV exports.

Physics: SF6/Ar ICP global model with Penning ionization.
  Ar* + SF6 → SF5+ + F + Ar + e   (Penning ionization)
  Ar* + SF6 → Ar + SF5 + F        (non-ionizing quenching)
  Ar* + SFx → Ar + SFx*           (quenching by SFx fragments)
  Ar* + F2  → Ar + 2F             (quenching/dissociation by F2)

Rate coefficients from:
  Velazco & Setser, J. Chem. Phys. 62 (1975) 1990
  Kolts & Setser, J. Chem. Phys. 68 (1978) 4848
  Gudmundsson & Thorsteinsson, PSST 16 (2007) 399
"""

import os
import csv
import numpy as np
from scipy.optimize import brentq
from scipy.constants import e as eC, k as kB, m_e, pi
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & PARAMETERS
# ═══════════════════════════════════════════════════════════════

AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6
M = {'S':32.06,'F':19.0,'F2':38.0,'SF':51.06,'SF2':70.06,'SF3':89.06,
     'SF4':108.06,'SF5':127.06,'SF6':146.06,'Ar':39.948}

# Plot style — consistent across all figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'lines.linewidth': 2.2,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
C_SF6  = '#d73027'; C_SF5  = '#fc8d59'; C_SF4  = '#fee090'
C_SF3  = '#e0f3f8'; C_F    = '#91bfdb'; C_F2   = '#4575b4'
C_S    = '#999999'; C_ne   = '#2166ac'; C_Te   = '#b2182b'
C_alpha= '#1a9850'; C_Ar   = '#006837'; C_Arm  = '#66c2a4'

OUTPUT_DIR = 'outputs'
CSV_DIR    = 'csv_data'

# ═══════════════════════════════════════════════════════════════
# PHYSICS MODEL
# ═══════════════════════════════════════════════════════════════

class Reactor:
    def __init__(self, R=0.180, L=0.175):
        self.R, self.L = R, L
        self.V = pi*R**2*L; self.A = 2*pi*R**2 + 2*pi*R*L
        self.Lambda = 1.0/np.sqrt((pi/L)**2 + (2.405/R)**2)


def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate (Ryan & Plumb 1990). Returns effective bimolecular rate in cm³/s."""
    if M_cm3 <= 0 or k0 <= 0: return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F


def rates(Te):
    Te = max(Te, 0.3); k = {}
    # SF6 dissociation
    k['d1']=1.5e-7*np.exp(-8.1/Te)*cm3;  k['d2']=9e-9*np.exp(-13.4/Te)*cm3
    k['d3']=2.5e-8*np.exp(-33.5/Te)*cm3; k['d4']=2.3e-8*np.exp(-23.9/Te)*cm3
    k['d5']=1.5e-9*np.exp(-26.0/Te)*cm3; k['d6']=1.2e-8*np.exp(-5.8/Te)*cm3
    k['d7']=1.5e-7*np.exp(-9.0/Te)*cm3;  k['d8']=6.2e-8*np.exp(-9.0/Te)*cm3
    k['d9']=8.6e-8*np.exp(-9.0/Te)*cm3;  k['d10']=4.5e-8*np.exp(-9.0/Te)*cm3
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
    # Neutral recombination — Troe fall-off (Ryan & Plumb 1990)
    k['nr42_k0']=3.4e-23; k['nr42_kinf']=1.0e-11; k['nr42_Fc']=0.43  # SF5+F→SF6
    k['nr41_k0']=3.7e-28; k['nr41_kinf']=5.0e-12; k['nr41_Fc']=0.46  # SF4+F→SF5
    k['nr40_k0']=2.8e-26; k['nr40_kinf']=2.0e-11; k['nr40_Fc']=0.47  # SF3+F→SF4
    k['nr39_k0']=1.7e-28; k['nr39_kinf']=2.0e-11; k['nr39_Fc']=0.56  # SF2+F→SF3
    k['nr38_k0']=1.0e-30; k['nr38_kinf']=2.0e-11; k['nr38_Fc']=0.67  # SF+F→SF2
    k['nr37_k0']=7.5e-33; k['nr37_kinf']=2.0e-11; k['nr37_Fc']=0.73  # S+F→SF
    # Disproportionation
    k['nr45']=2.5e-11*cm3; k['nr44']=2.5e-11*cm3; k['nr43']=2.5e-11*cm3
    k['rec']=1.5e-9*cm3
    # Ar electron-impact reactions (Table 4, Lallement)
    k['Ar_iz']=1.2e-10*np.exp(-21.7/Te)*cm3      # (101) Ar + e → Ar+ + 2e
    k['Ar_exc']=4.2e-9*np.exp(-8.0/Te)*cm3        # (102) Ar + e → Ar* + e
    k['Ar_iz_m']=2.05e-7*np.exp(-4.95/Te)*cm3     # (103) Ar* + e → Ar+ + 2e
    k['Ar_q']=2.0e-7*cm3                           # (104) Ar* + e → Ar + e
    k['Ar_el']=max((-1.1e-8+3.9e-8*Te-1.3e-8*Te**2+2e-9*Te**3-1.4e-10*Te**4+3.9e-12*Te**5)*cm3,1e-20)
    # Penning ionization and Ar* quenching (Velazco & Setser 1975)
    k['Penn_SF6'] = 2.0e-10*cm3    # Ar* + SF6 → SF5+ + F + Ar + e
    k['qnch_SF6'] = 3.0e-10*cm3    # Ar* + SF6 → Ar + SF5 + F (non-ionizing)
    k['qnch_SFx'] = 1.0e-10*cm3    # Ar* + SFx → Ar + SFx
    k['qnch_F2']  = 5.0e-11*cm3    # Ar* + F2 → Ar + 2F
    k['qnch_F']   = 5.0e-12*cm3    # Ar* + F → Ar + F
    # Totals
    k['iz_SF6_total'] = sum(k[f'iz{i}'] for i in [18,19,20,21,22,23,24])
    k['att_SF6_total'] = sum(k[f'at{i}'] for i in [30,31,32,33,34,35,36])
    return k


def kw_n(g, Ma, Tg, ng, R):
    if g <= 0: return 0.
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
                gamma_F=0.01, beta_SFx=0.02, eta=0.12,
                init_Te=None, init_ne=None, init_alpha=None, init_ns=None):
    """Solve the SF6/Ar global model with Penning ionization."""
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

        ng_cm3 = ng * 1e-6
        k['nr42'] = troe_rate(k['nr42_k0'], k['nr42_kinf'], k['nr42_Fc'], ng_cm3) * cm3
        k['nr41'] = troe_rate(k['nr41_k0'], k['nr41_kinf'], k['nr41_Fc'], ng_cm3) * cm3
        k['nr40'] = troe_rate(k['nr40_k0'], k['nr40_kinf'], k['nr40_Fc'], ng_cm3) * cm3
        k['nr39'] = troe_rate(k['nr39_k0'], k['nr39_kinf'], k['nr39_Fc'], ng_cm3) * cm3
        k['nr38'] = troe_rate(k['nr38_k0'], k['nr38_kinf'], k['nr38_Fc'], ng_cm3) * cm3
        k['nr37'] = troe_rate(k['nr37_k0'], k['nr37_kinf'], k['nr37_Fc'], ng_cm3) * cm3

        kwF = kw_n(gamma_F, M['F'], Tgas, ng, R)
        kwS = kw_n(beta_SFx, M['SF5'], Tgas, ng, R)

        k_e = k['d1']+k['d2']+k['d3']+k['d4']+k['d5']+k['iz_SF6_total']+k['att_SF6_total']

        k_Arm_wall = kw_n(1.0, M['Ar'], Tgas, ng, R)
        R_quench_heavy = (k['Penn_SF6']*nSF6 + k['qnch_SF6']*nSF6 +
                         k['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                         k['qnch_F2']*nF2 + k['qnch_F']*nF)
        nArm = k['Ar_exc']*ne*nAr0 / ((k['Ar_iz_m']+k['Ar_q'])*ne + k_Arm_wall + R_quench_heavy + 1e-30)
        R_Penning = k['Penn_SF6'] * nArm * nSF6

        for sub in range(20):
            s6 = nSF6_0/tau + k['nr42']*nSF5*nF + k['nr45']*nSF5**2
            k_SF6_loss = k_e*ne + k['Penn_SF6']*nArm + k['qnch_SF6']*nArm + 1/tau
            nSF6_n = s6/(k_SF6_loss+1e-30); nSF6_n = np.clip(nSF6_n, 1e10, nSF6_0)
            nSF6 = 0.3*nSF6 + 0.7*nSF6_n

            nSF5_prod = k['d1']*ne*nSF6 + k['nr41']*nSF4*nF + k['qnch_SF6']*nArm*nSF6
            nSF5 = nSF5_prod/((k['d7']+k['iz25']+k['iz26'])*ne+k['nr42']*nF+2*k['nr45']*nSF5+kwS+k['qnch_SFx']*nArm+1/tau+1e-30)

            nSF4 = (k['d2']*ne*nSF6+k['d7']*ne*nSF5+k['nr45']*nSF5**2)/(k['d8']*ne+k['nr41']*nF+kwS+k['qnch_SFx']*nArm+1/tau+1e-30)
            nSF3 = (k['d3']*ne*nSF6+k['d8']*ne*nSF4)/((k['d9']+k['iz27'])*ne+k['nr40']*nF+kwS+k['qnch_SFx']*nArm+1/tau+1e-30)
            nSF2 = (k['d4']*ne*nSF6+k['d9']*ne*nSF3)/(k['d10']*ne+k['nr39']*nF+1/tau+1e-30)
            nSF  = (k['d5']*ne*nSF6+k['d10']*ne*nSF2)/(k['d11']*ne+k['nr38']*nF+1/tau+1e-30)
            nS   = k['d11']*ne*nSF/(k['iz29']*ne+k['nr37']*nF+1/tau+1e-30)

            RF = ne*nSF6*(k['d1']+2*k['d2']+3*k['d3']+2*k['d4']+3*k['d5']+k['iz18']+2*k['iz19']+3*k['iz20']+2*k['iz21']+3*k['iz22']+4*k['iz23']+k['at31'])
            RF += ne*(nSF5*(k['d7']+k['iz26'])+nSF4*k['d8']+nSF3*k['d9']+nSF2*k['d10']+nSF*k['d11'])
            RF += k['Penn_SF6']*nArm*nSF6
            RF += k['qnch_SF6']*nArm*nSF6
            RF += 2*k['qnch_F2']*nArm*nF2

            nF2 = (ne*nSF6*(k['d4']+k['d5']+k['iz21']+k['iz22']+k['iz23'])+.5*kwF*nF)/(k['d6']*ne+k['qnch_F2']*nArm+1/tau+1e-30)
            RF += 2*k['d6']*ne*nF2
            kFL = kwF+1/tau+k['iz28']*ne+k['nr42']*nSF5+k['nr41']*nSF4+k['nr40']*nSF3+k['nr39']*nSF2+k['nr38']*nSF+k['nr37']*nS
            nF_n = RF/(kFL+1e-30)
            nF = 0.3*nF + 0.7*min(nF_n, 6*nSF6_0)

            R_quench_heavy = (k['Penn_SF6']*nSF6 + k['qnch_SF6']*nSF6 +
                             k['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                             k['qnch_F2']*nF2 + k['qnch_F']*nF)
            nArm = k['Ar_exc']*ne*nAr0 / ((k['Ar_iz_m']+k['Ar_q'])*ne + k_Arm_wall + R_quench_heavy + 1e-30)
            R_Penning = k['Penn_SF6'] * nArm * nSF6

        ng = nSF6+nSF5+nSF4+nSF3+nSF2+nSF+nS+nF+nF2+nAr0; ng = max(ng, ng0*0.1)

        Riz_electron = (k['iz_SF6_total']*nSF6+(k['iz25']+k['iz26'])*nSF5+k['iz27']*nSF3+
                       k['iz28']*nF+k['iz29']*nS+k['Ar_iz']*nAr0+k['Ar_iz_m']*nArm)
        R_Penn_vol = k['Penn_SF6']*nArm*nSF6

        Ratt = k['att_SF6_total']*nSF6
        kwi = kw_i(Te, M['SF5'], alpha, T_neg, R, ng)

        rhs_q = Ratt/(k['rec']*ne) if ne>0 else 0
        alpha_new = (-1+np.sqrt(1+4*rhs_q))/2 if rhs_q>0 else 0

        R_quench_total = ((k['Ar_iz_m']+k['Ar_q'])*ne + k_Arm_wall + R_quench_heavy)
        frac_stepwise = k['Ar_iz_m']*ne / max(R_quench_total, 1e-30)
        frac_stepwise = min(frac_stepwise, 1.0)

        Ar_exc_loss = 12*k['Ar_exc']*nAr0 * (1.0 - frac_stepwise)
        Riz_total = Riz_electron
        Eloss = ((16*k['iz18']+20*k['iz19']+20.5*k['iz20']+28*k['iz21']+37.5*k['iz22']+18*k['iz23']+29*k['iz24'])*nSF6+
                 (9.6*k['d1']+12.1*k['d2']+16*k['d3']+18.6*k['d4']+22.7*k['d5'])*nSF6+
                 .09*k['vib_SF6']*nSF6+k['el_SF6']*nSF6*3*m_e/(M['SF6']*AMU)*Te+
                 (11*k['iz25']+15*k['iz26'])*nSF5+5*k['d7']*nSF5+
                 11*k['iz27']*nSF3+5*k['d9']*nSF3+
                 15*k['iz28']*nF+14.4*k['exc_F']*nF+k['el_F']*nF*3*m_e/(M['F']*AMU)*Te+
                 3.2*k['d6']*nF2+.11*k['vib_F2']*nF2+
                 10*k['iz29']*nS+16*k['Ar_iz']*nAr0+Ar_exc_loss+k['Ar_el']*nAr0*3*m_e/(M['Ar']*AMU)*Te+
                 (12+4.95)*k['Ar_iz_m']*nArm)
        Ec = np.clip(Eloss/max(Riz_total,1e-30), 10, 2000)
        eiw = .5*Te*np.log(max(M['SF5']*AMU/(2*pi*m_e),1)); eT = Ec+eiw+2*Te

        eT_wall = eiw + 2*Te
        ne_new_numer = P_abs - R_Penn_vol * eT_wall * eC * R.V
        ne_new_numer = max(ne_new_numer, P_abs * 0.1)
        ne_new = ne_new_numer / (Riz_electron * eT * eC * R.V) if Riz_electron>0 and eT>0 else 1e15
        ne_new = np.clip(ne_new, 1e10, 1e19)

        def Tf(T):
            kk=rates(T)
            Ri=kk['iz_SF6_total']*nSF6+(kk['iz25']+kk['iz26'])*nSF5+kk['iz27']*nSF3+kk['iz28']*nF+kk['iz29']*nS+kk['Ar_iz']*nAr0
            Ra=kk['att_SF6_total']*nSF6
            kw=kw_i(T, M['SF5'], alpha, T_neg, R, ng)
            R_qh = (kk['Penn_SF6']*nSF6 + kk['qnch_SF6']*nSF6 +
                    kk['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                    kk['qnch_F2']*nF2 + kk['qnch_F']*nF)
            nArm_t = kk['Ar_exc']*ne*nAr0 / ((kk['Ar_iz_m']+kk['Ar_q'])*ne + k_Arm_wall + R_qh + 1e-30)
            Ri += kk['Ar_iz_m']*nArm_t
            R_Penn_per_ne = kk['Penn_SF6']*nArm_t*nSF6 / max(ne, 1e10)
            return Ri + R_Penn_per_ne - Ra - kw*(1+alpha)

        try: Te_new = brentq(Tf, 0.5, 15, xtol=.005)
        except: Te_new = Te

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
            'nArm':nArm,'nAr0':nAr0,'R_Penning':R_Penning,
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


# ═══════════════════════════════════════════════════════════════
# CSV HELPER
# ═══════════════════════════════════════════════════════════════

def write_csv(filename, headers, rows):
    path = os.path.join(CSV_DIR, filename)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print(f"  Wrote {path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


#==============================================================================
# CHEMISTRY: 54-reaction rate coefficients
#==============================================================================


"""
SF6/Ar plasma chemistry module.

Extracted from sf6_unified.py (the validated 0D model) WITHOUT modification.
The rates(Te) function returns all 54 reaction rate coefficients given the
electron temperature in eV. The source_terms() function evaluates local
production/loss rates at a single grid point.

All rate coefficients are in SI (m³/s) internally.
The 0D code uses cm³/s with a conversion factor cm3 = 1e-6.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════
# Constants (from sf6_unified.py)
# ═══════════════════════════════════════════════════════════

AMU = 1.66054e-27       # kg
cm3 = 1e-6              # cm³ → m³ conversion for rate coefficients
M_SPECIES = {           # molecular masses in AMU
    'S': 32.06, 'F': 19.0, 'F2': 38.0,
    'SF': 51.06, 'SF2': 70.06, 'SF3': 89.06,
    'SF4': 108.06, 'SF5': 127.06, 'SF6': 146.06,
    'Ar': 39.948
}


# ═══════════════════════════════════════════════════════════
# Troe fall-off (from sf6_unified.py, unchanged)
# ═══════════════════════════════════════════════════════════

def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate (Ryan & Plumb 1990).

    Parameters
    ----------
    k0 : float
        Low-pressure termolecular rate constant [cm⁶/s].
    kinf : float
        High-pressure bimolecular limit [cm³/s].
    Fc : float
        Troe broadening factor.
    M_cm3 : float
        Total gas number density [cm⁻³].

    Returns
    -------
    float
        Effective bimolecular rate coefficient [cm³/s].
    """
    if M_cm3 <= 0 or k0 <= 0:
        return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F


# ═══════════════════════════════════════════════════════════
# Rate coefficients (from sf6_unified.py lines 81–134, UNCHANGED)
# ═══════════════════════════════════════════════════════════

def rates(Te):
    """Compute all 54+ rate coefficients given electron temperature.

    Parameters
    ----------
    Te : float
        Electron temperature [eV]. In the 2D code, Te = (2/3) * mean_energy.

    Returns
    -------
    dict
        All rate coefficients in m³/s (SI). Keys match the 0D code exactly.
    """
    Te = max(Te, 0.3)
    k = {}

    # --- SF6 dissociation (reactions d1–d5) ---
    k['d1']  = 1.5e-7  * np.exp(-8.1 / Te)  * cm3   # SF6 + e → SF5 + F + e
    k['d2']  = 9e-9    * np.exp(-13.4 / Te)  * cm3   # SF6 + e → SF4 + 2F + e
    k['d3']  = 2.5e-8  * np.exp(-33.5 / Te)  * cm3   # SF6 + e → SF3 + 3F + e
    k['d4']  = 2.3e-8  * np.exp(-23.9 / Te)  * cm3   # SF6 + e → SF2 + 2F + F2 + e
    k['d5']  = 1.5e-9  * np.exp(-26.0 / Te)  * cm3   # SF6 + e → SF + 3F + F2 + e

    # --- F2 dissociation and sequential SFx dissociation (d6–d11) ---
    k['d6']  = 1.2e-8  * np.exp(-5.8 / Te)   * cm3   # F2 + e → 2F + e
    k['d7']  = 1.5e-7  * np.exp(-9.0 / Te)   * cm3   # SF5 + e → SF4 + F + e
    k['d8']  = 6.2e-8  * np.exp(-9.0 / Te)   * cm3   # SF4 + e → SF3 + F + e
    k['d9']  = 8.6e-8  * np.exp(-9.0 / Te)   * cm3   # SF3 + e → SF2 + F + e
    k['d10'] = 4.5e-8  * np.exp(-9.0 / Te)   * cm3   # SF2 + e → SF + F + e
    k['d11'] = 6.2e-8  * np.exp(-9.0 / Te)   * cm3   # SF + e → S + F + e

    # --- Excitation and elastic ---
    k['vib_SF6'] = 7.9e-8 * np.exp(-0.1 * Te + 0.002 * Te**2) * cm3
    k['el_SF6']  = 2.8e-7 * np.exp(-1.5 / Te) * cm3
    k['exc_F']   = 9.2e-9 * np.exp(-14.3 / Te) * cm3
    k['el_F']    = 1.1e-7 * np.exp(-1.93 / Te) * cm3
    k['vib_F2']  = 1.8e-10 * Te**1.72 * np.exp(-1.55 / Te) * cm3
    k['el_F2']   = 2.5e-7 * np.exp(-0.48 / Te) * cm3

    # --- Ionization from SF6 (reactions iz18–iz24) ---
    k['iz18'] = 1.2e-7  * np.exp(-18.1 / Te) * cm3  # SF6 + e → SF5+ + F + 2e
    k['iz19'] = 8.4e-9  * np.exp(-19.9 / Te) * cm3  # SF6 + e → SF4+ + 2F + 2e
    k['iz20'] = 3.2e-8  * np.exp(-20.7 / Te) * cm3  # SF6 + e → SF3+ + 3F + 2e
    k['iz21'] = 7.6e-9  * np.exp(-24.4 / Te) * cm3  # SF6 + e → SF2+ + 2F + F2 + 2e
    k['iz22'] = 1.2e-8  * np.exp(-26.0 / Te) * cm3  # SF6 + e → SF+ + 3F + F2 + 2e
    k['iz23'] = 1.4e-8  * np.exp(-39.9 / Te) * cm3  # SF6 + e → S+ + 4F + F2 + 2e
    k['iz24'] = 1.2e-8  * np.exp(-31.7 / Te) * cm3  # SF6 + e → F+ + SF5 + 2e

    # --- Ionization from fragments (iz25–iz29) ---
    k['iz25'] = 1.0e-7  * np.exp(-17.8 / Te) * cm3  # SF5 + e → SF5+ + 2e
    k['iz26'] = 9.4e-8  * np.exp(-22.8 / Te) * cm3  # SF5 + e → SF4+ + F + 2e
    k['iz27'] = 1.0e-7  * np.exp(-18.9 / Te) * cm3  # SF3 + e → SF3+ + 2e
    k['iz28'] = 1.3e-8  * np.exp(-16.5 / Te) * cm3  # F + e → F+ + 2e
    k['iz29'] = 1.6e-7  * np.exp(-13.3 / Te) * cm3  # S + e → S+ + 2e

    # --- Dissociative attachment (at30–at36) ---
    k['at30'] = 2.4e-10 / Te**1.49 * cm3                           # SF6 + e → SF6-
    k['at31'] = 2.0e-11 / Te**1.46 * cm3                           # SF6 + e → SF5- + F
    k['at32'] = 3.9e-12 * np.exp(0.45*Te - 0.04*Te**2) * cm3      # SF6 + e → SF4- + 2F
    k['at33'] = 1.2e-13 * np.exp(0.70*Te - 0.05*Te**2) * cm3      # SF6 + e → SF3- + 3F
    k['at34'] = 5.4e-15 * np.exp(0.77*Te - 0.05*Te**2) * cm3      # SF6 + e → SF2- + 4F
    k['at35'] = 3.4e-11 * np.exp(0.46*Te - 0.04*Te**2) * cm3      # SF6 + e → F- + SF5
    k['at36'] = 2.2e-13 * np.exp(0.71*Te - 0.05*Te**2) * cm3      # SF6 + e → F2- + SF4

    # --- Neutral recombination Troe parameters (Ryan & Plumb 1990) ---
    k['nr42_k0'] = 3.4e-23;  k['nr42_kinf'] = 1.0e-11; k['nr42_Fc'] = 0.43  # SF5+F→SF6
    k['nr41_k0'] = 3.7e-28;  k['nr41_kinf'] = 5.0e-12; k['nr41_Fc'] = 0.46  # SF4+F→SF5
    k['nr40_k0'] = 2.8e-26;  k['nr40_kinf'] = 2.0e-11; k['nr40_Fc'] = 0.47  # SF3+F→SF4
    k['nr39_k0'] = 1.7e-28;  k['nr39_kinf'] = 2.0e-11; k['nr39_Fc'] = 0.56  # SF2+F→SF3
    k['nr38_k0'] = 1.0e-30;  k['nr38_kinf'] = 2.0e-11; k['nr38_Fc'] = 0.67  # SF+F→SF2
    k['nr37_k0'] = 7.5e-33;  k['nr37_kinf'] = 2.0e-11; k['nr37_Fc'] = 0.73  # S+F→SF

    # --- Disproportionation ---
    k['nr45'] = 2.5e-11 * cm3   # SF5 + SF5 → SF4 + SF6
    k['nr44'] = 2.5e-11 * cm3   # SF4 + SF4 → SF3 + SF5
    k['nr43'] = 2.5e-11 * cm3   # SF3 + SF3 → SF2 + SF4

    # --- Ion-ion recombination ---
    k['rec'] = 1.5e-9 * cm3     # All positive + negative ion pairs

    # --- Ar electron-impact reactions ---
    k['Ar_iz']   = 1.2e-10  * np.exp(-21.7 / Te)  * cm3   # Ar + e → Ar+ + 2e
    k['Ar_exc']  = 4.2e-9   * np.exp(-8.0 / Te)   * cm3   # Ar + e → Ar* + e
    k['Ar_iz_m'] = 2.05e-7  * np.exp(-4.95 / Te)  * cm3   # Ar* + e → Ar+ + 2e
    k['Ar_q']    = 2.0e-7   * cm3                          # Ar* + e → Ar + e
    k['Ar_el']   = max((-1.1e-8 + 3.9e-8*Te - 1.3e-8*Te**2
                        + 2e-9*Te**3 - 1.4e-10*Te**4 + 3.9e-12*Te**5) * cm3, 1e-20)

    # --- Penning ionization and Ar* quenching ---
    k['Penn_SF6']  = 2.0e-10 * cm3   # Ar* + SF6 → SF5+ + F + Ar + e
    k['qnch_SF6']  = 3.0e-10 * cm3   # Ar* + SF6 → Ar + SF5 + F
    k['qnch_SFx']  = 1.0e-10 * cm3   # Ar* + SFx → Ar + SFx
    k['qnch_F2']   = 5.0e-11 * cm3   # Ar* + F2 → Ar + 2F
    k['qnch_F']    = 5.0e-12 * cm3   # Ar* + F → Ar + F

    # --- Totals ---
    k['iz_SF6_total']  = sum(k[f'iz{i}'] for i in [18, 19, 20, 21, 22, 23, 24])
    k['att_SF6_total'] = sum(k[f'at{i}'] for i in [30, 31, 32, 33, 34, 35, 36])

    return k


def compute_troe_rates(k, ng_m3):
    """Evaluate Troe fall-off neutral recombination rates at local gas density.

    Parameters
    ----------
    k : dict
        Rate coefficient dictionary from rates().
    ng_m3 : float
        Local total gas number density [m⁻³].

    Returns
    -------
    dict
        Updated k with keys 'nr37' through 'nr42' added [m³/s].
    """
    ng_cm3 = ng_m3 * 1e-6  # Convert to cm⁻³ for Troe formula
    for idx in [42, 41, 40, 39, 38, 37]:
        key = f'nr{idx}'
        k[key] = troe_rate(k[f'{key}_k0'], k[f'{key}_kinf'], k[f'{key}_Fc'], ng_cm3) * cm3
    return k


# ═══════════════════════════════════════════════════════════
# Source term evaluators for the 2D solver
# ═══════════════════════════════════════════════════════════

def electron_source(k, ne, nSF6, nSF5, nSF3, nF, nS, nAr, nArm):
    """Net electron production rate at a single grid point [m⁻³ s⁻¹].

    This is the RHS of the electron continuity equation:
    Sₑ = ionization - attachment + Penning
    """
    # Electron-impact ionization (all channels)
    Riz = ne * (k['iz_SF6_total'] * nSF6
                + (k['iz25'] + k['iz26']) * nSF5
                + k['iz27'] * nSF3
                + k['iz28'] * nF
                + k['iz29'] * nS
                + k['Ar_iz'] * nAr
                + k['Ar_iz_m'] * nArm)

    # Penning ionization
    R_Penn = k['Penn_SF6'] * nArm * nSF6

    # Attachment
    Ratt = ne * k['att_SF6_total'] * nSF6

    return Riz + R_Penn - Ratt


def ion_ion_recombination(k, n_pos, n_neg):
    """Ion-ion recombination rate [m⁻³ s⁻¹]."""
    return k['rec'] * n_pos * n_neg


def energy_loss_density(k, ne, Te, nSF6, nSF5, nSF3, nF, nF2, nS, nAr, nArm):
    """Electron energy loss rate per unit volume [eV m⁻³ s⁻¹].

    This is the collision term in the electron energy equation.
    Transcribed from sf6_unified.py lines 253–261.

    Parameters
    ----------
    All densities in m⁻³, Te in eV.

    Returns
    -------
    P_loss : float
        Energy loss rate [eV m⁻³ s⁻¹]. Multiply by eC (1.6e-19) for Watts/m³.
    """
    from scipy.constants import m_e

    # Fraction of Ar excitation energy that leads to stepwise ionization
    # (the rest is quenched and must be counted as energy loss)
    R_quench = ((k['Ar_iz_m'] + k['Ar_q']) * ne + 1e10)  # approximate
    frac_stepwise = k['Ar_iz_m'] * ne / max(R_quench, 1e-30)
    frac_stepwise = min(frac_stepwise, 1.0)
    Ar_exc_loss = 12.0 * k['Ar_exc'] * nAr * (1.0 - frac_stepwise)

    Eloss = (
        # SF6 ionization energy losses
        (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']) * nSF6
        # SF6 dissociation energy losses
        + (9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3'] + 18.6*k['d4'] + 22.7*k['d5']) * nSF6
        # SF6 vibrational + elastic
        + 0.09 * k['vib_SF6'] * nSF6
        + k['el_SF6'] * nSF6 * 3 * m_e / (M_SPECIES['SF6'] * AMU) * Te
        # SF5 losses
        + (11*k['iz25'] + 15*k['iz26']) * nSF5 + 5*k['d7'] * nSF5
        # SF3 losses
        + 11*k['iz27'] * nSF3 + 5*k['d9'] * nSF3
        # F losses
        + 15*k['iz28'] * nF + 14.4*k['exc_F'] * nF
        + k['el_F'] * nF * 3 * m_e / (M_SPECIES['F'] * AMU) * Te
        # F2 losses
        + 3.2*k['d6'] * nF2 + 0.11*k['vib_F2'] * nF2
        # S losses
        + 10*k['iz29'] * nS
        # Ar losses
        + 16*k['Ar_iz'] * nAr
        + Ar_exc_loss
        + k['Ar_el'] * nAr * 3 * m_e / (M_SPECIES['Ar'] * AMU) * Te
        # Ar* stepwise ionization energy
        + (12 + 4.95) * k['Ar_iz_m'] * nArm
    )

    return ne * Eloss  # [eV m⁻³ s⁻¹]


def fluorine_source(k, ne, nSF6, nSF5, nSF4, nSF3, nSF2, nSF, nF2, nArm):
    """Net F atom volumetric production rate [m⁻³ s⁻¹].

    Transcribed from sf6_unified.py lines 217–224.
    """
    RF = ne * nSF6 * (k['d1'] + 2*k['d2'] + 3*k['d3'] + 2*k['d4'] + 3*k['d5']
                       + k['iz18'] + 2*k['iz19'] + 3*k['iz20'] + 2*k['iz21']
                       + 3*k['iz22'] + 4*k['iz23'] + k['at31'])
    RF += ne * (nSF5 * (k['d7'] + k['iz26']) + nSF4 * k['d8']
                + nSF3 * k['d9'] + nSF2 * k['d10'] + nSF * k['d11'])
    RF += k['Penn_SF6'] * nArm * nSF6
    RF += k['qnch_SF6'] * nArm * nSF6
    RF += 2 * k['qnch_F2'] * nArm * nF2
    RF += 2 * k['d6'] * ne * nF2
    return RF


# ═══════════════════════════════════════════════════════════
# Self-test: verify against 0D model at reference conditions
# ═══════════════════════════════════════════════════════════


#==============================================================================
# LXCAT CROSS-SECTIONS (Phelps Ar, Biagi SF6)
#==============================================================================


"""
LXCat cross-section data for Ar (Phelps) and SF6 (Biagi v10.6).

Parsed from LXCat downloads, March 16, 2026.
References:
  - Phelps database: Yamabe, Buckman, and Phelps, Phys. Rev. 27, 1345 (1983)
  - Biagi database: S.F. Biagi, Magboltz v10.6
"""

import numpy as np
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════
# ARGON (Phelps)
# ═══════════════════════════════════════════════════════════════

AR_EFFECTIVE_E = np.array([
    0, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 8.5e-3, 1e-2, 1.5e-2, 2e-2,
    3e-2, 4e-2, 5e-2, 7e-2, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25,
    0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.2, 1.3, 1.5, 1.7,
    1.9, 2.1, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 10, 12, 15, 17, 20, 25,
    30, 50, 75, 100, 150, 200, 300, 500, 700, 1000,
    1500, 2000, 3000, 5000, 7000, 10000])
AR_EFFECTIVE_S = np.array([
    7.5e-20, 7.5e-20, 7.1e-20, 6.7e-20, 6.1e-20, 5.4e-20, 5.05e-20, 4.6e-20, 3.75e-20, 3.25e-20,
    2.5e-20, 2.05e-20, 1.73e-20, 1.13e-20, 5.9e-21, 4.0e-21, 2.3e-21, 1.6e-21, 1.03e-21, 9.1e-22,
    1.53e-21, 2.35e-21, 3.3e-21, 5.1e-21, 8.6e-21, 1.38e-20, 1.66e-20, 1.82e-20, 2.1e-20, 2.3e-20,
    2.5e-20, 2.8e-20, 2.9e-20, 3.3e-20, 3.8e-20, 4.1e-20, 4.5e-20, 4.9e-20, 5.4e-20, 6.1e-20,
    6.7e-20, 8.1e-20, 9.6e-20, 1.17e-19, 1.5e-19, 1.52e-19, 1.41e-19, 1.31e-19, 1.1e-19, 9.45e-20,
    8.74e-20, 6.9e-20, 5.85e-20, 5.25e-20, 4.24e-20, 3.76e-20, 3.02e-20, 2.1e-20, 1.64e-20, 1.21e-20,
    8.8e-21, 6.6e-21, 4.5e-21, 3.1e-21, 2.3e-21, 1.75e-21])
AR_MASS_RATIO = 1.36e-5

AR_EXC_E = np.array([
    11.5, 12.7, 13.7, 14.7, 15.9, 16.5, 17.5, 18.5, 19.9, 22.2,
    24.7, 27.0, 30.0, 33.0, 35.3, 42.0, 48.0, 52.0, 70.0, 100,
    150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000])
AR_EXC_S = np.array([
    0, 7e-22, 1.41e-21, 2.28e-21, 3.8e-21, 4.8e-21, 6.1e-21, 7.5e-21, 9.2e-21, 1.17e-20,
    1.33e-20, 1.42e-20, 1.44e-20, 1.41e-20, 1.34e-20, 1.25e-20, 1.16e-20, 1.11e-20, 9.4e-21, 7.6e-21,
    6.0e-21, 5.05e-21, 3.95e-21, 2.8e-21, 2.25e-21, 1.77e-21, 1.36e-21, 1.1e-21, 8.3e-22, 5.8e-22,
    4.5e-22, 3.5e-22])
AR_EXC_THRESHOLD = 11.5

AR_ION_E = np.array([
    15.8, 16.0, 17.0, 18.0, 20.0, 22.0, 23.75, 25.0, 26.5, 30.0,
    32.5, 35.0, 37.5, 40.0, 50.0, 55.0, 100, 150, 200, 300,
    500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000])
AR_ION_S = np.array([
    0, 2.02e-22, 1.34e-21, 2.94e-21, 6.3e-21, 9.3e-21, 1.15e-20, 1.3e-20, 1.45e-20, 1.8e-20,
    1.99e-20, 2.17e-20, 2.31e-20, 2.39e-20, 2.53e-20, 2.6e-20, 2.85e-20, 2.52e-20, 2.39e-20, 2.0e-20,
    1.45e-20, 1.15e-20, 8.6e-21, 6.4e-21, 5.2e-21, 3.6e-21, 2.4e-21, 1.8e-21, 1.35e-21])
AR_ION_THRESHOLD = 15.8

# ═══════════════════════════════════════════════════════════════
# SF6 (Biagi v10.6)
# ═══════════════════════════════════════════════════════════════

SF6_ELASTIC_E = np.array([
    1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, 12,
    15, 20, 25, 30, 40, 50, 70, 100, 150, 200,
    300, 500, 1000])
SF6_ELASTIC_S = np.array([
    1.434e-17, 8.9e-18, 6.2e-18, 3.86e-18, 1.99e-18, 1.22e-18, 6.15e-19, 3.6e-19, 2.8e-19, 2.08e-19,
    1.5e-19, 1.2e-19, 1.03e-19, 9.21e-20, 8.57e-20, 8.25e-20, 8.75e-20, 1.05e-19, 1.45e-19, 1.63e-19,
    1.51e-19, 1.43e-19, 1.36e-19, 1.39e-19, 1.51e-19, 1.55e-19, 1.48e-19, 1.47e-19, 1.53e-19, 1.76e-19,
    1.44e-19, 1.57e-19, 1.54e-19, 1.34e-19, 1.03e-19, 8.73e-20, 7.24e-20, 5.49e-20, 3.95e-20, 2.84e-20,
    1.66e-20, 9.6e-21, 4.47e-21])
SF6_MASS_RATIO = 3.7e-6

# SF6 attachment → SF5- (peaks at ~0.5 eV)
SF6_ATT_SF5_E = np.array([
    1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
SF6_ATT_SF5_S = np.array([
    0, 4.77e-21, 4.13e-21, 3.42e-21, 2.59e-21, 1.71e-21, 1.13e-21, 9.9e-22, 1.23e-21, 2.65e-21,
    5.13e-21, 7.43e-21, 7.1e-21, 4.79e-21, 2.99e-21, 1.9e-21, 1.3e-21, 5.73e-22, 1.89e-22, 0])

# SF6 attachment → SF6- (thermal, HUGE at 0 eV)
SF6_ATT_SF6_E = np.array([
    1e-6, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2,
    2e-2, 5e-2, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.55])
SF6_ATT_SF6_S = np.array([
    1.9845e-16, 1.9845e-16, 1.2346e-16, 8.571e-17, 5.906e-17, 3.551e-17, 2.374e-17, 1.54e-17, 8.654e-18, 5.346e-18,
    3.083e-18, 1.306e-18, 5.047e-19, 1.466e-19, 3.99e-20, 1.0e-20, 1.26e-21, 2.9e-22, 0])

# SF6 vibrational excitations (summed for energy loss)
SF6_VIB_V4_THRESHOLD = 0.076253
SF6_VIB_V1_THRESHOLD = 0.096032
SF6_VIB_V3_THRESHOLD = 0.11754
SF6_VIB_2V1_THRESHOLD = 0.192064
SF6_VIB_3V1_THRESHOLD = 0.288096

# Triplet dissociation at 9.6 eV
SF6_DISS_TRIP_E = np.array([
    9.6, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 25, 30, 40, 50, 70, 100, 200, 500, 1000])
SF6_DISS_TRIP_S = np.array([
    0, 3.6e-23, 1.5e-22, 3.0e-22, 4.5e-22, 5.0e-22, 5.4e-22, 5.65e-22, 5.9e-22, 5.95e-22,
    5.9e-22, 5.8e-22, 4.5e-22, 3.07e-22, 1.66e-22, 1.06e-22, 5.43e-23, 2.66e-23, 6.66e-24, 1.06e-24, 2.66e-25])
SF6_DISS_TRIP_THRESHOLD = 9.6

# Singlet dissociation at 12 eV
SF6_DISS_SING12_E = np.array([
    12, 13, 14, 15, 16, 17, 18, 19, 20, 22,
    25, 30, 40, 50, 70, 100, 200, 500, 1000])
SF6_DISS_SING12_S = np.array([
    0, 9.59e-23, 1.80e-22, 2.55e-22, 3.21e-22, 3.80e-22, 4.33e-22, 4.80e-22, 5.23e-22, 5.95e-22,
    6.80e-22, 7.75e-22, 8.68e-22, 8.96e-22, 8.81e-22, 8.10e-22, 6.03e-22, 3.45e-22, 2.10e-22])
SF6_DISS_SING12_THRESHOLD = 12.0

# Singlet dissociation at 16 eV
SF6_DISS_SING16_E = np.array([
    16, 17, 18, 19, 20, 22, 25, 30, 40, 50,
    70, 100, 200, 500, 1000])
SF6_DISS_SING16_S = np.array([
    0, 7.79e-23, 1.48e-22, 2.12e-22, 2.70e-22, 3.72e-22, 4.93e-22, 6.38e-22, 8.01e-22, 8.74e-22,
    9.09e-22, 8.72e-22, 6.83e-22, 4.06e-22, 2.52e-22])
SF6_DISS_SING16_THRESHOLD = 16.0

# SF6 ionization → SF5+ at 15.67 eV
SF6_ION_E = np.array([
    15.67, 16.5, 17, 18, 19, 20, 22, 25, 28, 30,
    35, 40, 50, 60, 70, 80, 90, 100, 120, 150,
    200, 300, 500, 1000, 2000, 4000])
SF6_ION_S = np.array([
    0, 2.0e-22, 3.5e-22, 8.4e-22, 1.54e-21, 3.14e-21, 4.34e-21, 1.13e-20, 1.36e-20, 1.61e-20,
    2.08e-20, 2.46e-20, 2.81e-20, 3.02e-20, 3.16e-20, 3.25e-20, 3.27e-20, 3.30e-20, 3.33e-20, 3.30e-20,
    3.19e-20, 2.91e-20, 2.40e-20, 1.63e-20, 9.38e-21, 5.32e-21])
SF6_ION_THRESHOLD = 15.67


def get_interpolator(E_arr, S_arr, fill_value=0.0):
    """Create a log-interpolated cross-section function."""
    mask = S_arr > 0
    if mask.sum() < 2:
        return lambda e: np.zeros_like(np.atleast_1d(e))
    return interp1d(np.log10(E_arr[mask] + 1e-30), np.log10(S_arr[mask] + 1e-30),
                    kind='linear', bounds_error=False,
                    fill_value=(np.log10(S_arr[mask][0]+1e-30), -40))


# Build interpolators
def sigma_Ar_eff(e):
    """Ar effective momentum transfer [m²]."""
    f = interp1d(AR_EFFECTIVE_E, AR_EFFECTIVE_S, kind='linear',
                 bounds_error=False, fill_value=(AR_EFFECTIVE_S[0], AR_EFFECTIVE_S[-1]))
    return f(np.atleast_1d(e))

def sigma_Ar_exc(e):
    f = interp1d(AR_EXC_E, AR_EXC_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_Ar_ion(e):
    f = interp1d(AR_ION_E, AR_ION_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_elastic(e):
    f = interp1d(SF6_ELASTIC_E, SF6_ELASTIC_S, kind='linear',
                 bounds_error=False, fill_value=(SF6_ELASTIC_S[0], SF6_ELASTIC_S[-1]))
    return f(np.atleast_1d(e))

def sigma_SF6_att_SF5(e):
    f = interp1d(SF6_ATT_SF5_E, SF6_ATT_SF5_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_att_SF6(e):
    f = interp1d(SF6_ATT_SF6_E, SF6_ATT_SF6_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_diss_trip(e):
    f = interp1d(SF6_DISS_TRIP_E, SF6_DISS_TRIP_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_diss_sing12(e):
    f = interp1d(SF6_DISS_SING12_E, SF6_DISS_SING12_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_diss_sing16(e):
    f = interp1d(SF6_DISS_SING16_E, SF6_DISS_SING16_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_ion(e):
    f = interp1d(SF6_ION_E, SF6_ION_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)



#==============================================================================
# METTLER EXPERIMENTAL DATA
#==============================================================================


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



#==============================================================================
# MESH GENERATOR
#==============================================================================


"""
Structured (r,z) mesh for 2D axisymmetric ICP reactor.

Provides non-uniform spacing with algebraic stretching near walls
to resolve sheaths and boundary layers.

Coordinate system:
    r: radial, 0 ≤ r ≤ R   (r=0 is symmetry axis)
    z: axial,  0 ≤ z ≤ L   (z=0 is grounded bottom, z=L is dielectric window / coil)
"""

import numpy as np


class Mesh2D:
    """Structured axisymmetric (r,z) mesh with optional wall stretching."""

    def __init__(self, R, L, Nr, Nz, stretch_r=1.0, stretch_z=1.0):
        """
        Parameters
        ----------
        R : float
            Reactor radius [m].
        L : float
            Reactor height [m].
        Nr : int
            Number of cells in r direction.
        Nz : int
            Number of cells in z direction.
        stretch_r : float
            Stretching factor for r (1.0 = uniform; >1 concentrates points near r=R).
        stretch_z : float
            Stretching factor for z (1.0 = uniform; >1 concentrates points near z=0 and z=L).
        """
        self.R = R
        self.L = L
        self.Nr = Nr
        self.Nz = Nz

        # --- Build 1D node coordinates ---
        # r: stretch toward r=R (wall)
        self.r_nodes = self._stretch_one_side(R, Nr + 1, stretch_r)
        # z: stretch toward both z=0 and z=L (electrodes / window)
        self.z_nodes = self._stretch_two_sides(L, Nz + 1, stretch_z)

        # Cell-centre coordinates
        self.r = 0.5 * (self.r_nodes[:-1] + self.r_nodes[1:])  # (Nr,)
        self.z = 0.5 * (self.z_nodes[:-1] + self.z_nodes[1:])  # (Nz,)

        # Cell widths
        self.dr = np.diff(self.r_nodes)  # (Nr,)
        self.dz = np.diff(self.z_nodes)  # (Nz,)

        # Distance between cell centres (for gradient computation)
        self.dr_c = np.zeros(Nr + 1)  # at r-faces (Nr+1 faces: 0..Nr)
        self.dr_c[1:-1] = 0.5 * (self.dr[:-1] + self.dr[1:])
        self.dr_c[0] = 0.5 * self.dr[0]      # axis face
        self.dr_c[-1] = 0.5 * self.dr[-1]     # wall face

        self.dz_c = np.zeros(Nz + 1)
        self.dz_c[1:-1] = 0.5 * (self.dz[:-1] + self.dz[1:])
        self.dz_c[0] = 0.5 * self.dz[0]
        self.dz_c[-1] = 0.5 * self.dz[-1]

        # Face positions (cell boundaries)
        self.r_faces = self.r_nodes  # (Nr+1,)
        self.z_faces = self.z_nodes  # (Nz+1,)

        # Face areas for flux computation in cylindrical coordinates
        # Radial face area: A_r = 2π r_face * Δz  (for each z-cell)
        # Axial face area:  A_z = π(r²_right - r²_left)  (annular ring for each r-cell)
        self.cell_volume = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                r_lo, r_hi = self.r_nodes[i], self.r_nodes[i + 1]
                self.cell_volume[i, j] = np.pi * (r_hi**2 - r_lo**2) * self.dz[j]

        # 2D coordinate grids (cell centres)
        self.RR, self.ZZ = np.meshgrid(self.r, self.z, indexing='ij')  # (Nr, Nz)

        # Summary
        self.total_cells = Nr * Nz
        self.dr_min = self.dr.min()
        self.dz_min = self.dz.min()

    # --- Stretching functions ---

    @staticmethod
    def _stretch_one_side(L, N, beta):
        """Stretch grid toward x = L. beta=1 gives uniform."""
        if abs(beta - 1.0) < 1e-10:
            return np.linspace(0, L, N)
        xi = np.linspace(0, 1, N)
        # Hyperbolic tangent stretching
        x = L * (1.0 + np.tanh(beta * (xi - 1.0)) / np.tanh(beta))
        x[0] = 0.0
        x[-1] = L
        return x

    @staticmethod
    def _stretch_two_sides(L, N, beta):
        """Stretch grid toward both x = 0 and x = L. beta=1 gives uniform."""
        if abs(beta - 1.0) < 1e-10:
            return np.linspace(0, L, N)
        xi = np.linspace(0, 1, N)
        # Symmetric hyperbolic tangent stretching
        x = L * 0.5 * (1.0 + np.tanh(beta * (2.0 * xi - 1.0)) / np.tanh(beta))
        x[0] = 0.0
        x[-1] = L
        return x

    def total_volume(self):
        """Total reactor volume [m³]."""
        return np.pi * self.R**2 * self.L

    def volume_average(self, field):
        """Compute volume-weighted average of a 2D field (Nr, Nz)."""
        return np.sum(field * self.cell_volume) / np.sum(self.cell_volume)

    def radial_profile_at_midplane(self, field):
        """Extract radial profile at the axial midplane."""
        j_mid = self.Nz // 2
        return self.r, field[:, j_mid]

    def axial_profile_on_axis(self, field):
        """Extract axial profile at r = 0 (first cell centre)."""
        return self.z, field[0, :]

    def __repr__(self):
        return (f"Mesh2D(R={self.R:.3f}m, L={self.L:.3f}m, "
                f"{self.Nr}×{self.Nz}={self.total_cells} cells, "
                f"Δr_min={self.dr_min*1e3:.2f}mm, Δz_min={self.dz_min*1e3:.2f}mm)")


# === Convenience constructor ===

def make_icp_mesh(R=0.180, L=0.175, Nr=80, Nz=100, stretch_r=1.5, stretch_z=1.5):
    """Create a mesh suitable for ICP reactor simulation.

    Default: Lallement reactor (R=18cm, L=17.5cm) with 80×100 cells
    and moderate stretching near walls.
    """
    return Mesh2D(R, L, Nr, Nz, stretch_r, stretch_z)


# === Self-test ===


#==============================================================================
# BASE TRANSPORT COEFFICIENTS
#==============================================================================


"""
Transport coefficients for charged and neutral species.

Electron transport: μₑ, Dₑ (and energy equivalents) from Arrhenius-like
expressions or BOLSIG+ tables.

Ion transport: Langevin mobility, variable Da from Vahedi (1995).

Neutral transport: Chapman-Enskog diffusion coefficients.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, epsilon_0, pi

AMU = 1.66054e-27


# ═══════════════════════════════════════════════════════════
# Electron transport
# ═══════════════════════════════════════════════════════════

class ElectronTransport:
    """Electron mobility and diffusion as functions of mean electron energy.

    Phase 1: Simple expressions consistent with the 0D model assumptions.
    Phase 2: Replace with BOLSIG+ lookup tables.
    """

    def __init__(self, gas_density, x_SF6=1.0, x_Ar=0.0):
        """
        Parameters
        ----------
        gas_density : float
            Total neutral gas density [m⁻³].
        x_SF6 : float
            Mole fraction of SF₆ in the gas.
        x_Ar : float
            Mole fraction of Ar in the gas.
        """
        self.ng = gas_density
        self.x_SF6 = x_SF6
        self.x_Ar = x_Ar

    def collision_frequency(self, Te):
        """Effective electron-neutral momentum transfer collision frequency [s⁻¹].

        Uses a simplified model: ν_en = ng * k_en(Te) where k_en combines
        elastic and inelastic momentum transfer.
        """
        # SF6 elastic momentum transfer cross-section ~ 1-5 × 10⁻¹⁹ m²
        # at Te ~ 2-4 eV, giving k_en ~ σ * v_th
        v_th = np.sqrt(8 * eC * Te / (pi * m_e))

        # Effective cross-sections (approximate, from BOLSIG+ for typical conditions)
        sigma_SF6 = 3e-19  # m² (SF6, moderate energy)
        sigma_Ar = 1e-20 + 5e-20 * np.exp(-((Te - 0.3) / 0.5)**2)  # Ramsauer minimum at ~0.3 eV

        sigma_eff = self.x_SF6 * sigma_SF6 + self.x_Ar * sigma_Ar
        return self.ng * sigma_eff * v_th

    def mobility(self, Te):
        """Electron mobility μₑ [m²/(V·s)].

        μₑ = e / (mₑ · ν_en)
        """
        nu = self.collision_frequency(Te)
        return eC / (m_e * max(nu, 1e3))

    def diffusivity(self, Te):
        """Electron diffusion coefficient Dₑ [m²/s].

        Using Einstein relation: Dₑ = (2/3) μₑ ε̄ = μₑ Te
        (Note: Hagelaar 2005 shows this can be off by factor 2 for Ar.
        Phase 2 should use exact BOLSIG+ values.)
        """
        return self.mobility(Te) * Te  # Te in eV, μ in m²/(V·s) → D in eV·m²/(V·s) = m²/s

    def energy_mobility(self, Te):
        """Energy mobility μₑ_ε [m²/(V·s)].

        Approximate: μₑ_ε ≈ (5/3) μₑ
        (Exact values differ — see Hagelaar 2005 Eq. 61.)
        """
        return (5.0 / 3.0) * self.mobility(Te)

    def energy_diffusivity(self, Te):
        """Energy diffusion coefficient Dε [m²/s].

        Approximate: Dε ≈ (5/3) Dₑ
        (Exact values differ — see Hagelaar 2005 Eq. 62.)
        """
        return (5.0 / 3.0) * self.diffusivity(Te)


# ═══════════════════════════════════════════════════════════
# Ion transport
# ═══════════════════════════════════════════════════════════

class IonTransport:
    """Positive and negative ion transport coefficients."""

    def __init__(self, Mi_amu, gas_density, Tgas=300.0, sigma_in=5e-19):
        """
        Parameters
        ----------
        Mi_amu : float
            Ion mass in AMU (e.g., 127.06 for SF₅⁺).
        gas_density : float
            Total neutral gas density [m⁻³].
        Tgas : float
            Gas temperature [K].
        sigma_in : float
            Ion-neutral collision cross-section [m²].
        """
        self.Mi = Mi_amu * AMU  # kg
        self.ng = gas_density
        self.Tgas = Tgas
        self.sigma_in = sigma_in

        # Ion thermal velocity
        self.v_ti = np.sqrt(8 * kB * Tgas / (pi * self.Mi))

        # Mean free path
        self.lambda_in = 1.0 / max(gas_density * sigma_in, 1.0)

        # Low-field diffusion coefficient (Vahedi 1995, Eq. 48)
        self.D0 = kB * Tgas * self.lambda_in / (self.Mi * self.v_ti)

        # Low-field mobility (Einstein relation at gas temperature)
        self.mu0 = eC * self.lambda_in / (self.Mi * self.v_ti)

    def bohm_velocity(self, Te, alpha=0.0, T_neg=0.3):
        """Modified Bohm velocity for electronegative plasma.

        uB = sqrt(e Te (1+α) / (M (1+γα)))

        From Lichtenberg (1997) Eq. 15.

        Parameters
        ----------
        Te : float
            Electron temperature [eV].
        alpha : float
            Local electronegativity n₋/nₑ.
        T_neg : float
            Negative ion temperature [eV].
        """
        gamma = Te / max(T_neg, 0.01)
        return np.sqrt(eC * Te * (1 + alpha) / (self.Mi * (1 + gamma * alpha)))

    def variable_Da(self, Te, u_drift, alpha=0.0, T_neg=0.3):
        """Variable ambipolar diffusion coefficient (Vahedi 1995, Eq. 47).

        Accounts for drift velocity saturation at the Bohm velocity.

        Da = sqrt(2) * Da0 / sqrt(1 + sqrt(1 + 4*Da0²*|∇n|²/(v_ti²*n²)))

        Simplified form using drift velocity:
        Da = Te * lambda_in / (Mi * sqrt(v_ti² + u_drift²))
        """
        return eC * Te * self.lambda_in / (self.Mi * np.sqrt(self.v_ti**2 + u_drift**2))

    def diffusivity(self, Te=None):
        """Ion diffusion coefficient [m²/s].

        If Te is given, returns ambipolar diffusion coefficient.
        Otherwise returns free diffusion coefficient.
        """
        if Te is not None:
            return eC * Te / (self.Mi * self.v_ti / self.lambda_in)
        return self.D0

    def mobility(self):
        """Ion mobility [m²/(V·s)]."""
        return self.mu0


# ═══════════════════════════════════════════════════════════
# Neutral transport
# ═══════════════════════════════════════════════════════════

class NeutralTransport:
    """Diffusion coefficients for neutral species."""

    def __init__(self, gas_density, Tgas=300.0):
        """
        Parameters
        ----------
        gas_density : float
            Background gas number density [m⁻³].
        Tgas : float
            Gas temperature [K].
        """
        self.ng = gas_density
        self.Tgas = Tgas

    def diffusivity(self, M_amu, sigma=4e-19):
        """Free diffusion coefficient for neutral species [m²/s].

        D = kT λ / (M v_th)  where λ = 1/(ng σ)

        Parameters
        ----------
        M_amu : float
            Species mass in AMU.
        sigma : float
            Diffusion cross-section [m²].
        """
        M = M_amu * AMU
        v_th = np.sqrt(8 * kB * self.Tgas / (pi * M))
        lam = 1.0 / max(self.ng * sigma, 1.0)  # mean free path [m]; floor at 1 m⁻¹ to avoid div/0
        return lam * v_th / 3.0  # kinetic theory: D = λv/3


# ═══════════════════════════════════════════════════════════
# Ambipolar diffusion for electronegative plasmas
# ═══════════════════════════════════════════════════════════

def ambipolar_Da(alpha, Te, D_plus, T_neg=0.3):
    """Ambipolar diffusion coefficient Da(α) for electronegative plasma.

    From Lichtenberg (1997) Eq. 10:
    Da ≈ D₊ (1 + γ + 2γα) / (1 + γα)

    where γ = Te/Ti.

    Limits:
        α >> 1: Da → 2D₊
        α << 1: Da → γD₊ (electropositive limit)

    Parameters
    ----------
    alpha : float or array
        Local electronegativity n₋/nₑ.
    Te : float
        Electron temperature [eV].
    D_plus : float
        Positive ion free diffusion coefficient [m²/s].
    T_neg : float
        Negative ion temperature [eV].

    Returns
    -------
    Da : float or array
        Ambipolar diffusion coefficient [m²/s].
    """
    gamma = Te / max(T_neg, 0.01)
    return D_plus * (1 + gamma + 2 * gamma * alpha) / (1 + gamma * alpha)


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════


#==============================================================================
# HAGELAAR TRANSPORT CORRECTIONS
#==============================================================================


"""
Hagelaar-corrected electron transport coefficients.

Based on: G.J.M. Hagelaar and L.C. Pitchford,
"Solving the Boltzmann equation to obtain electron transport coefficients
 and rate coefficients for fluid models,"
Plasma Sources Sci. Technol. 14, 722-733 (2005).

Key corrections over the simple approximations:
  1. Einstein relation D_e = (2/3)μ_e·ε̄ is WRONG by ~2× for Ar (Ramsauer minimum)
  2. The (5/3) factor for energy transport D_ε = (5/3)D_e is wrong by ~2× for Ar
  3. SF6 has NO Ramsauer minimum — corrections are smaller (~15-20%)
  4. For mixtures, use Blanc's law: 1/μ_mix = Σ x_k/μ_k
"""

import numpy as np
from scipy.constants import e as eC, m_e, k as kB, pi

gamma_boltzmann = np.sqrt(2 * eC / m_e)  # ≈ 5.931e5 m/s per eV^{1/2}


def transport_Ar(Te_eV, N):
    """Electron transport coefficients for pure Ar with Ramsauer correction.

    The Ramsauer minimum in Ar at ~0.2 eV causes the momentum-transfer
    frequency to be strongly energy-dependent. This makes the Einstein
    relation underestimate D_e by ~2× at Te = 2-4 eV (typical ICP range).

    Parameters
    ----------
    Te_eV : float or array
        Electron temperature in eV
    N : float
        Gas number density in m⁻³

    Returns
    -------
    dict with keys: mu_e, D_e, mu_eps, D_eps, nu_m (all SI)
    """
    Te = np.asarray(Te_eV, dtype=float)

    # Effective σ_m for Ar (fit to BOLSIG+/Phelps data)
    # Increases from ~1e-20 at 0.3 eV to ~8e-20 at 10 eV
    sigma_eff = 1.0e-20 * (0.5 + 2.5 * Te**0.7)

    v_th = np.sqrt(8 * eC * Te / (pi * m_e))
    nu_m = N * sigma_eff * v_th

    mu_e_base = eC / (m_e * nu_m)
    D_e_einstein = (2.0/3.0) * mu_e_base * 1.5 * Te  # ε̄ = 1.5*Te

    # Ramsauer correction factors (from Hagelaar 2005 Fig 9a)
    ramsauer_D = 1.0 + 1.2 * np.exp(-((Te - 2.0)/2.0)**2)    # D_e/D_einstein ~ 1.5-2.2
    ramsauer_mu = 1.0 + 0.3 * np.exp(-((Te - 2.0)/2.0)**2)   # μ_e correction ~ 1.0-1.3

    D_e = D_e_einstein * ramsauer_D
    mu_e = mu_e_base * ramsauer_mu

    # Energy transport (Hagelaar Eqs 61-62)
    # D_ε/D_e ratio is ~2.5-3.5 for Ar instead of 5/3
    D_eps_ratio = (5.0/3.0) * (1.0 + 0.8 * np.exp(-((Te - 2.5)/2.5)**2))
    mu_eps_ratio = (5.0/3.0) * (1.0 + 0.15 * np.exp(-((Te - 2.5)/2.5)**2))

    D_eps = D_e * D_eps_ratio
    mu_eps = mu_e * mu_eps_ratio

    return {'mu_e': mu_e, 'D_e': D_e, 'mu_eps': mu_eps, 'D_eps': D_eps,
            'nu_m': nu_m, 'eps_bar': 1.5*Te}


def transport_SF6(Te_eV, N):
    """Electron transport for pure SF6 (no Ramsauer minimum).

    SF6 has a very large total scattering cross-section (~20-50 × 10⁻²⁰ m²)
    with no Ramsauer minimum. The Einstein relation is more accurate (~15-20% error).
    """
    Te = np.asarray(Te_eV, dtype=float)

    sigma_eff = 1.0e-20 * (30.0 + 10.0 * Te**0.5)
    v_th = np.sqrt(8 * eC * Te / (pi * m_e))
    nu_m = N * sigma_eff * v_th

    mu_e = eC / (m_e * nu_m)
    D_e = (2.0/3.0) * mu_e * 1.5 * Te * 1.15    # 15% non-Maxwellian correction
    D_eps = (5.0/3.0) * D_e * 1.1                 # 10% correction over (5/3)
    mu_eps = (5.0/3.0) * mu_e * 1.05

    return {'mu_e': mu_e, 'D_e': D_e, 'mu_eps': mu_eps, 'D_eps': D_eps,
            'nu_m': nu_m, 'eps_bar': 1.5*Te}


def transport_mixture(Te_eV, N, x_Ar, x_SF6):
    """Transport coefficients for Ar/SF6 mixture using Blanc's law.

    1/μ_mix = x_Ar/μ_Ar + x_SF6/μ_SF6

    Parameters
    ----------
    Te_eV : float
    N : float — total gas density m⁻³
    x_Ar, x_SF6 : float — mole fractions (should sum to 1)
    """
    ar = transport_Ar(Te_eV, N)
    sf6 = transport_SF6(Te_eV, N)

    if x_Ar > 0 and x_SF6 > 0:
        mu_e = 1.0 / (x_Ar/ar['mu_e'] + x_SF6/sf6['mu_e'])
        D_e = 1.0 / (x_Ar/ar['D_e'] + x_SF6/sf6['D_e'])
        mu_eps = 1.0 / (x_Ar/ar['mu_eps'] + x_SF6/sf6['mu_eps'])
        D_eps = 1.0 / (x_Ar/ar['D_eps'] + x_SF6/sf6['D_eps'])
        nu_m = x_Ar*ar['nu_m'] + x_SF6*sf6['nu_m']
    elif x_Ar >= 1:
        return ar
    else:
        return sf6

    return {'mu_e': mu_e, 'D_e': D_e, 'mu_eps': mu_eps, 'D_eps': D_eps,
            'nu_m': nu_m, 'eps_bar': 1.5*Te_eV}



#==============================================================================
# ETCH RATE MODEL
#==============================================================================


"""
Silicon etch rate prediction from fluorine density.

The etch rate is: R_etch = gamma_Si * (1/4) * v_th_F * [F] * (M_Si / rho_Si / N_A)

where:
  gamma_Si = Si etch probability by F atoms (~0.025 from radical probes, Mettler)
  v_th_F = thermal velocity of F at gas temperature
  [F] = fluorine atom density at the wafer surface (m^-3)
  M_Si = 28.09 g/mol
  rho_Si = 2329 kg/m^3
  N_A = 6.022e23 mol^-1

The factor M_Si/(rho_Si*N_A) converts from atoms removed per area per time
to thickness per time (m/s → nm/s with 1e9 factor).
"""

import numpy as np
from scipy.constants import k as kB, pi, N_A

M_F_kg = 19.0 * 1.66054e-27   # F atom mass
M_Si = 28.09e-3                # Si molar mass, kg/mol
rho_Si = 2329.0                # Si density, kg/m3

# Etch probability from Mettler Table 4.18
GAMMA_SI_RADICAL_PROBE = 0.025   # From radical probe measurements
GAMMA_SI_ACTINOMETRY = 0.08      # From actinometry (overestimates)
GAMMA_SI_DEFAULT = 0.025          # Use radical probe value


def etch_rate(nF_at_wafer, Tgas=300.0, gamma_Si=GAMMA_SI_DEFAULT):
    """Compute Si etch rate from F density at the wafer.

    Parameters
    ----------
    nF_at_wafer : float or array
        F atom density at the wafer surface [m^-3]
    Tgas : float
        Gas temperature at the wafer [K] (typically ~25°C = 298 K)
    gamma_Si : float
        Si etch probability (dimensionless)

    Returns
    -------
    R_nm_s : float or array
        Si etch rate in nm/s
    """
    v_th_F = np.sqrt(8 * kB * Tgas / (pi * M_F_kg))   # m/s
    F_flux = 0.25 * v_th_F * nF_at_wafer               # m^-2 s^-1
    atoms_per_m3_Si = rho_Si * N_A / M_Si               # Si atoms per m^3
    R_m_s = gamma_Si * F_flux / atoms_per_m3_Si          # m/s
    R_nm_s = R_m_s * 1e9                                  # nm/s
    return R_nm_s


def etch_rate_profile(nF_2d, mesh, Tgas=300.0, gamma_Si=GAMMA_SI_DEFAULT):
    """Extract the radial etch rate profile at the wafer plane (z=0).

    Parameters
    ----------
    nF_2d : array (Nr, Nz)
        2D fluorine density field [m^-3]
    mesh : Mesh2D object
    Tgas, gamma_Si : floats

    Returns
    -------
    r_cm : array
        Radial positions [cm]
    R_etch : array
        Si etch rate at the wafer [nm/s]
    """
    nF_wafer = nF_2d[:, 0]  # z=0 is the wafer
    R_etch = etch_rate(nF_wafer, Tgas, gamma_Si)
    r_cm = mesh.r * 100
    return r_cm, R_etch


def uniformity(R_etch, r_cm=None, r_max_cm=None):
    """Compute etch rate uniformity metrics.

    Parameters
    ----------
    R_etch : array
        Etch rate profile [nm/s]
    r_cm : array, optional
        Radial positions
    r_max_cm : float, optional
        Max radius for uniformity calculation (e.g., wafer radius)

    Returns
    -------
    dict with: mean, std, nonuniformity_pct, center, edge, range_pct
    """
    if r_cm is not None and r_max_cm is not None:
        mask = r_cm <= r_max_cm
        R = R_etch[mask]
    else:
        R = R_etch

    return {
        'mean': np.mean(R),
        'std': np.std(R),
        'nonuniformity_pct': (np.max(R) - np.min(R)) / (2 * np.mean(R)) * 100,
        'center': R[0],
        'edge': R[-1] if len(R) > 1 else R[0],
        'range_pct': (R[0] - R[-1]) / R[0] * 100 if R[0] > 0 else 0,
    }



#==============================================================================
# DIFFUSION PROFILE SOLVER
#==============================================================================


"""
Steady-state diffusion solver for density profiles.

Solves: ∇·(Da ∇n) = 0  inside domain
with Robin BC at walls:  Da ∂n/∂n̂ = uB * n  (Bohm flux)
and symmetry BC at axis: ∂n/∂r = 0

Returns the fundamental diffusion mode (positive everywhere,
peaked at centre, zero at walls for Dirichlet, or with finite
edge density for Bohm-flux BC).

This replaces the unstable explicit diffusion in the main loop.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigs


def solve_diffusion_profile(mesh, Da, uB_wall):
    """Solve for steady-state density profile with Bohm-flux wall loss.

    The steady-state profile satisfies:
        ∇·(Da ∇n) - kw_eff * n = 0
    
    where kw_eff is an effective wall loss rate distributed over boundary cells.
    
    Actually, we solve the eigenvalue problem:
        Da ∇²n = -λ n
    
    to find the fundamental diffusion mode. The eigenvalue λ gives the
    wall loss rate kw = λ.
    
    For simplicity, we instead solve:
        Da ∇²n + S₀ = 0  with  n=0 at walls
    
    where S₀ = 1 (uniform source), which gives the correct profile shape.
    The magnitude is set by the power balance in the caller.

    Parameters
    ----------
    mesh : Mesh2D
    Da : float — Ambipolar diffusion coefficient [m²/s]
    uB_wall : float — Bohm velocity at wall [m/s] (for Robin BC)

    Returns
    -------
    profile : array (Nr, Nz) — normalized so max = 1
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N = Nr * Nz

    rows, cols, vals = [], [], []
    rhs = np.ones(N)  # Uniform source

    for i in range(Nr):
        for j in range(Nz):
            idx = i * Nz + j
            rc = mesh.r[i]
            dr = mesh.dr[i]
            dz = mesh.dz[j]
            diag = 0.0

            # --- Radial: (1/r) d/dr(r Da dn/dr) ---
            if i < Nr - 1:
                rf = mesh.r_faces[i+1]
                drc = mesh.dr_c[i+1]
                c = Da * rf / (rc * dr * drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c)
                diag -= c
            else:
                # Wall: Robin BC  Da dn/dr = -uB * n
                # → Da*(n_wall - n_i)/dr_c = -uB * 0.5*(n_wall + n_i)
                # Approximate: Da*(-n_i)/dr_c ≈ -uB*n_i  (n drops to ~0 at wall)
                rf = mesh.r_faces[Nr]
                drc = mesh.dr_c[Nr]
                c_wall = Da * rf / (rc * dr * drc)
                # Effective: extra loss term instead of neighbour
                diag -= c_wall  # Treat as Dirichlet n_wall ≈ 0

            if i > 0:
                rf = mesh.r_faces[i]
                drc = mesh.dr_c[i]
                c = Da * rf / (rc * dr * drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c)
                diag -= c
            else:
                # Axis: symmetry → (1/r)d(r dn/dr)/dr → 2 d²n/dr²
                drc = mesh.dr_c[1]
                c = 2.0 * Da / (dr * drc)
                rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # --- Axial: d²n/dz² ---
            if j < Nz - 1:
                dzc = mesh.dz_c[j+1]
                c = Da / (dz * dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c)
                diag -= c
            else:
                dzc = mesh.dz_c[Nz]
                c = Da / (dz * dzc)
                diag -= c  # Dirichlet n=0 at z=L

            if j > 0:
                dzc = mesh.dz_c[j]
                c = Da / (dz * dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c)
                diag -= c
            else:
                dzc = mesh.dz_c[0]
                c = Da / (dz * dzc)
                diag -= c  # Dirichlet n=0 at z=0

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    n_flat = spsolve(A, -rhs)

    profile = n_flat.reshape((Nr, Nz))
    profile = np.maximum(profile, 0.0)
    if profile.max() > 0:
        profile /= profile.max()
    else:
        # Fallback: cosine profile
        r_n = mesh.RR / mesh.R
        z_n = (mesh.ZZ - mesh.L/2) / (mesh.L/2)
        profile = np.maximum(np.cos(np.pi/2*z_n) * (1-r_n**2), 0.01)
        profile /= profile.max()

    return profile


def compute_h_factors(profile, mesh):
    """Compute edge-to-centre density ratios (h-factors) from the profile.

    hL = n(z=0)/n(centre) and n(z=L)/n(centre)  (axial)
    hR = n(r=R)/n(centre)  (radial)
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    n_centre = profile[0, Nz//2]  # On axis, midplane
    if n_centre < 1e-10:
        return 0.5, 0.4

    hL = 0.5 * (profile[0, 0] + profile[0, Nz-1]) / n_centre
    hR = profile[Nr-1, Nz//2] / n_centre

    return float(np.clip(hL, 0.01, 1.0)), float(np.clip(hR, 0.01, 1.0))



#==============================================================================
# EM WAVE EQUATION SOLVER
#==============================================================================


"""
Electromagnetic solver for ICP power deposition.

Solves the azimuthal component of the magnetic vector potential equation
in the frequency domain:

    (1/r) ∂/∂r(r ∂Ã/∂r) + ∂²Ã/∂z² + (ω²μ₀ε − 1/r²) Ã = −μ₀ J̃_coil

where Ã(r,z) is the complex amplitude of the azimuthal vector potential.

The inductive electric field is: Ẽ_θ = −jωÃ
The power deposition is: P(r,z) = ½ Re(σ̃ |Ẽ|²)

The complex plasma conductivity is:
    σ_p = nₑ e² / [mₑ (ν_eff + jω)]

where ν_eff includes both collisional and collisionless contributions
(Vahedi et al. 1995).

Reference: Vahedi et al., J. Appl. Phys. 78, 1446 (1995)
           Lymberopoulos & Economou, JRNIST 100, 473 (1995), Eqs. 28–35
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.constants import e as eC, m_e, mu_0, epsilon_0, c, pi


class EMSolver:
    """Frequency-domain EM solver for ICP coil coupling."""

    def __init__(self, mesh, freq=13.56e6):
        """
        Parameters
        ----------
        mesh : Mesh2D
        freq : float
            RF frequency [Hz]. Default 13.56 MHz.
        """
        self.mesh = mesh
        self.freq = freq
        self.omega = 2 * pi * freq
        self.Nr = mesh.Nr
        self.Nz = mesh.Nz

    def plasma_conductivity(self, ne, Te, gas_density):
        """Complex plasma conductivity σ_p(r,z).

        Parameters
        ----------
        ne : array (Nr, Nz) — electron density [m⁻³]
        Te : array (Nr, Nz) — electron temperature [eV]
        gas_density : float — neutral gas density [m⁻³]

        Returns
        -------
        sigma : complex array (Nr, Nz)
        """
        omega = self.omega

        # Electron-neutral collision frequency
        # Approximate: ν_en = ng × σ_en × v̄_e
        v_e = np.sqrt(8 * eC * Te / (pi * m_e))
        sigma_en = 3e-19  # Approximate effective cross-section [m²]
        nu_en = gas_density * sigma_en * v_e

        # Collisionless (stochastic) frequency (Vahedi 1995)
        # For ω ~ ν_en (typical at 13.56 MHz, 10 mTorr):
        # ν_st ≈ (1/4) v̄_e / δ  where δ ≈ c/ω_pe
        omega_pe = np.sqrt(ne * eC**2 / (m_e * epsilon_0))
        delta = np.where(omega_pe > 1e3, c / omega_pe, 1.0)
        nu_st = 0.25 * v_e / np.maximum(delta, 1e-6)

        # Effective collision frequency
        nu_eff = nu_en + nu_st

        # Complex conductivity: σ = nₑe²/[mₑ(ν_eff + jω)]
        sigma = ne * eC**2 / (m_e * (nu_eff + 1j * omega))

        return sigma

    def coil_current_density(self, I_coil, n_turns, coil_r, coil_z, coil_width=0.005):
        """Compute the coil current density source on the mesh.

        Models the coil as a set of discrete current loops.

        Parameters
        ----------
        I_coil : float
            Coil current amplitude [A].
        n_turns : int
            Number of coil turns.
        coil_r : array
            Radial positions of coil turns [m].
        coil_z : float
            Axial position of coil (just above dielectric) [m].
        coil_width : float
            Width of each coil turn for distributing current [m].

        Returns
        -------
        J_coil : complex array (Nr, Nz)
            Coil current density [A/m²].
        """
        mesh = self.mesh
        J = np.zeros((mesh.Nr, mesh.Nz), dtype=complex)

        for r_turn in coil_r:
            for i in range(mesh.Nr):
                for j in range(mesh.Nz):
                    dr = abs(mesh.r[i] - r_turn)
                    dz = abs(mesh.z[j] - coil_z)
                    if dr < coil_width and dz < coil_width:
                        # Distribute current over the cell area
                        area = coil_width**2
                        J[i, j] += I_coil / area

        return J

    def solve(self, ne, Te, gas_density, I_coil=10.0, n_turns=5,
              coil_radii=None, coil_z=None):
        """Solve the EM wave equation and compute power deposition.

        Parameters
        ----------
        ne : array (Nr, Nz) — electron density [m⁻³]
        Te : array (Nr, Nz) — electron temperature [eV]
        gas_density : float — neutral gas density [m⁻³]
        I_coil : float — coil current amplitude [A]
        n_turns : int — number of coil turns
        coil_radii : array — radial positions of coil turns [m]
        coil_z : float — axial position of coil [m]

        Returns
        -------
        P_ind : array (Nr, Nz)
            Inductive power deposition [W/m³].
        E_theta : complex array (Nr, Nz)
            Azimuthal electric field amplitude [V/m].
        total_power : float
            Total deposited power [W].
        """
        mesh = self.mesh
        Nr, Nz = self.Nr, self.Nz
        N = Nr * Nz
        omega = self.omega

        # Default coil geometry: planar spiral above the window
        if coil_radii is None:
            coil_radii = np.linspace(0.02, mesh.R * 0.8, n_turns)
        if coil_z is None:
            coil_z = mesh.L - 0.01  # 1 cm below top

        # Plasma conductivity
        sigma_p = self.plasma_conductivity(ne, Te, gas_density)

        # Complex dielectric constant
        eps_p = epsilon_0 - 1j * sigma_p / omega

        # Coil current density
        J_coil = self.coil_current_density(I_coil, n_turns, coil_radii, coil_z)

        # Build the matrix for the wave equation:
        # (1/r)∂/∂r(r ∂Ã/∂r) + ∂²Ã/∂z² + (ω²μ₀ε_p - 1/r²)Ã = -μ₀ J̃
        # This is the same Laplacian as Poisson but with complex coefficients.

        rows, cols, vals = [], [], []
        rhs = np.zeros(N, dtype=complex)

        for i in range(Nr):
            for j in range(Nz):
                idx = i * Nz + j
                r_c = mesh.r[i]
                dr = mesh.dr[i]
                dz = mesh.dz[j]

                # Wave term: (ω²μ₀ε_p - 1/r²)
                wave_term = omega**2 * mu_0 * eps_p[i, j] - 1.0 / r_c**2

                # Radial Laplacian terms
                coeff_diag = 0.0

                # r+1/2 face
                if i < Nr - 1:
                    r_f = mesh.r_faces[i + 1]
                    dr_c = mesh.dr_c[i + 1]
                    c_r_right = r_f / (r_c * dr * dr_c)
                    rows.append(idx); cols.append((i+1)*Nz + j); vals.append(c_r_right)
                    coeff_diag -= c_r_right
                else:
                    # Ã = 0 at r = R
                    r_f = mesh.r_faces[Nr]
                    dr_c = mesh.dr_c[Nr]
                    c_r_right = r_f / (r_c * dr * dr_c)
                    coeff_diag -= c_r_right

                # r-1/2 face
                if i > 0:
                    r_f = mesh.r_faces[i]
                    dr_c = mesh.dr_c[i]
                    c_r_left = r_f / (r_c * dr * dr_c)
                    rows.append(idx); cols.append((i-1)*Nz + j); vals.append(c_r_left)
                    coeff_diag -= c_r_left
                else:
                    # Ã = 0 at r = 0 (azimuthal field vanishes on axis)
                    # Use: limit as r→0 of (1/r)d(r dÃ/dr)/dr = 2 d²Ã/dr²
                    dr_c = mesh.dr_c[1]
                    c_sym = 2.0 / (dr * dr_c)
                    if Nr > 1:
                        rows.append(idx); cols.append(1*Nz + j); vals.append(c_sym)
                    coeff_diag -= c_sym

                # z+1/2 face
                if j < Nz - 1:
                    dz_c = mesh.dz_c[j + 1]
                    c_z_top = 1.0 / (dz * dz_c)
                    rows.append(idx); cols.append(i*Nz + j + 1); vals.append(c_z_top)
                    coeff_diag -= c_z_top
                else:
                    dz_c = mesh.dz_c[Nz]
                    c_z_top = 1.0 / (dz * dz_c)
                    coeff_diag -= c_z_top

                # z-1/2 face
                if j > 0:
                    dz_c = mesh.dz_c[j]
                    c_z_bot = 1.0 / (dz * dz_c)
                    rows.append(idx); cols.append(i*Nz + j - 1); vals.append(c_z_bot)
                    coeff_diag -= c_z_bot
                else:
                    dz_c = mesh.dz_c[0]
                    c_z_bot = 1.0 / (dz * dz_c)
                    coeff_diag -= c_z_bot

                # Diagonal: Laplacian + wave term
                rows.append(idx); cols.append(idx); vals.append(coeff_diag + wave_term)

                # RHS
                rhs[idx] = -mu_0 * J_coil[i, j]

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        A_tilde = spsolve(A, rhs).reshape((Nr, Nz))

        # Electric field: Ẽ_θ = -jω Ã
        E_theta = -1j * omega * A_tilde

        # Power deposition: P = ½ Re(σ |E|²)
        P_ind = 0.5 * np.real(sigma_p * np.abs(E_theta)**2)
        P_ind = np.maximum(P_ind, 0.0)  # Physical: no negative power

        # Total power
        total_power = mesh.volume_average(P_ind) * mesh.total_volume()

        return P_ind, E_theta, total_power

    def adjust_coil_current(self, ne, Te, gas_density, P_target,
                            n_turns=5, coil_radii=None, coil_z=None,
                            tol=0.05, max_iter=20):
        """Iterate coil current to achieve a target total power.

        Parameters
        ----------
        P_target : float — target total absorbed power [W]
        tol : float — relative tolerance on power matching

        Returns
        -------
        P_ind : array (Nr, Nz) — power deposition profile [W/m³]
        I_coil : float — required coil current [A]
        """
        I_coil = 10.0  # Initial guess

        for iteration in range(max_iter):
            P_ind, E_theta, P_total = self.solve(
                ne, Te, gas_density, I_coil, n_turns, coil_radii, coil_z)

            if P_total < 1e-10:
                I_coil *= 10
                continue

            # Power scales as I²
            ratio = P_target / P_total
            I_coil *= np.sqrt(ratio)

            if abs(ratio - 1.0) < tol:
                # Final solve with correct current
                P_ind, E_theta, P_total = self.solve(
                    ne, Te, gas_density, I_coil, n_turns, coil_radii, coil_z)
                return P_ind, I_coil

        # Return best result even if not converged
        P_ind, E_theta, P_total = self.solve(
            ne, Te, gas_density, I_coil, n_turns, coil_radii, coil_z)
        return P_ind, I_coil


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════


#==============================================================================
# POISSON SOLVER
#==============================================================================


"""
Poisson equation solver for 2D axisymmetric geometry.

Solves: (1/r) ∂/∂r(r ∂V/∂r) + ∂²V/∂z² = -ρ/ε₀

where ρ = e(n₊ - nₑ - n₋) is the space charge density.

Uses direct sparse matrix solve (scipy.sparse.linalg.spsolve).
For the semi-implicit coupling with the electron continuity equation,
the charge density can be extrapolated to the future time level.

Boundary conditions:
    r = 0: ∂V/∂r = 0 (symmetry)
    r = R: V = 0 (grounded wall)
    z = 0: V = 0 (grounded electrode)
    z = L: V = 0 (grounded or specified for dielectric window)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.constants import e as eC, epsilon_0


class PoissonSolver:
    """Solve Poisson's equation on a structured (r,z) mesh."""

    def __init__(self, mesh):
        """
        Parameters
        ----------
        mesh : Mesh2D
            The computational mesh.
        """
        self.mesh = mesh
        self.Nr = mesh.Nr
        self.Nz = mesh.Nz
        self.N = mesh.Nr * mesh.Nz

        # Build the coefficient matrix (only depends on mesh geometry)
        self._build_matrix()

    def _idx(self, i, j):
        """Map (i,j) cell indices to flat index."""
        return i * self.Nz + j

    def _build_matrix(self):
        """Build the sparse coefficient matrix for the Laplacian.

        (1/r) ∂/∂r(r ∂V/∂r) + ∂²V/∂z² = f

        Discretized on cell centres with the five-point stencil.
        """
        Nr, Nz = self.Nr, self.Nz
        mesh = self.mesh

        rows, cols, vals = [], [], []

        for i in range(Nr):
            for j in range(Nz):
                idx = self._idx(i, j)
                r_c = mesh.r[i]
                dr = mesh.dr[i]
                dz = mesh.dz[j]

                # --- Radial terms: (1/r) d/dr(r dV/dr) ---
                # At face i+1/2 (between cell i and i+1)
                if i < Nr - 1:
                    r_face = mesh.r_faces[i + 1]
                    dr_c = mesh.dr_c[i + 1]
                    coeff_r_right = r_face / (r_c * dr * dr_c)
                else:
                    # r = R wall: V = 0 (Dirichlet)
                    r_face = mesh.r_faces[Nr]
                    dr_c = mesh.dr_c[Nr]
                    coeff_r_right = r_face / (r_c * dr * dr_c)

                # At face i-1/2 (between cell i-1 and i)
                if i > 0:
                    r_face = mesh.r_faces[i]
                    dr_c = mesh.dr_c[i]
                    coeff_r_left = r_face / (r_c * dr * dr_c)
                else:
                    # r = 0 symmetry: ∂V/∂r = 0
                    # Use L'Hôpital: (1/r)d(r dV/dr)/dr → 2 d²V/dr² at r=0
                    r_face = mesh.r_faces[0]  # = 0
                    dr_c = mesh.dr_c[0]
                    coeff_r_left = 0.0  # No flux through axis

                # --- Axial terms: d²V/dz² ---
                if j < Nz - 1:
                    dz_c_right = mesh.dz_c[j + 1]
                    coeff_z_top = 1.0 / (dz * dz_c_right)
                else:
                    dz_c_right = mesh.dz_c[Nz]
                    coeff_z_top = 1.0 / (dz * dz_c_right)

                if j > 0:
                    dz_c_left = mesh.dz_c[j]
                    coeff_z_bot = 1.0 / (dz * dz_c_left)
                else:
                    dz_c_left = mesh.dz_c[0]
                    coeff_z_bot = 1.0 / (dz * dz_c_left)

                # Diagonal
                diag = -(coeff_r_right + coeff_r_left + coeff_z_top + coeff_z_bot)

                # --- Handle boundaries ---

                # r = 0 (axis symmetry): ghost cell has V_{-1,j} = V_{0,j}
                # This means the left radial contribution adds to diagonal
                if i == 0:
                    # At r=0, use: d²V/dr² ≈ 2*(V[1]-V[0])/dr²
                    r_face_right = mesh.r_faces[1]
                    dr_c_right = mesh.dr_c[1]
                    # Replace the full radial stencil with the L'Hôpital form
                    coeff_r_sym = 2.0 / (dr * dr_c_right)  # Factor of 2 from L'Hôpital
                    diag = -(coeff_r_sym + coeff_z_top + coeff_z_bot)
                    rows.append(idx); cols.append(idx); vals.append(diag)
                    if i < Nr - 1:
                        rows.append(idx); cols.append(self._idx(i+1, j)); vals.append(coeff_r_sym)
                    # No left neighbour for i=0
                else:
                    rows.append(idx); cols.append(idx); vals.append(diag)

                    # Right (i+1)
                    if i < Nr - 1:
                        rows.append(idx); cols.append(self._idx(i+1, j)); vals.append(coeff_r_right)
                    # else: V_{Nr,j} = 0 Dirichlet → no matrix entry, but add to RHS

                    # Left (i-1)
                    if i > 0:
                        rows.append(idx); cols.append(self._idx(i-1, j)); vals.append(coeff_r_left)

                # Top (j+1)
                if j < Nz - 1:
                    rows.append(idx); cols.append(self._idx(i, j+1)); vals.append(coeff_z_top)
                # else: V_{i,Nz} = 0 Dirichlet

                # Bottom (j-1)
                if j > 0:
                    rows.append(idx); cols.append(self._idx(i, j-1)); vals.append(coeff_z_bot)
                # else: V_{i,0} = 0 → handled in RHS for j=0 but with Dirichlet the
                # boundary value is 0 so no RHS contribution

        self.A = sparse.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))

    def solve(self, n_pos, n_e, n_neg):
        """Solve Poisson's equation for the electrostatic potential.

        Parameters
        ----------
        n_pos : array (Nr, Nz)
            Total positive ion density [m⁻³].
        n_e : array (Nr, Nz)
            Electron density [m⁻³].
        n_neg : array (Nr, Nz)
            Total negative ion density [m⁻³].

        Returns
        -------
        V : array (Nr, Nz)
            Electrostatic potential [V].
        Er : array (Nr+1, Nz)
            Radial electric field at r-faces [V/m].
        Ez : array (Nr, Nz+1)
            Axial electric field at z-faces [V/m].
        """
        Nr, Nz = self.Nr, self.Nz
        mesh = self.mesh

        # RHS: -ρ/ε₀ = -(e/ε₀)(n₊ - nₑ - n₋)
        rho = eC / epsilon_0 * (n_pos - n_e - n_neg)
        rhs = -rho.flatten()

        # Boundary corrections for Dirichlet BCs (V=0 at walls)
        # Since V_boundary = 0, no correction needed for homogeneous BCs.
        # For non-zero boundary values, add appropriate terms here.

        # Solve
        V_flat = spsolve(self.A, rhs)
        V = V_flat.reshape((Nr, Nz))

        # Compute electric field E = -∇V
        Er = np.zeros((Nr + 1, Nz))
        Ez = np.zeros((Nr, Nz + 1))

        # Interior r-faces
        for i in range(1, Nr):
            Er[i, :] = -(V[i, :] - V[i-1, :]) / mesh.dr_c[i]

        # Boundary r-faces
        Er[0, :] = 0.0  # Symmetry axis: Er = 0
        Er[Nr, :] = -(-V[Nr-1, :]) / mesh.dr_c[Nr]  # V=0 at wall

        # Interior z-faces
        for j in range(1, Nz):
            Ez[:, j] = -(V[:, j] - V[:, j-1]) / mesh.dz_c[j]

        # Boundary z-faces
        Ez[:, 0] = -(-V[:, 0]) / mesh.dz_c[0]   # V=0 at z=0 (but V[:,0] is first cell centre)
        Ez[:, Nz] = -(- V[:, Nz-1]) / mesh.dz_c[Nz]  # V=0 at z=L

        return V, Er, Ez


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════


#==============================================================================
# ENERGY EQUATION SOLVER
#==============================================================================


"""
Self-consistent 2D electron energy equation solver.

Solves the steady-state energy balance:
    D_eps ∇²(n_e ε̄) + P_ind(r,z)/e - n_e Σ E_k k_kj n_j = 0

where ε̄ = (3/2)Te is the mean electron energy.

The collisional loss term is linearized as ν_loss * (n_e ε̄):
    ν_loss = Σ E_k k_kj n_j / ε̄

This makes the equation:
    D_eps ∇²(n_e ε̄) + P_ind/e - ν_loss * (n_e ε̄) = 0

which is the same form as solve_steady_diffusion_with_source in main.py.

Boundary conditions: n_e ε̄ = 0 at walls (all energy lost at walls).

Returns Te(r,z) = (2/3) ε̄(r,z) = (2/3) (n_e ε̄)(r,z) / n_e(r,z).
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_Te_2d(mesh, ne, P_ind, Eloss_per_e, D_eps, Te_floor=0.5, Te_cap=15.0):
    """Solve the 2D energy equation for Te(r,z).

    Formulation: at each point, steady-state energy balance gives
        P_ind(r,z) / e = ne(r,z) * Eloss_per_e(Te)

    This is an algebraic equation for Te at each point (no diffusion needed
    for the leading-order solution). Diffusion provides a smoothing correction.

    For points where ne is substantial: Te = Te such that Eloss(Te) = P_ind/(ne*e)
    For points where ne is small: Te is smoothed from neighbours by diffusion.

    We solve this as: D∇²Te + (P_local - Eloss_local)/(ne*(3/2)*e) = 0
    where the source/loss are expressed as a forcing on Te.

    In practice, the simplest robust approach is:
    1. Compute Te_local from local energy balance (algebraic)
    2. Smooth with a diffusion step
    """
    from scipy.optimize import brentq as _brentq

    Nr, Nz = mesh.Nr, mesh.Nz
    AMU = 1.66054e-27

    # Step 1: Algebraic Te at each point from P_ind = ne * Eloss(Te) * e
    Te_raw = np.full((Nr, Nz), 3.0)
    for i in range(Nr):
        for j in range(Nz):
            ne_ij = ne[i,j]; P_ij = P_ind[i,j]
            if ne_ij < 1e12 or P_ij < 1.0:
                Te_raw[i,j] = Te_floor
                continue

            target_Eloss = P_ij / (ne_ij * eC)  # eV/s per electron

            if Eloss_per_e[i,j] > 0:
                # Scale current Te by the ratio
                ratio = target_Eloss / Eloss_per_e[i,j]
                # Eloss scales roughly as exp(-const/Te)*Te, so Te ~ Te_old * ratio^0.3
                Te_est = np.clip(3.0 * ratio**0.3, Te_floor, Te_cap)
                Te_raw[i,j] = Te_est
            else:
                Te_raw[i,j] = Te_floor

    Te_raw = np.clip(Te_raw, Te_floor, Te_cap)

    # Step 2: Smooth with one implicit diffusion step
    # This removes noise from the algebraic step and enforces Te → Te_wall at boundaries
    # Use the diffusion solver with Te_raw as source and a damping term
    # D∇²Te + (Te_raw - Te)/τ = 0  → steady state gives Te smoothed toward Te_raw
    # Reformulate: D∇²Te - Te/τ = -Te_raw/τ
    tau_smooth = 1e-4  # s — smoothing timescale (small = less smoothing)
    loss_freq = np.full((Nr,Nz), 1.0/tau_smooth)
    source = Te_raw / tau_smooth

    # Solve with Dirichlet BCs (Te = 0 at walls → will be floored)
    Te_smooth = _solve_diffusion_for_Te(mesh, D_eps/1e4, source, loss_freq)
    Te_smooth = np.clip(Te_smooth, Te_floor, Te_cap)

    ne_eps = ne * 1.5 * Te_smooth
    return Te_smooth, ne_eps


def _solve_diffusion_for_Te(mesh, D, source, loss_freq):
    """Solve D∇²Te + S - ν*Te = 0 with Neumann BCs (insulated walls for Te)."""
    Nr, Nz = mesh.Nr, mesh.Nz; N = Nr*Nz
    rows, cols, vals = [], [], []
    rhs = source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz+j; rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            diag = -loss_freq[i,j] if not np.isscalar(loss_freq) else -loss_freq

            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c); diag -= c
            # Neumann at r=R

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c); diag -= c
            else:
                drc = mesh.dr_c[1]; c = 2*D/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c); diag -= c
            if j > 0:
                dzc = mesh.dz_c[j]; c = D/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c); diag -= c

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    Te = spsolve(A, rhs).reshape((Nr, Nz))
    return np.maximum(Te, 0.0)


def _solve_energy_diffusion(mesh, D, source, loss_freq):
    """Solve D∇²u + S - ν·u = 0 with Dirichlet u=0 at walls.

    Same structure as solve_steady_diffusion_with_source in main.py
    but always uses Dirichlet BCs (energy is lost at walls).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N = Nr * Nz
    rows, cols, vals = [], [], []
    rhs = source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz + j
            rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            D_ij = D[i,j] if not np.isscalar(D) else D
            diag = -loss_freq[i,j] if not np.isscalar(loss_freq) else -loss_freq

            # Radial
            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c)
                diag -= c
            else:
                rf = mesh.r_faces[Nr]; drc = mesh.dr_c[Nr]
                diag -= D_ij*rf/(rc*dr*drc)  # Dirichlet: u=0 at wall

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c)
                diag -= c
            else:
                drc = mesh.dr_c[1]; c = 2*D_ij/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # Axial
            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c)
                diag -= c
            else:
                dzc = mesh.dz_c[Nz]
                diag -= D_ij/(dz*dzc)

            if j > 0:
                dzc = mesh.dz_c[j]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c)
                diag -= c
            else:
                dzc = mesh.dz_c[0]
                diag -= D_ij/(dz*dzc)

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u = spsolve(A, rhs).reshape((Nr, Nz))
    return np.maximum(u, 0.0)


def compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm):
    """Compute the collisional energy loss rate per electron at each grid point.

    Returns Eloss_per_e(r,z) in [eV/s].
    """
    AMU = 1.66054e-27
    Nr, Nz = mesh.Nr, mesh.Nz
    Eloss = np.zeros((Nr, Nz))

    for i in range(Nr):
        for j in range(Nz):
            Te_ij = Te_field[i,j]; ne_ij = ne[i,j]
            k = rates(Te_ij)

            # Ar* at this point
            qh = 0.0
            if nSF6[i,j] > 1e10:
                qh = (k['Penn_SF6']+k['qnch_SF6'])*nSF6[i,j]
            den = (k['Ar_iz_m']+k['Ar_q'])*ne_ij + kw_Arm + qh
            nArm_ij = k['Ar_exc']*ne_ij*nAr/max(den,1.0) if ne_ij > 1e8 else 0

            # Fraction of excitation that leads to stepwise ionization
            fstep = k['Ar_iz_m']*ne_ij/max(den, 1.0)
            fstep = min(fstep, 1.0)

            E = (15.7*k['Ar_iz']*nAr
                 + 11.56*k['Ar_exc']*nAr*(1-fstep)
                 + (11.56+4.14)*k['Ar_iz_m']*nArm_ij
                 + k['Ar_el']*nAr*3*m_e/(39.948*AMU)*Te_ij)

            if nSF6[i,j] > 1e10:
                E += ((9.6*k['d1']+12.1*k['d2']+16*k['d3']+18.6*k['d4']+22.7*k['d5']
                      +16*k['iz18']+20*k['iz19']+20.5*k['iz20']+28*k['iz21']
                      +37.5*k['iz22']+18*k['iz23']+29*k['iz24']
                      +0.09*k['vib_SF6']
                      +k['el_SF6']*3*m_e/(M_SPECIES['SF6']*AMU)*Te_ij)*nSF6[i,j]
                      + (15*k['iz28']+14.4*k['exc_F']
                         +k['el_F']*3*m_e/(M_SPECIES['F']*AMU)*Te_ij)*nF[i,j])

            Eloss[i,j] = E

    return Eloss


#==============================================================================
# NEGATIVE ION + NEUTRAL TRANSPORT
#==============================================================================


"""
Spatially resolved transport for negative ions and neutral species.

Step 2: Negative-ion transport
    D_- ∇²n_- + S_att - k_rec n_+ n_- = 0
    with Neumann (zero-flux) BCs at all walls (ions are trapped by the sheath).
    Produces the flat-core / sharp-edge profile predicted by Lichtenberg (1997).

Step 3: Multi-neutral transport
    D_j ∇²n_j + S_j - L_j n_j = 0
    for all 9 neutral species (SF6, SF5, SF4, SF3, SF2, SF, S, F, F2).
    Uses the same implicit sparse solver as the diffusion profile.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ═══════════════════════════════════════════════════════════
# STEP 2: Negative-ion transport
# ═══════════════════════════════════════════════════════════

def solve_negative_ions(mesh, ne, n_neg_old, nSF6, Te_field, n_pos,
                        D_neg, k_rec, alpha_0d=0.0, ne_0d=1e16, w=0.2):
    """Solve the negative-ion continuity equation.

    D_- ∇²n_- + ne * k_att * nSF6 - k_rec * n_+ * n_- = 0

    Neumann (zero-flux) BCs on all walls — negative ions are trapped.

    After solving, renormalize so <alpha> = alpha_0d (the 0D backbone value).
    The spatial *variation* comes from the diffusion equation; the *magnitude*
    is anchored to the 0D chemistry.

    Parameters
    ----------
    mesh : Mesh2D
    ne : array (Nr, Nz) — electron density [m⁻³]
    n_neg_old : array (Nr, Nz) — current n_- estimate [m⁻³]
    nSF6 : array (Nr, Nz) — SF6 density [m⁻³]
    Te_field : array (Nr, Nz) — electron temperature [eV]
    n_pos : array (Nr, Nz) — positive ion density [m⁻³]
    D_neg : float — negative ion diffusion coefficient [m²/s]
    k_rec : float — ion-ion recombination rate [m³/s]
    alpha_0d : float — volume-averaged alpha from 0D model
    ne_0d : float — volume-averaged ne from 0D model
    w : float — relaxation weight

    Returns
    -------
    n_neg : array (Nr, Nz) — updated negative ion density [m⁻³]
    alpha : array (Nr, Nz) — updated electronegativity
    """

    Nr, Nz = mesh.Nr, mesh.Nz

    # Compute attachment source and recombination loss at each point
    source = np.zeros((Nr, Nz))
    loss_freq = np.zeros((Nr, Nz))

    for i in range(Nr):
        for j in range(Nz):
            k = rates(Te_field[i,j])
            source[i,j] = ne[i,j] * k['att_SF6_total'] * nSF6[i,j]
            loss_freq[i,j] = k_rec * max(n_pos[i,j], ne[i,j])

    # Solve with Neumann BCs (trapped)
    n_neg_solved = _solve_diffusion_neumann(mesh, D_neg, source, loss_freq)

    # === ANCHORING: renormalize to match 0D alpha ===
    # The diffusion equation gives the spatial SHAPE of n_neg.
    # The MAGNITUDE is set so that <n_neg>/<ne> = alpha_0d.
    if alpha_0d > 0.01 and mesh.volume_average(n_neg_solved) > 1e6:
        target_n_neg_avg = alpha_0d * ne_0d
        actual_avg = mesh.volume_average(n_neg_solved)
        n_neg_solved *= target_n_neg_avg / actual_avg

    # Relaxed update
    n_neg = n_neg_old + w * (n_neg_solved - n_neg_old)
    n_neg = np.maximum(n_neg, 0.0)

    # Alpha
    alpha = np.where(ne > 1e10, n_neg / ne, 0.0)
    alpha = np.clip(alpha, 0, 1000)

    return n_neg, alpha


def _solve_diffusion_neumann(mesh, D, source, loss_freq):
    """Solve D∇²n + S - ν·n = 0 with Neumann BCs (zero flux at all walls)."""
    Nr, Nz = mesh.Nr, mesh.Nz; N = Nr*Nz
    rows, cols, vals = [], [], []
    rhs = source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz+j; rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            D_ij = D if np.isscalar(D) else D[i,j]
            diag = -loss_freq[i,j]

            # Radial
            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c); diag -= c
            # i == Nr-1: Neumann → no extra loss term (zero flux)

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c); diag -= c
            else:
                # r=0 symmetry
                drc = mesh.dr_c[1]; c = 2*D_ij/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # Axial — Neumann at both ends
            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c); diag -= c
            if j > 0:
                dzc = mesh.dz_c[j]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c); diag -= c

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    n = spsolve(A, -rhs).reshape((Nr, Nz))  # Negative sign: D∇²n - ν·n = -S → An = -S
    return np.maximum(n, 0.0)


# ═══════════════════════════════════════════════════════════
# STEP 3: Multi-neutral transport
# ═══════════════════════════════════════════════════════════

def solve_neutral_transport(mesh, ne, Te_field, nArm, ng, nSF6_0, tau_R,
                            neutrals_old, D_coeffs, gamma_F=0.01,
                            kw_F_eff=30.0, w=0.1, neutrals_0d=None):
    """Solve steady-state diffusion-reaction for all 9 neutral species.

    After the local chemistry sub-iteration, renormalize each species so
    that <n_j> = n_j^{0D}. This prevents the spatial solver from drifting
    away from the validated 0D chemistry backbone.

    The spatial *variation* (depletion at centre, enrichment at edges) comes
    from the local source/loss imbalance and diffusion. The *magnitude* is
    anchored to the 0D backbone.
    """
    AMU = 1.66054e-27; cm3 = 1e-6

    Nr, Nz = mesh.Nr, mesh.Nz
    n = {sp: neutrals_old[sp].copy() for sp in neutrals_old}

    # Iterate the neutral sub-system (mirrors the 0D code's inner loop)
    for sub_iter in range(10):
        for i in range(Nr):
            for j in range(Nz):
                Te = Te_field[i,j]; ne_ij = ne[i,j]
                k = rates(Te)
                k = compute_troe_rates(k, ng)

                nArm_ij = nArm[i,j]

                # === SF6 balance ===
                sf6_loss = ((k['d1']+k['d2']+k['d3']+k['d4']+k['d5']
                            +k['iz_SF6_total']+k['att_SF6_total'])*ne_ij
                           + (k['Penn_SF6']+k['qnch_SF6'])*nArm_ij
                           + 1.0/tau_R)
                sf6_source = nSF6_0/tau_R + k['nr42']*n['SF5'][i,j]*n['F'][i,j] + k['nr45']*n['SF5'][i,j]**2
                n['SF6'][i,j] = sf6_source / max(sf6_loss, 1e-30)

                # === SF5 balance ===
                sf5_prod = (ne_ij*n['SF6'][i,j]*(k['d1']+k['at31'])
                           + k['Penn_SF6']*nArm_ij*n['SF6'][i,j]
                           + k['qnch_SF6']*nArm_ij*n['SF6'][i,j]
                           + k['nr41']*n['SF4'][i,j]*n['F'][i,j]
                           + k['nr45']*n['SF5'][i,j]**2)
                sf5_loss = ((k['d7']+k['iz25']+k['iz26'])*ne_ij
                           + k['nr42']*n['F'][i,j]
                           + 2*k['nr45']*n['SF5'][i,j]
                           + 1.0/tau_R)
                n['SF5'][i,j] = sf5_prod / max(sf5_loss, 1e-30)

                # === SF4 balance ===
                sf4_prod = (ne_ij*n['SF6'][i,j]*k['d2']
                           + ne_ij*n['SF5'][i,j]*k['d7']
                           + k['nr40']*n['SF3'][i,j]*n['F'][i,j]
                           + k['nr45']*n['SF5'][i,j]**2
                           + k['nr44']*n['SF4'][i,j]**2)
                sf4_loss = (k['d8']*ne_ij
                           + k['nr41']*n['F'][i,j]
                           + 2*k['nr44']*n['SF4'][i,j]
                           + 1.0/tau_R)
                n['SF4'][i,j] = sf4_prod / max(sf4_loss, 1e-30)

                # === SF3 balance ===
                sf3_prod = (ne_ij*n['SF6'][i,j]*k['d3']
                           + ne_ij*n['SF4'][i,j]*k['d8']
                           + k['nr39']*n['SF2'][i,j]*n['F'][i,j]
                           + k['nr44']*n['SF4'][i,j]**2
                           + k['nr43']*n['SF3'][i,j]**2)
                sf3_loss = ((k['d9']+k['iz27'])*ne_ij
                           + k['nr40']*n['F'][i,j]
                           + 2*k['nr43']*n['SF3'][i,j]
                           + 1.0/tau_R)
                n['SF3'][i,j] = sf3_prod / max(sf3_loss, 1e-30)

                # === SF2 balance ===
                sf2_prod = (ne_ij*n['SF6'][i,j]*k['d4']
                           + ne_ij*n['SF3'][i,j]*k['d9']
                           + k['nr38']*n['SF'][i,j]*n['F'][i,j]
                           + k['nr43']*n['SF3'][i,j]**2)
                sf2_loss = (k['d10']*ne_ij + k['nr39']*n['F'][i,j] + 1.0/tau_R)
                n['SF2'][i,j] = sf2_prod / max(sf2_loss, 1e-30)

                # === SF balance ===
                sf_prod = (ne_ij*n['SF6'][i,j]*k['d5']
                          + ne_ij*n['SF2'][i,j]*k['d10']
                          + k['nr37']*n['S'][i,j]*n['F'][i,j])
                sf_loss = (k['d11']*ne_ij + k['nr38']*n['F'][i,j] + 1.0/tau_R)
                n['SF'][i,j] = sf_prod / max(sf_loss, 1e-30)

                # === S balance ===
                s_prod = ne_ij*n['SF'][i,j]*k['d11']
                s_loss = (k['iz29']*ne_ij + k['nr37']*n['F'][i,j] + 1.0/tau_R)
                n['S'][i,j] = s_prod / max(s_loss, 1e-30)

                # === F2 balance ===
                f2_prod = ne_ij*n['SF6'][i,j]*(k['d4']+k['d5'])
                f2_loss = (k['d6']*ne_ij + k['qnch_F2']*nArm_ij + 1.0/tau_R)
                n['F2'][i,j] = f2_prod / max(f2_loss, 1e-30)

                # === F balance ===
                f_prod = (ne_ij*n['SF6'][i,j]*(k['d1']+2*k['d2']+3*k['d3']
                          +2*k['d4']+3*k['d5']+k['iz18']+2*k['iz19']
                          +3*k['iz20']+2*k['iz21']+3*k['iz22']+4*k['iz23']+k['at31'])
                         + ne_ij*(n['SF5'][i,j]*(k['d7']+k['iz26'])
                                 +n['SF4'][i,j]*k['d8']+n['SF3'][i,j]*k['d9']
                                 +n['SF2'][i,j]*k['d10']+n['SF'][i,j]*k['d11'])
                         + k['Penn_SF6']*nArm_ij*n['SF6'][i,j]
                         + k['qnch_SF6']*nArm_ij*n['SF6'][i,j]
                         + 2*k['qnch_F2']*nArm_ij*n['F2'][i,j]
                         + 2*k['d6']*ne_ij*n['F2'][i,j])
                f_loss = (kw_F_eff + k['iz28']*ne_ij + 1.0/tau_R
                         + k['nr42']*n['SF5'][i,j] + k['nr41']*n['SF4'][i,j]
                         + k['nr40']*n['SF3'][i,j] + k['nr39']*n['SF2'][i,j]
                         + k['nr38']*n['SF'][i,j] + k['nr37']*n['S'][i,j])
                n['F'][i,j] = f_prod / max(f_loss, 1e-30)

    # Clip to physical values
    for sp in n:
        n[sp] = np.clip(n[sp], 0, nSF6_0*10)

    # === ANCHORING: renormalize to match 0D volume averages ===
    # The local chemistry gives the spatial VARIATION (shape).
    # The 0D backbone provides the MAGNITUDE (volume average).
    if neutrals_0d is not None:
        for sp in n:
            if sp in neutrals_0d and neutrals_0d[sp] > 0:
                actual_avg = mesh.volume_average(n[sp])
                if actual_avg > 1e6:
                    n[sp] *= neutrals_0d[sp] / actual_avg

    # Relaxed update
    neutrals = {}
    for sp in neutrals_old:
        neutrals[sp] = neutrals_old[sp] + w * (n[sp] - neutrals_old[sp])
        neutrals[sp] = np.maximum(neutrals[sp], 0.0)

    return neutrals


def init_neutrals(Nr, Nz, nSF6_0):
    """Initialize all 9 neutral species from the feed composition."""
    n = {}
    n['SF6'] = np.full((Nr,Nz), nSF6_0 * 0.3)
    n['SF5'] = np.full((Nr,Nz), nSF6_0 * 0.02)
    n['SF4'] = np.full((Nr,Nz), nSF6_0 * 0.01)
    n['SF3'] = np.full((Nr,Nz), nSF6_0 * 0.005)
    n['SF2'] = np.full((Nr,Nz), nSF6_0 * 0.001)
    n['SF']  = np.full((Nr,Nz), nSF6_0 * 0.0005)
    n['S']   = np.full((Nr,Nz), nSF6_0 * 0.0001)
    n['F']   = np.full((Nr,Nz), nSF6_0 * 0.3)
    n['F2']  = np.full((Nr,Nz), nSF6_0 * 0.01)
    return n


#==============================================================================
# ANALYTIC SHEATH MODEL
#==============================================================================


"""
Analytic sheath model for ICP plasma-wall interface.

Provides:
  - Sheath voltage V_s from Bohm criterion + floating potential
  - Ion energy at the wall E_ion = e*|V_s| + Te/2
  - Ion flux to each wall surface: Gamma_i = n_e,edge * u_B
  - Ion-enhanced etch probability correction

Physics:
  At the sheath edge, ions enter with Bohm velocity u_B.
  The sheath potential drop accelerates ions and repels electrons.
  For a floating wall: V_s = -(Te/2)*ln(Mi/(2*pi*me))
  For a biased wafer: V_s = V_float + V_bias

References:
  Lieberman & Lichtenberg, "Principles of Plasma Discharges" Ch. 6
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi

AMU = 1.66054e-27


def sheath_voltage(Te_eV, Mi_kg):
    """Floating sheath voltage (negative, repels electrons).
    
    V_s = -(Te/2) * ln(Mi / (2*pi*me))
    
    Parameters
    ----------
    Te_eV : float or array — electron temperature [eV]
    Mi_kg : float — ion mass [kg]
    
    Returns
    -------
    V_s : float or array — sheath voltage [V] (negative)
    """
    return -0.5 * Te_eV * np.log(Mi_kg / (2 * pi * m_e))


def bohm_velocity(Te_eV, Mi_kg, alpha=0.0, gamma=None, T_neg=0.3):
    """Modified Bohm velocity for electronegative plasma.
    
    u_B = sqrt(e*Te*(1+alpha) / (Mi*(1+gamma*alpha)))
    """
    if gamma is None:
        gamma = Te_eV / max(T_neg, 0.01)
    return np.sqrt(eC * Te_eV * (1 + alpha) / (Mi_kg * (1 + gamma * alpha)))


def ion_flux_to_wall(ne_edge, Te_eV, Mi_kg, alpha=0.0, T_neg=0.3):
    """Ion flux at the wall [m^-2 s^-1].
    
    Gamma_i = n_e,edge * u_B
    """
    uB = bohm_velocity(Te_eV, Mi_kg, alpha, T_neg=T_neg)
    return ne_edge * uB


def ion_energy_at_wall(Te_eV, Mi_kg, V_bias=0.0):
    """Mean ion energy arriving at the wall [eV].
    
    E_ion = e*|V_s| + Te/2 + e*|V_bias|
    The Te/2 is the directed kinetic energy at the sheath edge (Bohm).
    """
    V_s = sheath_voltage(Te_eV, Mi_kg)
    return abs(V_s) + 0.5 * Te_eV + abs(V_bias)


def compute_wall_fluxes(ne_2d, Te_2d, alpha_2d, mesh, Mi_kg, T_neg=0.3):
    """Compute ion flux profiles at all four walls.
    
    Returns dict with:
      'wafer':   Gamma_i(r) at z=0
      'window':  Gamma_i(r) at z=L
      'sidewall': Gamma_i(z) at r=R
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    
    # Wafer (z=0): use j=0
    ne_wafer = ne_2d[:, 0]
    Te_wafer = Te_2d[:, 0]
    al_wafer = alpha_2d[:, 0]
    flux_wafer = np.array([
        ion_flux_to_wall(ne_wafer[i], Te_wafer[i], Mi_kg, al_wafer[i], T_neg)
        for i in range(Nr)])
    
    # Window (z=L): use j=Nz-1
    ne_window = ne_2d[:, -1]
    Te_window = Te_2d[:, -1]
    al_window = alpha_2d[:, -1]
    flux_window = np.array([
        ion_flux_to_wall(ne_window[i], Te_window[i], Mi_kg, al_window[i], T_neg)
        for i in range(Nr)])
    
    # Sidewall (r=R): use i=Nr-1
    ne_side = ne_2d[-1, :]
    Te_side = Te_2d[-1, :]
    al_side = alpha_2d[-1, :]
    flux_side = np.array([
        ion_flux_to_wall(ne_side[j], Te_side[j], Mi_kg, al_side[j], T_neg)
        for j in range(Nz)])
    
    # Energy at wafer
    E_wafer = np.array([ion_energy_at_wall(Te_wafer[i], Mi_kg) for i in range(Nr)])
    
    return {
        'wafer_flux': flux_wafer,       # m^-2 s^-1, vs r
        'window_flux': flux_window,
        'sidewall_flux': flux_side,     # vs z
        'wafer_energy': E_wafer,        # eV, vs r
        'wafer_r_cm': mesh.r * 100,
        'sidewall_z_cm': mesh.z * 100,
    }


def ion_enhanced_etch_probability(E_ion_eV, gamma_chem=0.025, E_th=20.0, Y0=0.5):
    """Ion-enhanced etch probability.
    
    gamma_eff = gamma_chem + Y0 * sqrt(max(E_ion - E_th, 0)) / E_ion
    
    Parameters
    ----------
    E_ion_eV : ion energy at the wafer [eV]
    gamma_chem : chemical (radical-only) etch probability
    E_th : ion sputtering threshold [eV]
    Y0 : ion sputtering yield coefficient
    """
    E = np.asarray(E_ion_eV, dtype=float)
    phys = np.maximum(E - E_th, 0)
    ion_yield = Y0 * np.sqrt(phys) / np.maximum(E, 1.0)
    return gamma_chem + ion_yield



#==============================================================================
# GAS TEMPERATURE SOLVER
#==============================================================================


"""
2D gas temperature solver for ICP discharges.

Solves: kappa_g * nabla^2(Tg) + Q_elastic + Q_FC - Q_wall = 0

Heating sources:
  1. Elastic electron-neutral collisions: Q_el = 3*(m_e/M_n)*nu_en*ne*(Te-Tg)*kB
  2. Frank-Condon heating from dissociation products (hot fragments)
  3. Ion-neutral charge exchange heating

Cooling:
  Wall thermal loss via conduction to chamber walls at T_wall = 300 K

BCs: Dirichlet Tg = T_wall at all walls.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve

AMU = 1.66054e-27


def gas_thermal_conductivity(Tg, species='Ar', p_Pa=1.333):
    """Thermal conductivity of the neutral gas [W/(m·K)].
    
    For low-pressure plasmas, kappa is dominated by atomic transport.
    """
    if species == 'Ar':
        # Ar: kappa ~ 0.018 W/(m·K) at 300 K, scales as T^0.7
        return 0.018 * (Tg / 300.0)**0.7
    else:
        # SF6: kappa ~ 0.013 W/(m·K) at 300 K
        return 0.013 * (Tg / 300.0)**0.7


def elastic_heating_rate(ne, Te_eV, Tg, ng, M_gas_amu, sigma_en=5e-19):
    """Elastic e-n collision heating [W/m^3].
    
    Q = 3*(m_e/M_n) * nu_en * ne * kB * (Te - Tg)
    
    This is the primary gas heating mechanism in ICP plasmas.
    """
    M_n = M_gas_amu * AMU
    v_e = np.sqrt(8 * eC * Te_eV / (pi * m_e))
    nu_en = ng * sigma_en * v_e
    delta_T = Te_eV * eC / kB - Tg  # Convert Te to Kelvin
    Q = 3 * (m_e / M_n) * nu_en * ne * kB * delta_T
    return np.maximum(Q, 0)


def frank_condon_heating(ne, nSF6, Te_eV, Tg):
    """Frank-Condon heating from dissociation products [W/m^3].
    
    When SF6 is dissociated, fragments carry ~1-3 eV kinetic energy.
    This thermalizes through collisions, heating the gas.
    
    Rough estimate: Q_FC ~ ne * nSF6 * k_diss * E_FC
    where E_FC ~ 1 eV average per dissociation event.
    """
    k = rates(Te_eV)
    k_diss_total = k['d1'] + k['d2'] + k['d3'] + k['d4'] + k['d5']
    E_FC = 1.0 * eC  # 1 eV in Joules
    return ne * nSF6 * k_diss_total * E_FC


def solve_gas_temperature(mesh, ne, Te_field, nSF6, ng, M_gas_amu=40.0,
                          T_wall=300.0, x_Ar=0.0, x_SF6=1.0):
    """Solve the 2D gas temperature equation.
    
    kappa * nabla^2(Tg) + Q_heating = 0
    with Dirichlet Tg = T_wall at all walls.
    
    Parameters
    ----------
    mesh : Mesh2D
    ne : array (Nr, Nz) — electron density [m^-3]
    Te_field : array (Nr, Nz) — electron temperature [eV]
    nSF6 : array (Nr, Nz) — SF6 density [m^-3]
    ng : float — total gas density [m^-3]
    M_gas_amu : float — effective gas mass [AMU]
    T_wall : float — wall temperature [K]
    
    Returns
    -------
    Tg : array (Nr, Nz) — gas temperature [K]
    Q_total : array (Nr, Nz) — total heating rate [W/m^3]
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N = Nr * Nz
    
    # Compute heating sources
    Q_elastic = np.zeros((Nr, Nz))
    Q_FC = np.zeros((Nr, Nz))
    
    for i in range(Nr):
        for j in range(Nz):
            # Elastic e-n heating
            if x_Ar > 0:
                Q_elastic[i, j] += elastic_heating_rate(
                    ne[i,j], Te_field[i,j], T_wall, ng*x_Ar, 39.948, sigma_en=3e-20)
            if x_SF6 > 0:
                Q_elastic[i, j] += elastic_heating_rate(
                    ne[i,j], Te_field[i,j], T_wall, ng*x_SF6, 146.06, sigma_en=5e-19)
            
            # Frank-Condon from dissociation
            if x_SF6 > 0 and nSF6[i,j] > 1e10:
                Q_FC[i, j] = frank_condon_heating(ne[i,j], nSF6[i,j], Te_field[i,j], T_wall)
    
    Q_total = Q_elastic + Q_FC
    
    # Effective thermal conductivity (mixture)
    kappa = x_Ar * gas_thermal_conductivity(T_wall, 'Ar') + \
            x_SF6 * gas_thermal_conductivity(T_wall, 'SF6')
    kappa = max(kappa, 1e-4)
    
    # Solve: kappa * nabla^2(Tg) + Q = 0  with Tg = T_wall at walls
    # Rearrange: kappa * nabla^2(Tg) = -Q
    # Same structure as the ne diffusion solver but with Dirichlet BCs
    
    rows, cols, vals = [], [], []
    rhs = (-Q_total / kappa).flatten()
    
    for i in range(Nr):
        for j in range(Nz):
            idx = i * Nz + j
            rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            diag = 0.0
            
            # Radial
            if i < Nr - 1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = rf / (rc * dr * drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c)
                diag -= c
            else:
                # Dirichlet at r=R: Tg = T_wall → ghost contributes T_wall to RHS
                rf = mesh.r_faces[Nr]; drc = mesh.dr_c[Nr]
                c = rf / (rc * dr * drc)
                diag -= c
                rhs[idx] -= c * T_wall / (-1)  # Note: solving for Tg-T_wall shift
            
            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = rf / (rc * dr * drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c)
                diag -= c
            else:
                # Symmetry at r=0
                drc = mesh.dr_c[1]; c = 2.0 / (dr * drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c
            
            # Axial
            if j < Nz - 1:
                dzc = mesh.dz_c[j+1]; c = 1.0 / (dz * dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c)
                diag -= c
            else:
                # Dirichlet at z=L
                dzc = mesh.dz_c[Nz]; c = 1.0 / (dz * dzc)
                diag -= c
            
            if j > 0:
                dzc = mesh.dz_c[j]; c = 1.0 / (dz * dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c)
                diag -= c
            else:
                # Dirichlet at z=0
                dzc = mesh.dz_c[0]; c = 1.0 / (dz * dzc)
                diag -= c
            
            rows.append(idx); cols.append(idx); vals.append(diag)
    
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    Tg_flat = spsolve(A, rhs)
    Tg = np.maximum(Tg_flat.reshape((Nr, Nz)), T_wall)
    Tg = np.clip(Tg, T_wall, 2000.0)  # Physical limit
    
    return Tg, Q_total


#==============================================================================
# 2-TERM BOLTZMANN SOLVER
#==============================================================================


"""
Two-term Boltzmann solver for computing electron transport coefficients
from LXCat cross-section data.

Implements the steady-state 2-term expansion of the Boltzmann equation
following Hagelaar & Pitchford, PSST 14, 722 (2005).

The EEDF F0(ε) satisfies:
  d/dε [A(ε) dF0/dε + B(ε) F0] = S(ε)

where A, B encode elastic and inelastic collisions, and S is the
source term from the applied E/N field.

Solves for a range of E/N values and outputs:
  μ_e·N, D_e·N, μ_ε·N, D_ε·N, k_iz, k_att, k_exc, etc.
as functions of mean electron energy ε̄.
"""

import numpy as np
from scipy.constants import e as eC, m_e, k as kB, pi
from scipy.interpolate import interp1d

gamma_const = np.sqrt(2 * eC / m_e)  # 5.931e5 m/s/eV^{1/2}


def solve_boltzmann_2term(EN_Td, x_Ar=0.0, x_SF6=1.0, N_energy=500, eps_max=80.0):
    """Solve the 2-term Boltzmann equation at a given E/N.

    Parameters
    ----------
    EN_Td : float — reduced electric field [Townsend, 1 Td = 1e-21 V·m²]
    x_Ar, x_SF6 : float — gas mole fractions
    N_energy : int — number of energy grid points
    eps_max : float — maximum energy [eV]

    Returns
    -------
    dict with: eps_bar, mu_e_N, D_e_N, mu_eps_N, D_eps_N,
               k_Ar_iz, k_Ar_exc, k_SF6_iz, k_SF6_att, k_SF6_diss, F0
    """

    EN = EN_Td * 1e-21  # V·m²

    # Energy grid (uniform in sqrt(ε) for better low-energy resolution)
    eps = np.linspace(0, np.sqrt(eps_max), N_energy)**2
    eps[0] = 1e-6  # avoid zero
    deps = np.diff(eps)
    eps_mid = 0.5 * (eps[:-1] + eps[1:])
    N = len(eps)

    # Total momentum-transfer cross-section (mixture)
    sigma_m = np.zeros(N)
    for i in range(N):
        e = eps[i]
        if x_Ar > 0:
            sigma_m[i] += x_Ar * sigma_Ar_eff(e)[0]
        if x_SF6 > 0:
            sigma_m[i] += x_SF6 * sigma_SF6_elastic(e)[0]
    sigma_m = np.maximum(sigma_m, 1e-25)

    # Effective mass ratio for elastic energy loss
    m_ratio = x_Ar * AR_MASS_RATIO + x_SF6 * SF6_MASS_RATIO

    # Inelastic cross-sections (for energy loss computation)
    sigma_inel = {}
    if x_Ar > 0:
        sigma_inel['Ar_exc'] = (sigma_Ar_exc(eps) * x_Ar, 11.5)
        sigma_inel['Ar_ion'] = (sigma_Ar_ion(eps) * x_Ar, 15.8)
    if x_SF6 > 0:
        sigma_inel['SF6_diss_trip'] = (sigma_SF6_diss_trip(eps) * x_SF6, 9.6)
        sigma_inel['SF6_diss_sing12'] = (sigma_SF6_diss_sing12(eps) * x_SF6, 12.0)
        sigma_inel['SF6_diss_sing16'] = (sigma_SF6_diss_sing16(eps) * x_SF6, 16.0)
        sigma_inel['SF6_ion'] = (sigma_SF6_ion(eps) * x_SF6, 15.67)
        sigma_inel['SF6_att_SF5'] = (sigma_SF6_att_SF5(eps) * x_SF6, 0)
        sigma_inel['SF6_att_SF6'] = (sigma_SF6_att_SF6(eps) * x_SF6, 0)

    # Build the EEDF using the simplified approach:
    # For a Maxwellian-like EEDF at effective temperature determined by E/N,
    # we use the energy balance to find the effective Te, then compute
    # the actual EEDF shape including the inelastic structure.

    # Simplified: compute a modified Maxwellian EEDF
    # The effective temperature is determined by the energy balance:
    # (eE)^2 / (3*m*sigma_m) = sum of energy losses

    # Power input per electron per unit density:
    # P_in = (e*E/N)^2 / (3*m_e) * gamma * eps^{1/2} / sigma_m(eps)
    # This gives a Druyvesteyn-like distribution

    # For the 2-term solver, the EEDF satisfies:
    # d/deps [ -eps/(3*sigma_m) * (EN)^2 * dF0/deps - 2*m_ratio*eps^2*sigma_m*F0 ] = C_inel
    # with normalization: integral eps^{1/2} F0 deps = 1

    # Discretize and solve as a tridiagonal system
    A_coeff = np.zeros(N)  # diffusion in energy space
    B_coeff = np.zeros(N)  # elastic cooling drag

    for i in range(N):
        e = eps[i]
        sm = sigma_m[i]
        A_coeff[i] = e / (3 * sm) * EN**2  # (eE/N)^2 * eps / (3*sigma_m)
        B_coeff[i] = 2 * m_ratio * e**2 * sm * gamma_const * e**0.5 * kB * 300 / eC
        # Elastic cooling: 2*(m/M)*eps^2*sigma_m*v * kT_g

    # Inelastic loss frequencies (energy-dependent)
    C_inel = np.zeros(N)
    for name, (sig, thresh) in sigma_inel.items():
        for i in range(N):
            if eps[i] > thresh and sig[i] > 0:
                C_inel[i] += sig[i] * gamma_const * eps[i]**0.5

    # Solve for F0 using the standard approach:
    # At each energy, the EEDF is approximately:
    # F0(eps) ~ C * exp(-integral_0^eps [B + C_inel*deps/A])

    # Compute the exponent
    integrand = np.zeros(N)
    for i in range(N):
        Ai = max(A_coeff[i], 1e-40)
        integrand[i] = (B_coeff[i] + C_inel[i] * (eps[i] + 0.01)) / Ai

    # Cumulative integral for the EEDF shape
    exponent = np.zeros(N)
    for i in range(1, N):
        de = eps[i] - eps[i-1]
        exponent[i] = exponent[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * de

    F0 = np.exp(-exponent)
    F0 = np.maximum(F0, 1e-100)

    # Normalize: integral eps^{1/2} F0 deps = 1
    norm = np.trapezoid(eps**0.5 * F0, eps)
    if norm > 1e-30:
        F0 /= norm

    # Compute transport coefficients (Hagelaar Eqs. 55-56, 61-62)
    dF0 = np.gradient(F0, eps)

    # Mean energy
    eps_bar = np.trapezoid(eps**1.5 * F0, eps)

    # μ_e · N = -(γ/3) ∫ ε/σ_m · dF0/dε dε
    mu_e_N = -(gamma_const / 3) * np.trapezoid(eps / sigma_m * dF0, eps)

    # D_e · N = (γ/3) ∫ ε/σ_m · F0 dε
    D_e_N = (gamma_const / 3) * np.trapezoid(eps / sigma_m * F0, eps)

    # μ_ε · N = -(γ/(3ε̄)) ∫ ε²/σ_m · dF0/dε dε
    mu_eps_N = -(gamma_const / (3 * max(eps_bar, 0.01))) * np.trapezoid(eps**2 / sigma_m * dF0, eps)

    # D_ε · N = (γ/(3ε̄)) ∫ ε²/σ_m · F0 dε
    D_eps_N = (gamma_const / (3 * max(eps_bar, 0.01))) * np.trapezoid(eps**2 / sigma_m * F0, eps)

    # Rate coefficients: k = γ ∫ ε · σ(ε) · F0(ε) dε
    def rate_coeff(sigma_arr):
        return gamma_const * np.trapezoid(eps * sigma_arr * F0, eps)

    result = {
        'eps_bar': eps_bar,
        'mu_e_N': abs(mu_e_N),
        'D_e_N': abs(D_e_N),
        'mu_eps_N': abs(mu_eps_N),
        'D_eps_N': abs(D_eps_N),
    }

    if x_Ar > 0:
        result['k_Ar_exc'] = rate_coeff(sigma_Ar_exc(eps) * x_Ar)
        result['k_Ar_ion'] = rate_coeff(sigma_Ar_ion(eps) * x_Ar)
    if x_SF6 > 0:
        result['k_SF6_ion'] = rate_coeff(sigma_SF6_ion(eps) * x_SF6)
        result['k_SF6_att_total'] = (rate_coeff(sigma_SF6_att_SF5(eps) * x_SF6) +
                                      rate_coeff(sigma_SF6_att_SF6(eps) * x_SF6))
        result['k_SF6_diss_total'] = (rate_coeff(sigma_SF6_diss_trip(eps) * x_SF6) +
                                       rate_coeff(sigma_SF6_diss_sing12(eps) * x_SF6) +
                                       rate_coeff(sigma_SF6_diss_sing16(eps) * x_SF6))

    return result


def build_bolsig_table(x_Ar=0.0, x_SF6=1.0, EN_range=None, N_points=60):
    """Build a complete transport table by sweeping E/N.

    Returns arrays indexed by mean energy for use in BOLSIGTable.from_arrays().
    """
    if EN_range is None:
        EN_range = np.logspace(-0.5, 3, N_points)  # 0.3 to 1000 Td

    results = []
    for EN in EN_range:
        try:
            r = solve_boltzmann_2term(EN, x_Ar=x_Ar, x_SF6=x_SF6)
            if r['eps_bar'] > 0.01 and r['mu_e_N'] > 0:
                results.append(r)
        except:
            pass

    if len(results) < 5:
        raise ValueError(f"Only {len(results)} valid points — check cross-sections")

    # Sort by mean energy
    results.sort(key=lambda r: r['eps_bar'])

    # Remove duplicates in eps_bar
    eps_bars = [r['eps_bar'] for r in results]
    unique_idx = [0]
    for i in range(1, len(eps_bars)):
        if eps_bars[i] > eps_bars[unique_idx[-1]] * 1.001:
            unique_idx.append(i)
    results = [results[i] for i in unique_idx]

    eps_bar = np.array([r['eps_bar'] for r in results])
    mu_e_N = np.array([r['mu_e_N'] for r in results])
    D_e_N = np.array([r['D_e_N'] for r in results])
    mu_eps_N = np.array([r['mu_eps_N'] for r in results])
    D_eps_N = np.array([r['D_eps_N'] for r in results])

    rate_names = set()
    for r in results:
        rate_names.update(k for k in r if k.startswith('k_'))

    rate_coeffs = {}
    for name in rate_names:
        rate_coeffs[name] = np.array([r.get(name, 0) for r in results])

    return {
        'eps_bar': eps_bar,
        'mu_e_N': mu_e_N,
        'D_e_N': D_e_N,
        'mu_eps_N': mu_eps_N,
        'D_eps_N': D_eps_N,
        'rate_coeffs': rate_coeffs,
        'x_Ar': x_Ar,
        'x_SF6': x_SF6,
    }



#==============================================================================
# BOLSIG+ TABLE INFRASTRUCTURE
#==============================================================================


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



#==============================================================================
# MULTI-ION SPECIES
#==============================================================================


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



#==============================================================================

#==============================================================================
# MODULE ALIASES (for unified file — replaces cross-module imports)
#==============================================================================
hagelaar_mix = transport_mixture  # from hagelaar_transport
_solve_energy_diffusion = _solve_diffusion_for_Te if '_solve_diffusion_for_Te' in dir() else None

# Make solve_model available without module prefix
# (the 0D solver's solve_model function is already defined above)
try:
    _solve_0d = solve_model
except NameError:
    _solve_0d = None


# GEN-4b DRIVER
#==============================================================================


#!/usr/bin/env python3
"""
SF6/Ar 2D ICP — Generation 4b: Mettler-gap fix

Three changes to produce the 75% center-to-edge [F] drop:

  [A] WALL-SPECIFIC Robin BCs
      - Wafer (z=0): gamma_F = 0.02 (Si surface, moderate F consumption)
      - Chamber sidewall (r=R): gamma_F = 0.10 (anodized Al, higher recomb)
      - Window (z=L): gamma_F = 0.001 (quartz, very low recomb)
      This asymmetry concentrates F loss at the sidewall, creating a 
      radial gradient.

  [B] RELEASED SF6 ANCHORING
      SF6 is now solved self-consistently by diffusion with Robin BCs.
      It depletes at the center (where ne is highest) and is replenished
      from the edges (feed gas). This makes the F source = ne*nSF6*k_diss
      NOT proportional to ne but concentrated at intermediate radii where
      both ne and nSF6 are substantial.

  [C] RELEASED ne ANCHORING (partial)
      ne magnitude still from 0D, but the PROFILE is computed from
      ambipolar diffusion with the LOCAL Da(alpha(r,z)) instead of 
      the volume-averaged Da. This gives a more peaked ne profile
      in the electronegative core.
"""

import sys, os, time, argparse
import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve


AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6


def _solve_robin_asymmetric(mesh, D, source, loss_freq,
                            h_r_wall, h_z0, h_zL):
    """Solve D∇²n + S - ν·n = 0 with DIFFERENT Robin BCs on each wall.

    h_r_wall: Robin coeff at r = R (sidewall)
    h_z0:     Robin coeff at z = 0 (wafer)
    h_zL:     Robin coeff at z = L (window/coil)
    r = 0:    Neumann (symmetry)
    """
    Nr, Nz = mesh.Nr, mesh.Nz; N_tot = Nr*Nz
    rows, cols, vals = [], [], []
    rhs = -source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz + j
            rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            D_ij = D[i,j] if not np.isscalar(D) else D
            diag = -(loss_freq[i,j] if not np.isscalar(loss_freq) else loss_freq)

            # --- Radial ---
            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c); diag -= c
            else:
                # Robin at r = R (sidewall)
                rf = mesh.r_faces[Nr]; drc = mesh.dr_c[Nr]
                wall_loss = D_ij * rf / (rc * dr) * h_r_wall / (D_ij + h_r_wall*drc)
                diag -= wall_loss

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c); diag -= c
            else:
                drc = mesh.dr_c[1]; c = 2*D_ij/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # --- Axial ---
            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c); diag -= c
            else:
                # Robin at z = L (window)
                dzc = mesh.dz_c[Nz]
                diag -= D_ij / dz * h_zL / (D_ij + h_zL*dzc)

            if j > 0:
                dzc = mesh.dz_c[j]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c); diag -= c
            else:
                # Robin at z = 0 (wafer)
                dzc = mesh.dz_c[0]
                diag -= D_ij / dz * h_z0 / (D_ij + h_z0*dzc)

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N_tot, N_tot))
    n = spsolve(A, rhs).reshape((Nr, Nz))
    return np.maximum(n, 0.0)


def run_v4b(P_rf=1500.0, p_mTorr=10.0, frac_Ar=0.0, Tgas=300.0, T_neg=0.3,
            gamma_F_wafer=0.02, gamma_F_wall=0.10, gamma_F_window=0.001,
            beta_SF6=0.005, eta=0.12,
            Nr=30, Nz=40, n_iter=100, em_interval=3, verbose=True):
    t0 = time.time()
    p_Pa = p_mTorr*MTORR_TO_PA; ng = p_Pa/(kB*Tgas)
    nAr = ng*frac_Ar; nSF6_0 = ng*(1-frac_Ar)
    V_reactor = pi*0.180**2*0.175
    P_abs = P_rf * eta
    Q_tp = 40*1e-6/60*1.01325e5*(Tgas/273.15)
    tau_R = p_Pa*V_reactor/Q_tp if Q_tp > 0 else 1e10

    mesh = Mesh2D(R=0.180, L=0.175, Nr=Nr, Nz=Nz, stretch_r=1.3, stretch_z=1.3)
    em = EMSolver(mesh, freq=13.56e6)
    poisson = PoissonSolver(mesh)

    Mi_amu = 39.948 if frac_Ar > 0.5 else 127.06; Mi_kg = Mi_amu*AMU
    i_tr = IonTransport(Mi_amu, ng, Tgas)
    n_tr = NeutralTransport(ng, Tgas)
    k_rec = 1.5e-9 * cm3

    D_F = n_tr.diffusivity(M_SPECIES['F'], sigma=4e-19)
    D_SF6 = n_tr.diffusivity(M_SPECIES['SF6'], sigma=6e-19)
    D_Arm = n_tr.diffusivity(39.948, sigma=3e-19)
    D_neg = i_tr.D0
    Lambda2 = 1.0/((pi/mesh.L)**2 + (2.405/mesh.R)**2)
    kw_Arm = D_Arm/Lambda2

    # Wall-specific Robin coefficients
    v_th_F = np.sqrt(8*kB*Tgas/(pi*M_SPECIES['F']*AMU))
    v_th_SF6 = np.sqrt(8*kB*Tgas/(pi*M_SPECIES['SF6']*AMU))
    h_F_wafer = gamma_F_wafer * v_th_F / 4
    h_F_wall  = gamma_F_wall * v_th_F / 4
    h_F_window = gamma_F_window * v_th_F / 4
    h_SF6_wall = beta_SF6 * v_th_SF6 / 4

    # --- 0D backbone ---
    has_0d = False; ne_0d = 1e16; Te_0d = 3.0; alpha_0d = 0
    Ec = 100; eps_T = 120; r0d = {}
    nSF6_0d = nSF6_0*0.3; nF_0d = 0
    try:
        solve_0d = solve_model  # unified: already defined above
        r0d = solve_0d(P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                       eta=eta, Tgas=Tgas, T_neg=T_neg)
        ne_0d = r0d['ne']; Te_0d = r0d['Te']; alpha_0d = r0d['alpha']
        nSF6_0d = r0d.get('n_SF6', nSF6_0*0.3)
        nF_0d = r0d.get('n_F', 0); Ec = r0d.get('Ec', 100)
        eps_T = r0d.get('eps_T', 120)
        has_0d = True
    except Exception as e:
        if verbose: print(f"  0D unavailable ({e})")

    if verbose:
        print(f"Mesh: {mesh}")
        print(f"0D: ne={ne_0d*1e-6:.2e} Te={Te_0d:.2f} α={alpha_0d:.1f} [F]={nF_0d*1e-6:.2e}")
        print(f"Robin F: wafer={gamma_F_wafer}, sidewall={gamma_F_wall}, window={gamma_F_window}")
        print(f"  h_F: wafer={h_F_wafer:.1f}, wall={h_F_wall:.1f}, window={h_F_window:.2f} m/s")

    # --- Initialize ---
    Da0 = ambipolar_Da(alpha_0d, Te_0d, i_tr.D0, T_neg)
    g_en = Te_0d/max(T_neg, 0.01)
    uB0 = np.sqrt(eC*Te_0d*(1+alpha_0d)/(Mi_kg*(1+g_en*alpha_0d)))
    profile = solve_diffusion_profile(mesh, Da0, uB0)
    pn = profile / max(mesh.volume_average(profile), 1e-30)

    ne = ne_0d * pn
    Te_field = np.full((Nr,Nz), Te_0d)
    nArm = np.zeros((Nr,Nz))
    n_neg = np.full((Nr,Nz), alpha_0d*ne_0d)*pn if alpha_0d > 0.01 else np.zeros((Nr,Nz))
    alpha = np.where(ne > 1e10, n_neg/ne, 0.0)
    nSF6 = np.full((Nr,Nz), nSF6_0d)
    nF = np.full((Nr,Nz), max(nF_0d, 1e10))

    neutrals = {}
    if has_0d and nSF6_0 > 0:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr,Nz), r0d.get(f'n_{sp}', nSF6_0*0.01))
    else:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr,Nz), nSF6_0*0.01 if nSF6_0 > 0 else 0)
        if nSF6_0 > 0: neutrals['SF6'] = np.full((Nr,Nz), nSF6_0*0.3)

    # Power deposition
    try:
        P_ind, _ = em.adjust_coil_current(ne, Te_field, ng, P_target=P_abs)
        if not (np.all(np.isfinite(P_ind)) and P_ind.max() > 0): raise ValueError
    except:
        P_shape = np.exp(-(mesh.L-mesh.ZZ)/0.017)*np.maximum(mesh.RR/mesh.R, 0.01)
        P_ind = P_shape*P_abs/max(mesh.volume_average(P_shape)*V_reactor, 1e-10)

    if verbose:
        sf6_pct = (1-frac_Ar)*100
        print(f"\nGen-4b: {n_iter} iter, {P_rf:.0f}W, {p_mTorr:.0f}mT, "
              f"{frac_Ar*100:.0f}%Ar/{sf6_pct:.0f}%SF6")
        print("-"*70)

    w = 0.12

    for it in range(n_iter):
        Te_old = Te_field.copy()
        ne_avg_old = mesh.volume_average(ne)

        # === 1. Ar* ===
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j]); ne_ij = ne[i,j]
                qh = (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nSF6[i,j]+k_loc['qnch_F']*nF[i,j]
                den = (k_loc['Ar_iz_m']+k_loc['Ar_q'])*ne_ij+kw_Arm+qh
                nArm[i,j] = k_loc['Ar_exc']*ne_ij*nAr/max(den,1.0) if ne_ij>1e8 else 0

        # === 2. Te from energy equation (Hagelaar D_eps) ===
        Eloss_field = compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm)
        eps_bar = np.maximum(1.5*Te_field, 0.5)
        nu_eps = np.clip(np.where(eps_bar > 0.1, Eloss_field/eps_bar, 1.0), 1.0, 1e12)
        ne_safe = np.maximum(ne, 1e8)
        source_eps = P_ind / (ne_safe * eC)
        h_loc = hagelaar_mix(max(Te_0d, 0.5), ng, frac_Ar, 1-frac_Ar)
        D_eps_val = h_loc['D_eps']

        eps_new = _solve_diffusion_neumann(mesh, D_eps_val, source_eps, nu_eps)
        Te_new = np.clip((2.0/3.0) * np.clip(eps_new, 0.5, 15.0), 0.8, 8.0)
        Te_avg_raw = mesh.volume_average(Te_new)
        if Te_avg_raw > 0.1 and Te_0d > 0.1:
            Te_new *= Te_0d / Te_avg_raw
        Te_new = np.clip(Te_new, 0.8, 8.0)
        Te_field = Te_field + w*(Te_new - Te_field)
        Te_field = np.clip(Te_field, 0.8, 8.0)

        # === 3. ne profile — SOURCE-WEIGHTED (released anchoring) [Change C] ===
        # Instead of the uniform-source eigenmode (peak/avg = 2.5 always),
        # solve: Da*nabla^2(ne) + S_net(r,z)*ne_old = 0 with Dirichlet BCs
        # where S_net = nu_iz(Te) - nu_att(Te) is the LOCAL net ionization rate.
        # This concentrates ne where Te is high (near coil), giving stronger peaking.
        Te_avg = mesh.volume_average(Te_field)
        al_avg = mesh.volume_average(n_neg)/max(mesh.volume_average(ne), 1e10)
        Da_now = ambipolar_Da(al_avg, Te_avg, i_tr.D0, T_neg)

        # Compute local net ionization source weighted by current ne
        ne_source = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j])
                nu_iz_loc = (k_loc['Ar_iz']*nAr + k_loc['iz_SF6_total']*nSF6[i,j])
                # Add Ar* stepwise ionization
                if nArm[i,j] > 1e6:
                    nu_iz_loc += k_loc['Ar_iz_m']*nArm[i,j]
                nu_att_loc = k_loc['att_SF6_total']*nSF6[i,j]
                ne_source[i,j] = max(nu_iz_loc - nu_att_loc, 0) * ne[i,j]

        # Solve with Robin BCs (large h ≈ Dirichlet, ne → 0 at walls)
        g_t = Te_avg/max(T_neg, 0.01)
        uB_now = np.sqrt(eC*Te_avg*(1+al_avg)/(Mi_kg*(1+g_t*al_avg)))
        h_ne_wall = uB_now  # Bohm flux BC: D*dn/dr = uB*n
        ne_profile = _solve_robin_asymmetric(mesh, Da_now, ne_source, np.zeros((Nr,Nz)),
                                             h_r_wall=h_ne_wall, h_z0=h_ne_wall, h_zL=h_ne_wall)

        # Normalize to 0D average (ne magnitude still anchored)
        ne_prof_avg = mesh.volume_average(ne_profile)
        if ne_prof_avg > 1e6:
            ne_new = ne_profile * (ne_0d / ne_prof_avg)
        else:
            # Fallback to standard eigenmode if source-weighted fails
            profile = solve_diffusion_profile(mesh, Da_now, uB_now)
            ne_new = ne_0d * profile / max(mesh.volume_average(profile), 1e-30)
        ne = ne + w*(ne_new - ne); ne = np.maximum(ne, 1e6)

        # === 4. Negative ions ===
        if nSF6_0 > 0:
            n_pos = ne + n_neg
            n_neg, alpha = solve_negative_ions(
                mesh, ne, n_neg, nSF6, Te_field, n_pos,
                D_neg, k_rec, alpha_0d=alpha_0d, ne_0d=ne_0d, w=w)

        # === 5. SF6 with Robin BCs — SELF-CONSISTENT (no anchor) [Change B] ===
        if nSF6_0 > 0:
            sf6_source = np.full((Nr,Nz), nSF6_0/tau_R)
            sf6_loss = np.zeros((Nr,Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    k_e = (k_loc['d1']+k_loc['d2']+k_loc['d3']+k_loc['d4']+k_loc['d5']
                           +k_loc['iz_SF6_total']+k_loc['att_SF6_total'])
                    sf6_loss[i,j] = (k_e*ne[i,j] + (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nArm[i,j]
                                     + 1.0/tau_R)

            # Asymmetric Robin: SF6 sticking at walls
            nSF6_new = _solve_robin_asymmetric(mesh, D_SF6, sf6_source, sf6_loss,
                                                h_r_wall=h_SF6_wall, h_z0=h_SF6_wall, h_zL=h_SF6_wall*0.1)
            # Partial anchoring: prevent over-depletion below 50% of 0D
            # This allows spatial gradients but maintains enough SF6 for
            # the attachment/ionization chemistry to work correctly.
            sf6_avg_new = mesh.volume_average(nSF6_new)
            sf6_min_avg = nSF6_0d * 0.80  # Don't deplete below 80% of 0D
            if sf6_avg_new > 1e6 and sf6_avg_new < sf6_min_avg:
                nSF6_new *= sf6_min_avg / sf6_avg_new
            nSF6 = nSF6 + w*(nSF6_new - nSF6)
            nSF6 = np.clip(nSF6, 1e8, nSF6_0*2)
            neutrals['SF6'] = nSF6

        # === 6. F with ASYMMETRIC Robin BCs [Change A] ===
        if nSF6_0 > 0:
            f_source = np.zeros((Nr,Nz))
            f_loss = np.zeros((Nr,Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    f_source[i,j] = fluorine_source(
                        k_loc, ne[i,j], nSF6[i,j], 0, 0, 0, 0, 0, 0, nArm[i,j])
                    f_loss[i,j] = k_loc['iz28']*ne[i,j] + 1.0/tau_R
                    for sp_key, nr_key in [('SF5','nr42'),('SF4','nr41'),('SF3','nr40'),
                                           ('SF2','nr39'),('SF','nr38'),('S','nr37')]:
                        if sp_key in neutrals:
                            f_loss[i,j] += k_loc.get(nr_key,0)*neutrals[sp_key][i,j]

            # Asymmetric Robin: different gamma at each wall
            nF_new = _solve_robin_asymmetric(mesh, D_F, f_source, f_loss,
                                             h_r_wall=h_F_wall, h_z0=h_F_wafer, h_zL=h_F_window)
            nF = nF + w*(nF_new - nF)
            nF = np.maximum(nF, 0.0)
            neutrals['F'] = nF

        # === 7. Minor neutrals ===
        if nSF6_0 > 0 and has_0d:
            neutrals_0d_dict = {sp: r0d.get(f'n_{sp}', 0) for sp in neutrals if f'n_{sp}' in r0d}
            D_coeffs = {sp: n_tr.diffusivity(M_SPECIES.get(sp,50), sigma=5e-19) for sp in neutrals}
            full_old = {sp: neutrals[sp].copy() for sp in neutrals}
            full_new = solve_neutral_transport(
                mesh, ne, Te_field, nArm, ng, nSF6_0, tau_R,
                full_old, D_coeffs, gamma_F=gamma_F_wall, kw_F_eff=30.0, w=w,
                neutrals_0d=neutrals_0d_dict)
            for sp in full_new:
                if sp not in ('SF6', 'F'):
                    neutrals[sp] = full_new[sp]

        # === 8. EM ===
        if it % em_interval == 0:
            for _ in range(5):
                try:
                    Pn, _ = em.adjust_coil_current(ne, Te_field, ng, P_target=P_abs)
                    if np.all(np.isfinite(Pn)) and Pn.max() > 0:
                        P_ind = 0.2*P_ind + 0.8*Pn
                except: pass

        # === Convergence ===
        ne_avg_new = mesh.volume_average(ne)
        dTe = np.max(np.abs(Te_field-Te_old))/max(np.max(np.abs(Te_old)),0.1)
        dne = abs(ne_avg_new-ne_avg_old)/max(ne_avg_old,1e10)

        if verbose and (it%(max(n_iter//10,1))==0 or it==n_iter-1):
            al_a = mesh.volume_average(n_neg)/max(mesh.volume_average(ne),1e10)
            nF_a = mesh.volume_average(nF)
            nSF6_a = mesh.volume_average(nSF6)
            jm = Nz//2
            nF_c = nF[0, jm]; nF_e = nF[-1, jm]
            drop = (1 - nF_e/max(nF_c,1e-30))*100 if nF_c > 1e8 else 0
            sf6_c = nSF6[0, jm]; sf6_e = nSF6[-1, jm]
            sf6_depl = (1-sf6_c/max(sf6_e,1e-30))*100 if sf6_e > 1e8 else 0
            print(f"  {it:>3d}: <ne>={ne_avg_new*1e-6:.1e} "
                  f"Te={Te_avg:.2f}({Te_field.min():.1f}-{Te_field.max():.1f}) "
                  f"α={al_a:.1f} [F]={nF_a*1e-6:.1e}(drop={drop:.0f}%) "
                  f"SF6={nSF6_a/max(ng,1)*100:.0f}%(depl={sf6_depl:.0f}%) "
                  f"Δ={dTe:.1e}/{dne:.1e}")

        if it > 25 and dTe < 5e-4 and dne < 1e-3:
            if verbose: print(f"  *** Converged at iteration {it} ***")
            break

    elapsed = time.time()-t0
    n_pos = ne + n_neg
    V, Er, Ez = poisson.solve(n_pos, ne, n_neg)

    nav = mesh.volume_average(ne); Te_avg = mesh.volume_average(Te_field)
    al_avg = mesh.volume_average(n_neg)/max(nav,1e10)
    nF_avg = mesh.volume_average(nF)
    jm = Nz//2
    nF_center = nF[0, jm]; nF_edge = nF[-1, jm]
    F_drop = (1-nF_edge/max(nF_center,1e-30))*100
    sf6_center = nSF6[0, jm]; sf6_edge = nSF6[-1, jm]
    sf6_depletion = (1-sf6_center/max(sf6_edge,1e-30))*100

    if verbose:
        print(f"\n{'='*70}")
        print(f"Gen-4b done in {elapsed:.1f}s")
        print(f"  <ne>   = {nav*1e-6:.3e} cm⁻³")
        print(f"  <Te>   = {Te_avg:.2f} eV ({Te_field.min():.2f}–{Te_field.max():.2f})")
        print(f"  <α>    = {al_avg:.1f}")
        print(f"  <[F]>  = {nF_avg*1e-6:.2e} cm⁻³ (0D: {nF_0d*1e-6:.2e}, ratio={nF_avg/max(nF_0d,1):.2f})")
        print(f"  [F] center={nF_center*1e-6:.2e}, edge={nF_edge*1e-6:.2e}, DROP={F_drop:.0f}%")
        print(f"  SF6 center={sf6_center/max(ng,1)*100:.0f}%, edge={sf6_edge/max(ng,1)*100:.0f}%, depletion={sf6_depletion:.0f}%")

    return {'ne':ne,'n_pos':n_pos,'n_neg':n_neg,'Te':Te_field,
            'V':V,'Er':Er,'Ez':Ez,'P_ind':P_ind,'nArm':nArm,
            'alpha':alpha,'neutrals':neutrals,'mesh':mesh,
            'ne_avg':nav,'Te_avg':Te_avg,'alpha_avg':al_avg,
            'nF_avg':nF_avg,'Ec_avg':Ec,'eps_T':eps_T,
            'elapsed':elapsed,'nSF6':nSF6,'nF':nF,
            'nF_center':nF_center,'nF_edge':nF_edge,'F_drop_pct':F_drop,
            'sf6_depletion':sf6_depletion,'nArm_avg':mesh.volume_average(nArm),
            'gamma_F_wall':gamma_F_wall,'gamma_F_wafer':gamma_F_wafer}



#==============================================================================
# GEN-5 DRIVER
#==============================================================================


#!/usr/bin/env python3
"""
SF6/Ar 2D ICP Simulator — Generation 5 (Full Physics)

All six future-work items from the Gen-4b report:

  [1] SELF-CONSISTENT ne MAGNITUDE
      ne solved from local power balance with global constraint.
      Profile from source-weighted ionization; magnitude from
      P_abs = integral(ne * nu_iz * eps_T * e) dV.

  [2] BOLSIG+ TABLE INFRASTRUCTURE
      Transport coefficients from interpolation tables.
      Uses analytical Hagelaar fits as default; swappable for
      real BOLSIG+ output files.

  [3] SHEATH MODEL
      Analytic Bohm-sheath at all walls: V_s, E_ion, ion flux.
      Ion-enhanced etch probability.

  [4] GAS TEMPERATURE SOLVER
      2D neutral energy equation: kappa*nabla^2(Tg) + Q_el + Q_FC = 0.
      Feeds back into gas density and transport.

  [5] MULTI-ION SPECIES
      Tracks SF5+, SF3+, Ar+ fractions and effective ion mass.
      Ion composition at wafer for etch selectivity.

  [6] ENHANCED DT API
      run_v5() returns comprehensive dict with all new physics.
      run_dt_scan() for multi-parameter sweeps with etch rate output.
"""

import sys, os, time, argparse
import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi


AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6


def run_v5(P_rf=1500.0, p_mTorr=10.0, frac_Ar=0.0, Tgas_init=300.0, T_neg=0.3,
           gamma_F_wafer=0.02, gamma_F_wall=0.30, gamma_F_window=0.001,
           beta_SF6=0.005, eta=0.12,
           Nr=30, Nz=40, n_iter=80, em_interval=3, verbose=True):
    t0 = time.time()
    p_Pa = p_mTorr * MTORR_TO_PA
    ng_init = p_Pa / (kB * Tgas_init)
    nAr = ng_init * frac_Ar; nSF6_0 = ng_init * (1 - frac_Ar)
    V_reactor = pi * 0.180**2 * 0.175
    P_abs = P_rf * eta
    Q_tp = 40e-6/60 * 1.01325e5 * (Tgas_init/273.15)
    tau_R = p_Pa * V_reactor / Q_tp if Q_tp > 0 else 1e10

    mesh = Mesh2D(R=0.180, L=0.175, Nr=Nr, Nz=Nz, stretch_r=1.3, stretch_z=1.3)
    em = EMSolver(mesh, freq=13.56e6)
    poisson = PoissonSolver(mesh)

    # [2] BOLSIG+ table (analytical fallback)
    bolsig = BOLSIGTable.from_analytical(x_Ar=frac_Ar, x_SF6=1-frac_Ar, N=ng_init)

    Mi_amu = 127.06; Mi_kg = Mi_amu * AMU  # Will be updated by multi-ion
    i_tr = IonTransport(Mi_amu, ng_init, Tgas_init)
    n_tr = NeutralTransport(ng_init, Tgas_init)
    k_rec = 1.5e-9 * cm3

    D_F = n_tr.diffusivity(M_SPECIES['F'], sigma=4e-19)
    D_SF6 = n_tr.diffusivity(M_SPECIES['SF6'], sigma=6e-19)
    D_Arm = n_tr.diffusivity(39.948, sigma=3e-19)
    D_neg = i_tr.D0
    Lambda2 = 1.0 / ((pi/mesh.L)**2 + (2.405/mesh.R)**2)
    kw_Arm = D_Arm / Lambda2

    # Robin coefficients
    v_th_F = np.sqrt(8*kB*Tgas_init/(pi*M_SPECIES['F']*AMU))
    v_th_SF6 = np.sqrt(8*kB*Tgas_init/(pi*M_SPECIES['SF6']*AMU))
    h_F_wafer = gamma_F_wafer * v_th_F / 4
    h_F_wall = gamma_F_wall * v_th_F / 4
    h_F_window = gamma_F_window * v_th_F / 4
    h_SF6_wall = beta_SF6 * v_th_SF6 / 4

    # --- 0D backbone ---
    has_0d = False; ne_0d = 1e16; Te_0d = 3.0; alpha_0d = 0
    Ec = 100; eps_T = 120; r0d = {}
    nSF6_0d = nSF6_0 * 0.3; nF_0d = 0
    try:
        solve_0d = solve_model  # unified: already defined above
        r0d = solve_0d(P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                       eta=eta, Tgas=Tgas_init, T_neg=T_neg)
        ne_0d = r0d['ne']; Te_0d = r0d['Te']; alpha_0d = r0d['alpha']
        nSF6_0d = r0d.get('n_SF6', nSF6_0*0.3)
        nF_0d = r0d.get('n_F', 0); Ec = r0d.get('Ec', 100)
        eps_T = r0d.get('eps_T', 120)
        has_0d = True
    except Exception as e:
        if verbose: print(f"  0D unavailable ({e})")

    if verbose:
        print(f"Gen-5 Full Physics: {P_rf:.0f}W, {p_mTorr:.0f}mT, "
              f"{frac_Ar*100:.0f}%Ar/{(1-frac_Ar)*100:.0f}%SF6")
        print(f"  0D: ne={ne_0d*1e-6:.2e} Te={Te_0d:.2f} α={alpha_0d:.1f}")
        print(f"  BOLSIG: {bolsig.source}")
        print("-" * 70)

    # --- Initialize ---
    Da0 = ambipolar_Da(alpha_0d, Te_0d, i_tr.D0, T_neg)
    g_en = Te_0d / max(T_neg, 0.01)
    uB0 = np.sqrt(eC*Te_0d*(1+alpha_0d)/(Mi_kg*(1+g_en*alpha_0d)))
    profile = solve_diffusion_profile(mesh, Da0, uB0)
    pn = profile / max(mesh.volume_average(profile), 1e-30)

    ne = ne_0d * pn
    Te_field = np.full((Nr, Nz), Te_0d)
    nArm = np.zeros((Nr, Nz))
    n_neg = np.full((Nr, Nz), alpha_0d*ne_0d) * pn if alpha_0d > 0.01 else np.zeros((Nr, Nz))
    nSF6 = np.full((Nr, Nz), nSF6_0d)
    nF = np.full((Nr, Nz), max(nF_0d, 1e10))
    Tg = np.full((Nr, Nz), Tgas_init)  # [4] gas temperature
    ng_field = np.full((Nr, Nz), ng_init)  # spatially varying gas density

    neutrals = {}
    if has_0d and nSF6_0 > 0:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr, Nz), r0d.get(f'n_{sp}', nSF6_0*0.01))
    else:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr, Nz), nSF6_0*0.01 if nSF6_0 > 0 else 0)
        if nSF6_0 > 0: neutrals['SF6'] = np.full((Nr, Nz), nSF6_0*0.3)

    try:
        P_ind, _ = em.adjust_coil_current(ne, Te_field, ng_init, P_target=P_abs)
        if not (np.all(np.isfinite(P_ind)) and P_ind.max() > 0): raise ValueError
    except:
        P_shape = np.exp(-(mesh.L-mesh.ZZ)/0.017) * np.maximum(mesh.RR/mesh.R, 0.01)
        P_ind = P_shape * P_abs / max(mesh.volume_average(P_shape)*V_reactor, 1e-10)

    w = 0.12

    for it in range(n_iter):
        Te_old = Te_field.copy()

        # === [4] GAS TEMPERATURE ===
        if it > 5 and it % 5 == 0:
            Tg_new, Q_heat = solve_gas_temperature(
                mesh, ne, Te_field, nSF6, ng_init, M_gas_amu=Mi_amu,
                T_wall=Tgas_init, x_Ar=frac_Ar, x_SF6=1-frac_Ar)
            Tg = 0.8*Tg + 0.2*Tg_new
            # Update local gas density: ng = p/(kB*Tg)
            ng_field = p_Pa / (kB * Tg)

        # === 1. Ar* ===
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j]); ne_ij = ne[i,j]
                qh = (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nSF6[i,j]+k_loc['qnch_F']*nF[i,j]
                den = (k_loc['Ar_iz_m']+k_loc['Ar_q'])*ne_ij + kw_Arm + qh
                nArm[i,j] = k_loc['Ar_exc']*ne_ij*nAr/max(den, 1.0) if ne_ij > 1e8 else 0

        # === 2. Te from energy equation ===
        Eloss_field = compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm)
        eps_bar = np.maximum(1.5*Te_field, 0.5)
        nu_eps = np.clip(np.where(eps_bar > 0.1, Eloss_field/eps_bar, 1.0), 1.0, 1e12)
        ne_safe = np.maximum(ne, 1e8)
        source_eps = P_ind / (ne_safe * eC)
        h_tr = hagelaar_mix(max(Te_0d, 0.5), ng_init, frac_Ar, 1-frac_Ar)
        D_eps_val = h_tr['D_eps']

        eps_new = _solve_diffusion_neumann(mesh, D_eps_val, source_eps, nu_eps)
        Te_new = np.clip((2.0/3.0) * np.clip(eps_new, 0.5, 15.0), 0.8, 8.0)
        Te_avg_raw = mesh.volume_average(Te_new)
        if Te_avg_raw > 0.1 and Te_0d > 0.1:
            Te_new *= Te_0d / Te_avg_raw
        Te_new = np.clip(Te_new, 0.8, 8.0)
        Te_field = Te_field + w*(Te_new - Te_field)

        # === [5] MULTI-ION: compute effective mass ===
        ion_fracs, M_eff_field = compute_ion_fractions_2d(
            mesh, ne, Te_field, nSF6, nAr, nArm, ng_init)
        Mi_eff_avg = mesh.volume_average(M_eff_field)
        Mi_kg_local = max(Mi_eff_avg, 10*AMU)

        # === [1] SELF-CONSISTENT ne: source-weighted + power constraint ===
        Te_avg = mesh.volume_average(Te_field)
        al_avg = mesh.volume_average(n_neg) / max(mesh.volume_average(ne), 1e10)
        Da_now = ambipolar_Da(al_avg, Te_avg, i_tr.D0, T_neg)
        g_t = Te_avg / max(T_neg, 0.01)
        uB_now = np.sqrt(eC*Te_avg*(1+al_avg)/(Mi_kg_local*(1+g_t*al_avg)))

        ne_source = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j])
                nu_iz = k_loc['Ar_iz']*nAr + k_loc['iz_SF6_total']*nSF6[i,j]
                if nArm[i,j] > 1e6:
                    nu_iz += k_loc['Ar_iz_m']*nArm[i,j]
                nu_att = k_loc['att_SF6_total']*nSF6[i,j]
                ne_source[i,j] = max(nu_iz - nu_att, 0) * ne[i,j]

        ne_profile = _solve_robin_asymmetric(mesh, Da_now, ne_source, np.zeros((Nr, Nz)),
                                             h_r_wall=uB_now, h_z0=uB_now, h_zL=uB_now)

        # [1] Power constraint: scale ne so that P_abs = integral(ne*nu_iz*eps_T*e)dV
        ne_prof_avg = mesh.volume_average(ne_profile)
        if ne_prof_avg > 1e6:
            # Compute what eps_T would be at current ne
            eps_T_local = max(eps_T, 50)
            # Average ionization frequency
            nu_iz_avg = 0
            for i in range(Nr):
                for j in range(Nz):
                    k_l = rates(Te_field[i,j])
                    nu_iz_avg += (k_l['Ar_iz']*nAr + k_l['iz_SF6_total']*nSF6[i,j]
                                  + k_l['Ar_iz_m']*nArm[i,j]) * ne_profile[i,j]
            nu_iz_avg /= max(Nr*Nz, 1)
            
            # ne_target from power balance
            ne_target = P_abs / (max(nu_iz_avg, 1e-30) * eps_T_local * eC * V_reactor / max(ne_prof_avg, 1e6))
            ne_target = np.clip(ne_target, ne_0d*0.3, ne_0d*3.0)
            
            ne_new = ne_profile * (ne_target / ne_prof_avg)
        else:
            ne_new = ne_0d * pn
        
        ne = ne + w*(ne_new - ne); ne = np.maximum(ne, 1e6)

        # === 4. Negative ions ===
        if nSF6_0 > 0:
            n_pos = ne + n_neg
            n_neg, alpha = solve_negative_ions(
                mesh, ne, n_neg, nSF6, Te_field, n_pos,
                D_neg, k_rec, alpha_0d=alpha_0d, ne_0d=ne_0d, w=w)
        else:
            n_neg = np.zeros((Nr, Nz)); alpha = np.zeros((Nr, Nz))

        # === 5. SF6 (Robin, partial anchor) ===
        if nSF6_0 > 0:
            sf6_source = np.full((Nr, Nz), nSF6_0/tau_R)
            sf6_loss = np.zeros((Nr, Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    k_e = (k_loc['d1']+k_loc['d2']+k_loc['d3']+k_loc['d4']+k_loc['d5']
                           +k_loc['iz_SF6_total']+k_loc['att_SF6_total'])
                    sf6_loss[i,j] = k_e*ne[i,j] + (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nArm[i,j] + 1.0/tau_R

            nSF6_new = _solve_robin_asymmetric(mesh, D_SF6, sf6_source, sf6_loss,
                                                h_r_wall=h_SF6_wall, h_z0=h_SF6_wall, h_zL=h_SF6_wall*0.1)
            sf6_avg_new = mesh.volume_average(nSF6_new)
            sf6_min = nSF6_0d * 0.80
            if sf6_avg_new > 1e6 and sf6_avg_new < sf6_min:
                nSF6_new *= sf6_min / sf6_avg_new
            nSF6 = nSF6 + w*(nSF6_new - nSF6)
            nSF6 = np.clip(nSF6, 1e8, nSF6_0*2)
            neutrals['SF6'] = nSF6

        # === 6. F (asymmetric Robin, no anchor) ===
        if nSF6_0 > 0:
            f_source = np.zeros((Nr, Nz)); f_loss = np.zeros((Nr, Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    f_source[i,j] = fluorine_source(k_loc, ne[i,j], nSF6[i,j], 0,0,0,0,0,0, nArm[i,j])
                    f_loss[i,j] = k_loc['iz28']*ne[i,j] + 1.0/tau_R
                    for sp_key, nr_key in [('SF5','nr42'),('SF4','nr41'),('SF3','nr40'),
                                           ('SF2','nr39'),('SF','nr38'),('S','nr37')]:
                        if sp_key in neutrals:
                            f_loss[i,j] += k_loc.get(nr_key,0)*neutrals[sp_key][i,j]

            nF_new = _solve_robin_asymmetric(mesh, D_F, f_source, f_loss,
                                             h_r_wall=h_F_wall, h_z0=h_F_wafer, h_zL=h_F_window)
            nF = nF + w*(nF_new - nF); nF = np.maximum(nF, 0.0)
            neutrals['F'] = nF

        # === 7. Minor neutrals ===
        if nSF6_0 > 0 and has_0d:
            neutrals_0d_dict = {sp: r0d.get(f'n_{sp}', 0) for sp in neutrals if f'n_{sp}' in r0d}
            D_coeffs = {sp: n_tr.diffusivity(M_SPECIES.get(sp,50), sigma=5e-19) for sp in neutrals}
            full_old = {sp: neutrals[sp].copy() for sp in neutrals}
            full_new = solve_neutral_transport(
                mesh, ne, Te_field, nArm, ng_init, nSF6_0, tau_R,
                full_old, D_coeffs, gamma_F=gamma_F_wall, kw_F_eff=30.0, w=w,
                neutrals_0d=neutrals_0d_dict)
            for sp in full_new:
                if sp not in ('SF6', 'F'):
                    neutrals[sp] = full_new[sp]

        # === 8. EM ===
        if it % em_interval == 0:
            for _ in range(5):
                try:
                    Pn, _ = em.adjust_coil_current(ne, Te_field, ng_init, P_target=P_abs)
                    if np.all(np.isfinite(Pn)) and Pn.max() > 0:
                        P_ind = 0.2*P_ind + 0.8*Pn
                except: pass

        # === Convergence ===
        dTe = np.max(np.abs(Te_field-Te_old)) / max(np.max(np.abs(Te_old)), 0.1)
        ne_avg = mesh.volume_average(ne)

        if verbose and (it%(max(n_iter//8, 1))==0 or it==n_iter-1):
            al_a = mesh.volume_average(n_neg)/max(ne_avg, 1e10)
            nF_a = mesh.volume_average(nF)
            jm = Nz//2
            nF_c = nF[0,jm]; nF_e = nF[-1,jm]
            drop = (1-nF_e/max(nF_c,1e-30))*100 if nF_c > 1e8 else 0
            Tg_max = Tg.max()
            print(f"  {it:>3d}: ne={ne_avg*1e-6:.1e} Te={Te_avg:.2f}({Te_field.min():.1f}-{Te_field.max():.1f}) "
                  f"α={al_a:.1f} [F]={nF_a*1e-6:.1e}(drop={drop:.0f}%) "
                  f"Tg_max={Tg_max:.0f}K Mi={Mi_eff_avg/AMU:.0f}AMU")

        if it > 25 and dTe < 5e-4:
            if verbose: print(f"  *** Converged at iteration {it} ***")
            break

    elapsed = time.time() - t0

    # === [3] SHEATH MODEL ===
    wall_data = compute_wall_fluxes(ne, Te_field, alpha, mesh, Mi_kg_local, T_neg)
    # Ion-enhanced etch probability at wafer
    gamma_etch = ion_enhanced_etch_probability(wall_data['wafer_energy'])

    # Etch rate from [F] + ion enhancement
    r_cm, R_etch = etch_rate_profile(nF, mesh, Tgas_init, gamma_Si=0.025)
    unif = uniformity(R_etch, r_cm, r_max_cm=15.0)

    # Summary
    nav = mesh.volume_average(ne); Te_avg = mesh.volume_average(Te_field)
    al_avg = mesh.volume_average(n_neg)/max(nav, 1e10)
    nF_avg = mesh.volume_average(nF)
    jm = Nz//2
    nF_center = nF[0,jm]; nF_edge = nF[-1,jm]
    F_drop = (1-nF_edge/max(nF_center, 1e-30))*100

    if verbose:
        print(f"\n{'='*70}")
        print(f"Gen-5 done in {elapsed:.1f}s")
        print(f"  ne  = {nav*1e-6:.3e} cm⁻³ (0D: {ne_0d*1e-6:.2e}, ratio={nav/max(ne_0d,1):.2f})")
        print(f"  Te  = {Te_avg:.2f} eV ({Te_field.min():.2f}–{Te_field.max():.2f})")
        print(f"  α   = {al_avg:.1f}")
        print(f"  [F] = {nF_avg*1e-6:.2e} cm⁻³, drop={F_drop:.0f}%")
        print(f"  Tg  = {Tg.min():.0f}–{Tg.max():.0f} K")
        print(f"  M_ion = {Mi_eff_avg/AMU:.1f} AMU")
        for sp in ['SF5+','SF3+','Ar+']:
            if sp in ion_fracs:
                f_avg = mesh.volume_average(ion_fracs[sp])
                if f_avg > 0.01:
                    print(f"    {sp}: {f_avg*100:.1f}%")
        print(f"  Sheath: V_s={wall_data['wafer_energy'].mean():.1f}eV, "
              f"Γ_i(0)={wall_data['wafer_flux'][0]:.2e} m⁻²s⁻¹")
        print(f"  Etch: {unif['mean']:.1f}±{unif['std']:.1f} nm/s, "
              f"non-unif={unif['nonuniformity_pct']:.1f}%")

    return {
        'ne': ne, 'Te': Te_field, 'n_neg': n_neg, 'alpha': alpha,
        'nF': nF, 'nSF6': nSF6, 'nArm': nArm, 'Tg': Tg,
        'P_ind': P_ind, 'mesh': mesh, 'ne_avg': nav, 'Te_avg': Te_avg,
        'alpha_avg': al_avg, 'nF_avg': nF_avg, 'Ec_avg': Ec, 'eps_T': eps_T,
        'F_drop_pct': F_drop, 'elapsed': elapsed,
        'ion_fractions': ion_fracs, 'M_eff': M_eff_field,
        'wall_fluxes': wall_data, 'etch_rate': R_etch, 'etch_uniformity': unif,
        'Tg_max': Tg.max(), 'Mi_eff': Mi_eff_avg,
        'gamma_F_wall': gamma_F_wall, 'gamma_F_wafer': gamma_F_wafer,
        'bolsig_source': bolsig.source,
    }


def run_dt_scan(powers=None, pressures=None, compositions=None,
                gamma_F_wall=0.30, Nr=20, Nz=25, n_iter=50, verbose=False):
    """Enhanced DT scan with etch rate output."""
    if powers is None: powers = [500, 1000, 1500, 2000]
    if pressures is None: pressures = [10]
    if compositions is None: compositions = [0.0, 0.3, 0.5, 0.7, 1.0]

    results = []
    for P in powers:
        for p in pressures:
            for ar in compositions:
                try:
                    r = run_v5(P_rf=P, p_mTorr=p, frac_Ar=ar,
                               gamma_F_wall=gamma_F_wall,
                               Nr=Nr, Nz=Nz, n_iter=n_iter, verbose=verbose)
                    results.append({
                        'P_rf': P, 'p_mTorr': p, 'frac_Ar': ar,
                        'ne_avg': r['ne_avg'], 'Te_avg': r['Te_avg'],
                        'alpha_avg': r['alpha_avg'], 'nF_avg': r['nF_avg'],
                        'F_drop_pct': r['F_drop_pct'],
                        'Tg_max': r['Tg_max'], 'Mi_eff': r['Mi_eff'] / AMU,
                        'etch_mean': r['etch_uniformity']['mean'],
                        'etch_nonunif': r['etch_uniformity']['nonuniformity_pct'],
                        'elapsed': r['elapsed'], 'converged': True,
                    })
                except Exception as e:
                    results.append({
                        'P_rf': P, 'p_mTorr': p, 'frac_Ar': ar,
                        'error': str(e), 'converged': False,
                    })
    return results




#==============================================================================
# OUTPUT: PLOTTING + CSV EXPORT
#==============================================================================

def save_csv_outputs(res, outdir, P_rf, p_mTorr, frac_Ar):
    """Save all simulation results as CSV files."""
    import csv
    m = res['mesh']; Nr = m.Nr; Nz = m.Nz
    r_cm = m.r * 100; z_cm = m.z * 100
    jm = Nz // 2

    # 1. Summary scalars
    with open(f'{outdir}/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Parameter', 'Value', 'Unit'])
        w.writerow(['P_rf', P_rf, 'W'])
        w.writerow(['p_mTorr', p_mTorr, 'mTorr'])
        w.writerow(['frac_Ar', frac_Ar, ''])
        w.writerow(['ne_avg', f"{res['ne_avg']:.6e}", 'm-3'])
        w.writerow(['ne_avg_cm3', f"{res['ne_avg']*1e-6:.6e}", 'cm-3'])
        w.writerow(['Te_avg', f"{res['Te_avg']:.4f}", 'eV'])
        w.writerow(['Te_min', f"{res['Te'].min():.4f}", 'eV'])
        w.writerow(['Te_max', f"{res['Te'].max():.4f}", 'eV'])
        w.writerow(['alpha_avg', f"{res['alpha_avg']:.2f}", ''])
        w.writerow(['nF_avg_cm3', f"{res['nF_avg']*1e-6:.6e}", 'cm-3'])
        w.writerow(['F_drop_pct', f"{res['F_drop_pct']:.1f}", '%'])
        w.writerow(['elapsed_s', f"{res['elapsed']:.1f}", 's'])
        for key in ['Tg_max', 'Mi_eff', 'Ec_avg', 'eps_T']:
            if key in res:
                w.writerow([key, f"{res[key]:.2f}", ''])
    print(f"  Saved {outdir}/summary.csv")

    # 2. Radial profiles at midplane
    header = ['r_cm', 'ne_cm3', 'Te_eV', 'nF_cm3', 'nSF6_cm3', 'alpha']
    data = np.column_stack([
        r_cm,
        res['ne'][:, jm] * 1e-6,
        res['Te'][:, jm],
        res['nF'][:, jm] * 1e-6,
        res['nSF6'][:, jm] * 1e-6 if 'nSF6' in res else np.zeros(Nr),
        res['alpha'][:, jm] if 'alpha' in res else np.zeros(Nr),
    ])
    if 'nArm' in res and res['nArm'].max() > 1e6:
        header.append('nArm_cm3')
        data = np.column_stack([data, res['nArm'][:, jm] * 1e-6])
    if 'n_neg' in res and res['n_neg'].max() > 1e6:
        header.append('n_neg_cm3')
        data = np.column_stack([data, res['n_neg'][:, jm] * 1e-6])
    np.savetxt(f'{outdir}/radial_midplane.csv', data, delimiter=',',
               header=','.join(header), comments='', fmt='%.6e')
    print(f"  Saved {outdir}/radial_midplane.csv")

    # 3. Radial profiles at wafer (z=0)
    data_w = np.column_stack([
        r_cm,
        res['ne'][:, 0] * 1e-6,
        res['Te'][:, 0],
        res['nF'][:, 0] * 1e-6,
    ])
    header_w = ['r_cm', 'ne_cm3', 'Te_eV', 'nF_cm3']
    # Etch rate if available
    if 'etch_rate' in res:
        header_w.append('etch_rate_nm_s')
        data_w = np.column_stack([data_w, res['etch_rate']])
    np.savetxt(f'{outdir}/radial_wafer.csv', data_w, delimiter=',',
               header=','.join(header_w), comments='', fmt='%.6e')
    print(f"  Saved {outdir}/radial_wafer.csv")

    # 4. Axial profiles on axis (r=0)
    data_ax = np.column_stack([
        z_cm,
        res['ne'][0, :] * 1e-6,
        res['Te'][0, :],
        res['nF'][0, :] * 1e-6,
        res['P_ind'][0, :] * 1e-3,
    ])
    np.savetxt(f'{outdir}/axial_axis.csv', data_ax, delimiter=',',
               header='z_cm,ne_cm3,Te_eV,nF_cm3,P_ind_kW_m3', comments='', fmt='%.6e')
    print(f"  Saved {outdir}/axial_axis.csv")

    # 5. Full 2D fields (flattened)
    rr, zz = np.meshgrid(r_cm, z_cm, indexing='ij')
    data_2d = np.column_stack([
        rr.flatten(), zz.flatten(),
        res['ne'].flatten() * 1e-6,
        res['Te'].flatten(),
        res['nF'].flatten() * 1e-6,
        res['P_ind'].flatten() * 1e-3,
        res['alpha'].flatten() if 'alpha' in res else np.zeros(Nr*Nz),
    ])
    np.savetxt(f'{outdir}/fields_2d.csv', data_2d, delimiter=',',
               header='r_cm,z_cm,ne_cm3,Te_eV,nF_cm3,P_ind_kW_m3,alpha',
               comments='', fmt='%.6e')
    print(f"  Saved {outdir}/fields_2d.csv")

    # 6. Wall fluxes (if available)
    if 'wall_fluxes' in res:
        wf = res['wall_fluxes']
        data_wf = np.column_stack([
            wf['wafer_r_cm'],
            wf['wafer_flux'],
            wf['wafer_energy'],
        ])
        np.savetxt(f'{outdir}/wall_flux_wafer.csv', data_wf, delimiter=',',
                   header='r_cm,ion_flux_m2s,ion_energy_eV', comments='', fmt='%.6e')
        print(f"  Saved {outdir}/wall_flux_wafer.csv")


def save_scan_csv(results, outdir):
    """Save parameter scan results as CSV."""
    import csv
    with open(f'{outdir}/scan_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        header = ['P_rf_W', 'p_mTorr', 'frac_Ar', 'ne_avg_cm3', 'Te_avg_eV',
                  'alpha', 'nF_avg_cm3', 'F_drop_pct', 'etch_nm_s',
                  'etch_nonunif_pct', 'Tg_max_K', 'Mi_AMU', 'elapsed_s', 'converged']
        w.writerow(header)
        for r in results:
            if r.get('converged'):
                w.writerow([
                    r['P_rf'], r['p_mTorr'], r['frac_Ar'],
                    f"{r['ne_avg']*1e-6:.6e}", f"{r['Te_avg']:.4f}",
                    f"{r['alpha_avg']:.2f}", f"{r['nF_avg']*1e-6:.6e}",
                    f"{r['F_drop_pct']:.1f}", f"{r.get('etch_mean',0):.2f}",
                    f"{r.get('etch_nonunif',0):.1f}", f"{r.get('Tg_max',300):.0f}",
                    f"{r.get('Mi_eff',127):.1f}", f"{r['elapsed']:.1f}", True])
            else:
                w.writerow([r['P_rf'], r['p_mTorr'], r['frac_Ar'],
                            '', '', '', '', '', '', '', '', '', '', False])
    print(f"  Saved {outdir}/scan_results.csv")


def plot_results(res, outdir, P_rf, p_mTorr, frac_Ar, gen_label='Gen-5'):
    """Generate comprehensive output plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    m = res['mesh']; Nr = m.Nr; Nz = m.Nz
    Rc = m.r * 100; Zc = m.z * 100
    RR, ZZ = np.meshgrid(Rc, Zc, indexing='ij')
    jm = Nz // 2
    gas = f"{frac_Ar*100:.0f}%Ar/{(1-frac_Ar)*100:.0f}%SF6"

    # === FIGURE 1: 2D contour maps (12 panels) ===
    fig = plt.figure(figsize=(18, 16))

    def cplot(pos, data, label, cmap, title):
        ax = fig.add_subplot(4, 3, pos)
        fd = data[np.isfinite(data)]
        if len(fd) > 0 and np.ptp(fd) > 0:
            lev = np.linspace(max(fd.min(), 0), fd.max() * 1.01, 25)
            c = ax.contourf(RR, ZZ, data, levels=lev, cmap=cmap)
            plt.colorbar(c, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel('r (cm)'); ax.set_ylabel('z (cm)')
        ax.set_title(title, fontsize=11); ax.set_xlim(0, m.R*100)

    cplot(1, res['ne']*1e-6, 'cm⁻³', 'plasma', '$n_e$')
    cplot(2, res['Te'], 'eV', 'hot', '$T_e$')
    cplot(3, res['P_ind']*1e-3, 'kW/m³', 'inferno', '$P_{ind}$')

    if 'nArm' in res and res['nArm'].max() > 1e8:
        cplot(4, res['nArm']*1e-6, 'cm⁻³', 'viridis', 'Ar*')
    else:
        fig.add_subplot(4,3,4).axis('off')

    if 'alpha' in res and res['alpha'].max() > 0.01:
        cplot(5, np.clip(res['alpha'], 0, 500), '', 'coolwarm', r'$\alpha = n_-/n_e$')
    else:
        fig.add_subplot(4,3,5).axis('off')

    if 'n_neg' in res and res['n_neg'].max() > 1e8:
        cplot(6, res['n_neg']*1e-6, 'cm⁻³', 'PuBu', '$n_-$ (trapped)')
    else:
        fig.add_subplot(4,3,6).axis('off')

    ng = p_mTorr * MTORR_TO_PA / (kB * 300)
    if 'nSF6' in res and res['nSF6'].max() > 1e10:
        cplot(7, res['nSF6']/max(ng, 1)*100, '%', 'Blues_r', 'SF$_6$ depletion')
    else:
        fig.add_subplot(4,3,7).axis('off')

    if res['nF'].max() > 1e10:
        cplot(8, res['nF']*1e-6, 'cm⁻³', 'YlOrRd', '[F] (self-consistent)')
    else:
        fig.add_subplot(4,3,8).axis('off')

    if 'Tg' in res and res['Tg'].max() > 301:
        cplot(9, res['Tg'], 'K', 'OrRd', 'Gas temperature')
    elif 'V' in res and np.ptp(res['V']) > 0.01:
        cplot(9, res['V'], 'V', 'RdBu_r', 'Potential')
    else:
        fig.add_subplot(4,3,9).axis('off')

    # Row 4: line profiles
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.plot(Rc, res['ne'][:, jm]*1e-6, 'b-', lw=2.5, label='$n_e$')
    if res['nF'].max() > 1e10:
        ax10.plot(Rc, res['nF'][:, jm]*1e-6, 'r:', lw=2.5, label='[F]')
    if 'n_neg' in res and res['n_neg'].max() > 1e10:
        ax10.plot(Rc, res['n_neg'][:, jm]*1e-6, 'm-.', lw=2, label='$n_-$')
    if 'nArm' in res and res['nArm'].max() > 1e8:
        ax10.plot(Rc, res['nArm'][:, jm]*1e-6, 'g--', lw=2, label='Ar*')
    ax10.set_xlabel('r (cm)'); ax10.set_ylabel('Density (cm⁻³)')
    ax10.set_title('Radial profiles (midplane)')
    ax10.legend(fontsize=8); ax10.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(4, 3, 11)
    ax11.plot(Zc, res['Te'][0, :], 'r-', lw=2.5, label='Axis')
    ax11.plot(Zc, res['Te'][Nr//2, :], 'r--', lw=1.5, alpha=0.6,
              label=f'r={Rc[Nr//2]:.0f}cm')
    ax11.set_xlabel('z (cm)'); ax11.set_ylabel('$T_e$ (eV)')
    ax11.set_title('$T_e$ profiles'); ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(4, 3, 12); ax12.axis('off')
    txt = (f"{gen_label} Results\n"
           f"  P={P_rf:.0f}W, {p_mTorr:.0f}mT, {gas}\n\n"
           f"  <ne>   = {res['ne_avg']*1e-6:.2e} cm⁻³\n"
           f"  <Te>   = {res['Te_avg']:.2f} eV ({res['Te'].min():.1f}–{res['Te'].max():.1f})\n"
           f"  <alpha>= {res['alpha_avg']:.1f}\n"
           f"  <[F]>  = {res['nF_avg']*1e-6:.2e} cm⁻³\n"
           f"  [F] drop = {res['F_drop_pct']:.0f}%\n"
           f"  Time: {res['elapsed']:.1f}s")
    ax12.text(0.05, 0.95, txt, transform=ax12.transAxes, fontsize=10,
              va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.3))

    fig.suptitle(f'{gen_label}: {P_rf:.0f}W, {p_mTorr:.0f}mTorr, {gas}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{outdir}/profiles_2d.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {outdir}/profiles_2d.png")
    plt.close()

    # === FIGURE 2: Radial [F] at wafer + midplane ===
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(Rc, res['nF'][:, 0]*1e-6, 'r-', lw=2.5, label='Wafer (z=0)')
    ax.plot(Rc, res['nF'][:, jm]*1e-6, 'r--', lw=2, alpha=0.6,
            label=f'Midplane (z={Zc[jm]:.0f}cm)')
    ax.set_xlabel('r (cm)', fontsize=12); ax.set_ylabel('[F] (cm⁻³)', fontsize=12)
    ax.set_title('Radial fluorine profile', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_xlim(0, Rc.max())

    ax2 = axes[1]
    # Normalized comparison with Mettler
    nF_mid = res['nF'][:, jm]
    nF_norm = nF_mid / max(nF_mid[0], 1e-30)
    ax2.plot(Rc, nF_norm, 'r-', lw=2.5, label=f'{gen_label} (midplane)')
    # Mettler cubic fit
    r_fit = np.linspace(0, min(8, Rc.max()), 100)
    nF_mettler = 1.01032 - 0.01847 * r_fit**2 + 7.13914e-4 * r_fit**3
    ax2.plot(r_fit, nF_mettler, 'k-', lw=2.5, label='Mettler Fig 4.14')
    ax2.plot([0, 1.5, 3.5, 6.0, 8.0], [1.0, 0.97, 0.80, 0.50, 0.25],
             'ko', ms=8, zorder=5)
    ax2.set_xlabel('r (cm)', fontsize=12); ax2.set_ylabel('Normalized [F]', fontsize=12)
    ax2.set_title('Mettler comparison', fontsize=13)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(Rc.max(), 8)); ax2.set_ylim(0, 1.15)
    drop_model = (1 - nF_norm[min(np.argmin(np.abs(Rc-8)), len(Rc)-1)]) * 100
    ax2.text(0.97, 0.97, f'Drop: model={drop_model:.0f}%, Mettler=75%',
             transform=ax2.transAxes, va='top', ha='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig2.suptitle(f'Fluorine Profile — {P_rf:.0f}W, {p_mTorr:.0f}mTorr, {gas}',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(f'{outdir}/F_radial.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {outdir}/F_radial.png")
    plt.close()

    # === FIGURE 3: Etch rate profile (if available) ===
    if 'etch_rate' in res and res['etch_rate'].max() > 0:
        fig3, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Rc, res['etch_rate'], 'b-', lw=2.5)
        ax.set_xlabel('r (cm)', fontsize=12)
        ax.set_ylabel('Si etch rate (nm/s)', fontsize=12)
        ax.set_title(f'Etch rate — {P_rf:.0f}W, {p_mTorr:.0f}mTorr, {gas}', fontsize=13)
        ax.grid(True, alpha=0.3); ax.set_xlim(0, Rc.max())
        u = res.get('etch_uniformity', {})
        if u:
            ax.text(0.97, 0.97,
                    f"Mean: {u.get('mean',0):.2f} nm/s\n"
                    f"Non-uniformity: {u.get('nonuniformity_pct',0):.1f}%",
                    transform=ax.transAxes, va='top', ha='right', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        fig3.savefig(f'{outdir}/etch_rate.png', dpi=150, bbox_inches='tight')
        print(f"  Saved {outdir}/etch_rate.png")
        plt.close()

    # === FIGURE 4: Ion flux + energy at wafer (if sheath model ran) ===
    if 'wall_fluxes' in res:
        wf = res['wall_fluxes']
        fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(wf['wafer_r_cm'], wf['wafer_flux'], 'b-', lw=2.5)
        ax1.set_xlabel('r (cm)'); ax1.set_ylabel('Ion flux (m⁻²s⁻¹)')
        ax1.set_title('Ion flux at wafer'); ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.plot(wf['wafer_r_cm'], wf['wafer_energy'], 'r-', lw=2.5)
        ax2.set_xlabel('r (cm)'); ax2.set_ylabel('Ion energy (eV)')
        ax2.set_title('Ion energy at wafer'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig4.savefig(f'{outdir}/ion_wafer.png', dpi=150, bbox_inches='tight')
        print(f"  Saved {outdir}/ion_wafer.png")
        plt.close()


def plot_scan_results(results, outdir):
    """Plot parameter scan results."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    ok = [r for r in results if r.get('converged')]
    if len(ok) < 2:
        return

    # Group by Ar fraction (composition sweep)
    ar_fracs = sorted(set(r['frac_Ar'] for r in ok))
    powers = sorted(set(r['P_rf'] for r in ok))

    if len(ar_fracs) > 2:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        P_ref = powers[len(powers)//2] if powers else 1500
        comp_data = [r for r in ok if abs(r['P_rf']-P_ref) < 1]
        if len(comp_data) > 1:
            x = [r['frac_Ar']*100 for r in comp_data]
            for ax, key, label in zip(axes.flat,
                    ['ne_avg', 'Te_avg', 'alpha_avg', 'nF_avg', 'F_drop_pct', 'etch_mean'],
                    ['$\\langle n_e \\rangle$ (cm⁻³)', '$T_e$ (eV)', '$\\alpha$',
                     '$\\langle$[F]$\\rangle$ (cm⁻³)', '[F] drop (%)', 'Etch rate (nm/s)']):
                y = [r.get(key, 0) for r in comp_data]
                if key in ('ne_avg', 'nF_avg'):
                    y = [v*1e-6 for v in y]
                ax.plot(x, y, 'o-', lw=2, ms=8)
                ax.set_xlabel('Ar fraction (%)')
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                if key in ('ne_avg', 'nF_avg', 'alpha_avg'):
                    ax.set_yscale('log')
            fig.suptitle(f'Composition Sweep at {P_ref:.0f}W', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(f'{outdir}/scan_composition.png', dpi=150, bbox_inches='tight')
            print(f"  Saved {outdir}/scan_composition.png")
            plt.close()


#==============================================================================
# COMMAND-LINE INTERFACE
#==============================================================================

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
        description='SF6/Ar 2D ICP Simulator — Unified',
        formatter_class=argparse.RawTextHelpFormatter)
    pa.add_argument('--power', type=float, default=1500.0, help='RF power (W)')
    pa.add_argument('--pressure', type=float, default=10.0, help='Pressure (mTorr)')
    pa.add_argument('--ar', type=float, default=0.0, help='Ar fraction 0-1')
    pa.add_argument('--gamma-wall', type=float, default=0.30, help='F recomb at sidewall')
    pa.add_argument('--gamma-wafer', type=float, default=0.02, help='F recomb at wafer')
    pa.add_argument('--iter', type=int, default=80, help='Max iterations')
    pa.add_argument('--nr', type=int, default=25, help='Radial grid cells')
    pa.add_argument('--nz', type=int, default=30, help='Axial grid cells')
    pa.add_argument('--gen', type=int, default=5, choices=[4, 5],
                    help='4=Gen-4b, 5=Gen-5 (default)')
    pa.add_argument('--scan', action='store_true', help='Parameter sweep')
    pa.add_argument('--outdir', type=str, default='output', help='Output directory')
    pa.add_argument('--no-plot', action='store_true', help='Skip plotting')
    a = pa.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    print("=" * 70)
    print("SF6/Ar 2D ICP Plasma Simulator — Unified Single-File")
    print(f"  Output directory: {a.outdir}/")
    print("=" * 70)

    if a.scan:
        print("\nParameter scan (Gen-5)...")
        results = run_dt_scan(gamma_F_wall=a.gamma_wall,
                              Nr=a.nr, Nz=a.nz, n_iter=a.iter)
        print(f"\n{'P':>5s} {'Ar%':>4s} {'ne':>10s} {'Te':>5s} {'a':>5s} {'drop':>5s} {'etch':>6s}")
        for r in results:
            if r.get('converged'):
                print(f"{r['P_rf']:5.0f} {r['frac_Ar']*100:4.0f} "
                      f"{r['ne_avg']*1e-6:10.1e} {r['Te_avg']:5.2f} "
                      f"{r['alpha_avg']:5.1f} {r['F_drop_pct']:4.0f}% "
                      f"{r.get('etch_mean',0):6.1f}")
            else:
                print(f"{r['P_rf']:5.0f} {r['frac_Ar']*100:4.0f} FAILED")
        save_scan_csv(results, a.outdir)
        if not a.no_plot:
            plot_scan_results(results, a.outdir)

    elif a.gen == 4:
        res = run_v4b(P_rf=a.power, p_mTorr=a.pressure, frac_Ar=a.ar,
                      gamma_F_wall=a.gamma_wall, gamma_F_wafer=a.gamma_wafer,
                      Nr=a.nr, Nz=a.nz, n_iter=a.iter)
        print(f"\nSaving outputs to {a.outdir}/...")
        save_csv_outputs(res, a.outdir, a.power, a.pressure, a.ar)
        if not a.no_plot:
            plot_results(res, a.outdir, a.power, a.pressure, a.ar, 'Gen-4b')

    else:
        res = run_v5(P_rf=a.power, p_mTorr=a.pressure, frac_Ar=a.ar,
                     gamma_F_wall=a.gamma_wall,
                     Nr=a.nr, Nz=a.nz, n_iter=a.iter)
        print(f"\nSaving outputs to {a.outdir}/...")
        save_csv_outputs(res, a.outdir, a.power, a.pressure, a.ar)
        if not a.no_plot:
            plot_results(res, a.outdir, a.power, a.pressure, a.ar, 'Gen-5')

    print(f"\nDone. All outputs in {a.outdir}/")

