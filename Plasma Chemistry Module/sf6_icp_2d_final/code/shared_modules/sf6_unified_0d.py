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

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    print("="*65)
    print("SF6/Ar Global Model — Unified Run")
    print("="*65)

    base = dict(p_mTorr=10, Q_sccm=40, eta=0.12)
    powers = np.linspace(200, 2000, 37)

    # ── Quick validation ──────────────────────────────────────
    print("\n▶ Single point: 1500W, 10 mTorr, pure SF6")
    r = solve_model(**base, P_rf=1500, frac_Ar=0.)
    print(f"  ne={r['ne']*1e-6:.2e} Te={r['Te']:.2f} α={r['alpha']:.1f} nArm={r['nArm']:.2e} R_Penn={r['R_Penning']:.2e}")

    print("\n▶ Single point: 1500W, 10 mTorr, 50% Ar")
    r = solve_model(**base, P_rf=1500, frac_Ar=0.5)
    print(f"  ne={r['ne']*1e-6:.2e} Te={r['Te']:.2f} α={r['alpha']:.1f} nArm={r['nArm']:.2e} R_Penn={r['R_Penning']:.2e}")

    # ── Sweeps (run once, shared by all sections) ─────────────
    print("\n▶ Power sweeps at 0%, 20%, 50%, 80% Ar")
    sweeps = {}
    for fAr in [0.0, 0.2, 0.5, 0.8]:
        label = f"{int(fAr*100)}pct_Ar"
        print(f"  {int(fAr*100)}% Ar:")
        sweeps[label] = sweep_with_continuation(
            'P_rf', powers, {**base, 'frac_Ar': fAr}, verbose=False)
        print(f"    Done — {len(sweeps[label])} points")

    # Convenience aliases used in plot code below
    res_SF6  = sweeps['0pct_Ar']
    res_Ar20 = sweeps['20pct_Ar']
    res_Ar50 = sweeps['50pct_Ar']
    res_Ar80 = sweeps['80pct_Ar']

    print("\n▶ Ar fraction sweep (0–100%) at 1500W")
    res_Ar = sweep_with_continuation(
        'frac_Ar', np.linspace(0, 1.0, 21), {**base, 'P_rf': 1500}, verbose=False)
    print(f"  Done — {len(res_Ar)} points")

    print("\n▶ Alpha vs Ar% at 5, 10, 20 mTorr (1500W)")
    res_alpha = {}
    for p in [5, 10, 20]:
        b = {**base, 'p_mTorr': p, 'P_rf': 1500}
        res_alpha[p] = sweep_with_continuation(
            'frac_Ar', np.linspace(0, 0.8, 17), b, verbose=False)
        rl = res_alpha[p]
        print(f"  {p} mTorr: α(0%)={rl[0]['alpha']:.1f}  α(40%)={rl[8]['alpha']:.1f}  α(80%)={rl[-1]['alpha']:.1f}")

    # ── Summary table (console) ───────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY TABLE — key quantities at 1500 W, 10 mTorr")
    print("="*65)
    print(f"{'Quantity':<25} {'Pure SF6':>14} {'20% Ar':>14} {'50% Ar':>14} {'80% Ar':>14}")
    print("-"*83)
    idx = np.argmin(np.abs(powers - 1500))
    for qlabel, key, unit, scale in [
        ('ne', 'ne', 'cm⁻³', 1e-6), ('Te', 'Te', 'eV', 1),
        ('alpha', 'alpha', '—', 1), ('[F]', 'n_F', 'cm⁻³', 1e-6),
        ('[SF6]', 'n_SF6', 'cm⁻³', 1e-6), ('nAr (ground)', 'nAr0', 'cm⁻³', 1e-6),
        ('nAr* (meta)', 'nArm', 'cm⁻³', 1e-6), ('Ec', 'Ec', 'eV', 1),
        ('SF6 dissoc %', 'dissoc_frac', '%', 100),
    ]:
        vals = [res[idx][key]*scale for res in [res_SF6, res_Ar20, res_Ar50, res_Ar80]]
        if key in ['ne','n_F','n_SF6','nAr0','nArm']:
            print(f"{qlabel+' ('+unit+')':<25} {vals[0]:>14.2e} {vals[1]:>14.2e} {vals[2]:>14.2e} {vals[3]:>14.2e}")
        elif key == 'dissoc_frac':
            print(f"{qlabel:<25} {vals[0]:>13.0f}% {vals[1]:>13.0f}% {vals[2]:>13.0f}% {vals[3]:>13.0f}%")
        else:
            print(f"{qlabel+' ('+unit+')':<25} {vals[0]:>14.2f} {vals[1]:>14.2f} {vals[2]:>14.2f} {vals[3]:>14.2f}")

    # ═══════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════

    print("\n▶ Generating figures...")
    P_vals = [r['P_rf'] for r in res_SF6]

    # -- Fig 5: ne and Te vs power + vs Ar% (from sf6_global_model_final) --
    fig5, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0,0].plot(P_vals, [r['ne']*1e-6 for r in res_SF6], 'bs-', lw=2, ms=4)
    axes[0,0].set_ylabel('$n_e$ (cm$^{-3}$)'); axes[0,0].set_xlabel('Power (W)')
    axes[0,0].set_title('(a) $n_e$ vs power — 10 mTorr')
    axes[0,1].plot(P_vals, [r['Te'] for r in res_SF6], 'rs-', lw=2, ms=4)
    axes[0,1].set_ylabel('$T_e$ (eV)'); axes[0,1].set_xlabel('Power (W)')
    axes[0,1].set_title('(b) $T_e$ vs power'); axes[0,1].set_ylim(0, 5)
    fAr_pct = [r['frac_Ar']*100 for r in res_Ar]
    axes[1,0].plot(fAr_pct, [r['ne']*1e-6 for r in res_Ar], 'bs-', lw=2, ms=4)
    axes[1,0].set_ylabel('$n_e$ (cm$^{-3}$)'); axes[1,0].set_xlabel('Ar%')
    axes[1,0].set_title('(c) $n_e$ vs Ar%'); axes[1,0].set_yscale('log')
    axes[1,1].plot(fAr_pct, [r['Te'] for r in res_Ar], 'rs-', lw=2, ms=4)
    axes[1,1].set_ylabel('$T_e$ (eV)'); axes[1,1].set_xlabel('Ar%')
    axes[1,1].set_title('(d) $T_e$ vs Ar%'); axes[1,1].set_ylim(0, 5)
    fig5.suptitle('Figure 5 — With Penning Ionization', fontsize=13)
    plt.tight_layout()
    fig5.savefig(f'{OUTPUT_DIR}/fig5_reproduction.png', dpi=150, bbox_inches='tight')
    print("  Saved fig5_reproduction.png")

    # -- Fig 7: alpha vs Ar% at 3 pressures --
    fig7, ax = plt.subplots(figsize=(8, 6))
    for p, rl in res_alpha.items():
        ax.plot([r['frac_Ar']*100 for r in rl], [r['alpha'] for r in rl],
                'o-', lw=2, ms=5, label=f'{p} mTorr')
    ax.set_xlabel('Ar%'); ax.set_ylabel('α')
    ax.set_title('Figure 7 — Electronegativity (1500 W)')
    ax.legend()
    plt.tight_layout()
    fig7.savefig(f'{OUTPUT_DIR}/fig7_reproduction.png', dpi=150, bbox_inches='tight')
    print("  Saved fig7_reproduction.png")

    # -- Fig 8: [F] and ne vs power --
    fig8, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(P_vals, [r['n_F']*1e-6 for r in res_SF6], 'ro-', lw=2, ms=5, label='[F]')
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel('[F] (cm$^{-3}$)', color='r')
    ax2 = ax1.twinx()
    ax2.plot(P_vals, [r['ne']*1e-6 for r in res_SF6], 'bs-', lw=2, ms=5, label='$n_e$')
    ax2.set_ylabel('$n_e$ (cm$^{-3}$)', color='b')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax1.set_title('Figure 8 — [F] and $n_e$ vs power')
    plt.tight_layout()
    fig8.savefig(f'{OUTPUT_DIR}/fig8_reproduction.png', dpi=150, bbox_inches='tight')
    print("  Saved fig8_reproduction.png")

    # -- Species overview: neutral densities + alpha vs power --
    fig_sp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for sp, key in [('SF$_6$','n_SF6'),('SF$_5$','n_SF5'),('SF$_4$','n_SF4'),
                    ('SF$_3$','n_SF3'),('F','n_F'),('F$_2$','n_F2'),('S','n_S')]:
        vals = [r[key]*1e-6 for r in res_SF6]
        if max(vals) > 0:
            ax1.plot(P_vals, vals, 'o-', lw=1.5, ms=4, label=sp)
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel('Density (cm$^{-3}$)')
    ax1.set_title('Neutral densities'); ax1.legend(ncol=2); ax1.set_yscale('log')
    ax2.plot(P_vals, [r['alpha'] for r in res_SF6], 'go-', lw=2, ms=5)
    ax2.set_xlabel('Power (W)'); ax2.set_ylabel('α'); ax2.set_title('α vs power')
    plt.tight_layout()
    fig_sp.savefig(f'{OUTPUT_DIR}/species_overview.png', dpi=150, bbox_inches='tight')
    print("  Saved species_overview.png")

    # -- Argon density vs power (extended analysis) --
    fig_ar, ax = plt.subplots(figsize=(9, 6))
    for label, res, color, ls in [
        ('Ar ground (20% Ar)',      res_Ar20, '#2ca02c',  '-'),
        ('Ar* metastable (20% Ar)', res_Ar20, '#98df8a',  '--'),
        ('Ar ground (50% Ar)',      res_Ar50, C_Ar,       '-'),
        ('Ar* metastable (50% Ar)', res_Ar50, C_Arm,      '--'),
        ('Ar ground (80% Ar)',      res_Ar80, '#006400',  '-'),
        ('Ar* metastable (80% Ar)', res_Ar80, '#b2df8a',  '--'),
    ]:
        P = [r['P_rf'] for r in res]
        n = [r['nArm']*1e-6 if 'metastable' in label else r['nAr0']*1e-6 for r in res]
        ax.plot(P, n, ls, color=color, label=label)
    ax.set_xlabel('Coupled power (W)'); ax.set_ylabel('Density (cm$^{-3}$)')
    ax.set_title('Argon species density vs power — 10 mTorr')
    ax.set_yscale('log'); ax.set_ylim(1e8, 5e15); ax.set_xlim(0, 2100)
    ax.legend(loc='center right', fontsize=8.5)
    plt.tight_layout()
    fig_ar.savefig(f'{OUTPUT_DIR}/argon_density_vs_power.png', dpi=200, bbox_inches='tight')
    print("  Saved argon_density_vs_power.png")

    # -- Alpha vs power (extended analysis) --
    fig_alp, ax = plt.subplots(figsize=(9, 6))
    for label, res, color, marker in [
        ('Pure SF$_6$ (0% Ar)', res_SF6,  C_SF6, 'o'),
        ('20% Ar',              res_Ar20, C_SF5, 's'),
        ('50% Ar',              res_Ar50, C_F2,  '^'),
        ('80% Ar',              res_Ar80, C_ne,  'D'),
    ]:
        P = [r['P_rf'] for r in res]
        ax.plot(P, [r['alpha'] for r in res], '-', color=color,
                marker=marker, ms=4, markevery=3, label=label)
    ax.set_xlabel('Coupled power (W)'); ax.set_ylabel(r'$\alpha = n_{-}/n_e$')
    ax.set_title(r'Electronegativity $\alpha$ vs power — 10 mTorr')
    ax.set_yscale('log'); ax.set_ylim(0.05, 500); ax.set_xlim(0, 2100)
    ax.legend(); ax.axhline(y=1, color='gray', ls=':', lw=1)
    ax.text(2050, 1.15, r'$\alpha=1$', fontsize=9, color='gray', ha='right')
    plt.tight_layout()
    fig_alp.savefig(f'{OUTPUT_DIR}/alpha_vs_power.png', dpi=200, bbox_inches='tight')
    print("  Saved alpha_vs_power.png")

    # -- Te-based species diagnostics --
    fig_te, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    Te_sf6 = [r['Te'] for r in res_SF6]
    for label, key, color in [
        (r'SF$_6$','n_SF6',C_SF6),(r'SF$_5$','n_SF5',C_SF5),(r'SF$_4$','n_SF4','#fdae61'),
        (r'SF$_3$','n_SF3','#abd9e9'),('F','n_F',C_F),(r'F$_2$','n_F2',C_F2),('S','n_S',C_S),
    ]:
        vals = [r[key]*1e-6 for r in res_SF6]
        if max(vals) > 1e5:
            ax1.plot(Te_sf6, vals, '-', color=color, lw=2, label=label)
    ax1.set_xlabel(r'Electron temperature $T_e$ (eV)'); ax1.set_ylabel('Density (cm$^{-3}$)')
    ax1.set_title('SF$_6$-derived species vs $T_e$ — pure SF$_6$')
    ax1.set_yscale('log'); ax1.set_ylim(1e8, 5e14); ax1.legend(ncol=2); ax1.invert_xaxis()
    ax1.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax1.transAxes,
             fontsize=9, color='gray')

    Te_Ar = [r['Te'] for r in res_Ar50]
    ax2.plot(Te_Ar, [r['nAr0']*1e-6 for r in res_Ar50],  '-',  color=C_Ar,  lw=2.5, label='Ar (ground)')
    ax2.plot(Te_Ar, [r['nArm']*1e-6 for r in res_Ar50],  '--', color=C_Arm, lw=2.5, label=r'Ar$^*$ (metastable)')
    ax2.plot(Te_Ar, [r['ne']*1e-6 for r in res_Ar50],    '-',  color=C_ne,  lw=2,   label=r'$n_e$')
    ax2.plot(Te_Ar, [r['n_SF6']*1e-6 for r in res_Ar50], '-',  color=C_SF6, lw=1.5, alpha=0.7, label=r'SF$_6$ (mix)')
    ax2.plot(Te_Ar, [r['n_F']*1e-6 for r in res_Ar50],   '-',  color=C_F,   lw=1.5, alpha=0.7, label='F (mix)')
    ax2.set_xlabel(r'Electron temperature $T_e$ (eV)'); ax2.set_ylabel('Density (cm$^{-3}$)')
    ax2.set_title(r'Ar and mixed species vs $T_e$ — 50% Ar')
    ax2.set_yscale('log'); ax2.set_ylim(1e8, 5e14); ax2.legend(ncol=2, fontsize=8.5); ax2.invert_xaxis()
    ax2.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax2.transAxes,
             fontsize=9, color='gray')
    plt.tight_layout()
    fig_te.savefig(f'{OUTPUT_DIR}/Te_density_diagnostics.png', dpi=200, bbox_inches='tight')
    print("  Saved Te_density_diagnostics.png")

    # -- ne and alpha vs Te --
    fig_net, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for label, res, color in [
        ('Pure SF$_6$', res_SF6, C_SF6), ('20% Ar', res_Ar20, C_SF5),
        ('50% Ar', res_Ar50, C_F2),      ('80% Ar', res_Ar80, C_ne),
    ]:
        Te = [r['Te'] for r in res]
        ax1.plot(Te, [r['ne']*1e-6 for r in res],   'o-', color=color, ms=3, markevery=3, label=label)
        ax2.plot(Te, [r['alpha'] for r in res],      'o-', color=color, ms=3, markevery=3, label=label)
    ax1.set_xlabel(r'$T_e$ (eV)'); ax1.set_ylabel(r'$n_e$ (cm$^{-3}$)')
    ax1.set_title(r'Electron density vs $T_e$'); ax1.set_yscale('log'); ax1.legend(); ax1.invert_xaxis()
    ax1.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax1.transAxes, fontsize=9, color='gray')
    ax2.set_xlabel(r'$T_e$ (eV)'); ax2.set_ylabel(r'$\alpha = n_-/n_e$')
    ax2.set_title(r'Electronegativity vs $T_e$'); ax2.set_yscale('log')
    ax2.set_ylim(0.05, 500); ax2.legend(); ax2.axhline(y=1, color='gray', ls=':', lw=1); ax2.invert_xaxis()
    ax2.text(0.05, 0.05, r'$\longleftarrow$ increasing power', transform=ax2.transAxes, fontsize=9, color='gray')
    plt.tight_layout()
    fig_net.savefig(f'{OUTPUT_DIR}/ne_alpha_vs_Te.png', dpi=200, bbox_inches='tight')
    print("  Saved ne_alpha_vs_Te.png")

    # -- SF6 vs Ar species vs power --
    fig_cmp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for label, key, color, ls in [
        (r'SF$_6$','n_SF6',C_SF6,'-'),(r'SF$_5$','n_SF5',C_SF5,'-'),
        (r'SF$_4$','n_SF4','#fdae61','-'),(r'SF$_3$','n_SF3','#abd9e9','--'),
        ('F','n_F',C_F,'-'),(r'F$_2$','n_F2',C_F2,'-'),
        ('SF$_2$','n_SF2','#bcbddc','--'),('S','n_S',C_S,':'),
    ]:
        vals = [r[key]*1e-6 for r in res_SF6]
        if max(vals) > 1e6:
            ax1.plot(P_vals, vals, ls, color=color, lw=2 if ls=='-' else 1.5, label=label)
    ax1.set_xlabel('Coupled power (W)'); ax1.set_ylabel('Density (cm$^{-3}$)')
    ax1.set_title('SF$_6$-derived neutrals — pure SF$_6$')
    ax1.set_yscale('log'); ax1.set_ylim(1e7, 5e14); ax1.set_xlim(0, 2100); ax1.legend(ncol=2, fontsize=9)

    P50 = [r['P_rf'] for r in res_Ar50]
    ax2.plot(P50, [r['nAr0']*1e-6 for r in res_Ar50],  '-',  color=C_Ar,  lw=2.5, label='Ar (ground)')
    ax2.plot(P50, [r['nArm']*1e-6 for r in res_Ar50],  '--', color=C_Arm, lw=2.5, label=r'Ar$^*$ (metastable)')
    ax2.plot(P50, [r['ne']*1e-6 for r in res_Ar50],    '-',  color=C_ne,  lw=2,   label=r'$n_e$')
    ax2.plot(P50, [r['n_SF6']*1e-6 for r in res_Ar50], '-',  color=C_SF6, lw=1.5, label=r'SF$_6$')
    ax2.plot(P50, [r['n_F']*1e-6 for r in res_Ar50],   '-',  color=C_F,   lw=1.5, label='F')
    ax2.plot(P50, [r['n_SF5']*1e-6 for r in res_Ar50], '--', color=C_SF5, lw=1.5, label=r'SF$_5$')
    ax2.set_xlabel('Coupled power (W)'); ax2.set_ylabel('Density (cm$^{-3}$)')
    ax2.set_title('All species — 50% Ar / 50% SF$_6$')
    ax2.set_yscale('log'); ax2.set_ylim(1e7, 5e14); ax2.set_xlim(0, 2100); ax2.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    fig_cmp.savefig(f'{OUTPUT_DIR}/SF6_vs_Ar_species_vs_power.png', dpi=200, bbox_inches='tight')
    print("  Saved SF6_vs_Ar_species_vs_power.png")

    # ═══════════════════════════════════════════════════════════
    # CSV EXPORTS
    # ═══════════════════════════════════════════════════════════

    print("\n▶ Writing CSV files...")

    # 1. Power sweep — one file per Ar fraction
    power_headers = [
        'Power_W','Te_eV','ne_cm3','alpha',
        'nF_cm3','nF2_cm3','nSF6_cm3','nSF5_cm3','nSF4_cm3',
        'nSF3_cm3','nSF2_cm3','nSF_cm3','nS_cm3',
        'nAr_ground_cm3','nAr_metastable_cm3',
        'Ec_eV','eps_T_eV','SF6_dissoc_pct','R_Penning_m3s','converged',
    ]
    for label, res in sweeps.items():
        rows = []
        for r in res:
            rows.append([
                f"{r['P_rf']:.1f}",    f"{r['Te']:.4f}",       f"{r['ne']*1e-6:.6e}",
                f"{r['alpha']:.4f}",   f"{r['n_F']*1e-6:.6e}", f"{r['n_F2']*1e-6:.6e}",
                f"{r['n_SF6']*1e-6:.6e}", f"{r['n_SF5']*1e-6:.6e}", f"{r['n_SF4']*1e-6:.6e}",
                f"{r['n_SF3']*1e-6:.6e}", f"{r['n_SF2']*1e-6:.6e}", f"{r['n_SF']*1e-6:.6e}",
                f"{r['n_S']*1e-6:.6e}",   f"{r['nAr0']*1e-6:.6e}", f"{r['nArm']*1e-6:.6e}",
                f"{r['Ec']:.2f}",      f"{r['eps_T']:.2f}",
                f"{r['dissoc_frac']*100:.2f}", f"{r['R_Penning']:.4e}", str(r['converged']),
            ])
        write_csv(f"power_sweep_{label}.csv", power_headers, rows)

    # 2. Ar fraction sweep at 1500W
    ar_headers = [
        'Ar_fraction','Ar_percent','Power_W','Te_eV','ne_cm3','alpha',
        'nF_cm3','nSF6_cm3','nAr_ground_cm3','nAr_metastable_cm3',
        'Ec_eV','SF6_dissoc_pct','R_Penning_m3s',
    ]
    rows = []
    for r in res_Ar:
        rows.append([
            f"{r['frac_Ar']:.4f}", f"{r['frac_Ar']*100:.1f}", "1500.0",
            f"{r['Te']:.4f}",      f"{r['ne']*1e-6:.6e}",     f"{r['alpha']:.4f}",
            f"{r['n_F']*1e-6:.6e}", f"{r['n_SF6']*1e-6:.6e}",
            f"{r['nAr0']*1e-6:.6e}", f"{r['nArm']*1e-6:.6e}",
            f"{r['Ec']:.2f}", f"{r['dissoc_frac']*100:.2f}", f"{r['R_Penning']:.4e}",
        ])
    write_csv("ar_fraction_sweep_1500W.csv", ar_headers, rows)

    # 3. Alpha vs Ar at 3 pressures
    for p, res in res_alpha.items():
        headers = ['Ar_fraction','Ar_percent','alpha','Te_eV','ne_cm3',
                   'nSF6_cm3','nF_cm3','nAr_metastable_cm3']
        rows = []
        for r in res:
            rows.append([
                f"{r['frac_Ar']:.4f}", f"{r['frac_Ar']*100:.1f}",
                f"{r['alpha']:.4f}",   f"{r['Te']:.4f}",
                f"{r['ne']*1e-6:.6e}", f"{r['n_SF6']*1e-6:.6e}",
                f"{r['n_F']*1e-6:.6e}", f"{r['nArm']*1e-6:.6e}",
            ])
        write_csv(f"alpha_vs_Ar_{p}mTorr.csv", headers, rows)

    # 4. Summary table at 1500W
    sum_headers = ['Quantity','Units','Pure_SF6','20pct_Ar','50pct_Ar','80pct_Ar']
    rows = []
    for qlabel, key, unit, scale in [
        ('ne','ne','cm-3',1e-6),('Te','Te','eV',1),('alpha','alpha','-',1),
        ('nF','n_F','cm-3',1e-6),('nF2','n_F2','cm-3',1e-6),
        ('nSF6','n_SF6','cm-3',1e-6),('nSF5','n_SF5','cm-3',1e-6),
        ('nSF4','n_SF4','cm-3',1e-6),('nSF3','n_SF3','cm-3',1e-6),
        ('nAr_ground','nAr0','cm-3',1e-6),('nAr_metastable','nArm','cm-3',1e-6),
        ('Ec','Ec','eV',1),('eps_T','eps_T','eV',1),
        ('SF6_dissociation','dissoc_frac','%',100),
    ]:
        vals = []
        for sl in ['0pct_Ar','20pct_Ar','50pct_Ar','80pct_Ar']:
            v = sweeps[sl][idx][key] * scale
            vals.append(f"{v:.6e}" if abs(v) > 100 or abs(v) < 0.01 else f"{v:.4f}")
        rows.append([qlabel, unit] + vals)
    write_csv("summary_at_1500W_10mTorr.csv", sum_headers, rows)

    print(f"\n✓ All figures saved to {OUTPUT_DIR}/")
    print(f"✓ All CSV files saved to {CSV_DIR}/")
