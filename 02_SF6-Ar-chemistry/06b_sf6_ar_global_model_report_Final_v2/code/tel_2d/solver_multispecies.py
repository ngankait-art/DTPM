"""
Multi-Species 2D TEL Solver
============================
Extends the 2-species (F, SF6) solver to the full 9-species
Lallement chemistry on the masked TEL domain.

Species: SF6, SF5, SF4, SF3, SF2, SF, S, F, F2
Electrons: prescribed Bessel-cosine eigenmode (same as 2-species)
Ar*: local steady-state balance
"""

import numpy as np
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.constants import k as kB, e as eC, pi, atomic_mass as AMU
from scipy.special import j0

from mesh import Mesh2D as CylindricalMesh
from geometry import build_geometry_mask
from geometry import (BC_INTERIOR, BC_AXIS, BC_QUARTZ, BC_WINDOW,
                      BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER, BC_INACTIVE)
from sf6_chemistry import (SPECIES, compute_diffusion_coefficients, 
                           compute_thermal_speeds, compute_source_sink,
                           compute_rates, WALL_GAMMA, M)


class TELMultiSpeciesSolver:
    """9-species 2D neutral transport solver on masked TEL geometry."""
    
    def __init__(self, Nr=50, Nz=80, P_rf=700, p_mTorr=10, frac_Ar=0.0,
                 Tgas=313, Q_sccm=100, eta=0.43, f_rf=2e6,
                 beta_r=1.2, beta_z=1.0):
        
        self.P_rf = P_rf
        self.p_mTorr = p_mTorr
        self.frac_Ar = frac_Ar
        self.Tgas = Tgas
        self.Q_sccm = Q_sccm
        self.eta = eta
        self.f_rf = f_rf
        
        # Geometry (TEL reactor)
        self.R_icp = 0.038
        self.R_proc = 0.105
        self.L_icp = 0.1815
        self.L_proc = 0.050
        self.L_apt = 0.002
        self.z_apt_bot = self.L_proc
        self.z_apt_top = self.L_proc + self.L_apt
        self.z_top = self.L_proc + self.L_apt + self.L_icp
        
        # Mesh
        self.mesh = CylindricalMesh(self.R_proc, self.z_top, Nr, Nz, beta_r, beta_z)
        m = self.mesh
        
        # Geometry mask and boundaries
        self.inside, self.bc_type = build_geometry_mask(
            m, self.R_icp, self.R_proc, self.z_apt_bot, self.z_apt_top, self.z_top)
        self.n_active = int(np.sum(self.inside))
        
        # Gas properties
        p_Pa = p_mTorr * 0.1333
        self.ng = p_Pa / (kB * Tgas)
        self.ng_cm3 = self.ng * 1e-6
        
        # Diffusion coefficients
        self.D = compute_diffusion_coefficients(p_mTorr, Tgas)
        self.v_th = compute_thermal_speeds(Tgas)
        
        # Residence time
        V_total = np.sum(m.vol[self.inside])
        self.tau_R = p_Pa * V_total / (Q_sccm * 1.01325e5 / 60 * 1e-6) if Q_sccm > 0 else 1.0
        
        # Feed density
        self.nSF6_feed = self.ng * (1.0 - frac_Ar)
        
        # Flat index mapping
        self._build_index_map()
        
        # Material map for gamma
        self._build_material_map()
    
    def _build_index_map(self):
        m = self.mesh
        self.flat_idx = -np.ones((m.Nr, m.Nz), dtype=int)
        k = 0
        for i in range(m.Nr):
            for j in range(m.Nz):
                if self.inside[i, j]:
                    self.flat_idx[i, j] = k
                    k += 1
    
    def _build_material_map(self):
        """Map each boundary cell to its wall material string."""
        m = self.mesh
        self.material_map = {}
        for i in range(m.Nr):
            for j in range(m.Nz):
                bc = self.bc_type[i, j]
                if bc == BC_QUARTZ:
                    self.material_map[(i,j)] = 'quartz'
                elif bc in (BC_AL_SIDE, BC_AL_TOP, BC_SHOULDER):
                    self.material_map[(i,j)] = 'aluminium'
                elif bc == BC_WAFER:
                    self.material_map[(i,j)] = 'silicon'
                elif bc == BC_WINDOW:
                    self.material_map[(i,j)] = 'window'
    
    def _compute_ne_Te(self):
        """Prescribed ne and Te profiles (same as 2-species solver)."""
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        
        # Te from 0D particle balance (simplified)
        Te0 = 2.7  # eV, typical for SF6 at 10 mTorr
        Te = np.full((Nr, Nz), Te0 * 0.90)  # processing region default
        
        for i in range(Nr):
            r = m.rc[i]
            for j in range(Nz):
                z = m.zc[j]
                if self.inside[i,j] and z >= self.z_apt_top and r <= self.R_icp:
                    zl = (z - self.z_apt_top) / self.L_icp
                    Te[i,j] = Te0 * (0.92 + 0.16 * np.exp(-((1-zl)/0.3)**2) * 
                                     np.exp(-(r/(0.8*self.R_icp))**2))
        
        # ne from power balance
        P_abs = self.eta * self.P_rf
        eps_T = (280 + 7.2 * Te0) * eC  # J per ion pair
        k_iz = 1.2e-7 * np.exp(-18.1/Te0) * 1e-6  # m³/s
        V_icp = pi * self.R_icp**2 * self.L_icp
        ne_avg = P_abs / (k_iz * self.ng * eps_T * V_icp + 1e-30)
        ne_avg = min(ne_avg, 1e18)
        
        # Bessel-cosine eigenmode in ICP
        ne = np.zeros((Nr, Nz))
        for i in range(Nr):
            r = m.rc[i]
            for j in range(Nz):
                z = m.zc[j]
                if self.inside[i,j]:
                    if z >= self.z_apt_top and r <= self.R_icp:
                        ne[i,j] = ne_avg * j0(2.405*r/self.R_icp) * \
                                  max(np.cos(pi*(z-self.z_apt_top)/(2*self.L_icp)), 0.01)
                    else:
                        ne[i,j] = ne_avg * 0.005
        
        return ne, Te
    
    def _build_diffusion_matrix(self, D_val, species_name):
        """Build sparse diffusion matrix for one species with Robin BCs."""
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        N = self.n_active
        A = lil_matrix((N, N))
        
        v_th_sp = self.v_th[species_name]
        
        for i in range(Nr):
            for j in range(Nz):
                if not self.inside[i,j]:
                    continue
                k = self.flat_idx[i,j]
                ri = m.rc[i]; dri = m.dr[i]; dzj = m.dz[j]
                
                diag = 0.0
                
                # Radial fluxes
                if i == 0:
                    # Axis: L'Hopital
                    if i+1 < Nr and self.inside[i+1,j]:
                        drc = m.rc[i+1] - ri
                        coeff = 2.0 * D_val / (drc**2 * dri)
                        A[k, self.flat_idx[i+1,j]] += coeff
                        diag -= coeff
                else:
                    # Right face
                    r_face_r = 0.5*(m.rc[min(i+1,Nr-1)] + ri) if i+1 < Nr else ri + 0.5*dri
                    if i+1 < Nr and self.inside[i+1,j]:
                        drc = m.rc[i+1] - ri
                        coeff = D_val * r_face_r / (ri * dri * drc)
                        A[k, self.flat_idx[i+1,j]] += coeff
                        diag -= coeff
                    elif i+1 >= Nr or not self.inside[i+1,j]:
                        # Wall BC
                        mat = self.material_map.get((i,j), 'aluminium')
                        gamma = WALL_GAMMA.get(mat, {}).get(species_name, 0.001)
                        if gamma > 0:
                            h = gamma * v_th_sp / 4.0
                            dr_w = 0.5 * dri
                            wall_coeff = D_val * h / (D_val + h * dr_w)
                            diag -= wall_coeff * r_face_r / (ri * dri)
                    
                    # Left face
                    r_face_l = 0.5*(m.rc[max(i-1,0)] + ri) if i > 0 else 0.5*ri
                    if i-1 >= 0 and self.inside[i-1,j]:
                        drc = ri - m.rc[i-1]
                        coeff = D_val * r_face_l / (ri * dri * drc)
                        A[k, self.flat_idx[i-1,j]] += coeff
                        diag -= coeff
                
                # Axial fluxes
                for dj, sign in [(1, 'top'), (-1, 'bot')]:
                    jn = j + dj
                    if 0 <= jn < Nz and self.inside[i,jn]:
                        dzc = abs(m.zc[jn] - m.zc[j])
                        coeff = D_val / (dzj * dzc)
                        A[k, self.flat_idx[i,jn]] += coeff
                        diag -= coeff
                    elif jn < 0 or jn >= Nz or not self.inside[i,jn]:
                        mat = self.material_map.get((i,j), 'aluminium')
                        gamma = WALL_GAMMA.get(mat, {}).get(species_name, 0.001)
                        if gamma > 0:
                            h = gamma * v_th_sp / 4.0
                            dz_w = 0.5 * dzj
                            wall_coeff = D_val * h / (D_val + h * dz_w)
                            diag -= wall_coeff / dzj
                
                A[k, k] = diag
        
        return A.tocsr()
    
    def solve(self, n_iter=60, w=0.08, verbose=True):
        """Solve 9-species system with linearised implicit chemistry."""
        t0 = time.time()
        m = self.mesh; Nr, Nz = m.Nr, m.Nz; N = self.n_active
        
        if verbose:
            print(f"Multi-species solver: {Nr}x{Nz} mesh, {N} active, 9 species")
        
        ne, Te = self._compute_ne_Te()
        Te_avg = np.mean(Te[self.inside & (Te > 0)]) if np.any(self.inside & (Te > 0)) else 2.5
        k = compute_rates(Te_avg, self.ng_cm3, self.frac_Ar)
        
        # Initialise species (m^-3)
        fields = {}
        for sp, frac in [('SF6',0.5),('F',0.1),('SF5',0.02),('SF4',0.01),
                         ('SF3',0.005),('SF2',0.001),('SF',0.0005),('S',0.0001),('F2',0.01)]:
            fields[sp] = np.where(self.inside, self.nSF6_feed * frac, 0.0)
        
        if verbose: print("  Building diffusion matrices...")
        A_mats = {}
        for sp in SPECIES:
            A_mats[sp] = self._build_diffusion_matrix(self.D[sp], sp)
        
        if verbose: print("  Iterating (linearised implicit)...")
        from scipy.sparse import diags as sp_diags
        
        for it in range(n_iter):
            f = fields
            tau = self.tau_R
            nAr0 = self.frac_Ar * self.ng
            nArm = np.zeros((Nr, Nz))
            if self.frac_Ar > 0:
                Rq = (k['Penn_SF6']*f['SF6'] + k['qnch_SF6']*f['SF6'] +
                      k['qnch_SFx']*(f['SF5']+f['SF4']+f['SF3']+f['SF2']+f['SF']) +
                      k['qnch_F2']*f['F2'] + k['qnch_F']*f['F'])
                nArm = k['Ar_exc']*ne*nAr0 / ((k['Ar_iz_m']+k['Ar_q'])*ne + Rq + 1e-30)
            
            # Solve species in order: SF6 first, then fragments, F last
            solve_order = ['SF6','SF5','SF4','SF3','SF2','SF','S','F2','F']
            
            for sp in solve_order:
                src = np.zeros((Nr, Nz))
                loss_rate = np.zeros((Nr, Nz))
                
                if sp == 'SF6':
                    src = (self.nSF6_feed/tau + 
                           np.clip(k['nr42']*f['SF5']*f['F'], 0, 1e28) + 
                           np.clip(k['nr45']*f['SF5']**2, 0, 1e28))
                    loss_rate = k['k_e_SF6']*ne + (k['Penn_SF6']+k['qnch_SF6'])*nArm + 1/tau
                elif sp == 'SF5':
                    src = (k['d1']*ne*f['SF6'] + np.clip(k['nr41']*f['SF4']*f['F'],0,1e28) + 
                           k['qnch_SF6']*nArm*f['SF6'])
                    loss_rate = ((k['d7']+k['iz25']+k['iz26'])*ne + k['nr42']*np.clip(f['F'],0,1e22) + 
                                 k['qnch_SFx']*nArm + 1/tau)
                elif sp == 'SF4':
                    src = (k['d2']*ne*f['SF6'] + k['d7']*ne*f['SF5'] + 
                           np.clip(k['nr45']*f['SF5']**2,0,1e28) + np.clip(k['nr40']*f['SF3']*f['F'],0,1e28))
                    loss_rate = k['d8']*ne + k['nr41']*np.clip(f['F'],0,1e22) + k['qnch_SFx']*nArm + 1/tau
                elif sp == 'SF3':
                    src = (k['d3']*ne*f['SF6'] + k['d8']*ne*f['SF4'] + 
                           np.clip(k['nr39']*f['SF2']*f['F'],0,1e28))
                    loss_rate = (k['d9']+k['iz27'])*ne + k['nr40']*np.clip(f['F'],0,1e22) + k['qnch_SFx']*nArm + 1/tau
                elif sp == 'SF2':
                    src = k['d4']*ne*f['SF6'] + k['d9']*ne*f['SF3'] + np.clip(k['nr38']*f['SF']*f['F'],0,1e28)
                    loss_rate = k['d10']*ne + k['nr39']*np.clip(f['F'],0,1e22) + 1/tau
                elif sp == 'SF':
                    src = k['d5']*ne*f['SF6'] + k['d10']*ne*f['SF2'] + np.clip(k['nr37']*f['S']*f['F'],0,1e28)
                    loss_rate = k['d11']*ne + k['nr38']*np.clip(f['F'],0,1e22) + 1/tau
                elif sp == 'S':
                    src = k['d11']*ne*f['SF']
                    loss_rate = k['iz29']*ne + k['nr37']*np.clip(f['F'],0,1e22) + 1/tau
                elif sp == 'F2':
                    src = ne*f['SF6']*(k['d4']+k['d5']+k['iz21']+k['iz22']+k['iz23'])
                    loss_rate = k['d6']*ne + k['qnch_F2']*nArm + 1/tau
                elif sp == 'F':
                    src = (ne*f['SF6']*(k['d1']+2*k['d2']+3*k['d3']+2*k['d4']+3*k['d5']
                           +k['iz18']+2*k['iz19']+3*k['iz20']+2*k['iz21']+3*k['iz22']+4*k['iz23']+k['at31'])
                           + ne*(f['SF5']*(k['d7']+k['iz26'])+f['SF4']*k['d8']+f['SF3']*k['d9']
                                 +f['SF2']*k['d10']+f['SF']*k['d11'])
                           + k['Penn_SF6']*nArm*f['SF6'] + k['qnch_SF6']*nArm*f['SF6']
                           + 2*k['qnch_F2']*nArm*f['F2'] + 2*k['d6']*ne*f['F2'])
                    loss_rate = (1/tau + k['iz28']*ne + k['nr42']*np.clip(f['SF5'],0,1e22)
                                 + k['nr41']*np.clip(f['SF4'],0,1e22) + k['nr40']*np.clip(f['SF3'],0,1e22)
                                 + k['nr39']*np.clip(f['SF2'],0,1e22) + k['nr38']*np.clip(f['SF'],0,1e22)
                                 + k['nr37']*np.clip(f['S'],0,1e22))
                
                src = np.clip(src, 0, 1e30)
                loss_rate = np.clip(loss_rate, 1e-10, 1e20)
                
                rhs = np.zeros(N); loss_diag = np.zeros(N)
                for i in range(Nr):
                    for j in range(Nz):
                        if self.inside[i,j]:
                            idx = self.flat_idx[i,j]
                            rhs[idx] = -src[i,j]
                            loss_diag[idx] = -loss_rate[i,j]
                
                A_total = A_mats[sp] + sp_diags(loss_diag, 0, shape=(N,N))
                try:
                    x = spsolve(A_total, rhs)
                    x = np.maximum(x, 0)
                    new_field = np.zeros((Nr, Nz))
                    for i in range(Nr):
                        for j in range(Nz):
                            if self.inside[i,j]:
                                new_field[i,j] = x[self.flat_idx[i,j]]
                    fields[sp] = (1-w)*fields[sp] + w*new_field
                    fields[sp] = np.clip(fields[sp], 0, self.ng * 10)
                except: pass
            
            if it % 10 == 0 or it == n_iter - 1:
                nF = fields['F']; Fw = nF[:,0]
                Fc = max(Fw[0], 1e-10)
                ie = np.argmin(np.abs(m.rc - self.R_proc))
                Fe = Fw[ie]
                drop = (1 - Fe/Fc) * 100
                sf6d = (1 - np.mean(fields['SF6'][self.inside])/max(self.nSF6_feed,1e-10))*100
                if verbose:
                    print(f"    iter {it:3d}: [F] drop = {drop:.1f}%, SF6 depl = {sf6d:.0f}%")
        
        elapsed = time.time() - t0
        icp = (self.inside & (np.outer(m.rc, np.ones(Nz)) <= self.R_icp) &
               (np.outer(np.ones(Nr), m.zc) >= self.z_apt_top))
        proc = self.inside & (np.outer(np.ones(Nr), m.zc) < self.z_apt_bot)
        
        results = {'fields':fields,'ne':ne,'Te':Te,'mesh':m,'inside':self.inside,
                   'icp_mask':icp,'proc_mask':proc,'elapsed':elapsed,'F_drop_pct':drop}
        for sp in SPECIES:
            fld = fields[sp]
            results[f'{sp}_icp'] = np.sum(fld[icp]*m.vol[icp])/max(np.sum(m.vol[icp]),1e-30)
            results[f'{sp}_proc'] = np.sum(fld[proc]*m.vol[proc])/max(np.sum(m.vol[proc]),1e-30)
        
        if verbose:
            print(f"\n  Completed in {elapsed:.1f}s")
            print(f"  [F] drop = {drop:.1f}%")
            for sp in SPECIES:
                print(f"  [{sp}] ICP = {results[f'{sp}_icp']*1e-6:.2e} cm-3")
        return results
