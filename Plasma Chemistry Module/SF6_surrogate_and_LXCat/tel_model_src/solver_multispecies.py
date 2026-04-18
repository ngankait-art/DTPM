"""
TEL 9-Species Solver — subclass of TELSolver.

Inherits from solver.py:
  - _compute_ne(Te) → ne field with physical power scaling
  - _compute_Te_spatial(ne) → Te field  
  - _build_diffusion_matrix(D_field, gamma_map) → Robin BC sparse matrix
  - mesh, geometry mask, bc_type, flat_to_ij, ij_to_flat, n_active

Extends:
  - Solves ALL 9 neutral species with 2D diffusion (smooth aperture)
  - Charged species from Kokkoris quasi-neutrality + Gaussian smoothing
"""
import numpy as np
import time
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
from scipy.constants import k as kB, pi, atomic_mass as AMU

from solver import TELSolver
from sf6_chemistry import compute_rates, SPECIES, M, WALL_GAMMA
from boundary_conditions import get_gamma_map
from geometry import (BC_QUARTZ, BC_WINDOW, BC_AL_SIDE, BC_AL_TOP,
                      BC_WAFER, BC_SHOULDER)


class TELMultiSpeciesSolver(TELSolver):
    
    def __init__(self, Nr=50, Nz=80, P_rf=700, p_mTorr=10, frac_Ar=0.0,
                 gamma_Al=0.18, eta=0.43):
        super().__init__(Nr=Nr, Nz=Nz, P_rf=P_rf, p_mTorr=p_mTorr,
                         gamma_Al=gamma_Al, eta=eta, frac_Ar=frac_Ar)

    def _gamma_map_for_species(self, species_name):
        """Build a bc_type → gamma dict for a given species."""
        bc_to_mat = {
            BC_QUARTZ: 'quartz', BC_WINDOW: 'window',
            BC_AL_SIDE: 'aluminium', BC_AL_TOP: 'aluminium',
            BC_SHOULDER: 'aluminium', BC_WAFER: 'silicon',
        }
        gmap = {}
        for bc_val, mat in bc_to_mat.items():
            g = WALL_GAMMA.get(mat, {}).get(species_name, 0.001)
            if species_name == 'F' and mat == 'aluminium':
                g = self.gamma_Al  # use the validated value
            gmap[bc_val] = g
        return gmap

    def _build_species_matrix(self, D_val, species_name):
        """Build diffusion matrix for any species using parent's validated builder."""
        m = self.mesh
        D_field = np.full((m.Nr, m.Nz), D_val)
        gamma_map = self._gamma_map_for_species(species_name)
        
        # Temporarily set v_th_F to this species' thermal velocity
        mass = M.get(species_name, 32)
        v_th = np.sqrt(8 * kB * self.Tgas / (pi * mass * AMU))
        old_vth = self.v_th_F
        self.v_th_F = v_th
        A = self._build_diffusion_matrix(D_field, gamma_map)
        self.v_th_F = old_vth
        return A

    def solve(self, n_iter=60, w=0.08, verbose=True):
        t0 = time.time()
        m = self.mesh; Nr, Nz = m.Nr, m.Nz; N = self.n_active
        
        # ne/Te from parent's validated methods
        Te = self._compute_Te_spatial(np.zeros((Nr, Nz)))
        ne, ne_avg = self._compute_ne(Te)
        Te_avg = np.mean(Te[self.inside & (Te > 0)])
        k = compute_rates(Te_avg, self.ng * 1e-6)
        tau = max(0.1333 * self.p_mTorr * np.sum(m.vol[self.inside]) /
                  (100 * 1.01325e5 / 60 * 1e-6), 0.01)

        # Diffusion coefficients
        from sf6_chemistry import compute_diffusion_coefficients
        D_all = compute_diffusion_coefficients(self.Tgas, self.p_mTorr)

        # Build diffusion matrices for all 9 species
        A_mats = {sp: self._build_species_matrix(D_all[sp], sp) for sp in SPECIES}

        # Initialise
        fields = {}
        fields['SF6'] = np.where(self.inside, self.nSF6_feed * 0.5, 0.0)
        fields['F'] = np.where(self.inside, self.nSF6_feed * 0.1, 0.0)
        for sp in ['SF5','SF4','SF3','SF2','SF','S','F2']:
            fields[sp] = np.where(self.inside, self.nSF6_feed * 0.01, 0.0)

        for it in range(n_iter):
            # Recompute ne with updated Te (every 10 iters)
            if it > 0 and it % 5 == 0:
                # Update SF6 and alpha for self-consistent Te and ne
                icp_m = (self.inside &
                         (np.outer(m.rc, np.ones(Nz)) <= self.R_icp) &
                         (np.outer(np.ones(Nr), m.zc) >= self.z_apt_top))
                self._nSF6_avg = np.sum(fields['SF6'][icp_m]*m.vol[icp_m])/max(np.sum(m.vol[icp_m]),1e-30)
                # Compute alpha from current fields for EN correction
                k_rec_local = 1.5e-9*1e-6
                R_att_local = (k['at30']+k['at31'])*fields['SF6']
                alpha_field = (-1+np.sqrt(np.clip(1+4*R_att_local/(k_rec_local*ne+1e-30),0,1e12)))/2
                self._alpha_avg = np.sum(alpha_field[icp_m]*m.vol[icp_m])/max(np.sum(m.vol[icp_m]),1e-30)
                Te = self._compute_Te_spatial(ne)
                ne, ne_avg = self._compute_ne(Te)
                k = compute_rates(np.mean(Te[self.inside & (Te>0)]), self.ng*1e-6)
                Te_avg = np.mean(Te[self.inside & (Te > 0)])
                k = compute_rates(Te_avg, self.ng * 1e-6)

            for sp in SPECIES:
                rhs = np.zeros(N); loss_diag = np.zeros(N)
                for idx in range(N):
                    i, j = self.flat_to_ij[idx]
                    ne_l = ne[i,j]; n6 = fields['SF6'][i,j]; nFl = fields['F'][i,j]

                    if sp == 'F':
                        src = ne_l*n6*(k['d1']+2*k['d2']+3*k['d3']+2*k['d4']+3*k['d5']) \
                            + ne_l*fields['SF5'][i,j]*k['d7'] \
                            + ne_l*fields['SF4'][i,j]*k['d8'] \
                            + 2*ne_l*fields['F2'][i,j]*k['d6']
                        loss = k['nr42']*fields['SF5'][i,j] + 1/tau
                    elif sp == 'SF6':
                        src = self.nSF6_feed/tau + k['nr42']*fields['SF5'][i,j]*nFl
                        loss = ne_l*(k['d1']+k['d2']+k['d3']+k['d4']+k['d5']) + 1/tau
                    elif sp == 'SF5':
                        src = k['d1']*ne_l*n6
                        loss = (k['d7']+k['iz25']+k['iz26'])*ne_l + k['nr42']*nFl + 1/tau
                    elif sp == 'SF4':
                        src = k['d2']*ne_l*n6 + k['d7']*ne_l*fields['SF5'][i,j]
                        loss = k['d8']*ne_l + k['nr41']*nFl + 1/tau
                    elif sp == 'SF3':
                        src = k['d3']*ne_l*n6 + k['d8']*ne_l*fields['SF4'][i,j]
                        loss = (k['d9']+k['iz27'])*ne_l + k['nr40']*nFl + 1/tau
                    elif sp == 'SF2':
                        src = k['d4']*ne_l*n6 + k['d9']*ne_l*fields['SF3'][i,j]
                        loss = k['d10']*ne_l + 1/tau
                    elif sp == 'SF':
                        src = k['d5']*ne_l*n6 + k['d10']*ne_l*fields['SF2'][i,j]
                        loss = k['d11']*ne_l + 1/tau
                    elif sp == 'S':
                        src = k['d11']*ne_l*fields['SF'][i,j]
                        # S wall loss: S sticks to walls (Kokkoris effective γ_S ≈ 0.01)
                        v_th_S = 354.0  # m/s (S thermal velocity, M=32)
                        gamma_S = 0.01  # Kokkoris WALL_GAMMA
                        r_val_s = m.rc[i]; z_val_s = m.zc[j]
                        dist_s = min(
                            abs(r_val_s - self.R_proc) if z_val_s < self.z_apt_bot else abs(r_val_s - self.R_icp),
                            z_val_s, abs(self.z_top - z_val_s) if z_val_s > self.z_apt_top else 1.0)
                        dist_s = max(dist_s, 0.002)
                        kwS = gamma_S * v_th_S / 4 / (dist_s * 2.5)  # enhanced S wall loss
                        loss = k['iz29']*ne_l + k['nr37']*nFl + kwS + 1/tau
                    elif sp == 'F2':
                        src = ne_l*n6*(k['d4']+k['d5'])
                        # Wall F→F2 production: 50% of F lost at walls becomes F2
                        # Use local F density as proxy for wall F flux
                        # Scale: kwF_local ≈ gamma_eff * v_th/4 / L_diff
                        # where L_diff is the diffusion length to nearest wall
                        r_val = m.rc[i]; z_val = m.zc[j]
                        dist_to_wall = min(
                            abs(r_val - self.R_proc) if z_val < self.z_apt_bot else abs(r_val - self.R_icp),
                            z_val,  # distance to wafer
                            abs(self.z_top - z_val) if z_val > self.z_apt_top else 1.0
                        )
                        dist_to_wall = max(dist_to_wall, 0.002)  # floor at 2mm
                        v_th_F_l = 591.0  # m/s (F thermal velocity at 313K)
                        gamma_wall = 0.15  # Kokkoris S1 sticking probability
                        kwF_local = gamma_wall * v_th_F_l / 4 / (dist_to_wall * 55)  # tuned to match 0D F2
                        src += 0.5 * kwF_local * nFl  # 50% of wall-lost F becomes F2
                        loss = k['d6']*ne_l + 1/tau
                    else:
                        src = 0; loss = 1/tau

                    rhs[idx] = -src; loss_diag[idx] = -loss

                A_total = A_mats[sp] + sp_diags(loss_diag, 0, shape=(N, N))
                try:
                    x = spsolve(A_total, rhs)
                    x = np.maximum(x, 0)
                    new = self._flat_to_field(x, fill=0.0)
                    fields[sp] = (1 - w) * fields[sp] + w * new
                    fields[sp] = np.clip(fields[sp], 0, self.ng * 10)
                except:
                    pass

            if verbose and (it % 10 == 0 or it == n_iter - 1):
                Fw = fields['F'][:,0]; Fc = max(Fw[0], 1e-10)
                ie = np.argmin(np.abs(m.rc - self.R_proc))
                drop = (1 - Fw[ie]/Fc) * 100
                sf6d = (1 - np.mean(fields['SF6'][self.inside])/self.nSF6_feed)*100
                print(f"  iter {it:3d}: [F] drop={drop:.1f}%  SF6 depl={sf6d:.0f}%")

        # Charged species
        k_rec = 1.5e-9 * 1e-6
        R_att = (k['at30'] + k['at31']) * fields['SF6']
        alpha = (-1 + np.sqrt(np.clip(1 + 4*R_att/(k_rec*ne+1e-30), 0, 1e12))) / 2
        alpha = np.clip(alpha, 0, 500)
        n_neg = alpha * ne; n_pos = ne + n_neg

        def smooth(f):
            f2 = f.copy(); f2[~self.inside] = 0
            for _ in range(3):
                fs = gaussian_filter(f2, 2.5)
                ms = gaussian_filter(self.inside.astype(float), 2.5)
                f2 = np.where(ms > 0.05, fs/ms, 0); f2[~self.inside] = 0
            return f2

        ne_s = smooth(ne); np_s = smooth(n_pos); nn_s = smooth(n_neg)
        ions = {'ne': ne_s, 'n+': np_s, 'n-': nn_s,
                'alpha': np.clip(nn_s/(ne_s+1e-20), 0, 500),
                'SF5+': np_s*0.65, 'SF3+': np_s*0.20,
                'SF4+': np_s*0.10, 'F+': np_s*0.05,
                'F-': nn_s*0.55, 'SF6-': nn_s*0.25,
                'SF5-': nn_s*0.15, 'SF4-': nn_s*0.05}

        Fw = fields['F'][:,0]; Fc = max(Fw[0], 1e-10)
        ie = np.argmin(np.abs(m.rc - self.R_proc))
        drop = (1 - Fw[ie]/Fc) * 100

        icp = (self.inside &
               (np.outer(m.rc, np.ones(Nz)) <= self.R_icp) &
               (np.outer(np.ones(Nr), m.zc) >= self.z_apt_top))
        proc = self.inside & (np.outer(np.ones(Nr), m.zc) < self.z_apt_bot)

        return {'fields': fields, 'ne': ne, 'ions': ions, 'Te': Te_avg,
                'ne_avg': ne_avg, 'mesh': m, 'inside': self.inside,
                'icp_mask': icp, 'proc_mask': proc,
                'F_drop_pct': drop, 'elapsed': time.time() - t0}
