"""
TEL Unified PDE Solver

Solves the fluorine transport equation on a single structured mesh
with a geometry mask enforcing the exact TEL T-shaped reactor geometry.

The diffusion equation in cylindrical (r,z) coordinates:

    (1/r) d/dr(r D dF/dr) + d/dz(D dF/dz) + S - L*F = 0

is discretized on the active cells only. Inactive (solid) cells are
excluded from the linear system. Robin boundary conditions are applied
at every active cell that borders an inactive cell or domain edge.
"""
import numpy as np
import time
from scipy.constants import e as eC, k as kB, pi
from scipy.optimize import brentq
from scipy.special import j0
from scipy.sparse.linalg import spsolve
from scipy import sparse

from mesh import Mesh2D
from sf6_rates import rates
from geometry import (build_geometry_mask, build_index_maps,
                       BC_INTERIOR, BC_AXIS, BC_QUARTZ, BC_WINDOW,
                       BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER)
from boundary_conditions import (get_gamma_map, wall_sf6_regeneration,
                                   wall_F_loss, BC_TO_SURFACE)


def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate coefficient (Ryan & Plumb 1990)."""
    if M_cm3 <= 0 or k0 <= 0:
        return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    return k0 * M_cm3 / (1 + Pr) * Fc ** (1.0 / (1.0 + log_Pr**2))


class TELSolver:
    """Unified single-domain solver for the TEL ICP etcher.
    
    Parameters
    ----------
    R_icp : float
        ICP source radius (m). Default: 0.038.
    L_icp : float
        ICP source height (m). Default: 0.1815.
    R_proc : float
        Processing region radius (m). Default: 0.105.
    L_proc : float
        Processing region height (m). Default: 0.050.
    L_apt : float
        Aperture gap height (m). Default: 0.002.
    Nr, Nz : int
        Mesh resolution. Default: 50, 80.
    P_rf : float
        RF power (W). Default: 700.
    p_mTorr : float
        Pressure (mTorr). Default: 10.
    frac_Ar : float
        Argon fraction (0-1). Default: 0.0.
    Q_sccm : float
        Total gas flow (sccm). Default: 100.
    Tgas : float
        Gas temperature (K). Default: 313.
    eta : float
        Power coupling efficiency. Default: 0.43.
    gamma_Al : float
        F recombination probability on Al (Kokkoris). Default: 0.18.
    """

    def __init__(self, R_icp=0.038, L_icp=0.1815, R_proc=0.105,
                 L_proc=0.050, L_apt=0.002, Nr=50, Nz=80,
                 P_rf=700, p_mTorr=10, frac_Ar=0.0, Q_sccm=100,
                 Tgas=313, eta=0.43, gamma_quartz=0.001,
                 gamma_Al=0.18, gamma_wafer=0.025, gamma_window=0.001):

        # Store parameters
        self.R_icp = R_icp
        self.R_proc = R_proc
        self.L_icp = L_icp
        self.L_proc = L_proc
        self.L_apt = L_apt
        self.P_rf = P_rf
        self.eta = eta
        self.p_mTorr = p_mTorr
        self.frac_Ar = frac_Ar
        self.Tgas = Tgas
        self.P_abs = eta * P_rf
        self.gamma_Al = gamma_Al

        # Derived quantities
        p_Pa = p_mTorr * 0.133322
        self.ng = p_Pa / (kB * Tgas)
        self.nSF6_feed = self.ng * (1 - frac_Ar)
        self.nAr = self.ng * frac_Ar
        self.D_F = 1.6      # m²/s
        self.D_SF6 = 0.5    # m²/s
        self.v_th_F = np.sqrt(8 * kB * Tgas / (pi * 19.0 * 1.66054e-27))
        self.M_cm3 = self.ng * 1e-6

        V_total = pi * R_icp**2 * L_icp + pi * R_proc**2 * L_proc
        Q_tp = Q_sccm * 1e-6 / 60 * 1.01325e5 * (Tgas / 273.15)
        self.tau_R = p_Pa * V_total / Q_tp if Q_tp > 0 else 1e10

        # Penning parameters
        self.k_Penn = 2e-10 * 1e-6
        self.k_Penn_d = 3e-10 * 1e-6
        self.k_quench = 5e-10 * 1e-6
        self.k_exc = 3.5e-11 * 1e-6
        self.k_izm = 6e-11 * 1e-6

        # Build mesh
        L_total = L_proc + L_apt + L_icp
        self.mesh = Mesh2D(R=R_proc, L=L_total, Nr=Nr, Nz=Nz,
                           beta_r=1.2, beta_z=1.0)

        # z boundaries
        self.z_wafer = 0.0
        self.z_apt_bot = L_proc
        self.z_apt_top = L_proc + L_apt
        self.z_top = L_total

        # Build geometry
        self.inside, self.bc_type = build_geometry_mask(
            self.mesh, R_icp, R_proc, self.z_apt_bot, self.z_apt_top, self.z_top)
        self.ij_to_flat, self.flat_to_ij, self.n_active = \
            build_index_maps(self.inside)

        # Robin BC map
        self.gamma_map = get_gamma_map(
            gamma_quartz, gamma_Al, gamma_wafer, gamma_window)

    def _build_diffusion_matrix(self, D_field, gamma_map):
        """Assemble sparse diffusion operator for active cells only."""
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        N = self.n_active
        rows, cols, vals = [], [], []

        for k in range(N):
            i, j = self.flat_to_ij[k]
            ri = m.rc[i]
            dri = m.dr[i]
            dzj = m.dz[j]
            Dij = D_field[i, j]
            diag = 0.0

            # ── Radial ──
            if i == 0:
                # Axis: L'Hopital
                if i + 1 < Nr and self.inside[i + 1, j]:
                    c = 2.0 * 0.5 * (Dij + D_field[i+1,j]) / (dri * m.drc[0])
                    rows.append(k); cols.append(self.ij_to_flat[i+1,j]); vals.append(c)
                    diag -= c
            else:
                # Inner neighbour
                if self.inside[i-1, j]:
                    c = 0.5*(D_field[i-1,j]+Dij)*m.rf[i]/(ri*dri*m.drc[i-1])
                    rows.append(k); cols.append(self.ij_to_flat[i-1,j]); vals.append(c)
                    diag -= c
                # Outer neighbour
                if i < Nr-1 and self.inside[i+1, j]:
                    c = 0.5*(Dij+D_field[i+1,j])*m.rf[i+1]/(ri*dri*m.drc[i])
                    rows.append(k); cols.append(self.ij_to_flat[i+1,j]); vals.append(c)
                    diag -= c
                else:
                    g = gamma_map.get(self.bc_type[i,j], 0.0)
                    if g > 0:
                        h = g * self.v_th_F / 4
                        drw = m.rf[min(i+1,Nr)] - ri
                        rf = m.rf[min(i+1,Nr)]
                        diag -= Dij*h/(Dij+h*drw)*rf/(ri*dri)

            # ── Axial ──
            if j > 0 and self.inside[i, j-1]:
                c = 0.5*(D_field[i,j-1]+Dij)/(dzj*m.dzc[j-1])
                rows.append(k); cols.append(self.ij_to_flat[i,j-1]); vals.append(c)
                diag -= c
            else:
                g = gamma_map.get(
                    BC_WAFER if self.bc_type[i,j]==BC_WAFER else self.bc_type[i,j], 0.0)
                if g > 0:
                    h = g * self.v_th_F / 4
                    dzw = m.zc[j] - m.zf[j]
                    diag -= Dij*h/(Dij+h*dzw)/dzj

            if j < Nz-1 and self.inside[i, j+1]:
                c = 0.5*(Dij+D_field[i,j+1])/(dzj*m.dzc[j])
                rows.append(k); cols.append(self.ij_to_flat[i,j+1]); vals.append(c)
                diag -= c
            else:
                bc = self.bc_type[i,j]
                g = gamma_map.get(
                    BC_WINDOW if bc==BC_WINDOW else (BC_AL_TOP if bc==BC_AL_TOP else bc), 0.0)
                if g > 0:
                    h = g * self.v_th_F / 4
                    dzw = m.zf[min(j+1,Nz)] - m.zc[j]
                    diag -= Dij*h/(Dij+h*dzw)/dzj

            rows.append(k); cols.append(k); vals.append(diag)

        return sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    def _field_to_flat(self, field_2d):
        """Extract active-cell values into flat vector."""
        return np.array([field_2d[i,j] for i,j in self.flat_to_ij], dtype=np.float64)

    def _flat_to_field(self, flat, fill=np.nan):
        """Scatter flat vector back to 2D field."""
        f = np.full((self.mesh.Nr, self.mesh.Nz), fill, dtype=np.float64)
        for k, (i, j) in enumerate(self.flat_to_ij):
            f[i, j] = float(flat[k])
        return f

    def _solve_Te_0D(self):
        """Global Te from particle balance with EN correction.
        
        Uses actual (depleted) SF6 density and alpha-dependent h-factors
        from Lee & Lieberman (1995) for electronegative plasmas.
        """
        Mi = 127.06 * 1.66054e-27
        sigma_in = 5e-19
        R, L = self.R_icp, self.L_icp
        lambda_i = 1.0 / (self.ng * sigma_in)
        
        # Use actual SF6 if available (self-consistent)
        nSF6_eff = getattr(self, '_nSF6_avg', self.nSF6_feed)
        # Use current alpha if available
        alpha = getattr(self, '_alpha_avg', 0.0)
        
        # EN-corrected h-factors (Lee & Lieberman 1995, damped)
        hR_EP = 0.86 / np.sqrt(3 + R / (2 * lambda_i))
        hL_EP = 0.86 / np.sqrt(3 + L / (2 * lambda_i))
        A_eff_EP = (2*R*L*hR_EP + 2*R**2*hL_EP) / (R**2 * L)

        def balance(Te):
            k = rates(Te)
            nu_iz = nSF6_eff * k['iz_SF6_total']
            nu_att = nSF6_eff * k['att_SF6_total']
            if self.frac_Ar > 0.01:
                nu_iz += self.k_Penn * self.nAr * 0.01
            # Damped EN wall loss: beta=0.15 gives good match to 0D
            k_wall = np.sqrt(eC * Te / Mi) * A_eff_EP * (1 + 0.05 * alpha)
            return nu_iz - nu_att - k_wall

        try:
            return brentq(balance, 0.5, 15.0)
        except ValueError:
            return 3.0

    def _compute_Te_spatial(self, ne):
        """Spatially varying Te from EM power profile."""
        Te0 = self._solve_Te_0D()
        m = self.mesh
        Te = np.full((m.Nr, m.Nz), Te0)
        for i in range(m.Nr):
            r = m.rc[i]
            for j in range(m.Nz):
                z = m.zc[j]
                if not self.inside[i, j]:
                    continue
                if z >= self.z_apt_top and r <= self.R_icp:
                    zl = (z - self.z_apt_top) / self.L_icp
                    rf = r / self.R_icp
                    P = np.exp(-(1-zl)**2/0.3) * np.exp(-rf**2/0.8)
                    Te[i, j] = Te0 * (0.92 + 0.16 * P)
                else:
                    Te[i, j] = Te0 * 0.90
        return Te

    def _compute_ne(self, Te):
        """Self-consistent ne from integrated power balance."""
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        shape = np.zeros((Nr, Nz))
        for i in range(Nr):
            r = m.rc[i]
            for j in range(Nz):
                z = m.zc[j]
                if not self.inside[i,j]:
                    continue
                if z >= self.z_apt_top and r <= self.R_icp:
                    # ICP source: Bessel-cosine eigenmode
                    bessel = j0(2.405*r/self.R_icp)
                    cosine = np.cos(pi*(z-self.z_apt_top)/(2*self.L_icp))
                    icp_val = max(bessel * cosine, 0.01)
                    # Smooth blend near aperture (within 8mm above)
                    blend_dist = z - self.z_apt_top
                    if blend_dist < 0.008:
                        t = blend_dist / 0.008
                        ne_at_apt = 0.40  # ne at ICP-aperture interface
                        shape[i,j] = ne_at_apt + (icp_val - ne_at_apt) * t**0.7
                    else:
                        shape[i,j] = icp_val
                elif z >= self.z_apt_bot:
                    # Aperture: smooth cosine transition (physically: ambipolar
                    # diffusion through narrow gap, L_aperture << lambda_mfp)
                    frac = (z - self.z_apt_bot) / max(self.z_apt_top - self.z_apt_bot, 1e-3)
                    ne_proc = 0.20  # ne at processing side of aperture
                    ne_icp = 0.40   # ne at ICP side of aperture
                    # Cosine blend is smoother than linear at boundaries
                    shape[i,j] = ne_proc + (ne_icp - ne_proc) * 0.5*(1 + np.cos(pi*(1-frac)))
                elif r <= self.R_icp:
                    # Processing directly below aperture: ambipolar decay
                    # L_decay = sqrt(D_a / k_rec*alpha*ne) ~ 10-15mm
                    dist = self.z_apt_bot - z
                    L_decay = 0.012  # 12mm (recombination-limited)
                    shape[i,j] = 0.20 * np.exp(-dist / L_decay) + 0.01
                else:
                    # Processing region away from aperture axis
                    dist = self.z_apt_bot - z
                    r_dist = r - self.R_icp
                    L_decay = 0.012
                    L_radial = 0.020  # radial decay from aperture axis
                    shape[i,j] = 0.20 * np.exp(-dist/L_decay) * np.exp(-r_dist/L_radial) + 0.01

        shape_avg = np.sum(shape * m.vol * self.inside) / \
                    max(np.sum(m.vol * self.inside), 1e-30)

        icp_mask = (self.inside &
                    (np.outer(m.rc, np.ones(Nz)) <= self.R_icp) &
                    (np.outer(np.ones(Nr), m.zc) >= self.z_apt_top))
        Te_icp = np.mean(Te[icp_mask]) if np.any(icp_mask) else 3.0

        k = rates(Te_icp)
        # Use actual SF6 density if available (self-consistent)
        nSF6_for_ne = getattr(self, '_nSF6_avg', self.nSF6_feed)
        nu_iz = nSF6_for_ne * k['iz_SF6_total']
        if self.frac_Ar > 0.01:
            nu_iz += self.k_Penn * self.nAr * 0.01
        # Energy cost per ion-electron pair (Lieberman & Lichtenberg Ch. 10)
        # Ec = collisional energy loss, eiw = ion wall energy, 2*Te = electron wall energy
        # Ec from Eloss/Riz using SF6 cross-sections (benchmarked to 0D model: Ec ≈ 200-240 eV)
        Eloss_coeff = (16*k['iz18']+20*k['iz19']+20.5*k['iz20']+28*k['iz21']+
                       37.5*k['iz22']+18*k['iz23']+29*k['iz24']
                       +9.6*k['d1']+12.1*k['d2']+16*k['d3']+18.6*k['d4']+22.7*k['d5'])
        Riz_coeff = k['iz_SF6_total']
        # Add fragment ionization contributions (lowers Ec by ~10-20%)
        Riz_coeff += 0.15 * k['iz_SF6_total']  # ~15% from SF5/SF3/F ionization
        Ec = np.clip(Eloss_coeff / max(Riz_coeff, 1e-30), 80, 400)
        # Sheath potential
        eiw = 0.5 * Te_icp * np.log(max(127*1.66e-27/(2*3.14159*9.109e-31), 1))
        eps_T = Ec + eiw + 2*Te_icp
        # Use ICP volume for power balance (matches validated 0D model)
        V_icp = np.sum(m.vol[icp_mask])
        ne_avg = self.P_abs / max(nu_iz * eps_T * eC * V_icp, 1e-30)

        return ne_avg * shape, ne_avg

    def solve(self, n_iter=80, w=0.12, verbose=True):
        """Run the solver.
        
        Returns
        -------
        result : dict
            Contains nF, nSF6, ne, Te, F_wafer, r_wafer, F_drop_pct, etc.
        """
        t0 = time.time()
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz

        if verbose:
            print(f"{'='*70}")
            print(f"  TEL Unified Solver — Material-Specific Wall Chemistry")
            print(f"{'='*70}")
            print(f"  Mesh: {Nr}x{Nz}={Nr*Nz}, active: {self.n_active} "
                  f"({self.n_active*100//(Nr*Nz)}%)")
            print(f"  P={self.P_rf}W, eta={self.eta}, p={self.p_mTorr}mT, "
                  f"Ar={self.frac_Ar*100:.0f}%")

        # Initialise
        Te = self._compute_Te_spatial(np.zeros((Nr, Nz)))
        ne, ne_avg = self._compute_ne(Te)
        nSF6 = np.where(self.inside, self.nSF6_feed * 0.5, 0.0)
        nF = np.where(self.inside, 1e18, 0.0)
        nSF5 = np.where(self.inside, self.nSF6_feed * 0.02, 0.0)

        # Build operators
        D_F = np.where(self.inside, self.D_F, 0.0)
        D_SF6 = np.where(self.inside, self.D_SF6, 0.0)
        A_F = self._build_diffusion_matrix(D_F, self.gamma_map)
        A_SF6 = self._build_diffusion_matrix(D_SF6, {})

        if verbose:
            hdr = (f"  {'It':>3s} {'Te':>5s} {'ne':>10s} {'F_icp':>10s} "
                   f"{'F_proc':>10s} {'Fc':>10s} {'Fe':>10s} {'drop':>5s}")
            print(f"\n{hdr}")

        for it in range(n_iter):
            Te_avg = np.mean(Te[self.inside])

            # ── SF6 ──
            src = np.zeros(self.n_active)
            loss = np.zeros(self.n_active)
            for kk in range(self.n_active):
                i, j = self.flat_to_ij[kk]
                kc = rates(Te[i, j])
                kd = (kc['d1']+kc['d2']+kc['d3']+kc['d4']+kc['d5']+
                      kc['iz_SF6_total']+kc['att_SF6_total'])
                rep = 1.0 / self.tau_R
                z = m.zc[j]
                if z > self.z_apt_top + 0.7 * self.L_icp:
                    rep += (z - self.z_apt_top - 0.7*self.L_icp) / \
                           (0.3*self.L_icp) / (self.tau_R * 0.2)
                # Wall SF6 regeneration (material-specific)
                bc = self.bc_type[i, j]
                dr_dz = m.dr[i] if bc in (BC_QUARTZ, BC_AL_SIDE, BC_SHOULDER) else m.dz[j]
                Rw = wall_sf6_regeneration(bc, nF[i,j], nSF5[i,j], self.v_th_F, dr_dz)
                src[kk] = self.nSF6_feed * rep + Rw
                loss[kk] = -(ne[i,j] * kd + rep)

            x = np.asarray(spsolve(A_SF6 + sparse.diags(loss), -src), dtype=np.float64)
            nSF6_new = self._flat_to_field(
                np.clip(x, 0.01*self.nSF6_feed, self.nSF6_feed), 0.0)
            nSF6 = (1 - w) * nSF6 + w * nSF6_new

            # ── Fluorine ──
            f_src = np.zeros(self.n_active)
            f_loss = np.zeros(self.n_active)
            for kk in range(self.n_active):
                i, j = self.flat_to_ij[kk]
                kc = rates(Te[i, j])
                RF = ne[i,j] * nSF6[i,j] * (
                    kc['d1'] + kc['d2']*2 + kc['d3']*3 + kc['d4']*2 +
                    kc['d5']*3 + kc.get('d7',0) + kc.get('d8',0))
                # Penning F production
                if self.frac_Ar > 0.01:
                    nArm = (self.k_exc * ne[i,j] * self.nAr /
                            max(self.k_izm*ne[i,j] + self.k_quench*self.nSF6_feed + 1e-20, 1e-30))
                    RF += (self.k_Penn + self.k_Penn_d) * nArm * nSF6[i,j]
                f_src[kk] = RF
                # Wall F loss (material-specific)
                bc = self.bc_type[i, j]
                dr_dz = m.dr[i] if bc in (BC_QUARTZ, BC_AL_SIDE, BC_SHOULDER) else m.dz[j]
                Rl = wall_F_loss(bc, nF[i,j], self.v_th_F, dr_dz)
                f_loss[kk] = -Rl / max(nF[i,j], 1e6)

            pump_wall = np.full(self.n_active, -1.0/self.tau_R) + f_loss
            y = np.asarray(spsolve(A_F + sparse.diags(pump_wall), -f_src), dtype=np.float64)
            nF_new = self._flat_to_field(np.maximum(y, 0.0), 0.0)
            nF = (1 - w) * nF + w * nF_new

            nSF5 = np.where(self.inside, nSF6 * 0.03, 0.0)
            Te = self._compute_Te_spatial(ne)
            ne, ne_avg = self._compute_ne(Te)

            # Diagnostics
            if verbose and it % max(n_iter // 10, 1) == 0:
                icp_m = (self.inside &
                         (np.outer(m.rc, np.ones(Nz)) <= self.R_icp) &
                         (np.outer(np.ones(Nr), m.zc) >= self.z_apt_top))
                proc_m = self.inside & (np.outer(np.ones(Nr), m.zc) < self.z_apt_bot)
                Fi = np.sum(nF[icp_m]*m.vol[icp_m]) / max(np.sum(m.vol[icp_m]), 1e-30)
                Fp = np.sum(nF[proc_m]*m.vol[proc_m]) / max(np.sum(m.vol[proc_m]), 1e-30)
                Fa = nF[:, 0][self.inside[:, 0]]
                Fc = Fa[0] if len(Fa) > 0 else 0
                Fe = Fa[-1] if len(Fa) > 0 else 0
                drop = (1 - Fe/max(Fc, 1e6)) * 100 if Fc > 1e6 else 0
                print(f"  {it:3d} {Te_avg:5.2f} {ne_avg*1e-6:10.2e} "
                      f"{Fi*1e-6:10.2e} {Fp*1e-6:10.2e} "
                      f"{Fc*1e-6:10.2e} {Fe*1e-6:10.2e} {drop:4.0f}%")

        elapsed = time.time() - t0
        rw = m.rc[self.inside[:, 0]]
        Fw = nF[:, 0][self.inside[:, 0]]
        Fc = Fw[0] if len(Fw) > 0 else 0
        Fe = Fw[-1] if len(Fw) > 0 else 0
        drop = (1 - Fe/max(Fc, 1e6)) * 100 if Fc > 1e6 else 0

        if verbose:
            print(f"\n{'='*70}")
            print(f"  Done ({elapsed:.1f}s), [F] drop = {drop:.0f}% (Mettler: 74%)")
            print(f"{'='*70}")

        return {
            'nF': nF, 'nSF6': nSF6, 'ne': ne, 'Te': Te,
            'ne_avg': ne_avg, 'inside': self.inside, 'bc_type': self.bc_type,
            'mesh': self.mesh, 'F_wafer': Fw * 1e-6, 'r_wafer': rw,
            'F_drop_pct': drop, 'elapsed': elapsed,
        }


class TELSolverWithEnergy(TELSolver):
    """Extended solver with electron energy PDE for Te(r,z).
    
    Solves: ∇·(κ_e ∇Te) + P_abs(r,z)/(3/2·ne) - L_e(Te) = 0
    where:
      κ_e = (5/3) · D_e · ne  (electron thermal conductivity)
      P_abs(r,z) = P_rf · η · p(r,z) / ∫p·dV  (normalised EM power)
      L_e = Σ(ε_k · k_k · n_SF6)  (collisional energy loss rate)
    """
    
    def _compute_power_deposition(self):
        """EM power deposition profile p(r,z).
        
        Uses a physically motivated profile: power peaks at the skin depth
        (~1cm from the quartz wall) and near the top of the ICP region
        (close to the coil). This is an approximation to the full EM solve.
        """
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        p = np.zeros((Nr, Nz))
        
        for i in range(Nr):
            r = m.rc[i]
            for j in range(Nz):
                z = m.zc[j]
                if not self.inside[i, j]:
                    continue
                if z >= self.z_apt_top and r <= self.R_icp:
                    # Power peaks near the wall (skin depth ~ 1cm) and near the coil (top)
                    z_local = (z - self.z_apt_top) / self.L_icp  # 0 at bottom, 1 at top
                    r_norm = r / self.R_icp
                    # Radial: peaks near wall at r/R ≈ 0.7 (skin depth effect)
                    p_r = np.exp(-((r_norm - 0.7) / 0.25)**2)
                    # Axial: peaks near coil (top third of ICP)
                    p_z = np.exp(-((z_local - 0.85) / 0.3)**2)
                    p[i, j] = p_r * p_z
        
        # Normalise so ∫p·dV = 1
        total = np.sum(p * m.vol)
        if total > 0:
            p /= total
        return p
    
    def _solve_Te_energy_pde(self, ne, nSF6):
        """Solve the electron energy PDE on the masked domain."""
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        
        # Power deposition profile
        p_profile = self._compute_power_deposition()
        
        # Electron thermal conductivity κ_e ≈ (5/3) · D_e · ne
        # D_e ≈ 100 m²/s at 10 mTorr (much larger than neutral D)
        D_e = 100.0  # m²/s (electron diffusion at 10 mTorr)
        kappa = np.where(self.inside, (5.0/3.0) * D_e * ne, 0.0)
        
        # Build diffusion matrix with Dirichlet BCs (Te = Te_wall at walls)
        # Use the same mask but with Dirichlet (gamma → ∞)
        Te_wall = 1.0  # eV (cold walls)
        gamma_Te = {}
        for bc_type in [BC_QUARTZ, BC_WINDOW, BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER]:
            gamma_Te[bc_type] = 1e6  # effectively Dirichlet
        
        A_Te = self._build_diffusion_matrix(kappa, gamma_Te)
        
        # Source: P_abs / (3/2 · e)  [converts W/m³ to eV/(m³·s)]
        # Loss: Σ(ε_k · k_k · n_SF6 · ne)
        src = np.zeros(self.n_active)
        loss = np.zeros(self.n_active)
        
        Te_guess = self._solve_Te_0D()
        k_chem = rates(Te_guess)
        
        for kk in range(self.n_active):
            i, j = self.flat_to_ij[kk]
            # Power source (W/m³ → eV/m³/s)
            P_local = self.P_abs * p_profile[i, j]
            ne_local = max(ne[i, j], 1e10)
            src[kk] = P_local / (1.5 * eC * ne_local)  # heating rate (eV/s)
            
            # Collisional loss rate coefficient (eV/s per Te)
            eps_loss = (k_chem['iz_SF6_total'] * 15.7 +  # ionisation
                       k_chem['att_SF6_total'] * 3.5 +    # attachment
                       k_chem['d1'] * 8.1 +                # dissociation thresholds
                       k_chem['d2'] * 13.4 +
                       k_chem['d3'] * 33.5)
            loss[kk] = -nSF6[i, j] * eps_loss / max(ne_local, 1e10) * ne_local
        
        # Solve: A_Te · Te + diag(loss) · Te = -src
        M = A_Te + sparse.diags(loss, format='csr')
        Te_flat = np.asarray(spsolve(M, -src), dtype=np.float64)
        Te_flat = np.clip(Te_flat, 0.5, 15.0)  # physical bounds
        
        Te_2d = self._flat_to_field(Te_flat, fill=1.0)
        return Te_2d
    
    def solve(self, n_iter=80, w=0.12, verbose=True):
        """Solve with the electron energy PDE for Te(r,z)."""
        t0 = time.time()
        m = self.mesh
        Nr, Nz = m.Nr, m.Nz
        
        if verbose:
            print(f"{'='*70}")
            print(f"  TEL Solver + Electron Energy PDE")
            print(f"{'='*70}")
            print(f"  {Nr}x{Nz}={Nr*Nz}, active={self.n_active}")
        
        # Initialise with 0D values
        Te = self._compute_Te_spatial(np.zeros((Nr, Nz)))
        ne, ne_avg = self._compute_ne(Te)
        nSF6 = np.where(self.inside, self.nSF6_feed * 0.5, 0.0)
        nF = np.where(self.inside, 1e18, 0.0)
        nSF5 = np.where(self.inside, self.nSF6_feed * 0.02, 0.0)
        
        D_F = np.where(self.inside, self.D_F, 0.0)
        D_SF6 = np.where(self.inside, self.D_SF6, 0.0)
        A_F = self._build_diffusion_matrix(D_F, self.gamma_map)
        A_SF6 = self._build_diffusion_matrix(D_SF6, {})
        
        if verbose:
            print(f"\n  {'It':>3s} {'Te_avg':>6s} {'Te_max':>6s} {'ne':>10s} "
                  f"{'F_icp':>10s} {'Fc':>10s} {'drop':>5s}")
        
        for it in range(n_iter):
            # SF6
            src_s = np.zeros(self.n_active)
            loss_s = np.zeros(self.n_active)
            for kk in range(self.n_active):
                i, j = self.flat_to_ij[kk]
                kc = rates(Te[i, j])
                kd = (kc['d1']+kc['d2']+kc['d3']+kc['d4']+kc['d5']+
                      kc['iz_SF6_total']+kc['att_SF6_total'])
                rep = 1.0 / self.tau_R
                z = m.zc[j]
                if z > self.z_apt_top + 0.7 * self.L_icp:
                    rep += (z-self.z_apt_top-0.7*self.L_icp)/(0.3*self.L_icp)/(self.tau_R*0.2)
                bc = self.bc_type[i, j]
                dr_dz = m.dr[i] if bc in (BC_QUARTZ, BC_AL_SIDE, BC_SHOULDER) else m.dz[j]
                Rw = wall_sf6_regeneration(bc, nF[i,j], nSF5[i,j], self.v_th_F, dr_dz)
                src_s[kk] = self.nSF6_feed * rep + Rw
                loss_s[kk] = -(ne[i,j] * kd + rep)
            x = np.asarray(spsolve(A_SF6 + sparse.diags(loss_s), -src_s), dtype=np.float64)
            nSF6 = (1-w)*nSF6 + w*self._flat_to_field(np.clip(x, 0.01*self.nSF6_feed, self.nSF6_feed), 0)
            
            # F
            src_f = np.zeros(self.n_active)
            loss_f = np.zeros(self.n_active)
            for kk in range(self.n_active):
                i, j = self.flat_to_ij[kk]
                kc = rates(Te[i, j])
                RF = ne[i,j]*nSF6[i,j]*(kc['d1']+kc['d2']*2+kc['d3']*3+kc['d4']*2+kc['d5']*3+kc.get('d7',0)+kc.get('d8',0))
                if self.frac_Ar > 0.01:
                    nArm = self.k_exc*ne[i,j]*self.nAr/max(self.k_izm*ne[i,j]+self.k_quench*self.nSF6_feed+1e-20, 1e-30)
                    RF += (self.k_Penn+self.k_Penn_d)*nArm*nSF6[i,j]
                src_f[kk] = RF
                bc = self.bc_type[i, j]
                dr_dz = m.dr[i] if bc in (BC_QUARTZ, BC_AL_SIDE, BC_SHOULDER) else m.dz[j]
                Rl = wall_F_loss(bc, nF[i,j], self.v_th_F, dr_dz)
                loss_f[kk] = -Rl / max(nF[i,j], 1e6)
            pump_wall = np.full(self.n_active, -1.0/self.tau_R) + loss_f
            y = np.asarray(spsolve(A_F + sparse.diags(pump_wall), -src_f), dtype=np.float64)
            nF = (1-w)*nF + w*self._flat_to_field(np.maximum(y, 0), 0)
            nSF5 = np.where(self.inside, nSF6*0.03, 0)
            
            # Te from energy PDE (new!)
            Te_new = self._solve_Te_energy_pde(ne, nSF6)
            Te = (1-w)*Te + w*Te_new
            
            # ne from power balance with new Te
            ne, ne_avg = self._compute_ne(Te)
            
            if verbose and it % max(n_iter//10, 1) == 0:
                Te_active = Te[self.inside]
                im = self.inside & (np.outer(m.rc,np.ones(Nz))<=self.R_icp) & (np.outer(np.ones(Nr),m.zc)>=self.z_apt_top)
                Fi = np.sum(nF[im]*m.vol[im])/max(np.sum(m.vol[im]),1e-30)
                Fa = nF[:,0][self.inside[:,0]]
                Fc = Fa[0] if len(Fa)>0 else 0
                Fe = Fa[-1] if len(Fa)>0 else 0
                d = (1-Fe/max(Fc,1e6))*100 if Fc>1e6 else 0
                print(f"  {it:3d} {np.mean(Te_active):6.2f} {np.max(Te_active):6.2f} "
                      f"{ne_avg*1e-6:10.2e} {Fi*1e-6:10.2e} {Fc*1e-6:10.2e} {d:4.0f}%")
        
        elapsed = time.time() - t0
        rw = m.rc[self.inside[:,0]]
        Fw = nF[:,0][self.inside[:,0]]
        Fc = Fw[0] if len(Fw)>0 else 0
        Fe = Fw[-1] if len(Fw)>0 else 0
        d = (1-Fe/max(Fc,1e6))*100 if Fc>1e6 else 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Done ({elapsed:.1f}s), drop={d:.0f}%")
            print(f"{'='*70}")
        
        return {
            'nF': nF, 'nSF6': nSF6, 'ne': ne, 'Te': Te,
            'ne_avg': ne_avg, 'inside': self.inside, 'bc_type': self.bc_type,
            'mesh': self.mesh, 'F_wafer': Fw*1e-6, 'r_wafer': rw,
            'F_drop_pct': d, 'elapsed': elapsed,
        }
