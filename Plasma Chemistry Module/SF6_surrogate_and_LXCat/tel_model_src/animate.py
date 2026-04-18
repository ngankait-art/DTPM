"""
Animation module for TEL ICP reactor simulations.
Generates convergence animations showing pseudo-time evolution to steady state.
"""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_convergence(solver_class, output_path='animations/convergence.gif',
                        Nr=40, Nz=60, n_frames=20, fps=4):
    """Animate the iterative convergence of the [F] field."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    
    s = solver_class(Nr=Nr, Nz=Nz)
    m = s.mesh
    
    # Collect snapshots at different iteration counts
    snapshots = []
    drops = []
    for n_iter in np.linspace(1, 70, n_frames).astype(int):
        r = s.solve(n_iter=int(n_iter), w=0.12, verbose=False)
        nF_cm3 = r['nF'] * 1e-6
        F_log = np.full_like(nF_cm3, np.nan)
        mask = r['inside'] & (nF_cm3 > 0)
        F_log[mask] = np.log10(nF_cm3[mask])
        snapshots.append(F_log.copy())
        drops.append(r['F_drop_pct'])
    
    # Create animation
    Ri = s.R_icp*1000; Rp = s.R_proc*1000
    zAb = s.z_apt_bot*1000; zAt = s.z_apt_top*1000; zT = s.z_top*1000
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    plt.rcParams.update({'figure.facecolor':'white','savefig.facecolor':'white'})
    
    norm = Normalize(vmin=12, vmax=14.8)
    cmap = plt.cm.inferno.copy()
    
    # Mirror the field for display
    def mirror(fld):
        rl = -m.rc[1:][::-1]*1000
        rf = np.concatenate([rl, m.rc*1000])
        ff = np.concatenate([fld[1:,:][::-1,:], fld], axis=0)
        return rf, ff
    
    rf, _ = mirror(snapshots[0])
    
    # Initial frame
    _, ff0 = mirror(snapshots[0])
    rgba0 = cmap(norm(np.nan_to_num(ff0, nan=0)))
    rgba0[np.isnan(ff0), 3] = 0
    
    im = ax1.imshow(rgba0.transpose(1,0,2)[::-1], origin='lower', aspect='auto',
                     extent=[rf[0],rf[-1],0,zT], interpolation='bilinear')
    ax1.set_xlabel('$r$ (mm)'); ax1.set_ylabel('$z$ (mm)')
    title = ax1.set_title('Iteration 1', fontweight='bold')
    
    # Radial profile panel
    line_F, = ax2.plot([], [], 'r-', lw=3)
    ax2.set_xlim(0, Rp+5); ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('$r$ (mm)'); ax2.set_ylabel('[F] / [F]$_{r=0}$')
    ax2.grid(alpha=0.2, ls=':')
    drop_text = ax2.text(60, 0.9, '', fontsize=14, fontweight='bold')
    
    iters = np.linspace(1, 70, n_frames).astype(int)
    
    def update(frame):
        fld = snapshots[frame]
        _, ff = mirror(fld)
        rgba = cmap(norm(np.nan_to_num(ff, nan=0)))
        rgba[np.isnan(ff), 3] = 0
        im.set_data(rgba.transpose(1,0,2)[::-1])
        title.set_text(f'Iteration {iters[frame]}')
        
        # Radial profile
        F_wafer = 10**np.nan_to_num(fld[:, 0], nan=0)
        F_norm = F_wafer / max(F_wafer[0], 1e-10)
        line_F.set_data(m.rc*1000, F_norm)
        drop_text.set_text(f'Drop = {drops[frame]:.0f}%')
        ax2.set_title(f'Radial Profile at Wafer', fontweight='bold')
        return im, line_F, title, drop_text
    
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Animation saved: {output_path}")
    
    # Save key frames as PNG
    for i in [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]:
        fig2, ax = plt.subplots(figsize=(8, 10))
        fld = snapshots[i]; _, ff = mirror(fld)
        rgba = cmap(norm(np.nan_to_num(ff, nan=0))); rgba[np.isnan(ff),3] = 0
        ax.imshow(rgba.transpose(1,0,2)[::-1], origin='lower', aspect='auto',
                  extent=[rf[0],rf[-1],0,zT], interpolation='bilinear')
        ax.set_title(f'Iteration {iters[i]}, Drop = {drops[i]:.0f}%', fontweight='bold')
        ax.set_xlabel('$r$ (mm)'); ax.set_ylabel('$z$ (mm)')
        fig2.savefig(f'animations/frame_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Key frames saved: animations/frame_*.png")


if __name__ == '__main__':
    from solver import TELSolver
    animate_convergence(TELSolver, 'animations/convergence.gif', Nr=35, Nz=55, n_frames=15)
