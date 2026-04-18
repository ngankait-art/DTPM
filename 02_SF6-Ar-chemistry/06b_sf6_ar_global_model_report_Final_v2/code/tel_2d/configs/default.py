"""
TEL ICP Reactor Model Configuration
====================================
All parameters in one place. Import and override as needed.
"""

# ── Geometry (TEL reactor) ──
geometry = dict(
    R_icp    = 0.038,    # m, ICP source radius
    R_proc   = 0.105,    # m, processing region radius
    L_icp    = 0.1815,   # m, ICP source height
    L_proc   = 0.050,    # m, processing region height
    L_apt    = 0.002,    # m, aperture height
    R_wafer  = 0.075,    # m, wafer radius (150 mm diameter)
)

# ── Operating conditions ──
operating = dict(
    P_rf     = 700,      # W, RF power
    p_mTorr  = 10,       # mTorr, pressure
    frac_Ar  = 0.0,      # Ar fraction (0 = pure SF6)
    Q_sccm   = 100,      # sccm, total gas flow
    Tgas     = 313,      # K, gas temperature
    f_rf     = 2e6,      # Hz, RF frequency
)

# ── Wall recombination probabilities ──
# Kokkoris 2009 literature values (uncalibrated)
gamma_literature = dict(
    quartz_F    = 0.001,
    aluminium_F = 0.015,   # Kokkoris beam-surface value
    silicon_F   = 0.025,
    window_F    = 0.001,
)

# Calibrated values (fitted to Mettler 74% drop)
gamma_calibrated = dict(
    quartz_F    = 0.001,
    aluminium_F = 0.18,    # Calibrated effective value
    silicon_F   = 0.025,
    window_F    = 0.001,
)

# ── Solver settings ──
solver = dict(
    Nr       = 50,
    Nz       = 80,
    beta_r   = 1.2,
    beta_z   = 1.0,
    n_iter   = 70,
    omega    = 0.12,      # relaxation factor
    eta      = 0.43,      # power coupling efficiency
)

# ── Model definitions ──
models = dict(
    A = dict(
        name        = "Model A: Calibrated 2-species",
        description = "F + SF6 transport, gamma_Al=0.18 (calibrated to 74% drop)",
        species     = ['F', 'SF6'],
        gamma_set   = 'calibrated',
        transport   = '2D_PDE',
        purpose     = "Baseline / proof-of-concept",
    ),
    B = dict(
        name        = "Model B: Hybrid multi-species (calibrated)",
        description = "F+SF6 2D transport + 7 intermediates local, gamma_Al=0.18",
        species     = ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2'],
        gamma_set   = 'calibrated',
        transport   = 'hybrid',
        purpose     = "Chemistry distribution with calibrated backbone",
    ),
    C = dict(
        name        = "Model C: Hybrid multi-species (uncalibrated)",
        description = "Same as B but gamma_Al=0.015 (Kokkoris literature)",
        species     = ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2'],
        gamma_set   = 'literature',
        transport   = 'hybrid',
        purpose     = "Prediction without calibration; quantify gamma gap",
    ),
    D = dict(
        name        = "Model D: Full multi-species transport (future)",
        description = "All 9 species solved via 2D PDE transport",
        species     = ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2'],
        gamma_set   = 'literature',
        transport   = '2D_PDE_all',
        purpose     = "Target architecture for publication",
    ),
)
