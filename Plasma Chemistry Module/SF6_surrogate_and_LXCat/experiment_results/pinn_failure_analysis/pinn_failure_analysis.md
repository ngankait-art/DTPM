# PINN Failure Analysis

## System
2D axisymmetric reaction-diffusion with 54-reaction SF6 chemistry in T-shaped ICP reactor.

## Results

| Training mode | Result | Loss reduction |
|---|---|---|
| Data-only (supervised) | **Converges** | 23,000x |
| PDE-constrained (PINN) | **Diverges** | N/A |

## Root causes of PINN failure

1. **Stiff Arrhenius chemistry** (Critical): Rate coefficients span 15+ orders of magnitude (k ~ exp(-E/Te)). Autograd gradients through these rates are numerically unstable.
2. **Second-derivative instability** (High): The diffusion operator requires d²n/dr² and d²n/dz². Second derivatives through neural networks amplify noise.
3. **Axis singularity (1/r term)** (Medium): Cylindrical coordinates have 1/r * dn/dr at r=0. Even with L'Hopital fix (replacing with d²n/dr²), the gradient is ill-conditioned near the axis.
4. **Energy equation coupling** (High): Te PDE is coupled to species through collisional energy loss. The coupled system creates circular gradient dependencies.
5. **Multi-scale loss competition** (Critical): PDE residual, BC loss, data loss, and energy loss operate at different scales. No single loss weighting allows all terms to converge simultaneously.

## Attempted fixes (all unsuccessful)

- L'Hopital rule for 1/r singularity at axis
- BC residual normalization by characteristic scales (D*N_ref/R_ref)
- Gradient clipping (max norm 1.0)
- Learning rate reduction (1e-3 to 1e-5)
- Curriculum training (data-only first, then add PDE)
- Loss weighting sweeps (PDE weight 0.001 to 1.0)
- Separate optimizers for different loss terms

## Conclusion

Physics-informed regularization (soft constraints) succeeds where physics-informed neural networks (hard PDE constraints) fail, for stiff multi-scale reaction-diffusion systems.
