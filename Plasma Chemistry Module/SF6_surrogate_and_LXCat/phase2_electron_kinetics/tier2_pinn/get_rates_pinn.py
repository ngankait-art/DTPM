"""
tier2_pinn/get_rates_pinn.py
=============================

Phase 2 electron-kinetics production API.

This is the callable interface the Phase 2 workplan (§4.4, "Build the
rate-coefficient query interface") asks for. The DTPM fluid solver
calls this function at each Picard iteration, once per active cell
(or in a vectorised batch over all cells), to replace its current
Arrhenius rate evaluation.

The surrogate wrapped here is the Tier 2 supervised MLP trained on
the Tier 1 BOLSIG+ lookup table at the (E/N, x_Ar) grid documented
in ``data/raw/bolsig_data.h5``. Training code lives at
``tier2_pinn/train_mlp.py``; pre-trained weights are at
``tier2_pinn/weights/m5_surrogate.pt``. The function below does
nothing but (i) load those weights once per process, (ii) apply
the input scaler, (iii) run a forward pass on a GPU-free batch,
and (iv) apply the output scaler and de-log the rate-coefficient
outputs.

Notes
-----
- Pressure is currently ignored. The Tier 1 BOLSIG+ grid was
  generated at 10 mTorr with a weak pressure-dependence assumption;
  the workplan §4.6 explicitly notes that "at our operating
  pressures (10-30 mTorr) and low ionisation fraction, the EEDF
  depends almost entirely on E/N and composition, with only weak
  pressure dependence" and permits the surrogate to drop pressure
  from its input dimension. The ``pressure_mTorr`` argument is
  retained in the signature so that DTPM callers do not need to
  change when a pressure-dependent surrogate is added in a future
  revision.
- The surrogate output names are ``Te_eff``, ``k_att_total``,
  ``k_iz_total``, ``k_exc``, ``k_diss``. These correspond to the
  aggregated BOLSIG+ channels in ``data/raw/bolsig_data.h5``. The
  workplan's six "key reactions" (G1, G7, G12, G24, G25, G29) can
  be recovered from these aggregates via the branching ratios
  tabulated in ``tier1_bolsig/outputs/rates_boltzmann_pure_SF6.csv``.

Usage
-----

    >>> from tier2_pinn.get_rates_pinn import get_rates_pinn
    >>> import numpy as np
    >>> E_over_N = np.array([10, 30, 50, 100, 300])
    >>> rates = get_rates_pinn(E_over_N, x_Ar=0.0, pressure_mTorr=10.0)
    >>> rates["k_iz"].shape
    (5,)
    >>> rates["Te_eff"]
    array([...])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .models import load_from_checkpoint


# Module-level cache so DTPM can call this function millions of times
# without paying the checkpoint-load cost on each call.
_MODEL_CACHE: Dict[str, torch.nn.Module] = {}

_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent / "weights" / "m5_surrogate.pt"
)


def _load_model(weights_path: Path) -> torch.nn.Module:
    key = str(weights_path.resolve())
    if key not in _MODEL_CACHE:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Tier 2 weights not found at {weights_path}. "
                "Run `python tier2_pinn/train_mlp.py` to train from scratch, "
                "or restore the checkpoint from the release bundle."
            )
        _MODEL_CACHE[key] = load_from_checkpoint(weights_path)
    return _MODEL_CACHE[key]


def get_rates_pinn(
    E_over_N,
    x_Ar: float = 0.0,
    pressure_mTorr: float = 10.0,
    weights_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Query the trained Tier 2 surrogate for rate coefficients.

    Parameters
    ----------
    E_over_N : array-like, shape (N,)
        Reduced electric field in Townsend (1 Td = 1e-21 V m^2).
        Can be a scalar or any array. Internally broadcast to 1D.
    x_Ar : float, optional
        Argon mole fraction in the SF6/Ar mixture, in [0, 0.9].
        Default 0.0 (pure SF6).
    pressure_mTorr : float, optional
        Chamber pressure in mTorr. Currently unused by the surrogate
        (see note in module docstring). Accepted for API stability.
    weights_path : Path, optional
        Override the default Tier 2 checkpoint location. Useful for
        swapping between the supervised (M5) and physics-informed (M6)
        surrogates.

    Returns
    -------
    dict with keys:

    - ``"Te_eff"``     : array (N,) of effective electron temperature (eV)
    - ``"k_iz"``       : array (N,) of total SF6 ionisation rate (m^3/s)
    - ``"k_att"``      : array (N,) of total SF6 attachment rate (m^3/s)
    - ``"k_diss"``     : array (N,) of aggregated SF6 dissociation rate (m^3/s)
    - ``"k_exc"``      : array (N,) of aggregated SF6 excitation rate (m^3/s)

    Raises
    ------
    FileNotFoundError
        If the Tier 2 weights are not found at ``weights_path`` or
        the default location.
    ValueError
        If ``x_Ar`` is outside [0, 0.9] or ``E_over_N`` has negative
        values.
    """
    path = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    model = _load_model(path)

    EN = np.atleast_1d(np.asarray(E_over_N, dtype=np.float64))
    if np.any(EN <= 0):
        raise ValueError(
            "E_over_N must be strictly positive (got non-positive values)"
        )
    if not 0.0 <= x_Ar <= 0.9:
        raise ValueError(f"x_Ar must be in [0, 0.9] (got {x_Ar})")
    _ = pressure_mTorr  # intentionally unused; see docstring

    # Surrogate input: (log10(E/N), x_Ar), z-score normalised
    x_raw = np.stack([np.log10(EN), np.full_like(EN, x_Ar)], axis=1)
    x_mean = np.asarray(model.x_scaler["mean"])
    x_std = np.asarray(model.x_scaler["std"])
    x_norm = (x_raw - x_mean) / x_std

    # Forward pass
    with torch.no_grad():
        y_norm = model(torch.from_numpy(x_norm).float()).numpy()

    # De-normalise
    y_mean = np.asarray(model.y_scaler["mean"])
    y_std = np.asarray(model.y_scaler["std"])
    y_raw = y_norm * y_std + y_mean

    # Output layout: Te_eff (linear), then four rates in
    # log10(k + 1e-22) space. See tier2_pinn/README.md for the
    # derivation of the additive floor.
    names = list(model.output_names)
    te_idx = names.index("Te_eff")
    Te_eff = y_raw[:, te_idx]

    def _rate(name: str) -> np.ndarray:
        idx = names.index(name)
        return 10.0 ** y_raw[:, idx] - 1e-22

    result = {
        "Te_eff": Te_eff,
        "k_att": _rate("k_att_total"),
        "k_iz":  _rate("k_iz_total"),
        "k_exc": _rate("k_exc"),
        "k_diss": _rate("k_diss"),
    }
    # Enforce non-negativity (small numerical undershoot near log floor)
    for k in ("k_att", "k_iz", "k_exc", "k_diss"):
        result[k] = np.clip(result[k], 0.0, None)
    return result


def _example() -> None:
    """Runnable example — `python -m tier2_pinn.get_rates_pinn`."""
    E_over_N = np.array([10.0, 30.0, 50.0, 100.0, 300.0, 1000.0])
    rates = get_rates_pinn(E_over_N, x_Ar=0.0, pressure_mTorr=10.0)
    print("# Tier 2 surrogate inference at pure-SF6, 10 mTorr")
    header = ["E/N (Td)", "Te_eff (eV)", "k_iz (m3/s)",
              "k_att (m3/s)", "k_diss (m3/s)", "k_exc (m3/s)"]
    print("  ".join(f"{h:>14s}" for h in header))
    for i, en in enumerate(E_over_N):
        print(
            f"  {en:>14.3g}"
            f"  {rates['Te_eff'][i]:>14.4f}"
            f"  {rates['k_iz'][i]:>14.3e}"
            f"  {rates['k_att'][i]:>14.3e}"
            f"  {rates['k_diss'][i]:>14.3e}"
            f"  {rates['k_exc'][i]:>14.3e}"
        )


if __name__ == "__main__":
    _example()
