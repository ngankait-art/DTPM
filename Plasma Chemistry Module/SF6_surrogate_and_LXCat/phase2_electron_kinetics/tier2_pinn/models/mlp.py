"""
tier2_pinn/models/mlp.py
========================

Tier 2 MLP surrogate model definition.

This is a plain 3-layer MLP with GELU activations. Input is
(log10(E/N_in_Td), x_Ar) after z-score normalisation; output is
(Te_eff_eV, log10(k_att + eps), log10(k_iz + eps), log10(k_exc + eps),
log10(k_diss + eps)) also after z-score normalisation. The z-score
statistics are stored in the checkpoint alongside the weights, so
inference only needs the checkpoint file; no external config is
required.

Matches the architecture of the M5 supervised surrogate trained in
the upstream plasma-dtpm repository, whose checkpoint ships at
``tier2_pinn/weights/m5_surrogate.pt``.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPSurrogate(nn.Module):
    """3-layer MLP: 2 -> 96 -> 96 -> 96 -> 5 with GELU activations."""

    def __init__(self, hidden: int = 96, n_layers: int = 3,
                 in_dim: int = 2, out_dim: int = 5) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden] * n_layers + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_from_checkpoint(path) -> "MLPSurrogate":
    """Instantiate a surrogate and load weights from a Phase 2 checkpoint.

    The checkpoint is expected to be a dict with keys
    ``model_state`` (PyTorch state dict), ``config`` (training config),
    ``x_scaler`` (input mean/std), ``y_scaler`` (output mean/std), and
    ``output_names`` (list of column names).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MLPSurrogate(
        hidden=cfg.get("hidden", 96),
        n_layers=cfg.get("n_layers", 3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # attach scalers / metadata as attributes so callers can access them
    model.x_scaler = ckpt["x_scaler"]  # type: ignore[attr-defined]
    model.y_scaler = ckpt["y_scaler"]  # type: ignore[attr-defined]
    model.output_names = ckpt["output_names"]  # type: ignore[attr-defined]
    return model
