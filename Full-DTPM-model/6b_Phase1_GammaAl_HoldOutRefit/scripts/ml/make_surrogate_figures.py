#!/usr/bin/env python3
"""Regenerate the §10 surrogate diagnostic figures from the production
ensemble checkpoints.

Produces 7 PDF/PNG pairs into the report's figure directory plus a
small inference-timing JSON next to the ensemble checkpoints.

Usage:
    cd Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit
    python scripts/ml/make_surrogate_figures.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIXB_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIXB_ROOT, "..", ".."))
FIG_DIR = os.path.join(
    REPO_ROOT,
    "Plasma Chemistry Module",
    "SF6_surrogate_and_LXCat",
    "figures",
)
ENS_DIR = os.path.join(SIXB_ROOT, "results", "ml_production_ensemble_lxcat")
DATASET_DIR = os.path.join(SIXB_ROOT, "results", "ml_dataset", "lxcat")

sys.path.insert(0, HERE)
import ml_dataset_loader as mdl  # noqa: E402
from train_ensemble import SeparateHeadsMLP, to_tensors  # noqa: E402

# Solver per-eval mean times (s) reported in §10.7 of the report.
SOLVER_LEGACY_S = 7.76
SOLVER_LXCAT_S = 14.4


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────
def load_summary():
    with open(os.path.join(ENS_DIR, "summary.json")) as f:
        return json.load(f)


def load_ensemble(device="cpu"):
    models = []
    for i in range(5):
        m = SeparateHeadsMLP(n_in=5).to(device)
        sd = torch.load(
            os.path.join(ENS_DIR, f"model_{i}.pt"),
            map_location=device,
            weights_only=True,
        )
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


def ensemble_predict(models, X):
    with torch.no_grad():
        preds = torch.stack([m(X) for m in models])  # (M, N, 2)
    mean = preds.mean(0)
    std = preds.std(0)
    return preds, mean, std


def load_data():
    train_d, val_d, meta, val_idx = mdl.load_dataset(
        mode="lxcat", val_frac=0.15, enhanced_features=False
    )
    return train_d, val_d, meta, val_idx


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────
def fig_pred_vs_true(Yv_np, mean_np):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    species = [("$n_\\mathrm{F}$", 0), ("$n_{\\mathrm{SF}_6}$", 1)]
    for ax, (label, c) in zip(axes, species):
        x = Yv_np[:, c]
        y = mean_np[:, c]
        ax.scatter(x, y, s=2, alpha=0.25, color="#1f77b4", rasterized=True)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ideal")
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        ax.set_xlabel(f"True log$_{{10}}$ {label}")
        ax.set_ylabel(f"Predicted log$_{{10}}$ {label}")
        ax.set_title(f"{label} — ensemble RMSE = {rmse:.5f}")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save(fig, "surrogate_v4_pred_vs_true")


def fig_per_model_val_loss(summary):
    vals = np.array(summary["ensemble_vals"])
    mean = float(summary["ens_mean"])
    std = float(summary["ens_std"])
    labels = [f"M{i}" for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(
        labels, vals, color="#1f77b4", edgecolor="black", linewidth=0.5
    )
    ax.axhline(mean, color="#d62728", linestyle="--", lw=1.2,
               label=f"mean = {mean:.2e}")
    ax.axhspan(mean - std, mean + std, color="#d62728", alpha=0.15,
               label=f"±σ = {std:.2e}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2e}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Best validation loss")
    ax.set_title("Per-model best validation loss across 5-model ensemble")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ymax = float(max(vals.max(), mean + std)) * 1.25
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    save(fig, "surrogate_v4_per_model_val_loss")


def fig_uncertainty(Yv_np, mean_np, std_np):
    err = np.abs(mean_np - Yv_np)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, c, label in zip(axes, [0, 1],
                            ["$n_\\mathrm{F}$", "$n_{\\mathrm{SF}_6}$"]):
        s = std_np[:, c]
        e = err[:, c]
        ax.scatter(s, e, s=2, alpha=0.25, color="#2ca02c", rasterized=True)
        hi = float(max(s.max(), e.max()))
        ax.plot([0, hi], [0, hi], "k--", lw=0.8,
                label="ideal calibration (|err|=σ)")
        ax.set_xlabel(f"Ensemble σ ({label})")
        ax.set_ylabel(f"|prediction error| ({label})")
        rho = float(np.corrcoef(s, e)[0, 1]) if s.std() > 0 else float("nan")
        ax.set_title(f"{label} — Pearson r = {rho:.2f}")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "surrogate_v4_uncertainty")


def fig_residuals_spatial(meta, Yv_np, mean_np, val_d, std_np):
    """Spatial residuals + per-species diagnostics figures."""
    rep_case_id = "P0700W_p10mT_xAr000"
    rep_idx = next((m["idx"] for m in meta if m["case_id"] == rep_case_id), None)

    if rep_idx is None:
        print(f"  WARN: {rep_case_id} not in meta; using first val case")
        case_arr = np.array(val_d["case"])
        rep_idx = int(case_arr[0])

    case_arr = np.array(val_d["case"])
    sel = case_arr == rep_idx
    if not sel.any():
        print(f"  WARN: case idx {rep_idx} not in val split; using first val case")
        rep_idx = int(case_arr[0])
        sel = case_arr == rep_idx

    r_sel = np.array(val_d["r"])[sel]
    z_sel = np.array(val_d["z"])[sel]
    err = mean_np[sel] - Yv_np[sel]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(r_sel * 100, z_sel * 100, c=np.abs(err[:, 0]),
                    cmap="viridis", s=8, marker="s", rasterized=True)
    plt.colorbar(sc, ax=ax, label="|residual| log$_{10}$ $n_\\mathrm{F}$")
    ax.set_xlabel("r (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title(f"Spatial residual map — {rep_case_id} (case idx {rep_idx})")
    fig.tight_layout()
    save(fig, "residuals")

    species = [("nF", 0, "$n_\\mathrm{F}$"),
               ("nSF6", 1, "$n_{\\mathrm{SF}_6}$")]
    case_arr_full = np.array(val_d["case"])
    P_arr = np.array(val_d["P"])
    p_arr = np.array(val_d["p"])
    for tag, c, label in species:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
        all_err = mean_np[:, c] - Yv_np[:, c]
        axes[0].hist(all_err, bins=80, color="#1f77b4",
                     edgecolor="black", linewidth=0.3)
        axes[0].set_xlabel(f"residual log$_{{10}}$ {label}")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"Residual histogram (val set, all cases)")
        axes[0].grid(alpha=0.3, linestyle=":")
        axes[0].axvline(0, color="k", linestyle="--", lw=0.8)

        sc = axes[1].scatter(r_sel * 100, z_sel * 100,
                             c=np.abs(err[:, c]), cmap="viridis",
                             s=8, marker="s", rasterized=True)
        plt.colorbar(sc, ax=axes[1], label=f"|residual| log$_{{10}}$ {label}")
        axes[1].set_xlabel("r (cm)")
        axes[1].set_ylabel("z (cm)")
        axes[1].set_title(f"Spatial map — {rep_case_id}")

        unique_cases = np.unique(case_arr_full)
        per_case_rmse = []
        per_case_P = []
        for cidx in unique_cases:
            cmask = case_arr_full == cidx
            if not cmask.any():
                continue
            per_case_rmse.append(
                float(np.sqrt(np.mean(all_err[cmask] ** 2))))
            per_case_P.append(float(P_arr[cmask].mean()))
        axes[2].scatter(per_case_P, per_case_rmse,
                        s=18, color="#d62728", edgecolor="black",
                        linewidth=0.5, alpha=0.8)
        axes[2].set_xlabel("$P_\\mathrm{rf}$ (W)")
        axes[2].set_ylabel(f"per-case RMSE ({label})")
        axes[2].set_title("Per-condition RMSE vs power")
        axes[2].grid(alpha=0.3, linestyle=":")

        fig.tight_layout()
        save(fig, f"surrogate_v4_diagnostics_{tag}")


def fig_surrogate_vs_solver(models, summary):
    """Overlay solver radial F at the wafer with surrogate ±1σ band."""
    rep_case_id = "P0700W_p10mT_xAr000"
    case_dir = os.path.join(DATASET_DIR, rep_case_id)
    nF_path = os.path.join(case_dir, "nF.npy")
    case_summary = os.path.join(case_dir, "summary.json")
    if not (os.path.exists(nF_path) and os.path.exists(case_summary)):
        print(f"  WARN: {rep_case_id} not found; skipping overlay")
        return

    rc, zc, inside = mdl.get_mesh()
    nF = np.load(nF_path)
    with open(case_summary) as f:
        s = json.load(f)
    P_val = float(s.get("P_rf_W", s.get("P_rf", 700.0)))
    p_val = float(s.get("p_mTorr", 10.0))
    Ar_val = float(s.get("x_Ar", s.get("frac_Ar", 0.0)))

    z_wafer_idx = int(np.argmin(np.abs(zc - 0.0)))
    if not inside[:, z_wafer_idx].any():
        z_wafer_idx = int(np.argmin(zc[inside.any(axis=0)]))
    r_in = rc[inside[:, z_wafer_idx]]
    nF_solver = nF[inside[:, z_wafer_idx], z_wafer_idx]
    log_solver = np.log10(np.clip(nF_solver, 1e6, None))

    pts = np.column_stack([
        r_in,
        np.full_like(r_in, zc[z_wafer_idx]),
        np.full_like(r_in, P_val),
        np.full_like(r_in, p_val),
        np.full_like(r_in, Ar_val),
    ]).astype(np.float32)
    X = torch.from_numpy(pts)
    with torch.no_grad():
        preds = torch.stack([m(X) for m in models])
    mean = preds.mean(0).numpy()
    std = preds.std(0).numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(r_in * 100, log_solver, "k-", lw=1.6, label="Solver")
    ax.plot(r_in * 100, mean[:, 0], color="#d62728", lw=1.4,
            label="Surrogate ensemble mean")
    ax.fill_between(r_in * 100, mean[:, 0] - std[:, 0],
                    mean[:, 0] + std[:, 0], color="#d62728", alpha=0.25,
                    label="±1σ")
    ax.set_xlabel("r (cm)")
    ax.set_ylabel("log$_{10}$ $n_\\mathrm{F}$ at wafer")
    ax.set_title(f"Surrogate vs. solver — {rep_case_id}  "
                 f"(P={P_val:.0f} W, p={p_val:.0f} mTorr, x$_\\mathrm{{Ar}}$={Ar_val:.2f})")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "surrogate_vs_solver_overlay")


# ─────────────────────────────────────────────────────────────────────
# Inference timing
# ─────────────────────────────────────────────────────────────────────
def time_inference(models, val_d):
    """Time a single operating-point evaluation: ensemble inference on the
    inside-mask cells of one case. Mirrors how the surrogate is used in
    practice (one (P_rf, p, x_Ar) tuple → full-mesh density field), which
    is what the §10.7 solver-vs-surrogate ratio compares against."""
    case_arr = np.array(val_d["case"])
    counts = np.bincount(case_arr)
    rep_idx = int(np.argmax(counts))
    sel = case_arr == rep_idx
    r_s = np.array(val_d["r"])[sel].astype(np.float32)
    z_s = np.array(val_d["z"])[sel].astype(np.float32)
    P_s = np.array(val_d["P"])[sel].astype(np.float32)
    p_s = np.array(val_d["p"])[sel].astype(np.float32)
    Ar_s = np.array(val_d["Ar"])[sel].astype(np.float32)
    R_PROC = 0.105
    Z_TOP = 0.234
    X = torch.from_numpy(np.column_stack([
        r_s / R_PROC, z_s / Z_TOP, P_s / 1200, p_s / 20, Ar_s,
    ])).float()

    n_warm = 20
    n_runs = 200
    with torch.no_grad():
        for _ in range(n_warm):
            for m in models:
                _ = m(X)
        wall = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for m in models:
                _ = m(X)
            wall.append(time.perf_counter() - t0)
    arr = np.array(wall)
    out = {
        "n_passes": int(n_runs),
        "n_warmup": int(n_warm),
        "ensemble_size": len(models),
        "batch_size": int(X.shape[0]),
        "batch_definition": "single representative operating point "
                            "(inside-mask cells of one validation case)",
        "ms_per_eval_mean": float(arr.mean() * 1000),
        "ms_per_eval_std": float(arr.std() * 1000),
        "eval_machine_id": socket.gethostname(),
        "speedup_vs_legacy_solver": float(SOLVER_LEGACY_S / arr.mean()),
        "speedup_vs_lxcat_solver": float(SOLVER_LXCAT_S / arr.mean()),
    }
    out_path = os.path.join(ENS_DIR, "inference_timing.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote {out_path}")
    print(f"  ms/eval: {out['ms_per_eval_mean']:.2f} ± {out['ms_per_eval_std']:.2f}  "
          f"(batch_size = {out['batch_size']} cells)")
    print(f"  speedup vs legacy solver: {out['speedup_vs_legacy_solver']:.0f}×")
    print(f"  speedup vs LXCat solver:  {out['speedup_vs_lxcat_solver']:.0f}×")
    return out


# ─────────────────────────────────────────────────────────────────────
def save(fig, stem):
    pdf_path = os.path.join(FIG_DIR, f"{stem}.pdf")
    png_path = os.path.join(FIG_DIR, f"{stem}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {pdf_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary = load_summary()
    print(f"Loaded summary: nF RMSE = {summary['metrics']['nF']['rmse']:.5f}")

    print("Loading dataset (lxcat)…")
    train_d, val_d, meta, val_idx = load_data()
    print(f"  train pts: {len(train_d['r'])}; val pts: {len(val_d['r'])}; "
          f"cases: {len(meta)}; val cases: {len(val_idx)}")

    print("Loading ensemble checkpoints…")
    models = load_ensemble(device="cpu")

    Xv, Yv = to_tensors(val_d, device="cpu", enhanced=False)
    _, mean, std = ensemble_predict(models, Xv)
    Yv_np = Yv.numpy()
    mean_np = mean.numpy()
    std_np = std.numpy()

    print("Generating figures…")
    fig_pred_vs_true(Yv_np, mean_np)
    fig_per_model_val_loss(summary)
    fig_uncertainty(Yv_np, mean_np, std_np)
    fig_residuals_spatial(meta, Yv_np, mean_np, val_d, std_np)
    fig_surrogate_vs_solver(models, summary)

    print("Timing inference…")
    time_inference(models, val_d)

    print("Done.")


if __name__ == "__main__":
    main()
