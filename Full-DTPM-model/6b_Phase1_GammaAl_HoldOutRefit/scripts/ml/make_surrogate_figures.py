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


def _load_case_field(case_id, field):
    p = os.path.join(DATASET_DIR, case_id, f"{field}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def _vol_avg_inside(field, inside):
    """Volume-average a 2D (Nr,Nz) field over the inside-mask cells."""
    return float(np.mean(field[inside]))


def fig_surrogate_vs_solver_neutrals_benchmark(models):
    """Surrogate vs solver volume-averaged density vs (power, pressure) for
    the two species the surrogate predicts.  Replacement for Figure 15 of
    the original v1 report — same visual style, but the legend is explicit
    about which curve is solver and which is surrogate."""
    rc, zc, inside = mdl.get_mesh()

    powers = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    pressures = [3.0, 5.0, 10.0, 15.0, 20.0]
    P_FIXED = 10.0
    P_FIXED_W = 700

    def case_id(P, p, ar=0.0):
        return f"P{int(P):04d}W_p{int(p):02d}mT_xAr{int(round(ar*100)):03d}"

    def predict_avg(P, p, ar=0.0):
        Nr, Nz = inside.shape
        rg, zg = np.meshgrid(rc, zc, indexing="ij")
        r_in = rg[inside].astype(np.float32)
        z_in = zg[inside].astype(np.float32)
        n = r_in.size
        R_PROC = 0.105
        Z_TOP = 0.234
        X = torch.from_numpy(np.column_stack([
            r_in / R_PROC, z_in / Z_TOP,
            np.full(n, P / 1200, dtype=np.float32),
            np.full(n, p / 20, dtype=np.float32),
            np.full(n, ar, dtype=np.float32),
        ]))
        with torch.no_grad():
            preds = torch.stack([m(X) for m in models]).numpy()
        # Predicted log10 density per ensemble member; convert to linear,
        # volume-average, then log10 again so we plot in same units as solver.
        lin = 10 ** preds
        per_member_avg = lin.mean(axis=1)  # (M, 2)
        ens_log = np.log10(np.clip(per_member_avg, 1.0, None))
        return ens_log.mean(axis=0), ens_log.std(axis=0)

    def solver_avg(case, field):
        f = _load_case_field(case, field)
        if f is None:
            return np.nan
        f = np.clip(f, 1.0, None)
        return np.log10(_vol_avg_inside(f, inside))

    # Power sweep at p=10 mTorr, pure SF6
    P_arr = []
    sol_nF = []; sol_nSF6 = []
    sur_nF = []; sur_nSF6 = []
    sur_nF_sd = []; sur_nSF6_sd = []
    for P in powers:
        cid = case_id(P, P_FIXED, 0.0)
        sol_nF.append(solver_avg(cid, "nF"))
        sol_nSF6.append(solver_avg(cid, "nSF6"))
        m, s = predict_avg(P, P_FIXED, 0.0)
        sur_nF.append(m[0]);  sur_nF_sd.append(s[0])
        sur_nSF6.append(m[1]); sur_nSF6_sd.append(s[1])
        P_arr.append(P)
    P_arr = np.array(P_arr)
    sol_nF = np.array(sol_nF); sol_nSF6 = np.array(sol_nSF6)
    sur_nF = np.array(sur_nF); sur_nSF6 = np.array(sur_nSF6)
    sur_nF_sd = np.array(sur_nF_sd); sur_nSF6_sd = np.array(sur_nSF6_sd)

    # Pressure sweep at P=700W, pure SF6
    p_arr = []
    psol_nF = []; psol_nSF6 = []
    psur_nF = []; psur_nSF6 = []
    psur_nF_sd = []; psur_nSF6_sd = []
    for p in pressures:
        cid = case_id(P_FIXED_W, p, 0.0)
        psol_nF.append(solver_avg(cid, "nF"))
        psol_nSF6.append(solver_avg(cid, "nSF6"))
        m, s = predict_avg(P_FIXED_W, p, 0.0)
        psur_nF.append(m[0]);  psur_nF_sd.append(s[0])
        psur_nSF6.append(m[1]); psur_nSF6_sd.append(s[1])
        p_arr.append(p)
    p_arr = np.array(p_arr)
    psol_nF = np.array(psol_nF); psol_nSF6 = np.array(psol_nSF6)
    psur_nF = np.array(psur_nF); psur_nSF6 = np.array(psur_nSF6)
    psur_nF_sd = np.array(psur_nF_sd); psur_nSF6_sd = np.array(psur_nSF6_sd)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    cF = "#d62728"
    cSF6 = "#1f77b4"

    ax = axes[0]
    ax.plot(P_arr, sol_nF, "-", color=cF, lw=2.0, label=r"$n_\mathrm{F}$ — solver")
    ax.plot(P_arr, sur_nF, "--", color=cF, lw=1.6,
            marker="o", ms=4, label=r"$n_\mathrm{F}$ — surrogate (mean)")
    ax.fill_between(P_arr, sur_nF - sur_nF_sd, sur_nF + sur_nF_sd,
                    color=cF, alpha=0.20, label=r"$n_\mathrm{F}$ ensemble $\pm\sigma$")
    ax.plot(P_arr, sol_nSF6, "-", color=cSF6, lw=2.0,
            label=r"$n_{\mathrm{SF}_6}$ — solver")
    ax.plot(P_arr, sur_nSF6, "--", color=cSF6, lw=1.6,
            marker="s", ms=4, label=r"$n_{\mathrm{SF}_6}$ — surrogate (mean)")
    ax.fill_between(P_arr, sur_nSF6 - sur_nSF6_sd, sur_nSF6 + sur_nSF6_sd,
                    color=cSF6, alpha=0.20,
                    label=r"$n_{\mathrm{SF}_6}$ ensemble $\pm\sigma$")
    ax.set_xlabel("RF power (W)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ volume-averaged density (cm$^{-3}$)", fontsize=12)
    ax.set_title("(a) Power sweep — p = 10 mTorr, pure SF$_6$", fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    ax.plot(p_arr, psol_nF, "-", color=cF, lw=2.0, label=r"$n_\mathrm{F}$ — solver")
    ax.plot(p_arr, psur_nF, "--", color=cF, lw=1.6,
            marker="o", ms=4, label=r"$n_\mathrm{F}$ — surrogate (mean)")
    ax.fill_between(p_arr, psur_nF - psur_nF_sd, psur_nF + psur_nF_sd,
                    color=cF, alpha=0.20, label=r"$n_\mathrm{F}$ ensemble $\pm\sigma$")
    ax.plot(p_arr, psol_nSF6, "-", color=cSF6, lw=2.0,
            label=r"$n_{\mathrm{SF}_6}$ — solver")
    ax.plot(p_arr, psur_nSF6, "--", color=cSF6, lw=1.6,
            marker="s", ms=4, label=r"$n_{\mathrm{SF}_6}$ — surrogate (mean)")
    ax.fill_between(p_arr, psur_nSF6 - psur_nSF6_sd, psur_nSF6 + psur_nSF6_sd,
                    color=cSF6, alpha=0.20,
                    label=r"$n_{\mathrm{SF}_6}$ ensemble $\pm\sigma$")
    ax.set_xlabel("Gas pressure (mTorr)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ volume-averaged density (cm$^{-3}$)", fontsize=12)
    ax.set_title(f"(b) Pressure sweep — P$_\\mathrm{{rf}}$ = {P_FIXED_W} W, pure SF$_6$",
                 fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
    ax.tick_params(labelsize=10)

    fig.suptitle("Surrogate vs. solver benchmark — volume-averaged densities",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "surrogate_vs_solver_neutrals_benchmark")

    # ─── Redesigned figure: absolute values + residual subpanels ──────────
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.2, 1.0],
        hspace=0.05, wspace=0.22,
        left=0.07, right=0.98, bottom=0.08, top=0.93,
    )
    axTL = fig.add_subplot(gs[0, 0])  # absolute, power
    axTR = fig.add_subplot(gs[0, 1])  # absolute, pressure
    axBL = fig.add_subplot(gs[1, 0], sharex=axTL)  # residual, power
    axBR = fig.add_subplot(gs[1, 1], sharex=axTR)  # residual, pressure

    def _abs_panel(ax, x, sol_F, sur_F, sd_F, sol_SF6, sur_SF6, sd_SF6,
                   xlabel, title):
        ax.plot(x, sol_F, "-",  color=cF, lw=2.0,
                label=r"$n_\mathrm{F}$ — solver")
        ax.plot(x, sur_F, "--", color=cF, lw=1.6, marker="o", ms=4,
                label=r"$n_\mathrm{F}$ — surrogate")
        ax.fill_between(x, sur_F - sd_F, sur_F + sd_F, color=cF, alpha=0.2)
        ax.plot(x, sol_SF6, "-",  color=cSF6, lw=2.0,
                label=r"$n_{\mathrm{SF}_6}$ — solver")
        ax.plot(x, sur_SF6, "--", color=cSF6, lw=1.6, marker="s", ms=4,
                label=r"$n_{\mathrm{SF}_6}$ — surrogate")
        ax.fill_between(x, sur_SF6 - sd_SF6, sur_SF6 + sd_SF6,
                        color=cSF6, alpha=0.2)
        ax.set_ylabel(r"$\log_{10}$ vol-avg density (cm$^{-3}$)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
        ax.tick_params(labelsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)

    def _res_panel(ax, x, sol_F, sur_F, sd_F, sol_SF6, sur_SF6, sd_SF6,
                   xlabel):
        rF = sur_F - sol_F
        rSF6 = sur_SF6 - sol_SF6
        ax.axhline(0, color="k", lw=0.8, linestyle="-")
        ax.plot(x, rF, "--o", color=cF, lw=1.4, ms=4,
                label=r"$n_\mathrm{F}$ residual")
        ax.fill_between(x, rF - sd_F, rF + sd_F, color=cF, alpha=0.2)
        ax.plot(x, rSF6, "--s", color=cSF6, lw=1.4, ms=4,
                label=r"$n_{\mathrm{SF}_6}$ residual")
        ax.fill_between(x, rSF6 - sd_SF6, rSF6 + sd_SF6, color=cSF6, alpha=0.2)
        ymax = max(np.abs(rF).max(), np.abs(rSF6).max(),
                   sd_F.max(), sd_SF6.max()) * 1.4 + 0.005
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r"surrogate $-$ solver", fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
        ax.tick_params(labelsize=10)

    _abs_panel(axTL, P_arr, sol_nF, sur_nF, sur_nF_sd,
               sol_nSF6, sur_nSF6, sur_nSF6_sd,
               "RF power (W)",
               "(a) Power sweep — p = 10 mTorr, pure SF$_6$")
    _abs_panel(axTR, p_arr, psol_nF, psur_nF, psur_nF_sd,
               psol_nSF6, psur_nSF6, psur_nSF6_sd,
               "Gas pressure (mTorr)",
               f"(b) Pressure sweep — P$_\\mathrm{{rf}}$ = {P_FIXED_W} W, pure SF$_6$")
    _res_panel(axBL, P_arr, sol_nF, sur_nF, sur_nF_sd,
               sol_nSF6, sur_nSF6, sur_nSF6_sd, "RF power (W)")
    _res_panel(axBR, p_arr, psol_nF, psur_nF, psur_nF_sd,
               psol_nSF6, psur_nSF6, psur_nSF6_sd, "Gas pressure (mTorr)")

    fig.suptitle("Surrogate vs. solver benchmark with residuals "
                 "(top: absolute log-density; bottom: surrogate $-$ solver, $\\pm\\sigma$ band)",
                 fontsize=13)
    save(fig, "surrogate_vs_solver_neutrals_benchmark_residuals")


def fig_parameter_sweeps_clean():
    """Publication-grade replacement for Figure 20.  Drops the squeezed
    reactor-schematic strips and shows full-width 1D trend plots for both
    neutral and charged species across power and pressure sweeps."""
    rc, zc, inside = mdl.get_mesh()

    powers = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    pressures = [3.0, 5.0, 10.0, 15.0, 20.0]
    P_FIXED = 10.0
    P_FIXED_W = 700

    NEUTRALS = [("nSF6", r"SF$_6$", "#e41a1c"),
                ("nSF5", r"SF$_5$", "#ff7f00"),
                ("nSF4", r"SF$_4$", "#ffd700"),
                ("nSF3", r"SF$_3$", "#4daf4a"),
                ("nSF2", r"SF$_2$", "#377eb8"),
                ("nF",   r"F",      "#984ea3"),
                ("nF2",  r"F$_2$",  "#a65628")]
    CHARGED = [("ion_ne", r"$n_e$",  "#1f77b4"),
               ("ion_n+", r"$n_+$",  "#d62728"),
               ("ion_n-", r"$n_-$",  "#2ca02c")]

    def case_id(P, p, ar=0.0):
        return f"P{int(P):04d}W_p{int(p):02d}mT_xAr{int(round(ar*100)):03d}"

    def avg_log10(case, field):
        f = _load_case_field(case, field)
        if f is None:
            return np.nan
        f = np.clip(f, 1.0, None)
        return np.log10(_vol_avg_inside(f, inside))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Neutrals vs power
    ax = axes[0, 0]
    for fld, lab, col in NEUTRALS:
        y = [avg_log10(case_id(P, P_FIXED, 0.0), fld) for P in powers]
        ax.plot(powers, y, "-o", color=col, lw=1.8, ms=5, label=lab)
    ax.set_xlabel("RF power (W)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ volume-avg.\ density (cm$^{-3}$)", fontsize=12)
    ax.set_title("(a) Neutral species vs RF power (p = 10 mTorr, pure SF$_6$)",
                 fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=True, fontsize=10, ncol=2, loc="best")
    ax.tick_params(labelsize=10)

    # (b) Neutrals vs pressure
    ax = axes[0, 1]
    for fld, lab, col in NEUTRALS:
        y = [avg_log10(case_id(P_FIXED_W, p, 0.0), fld) for p in pressures]
        ax.plot(pressures, y, "-s", color=col, lw=1.8, ms=5, label=lab)
    ax.set_xlabel("Gas pressure (mTorr)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ volume-avg.\ density (cm$^{-3}$)", fontsize=12)
    ax.set_title(f"(b) Neutral species vs pressure (P$_\\mathrm{{rf}}$ = {P_FIXED_W} W, pure SF$_6$)",
                 fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=True, fontsize=10, ncol=2, loc="best")
    ax.tick_params(labelsize=10)

    # (c) Charged vs power
    ax = axes[1, 0]
    for fld, lab, col in CHARGED:
        y = [avg_log10(case_id(P, P_FIXED, 0.0), fld) for P in powers]
        ax.plot(powers, y, "-o", color=col, lw=1.8, ms=5, label=lab)
    ax.set_xlabel("RF power (W)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ volume-avg.\ density (cm$^{-3}$)", fontsize=12)
    ax.set_title("(c) Charged species vs RF power (p = 10 mTorr, pure SF$_6$)",
                 fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=True, fontsize=10, loc="best")
    ax.tick_params(labelsize=10)

    # (d) Charged vs pressure
    ax = axes[1, 1]
    for fld, lab, col in CHARGED:
        y = [avg_log10(case_id(P_FIXED_W, p, 0.0), fld) for p in pressures]
        ax.plot(pressures, y, "-s", color=col, lw=1.8, ms=5, label=lab)
    ax.set_xlabel("Gas pressure (mTorr)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ volume-avg.\ density (cm$^{-3}$)", fontsize=12)
    ax.set_title(f"(d) Charged species vs pressure (P$_\\mathrm{{rf}}$ = {P_FIXED_W} W, pure SF$_6$)",
                 fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=True, fontsize=10, loc="best")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    save(fig, "parameter_sweeps_clean")


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
    is what the §10.7 solver-vs-surrogate ratio compares against.

    Skips re-timing if inference_timing.json already exists; pass
    --retime to force a fresh measurement."""
    out_path = os.path.join(ENS_DIR, "inference_timing.json")
    if os.path.exists(out_path) and "--retime" not in sys.argv:
        with open(out_path) as f:
            cached = json.load(f)
        print(f"  Skipping inference timing — {out_path} exists "
              f"(ms/eval = {cached['ms_per_eval_mean']:.2f}, "
              f"{cached['speedup_vs_legacy_solver']:.0f}×/{cached['speedup_vs_lxcat_solver']:.0f}×). "
              f"Pass --retime to force a fresh measurement.")
        return cached
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
    fig_surrogate_vs_solver_neutrals_benchmark(models)
    fig_parameter_sweeps_clean()

    print("Timing inference…")
    time_inference(models, val_d)

    print("Done.")


if __name__ == "__main__":
    main()
