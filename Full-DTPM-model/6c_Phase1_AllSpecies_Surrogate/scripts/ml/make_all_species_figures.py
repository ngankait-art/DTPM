#!/usr/bin/env python3
"""Multi-species surrogate-vs-solver figures for the all-species report.

Reads each per-species ensemble from
    results/ml_production_ensemble_all_species/<species>/{model_{0..4}.pt,summary.json}
and overlays surrogate (ensemble mean ± σ) on solver volume-averaged
log-density across a power sweep and a pressure sweep.

Outputs (into the SF6 figures dir, target of \graphicspath in
report_all_species/main.tex):
    fig_all_species_neutrals_power.{pdf,png}        # 3×3
    fig_all_species_neutrals_pressure.{pdf,png}     # 3×3
    fig_all_species_charged_power.{pdf,png}         # 4×3 (11 charged + Te)
    fig_all_species_charged_pressure.{pdf,png}      # 4×3
    fig_all_species_rmse_summary.{pdf,png}          # one bar per species
    inference_timing_all_species.json               # per-species ms/eval
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
SIXC_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIXC_ROOT, "..", ".."))
FIG_DIR = os.path.join(
    REPO_ROOT, "Plasma Chemistry Module",
    "SF6_surrogate_and_LXCat", "figures")
ENS_BASE = os.path.join(SIXC_ROOT, "results",
                        "ml_production_ensemble_all_species")
DATASET_BASE = os.path.join(SIXC_ROOT, "results", "ml_dataset", "lxcat")

sys.path.insert(0, os.path.join(SIXC_ROOT, "scripts", "ml"))
from species_loader import mdl  # noqa: E402
from single_head_arch import SingleHeadMLP  # noqa: E402

NEUTRALS = ["nSF6", "nSF5", "nSF4", "nSF3", "nSF2", "nSF", "nF", "nF2", "nS"]
CHARGED = ["ion_ne", "ion_n+", "ion_n-", "ion_F+", "ion_F-",
           "ion_SF3+", "ion_SF4+", "ion_SF4-", "ion_SF5+",
           "ion_SF5-", "ion_SF6-"]
EXTRA = ["Te"]
ALL = NEUTRALS + CHARGED + EXTRA

POWERS = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
PRESSURES = [3.0, 5.0, 10.0, 15.0, 20.0]
P_FIXED = 10.0
P_FIXED_W = 700

R_PROC = 0.105
Z_TOP = 0.234


# ─────────────────────────────────────────────────────────────────────
def case_id(P, p, ar=0.0):
    return f"P{int(P):04d}W_p{int(p):02d}mT_xAr{int(round(ar*100)):03d}"


def load_species_ensemble(species, device="cpu"):
    sp_dir = os.path.join(ENS_BASE, species)
    summary_path = os.path.join(sp_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None, None
    with open(summary_path) as f:
        summary = json.load(f)
    bias = float(summary["output_bias_init"])
    models = []
    for i in range(summary["n_ensemble"]):
        m = SingleHeadMLP(n_in=5, output_bias=bias).to(device)
        sd = torch.load(os.path.join(sp_dir, f"model_{i}.pt"),
                        map_location=device, weights_only=True)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, summary


def vol_avg_inside(field, inside):
    return float(np.mean(field[inside]))


def solver_logavg(species, P, p, ar=0.0, *, inside):
    cid = case_id(P, p, ar)
    f_path = os.path.join(DATASET_BASE, cid, f"{species}.npy")
    if not os.path.exists(f_path):
        return np.nan
    f = np.load(f_path)
    f = np.clip(f, 1.0, None)
    return np.log10(vol_avg_inside(f, inside))


def surrogate_logavg(models, P, p, ar, rc, zc, inside):
    """Mean over inside-mask cells of 10**ensemble_pred, then log10.
    Returns (mean, std) across ensemble members."""
    rg, zg = np.meshgrid(rc, zc, indexing="ij")
    r_in = rg[inside].astype(np.float32)
    z_in = zg[inside].astype(np.float32)
    n = r_in.size
    X = torch.from_numpy(np.column_stack([
        r_in / R_PROC, z_in / Z_TOP,
        np.full(n, P / 1200, dtype=np.float32),
        np.full(n, p / 20, dtype=np.float32),
        np.full(n, ar, dtype=np.float32),
    ]))
    with torch.no_grad():
        preds = torch.stack([m(X) for m in models]).numpy()  # (M, n, 1)
    lin = 10 ** preds[:, :, 0]
    per_member_avg = lin.mean(axis=1)            # (M,)
    ens_log = np.log10(np.clip(per_member_avg, 1.0, None))
    return float(ens_log.mean()), float(ens_log.std())


# ─────────────────────────────────────────────────────────────────────
def collect_sweep_data(species_list, rc, zc, inside):
    """Build solver/surrogate sweep tables for the given species list.
    Returns nested dict[species] = {power: {...}, pressure: {...}}."""
    out = {}
    for sp in species_list:
        sp_dir = os.path.join(ENS_BASE, sp)
        if not os.path.exists(os.path.join(sp_dir, "summary.json")):
            print(f"  skip {sp} — no summary.json yet")
            continue
        models, summary = load_species_ensemble(sp, device="cpu")
        if models is None:
            continue
        d = {"summary": summary, "power": {}, "pressure": {}}
        # Power sweep at p=10mTorr, pure SF6
        Px, sol, mu, sd = [], [], [], []
        for P in POWERS:
            sv = solver_logavg(sp, P, P_FIXED, 0.0, inside=inside)
            mv, ss = surrogate_logavg(models, P, P_FIXED, 0.0, rc, zc, inside)
            if not np.isfinite(sv):
                continue
            Px.append(P); sol.append(sv); mu.append(mv); sd.append(ss)
        d["power"] = {"P": np.array(Px), "sol": np.array(sol),
                      "sur": np.array(mu), "sd": np.array(sd)}
        # Pressure sweep at P=700W, pure SF6
        px, sol, mu, sd = [], [], [], []
        for p in PRESSURES:
            sv = solver_logavg(sp, P_FIXED_W, p, 0.0, inside=inside)
            mv, ss = surrogate_logavg(models, P_FIXED_W, p, 0.0, rc, zc, inside)
            if not np.isfinite(sv):
                continue
            px.append(p); sol.append(sv); mu.append(mv); sd.append(ss)
        d["pressure"] = {"p": np.array(px), "sol": np.array(sol),
                         "sur": np.array(mu), "sd": np.array(sd)}
        out[sp] = d
        print(f"  collected {sp} (RMSE={summary['metrics']['rmse']:.5f})")
    return out


# ─────────────────────────────────────────────────────────────────────
def fig_grid(data, species_list, sweep, xlabel, title, outname,
             ncols=3):
    n = len(species_list)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    for k, sp in enumerate(species_list):
        i, j = divmod(k, ncols)
        ax = axes[i, j]
        if sp not in data:
            ax.text(0.5, 0.5, f"{sp}\n(missing)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        d = data[sp][sweep]
        x = d["P"] if sweep == "power" else d["p"]
        ax.plot(x, d["sol"], "-", color="#222", lw=1.6, label="Solver")
        ax.plot(x, d["sur"], "--", color="#d62728", lw=1.4,
                marker="o", ms=4, label="Surrogate")
        ax.fill_between(x, d["sur"] - d["sd"], d["sur"] + d["sd"],
                        color="#d62728", alpha=0.20, label=r"$\pm 1\sigma$")
        ax.set_title(sp, fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)
        if i == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=10)
        if j == 0:
            ax.set_ylabel(r"$\log_{10}$ vol-avg (cm$^{-3}$)", fontsize=10)
        if k == 0:
            ax.legend(frameon=True, fontsize=8, loc="best")
    # Hide empty axes
    for k in range(n, nrows * ncols):
        i, j = divmod(k, ncols)
        axes[i, j].set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf = os.path.join(FIG_DIR, f"{outname}.pdf")
    png = os.path.join(FIG_DIR, f"{outname}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


def _fmt_species_label(sp):
    if sp == "ion_ne":
        return r"$n_e$"
    if sp == "ion_n+":
        return r"$n_{+}$"
    if sp == "ion_n-":
        return r"$n_{-}$"
    if sp.startswith("ion_"):
        s = sp[4:]
        s = s.replace("SF6", "SF$_6$").replace("SF5", "SF$_5$") \
             .replace("SF4", "SF$_4$").replace("SF3", "SF$_3$") \
             .replace("SF2", "SF$_2$")
        s = s.replace("+", "$^{+}$").replace("-", "$^{-}$")
        return s
    if sp == "Te":
        return r"$T_e$"
    return sp.replace("nSF6", "SF$_6$").replace("nSF5", "SF$_5$") \
             .replace("nSF4", "SF$_4$").replace("nSF3", "SF$_3$") \
             .replace("nSF2", "SF$_2$").replace("nSF", "SF") \
             .replace("nF2", "F$_2$").replace("nF", "F").replace("nS", "S")


def fig_consolidated(data, species_list, title, outname,
                     palette="tab10"):
    """One figure, 2 panels (power left, pressure right), all species
    overlaid on each panel.  Solid = solver, dashed = surrogate ensemble
    mean.  Publication-grade styling: large fonts, distinct colour
    cycle, white background, thin grid, no shading (clutter)."""
    cmap = plt.get_cmap("tab20" if len(species_list) > 10 else palette)
    colors = [cmap(i / max(len(species_list) - 1, 1))
              if cmap.N >= 256 else cmap(i % cmap.N)
              for i in range(len(species_list))]
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X",
               "p", "h", "H", "*", "d"]
    available = [sp for sp in species_list if sp in data]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(left=0.07, right=0.84, top=0.90, bottom=0.13,
                        wspace=0.22)

    for k, sp in enumerate(available):
        col = colors[k]
        mk = markers[k % len(markers)]
        lab = _fmt_species_label(sp)
        # Power sweep (left)
        d = data[sp]["power"]
        axL.plot(d["P"], d["sol"], "-", color=col, lw=1.8)
        axL.plot(d["P"], d["sur"], "--", color=col, lw=1.4,
                 marker=mk, ms=5, label=lab)
        # Pressure sweep (right)
        d = data[sp]["pressure"]
        axR.plot(d["p"], d["sol"], "-", color=col, lw=1.8)
        axR.plot(d["p"], d["sur"], "--", color=col, lw=1.4,
                 marker=mk, ms=5)

    for ax, x_label, panel_title in [
        (axL, "RF power (W)",
         "(a) Power sweep — $p$ = 10 mTorr, pure SF$_6$"),
        (axR, "Gas pressure (mTorr)",
         "(b) Pressure sweep — $P_\\mathrm{rf}$ = 700 W, pure SF$_6$"),
    ]:
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(r"$\log_{10}$ volume-averaged density (cm$^{-3}$)",
                      fontsize=13)
        ax.set_title(panel_title, fontsize=13)
        ax.grid(alpha=0.30, linestyle=":", linewidth=0.7)
        ax.tick_params(labelsize=11)

    # Add a single legend block for species (right of right panel)
    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(0.995, 0.5), frameon=True, fontsize=11,
               title="Species", title_fontsize=12, ncol=1)

    # Add a separate legend for solid/dashed convention
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", lw=1.8, ls="-",  label="Solver"),
        Line2D([0], [0], color="black", lw=1.4, ls="--",
               marker="o", ms=5, label="Surrogate"),
    ]
    axL.legend(handles=style_handles, loc="best", frameon=True,
               fontsize=10, title="Curve type", title_fontsize=11)

    fig.suptitle(title, fontsize=15, y=0.985)
    pdf = os.path.join(FIG_DIR, f"{outname}.pdf")
    png = os.path.join(FIG_DIR, f"{outname}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


def fig_rmse_summary(data, outname="fig_all_species_rmse_summary"):
    sp_in_order = [sp for sp in ALL if sp in data]
    rmse = [data[sp]["summary"]["metrics"]["rmse"] for sp in sp_in_order]
    is_charged = ["ion" in sp for sp in sp_in_order]
    is_te = [sp == "Te" for sp in sp_in_order]
    colors = ["#1f77b4" if not c and not t else
              ("#d62728" if c else "#2ca02c")
              for c, t in zip(is_charged, is_te)]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(sp_in_order)), 4.2))
    bars = ax.bar(sp_in_order, rmse, color=colors,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=0)
    ax.set_ylabel(r"RMSE on $\log_{10}$ density", fontsize=12)
    ax.set_title("Per-species ensemble RMSE on validation set "
                 "(blue=neutral, red=charged, green=Te)", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    pdf = os.path.join(FIG_DIR, f"{outname}.pdf")
    png = os.path.join(FIG_DIR, f"{outname}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


# ─────────────────────────────────────────────────────────────────────
def time_inference_per_species(species_list):
    """Cache total per-eval ms across all species (each species is one
    ensemble inference call).  Skip if cached."""
    out_path = os.path.join(ENS_BASE, "inference_timing_all_species.json")
    if os.path.exists(out_path) and "--retime" not in sys.argv:
        with open(out_path) as f:
            return json.load(f)
    rc, zc, inside = mdl.get_mesh()
    rg, zg = np.meshgrid(rc, zc, indexing="ij")
    r_in = rg[inside].astype(np.float32)
    z_in = zg[inside].astype(np.float32)
    n = r_in.size
    X = torch.from_numpy(np.column_stack([
        r_in / R_PROC, z_in / Z_TOP,
        np.full(n, 700 / 1200, dtype=np.float32),
        np.full(n, 10 / 20, dtype=np.float32),
        np.full(n, 0.0, dtype=np.float32),
    ]))
    timings = {}
    for sp in species_list:
        sp_dir = os.path.join(ENS_BASE, sp)
        if not os.path.exists(os.path.join(sp_dir, "summary.json")):
            continue
        models, _ = load_species_ensemble(sp, device="cpu")
        with torch.no_grad():
            for _ in range(10):  # warm
                for m in models:
                    _ = m(X)
            wall = []
            for _ in range(50):
                t0 = time.perf_counter()
                for m in models:
                    _ = m(X)
                wall.append(time.perf_counter() - t0)
        timings[sp] = {
            "ms_per_eval_mean": float(np.mean(wall) * 1000),
            "ms_per_eval_std": float(np.std(wall) * 1000),
            "batch_size": int(n),
            "ensemble_size": len(models),
        }
        print(f"  timed {sp}: {timings[sp]['ms_per_eval_mean']:.2f} ms/eval")

    total = sum(t["ms_per_eval_mean"] for t in timings.values())
    out = {
        "per_species": timings,
        "total_ms_per_eval": float(total),
        "n_species": len(timings),
        "eval_machine_id": socket.gethostname(),
        "speedup_vs_legacy_solver": float(7.76 / (total / 1000)) if total > 0 else None,
        "speedup_vs_lxcat_solver": float(14.4 / (total / 1000)) if total > 0 else None,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  Total ms/eval (all species): {total:.1f} ms")
    if out["speedup_vs_legacy_solver"]:
        print(f"  Speedup vs legacy solver: {out['speedup_vs_legacy_solver']:.0f}×")
        print(f"  Speedup vs LXCat solver:  {out['speedup_vs_lxcat_solver']:.0f}×")
    return out


# ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rc, zc, inside = mdl.get_mesh()

    print("Collecting neutral sweep data…")
    neutrals_data = collect_sweep_data(NEUTRALS, rc, zc, inside)
    print("Collecting charged + Te sweep data…")
    charged_data = collect_sweep_data(CHARGED + EXTRA, rc, zc, inside)
    all_data = {**neutrals_data, **charged_data}

    print("Generating figures…")
    fig_grid(neutrals_data, NEUTRALS, "power", "RF power (W)",
             "Neutral species — power sweep at p=10 mTorr, pure SF$_6$",
             "fig_all_species_neutrals_power", ncols=3)
    fig_grid(neutrals_data, NEUTRALS, "pressure", "Pressure (mTorr)",
             "Neutral species — pressure sweep at P$_\\mathrm{rf}$=700 W, pure SF$_6$",
             "fig_all_species_neutrals_pressure", ncols=3)
    fig_grid(charged_data, CHARGED + EXTRA, "power", "RF power (W)",
             "Charged species + T$_e$ — power sweep at p=10 mTorr, pure SF$_6$",
             "fig_all_species_charged_power", ncols=4)
    fig_grid(charged_data, CHARGED + EXTRA, "pressure", "Pressure (mTorr)",
             "Charged species + T$_e$ — pressure sweep at P$_\\mathrm{rf}$=700 W, pure SF$_6$",
             "fig_all_species_charged_pressure", ncols=4)

    # Consolidated single-axes overlays (publication grade)
    fig_consolidated(neutrals_data, NEUTRALS,
                     "Neutral species — surrogate vs. solver "
                     "(solid: solver, dashed: surrogate ensemble mean)",
                     "fig_all_species_neutrals_consolidated")
    fig_consolidated(charged_data, CHARGED,
                     "Charged species — surrogate vs. solver "
                     "(solid: solver, dashed: surrogate ensemble mean)",
                     "fig_all_species_charged_consolidated")

    fig_rmse_summary(all_data)

    print("Timing inference (cached on second run)…")
    time_inference_per_species(ALL)

    print("Done.")


if __name__ == "__main__":
    main()
