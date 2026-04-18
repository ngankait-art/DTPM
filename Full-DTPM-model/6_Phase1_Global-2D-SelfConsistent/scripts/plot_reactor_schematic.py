#!/usr/bin/env python3
"""
Generate a dimensioned cross-section schematic of the TEL ICP reactor.

Uses corrected TEL-spec dimensions:
  - Quartz cylinder inner radius: 38 mm
  - ICP coil radius: 40.5 mm
  - Domain: 320 x 220 mm (axis at x=160 mm)
  - 6 coil positions, 10 mm pitch, starting at y=145 mm

Usage:
    python scripts/plot_reactor_schematic.py [--output path/to/save.pdf]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


# ──────────────────────────────────────────────────────────────────
# TEL ICP Reactor dimensions (all in mm)
# ──────────────────────────────────────────────────────────────────
DOMAIN_W = 320.0
DOMAIN_H = 220.0
AXIS_X = 160.0  # center / axis of symmetry

# Quartz tube
QZ_INNER_R = 38.0      # inner radius [mm]
QZ_WALL_T = 0.5        # wall thickness [mm] (corrected per TEL drawing)
QZ_OUTER_R = QZ_INNER_R + QZ_WALL_T  # 40 mm
QZ_BOTTOM = 50.0       # bottom of quartz tube region
QZ_TOP = 200.0         # top of quartz tube region

# ICP coils (TEL spec: R_coil = 40.5 mm from axis)
COIL_R = 40.5           # coil center radius from axis [mm]
COIL_SIZE = 3.0         # visual size of coil cross-section [mm] (physical wire ~2mm)
NUM_COILS = 6
COIL_START_Y = 145.0
COIL_PITCH = 10.0

# Chamber
CHAMBER_R = 157.0       # main chamber inner radius [mm] (Ø314mm)
CHAMBER_LEFT = AXIS_X - CHAMBER_R    # ~3 mm
CHAMBER_RIGHT = AXIS_X + CHAMBER_R   # ~317 mm

# Wafer
WAFER_D = 150.0         # diameter [mm]
WAFER_Y = 25.0          # wafer surface height [mm]
WAFER_THICK = 3.0       # visual thickness

# ESC / chuck
CHUCK_W = 170.0
CHUCK_H = 15.0
CHUCK_Y = WAFER_Y - WAFER_THICK - 2


def dim_arrow(ax, x1, y1, x2, y2, text, offset=0, fontsize=8,
              color='#2060C0', text_side='above', text_offset=4):
    """Draw a dimension annotation with arrows and label."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color=color,
                                lw=1.0, shrinkA=0, shrinkB=0))
    # Label position
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if text_side == 'above':
        my += text_offset
    elif text_side == 'below':
        my -= text_offset
    elif text_side == 'left':
        mx -= text_offset
    elif text_side == 'right':
        mx += text_offset

    ax.text(mx, my, text, fontsize=fontsize, color=color,
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.85))


def draw_schematic(ax):
    """Draw the full reactor cross-section on the given axes."""

    left_coil_x = AXIS_X - COIL_R    # 119.5 mm
    right_coil_x = AXIS_X + COIL_R   # 200.5 mm
    wafer_left = AXIS_X - WAFER_D / 2
    wafer_right = AXIS_X + WAFER_D / 2
    top_coil_y = COIL_START_Y + (NUM_COILS - 1) * COIL_PITCH

    # ── Background ──────────────────────────────────────
    ax.set_xlim(-30, DOMAIN_W + 55)
    ax.set_ylim(-25, DOMAIN_H + 25)
    ax.set_aspect('equal')
    ax.set_xlabel('x  [mm]  (radial)', fontsize=10)
    ax.set_ylabel('y  [mm]  (axial)', fontsize=10)
    ax.set_title('TEL ICP Reactor — 2D Cross-Section (Corrected Dimensions)',
                 fontsize=12, fontweight='bold', pad=10)

    # Domain outline (light gray dashed)
    ax.add_patch(patches.Rectangle((0, 0), DOMAIN_W, DOMAIN_H,
                 linewidth=0.8, edgecolor='#AAAAAA', facecolor='none',
                 linestyle='--', zorder=1))

    # ── Axis of symmetry ─────────────────────────────────
    ax.axvline(AXIS_X, color='#888888', linewidth=0.6, linestyle='-.',
               zorder=1, alpha=0.6)
    ax.text(AXIS_X + 3, DOMAIN_H + 18, 'axis', fontsize=7, color='#888888',
            ha='left', va='bottom', fontstyle='italic')

    # ── Main chamber walls ────────────────────────────────
    wall_color = '#555555'
    wall_lw = 2.5
    # Bottom-left wall (shelf)
    ax.plot([CHAMBER_LEFT, AXIS_X - QZ_INNER_R],
            [QZ_BOTTOM, QZ_BOTTOM], color=wall_color, lw=wall_lw, zorder=5)
    # Bottom-right wall (shelf)
    ax.plot([AXIS_X + QZ_INNER_R, CHAMBER_RIGHT],
            [QZ_BOTTOM, QZ_BOTTOM], color=wall_color, lw=wall_lw, zorder=5)
    # Left chamber sidewall
    ax.plot([CHAMBER_LEFT, CHAMBER_LEFT], [0, QZ_BOTTOM],
            color=wall_color, lw=wall_lw, zorder=5)
    # Right chamber sidewall
    ax.plot([CHAMBER_RIGHT, CHAMBER_RIGHT], [0, QZ_BOTTOM],
            color=wall_color, lw=wall_lw, zorder=5)
    # Bottom
    ax.plot([CHAMBER_LEFT, CHAMBER_RIGHT], [0, 0],
            color=wall_color, lw=wall_lw, zorder=5)

    # ── Quartz tube (blue fill) ──────────────────────────
    qz_color = '#B8D8F0'
    qz_edge = '#3070B0'
    # Left quartz wall
    ax.add_patch(patches.Rectangle(
        (AXIS_X - QZ_OUTER_R, QZ_BOTTOM), QZ_WALL_T, QZ_TOP - QZ_BOTTOM,
        facecolor=qz_color, edgecolor=qz_edge, linewidth=1.5, zorder=4))
    # Right quartz wall
    ax.add_patch(patches.Rectangle(
        (AXIS_X + QZ_INNER_R, QZ_BOTTOM), QZ_WALL_T, QZ_TOP - QZ_BOTTOM,
        facecolor=qz_color, edgecolor=qz_edge, linewidth=1.5, zorder=4))
    # Top quartz cap
    ax.add_patch(patches.Rectangle(
        (AXIS_X - QZ_OUTER_R, QZ_TOP - QZ_WALL_T), 2 * QZ_OUTER_R, QZ_WALL_T,
        facecolor=qz_color, edgecolor=qz_edge, linewidth=1.5, zorder=4))

    # Label quartz (on the inside of the tube, near the left wall)
    ax.text(AXIS_X - QZ_INNER_R + 3, QZ_BOTTOM + 15,
            'Quartz\n(SiO₂)', fontsize=6.5, color=qz_edge,
            ha='left', va='bottom', fontstyle='italic', zorder=8)

    # ── ICP coils (orange circles) ────────────────────────
    coil_face = '#F0A030'
    coil_edge = '#8B4513'

    for i in range(NUM_COILS):
        y = COIL_START_Y + i * COIL_PITCH
        # Left coil cross-section
        ax.add_patch(patches.Circle(
            (left_coil_x, y), COIL_SIZE / 2,
            facecolor=coil_face, edgecolor=coil_edge, linewidth=1.2, zorder=6))
        # Right coil cross-section
        ax.add_patch(patches.Circle(
            (right_coil_x, y), COIL_SIZE / 2,
            facecolor=coil_face, edgecolor=coil_edge, linewidth=1.2, zorder=6))
        # Current direction markers
        ax.text(left_coil_x, y, '×', fontsize=7, ha='center', va='center',
                color=coil_edge, fontweight='bold', zorder=7)
        ax.text(right_coil_x, y, '·', fontsize=10, ha='center', va='center',
                color=coil_edge, fontweight='bold', zorder=7)

    # ── Wafer ─────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (wafer_left, WAFER_Y - WAFER_THICK), WAFER_D, WAFER_THICK,
        facecolor='#C0C0C0', edgecolor='#404040', linewidth=1.5, zorder=6))
    ax.text(AXIS_X, WAFER_Y - WAFER_THICK / 2, 'Wafer (6")',
            fontsize=8, ha='center', va='center', zorder=7)

    # ── ESC / Chuck ───────────────────────────────────────
    chuck_left = AXIS_X - CHUCK_W / 2
    ax.add_patch(patches.Rectangle(
        (chuck_left, CHUCK_Y - CHUCK_H), CHUCK_W, CHUCK_H,
        facecolor='#E8E8D0', edgecolor='#808060', linewidth=1.0, zorder=5))
    ax.text(AXIS_X, CHUCK_Y - CHUCK_H / 2, 'ESC (He cooled)',
            fontsize=7, ha='center', va='center', color='#606040', zorder=6)

    # ── Plasma region (shaded) ────────────────────────────
    plasma_h = QZ_TOP - QZ_BOTTOM
    ax.add_patch(patches.Rectangle(
        (AXIS_X - QZ_INNER_R, QZ_BOTTOM), 2 * QZ_INNER_R, plasma_h,
        facecolor='#FFE0FF', edgecolor='none', alpha=0.25, zorder=2))
    ax.text(AXIS_X, (QZ_BOTTOM + QZ_TOP) / 2 + 10, 'Plasma\nregion',
            fontsize=9, ha='center', va='center', color='#A020A0',
            alpha=0.5, zorder=3)

    # ── Process volume below quartz ───────────────────────
    ax.add_patch(patches.Rectangle(
        (CHAMBER_LEFT, QZ_BOTTOM - 1), CHAMBER_RIGHT - CHAMBER_LEFT,
        -(QZ_BOTTOM - WAFER_Y) + 1,
        facecolor='#FFFFE0', edgecolor='none', alpha=0.2, zorder=2))

    # ──────────────────────────────────────────────────────
    # DIMENSION ANNOTATIONS
    # ──────────────────────────────────────────────────────

    # --- Horizontal dimensions ---

    # Quartz inner diameter — placed well above the top cap
    y_qz_dim = QZ_TOP + 16
    dim_arrow(ax, AXIS_X - QZ_INNER_R, y_qz_dim, AXIS_X + QZ_INNER_R, y_qz_dim,
              f'Ø{2*QZ_INNER_R:.0f} mm  (quartz inner, R={QZ_INNER_R:.0f})',
              fontsize=7, text_side='above', text_offset=5)

    # ICP coil diameter — just above quartz top
    y_coil_dim = QZ_TOP + 5
    dim_arrow(ax, left_coil_x, y_coil_dim, right_coil_x, y_coil_dim,
              f'Ø{2*COIL_R:.0f} mm  (coil center)',
              fontsize=6.5, color='#A06020', text_side='above', text_offset=5)

    # Wafer diameter — below wafer to avoid chamber overlap
    dim_arrow(ax, wafer_left, WAFER_Y + 4, wafer_right, WAFER_Y + 4,
              f'{WAFER_D:.0f} mm (wafer)', fontsize=7,
              text_side='above', text_offset=4)

    # Chamber half-width — place BELOW the bottom wall, outside the chamber
    y_cham = -6
    dim_arrow(ax, CHAMBER_LEFT, y_cham, AXIS_X, y_cham,
              f'{CHAMBER_R:.0f} mm', fontsize=7,
              text_side='above', text_offset=4)
    dim_arrow(ax, AXIS_X, y_cham, CHAMBER_RIGHT, y_cham,
              f'{CHAMBER_R:.0f} mm', fontsize=7,
              text_side='above', text_offset=4)

    # Full domain width — further below
    dim_arrow(ax, 0, -18, DOMAIN_W, -18,
              f'{DOMAIN_W:.0f} mm (domain)', fontsize=7,
              text_side='below', text_offset=5, color='#888888')

    # --- Vertical dimensions ---

    # ICP-to-wafer distance (right side, far out)
    x_icp = CHAMBER_RIGHT + 22
    dim_arrow(ax, x_icp, WAFER_Y, x_icp, COIL_START_Y,
              f'{COIL_START_Y - WAFER_Y:.0f} mm\n(ICP → wafer)',
              fontsize=7.5, text_side='right', text_offset=20, color='#C02020')

    # Shelf-to-lowest-coil (left side, outside coils)
    x_shelf = left_coil_x - 18
    dim_arrow(ax, x_shelf, QZ_BOTTOM, x_shelf, COIL_START_Y,
              f'{COIL_START_Y - QZ_BOTTOM:.0f} mm',
              fontsize=7, text_side='left', text_offset=14)

    # Coil extent (left side, close to coils)
    x_coilext = left_coil_x - 8
    dim_arrow(ax, x_coilext, COIL_START_Y, x_coilext, top_coil_y,
              f'{(NUM_COILS-1)*COIL_PITCH:.0f} mm\n({NUM_COILS} coils)',
              fontsize=6.5, text_side='left', text_offset=16, color='#A06020')

    # Wafer-to-shelf (right side, near chamber wall)
    x_w2s = CHAMBER_RIGHT + 8
    dim_arrow(ax, x_w2s, WAFER_Y, x_w2s, QZ_BOTTOM,
              f'{QZ_BOTTOM - WAFER_Y:.0f} mm',
              fontsize=7, text_side='right', text_offset=12)

    # Wafer height from bottom (far left)
    x_wh = CHAMBER_LEFT - 12
    dim_arrow(ax, x_wh, 0, x_wh, WAFER_Y,
              f'{WAFER_Y:.0f} mm',
              fontsize=7, text_side='left', text_offset=12)

    # Full domain height (far far left)
    dim_arrow(ax, -22, 0, -22, DOMAIN_H,
              f'{DOMAIN_H:.0f} mm', fontsize=7, text_side='left',
              text_offset=12, color='#888888')

    # Quartz tube height (right side, near quartz wall)
    x_qh = AXIS_X + QZ_OUTER_R + 8
    dim_arrow(ax, x_qh, QZ_BOTTOM, x_qh, QZ_TOP,
              f'{QZ_TOP - QZ_BOTTOM:.0f} mm\n(tube height)',
              fontsize=6.5, text_side='right', text_offset=14, color='#3070B0')

    # --- Coil pitch annotation ---
    if NUM_COILS >= 2:
        y1 = COIL_START_Y
        y2 = COIL_START_Y + COIL_PITCH
        xp = right_coil_x + COIL_SIZE / 2 + 5
        dim_arrow(ax, xp, y1, xp, y2,
                  f'{COIL_PITCH:.0f} mm\npitch',
                  fontsize=6, text_side='right', text_offset=12, color='#A06020')

    # ── Grid lines (subtle) ──────────────────────────────
    for x in [0, AXIS_X, DOMAIN_W]:
        ax.axvline(x, color='#DDDDDD', linewidth=0.3, zorder=0)
    for y in [0, WAFER_Y, QZ_BOTTOM, COIL_START_Y, QZ_TOP, DOMAIN_H]:
        ax.axhline(y, color='#DDDDDD', linewidth=0.3, zorder=0)

    # ── Legend (moved to avoid overlap with geometry) ─────
    legend_items = [
        patches.Patch(facecolor=qz_color, edgecolor=qz_edge, label='Quartz (SiO₂)'),
        patches.Patch(facecolor=coil_face, edgecolor=coil_edge, label='ICP coils (Cu)'),
        patches.Patch(facecolor='#C0C0C0', edgecolor='#404040', label='Wafer'),
        patches.Patch(facecolor='#FFE0FF', edgecolor='#A020A0', alpha=0.4, label='Plasma region'),
        patches.Patch(facecolor='#E8E8D0', edgecolor='#808060', label='ESC chuck'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=7,
              framealpha=0.95, edgecolor='#CCCCCC')

    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description='Generate TEL ICP reactor schematic')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (e.g., docs/reactor_schematic.pdf)')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.linewidth': 0.8,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    draw_schematic(ax)

    fig.tight_layout(pad=1.5)

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved schematic to {args.output}")
    else:
        # Default: save to docs/
        import os
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
        os.makedirs(docs_dir, exist_ok=True)
        out_path = os.path.join(docs_dir, 'reactor_schematic.pdf')
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved schematic to {out_path}")
        # Also save PNG for quick viewing
        out_png = os.path.join(docs_dir, 'reactor_schematic.png')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Saved PNG preview to {out_png}")

    plt.close(fig)


if __name__ == '__main__':
    main()
