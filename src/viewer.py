"""
src/viewer.py

Visualisation helpers for VoxelSense 3D.

Provides a uniform `show()` entry point that prefers PyVista and falls
back to Matplotlib when PyVista is unavailable.  Also provides
`run_compare()` for the RGB-vs-CIELAB side-by-side window used in
report screenshots.

v5.1 (2026-05) fixes:
  - P2: dist <= 0 fallback for degenerate-extent voxel clouds.
  - empty-coords guard for both single and compare views.
"""

import numpy as np


# ============================================================================
# Single-view rendering
# ============================================================================

def show(coords:  np.ndarray,
         colors:  np.ndarray,
         title:   str = "VoxelSense 3D") -> None:
    """Display voxels — try PyVista first, fall back to Matplotlib on error."""
    if len(coords) == 0:
        print("[vis] WARNING: 0 voxels — nothing to display")
        return

    try:
        _show_pyvista(coords, colors, title)
    except Exception as e:
        print(f"[vis] PyVista unavailable ({e}), using Matplotlib.")
        _show_matplotlib(coords, colors, title)


def _show_pyvista(coords, colors, title) -> None:
    import pyvista as pv

    cloud = pv.PolyData(coords)
    cloud["colors"] = colors

    pl = pv.Plotter(title=title)
    pl.background_color = "black"
    pl.add_mesh(cloud, scalars="colors", rgb=True,
                point_size=6, render_points_as_spheres=True)
    pl.add_text(title, font_size=11, color="white")

    # Front view — camera on +Z, looking at -Z, with Y up.
    cx, cy, cz = coords.mean(axis=0)
    dist = max(np.ptp(coords[:, 0]),
               np.ptp(coords[:, 1]),
               np.ptp(coords[:, 2])) * 2.0
    if dist <= 0:
        dist = 1.0
    pl.camera_position = [
        (cx, cy, cz + dist),
        (cx, cy, cz),
        (0,  1,  0),
    ]
    pl.show()


def _show_matplotlib(coords, colors, title) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 — registers projection

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=1, depthshade=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ============================================================================
# RGB vs CIELAB comparison
# ============================================================================

def run_compare(palette, voxel_colors, coords) -> None:
    """Side-by-side RGB vs CIELAB matching, plus ΔE statistics.

    Used to generate the headline number for the report:
      "CIELAB reduces mean ΔE by N% over RGB."
    """
    if len(coords) == 0:
        print("[compare] WARNING: 0 voxels — skipping comparison")
        return

    print("\n[compare] RGB vs CIELAB matching")

    _, colors_lab = palette.match_lab(voxel_colors)
    _, colors_rgb = palette.match_rgb(voxel_colors)

    de_rgb = palette.compute_delta_e(
        voxel_colors, (colors_rgb * 255).astype(np.float32))
    de_lab = palette.compute_delta_e(
        voxel_colors, (colors_lab * 255).astype(np.float32))

    print(f"  Delta E (CIEDE2000):")
    print(f"  RGB — mean {de_rgb.mean():.2f}  "
          f"median {np.median(de_rgb):.2f}  max {de_rgb.max():.2f}")
    print(f"  Lab — mean {de_lab.mean():.2f}  "
          f"median {np.median(de_lab):.2f}  max {de_lab.max():.2f}")
    diff = de_rgb.mean() - de_lab.mean()
    pct  = diff / de_rgb.mean() * 100 if de_rgb.mean() > 0 else 0.0
    print(f"  Improvement: {diff:.2f} avg ΔE ({pct:.1f}% reduction)")
    print(f"  Voxels with ΔE < 5:  "
          f"RGB {(de_rgb < 5).mean()*100:.1f}%   "
          f"Lab {(de_lab < 5).mean()*100:.1f}%")

    _render_compare(coords, colors_rgb, colors_lab)


def _render_compare(coords, colors_rgb, colors_lab) -> None:
    """Try PyVista linked-view; fall back to side-by-side Matplotlib."""
    try:
        import pyvista as pv

        pl = pv.Plotter(shape=(1, 2),
                        title="VoxelSense — RGB vs CIELAB")
        pl.background_color = "black"
        for i, (c, t) in enumerate([
            (colors_rgb, "RGB matching (traditional)"),
            (colors_lab, "CIELAB matching (VoxelSense)"),
        ]):
            cloud = pv.PolyData(coords)
            cloud["colors"] = c
            pl.subplot(0, i)
            pl.add_mesh(cloud, scalars="colors", rgb=True,
                        point_size=6, render_points_as_spheres=True)
            pl.add_text(t, font_size=10, color="white")
        pl.link_views()
        pl.show()

    except Exception as e:
        print(f"[vis] PyVista failed ({e}), using Matplotlib.")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D    # noqa: F401

        fig = plt.figure(figsize=(14, 6))
        for i, (c, t) in enumerate([
            (colors_rgb, "RGB matching (traditional)"),
            (colors_lab, "CIELAB matching (VoxelSense)"),
        ]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=c, s=1, depthshade=True)
            ax.set_title(t)
        plt.tight_layout()
        plt.show()