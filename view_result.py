"""
view_result.py — Quick viewer + HTML exporter for VoxelSense 3D output.

Usage
-----
    python view_result.py                       open PyVista window
    python view_result.py --html                also export voxel_view.html
    python view_result.py --html-only           export HTML and exit
    python view_result.py --backend matplotlib  force Matplotlib

v5.7: simplified HTML export — uses pyvista.Plotter.export_html() when
available, falls back to a static screenshot wrapped in HTML otherwise.
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.config import COORDS_NPY, COLORS_NPY, HTML_VIEWER, OUTPUT_DIR


# ============================================================================
# Helpers
# ============================================================================

def _load_data():
    """Loads coords + colors from output/, returns (coords, colors) or None."""
    if not os.path.exists(COORDS_NPY) or not os.path.exists(COLORS_NPY):
        print(f"[error] Missing {COORDS_NPY} or {COLORS_NPY}")
        print(f"        Run main.py first.")
        return None, None

    coords = np.load(COORDS_NPY).astype(float)
    colors = np.load(COLORS_NPY)

    if len(coords) == 0:
        print("[error] coords.npy is empty.")
        return None, None

    if colors.max() > 1.0:
        print("[warning] Colors > 1.0 — normalising by /255.")
        colors = colors / 255.0

    print(f"[view] {len(coords):,} voxels loaded")
    return coords, colors


def _build_plotter(coords, colors, title, off_screen=False):
    """Builds a PyVista Plotter with front-facing camera."""
    import pyvista as pv

    cloud = pv.PolyData(coords)
    cloud["colors"] = colors

    pl = pv.Plotter(off_screen=off_screen,
                    window_size=(1280, 800),
                    title=title)
    pl.background_color = "black"
    pl.add_mesh(cloud, scalars="colors", rgb=True,
                point_size=6, render_points_as_spheres=True)
    pl.add_text(title, font_size=11, color="white")

    cx, cy, cz = coords.mean(axis=0)
    dist = max(np.ptp(coords[:, 0]),
               np.ptp(coords[:, 1]),
               np.ptp(coords[:, 2])) * 2.0
    if dist <= 0:
        dist = 1.0
    pl.camera_position = [(cx, cy, cz + dist), (cx, cy, cz), (0, 1, 0)]
    return pl


# ============================================================================
# HTML export
# ============================================================================

def export_html(coords, colors, out_path):
    """Try multiple export strategies, in order of fidelity."""
    print(f"\n[html] Exporting to {out_path} ...")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        import pyvista as pv
        print(f"[html] PyVista version: {pv.__version__}")
    except Exception as e:
        print(f"[html] ERROR: PyVista not available: {e}")
        return False

    pl = _build_plotter(coords, colors, "VoxelSense 3D", off_screen=True)

    # ── Strategy 1: modern export_html() ─────────────────────────────────
    if hasattr(pl, "export_html"):
        try:
            pl.export_html(out_path)
            if os.path.exists(out_path):
                size_mb = os.path.getsize(out_path) / 1024 / 1024
                print(f"[html] ✓ Exported via export_html()  ({size_mb:.1f} MB)")
                print(f"[html] Open this file in Chrome / Edge:")
                print(f"       {out_path}")
                pl.close()
                return True
        except Exception as e:
            print(f"[html] export_html() failed: {e}")

    # ── Strategy 2: screenshot + HTML wrapper ────────────────────────────
    try:
        png_path = out_path.replace(".html", ".png")
        pl.screenshot(png_path, return_img=False)
        pl.close()

        if os.path.exists(png_path):
            _write_image_wrapper(out_path, os.path.basename(png_path))
            print(f"[html] ✓ Exported as static PNG-in-HTML")
            print(f"[html] PNG : {png_path}")
            print(f"[html] HTML: {out_path}")
            return True
    except Exception as e:
        print(f"[html] Screenshot fallback failed: {e}")
        try:
            pl.close()
        except Exception:
            pass

    print(f"[html] ✗ All export strategies failed.")
    return False


def _write_image_wrapper(html_path, png_filename):
    """Writes an HTML shell that displays the PNG screenshot."""
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>VoxelSense 3D — Result Viewer</title>
  <style>
    body{{margin:0;background:#000;display:flex;align-items:center;
         justify-content:center;height:100vh;font-family:sans-serif}}
    img{{max-width:100%;max-height:100vh;object-fit:contain}}
    .note{{position:fixed;top:10px;left:10px;color:#888;
           background:rgba(0,0,0,0.7);padding:8px 12px;border-radius:4px}}
  </style>
</head>
<body>
  <div class="note">VoxelSense 3D — static result preview</div>
  <img src="{png_filename}" alt="VoxelSense voxel result">
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================================
# Matplotlib fallback
# ============================================================================

def _show_matplotlib(coords, colors, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D    # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=1, depthshade=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="View VoxelSense 3D output")
    parser.add_argument("--backend", choices=["pyvista", "matplotlib"],
                        default="pyvista")
    parser.add_argument("--html", action="store_true",
                        help="Also export voxel_view.html")
    parser.add_argument("--html-only", action="store_true",
                        help="Only export HTML, skip the interactive window")
    args = parser.parse_args()

    coords, colors = _load_data()
    if coords is None:
        return

    # ── HTML export (if requested) ───────────────────────────────────────
    if args.html or args.html_only:
        ok = export_html(coords, colors, HTML_VIEWER)
        if args.html_only:
            if ok:
                print("\n[html] Done.  Drag the .html file into Chrome/Edge.")
            return

    # ── Interactive view ─────────────────────────────────────────────────
    if args.backend == "pyvista":
        try:
            pl = _build_plotter(coords, colors, "VoxelSense 3D",
                                off_screen=False)
            pl.show()
            return
        except Exception as e:
            print(f"[view] PyVista failed: {e}")
            print(f"[view] Falling back to Matplotlib.")

    _show_matplotlib(coords, colors, "VoxelSense 3D")


if __name__ == "__main__":
    main()