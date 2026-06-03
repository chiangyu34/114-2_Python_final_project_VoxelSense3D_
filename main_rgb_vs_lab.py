"""
main_rgb_vs_lab.py
Side-by-side RGB vs CIELAB comparison + saves report_delta_e.csv

Usage:  python main_rgb_vs_lab.py
"""

import os, sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.config import BLOCKS_JSON
from src.geometry import load_and_voxelize
from src.color_engine import BlockPalette, make_default_voxel_colors

# ===========================================================================
INPUT_OBJ   = os.path.join(PROJECT_ROOT, "models", "usagi_chiikawa.glb")
QUALITY     = "low"
ROTATE_Z_UP = True
# ===========================================================================

QUALITY_MAP = {"low": 32, "medium": 64, "high": 96, "extra_high": 128}

def rotate(coords):
    c = coords[:, [0, 2, 1]].copy()
    c[:, 2] *= -1
    return c

def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

    # 1. Load
    coords, surface_colors, mesh, *_ = load_and_voxelize(
        INPUT_OBJ, QUALITY_MAP[QUALITY])
    if ROTATE_Z_UP:
        coords = rotate(coords)
    voxel_colors = (surface_colors.astype(np.float32)
                    if surface_colors is not None
                    else make_default_voxel_colors(coords).astype(np.float32))

    # 2. Match
    palette = BlockPalette(BLOCKS_JSON)
    names_lab, colors_lab = palette.match_lab(voxel_colors)
    names_rgb, colors_rgb = palette.match_rgb(voxel_colors)

    # 3. Delta E
    de_rgb = palette.compute_delta_e(voxel_colors, (colors_rgb*255).astype(np.float32))
    de_lab = palette.compute_delta_e(voxel_colors, (colors_lab*255).astype(np.float32))

    print(f"\n  RGB  mean={de_rgb.mean():.2f}  median={np.median(de_rgb):.2f}")
    print(f"  Lab  mean={de_lab.mean():.2f}  median={np.median(de_lab):.2f}")
    improvement = (de_rgb.mean()-de_lab.mean())/de_rgb.mean()*100
    print(f"  Improvement: {improvement:.1f}%  |  "
          f"ΔE<5: RGB {(de_rgb<5).mean()*100:.1f}%  Lab {(de_lab<5).mean()*100:.1f}%")

    # 4. Save report_delta_e.csv
    csv_path = os.path.join(OUTPUT_DIR, "report_delta_e.csv")
    pd.DataFrame({
        "method":   ["RGB", "CIELAB"],
        "mean_dE":  [round(float(de_rgb.mean()),2), round(float(de_lab.mean()),2)],
        "median_dE":[round(float(np.median(de_rgb)),2), round(float(np.median(de_lab)),2)],
        "max_dE":   [round(float(de_rgb.max()),2), round(float(de_lab.max()),2)],
        "pct_under5":[round(float((de_rgb<5).mean()*100),1),
                      round(float((de_lab<5).mean()*100),1)],
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {csv_path}")

    # 5. Visualise
    try:
        import pyvista as pv
        cam = [(coords[:,0].mean(), coords[:,1].mean(),
                coords[:,2].mean() + np.ptp(coords,axis=0).max()*2),
               tuple(coords.mean(axis=0)), (0,1,0)]
        pl = pv.Plotter(shape=(1,2), title="RGB vs CIELAB")
        pl.background_color = "black"
        for i, (c, t) in enumerate([
            (colors_rgb, f"RGB  (mean ΔE={de_rgb.mean():.1f})"),
            (colors_lab, f"CIELAB  (mean ΔE={de_lab.mean():.1f})"),
        ]):
            cloud = pv.PolyData(coords.astype(float))
            cloud["colors"] = c
            pl.subplot(0, i)
            pl.add_mesh(cloud, scalars="colors", rgb=True,
                        point_size=5, render_points_as_spheres=True)
            pl.add_text(t, font_size=11, color="white")
            pl.camera_position = cam
        pl.link_views()
        shot = os.path.join(OUTPUT_DIR, "compare_rgb_vs_lab.png")
        pl.screenshot(shot)
        print(f"  Screenshot: {shot}")
        pl.show()
    except Exception as e:
        print(f"  PyVista failed ({e}), using Matplotlib")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14,6), facecolor="black")
        for i, (c, t) in enumerate([
            (colors_rgb, f"RGB  (mean ΔE={de_rgb.mean():.1f})"),
            (colors_lab, f"CIELAB  (mean ΔE={de_lab.mean():.1f})"),
        ]):
            ax = fig.add_subplot(1,2,i+1,projection="3d")
            ax.scatter(coords[:,0],coords[:,1],coords[:,2],c=c,s=1)
            ax.set_title(t, color="white"); ax.set_facecolor("black")
        plt.tight_layout()
        shot = os.path.join(OUTPUT_DIR, "compare_rgb_vs_lab.png")
        plt.savefig(shot, dpi=150, facecolor="black")
        print(f"  Screenshot: {shot}")
        plt.show()

if __name__ == "__main__":
    main()
