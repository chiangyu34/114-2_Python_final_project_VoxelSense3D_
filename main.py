"""
main.py — VoxelSense 3D entry point.

Reads INPUT_OBJ, runs ModelLoader → VoxelEngine → ColorMatcher, writes:
    output/coords.npy           voxel coordinates  (N, 3) float
    output/colors.npy           matched RGB        (N, 3) float, 0-1
    output/material_list.csv    block usage statistics

Edit only the SETTINGS block below.

v5.1 (2026-05) fixes:
  - P4: CSV export catches PermissionError so an Excel-locked file
    doesn't kill the whole demo.
  - Friendly top-level error handler around the pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.config   import (OUTPUT_DIR, COORDS_NPY, COLORS_NPY,
                      MATERIAL_CSV, QUALITY_MAP)
from src.pipeline import ModelLoader, VoxelEngine, ColorMatcher
from src.viewer   import show, run_compare


# =========================== SETTINGS ===========================
OBJ_FILE               = input("Voxelizing file in dir models (example: usagi_chiikawa.glb): ")
INPUT_OBJ              = os.path.join(PROJECT_ROOT, "models", OBJ_FILE)
QUALITY                = "medium"                                   # low | medium | high
COMPARE                = os.environ.get("VOXELSENSE_COMPARE") == "1"
USE_MATERIAL_WEIGHTING = False
# ================================================================


def export_csv(assigned_names: list[str], output_path: str) -> None:
    """Writes a sorted block-usage CSV."""
    if len(assigned_names) == 0:
        print("[output] WARNING: no voxels to export — skipping CSV")
        return

    count = Counter(assigned_names)
    total = sum(count.values())
    df = pd.DataFrame([
        {"block_name": name, "count": qty,
         "percent": f"{qty / total * 100:.1f}%"}
        for name, qty in sorted(count.items(), key=lambda x: -x[1])
    ])

    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_path.replace(".csv", f"_{stamp}.csv")
        print(f"[output] WARNING: {output_path} is locked (open in Excel?)")
        print(f"        Writing to {fallback} instead.")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        output_path = fallback

    print(f"[output] Top 10 blocks:")
    print(df.head(10).to_string(index=False))
    print(f"  Total voxels : {total:,}")
    print(f"  Saved to     : {output_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    resolution = QUALITY_MAP[QUALITY]
    print(f"\n[pipeline] quality={QUALITY} ({resolution}^3)")
    print(f"[pipeline] input={INPUT_OBJ}")

    # Stage 1 — Load + bake textures + fix orientation
    mesh = ModelLoader(INPUT_OBJ).load()

    # Stage 2 — Voxelize
    engine = VoxelEngine(mesh, resolution)
    coords, mesh_scaled, final_res = engine.voxelize()
    if final_res != resolution:
        print(f"[pipeline] Resolution auto-reduced "
              f"{resolution} → {final_res} (memory ceiling)")

    # Stage 3 — Sample colors + CIELAB matching
    matcher = ColorMatcher(mesh_scaled, coords,
                           use_material_weighting=USE_MATERIAL_WEIGHTING)
    matcher.sample_voxel_colors()
    assigned_names, matched_colors = matcher.assign()

    # Persist results
    np.save(COORDS_NPY, coords)
    np.save(COLORS_NPY, matched_colors)
    export_csv(assigned_names, MATERIAL_CSV)
    print(f"[output] coords.npy + colors.npy → {OUTPUT_DIR}")

    # Visualise
    if COMPARE:
        run_compare(matcher.palette, matcher.voxel_colors_rgb, coords)
    else:
        show(coords, matched_colors,
             f"VoxelSense 3D — {os.path.basename(INPUT_OBJ)} ({QUALITY})")

    print("\n[pipeline] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("\nIf you're demoing now, you can fall back to:")
        print("  - Open output/voxel_view.html in a browser (pre-generated)")
        print("  - Run: python view_result.py  (uses last successful output)")
        sys.exit(1)