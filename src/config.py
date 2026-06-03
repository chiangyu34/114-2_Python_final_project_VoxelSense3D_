"""
src/config.py

Centralised path management for VoxelSense 3D.

All other modules import paths from here.  Hard-coding paths inside
scripts breaks reproducibility when the project is moved or cloned;
keeping every directory string in one file makes the layout explicit
and one-line-changeable.

v5.2 (2026-05) — same as v5.1; verified no further bugs in this file.

Project layout
--------------
VoxelSense3D_/
├── src/                      Python modules (this file lives here)
├── data/
│   └── blocks.json           Pre-computed palette: name → [R,G,B]
├── block/                    Raw Minecraft texture PNGs + _list.json
├── models/                   User-supplied .obj / .glb / .gltf
└── output/                   Generated: coords / colors / *.csv / *.html
"""

import os

# Project root is one level up from this file.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folders
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
BLOCK_DIR    = os.path.join(PROJECT_ROOT, "block")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")

# Specific files
BLOCKS_JSON  = os.path.join(DATA_DIR,   "blocks.json")
LIST_JSON    = os.path.join(BLOCK_DIR,  "_list.json")

# Output files (used by main.py and view_result.py)
COORDS_NPY   = os.path.join(OUTPUT_DIR, "coords.npy")
COLORS_NPY   = os.path.join(OUTPUT_DIR, "colors.npy")
MATERIAL_CSV = os.path.join(OUTPUT_DIR, "material_list.csv")
HTML_VIEWER  = os.path.join(OUTPUT_DIR, "voxel_view.html")

# Pipeline tunables
MAX_VOXELS              = 300_000   # safe_voxelize ceiling — see Voxelizer
GRAY_TOL                = 10.0      # ±tolerance for _is_default_gray (mean 128)
MAX_VOXEL_WEIGHT_BATCH  = 50_000    # batch size for match_lab_weighted's
                                    # (N × ~600) ΔE matrix.  Keeps peak RAM
                                    # under ~150 MB even at 300K voxels.

QUALITY_MAP  = {
    "low":               32,
    "medium":            64,
    "high":             128,
    "extra_high":       256,
    "__extreme_high": 512,
}


def ensure_output_dirs() -> None:
    """Create OUTPUT_DIR and DATA_DIR if they don't exist.

    Called at import time below.  Idempotent.  Prevents the race where
    build_blocks_json.py is invoked before main.py has had a chance to
    os.makedirs the output folder.
    """
    for d in (DATA_DIR, OUTPUT_DIR):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            # Don't crash on import — let the caller decide
            print(f"[config] WARNING: could not create {d}: {e}")


# Run on import so any consumer can rely on the folders existing.
ensure_output_dirs()