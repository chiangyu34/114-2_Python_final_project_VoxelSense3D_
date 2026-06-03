"""
src/pipeline.py

Three-class façade over geometry.py + color_engine.py.

    ModelLoader   →  load + bake textures + fix orientation
    VoxelEngine   →  scale + fill + memory-safe downsample
    ColorMatcher  →  barycentric sample + CIELAB matching (+ optional weighting)

Each class is a thin wrapper that holds intermediate state so main.py
reads top-to-bottom like a recipe, and each stage can be unit-tested
in isolation.

v5.1 (2026-05) fixes:
  - B4: ModelLoader catches "No Trimesh geometry" and re-raises with
    actionable error text pointing the user to check their .glb file.
  - ColorMatcher.assign() guards against empty voxel arrays so a
    degenerate input doesn't crash the demo silently.
"""

import os
import numpy as np
import trimesh

from geometry import (
    load_model, Voxelizer, SurfaceColorSampler,
)
from color_engine import BlockPalette, make_default_voxel_colors
from config import BLOCKS_JSON


class ModelLoader:
    """Loads a 3D asset and produces a single merged Trimesh with vertex
    colors baked from PBR textures.

    Delegates to geometry.load_model(), which handles Scene → Trimesh
    concatenation, PBR UV sampling, baseColorFactor fallback, Z-up
    correction, and the _is_default_gray() gate.
    """

    def __init__(self, obj_path: str):
        if not os.path.exists(obj_path):
            raise FileNotFoundError(
                f"Model not found: {obj_path}\n"
                f"Check that the file exists and the path in main.py's "
                f"INPUT_OBJ setting is correct."
            )
        self.obj_path: str                  = obj_path
        self.mesh:    trimesh.Trimesh | None = None

    def load(self) -> trimesh.Trimesh:
        """Idempotent — caches the result on first call."""
        if self.mesh is not None:
            return self.mesh

        try:
            self.mesh = load_model(self.obj_path)
        except ValueError as e:
            if "No Trimesh geometry" in str(e):
                raise ValueError(
                    f"The file {os.path.basename(self.obj_path)} contains no "
                    f"3D geometry — only lights/cameras/empty nodes.\n"
                    f"Try a different model file."
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {os.path.basename(self.obj_path)}: "
                f"{type(e).__name__}: {e}"
            ) from e

        return self.mesh


class VoxelEngine:
    """Voxelizes a loaded mesh.

    Exposes both `coords` (voxel centres) and `mesh_scaled` (the scaled
    working copy in the same coord space).  ColorMatcher needs both,
    because its KDTree must be queried with points in the SAME coordinate
    space as mesh.vertices — voxelize() scales the mesh to 0..resolution,
    so passing the original mesh would collapse all colours to a single
    grey block.
    """

    def __init__(self, mesh: trimesh.Trimesh, resolution: int = 64):
        self.mesh                            = mesh
        self.resolution                      = resolution
        self.coords:      np.ndarray | None  = None
        self.mesh_scaled: trimesh.Trimesh | None = None
        self.final_res:   int | None         = None

    def voxelize(self) -> tuple[np.ndarray, trimesh.Trimesh, int]:
        points, _, work, final_res = Voxelizer(self.mesh,
                                               self.resolution).voxelize()
        self.coords      = points
        self.mesh_scaled = work
        self.final_res   = final_res
        return points, work, final_res


class ColorMatcher:
    """Per-voxel block assignment via CIELAB nearest-neighbour.

    Pipeline:
      1. Sample per-voxel RGB from the mesh surface (barycentric).
      2. Fall back to height-shaded pseudo-colors if the mesh has none.
      3. Convert all voxel RGBs to CIELAB and query the palette KDTree.
      4. Optionally apply material-family weighting.

    Optional flag `use_material_weighting` enables semantic-aware matching:
    skin tones bias toward smooth_sandstone/terracotta, dark hair biases
    toward black_concrete/coal_block, etc.
    """

    def __init__(self,
                 mesh_scaled:  trimesh.Trimesh,
                 coords:       np.ndarray,
                 palette_path: str = BLOCKS_JSON,
                 use_material_weighting: bool = False):
        self.mesh_scaled            = mesh_scaled
        self.coords                 = coords
        self.palette                = BlockPalette(palette_path)
        self.use_material_weighting = use_material_weighting
        self.voxel_colors_rgb:  np.ndarray | None  = None
        self.assigned_names:    list[str] | None   = None
        self.matched_colors:    np.ndarray | None  = None

    def sample_voxel_colors(self) -> np.ndarray:
        sampled = SurfaceColorSampler(self.mesh_scaled, self.coords).sample()
        if sampled is None:
            print("[ColorMatcher] No real colors found — using pseudo-colors")
            sampled = make_default_voxel_colors(self.coords)
        self.voxel_colors_rgb = sampled.astype(np.float32)
        return self.voxel_colors_rgb

    def assign(self) -> tuple[list[str], np.ndarray]:
        if self.voxel_colors_rgb is None:
            self.sample_voxel_colors()

        if len(self.voxel_colors_rgb) == 0:
            print("[ColorMatcher] WARNING: 0 voxels to match")
            self.assigned_names = []
            self.matched_colors = np.zeros((0, 3), dtype=np.float32)
            return self.assigned_names, self.matched_colors

        if self.use_material_weighting:
            names, colors = self.palette.match_lab_weighted(
                self.voxel_colors_rgb, enable=True)
        else:
            names, colors = self.palette.match_lab(self.voxel_colors_rgb)

        self.assigned_names = names
        self.matched_colors = colors
        return names, colors