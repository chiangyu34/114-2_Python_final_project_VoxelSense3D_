"""
src/color_engine.py

Block color palette manager.

Loads blocks.json, builds KD-Trees in both CIELAB and RGB space, and
provides fast perceptual color matching for voxel assignment.

Public API
----------
    BlockPalette(json_path)
        .match_lab(voxel_colors_rgb)            CIELAB nearest-neighbour
        .match_lab_weighted(...)                CIELAB + material-family bias
        .match_rgb(voxel_colors_rgb)            RGB nearest-neighbour (baseline)
        .compute_delta_e(orig_rgb, matched_rgb) CIEDE2000 verification
        .find_best_block(rgb)                   single-color lookup

    make_default_voxel_colors(coords, base_color=None)
        Height-shaded pseudo-colors for textureless models.

v5.2 (2026-05) fixes:
  - B1: match_lab_weighted batched for memory safety.
  - B6: empty-input guard on match_lab_weighted.
  - C3: documented family-window overlap behaviour; bonus stacks
    intentionally if a voxel falls into multiple windows (rare in
    practice with the current curated families).
"""

import json
import os
import numpy as np
from skimage import color as skcolor
from scipy.spatial import KDTree

from config import MAX_VOXEL_WEIGHT_BATCH


# ============================================================================
# Material families — for semantic weighting
# ============================================================================

# Each entry maps a semantic tag → curated vanilla blocks that read as that
# material.  A voxel whose CIELAB falls inside `lab_window` receives a
# negative ΔE bonus when matched against blocks in `blocks`, pulling the
# nearest-neighbour search toward the "right" material family.
#
# Bonus values are intentionally small (2-4) so the bias only flips the
# choice between near-equivalent matches (ΔE diff < bonus).  It never
# overrides an obviously better match.
#
# NOTE on overlap (C3): the current three windows are disjoint.  If you
# add a new family whose window overlaps an existing one, the bonus will
# STACK on overlapping voxels — that may or may not be desirable.  Audit
# new entries with a CIELAB scatter plot before merging.
MATERIAL_FAMILIES = {
    "skin": {
        "blocks":  ["white_terracotta", "pink_terracotta",
                    "smooth_sandstone",  "pink_concrete_powder",
                    "white_concrete_powder"],
        "lab_window": {"L": (55, 85), "a": (5,   30), "b": (10, 40)},
        "bonus": 4.0,
    },
    "hair_dark": {
        "blocks":  ["black_concrete", "black_terracotta",
                    "polished_blackstone", "coal_block"],
        "lab_window": {"L": (0, 30), "a": (-5, 10), "b": (-5, 15)},
        "bonus": 3.0,
    },
    "foliage": {
        "blocks":  ["oak_leaves", "moss_block", "grass_block_top",
                    "azalea_leaves", "green_wool"],
        "lab_window": {"L": (30, 70), "a": (-40, -10), "b": (10, 50)},
        "bonus": 2.5,
    },
}


# ============================================================================
# BlockPalette
# ============================================================================

class BlockPalette:
    """Manages the Minecraft block color database.

    Loads block RGB data from blocks.json and exposes:
      - CIELAB KD-Tree matching            (primary, perceptually accurate)
      - CIELAB + material-family weighting (semantic-aware variant)
      - RGB KD-Tree matching               (baseline for comparison)
      - CIEDE2000 ΔE                       (used for report data)
    """

    def __init__(self, json_path: str):
        self.block_names:      list[str]        = []
        self.block_colors:     np.ndarray | None = None  # (N, 3) uint8
        self.block_colors_lab: np.ndarray | None = None  # (N, 3) float32
        self._tree_lab: KDTree | None = None
        self._tree_rgb: KDTree | None = None
        self._load(json_path)

    # ------------------------------------------------------------------
    # Public API — single-color lookup
    # ------------------------------------------------------------------

    def find_best_block(self, rgb) -> tuple[str, float]:
        """Perceptually closest block for a single RGB color (0-255).

        Returns (block_name, ΔE_distance).
        """
        lab = self._rgb_to_lab_single(rgb)
        distance, index = self._tree_lab.query(lab)
        return self.block_names[index], float(distance)

    # ------------------------------------------------------------------
    # Public API — batch matching
    # ------------------------------------------------------------------

    def match_lab(self, voxel_colors_rgb: np.ndarray
                  ) -> tuple[list[str], np.ndarray]:
        """Batch CIELAB matching — primary method, perceptually accurate.

        Parameters
        ----------
        voxel_colors_rgb : (N, 3) array, uint8 or float32, range 0-255

        Returns
        -------
        matched_names  : list of N block name strings
        matched_colors : (N, 3) float32, range 0-1 (ready for PyVista rgb=True)
        """
        if len(voxel_colors_rgb) == 0:
            return [], np.zeros((0, 3), dtype=np.float32)

        lab = self._rgb_to_lab_batch(voxel_colors_rgb.astype(np.float32))
        _, indices = self._tree_lab.query(lab)
        indices = np.asarray(indices).flatten()
        return self._indices_to_result(indices)

    def match_lab_weighted(self, voxel_colors_rgb: np.ndarray,
                           enable: bool = True
                           ) -> tuple[list[str], np.ndarray]:
        """CIELAB matching with optional material-family bias.

        Algorithm
        ---------
        1. Compute Lab for every voxel.
        2. For each MATERIAL_FAMILIES entry whose lab_window contains
           the voxel, subtract `bonus` from the ΔE of every block in
           that family BEFORE argmin.
        3. Argmin over the adjusted distance matrix.

        v5.1 fixes
        ----------
        - B1: process voxels in batches of MAX_VOXEL_WEIGHT_BATCH to keep
          peak memory under ~150MB even at 300K voxels.  Pre-v5.1 a 300K
          × 600 × 3 float32 array was 2.1GB and OOM'd on 8GB laptops.
        - B6: empty-input guard.

        Falls back to plain match_lab() when enable=False.
        """
        if not enable:
            return self.match_lab(voxel_colors_rgb)

        n = len(voxel_colors_rgb)
        if n == 0:
            return [], np.zeros((0, 3), dtype=np.float32)

        all_indices = np.empty(n, dtype=np.int64)

        # ── B1 fix: batched processing ───────────────────────────────────
        # Peak RAM = batch_size × n_blocks × 3 × 4 bytes
        # 50,000 × 600 × 3 × 4 = 360 MB → comfortably under 1GB ceiling
        for start in range(0, n, MAX_VOXEL_WEIGHT_BATCH):
            end   = min(start + MAX_VOXEL_WEIGHT_BATCH, n)
            chunk = voxel_colors_rgb[start:end].astype(np.float32)
            lab   = self._rgb_to_lab_batch(chunk)

            # Pairwise ΔE76 (Euclidean in Lab) — fast ranker
            diff = lab[:, None, :] - self.block_colors_lab[None, :, :]
            de   = np.linalg.norm(diff, axis=2)

            for family in MATERIAL_FAMILIES.values():
                L_lo, L_hi = family["lab_window"]["L"]
                a_lo, a_hi = family["lab_window"]["a"]
                b_lo, b_hi = family["lab_window"]["b"]
                in_window = (
                    (lab[:, 0] >= L_lo) & (lab[:, 0] <= L_hi)
                    & (lab[:, 1] >= a_lo) & (lab[:, 1] <= a_hi)
                    & (lab[:, 2] >= b_lo) & (lab[:, 2] <= b_hi)
                )
                if not in_window.any():
                    continue
                family_idx = [
                    i for i, name in enumerate(self.block_names)
                    if any(sub in name for sub in family["blocks"])
                ]
                if not family_idx:
                    continue
                de[np.ix_(in_window, family_idx)] -= family["bonus"]

            all_indices[start:end] = de.argmin(axis=1)

            del diff, de, lab, chunk

        return self._indices_to_result(all_indices)

    def match_rgb(self, voxel_colors_rgb: np.ndarray
                  ) -> tuple[list[str], np.ndarray]:
        """Batch RGB matching — baseline for comparison screenshots."""
        if len(voxel_colors_rgb) == 0:
            return [], np.zeros((0, 3), dtype=np.float32)

        arr = voxel_colors_rgb.astype(np.float32)
        _, indices = self._tree_rgb.query(arr)
        indices = np.asarray(indices).flatten()
        return self._indices_to_result(indices)

    def compute_delta_e(self,
                        original_rgb: np.ndarray,
                        matched_rgb:  np.ndarray) -> np.ndarray:
        """CIEDE2000 perceptual color difference per voxel pair.

        Used to generate report data comparing RGB vs CIELAB matching.
        ΔE < 5   ≈ visually nearly identical
        ΔE < 2.3 ≈ Just Noticeable Difference (JND)
        """
        if len(original_rgb) == 0:
            return np.zeros(0, dtype=np.float32)

        lab1 = self._rgb_to_lab_batch(original_rgb).reshape(-1, 1, 3)
        lab2 = self._rgb_to_lab_batch(matched_rgb ).reshape(-1, 1, 3)
        return skcolor.deltaE_ciede2000(lab1, lab2).flatten().astype(np.float32)

    def get_random_colors(self, count: int) -> np.ndarray:
        """Randomly assigned block colors.  Used as a Day-04 placeholder."""
        ids = np.random.randint(0, len(self.block_names), size=count)
        return self.block_colors[ids].astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _indices_to_result(self, indices: np.ndarray
                           ) -> tuple[list[str], np.ndarray]:
        """Shared back-half: indices → (names list, normalized colors)."""
        names  = [self.block_names[i] for i in indices]
        colors = self.block_colors[indices].astype(np.float32) / 255.0
        return names, colors

    def _load(self, json_path: str) -> None:
        """Reads blocks.json and builds both KD-Trees."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"blocks.json not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        self.block_names       = list(data.keys())
        self.block_colors      = np.array(list(data.values()), dtype=np.uint8)
        self.block_colors_lab  = self._rgb_to_lab_batch(
                                     self.block_colors.astype(np.float32))

        self._tree_lab = KDTree(self.block_colors_lab)
        self._tree_rgb = KDTree(self.block_colors.astype(np.float32))

        print(f"[BlockPalette] Loaded {len(self.block_names)} blocks "
              f"(CIELAB + RGB KD-Trees ready)")

    @staticmethod
    def _rgb_to_lab_single(rgb) -> np.ndarray:
        """Single [R, G, B] (0-255) → CIELAB (shape (3,))."""
        rgb_norm = np.array(rgb, dtype=float) / 255.0
        lab = skcolor.rgb2lab(rgb_norm.reshape(1, 1, 3))
        return lab[0, 0].astype(np.float32)

    @staticmethod
    def _rgb_to_lab_batch(rgb_array: np.ndarray) -> np.ndarray:
        """Vectorised (N, 3) RGB → CIELAB.

        skimage.rgb2lab accepts arbitrary (..., 3) shapes; we reshape to
        (1, N, 3) for one batch call, then flatten back to (N, 3).
        Input must be in [0, 1] — we divide by 255 here.
        """
        rgb_01 = rgb_array / 255.0
        lab = skcolor.rgb2lab(rgb_01.reshape(1, -1, 3)).reshape(-1, 3)
        return lab.astype(np.float32)

    def __len__(self)  -> int: return len(self.block_names)
    def __repr__(self) -> str: return f"BlockPalette({len(self)} blocks)"


# ============================================================================
# Pseudo-color generator (module-level utility)
# ============================================================================

def make_default_voxel_colors(coords:     np.ndarray,
                              base_color: list | None = None) -> np.ndarray:
    """Generates pseudo-colors from voxel height when the model has no texture.

    Applies a vertical ambient-occlusion-style shading: bottom voxels are
    darker (factor 0.6), top voxels are lighter (factor 1.0).  This gives
    the voxel model a sense of depth even without real surface colors.
    """
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    base = np.array(base_color if base_color else [160, 120, 80],
                    dtype=np.float32)

    y = coords[:, 1]
    y_min, y_max = y.min(), y.max()
    shade = (np.ones(len(coords)) if y_max == y_min
             else 0.6 + 0.4 * (y - y_min) / (y_max - y_min))

    return np.outer(shade, base).clip(0, 255).astype(np.uint8)