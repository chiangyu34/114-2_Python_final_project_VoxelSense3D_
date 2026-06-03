"""
src/build_blocks_json.py

Builds data/blocks.json from the Minecraft 1.20 default resource pack.

Scans the block/ folder using _list.json as the file manifest, computes
the average RGB for each block texture (ignoring transparent pixels),
applies biome tinting for blocks that need it, and writes the result
to data/blocks.json.

Usage
-----
    python -m src.build_blocks_json
or  python src/build_blocks_json.py

Output
------
    data/blocks.json  →  {"block_name": [R, G, B], ...}

v5.2 (2026-05) fixes:
  - P3: _compute_avg_color() forces 8-bit RGBA conversion and clips the
    final mean to 0-255 before casting, preventing overflow on 16-bit PNGs.
"""

import json
import os
import sys

import numpy as np
from PIL import Image

# Make config.py importable whether this is run as a script or a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    DATA_DIR, BLOCK_DIR, BLOCKS_JSON, LIST_JSON,
)


# ============================================================================
# Filter rules — module constants
# ============================================================================

FACE_SUFFIXES = ["_top", "_side", "_bottom", "_front",
                 "_back", "_inner", "_outer"]

FACE_PRIORITY = {"_top": 0, "_side": 1, "_front": 2, None: 3}

SKIP_SUFFIXES = {
    "_lit", "_on", "_off", "_open", "_flow", "_overlay",
}

SKIP_EXACT = {"debug", "debug2"}

# ── Biome tinting constants ────────────────────────────────────────────────
GRASS_TINT   = np.array([121, 192,  90], dtype=float) / 255.0
FOLIAGE_TINT = np.array([119, 171,  47], dtype=float) / 255.0

GRASS_TINTED_BLOCKS = {
    "grass_block_top",  "grass_block_side",
    "tall_grass_top",   "tall_grass_bottom",
    "fern",             "large_fern_top",   "large_fern_bottom",
    "sugar_cane",       "grass",
}

FOLIAGE_TINTED_BLOCKS = {
    "oak_leaves",      "acacia_leaves",   "dark_oak_leaves",
    "jungle_leaves",   "azalea_leaves",   "mangrove_leaves",
    "cherry_leaves",   "vine",
}


# ============================================================================
# PaletteBuilder
# ============================================================================

class PaletteBuilder:
    """Reads block textures, filters & groups them, then writes blocks.json."""

    def __init__(self,
                 block_dir: str = BLOCK_DIR,
                 list_json: str = LIST_JSON,
                 output_path: str = BLOCKS_JSON):
        self.block_dir   = block_dir
        self.list_json   = list_json
        self.output_path = output_path

    def build(self) -> dict[str, list[int]]:
        if not os.path.isdir(self.block_dir):
            raise FileNotFoundError(f"block/ folder not found: {self.block_dir}")

        all_files   = self._load_manifest()
        png_stems   = self._strip_pngs(all_files)
        valid_stems = self._apply_skip_rules(png_stems)
        groups      = self._group_by_root(valid_stems)
        palette     = self._build_palette(groups)

        self._write_json(palette)
        return palette

    def _load_manifest(self) -> list[str]:
        if not os.path.exists(self.list_json):
            raise FileNotFoundError(
                f"_list.json not found at {self.list_json}\n"
                "Place the _list.json that came with the ZIP inside block/."
            )
        with open(self.list_json, encoding="utf-8") as f:
            manifest = json.load(f)
        files = manifest["files"]
        print(f"[manifest]  {len(files)} files listed in _list.json")
        return files

    @staticmethod
    def _strip_pngs(all_files: list[str]) -> list[str]:
        stems = [
            f[:-4]
            for f in all_files
            if f.endswith(".png") and not f.endswith(".png.mcmeta")
        ]
        print(f"[filter]    {len(stems)} PNG files after dropping .mcmeta")
        return stems

    def _apply_skip_rules(self, png_stems: list[str]) -> list[str]:
        valid = [s for s in png_stems if not self._is_skipped(s)]
        skipped = len(png_stems) - len(valid)
        print(f"[filter]    {len(valid)} kept  |  {skipped} skipped "
              f"(animation frames, crop stages, state variants)")
        return valid

    @staticmethod
    def _is_skipped(stem: str) -> bool:
        if stem in SKIP_EXACT:
            return True
        if stem.startswith("destroy_stage_"):
            return True
        for s in SKIP_SUFFIXES:
            if stem.endswith(s):
                return True
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return True
        return False

    @staticmethod
    def _get_face_suffix(stem: str) -> str | None:
        for s in FACE_SUFFIXES:
            if stem.endswith(s):
                return s
        return None

    def _group_by_root(self, stems: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for stem in stems:
            face = self._get_face_suffix(stem)
            root = stem[: -len(face)] if face else stem
            groups.setdefault(root, []).append(stem)
        print(f"[grouping]  {len(groups)} unique block roots "
              f"after merging face variants")
        return groups

    @staticmethod
    def _pick_best_face(candidates: list[str]) -> str:
        def priority(stem: str) -> tuple[int, str]:
            face = PaletteBuilder._get_face_suffix(stem)
            rank = FACE_PRIORITY.get(face, 4)
            return (rank, stem)
        return min(candidates, key=priority)

    @staticmethod
    def _compute_avg_color(filepath: str) -> list[int] | None:
        """Average RGB of opaque pixels (alpha > 128), or None if all transparent.

        v5.2 (P3 fix): forces 8-bit RGBA regardless of source bit depth.
        Some third-party resource packs use 16-bit PNGs; without this
        normalisation the np.uint8 cast at the end would overflow.
        """
        img = Image.open(filepath)
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        data = np.array(img, dtype=np.float32)
        if data.ndim != 3 or data.shape[2] < 4:
            return None

        opaque_mask   = data[:, :, 3] > 128
        opaque_pixels = data[opaque_mask]
        if len(opaque_pixels) == 0:
            return None

        rgb_mean = opaque_pixels[:, :3].mean(axis=0)
        rgb_mean = np.clip(rgb_mean, 0, 255)
        return [int(round(float(rgb_mean[0]))),
                int(round(float(rgb_mean[1]))),
                int(round(float(rgb_mean[2])))]

    @staticmethod
    def _apply_biome_tint(stem: str, rgb: list[int]) -> list[int]:
        arr = np.array(rgb, dtype=float) / 255.0
        if stem in GRASS_TINTED_BLOCKS:
            tinted = arr * GRASS_TINT * 255.0
        elif stem in FOLIAGE_TINTED_BLOCKS:
            tinted = arr * FOLIAGE_TINT * 255.0
        else:
            return rgb
        return [int(round(float(v))) for v in tinted]

    def _build_palette(self, groups: dict[str, list[str]]
                       ) -> dict[str, list[int]]:
        palette: dict[str, list[int]] = {}
        skipped_transparent = 0

        for root, candidates in sorted(groups.items()):
            chosen   = self._pick_best_face(candidates)
            filepath = os.path.join(self.block_dir, chosen + ".png")
            color    = self._compute_avg_color(filepath)

            if color is None:
                skipped_transparent += 1
                print(f"  [skip-transparent]  {chosen}")
                continue

            palette[chosen] = self._apply_biome_tint(chosen, color)

        print(f"[palette]   {len(palette)} blocks written  |  "
              f"{skipped_transparent} skipped (fully transparent)")
        return palette

    def _write_json(self, palette: dict[str, list[int]]) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(palette, f, indent=2)
        print(f"\n[output]    {self.output_path}")
        self._spot_check(palette)

    @staticmethod
    def _spot_check(palette: dict[str, list[int]]) -> None:
        print("\n[spot-check] Verify these colors look reasonable:")
        spot = [
            ("grass_block_top",  "should be medium green"),
            ("dirt",             "should be brown"),
            ("stone",            "should be mid grey"),
            ("sand",             "should be warm beige"),
            ("oak_planks",       "should be pale tan"),
            ("oak_log",          "should be brown bark"),
            ("oak_log_top",      "should be tan end-grain"),
            ("bricks",           "should be terracotta red"),
            ("diamond_block",    "should be cyan"),
            ("redstone_block",   "should be red"),
        ]
        for name, hint in spot:
            if name in palette:
                r, g, b = palette[name]
                print(f"  {name:<30}  RGB({r:3d}, {g:3d}, {b:3d})  # {hint}")
            else:
                print(f"  {name:<30}  (not in palette — check texture name)")


def main():
    print("=" * 60)
    print("  VoxelSense 3D  —  build_blocks_json.py")
    print("=" * 60)
    PaletteBuilder().build()
    print("\nDone.  Run main.py to verify the new blocks.json works end-to-end.")


if __name__ == "__main__":
    main()