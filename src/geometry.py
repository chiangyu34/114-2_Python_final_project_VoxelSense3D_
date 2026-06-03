"""
src/geometry.py

Geometry pipeline for VoxelSense 3D — loads .obj/.glb/.gltf, voxelizes,
and samples per-voxel surface colors with full PBR texture support.

Fix log
-------
v4 (2026-05):   _is_default_gray gate replaces std-based rejection.
v5 (2026-05):   reorganised into four classes.
v5.1 (2026-05): process=True for face_normals, degenerate-extent guard.
v5.2 (2026-05): C2 degenerate triangle vertex-mean fallback.
v5.3 (2026-05): _manual_concat_with_colors fixes "all one color" bug.
v5.4 (2026-05): Fixes the "to_color() returns flat white for textured body meshes" bug.

v5.5 (2026-05):  ← THIS VERSION
  The "Hachiware body stays gray" bug.

  Symptom
  -------
  Chiikawa & Hachiware bodies — whose materials are TextureVisuals but
  whose `material.baseColorTexture.image` is somehow not findable by
  _find_any_image() — were falling all the way to Case 4 safety net,
  which returned [128,128,128] gray because _material_avg_color also
  couldn't find a texture image.  Result: every body voxel CIELAB-matched
  to "tripwire" (gray) = ~30% of voxels.

  Cause
  -----
  Two issues compounding:
    1. _find_any_image walks attribute names but skips properties whose
       getter raises an exception (some trimesh material objects raise
       on access to lazily-loaded texture slots).
    2. When the image really IS unfindable, we used to_color()'s output
       to detect "extreme" results and reject them — but then fell back
       to a safety net that returned EVEN WORSE [128,128,128] gray.

  Fix
  ---
  Case 3 (to_color) and Case 4 (safety net) now coordinate:
    - to_color result is RECORDED as `_to_color_result`.
    - Safety net first asks _material_avg_color for a non-gray colour.
    - If safety net would also produce gray, we KEEP the to_color result
      even if it was "uniform-extreme" — a uniform white body is still
      visually correct for Chiikawa, and far better than uniform gray
      mapping to tripwire.

  Plus: _find_any_image now wraps every getattr in try/except so lazy
  texture loaders don't break the walk.
"""

import os
import numpy as np
import trimesh
import trimesh.proximity
import trimesh.repair

from config import MAX_VOXELS, GRAY_TOL


# ============================================================================
# Internal helpers
# ============================================================================

def _color_std(vertex_colors: np.ndarray) -> float:
    """std of RGB channels across all vertices, in the 0-255 range."""
    if vertex_colors is None or len(vertex_colors) == 0:
        return 0.0
    return float(vertex_colors[:, :3].astype(np.float32).std())


def _is_default_gray(vertex_colors: np.ndarray) -> bool:
    """True when mean RGB ≈ [128,128,128] (trimesh placeholder)."""
    if vertex_colors is None or len(vertex_colors) == 0:
        return True
    mean_rgb = vertex_colors[:, :3].astype(np.float32).mean(axis=0)
    return bool(np.all(np.abs(mean_rgb - 128.0) < GRAY_TOL))


def _is_uniform_extreme(vertex_colors: np.ndarray) -> bool:
    """True when colour is uniform AND looks like a trimesh "lost detail"
    fallback.

    v5.6: expanded to catch "near-gray uniform" outputs that CIELAB-match
    to tripwire.  Previously we only caught pure white/black; but trimesh's
    to_color() often returns something like [140,130,125] (slightly off-gray
    material mean) which:
      - passes _is_default_gray (mean ≈ 131, beyond ±10 of 128)
      - CIELAB-matches to tripwire (129,129,129) at ΔE=0.39
    Result: every voxel becomes tripwire.

    Detection criteria (any one triggers):
      - uniform (std == 0) AND mean is pure white/black (extreme luminance)
      - uniform (std == 0) AND nearly desaturated (channel diff < 25)
        AND mean luminance is in the "suspicious mid-gray" zone (60-200)
    """
    if vertex_colors is None or len(vertex_colors) == 0:
        return False
    arr  = vertex_colors[:, :3].astype(np.float32)
    if arr.std() > 1.0:
        return False
    mean = arr.mean(axis=0)

    # Pure white or pure black uniform → lost detail
    if mean.mean() > 240 or mean.mean() < 15:
        return True

    # v5.6: near-desaturated uniform in the mid-luminance zone
    # — this is what trimesh's to_color() typically produces when it
    # averaged a textured material into a single value.
    channel_spread = mean.max() - mean.min()
    if channel_spread < 25 and 60 < mean.mean() < 200:
        return True

    return False

def _safe_getattr(obj, name):
    """getattr that swallows exceptions from lazy property getters."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _find_any_image(material):
    """Walks EVERY attribute of the material looking for a PIL image.

    v5.5: wraps every getattr in _safe_getattr so lazily-loaded texture
    slots that raise on access don't terminate the walk.
    """
    visited = set()

    def _is_image_like(obj) -> bool:
        return (hasattr(obj, "size")
                and hasattr(obj, "convert")
                and not isinstance(obj, type))

    def _probe(obj, depth=0):
        if depth > 3 or id(obj) in visited:
            return None
        visited.add(id(obj))

        if _is_image_like(obj):
            try:
                _ = obj.size
                return obj
            except Exception:
                pass

        for name in dir(obj):
            if name.startswith("_"):
                continue
            val = _safe_getattr(obj, name)
            if val is None or callable(val):
                continue
            if isinstance(val, (int, float, str, bool, list, tuple, dict,
                                np.ndarray, type)):
                continue
            result = _probe(val, depth + 1)
            if result is not None:
                return result
        return None

    try:
        return _probe(material)
    except Exception:
        return None


def _material_avg_color(visual) -> np.ndarray:
    """Best-effort dominant color of a sub-mesh visual.  Returns uint8 RGBA.

    Order:
      1. Average of opaque texture pixels (alpha > 128)
      2. baseColorFactor RGBA from PBR material
      3. Mid-gray [128,128,128,255] — last resort (avoided by v5.5
         coordination with to_color result)
    """
    try:
        if isinstance(visual, trimesh.visual.TextureVisuals):
            mat = visual.material
            img = _find_any_image(mat)
            if img is not None:
                arr  = np.array(img.convert("RGBA"), dtype=np.uint8)
                mask = arr[:, :, 3] > 128
                if mask.any():
                    avg = arr[mask, :3].mean(axis=0)
                    return np.array([avg[0], avg[1], avg[2], 255],
                                    dtype=np.uint8)
            bcf = _safe_getattr(mat, "baseColorFactor")
            if bcf is not None:
                c = np.asarray(bcf, dtype=np.float32)
                if c.max() <= 1.0:
                    c = c * 255
                return (c[:4].astype(np.uint8) if len(c) >= 4
                        else np.array([c[0], c[1], c[2], 255], dtype=np.uint8))
    except Exception:
        pass
    return np.array([128, 128, 128, 255], dtype=np.uint8)


# ============================================================================
# Class 1 — TextureBaker
# ============================================================================

class TextureBaker:
    """Converts any trimesh visual type into vertex colors.

    Strategy priority:
      1. ColorVisuals with real data (not default gray)
      2a. TextureVisuals + image + UV → UV-sample
      2b. TextureVisuals + baseColorFactor (no image) → solid fill
      3. to_color() — record result, accept if not gray/extreme
      4. Safety net — coordinate with Case 3: if safety net is gray,
         keep the to_color result instead (v5.5 fix)
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        # v5.5: remember the to_color result so safety net can fall back to it
        self._to_color_result: np.ndarray | None = None

    def bake(self) -> trimesh.Trimesh:
        visual = self.mesh.visual

        # ── Case 1: ColorVisuals with real data ──────────────────────────
        if isinstance(visual, trimesh.visual.ColorVisuals):
            vc = visual.vertex_colors
            if (vc is not None
                    and len(vc) == len(self.mesh.vertices)
                    and not _is_default_gray(vc)):
                return self.mesh
            std = _color_std(vc) if vc is not None else 0.0
            print(f"    [bake] ColorVisuals is default gray "
                  f"(std={std:.1f})  →  trying to_color()")

        # ── Case 2: TextureVisuals ───────────────────────────────────────
        elif isinstance(visual, trimesh.visual.TextureVisuals):
            if self._try_uv_sample(visual):
                return self.mesh
            if self._try_base_color_factor(visual):
                return self.mesh

        # ── Case 3: trimesh's to_color() ─────────────────────────────────
        if self._try_to_color():
            return self.mesh

        # ── Case 4: safety net ───────────────────────────────────────────
        self._apply_safety_net()
        return self.mesh

    def _try_uv_sample(self, visual) -> bool:
        try:
            material = visual.material
            image    = _find_any_image(material)
            if image is None or not hasattr(visual, "uv") or visual.uv is None:
                return False

            uv  = np.array(visual.uv, dtype=np.float64)
            img = np.array(image.convert("RGBA"), dtype=np.uint8)
            H, W = img.shape[:2]
            imgf = img.astype(np.float32)

            u = np.clip(uv[:, 0],       0.0, 1.0)
            v = np.clip(1.0 - uv[:, 1], 0.0, 1.0)

            uf = u * (W - 1)
            vf = v * (H - 1)
            x0 = np.round(uf).astype(int).clip(0, W - 1)
            y0 = np.round(vf).astype(int).clip(0, H - 1)
            sampled = imgf[y0, x0].clip(0, 255).astype(np.uint8)

            std_sampled = _color_std(sampled)
            print(f"    [bake] UV-sampled {H}×{W} texture  std={std_sampled:.1f}")
            self.mesh.visual = trimesh.visual.ColorVisuals(
                mesh=self.mesh, vertex_colors=sampled
            )
            return True

        except Exception as e:
            print(f"    [bake] UV sampling failed "
                  f"({type(e).__name__}: {e}) — trying to_color()")
            return False

    def _try_base_color_factor(self, visual) -> bool:
        material = visual.material
        if _find_any_image(material) is not None:
            return False

        bcf = _safe_getattr(material, "baseColorFactor")
        if bcf is None:
            return False

        c = np.asarray(bcf, dtype=np.float32)
        if c.max() <= 1.0:
            c = c * 255
        rgba = (c[:4].astype(np.uint8) if len(c) >= 4
                else np.array([c[0], c[1], c[2], 255], dtype=np.uint8))
        if _is_default_gray(rgba.reshape(1, -1)):
            return False

        uniform = np.tile(rgba, (len(self.mesh.vertices), 1))
        self.mesh.visual = trimesh.visual.ColorVisuals(
            mesh=self.mesh, vertex_colors=uniform
        )
        print(f"    [bake] baseColorFactor (solid color)  rgba={rgba}")
        return True

    def _try_to_color(self) -> bool:
        """v5.5: record the result for safety-net coordination."""
        try:
            self.mesh.visual = self.mesh.visual.to_color()
            vc = self.mesh.visual.vertex_colors
            std        = _color_std(vc) if vc is not None else 0.0
            is_gray    = _is_default_gray(vc) if vc is not None else True
            is_extreme = _is_uniform_extreme(vc) if vc is not None else False

            print(f"    [bake] to_color() fallback  std={std:.1f}  "
                  f"gray={is_gray}  uniform_extreme={is_extreme}")

            # Record for potential fallback in safety net
            if (vc is not None and len(vc) == len(self.mesh.vertices)):
                self._to_color_result = vc.astype(np.uint8).copy()

            if (vc is not None
                    and len(vc) == len(self.mesh.vertices)
                    and not is_gray
                    and not is_extreme):
                return True

            if is_extreme:
                print(f"    [bake] to_color() result is uniform-extreme "
                      f"(may be lost detail) — trying safety net first")
        except Exception as e:
            print(f"    [bake] to_color() also failed ({e})")
        return False

    def _apply_safety_net(self) -> None:
        """v5.6: better tie-breaking when both to_color and material_avg fail.

        Decision tree:
        1. material_avg_color returns non-gray → use it (best case)
        2. _to_color_result is non-gray-zone → keep it
        3. _to_color_result is near-gray → REPLACE with a sensible
            off-palette colour:
            - if it's brightish (luma > 150) → cream/white
            - if it's darkish  (luma < 100) → dark grey/black
            - otherwise → light cream (rare middle case)
            Cream maps to bone_block_top or white_terracotta in CIELAB,
            which is far better than tripwire.
        """
        avg_rgba = _material_avg_color(self.mesh.visual)
        avg_is_gray = bool(np.all(np.abs(
            avg_rgba[:3].astype(np.float32) - 128.0) < GRAY_TOL))

        # ── Case A: material_avg gave us a real colour ────────────────
        if not avg_is_gray:
            fallback = np.tile(avg_rgba, (len(self.mesh.vertices), 1))
            self.mesh.visual = trimesh.visual.ColorVisuals(
                mesh=self.mesh, vertex_colors=fallback
            )
            std_fb = _color_std(fallback)
            print(f"    [bake] material-avg safety net  rgba={avg_rgba}  "
                f"std={std_fb:.1f}  ({len(fallback)} verts)")
            return

        # ── Case B: material_avg gave gray, try _to_color_result ──────
        if self._to_color_result is not None:
            tc_rgb = self._to_color_result[:, :3].astype(np.float32).mean(axis=0)

            # Check if to_color is actually useful (not near-gray)
            channel_spread = tc_rgb.max() - tc_rgb.min()
            tc_is_near_gray = (channel_spread < 25
                            and 60 < tc_rgb.mean() < 200)

            if not tc_is_near_gray:
                # to_color is colourful enough — keep it
                self.mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=self.mesh, vertex_colors=self._to_color_result
                )
                std_tc = _color_std(self._to_color_result)
                print(f"    [bake] safety-net coord: keeping to_color "
                    f"(mean RGB={tc_rgb})  std={std_tc:.1f}")
                return

            # ── Case C: to_color was near-gray — pick a sensible fallback ─
            # Use luminance to decide between white-ish and black-ish
            luma = tc_rgb.mean()
            if luma > 150:
                chosen = np.array([245, 240, 230, 255], dtype=np.uint8)  # cream
                label  = "cream (luma > 150)"
            elif luma < 100:
                chosen = np.array([30, 30, 30, 255], dtype=np.uint8)     # near-black
                label  = "near-black (luma < 100)"
            else:
                chosen = np.array([220, 215, 205, 255], dtype=np.uint8)  # light cream
                label  = "light cream (mid luma)"

            fallback = np.tile(chosen, (len(self.mesh.vertices), 1))
            self.mesh.visual = trimesh.visual.ColorVisuals(
                mesh=self.mesh, vertex_colors=fallback
            )
            print(f"    [bake] safety-net rescue: to_color was near-gray "
                f"(RGB={tc_rgb})  →  forcing {label}  rgba={chosen}")
            return

        # ── Case D: everything failed, use gray as last resort ────────
        fallback = np.tile(avg_rgba, (len(self.mesh.vertices), 1))
        self.mesh.visual = trimesh.visual.ColorVisuals(
            mesh=self.mesh, vertex_colors=fallback
        )
        print(f"    [bake] material-avg safety net (last resort)  "
            f"rgba={avg_rgba}  ({len(fallback)} verts)")

# ============================================================================
# Class 2 — MeshOrientation
# ============================================================================

class MeshOrientation:
    """Detects and fixes common GLB Z-up vs Y-up mismatches."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh

    def fix(self) -> trimesh.Trimesh:
        ex, ey, ez = self.mesh.extents
        if ez > ey * 1.5:
            R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            self.mesh.apply_transform(R)
            print("[geometry] Orientation: Z-up detected, rotated to Y-up")
        return self.mesh


# ============================================================================
# Class 3 — Voxelizer
# ============================================================================

class Voxelizer:
    """Scales a mesh to fit a cube of `resolution` voxels per side, then
    fills it into a solid voxel grid.
    """

    def __init__(self, mesh: trimesh.Trimesh, resolution: int = 32):
        self.mesh = mesh
        self.resolution = resolution

    def voxelize(self) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh, int]:
        ex = self.mesh.extents
        if ex.min() <= 0:
            raise ValueError(
                f"Mesh has a degenerate (zero) extent on at least one axis: "
                f"{tuple(ex)}.  Check your input file."
            )

        max_side = ex.max()
        if max_side == 0:
            raise ValueError("Mesh has zero extent on all axes.")

        work = self._build_geometry_only_copy()
        work.apply_scale(self.resolution / max_side)

        voxel_grid = work.voxelized(pitch=1.0)

        if voxel_grid.matrix.sum() == 0:
            raise ValueError("Voxelization produced 0 occupied voxels.")

        voxel_grid = self._try_solid_fill(voxel_grid, work)

        n_voxels = int(voxel_grid.matrix.sum())
        if n_voxels > MAX_VOXELS:
            half_res = max(self.resolution // 2, 16)
            print(f"[geometry] ⚠ {n_voxels:,} voxels > {MAX_VOXELS:,} limit  "
                  f"— auto-halving resolution {self.resolution} → {half_res}")
            sub = Voxelizer(self.mesh, half_res)
            return sub.voxelize()

        points = voxel_grid.points
        matrix = voxel_grid.matrix
        print(f"[geometry] Result : {len(points):,} voxels  grid {matrix.shape}")
        return points, matrix, work, self.resolution

    def _build_geometry_only_copy(self) -> trimesh.Trimesh:
        """v5.3 critical: process=False so vertex order is preserved."""
        work = trimesh.Trimesh(
            vertices=self.mesh.vertices.copy(),
            faces=self.mesh.faces.copy(),
            process=False,
        )
        if (hasattr(self.mesh.visual, "vertex_colors")
                and self.mesh.visual.vertex_colors is not None
                and len(self.mesh.visual.vertex_colors) == len(self.mesh.vertices)):
            work.visual = trimesh.visual.ColorVisuals(
                mesh=work,
                vertex_colors=self.mesh.visual.vertex_colors.copy(),
            )
        return work

    @staticmethod
    def _try_solid_fill(voxel_grid, work):
        filled = False

        try:
            candidate = voxel_grid.fill()
            if candidate.matrix.sum() > voxel_grid.matrix.sum() * 1.5:
                voxel_grid = candidate
                filled = True
                print("[geometry] Fill   : solid")
        except Exception:
            pass

        if not filled:
            try:
                trimesh.repair.fill_holes(work)
                candidate = work.voxelized(pitch=1.0).fill()
                if candidate.matrix.sum() > voxel_grid.matrix.sum() * 1.5:
                    voxel_grid = candidate
                    filled = True
                    print("[geometry] Fill   : repaired + solid")
            except Exception:
                pass

        if not filled:
            print("[geometry] Fill   : surface shell only (mesh not watertight)")

        return voxel_grid


# ============================================================================
# Class 4 — SurfaceColorSampler
# ============================================================================

class SurfaceColorSampler:
    """Per-voxel barycentric color interpolation on the closest triangle."""

    def __init__(self, mesh: trimesh.Trimesh, voxel_points: np.ndarray):
        self.mesh = mesh
        self.voxel_points = voxel_points

    def sample(self) -> np.ndarray | None:
        if self.voxel_points is None or len(self.voxel_points) == 0:
            print("[geometry] WARNING: empty voxel_points — skipping sampling")
            return None

        if not self._has_real_color():
            return None

        result = self._barycentric()
        if result is not None:
            return result
        return self._kdtree_fallback()

    def _has_real_color(self) -> bool:
        has_color = (
            hasattr(self.mesh.visual, "vertex_colors")
            and self.mesh.visual.vertex_colors is not None
            and len(self.mesh.visual.vertex_colors) == len(self.mesh.vertices)
        )
        if not has_color:
            print("[geometry] No vertex_colors aligned to vertices — pseudo-colors will be used")
            return False

        vc = self.mesh.visual.vertex_colors
        if _is_default_gray(vc):
            std = _color_std(vc)
            print(f"[geometry] Skipping default-gray vertex colors "
                  f"(std={std:.1f}) — pseudo-colors will be used")
            return False

        std = _color_std(vc)
        mean_rgb = vc[:, :3].astype(np.float32).mean(axis=0)
        print(f"[geometry] Real colors found: std={std:.1f}  mean RGB={mean_rgb}")
        return True

    def _barycentric(self) -> np.ndarray | None:
        try:
            import trimesh.proximity as prox

            _, _, face_ids = prox.closest_point(self.mesh, self.voxel_points)
            tri = self.mesh.faces[face_ids]
            A = self.mesh.vertices[tri[:, 0]]
            B = self.mesh.vertices[tri[:, 1]]
            C = self.mesh.vertices[tri[:, 2]]

            v0 = B - A
            v1 = C - A
            v2 = self.voxel_points - A

            d00 = (v0 * v0).sum(1)
            d01 = (v0 * v1).sum(1)
            d11 = (v1 * v1).sum(1)
            d20 = (v2 * v0).sum(1)
            d21 = (v2 * v1).sum(1)

            denom = d00 * d11 - d01 * d01
            DEGEN_EPS = 1e-9
            is_degenerate = np.abs(denom) < DEGEN_EPS
            denom_safe    = np.where(is_degenerate, 1.0, denom)

            w1 = (d11 * d20 - d01 * d21) / denom_safe
            w2 = (d00 * d21 - d01 * d20) / denom_safe
            w0 = 1.0 - w1 - w2

            w0 = np.clip(w0, 0.0, 1.0)
            w1 = np.clip(w1, 0.0, 1.0)
            w2 = np.clip(w2, 0.0, 1.0)
            ws = w0 + w1 + w2
            ws = np.where(ws < 1e-12, 1.0, ws)
            w0 /= ws; w1 /= ws; w2 /= ws

            vc_all = self.mesh.visual.vertex_colors
            c0 = vc_all[tri[:, 0]].astype(np.float32)
            c1 = vc_all[tri[:, 1]].astype(np.float32)
            c2 = vc_all[tri[:, 2]].astype(np.float32)

            interp = (w0[:, None] * c0
                      + w1[:, None] * c1
                      + w2[:, None] * c2)

            if is_degenerate.any():
                mean_c = (c0 + c1 + c2) / 3.0
                interp[is_degenerate] = mean_c[is_degenerate]
                print(f"[geometry] Substituted vertex-mean for "
                      f"{is_degenerate.sum():,} degenerate triangle hits")

            interp_u8 = interp.clip(0, 255).astype(np.uint8)
            colors    = interp_u8[:, :3]
            std_out   = _color_std(colors)
            print(f"[geometry] Sampled (barycentric) {len(colors):,} voxels"
                  f"  std={std_out:.1f}")
            return colors

        except Exception as exc:
            print(f"[geometry] Barycentric sampling failed ({exc})"
                  f" — falling back to KDTree nearest-vertex")
            return None

    def _kdtree_fallback(self) -> np.ndarray | None:
        try:
            from scipy.spatial import KDTree

            tree   = KDTree(self.mesh.vertices)
            _, idx = tree.query(self.voxel_points, workers=1)

            vc_all  = self.mesh.visual.vertex_colors
            colors  = vc_all[idx, :3].astype(np.uint8)
            std_out = _color_std(colors)
            print(f"[geometry] Sampled (KDTree fallback) {len(colors):,} voxels"
                  f"  std={std_out:.1f}")
            return colors

        except Exception as exc2:
            print(f"[geometry] Color sampling failed entirely ({exc2})"
                  f" — using pseudo-colors")
            return None


# ============================================================================
# Manual concat with colors — v5.3 ROOT-CAUSE FIX
# ============================================================================

def _manual_concat_with_colors(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Manually concatenate sub-meshes preserving per-vertex colors.

    trimesh.util.concatenate() de-duplicates vertices and breaks the 1:1
    mapping between input colors and output vertices.  We stack vertices
    WITHOUT de-duplication, re-index faces with the running vertex offset,
    and stack colors in the same order.
    """
    if len(meshes) == 0:
        raise ValueError("Cannot concatenate empty mesh list.")
    if len(meshes) == 1:
        return meshes[0]

    all_vertices = []
    all_faces    = []
    all_colors   = []
    vertex_offset = 0

    for m in meshes:
        n_verts = len(m.vertices)
        all_vertices.append(m.vertices.copy())
        all_faces.append(m.faces.copy() + vertex_offset)

        if (hasattr(m.visual, 'vertex_colors')
                and m.visual.vertex_colors is not None
                and len(m.visual.vertex_colors) == n_verts):
            vc = m.visual.vertex_colors[:, :4].astype(np.uint8)
        else:
            vc = np.full((n_verts, 4), 128, dtype=np.uint8)
        all_colors.append(vc)

        vertex_offset += n_verts

    merged_verts  = np.vstack(all_vertices)
    merged_faces  = np.vstack(all_faces)
    merged_colors = np.vstack(all_colors)

    mesh = trimesh.Trimesh(
        vertices=merged_verts,
        faces=merged_faces,
        process=False,
    )
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=merged_colors
    )

    std_check  = _color_std(merged_colors)
    mean_check = merged_colors[:, :3].astype(np.float32).mean(axis=0)
    print(f"[geometry] Manual concat: {len(merged_verts):,} verts  "
          f"{len(merged_faces):,} faces  "
          f"color std={std_check:.1f}  mean RGB={mean_check}")
    return mesh


# ============================================================================
# Public API
# ============================================================================

def load_model(obj_path: str) -> trimesh.Trimesh:
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Model not found: {obj_path}")

    loaded = trimesh.load(obj_path)

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for name, geom in loaded.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue

            piece = geom.copy()
            try:
                matrix, _ = loaded.graph.get(name)
                piece.apply_transform(matrix)
            except Exception:
                pass

            vis_type = type(piece.visual).__name__
            vc       = getattr(piece.visual, "vertex_colors", None)
            std_pre  = _color_std(vc) if vc is not None else 0.0
            gray_pre = _is_default_gray(vc) if vc is not None else True
            print(f"  [mesh] {name[:50]:<52} "
                  f"visual={vis_type:<18} std_pre={std_pre:.1f}  gray={gray_pre}")

            piece = TextureBaker(piece).bake()

            after_vc = piece.visual.vertex_colors if hasattr(piece.visual, 'vertex_colors') else None
            if after_vc is not None and len(after_vc) == len(piece.vertices):
                std_post  = _color_std(after_vc)
                mean_post = after_vc[:, :3].astype(np.float32).mean(axis=0)
                print(f"    [post-bake] std={std_post:.1f}  "
                      f"mean RGB={mean_post}  n_verts={len(piece.vertices)}")

            meshes.append(piece)

        if not meshes:
            raise ValueError("No Trimesh geometry found in the scene.")

        mesh = _manual_concat_with_colors(meshes)
    else:
        mesh = loaded
        mesh = TextureBaker(mesh).bake()

    mesh = MeshOrientation(mesh).fix()
    _report_color_status(mesh)
    return mesh


def _report_color_status(mesh: trimesh.Trimesh) -> None:
    has_color = (
        hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    )
    print(f"[geometry] Loaded : {len(mesh.vertices):,} vertices  "
          f"{len(mesh.faces):,} faces")

    if not has_color:
        print("[geometry] Colors : none — pseudo-colors will be used")
        return

    vc_final      = mesh.visual.vertex_colors
    std_final     = _color_std(vc_final)
    is_gray_final = _is_default_gray(vc_final)
    mean_final    = vc_final[:, :3].astype(np.float32).mean(axis=0)
    print(f"[geometry] Colors : "
          f"{'vertex colors ready' if not is_gray_final else 'WARNING: still default gray'}"
          f"  std={std_final:.1f}  mean RGB={mean_final}")


# Procedural aliases
def voxelize(mesh: trimesh.Trimesh, resolution: int = 32):
    return Voxelizer(mesh, resolution).voxelize()


def sample_surface_colors(mesh: trimesh.Trimesh,
                          voxel_points: np.ndarray) -> np.ndarray | None:
    return SurfaceColorSampler(mesh, voxel_points).sample()


def load_and_voxelize(obj_path: str, resolution: int = 64):
    mesh = load_model(obj_path)
    points, _, mesh_scaled, final_res = Voxelizer(mesh, resolution).voxelize()
    surface_colors = SurfaceColorSampler(mesh_scaled, points).sample()
    return points, surface_colors, mesh, final_res