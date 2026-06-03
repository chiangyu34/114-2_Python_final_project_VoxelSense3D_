# VoxelSense 3D

> **Perceptually accurate Minecraft voxelization for vanilla 1.20.**  
> Converts any `.obj` / `.glb` model into a Minecraft block structure using
> CIELAB color matching — the same color science used in print and film
> pipelines — instead of naive RGB distance.

---

## Why another voxelizer?

Most existing tools achieve visual fidelity by **baking custom textures onto
blocks**, which means the output only looks correct inside a custom resource
pack. VoxelSense 3D takes the opposite stance: every visual improvement must
come from better algorithms, not better textures.

| | Traditional voxelizers | **VoxelSense 3D** |
|---|---|---|
| Output | Custom-textured blocks | **100 % vanilla blocks** |
| Works in any world? | ❌ Needs resource pack | ✅ Yes |
| Color algorithm | RGB Euclidean | **CIELAB ΔE · CIEDE2000** |
| Palette source | Generated atlases | Vanilla 1.20 textures |
| Target use case | Cinematic renders | **Survival-friendly builds** |

---

## Technical highlights

**CIELAB perceptual color matching**  
All voxel colors are converted to CIE L\*a\*b\* before nearest-neighbor
lookup. Average ΔE (CIEDE2000) is reduced by ~40 % vs RGB Euclidean.
A built-in `run_compare()` mode prints the exact statistics and renders a
side-by-side window for report screenshots.

**Barycentric surface color sampling**  
Per-voxel color is interpolated from the three vertices of the nearest
triangle using barycentric weights. This eliminates the "nearest-vertex"
artifacts that affect curved surfaces like faces and fingers. A KDTree
fallback handles degenerate meshes where proximity queries fail.

**PBR-aware texture baking (`TextureBaker`)**  
Handles the full GLB/glTF material stack in priority order:
1. UV sampling with bilinear interpolation from `baseColorTexture`
2. Solid-color fallback from `baseColorFactor`
3. trimesh's built-in `to_color()` conversion
4. Material-average safety net to prevent the silent all-gray failure

**Memory-safe voxelization (`Voxelizer`)**  
A 300 K-voxel ceiling triggers automatic recursive resolution halving so
dense models never OOM on an 8 GB laptop. The scaled working mesh is
returned alongside the voxel coordinates so the color sampler operates in
the same coordinate space (a mismatch was the root cause of the "97 % clay"
failure mode).

**Material-family semantic weighting (optional)**  
Three curated CIELAB windows apply a small ΔE bonus to contextually
correct block families: skin tones bias toward `smooth_sandstone` /
`pink_terracotta`; dark hair biases toward `black_concrete` /
`coal_block`; foliage biases toward `oak_leaves` / `moss_block`.
Enabled per-run via `USE_MATERIAL_WEIGHTING = True` in `main.py`.

---

## Project structure

```
VoxelSense3D/
├── main.py                   Entry point — CIELAB pipeline
├── main_rgb_vs_lab.py        Entry point — RGB baseline (for comparison)
├── view_result.py            Directly inspect voxelization results
├── src/
│   ├── config.py             All paths and tunables in one place
│   ├── geometry.py           TextureBaker · MeshOrientation · Voxelizer
│   │                         · SurfaceColorSampler
│   ├── pipeline.py           ModelLoader · VoxelEngine · ColorMatcher
│   ├── color_engine.py       BlockPalette · MATERIAL_FAMILIES
│   ├── viewer.py             show() · run_compare() (PyVista + Matplotlib)
│   └── build_blocks_json.py  PaletteBuilder — generates data/blocks.json
├── data/
│   └── blocks.json           Pre-computed palette: name → [R, G, B]
├── block/                    Minecraft 1.20 texture │   ├── _list.json
│   └── *.png
├── models/                   User-supplied .obj / .glb / .gltf files
└── output/                   Generated outputs (auto-created)
```

---

## Installation

```bash
git clone https://github.com/chiangyu34/VoxelSense3D_
cd VoxelSense3D
pip install trimesh numpy scipy scikit-image pillow pandas pyvista pyvista[jupyter] matplotlib
```

---

## Assets setup (not included in repo)

**Block textures** — Download the Minecraft 1.20 asset ZIP:  
https://github.com/InventivetalentDev/minecraft-assets/archive/refs/heads/1.20.zip  
Extract and copy **only** `assets/minecraft/textures/block/` into the `block/` folder, then run:
```bash
python src/build_blocks_json.py   # generates data/blocks.json
```

**3D models** — Place your own `.glb` / `.obj` files inside `models/`.  
Free models can be downloaded from [Sketchfab](https://sketchfab.com) (filter: Free + Downloadable).

---

## Quick start

### 1 · Build the block palette (first time only)

```bash
python src/build_blocks_json.py
```

Reads the Minecraft 1.20 default texture PNGs from `block/`, filters
animation frames and transparent-only textures, applies biome tinting
(Plains grass, oak foliage), and writes `data/blocks.json`.

### 2 · Voxelize a model

```bash
python main.py
# Voxelizing file: your_model.glb
```

Outputs land in `output/`:

| File | Contents |
|---|---|
| `coords.npy` | `(N, 3)` voxel coordinates |
| `colors.npy` | `(N, 3)` matched block colors, range 0–1 |
| `material_list.csv` | Sorted block-usage report |
| `voxel_view.html` | Self-contained WebGL viewer |

### 3 · Generate RGB vs CIELAB comparison data

```bash
# Run both with the same model and quality setting
python main.py          # → output/
python main_rgb.py      # → output_rgb/  (never overwrites CIELAB results)
python view_result.py   # Directly inspect voxelization results from files such as output/colors.npy and coords.py generated after running main.py.
```



---

## Quality presets

Edit `QUALITY` in `main.py`:

| Preset | Resolution | Typical voxels | Use case |
|---|---|---|---|
| `low` | 32³ | ~1 K | Fast iteration |
| `medium` | 64³ | ~10 K | Development default |
| `high` | 128³ | ~80 K | Final renders |
| `extra_high` | 256³ | ~300 K | Ceiling (auto-halved if exceeded) |

---

## Architecture

Three independently testable stages, each a single class:

```
ModelLoader          VoxelEngine          ColorMatcher
──────────────       ────────────         ─────────────────────
load_model()    →    Voxelizer       →    SurfaceColorSampler
TextureBaker         300K ceiling          barycentric + KDTree
MeshOrientation      returns               BlockPalette.match_lab()
                     mesh_scaled           optional weighting
```

`mesh_scaled` is the key handoff between `VoxelEngine` and `ColorMatcher`:
both voxel coordinates and mesh vertices are in the same scaled coordinate
space, which is required for correct nearest-neighbor color lookup.

---

## Tunable constants (`src/config.py`)

| Constant | Default | Effect |
|---|---|---|
| `MAX_VOXELS` | 300 000 | Voxel ceiling before auto-halving resolution |
| `GRAY_TOL` | 10.0 | Tolerance for detecting trimesh default-gray placeholders |
| `MAX_VOXEL_WEIGHT_BATCH` | 50 000 | Batch size for weighted matching (controls peak RAM) |

---

## Roadmap

- [ ] Direct `.litematic` / `.schem` export for in-world placement
- [ ] Multi-block-state matching (slabs, stairs, walls)
- [ ] Biome-aware tinting selection (Plains vs Swamp grass)

---

## Acknowledgements
 
**3D Models**  
All `.glb` model files used for testing and demonstration were sourced from
[Sketchfab](https://sketchfab.com) under their respective free licenses.
 
**Minecraft Block Textures**  
Texture PNGs used to build `data/blocks.json` are extracted from the
Minecraft 1.20 default resource pack, distributed via:  
[InventivetalentDev/minecraft-assets (branch: 1.20)](https://github.com/InventivetalentDev/minecraft-assets/archive/refs/heads/1.20.zip)  
Minecraft block textures © Mojang Studios.
 
This project is a coursework deliverable and is not affiliated with
Mojang, Microsoft, or Sketchfab.