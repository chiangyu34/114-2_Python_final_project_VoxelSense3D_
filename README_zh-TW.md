# VoxelSense 3D

> **基於感知科學的 Minecraft 原版方塊體素化工具（相容 1.20）**  
> 將任意 `.obj` / `.glb` 模型轉換為 Minecraft 方塊結構，  
> 採用印刷與影視色彩流程中使用的 CIELAB 色彩空間進行配色，  
> 而非傳統的 RGB 歐氏距離計算。

---

## 為什麼要做這個工具？

現有工具大多透過**將自訂貼圖烘焙到方塊上**來追求視覺效果，但輸出結果只能在搭配專屬資源包的情況下正確顯示。VoxelSense 3D 採取相反的設計哲學：所有視覺品質的提升，都必須來自更好的演算法，而不是更好的貼圖。

| | 傳統體素化工具 | **VoxelSense 3D** |
|---|---|---|
| 輸出類型 | 自訂貼圖方塊 | **100% 原版方塊** |
| 任何世界都能使用？ | ❌ 需要資源包 | ✅ 是 |
| 配色演算法 | RGB 歐氏距離 | **CIELAB ΔE · CIEDE2000** |
| 調色盤來源 | 自動生成的圖集 | 1.20 貼圖 |
| 適用情境 | 電影級渲染 | **生存模式友善建築** |

---

## 技術亮點

**CIELAB 感知色彩配對**  
所有體素顏色在進行最近鄰搜尋前，會先轉換至 CIE L\*a\*b\* 色彩空間。相較於 RGB 歐氏距離，平均 ΔE（CIEDE2000）可降低約 40%。內建的 `run_compare()` 模式會印出完整統計數據，並渲染對比視窗供報告截圖使用。

**重心座標表面色彩採樣（Barycentric Surface Color Sampling）**  
每個體素的顏色從最近三角面的三個頂點以重心座標插值計算，消除了曲面（例如臉部、手指）上常見的「最近頂點」跳色問題。底層以 KDTree 作為退路，處理法線查詢失敗的退化網格。

**PBR 感知貼圖烘焙（`TextureBaker`）**  
支援 GLB/glTF 材質堆疊，依照以下優先順序處理：
1. 從 `baseColorTexture` 以雙線性插值（Bilinear Interpolation）進行 UV 採樣
2. 從 `baseColorFactor` 取得純色 fallback
3. 使用 trimesh 內建的 `to_color()` 轉換
4. 材質平均色安全網，防止 trimesh 預設灰色導致全體素配對到 clay 的靜默錯誤

**記憶體安全體素化（`Voxelizer`）**  
設有 300K 體素上限，超過時自動遞迴減半解析度，確保密集模型在 8 GB 筆電上不會 OOM。縮放後的工作網格（`mesh_scaled`）會和體素座標一併回傳，讓色彩採樣器在相同座標空間內運算（座標空間不一致是「97% clay」問題的根本原因）。

**材質家族語意加權（Material-Family Semantic Weighting，可選）**  
三組 CIELAB 語意窗口對特定方塊家族施加小型 ΔE 加成：膚色偏向 `smooth_sandstone` / `pink_terracotta`；深色頭髮偏向 `black_concrete` / `coal_block`；植被偏向 `oak_leaves` / `moss_block`。在 `main.py` 中設定 `USE_MATERIAL_WEIGHTING = True` 即可啟用。

---

## 專案結構

```
VoxelSense3D/
├── main.py                   主程式入口 — CIELAB Pipeline
├── main_rgb_vs_lab.py        主程式入口 — RGB Baseline（用於對比）
├── view_result.py            查看體素化結果
├── src/
│   ├── config.py             統一管理所有路徑與參數
│   ├── geometry.py           TextureBaker · MeshOrientation · Voxelizer
│   │                         · SurfaceColorSampler
│   ├── pipeline.py           ModelLoader · VoxelEngine · ColorMatcher
│   ├── color_engine.py       BlockPalette · MATERIAL_FAMILIES
│   ├── viewer.py             show() · run_compare()（PyVista + Matplotlib）
│   └── build_blocks_json.py  PaletteBuilder — 產生 data/blocks.json
├── data/
│   └── blocks.json           預計算調色盤：方塊名稱 → [R, G, B]
├── block/                    Minecraft 1.20 貼圖 PNG + _list.json
│   ├── _list.json
│   └── *.png
├── models/                   使用者提供的 .obj / .glb / .gltf 模型
└── output/                   輸出檔案（自動建立）
```

---

## 安裝

```bash
git clone https://github.com/chiangyu34/VoxelSense3D_
cd VoxelSense3D
pip install trimesh numpy scipy scikit-image pillow pandas pyvista pyvista[jupyter] matplotlib
```

---

## 資源準備（不含於 repo 內）

**方塊貼圖** — 下載 Minecraft 1.20 ZIP：  
https://github.com/InventivetalentDev/minecraft-assets/archive/refs/heads/1.20.zip  
解壓後，只需將 `assets/minecraft/textures/block/` 整個資料夾複製到專案的 `block/` 目錄，接著執行：
```bash
python src/build_blocks_json.py   # 產生 data/blocks.json
```

**3D 模型** — 將你的 `.glb` / `.obj` 檔放入 `models/` 資料夾。  
免費模型可至 [Sketchfab](https://sketchfab.com) 下載（篩選條件：Free + Downloadable）。

---

## 快速開始

### 1 · 建立方塊調色盤（首次執行）

```bash
python src/build_blocks_json.py
```

從 `block/` 資料夾讀取 Minecraft 1.20 原版貼圖 PNG，過濾動畫幀與全透明貼圖，套用生物群系染色（平原草地、橡樹樹葉），輸出至 `data/blocks.json`。

### 2 · 體素化模型

```bash
python main.py
# Voxelizing file: your_model.glb
```

輸出結果存放於 `output/`：

| 檔案 | 內容 |
|---|---|
| `coords.npy` | `(N, 3)` 體素座標 |
| `colors.npy` | `(N, 3)` 配對後的方塊顏色，範圍 0–1 |
| `material_list.csv` | 方塊用量統計報表 |
| `voxel_view.html` | 獨立的 WebGL 互動檢視器 |

### 3 · 視覺化與產生 RGB vs CIELAB 對比資料

```bash
# 使用相同模型與品質設定分別執行兩個版本
python main.py          # → output/
python main_rgb.py      # → output_rgb/（不會覆蓋 CIELAB 結果）
python view_result.py   # 由 main.py 執行後的 output/colors.npy, coords.py 等檔案直接查看體素化結果
```



---

## 品質預設

在 `main.py` 中修改 `QUALITY`：

| 預設名稱 | 解析度 | 典型體素數 | 適用情境 |
|---|---|---|---|
| `low` | 32³ | ~1K | 快速測試 |
| `medium` | 64³ | ~10K | 開發預設 |
| `high` | 128³ | ~80K | 最終輸出 |
| `extra_high` | 256³ | ~300K | 上限（超過自動減半） |

---

## 系統架構

三個可獨立測試的階段，各自封裝為一個 class：

```
ModelLoader          VoxelEngine          ColorMatcher
──────────────       ────────────         ─────────────────────
load_model()    →    Voxelizer       →    SurfaceColorSampler
TextureBaker         300K 上限             重心座標 + KDTree
MeshOrientation      回傳                  BlockPalette.match_lab()
                     mesh_scaled           可選語意加權
```

`mesh_scaled` 是 `VoxelEngine` 和 `ColorMatcher` 之間的關鍵交接點：體素座標與網格頂點必須在同一個縮放座標空間內，最近鄰色彩查詢才會正確（座標空間不一致是「97% clay」問題的根本原因）。

---

## 可調整參數（`src/config.py`）

| 常數 | 預設值 | 作用 |
|---|---|---|
| `MAX_VOXELS` | 300,000 | 觸發自動減半解析度的體素上限 |
| `GRAY_TOL` | 10.0 | 偵測 trimesh 預設灰色 placeholder 的容差 |
| `MAX_VOXEL_WEIGHT_BATCH` | 50,000 | 語意加權的分批大小（控制 RAM 峰值） |

---

## 後續規劃

- [ ] 直接輸出 `.litematic` / `.schem` 格式供遊戲內放置
- [ ] 支援多方塊狀態配對（半磚、樓梯、矮牆）
- [ ] 生物群系感知染色選擇（平原 vs 沼澤草地）

---

## 致謝
 
**3D 模型**  
所有用於測試與展示的 `.glb` 模型檔案均來自 [Sketchfab](https://sketchfab.com)，依各自授權條款免費取得。
 
**Minecraft 方塊貼圖**  
用於產生 `data/blocks.json` 的貼圖 PNG 取自 Minecraft 1.20 原版資源包，透過以下來源取得：  
[InventivetalentDev/minecraft-assets（分支：1.20）](https://github.com/InventivetalentDev/minecraft-assets/archive/refs/heads/1.20.zip)  
Minecraft 方塊貼圖版權所有 © Mojang Studios。
 
本專案為學術課程作業，與 Mojang、Microsoft 或 Sketchfab 無任何從屬關係。