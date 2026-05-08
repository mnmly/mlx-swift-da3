# Port mlx-da3 (Depth Anything 3) to mlx-swift

## Skill

This port uses the **`port-mlx-to-swift`** skill (`~/.claude/skills/port-mlx-to-swift/`).
Refer to its reference docs for Python→Swift translation patterns:

- `references/operation-mapping.md` — MLX op/module translation table
- `references/package-patterns.md` — Package.swift template, module structure, config patterns
- `references/weight-loading.md` — safetensors loading, key remapping, conv transposition

## Source

Python: `/Users/mnmly/Development-local/GitHub/python/mlx-da3/mlx_da3/` (~800 LOC, 12 files)

## Architecture

```
DepthAnything3 (model.py)
├── DinoVisionTransformer (dinov2.py)        ← backbone
│   ├── Embeddings (embeddings.py)           ← PatchEmbed + CLS + pos interp
│   ├── Block[] (block.py)                   ← pre-norm transformer
│   │   ├── Attention (attention.py)         ← MHA, QK-norm, RoPE
│   │   ├── Mlp / SwiGLUFFN (ffn.py)        ← feed-forward
│   │   └── LayerScale (block.py)            ← per-dim scaling
│   └── RotaryPositionEmbedding2D (rope.py)  ← 2D RoPE
└── DPT / DualDPT (dpt.py)                  ← decoder head
    ├── Scratch                              ← stage adapters (3x3 conv)
    ├── FeatureFusionBlock                   ← top-down fusion + upsample
    ├── ResidualConvUnit                     ← residual conv block
    └── AuxHead                              ← auxiliary ray head (DualDPT)

Supporting:
  configs.py          ← hardcoded model configs (small/base/large/giant/mono-large)
  loading.py          ← PyTorch→MLX key remap + conv transposition
  image_processor.py  ← resize + ImageNet normalize
```

## Target Structure

```
mlx-swift-da3/
├── Package.swift
├── Sources/MLXDA3/
│   ├── DepthAnything3.swift          ← model.py
│   ├── DinoVisionTransformer.swift   ← dinov2.py
│   ├── Embeddings.swift              ← embeddings.py
│   ├── Attention.swift               ← attention.py
│   ├── FFN.swift                     ← ffn.py
│   ├── Block.swift                   ← block.py
│   ├── RoPE2D.swift                  ← rope.py
│   ├── DPT.swift                     ← dpt.py
│   ├── Configurations.swift          ← configs.py
│   ├── WeightLoading.swift           ← loading.py
│   └── ImageProcessor.swift          ← image_processor.py
├── Tools/da3-tool/
│   └── DA3Tool.swift                 ← CLI demo
└── Tests/MLXDA3Tests/
    └── ForwardTests.swift
```

## Progress

### ✅ Step 1: Package.swift + Configurations - COMPLETED

**Status:** All files created and compiling successfully.

**Files created:**
- `Package.swift` - Dependencies: mlx-swift, swift-transformers, swift-argument-parser
- `Sources/MLXDA3/Configurations.swift` - `BackboneConfig`, `HeadConfig`, `DA3Config` structs

**Verification:** `swift build` passes ✅

---

### ✅ Step 2: Low-level building blocks - COMPLETED (revised 2026-04-08)

All modules ported and compiling. **Revised** to fix bugs and add `@ParameterInfo` keys for weight loading.

**a) `RoPE2D.swift`** ← `rope.py` ✅
- `PositionGetter` — generates 2D grid positions `[B, H*W, 2]` using `MLX.repeated`/`MLX.tiled` + cache
- `RotaryPositionEmbedding2D: Module` — 2D RoPE via separate vertical/horizontal halves
  - Fixed `rotateHalf` slicing (`..<half` / `half...` instead of `0...half` / `half...d`)
  - `callAsFunction(tokens:positions:)` → split via `.split(axis:)` tuple, apply 1D RoPE to each half

**b) `FFN.swift`** ← `ffn.py` ✅
- `Mlp: Module` — Linear → GELU → Linear, `@ParameterInfo` keys: fc1, fc2
- `SwiGLUFFN: Module` — Linear(2*hidden) → `.split(axis:)` tuple → SiLU gate → Linear
  - `@ParameterInfo` keys: w12, w3

**c) `Attention.swift`** ← `attention.py` ✅
- `DA3Attention: Module`
  - Fixed: scale = `1.0 / Float(headDim).squareRoot()` (was broken `.reciprocal!`)
  - Fixed: mask expansion uses `MLX.repeated` (was broken `ones * expanded`)
  - Uses `MLXFast.scaledDotProductAttention` (import MLXFast)
  - `@ParameterInfo` keys: qkv, q_norm, k_norm, proj

**d) `Embeddings.swift`** ← `embeddings.py` ✅
- Fixed: `posEmbed` slicing uses `posEmbed[0..., ..<1]` / `posEmbed[0..., 1...]` (was dropping batch dim)
- Fixed: CLS broadcast uses `broadcast(clsToken, to:)` instead of `ones *`
- `@ParameterInfo` keys: patch_embed, cls_token, pos_embed

**Verification:** `swift build` passes ✅

---

### ✅ Step 3: Transformer block - COMPLETED (revised 2026-04-08)

**Files:**
- `Sources/MLXDA3/Block.swift`

**Modules:**
- `LayerScale: Module` — learnable `gamma` via `@ParameterInfo`
- `FFNLayer` protocol — shared interface for Mlp/SwiGLUFFN callability
- `DA3Block: Module`
  - LN → Attention → LayerScale → residual
  - LN → FFN (via `FFNLayer` protocol) → LayerScale → residual
  - All sub-modules have `@ParameterInfo` keys: norm1, attn, ls1, norm2, mlp, ls2

**Verification:** `swift build` passes ✅

---

### ✅ Step 4: Backbone - COMPLETED

**File:** `Sources/MLXDA3/DinoVisionTransformer.swift`

**Modules:**
- `DinoVisionTransformer: Module`
  - Init: Embeddings, optional RoPE + PositionGetter, optional camera_token, Block array, final LayerNorm
  - Per-block config: `qkNorm` correctly gated by `qknormStart != -1 && i >= qknormStart`
  - `prepareRope(B:S:H:W:)` → position tensors with CLS offset via `MLXArray.zeros(like:)`
  - `processAttention(_:block:attnType:pos:mask:)` → local (per-view) or global (cross-view) reshape
  - `callAsFunction(_:camToken:)` → extract intermediate features at `outLayers`
    - Camera token injection at `altStart` using `.at[...].add()` for in-place-like update
    - Learned camera token via `broadcast` + `expandedDimensions`
    - Alternating local/global attention (odd blocks after altStart → global)
    - `catToken`: concatenate local+global features at output layers
    - Final norm: full or half (when catToken doubles the dim)
    - Returns `([(MLXArray, MLXArray)], [MLXArray])` — (features+camTok pairs, aux)
  - `@ParameterInfo` keys: embeddings, camera_token, blocks, norm

**Verification:** `swift build` passes ✅

---

### ✅ Step 5: Decoder heads - COMPLETED

**File:** `Sources/MLXDA3/DPT.swift` ← `dpt.py`

**Utility functions (module-level):**
- `createUVGrid(width:height:aspectRatio:)` → normalized UV grid `[H, W, 2]`
- `makeSincosPosEmbed(embedDim:pos:omega0:)` → 1D sincos positional embedding `[M, D]`
- `positionGridToEmbed(posGrid:embedDim:)` → grid to sincos `[H, W, D]`
- `addPosEmbed(_:wImg:hImg:ratio:)` → add UV pos embed to NHWC feature map
- `bilinearInterpolate(_:size:)` → NHWC bilinear via `Upsample`
- `applyActivation(_:_:)` → named activation dispatch (exp/expp1/relu/sigmoid/softplus/tanh/linear)

**Modules:**
- `ResidualConvUnit` — two 3x3 convs with ReLU + residual
- `FeatureFusionBlock` — optional lateral merge + RCU + upsample + 1x1 conv
  - `callAsFunction(top:lateral:size:)` with optional lateral and target size
- `Scratch` — 4× stage adapters (3x3 conv, no bias)
- `AuxHead` — conv → LN → ReLU → 1x1 conv
- `DPT` — single-head decoder with optional sky head
  - `ConvTransposed2d` for upsampling (note: Swift class is `ConvTransposed2d`, not `ConvTranspose2d`)
  - Fusion pyramid: refinenet4 → refinenet3 → refinenet2 → refinenet1
  - Output: `[String: MLXArray]` dict with depth/conf/sky keys
- `DualDPT` — dual-head decoder (depth + ray)
  - Separate main + aux fusion chains
  - Nested `[[Conv2d]]` for aux pre-head stacks
  - Output: depth, depth_conf, ray, ray_conf

**All modules have `@ParameterInfo` keys matching Python weight names.**

**Verification:** `swift build` passes ✅

---

### Pending Steps

### ✅ Step 6: Top-level model - COMPLETED

**File:** `Sources/MLXDA3/DepthAnything3.swift` ← `model.py`

**Modules:**
- `DepthAnything3: Module` — combines backbone + head
  - Two init overloads: `init(backbone:, head: DPT)` and `init(backbone:, head: DualDPT)`
  - `callAsFunction(_:)` dispatches to DPT or DualDPT head
  - Returns `[String: MLXArray]` with depth/conf/ray/sky keys
- `buildModel(configName:)` — factory function using `DA3Config`

**Verification:** `swift build` passes ✅

---

### ✅ Step 7: Weight loading - COMPLETED

**File:** `Sources/MLXDA3/WeightLoading.swift` ← `loading.py`

**Functions:**
- `remapKey(_:)` — PyTorch → MLX key remapping using string parsing
  - Strips `model.` prefix, skips `cam_enc`/`cam_dec`
  - Backbone: `backbone.pretrained.X` → `backbone.X` (with embeddings redirect)
  - Head: `scratch.refinenetN` → `head.refinenetN`, sequential indices → named keys
  - Aux: `output_conv2_aux.LEVEL.IDX` → `conv0`/`ln`/`conv1` by index
- `isConvWeight(key:shape:)` / `isConvTranspose(key:)` — detection
- `transposeConv2d(_:)` — OIHW → OHWI
- `transposeConvTranspose2d(_:)` — IOHW → OHWI
- `loadWeights(model:url:dtype:)` — full pipeline: load → remap → transpose → cast → update
- `loadModel(configName:weightsURL:dtype:)` — build + load convenience

**Note:** Used string-based parsing (not regex literals) due to Swift regex literal parsing conflicts.

**Verification:** `swift build` passes ✅

---

### ✅ Step 8: Image processor - COMPLETED

**File:** `Sources/MLXDA3/ImageProcessor.swift` ← `image_processor.py`

**Struct:**
- `ImageProcessor` — CGImage → MLXArray `[1, H', W', 3]`
  - Resize maintaining aspect ratio to `processRes` (default 518)
  - Round to nearest multiple of `patchSize` (14)
  - ImageNet normalization (mean/std)
  - Uses CoreGraphics for resize (no PIL/numpy dependency)

**Verification:** `swift build` passes ✅

---

### ✅ Step 9: CLI tool + tests - COMPLETED

**Files:**
- `Tools/da3-tool/DA3Tool.swift` — CLI inference tool
- `Tests/MLXDA3Tests/ForwardTests.swift` — unit tests

**CLI tool (`da3-tool`):**
- `--model` config name (default: da3mono-large)
- `--weights` path to safetensors file
- `--input` image path (loads via ImageIO/CGImage)
- `--output` depth map output path (raw float32 binary)
- `--resolution` processing resolution (default 518)
- `--dtype` float16/float32
- Reports inference time and output shapes

**Tests (4 test suites, 28 tests):**
- `BuildModelTests` — constructs all 5 model configs (requires Metal)
- `ComponentShapeTests` — shape validation for PatchEmbed, Embeddings, Attention, FFN, Block, RoPE2D, ResidualConvUnit, AuxHead (requires Metal)
- `ImageProcessorTests` — divisibility, aspect ratio preservation (requires Metal for MLXArray creation)
- `WeightKeyRemapTests` — 17 tests for key remapping logic (**all pass**, no Metal needed)

**Package.swift fix:** Moved `ArgumentParser` dependency from library to tool target only.

**Verification:** `swift build --build-tests` passes ✅, `swift test --filter WeightKeyRemapTests` — 17/17 pass ✅
Shape/model tests require Metal runtime (run via Xcode or on-device).

---

### ✅ Step 10: End-to-end inference - COMPLETED

**Critical fix:** Changed all `@ParameterInfo(key:)` on Module-typed properties to `@ModuleInfo(key:)` across all files. `@ParameterInfo` is for raw `MLXArray` only; using it on `Conv2d`/`Linear`/`LayerNorm`/etc. causes runtime crash.

**Additional fix:** `createUVGrid()` in DPT.swift was passing `Double` to `linspace()`, producing `float64` arrays that crash on Metal GPU. Changed to `Float`.

**Test result with DA3-LARGE-1.1:**
```
Model: da3-large (depth-anything/DA3-LARGE-1.1)
Weights: 563 tensors loaded, 74 skipped (cam_enc/cam_dec)
Input: 640x480 → processed 518x378
Inference time: 2.25s
Output:
  depth: [1, 1, 378, 518] float32
  depth_conf: [1, 1, 378, 518] float32
  ray: [1, 1, 216, 296, 6] float32
  ray_conf: [1, 1, 216, 296] float32
```

**Build command:** `xcodebuild -scheme da3-tool -destination "platform=macOS" -configuration Release -derivedDataPath ./build`

---

## All Steps Complete

All 10 steps of the port are finished. The library compiles, the CLI tool runs end-to-end, and tests validate the weight remapping logic. Remaining work for production use:
- Numerical comparison against Python reference output
- HuggingFace Hub integration for automatic weight download
