# Agent handoff — mlx-swift-da3 streaming port

This file is a runbook for the next AI agent (or human) picking up the DA3-Streaming Swift port. Read it before touching anything. It assumes you know the codebase structure from `README.md` and the slice history in `tasks/todo.md`.

---

## TL;DR — what's where

- **Phase 1 (chunked streaming, no loop closure) is complete.**
- **Phase 2 (SALAD detector + Sim3LoopOptimizer + StreamingPipeline integration) is complete.**
- The Python reference (`Depth-Anything-3`) is required for any parity work — clone it locally and set the path in `.claude/local-runners/.env` (template at `docs/local-runners.env.example`).
- Streaming weights: `model.safetensors` from `da3_streaming/weights/` (the nested giant+large checkpoint produced by `download_weights.sh`).
- Single-image weights: any DA3 HuggingFace repo (`depth-anything/DA3-LARGE-1.1` etc.), downloaded via `huggingface-cli download`.
- SALAD weights: `dino_salad.ckpt` → run `Scripts/convert_salad_to_safetensors.py` once.
- Test inputs live under `tmp/da3_frames/` (8-frame chunked-streaming fixture) and `tmp/da3_robot_frames/` (42-frame loop-closure fixture).

---

## Hard rules (non-negotiable, will be enforced)

1. **Use `xcodebuild` only.** Never `swift build`, `swift test`, `swift package` — they've corrupted `.build/` state on this machine. Always pass `-derivedDataPath .xcdd` so artifacts land in a stable repo-local path.
2. **Use `uv` for any Python work.** Never call `python` or `python3` directly. For one-shot scripts: `uv run --no-project --with <pkg> python -c "..."`.
3. **Don't write CLAUDE.md, .md docs, or commit messages unless asked.** Working notes go in `tasks/`.
4. **Stop-the-line on unexpected behavior.** If a build error or test failure looks unfamiliar, diagnose root cause — don't bypass with `--no-verify` or destructive ops.
5. **Don't add features beyond the requested task.** A bug fix doesn't need surrounding cleanup; new abstractions need explicit user approval.
6. Make sure you refer to the actual API. Swift mlx is at ../mlx-swift you can also refer to ../mlx-swift-examples if needed.
The user prefers terse, concrete output. No long preambles. State what you're doing in one sentence, then do it.

---

## Build & run cheatsheet

```bash
# Build the streaming tool (release, into .xcdd)
xcodebuild -scheme da3-streaming-tool -destination 'platform=macOS' \
  -configuration release -derivedDataPath .xcdd build

# Smoke test (8 frames, 3 chunks, ~3 seconds). Source the env first
# (.claude/local-runners/.env contains your local weights path):
source .claude/local-runners/env.sh

.xcdd/Build/Products/Release/da3-streaming-tool \
  --model da3-giant \
  --weights "$DA3_STREAMING_WEIGHTS" \
  --image-dir tmp/da3_frames \
  --output-dir /tmp/da3_out \
  --chunk-size 4 --overlap 2 --limit 8

# List schemes
xcodebuild -list

# Tests (DA3 model parity)
xcodebuild -scheme MLXDA3 -destination 'platform=macOS' \
  -derivedDataPath .xcdd test
```

---

## Architecture map (streaming side)

Read these files in order if you want to grok the pipeline:

1. `Sources/MLXDA3Streaming/StreamingOrchestrator.swift` — entry point, mirrors Python `process_long_sequence` (no-loop branch).
2. `Sources/MLXDA3Streaming/StreamingInference.swift` — multi-view forward pass + post-processing into `ChunkPredictions`.
3. `Sources/MLXDA3Streaming/RayPose.swift` — geometric ray → extrinsics+intrinsics, port of `get_extrinsic_from_camray`. Heaviest math file. **All SVD/QR pinned to `stream: .cpu`.**
4. `Sources/MLXDA3Streaming/Sim3Alignment.swift` — weighted Sim(3) IRLS with Huber re-weighting. SVD also CPU-only.
5. `Sources/MLXDA3Streaming/ChunkAlignment.swift` — gathers valid-pixel pairs for `Sim3Alignment` (CPU-side because MLX fancy indexing is limited).
6. `Sources/MLXDA3Streaming/PointMaps.swift` — `depth_to_point_cloud` (closed-form K^-1, manual w2c→c2w via R^T to avoid GPU-incompatible `inv`).
7. `Sources/MLXDA3Streaming/CameraPosesIO.swift`, `PLYWriter.swift` — file outputs.

---

## MLX-Swift gotchas you WILL hit

I burned time on each of these — save yourself the round trip:

### Linalg ops are CPU-only

`MLX.svd`, `MLX.qr`, `MLX.inv` all crash on the GPU stream with "This op is not yet supported on the GPU." Pin them with `stream: .cpu`:

```swift
let (U, _, Vh) = MLX.svd(A, stream: .cpu)
let (Q, R)     = MLX.qr(A, stream: .cpu)
```

`MLXLinalg.det` does **not** exist. Compute 3×3 det manually (see `RayPose.det3x3`).

### Don't use `inv` if you can avoid it

For 3×3 intrinsic inversion: closed-form `K^-1 = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]`.

For 4×4 / 3×4 SE(3) inverse (w2c → c2w): `[R | t]^-1 = [R^T | -R^T t]`. See `PointMaps.depthToWorldPoints` and `CameraPosesIO.invertW2C`.

### `broadcast(to:)` on MLXArray is internal

The instance method has `internal` access. Use the free function:

```swift
// ❌ won't compile from outside MLX module
arr.broadcast(to: [N, H, W, 3])
// ✅
MLX.broadcast(arr, to: [N, H, W, 3])
```

### `stacked` and similar are free functions

```swift
// ❌
let s = stacked([a, b], axis: 0)
// ✅
let s = MLX.stacked([a, b], axis: 0)
```

### Comparison operators

MLX-Swift has both `>` (returns `MLXArray`) and `.>` (also returns `MLXArray`). Either works for elementwise comparison. Use plain `&` and `|` for bool ops:

```swift
let mask = (MLX.abs(zo) .> Float(1e-4)) & (MLX.abs(zt) .> Float(1e-4))
```

### `MLX.which(cond, a, b)` overload resolution

`a` and `b` are both `some ScalarOrArray`. Mixing `Float` and `MLXArray` in the same call sometimes confuses the type checker. Bind to typed locals first:

```swift
// fragile
MLX.which(r .< d, halfRsq, Float(d) * (r - 0.5*d))   // sometimes errors
// reliable
let halfRsq = r * r * Float(0.5)
let linearTerm = (r - Float(0.5*d)) * Float(d)
MLX.which(r .< Float(d), halfRsq, linearTerm)
```

### "Compiler unable to type-check in reasonable time"

If you see this, you've chained too many MLX ops on one line. Break into named locals.

### MLXArray construction quirks

`MLXArray(0.0, dtype: .float32)` — works, gives a scalar array.
`MLXArray.zeros([N, 1], dtype: .float32)` — works for shapes.
`MLXArray([Float(1), 2, 3])` — works for 1-D.
`MLXArray([Float(0), 0, 1, ...], [3, 3])` — works for 2-D + shape.

### No `median` op

The Python reference uses `np.median` for the conf threshold heuristic. MLX has no median. Current code uses `mean` as a stand-in (good enough for a soft threshold, but not faithful). If you need true median, do it CPU-side via `.asArray(Float.self)` + sort.

---

## Weight-loading conventions (already wired)

The Swift loader (`Sources/MLXDA3/WeightLoading.swift`) has been extended to handle the nested `da3nested-giant-large` checkpoint:

- Strips both `model.` and (if present) the `da3.` sub-prefix.
- Skips `da3_metric.*` (metric mono branch — not used in streaming).
- Skips `cam_enc.*`, `cam_dec.*`, `gs_head.*`, `gs_adapter.*` (not ported).

If you add new weights or modules later, update `remapKey()` in that file.

For the test sample we load 831 / 1377 keys; the 546 skipped are intentional.

---

## Python reference map

Want to verify or extend? The relevant Python files:

```
python/Depth-Anything-3/
├── src/depth_anything_3/
│   ├── api.py                            # high-level inference API
│   ├── model/da3.py                      # forward + ray-pose path
│   └── utils/
│       ├── ray_utils.py                  # ⭐ get_extrinsic_from_camray + RANSAC
│       └── geometry.py                   # unproject_depth (only ixt_normalized path needed)
└── da3_streaming/
    ├── da3_streaming.py                  # ⭐ orchestrator (process_long_sequence)
    ├── loop_utils/
    │   ├── alignment_torch.py            # ⭐ IRLS + weighted Sim(3) (torch)
    │   ├── alignment_triton.py           # CUDA-only, do not port
    │   ├── sim3utils.py                  # ⭐ weighted_align_point_maps + accumulate + PLY IO
    │   ├── sim3loop.py                   # Sim(3) BA optimizer (Phase 2)
    │   ├── loop_detector.py              # NetVLAD detector (Phase 2)
    │   ├── loop_refinement.py            # (Phase 2)
    │   └── salad/                        # NetVLAD model + ckpt (Phase 2)
    ├── fastloop/
    │   ├── solve.cpp                     # CUDA solver — do NOT port; use solve_python.py instead
    │   └── solve_python.py               # Python fallback
    └── configs/base_config.yaml          # default hyperparameters
```

Stars (⭐) mark files I actively referenced during the Phase-1 port.

---

## Verifying your work

### Smoke test (must pass before opening a PR)

```bash
source .claude/local-runners/env.sh
xcodebuild -scheme da3-streaming-tool -destination 'platform=macOS' \
  -configuration release -derivedDataPath .xcdd build && \
.xcdd/Build/Products/Release/da3-streaming-tool \
  --model da3-giant \
  --weights "$DA3_STREAMING_WEIGHTS" \
  --image-dir tmp/da3_frames \
  --output-dir /tmp/da3_out \
  --chunk-size 4 --overlap 2 --limit 8
```

Expected:
- 8 images, 3 chunks
- 2 alignment lines like `sim3 s≈1.0 t=[~0.5, ~0, ~0.5]`
- `camera_poses.txt` with 8 lines (16 numbers each)
- `intrinsic.txt` with 8 lines (4 numbers each)
- `pcd/combined_pcd.ply` ~13 MB, ~875k vertices

### Numerical parity vs Python (NOT YET DONE)

There is no parity test against the Python reference. Adding one would catch regressions in any of the algorithmic ports. Suggested approach:
1. Run Python `da3_streaming.py` on the same `/tmp/da3_frames/` with `align_lib=torch`, save `camera_poses.txt` + first chunk's depth/conf as `.npy`.
2. Run Swift tool on same input.
3. Compare poses (Frobenius norm of c2w difference < threshold) and depths (relative MSE).
4. Add to `Tests/MLXDA3Tests/` as a fixture-based test.

This is the single highest-value next task.

---

## Suggested next tasks (in priority order)

### 1. Numerical parity check vs Python ⭐ high value
Without this we don't know if the ports are correct, only that they don't crash. Steps in the Verifying section above.

### 2. `ref_view_strategy=saddle_balanced` in the backbone
Python `vision_transformer.py:316` selects a reference view among the multi-view stack for camera-token broadcasting. The Swift `DinoVisionTransformer` has a stubbed `cam_token` comment but no ref-view logic. Likely a meaningful accuracy delta. Port this once parity tests are in place to measure the effect.

### 3. Loop closure (Phase 2 — biggest single chunk)
- Port `LoopDetector` (`loop_utils/loop_detector.py` + `salad/`) — needs the `dino_salad.ckpt` weights, which is a NetVLAD-style model. This is a separate model port.
- Port `Sim3LoopOptimizer` (`loop_utils/sim3loop.py`) — pose-graph optimization with Huber.
- Port `loop_refinement.py`.
- C++ `fastloop/solve.cpp` — skip; port `solve_python.py` instead.
- Wire into `StreamingOrchestrator` behind a `--loop-closure` flag.

### 4. Memory: spill predictions to disk
`StreamingOrchestrator.run()` keeps every chunk's predictions in RAM. For long sequences (~50+ frames at chunk_size=8) this OOMs. Python writes per-chunk `.npy` files between phases. Mirror that — define a serialization for `ChunkPredictions` (e.g. safetensors per chunk) and stream-load during the alignment + PLY-write phases.

### 5. PLY reservoir sampling
Currently `sample_ratio=1.0` is hardcoded in `PLYWriter.saveConfidentPointCloud`. Port `optimized_vectorized_reservoir_sampling` from `sim3utils.py` for `sample_ratio<1.0`. Useful for keeping the combined PLY size bounded.

### 6. ImageProcessor MLX-vectorize
`Sources/MLXDA3/ImageProcessor.swift` does CPU per-pixel normalization in pure Swift loops — slow for large/many-frame chunks. Replace with MLX tensor ops on the raw RGB UInt8 buffer.

### 7. True median for IRLS conf threshold
Replace the `mean` stand-in in `ChunkAlignment.median()` with a CPU-side `asArray + sort` median. Will only matter once parity tests reveal whether the heuristic difference shifts results.

---

## Conventions for new files

- One concept per file. Don't create grab-bag utility files.
- File-level doc comment explains what the file maps to in the Python reference.
- No emojis in code or comments unless explicitly requested.
- Default to no comments. Add a one-liner only when the *why* is non-obvious (constraint, invariant, surprise).
- Don't add `// removed` comments or `_unused` renames. If unused, delete.
- `tasks/todo.md` is the canonical status doc. Update it as you complete slices.

---

## File map (write/edit cheatsheet)

If you're adding...

| Feature | Add to | Touch |
|---|---|---|
| New CLI flag | `Tools/da3-streaming-tool/DA3StreamingTool.swift` | `StreamingOrchestrator.Config` |
| New chunk-level math | `Sources/MLXDA3Streaming/<New>.swift` | `StreamingOrchestrator` |
| New camera math | `Sources/MLXDA3Streaming/RayPose.swift` |  |
| Loop closure | `Sources/MLXDA3Streaming/LoopClosure*.swift` (new) | `StreamingOrchestrator` |
| Test fixture | `Tests/MLXDA3Tests/` | — |
| Weight-key handling | `Sources/MLXDA3/WeightLoading.swift` (`remapKey`) | — |
| New model config | `Sources/MLXDA3/Configurations.swift` | — |

---

## Final notes

- Don't trust subagents' summaries blindly. After delegating work, read the diff.
- If a memory in the user's `~/.claude/projects/.../memory/` says "X must be true," verify it's still true before acting on it. Memories rot.
- The user is working on creative/film projects — avoid breaking the binary in subtle ways. Prefer adding new flags/code paths over changing defaults.
