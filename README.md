# mlx-swift-da3

Swift port of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for Apple Silicon, built on [mlx-swift](https://github.com/ml-explore/mlx-swift).

Two products:

- **`MLXDA3`** — the DA3 model (DinoV2 backbone + DPT/DualDPT head) and a `da3-tool` CLI for single-image / multi-view inference.
- **`MLXDA3Streaming`** — port of the [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming) chunked inference pipeline (Phase 1: chunked alignment, **no loop closure**) and a `da3-streaming-tool` CLI that produces camera poses + a combined point cloud from a directory of frames.

## Status

| Component | Status |
|---|---|
| DinoV2 backbone (vits/b/l/g, multi-view) | ✅ |
| DPT / DualDPT head | ✅ |
| Single-image inference (`da3-tool`) | ✅ |
| Streaming: multi-view inference + ray-based camera estimation | ✅ |
| Streaming: chunk-pair Sim(3) alignment (IRLS, Huber-weighted) | ✅ |
| Streaming: per-frame camera_poses.txt + intrinsic.txt + combined PLY | ✅ |
| Streaming: SALAD VPR loop detection (`LoopDetector`) | ✅ Phase 2 |
| Streaming: Sim(3) loop optimizer (`Sim3LoopOptimizer`, LM with numerical Jacobian) | ✅ Phase 2 |
| Streaming: loop-pair Sim(3) measurement helper (`computeLoopMeasurement`) | ✅ Phase 2 |
| Streaming: `StreamingPipeline.predict(images:loopConstraints:)` opt-in | ✅ Phase 2 |
| Streaming: end-to-end loop-enabled parity vs python (`LoopClosureParityFixtureTests` on 42 robot_unitree frames) | ✅ Phase 2 (plumbing parity verified; coverage of the optimizer-with-actual-loops path requires KITTI-style data not bundled in either repo) |
| Streaming: `da3-streaming-tool` `--enable-loop-closure` CLI flag | ❌ planned (low priority — API already usable from Swift) |
| `ref_view_strategy=saddle_balanced` | ❌ defers to backbone default |
| Numerical parity check vs Python reference | ❌ not yet run |

See [`tasks/todo.md`](tasks/todo.md) for the full breakdown of what's done and what's still missing.

## Requirements

- macOS 14+ (or iOS 17+ / visionOS 1+)
- Xcode 15+
- Apple Silicon (Metal-capable GPU)

## Building

Always pass `-derivedDataPath .xcdd` so build products land in a stable repo-local path:

```bash
xcodebuild -scheme da3-streaming-tool -destination 'platform=macOS' \
  -configuration release -derivedDataPath .xcdd build
```

Schemes:

```
xcodebuild -list
# da3-tool, da3-bench, da3-streaming-tool, MLXDA3
```

## Weights

You'll need different checkpoints depending on which pipeline you run.

### Single-image pipeline (`DepthAnything3Pipeline`)

Download from HuggingFace — any of the public DA3 variants work:

```bash
huggingface-cli download depth-anything/DA3-LARGE-1.1
# or:
huggingface-cli download depth-anything/DA3-GIANT-1.1
huggingface-cli download depth-anything/DA3-BASE-1.1
huggingface-cli download depth-anything/DA3-SMALL-1.1
```

This populates `~/.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/<hash>/model.safetensors`. Either point the CLI tools / demo app at that path directly, or copy/symlink the file somewhere convenient. The Swift loader transparently shims HuggingFace's symlink-to-blob-with-no-extension layout, so picking the snapshot path in a file picker works without further setup.

Use `--model da3-large` (or matching variant) when invoking `da3-tool` / `da3-bench`, or pick the config from the dropdown in the demo app.

### Streaming pipeline (`StreamingPipeline`)

The DA3-Streaming release ships a *nested* checkpoint that contains both the multi-view giant branch and a metric mono branch. The Swift loader strips the `model.da3.*` prefix and skips the metric branch + camera encoder/decoder + GS head automatically; pass `--model da3-giant` to match the multi-view config.

```bash
# Clone DA3 reference repo and run the bundled download script
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
bash da3_streaming/scripts/download_weights.sh
# → da3_streaming/weights/{model.safetensors, config.json, dino_salad.ckpt}
```

### Loop closure (SALAD VPR)

Loop closure additionally requires the SALAD descriptor weights. The upstream release ships them as a PyTorch `.ckpt` saved on CUDA — convert to MLX-loadable safetensors first:

```bash
uv run --script Scripts/convert_salad_to_safetensors.py \
    --in  /path/to/Depth-Anything-3/da3_streaming/weights/dino_salad.ckpt \
    --out /path/to/Depth-Anything-3/da3_streaming/weights/dino_salad.safetensors
```

### Quick reference

| Use case | Config / API | Weights file |
|---|---|---|
| Single image (e.g. `da3-tool`, demo app's "Single image" mode) | `--model da3-large` / `DepthAnything3Pipeline` | `model.safetensors` from any DA3 HuggingFace repo |
| Multi-frame streaming (e.g. `da3-streaming-tool`, demo app's "Video streaming" mode) | `--model da3-giant` / `StreamingPipeline` | `model.safetensors` from `da3_streaming/weights/` (nested giant+large) |
| Loop closure | `LoopDetector.fromPretrained(...)` | `dino_salad.safetensors` (converted from `.ckpt`) |

## Usage

Use the high-level pipeline APIs in your own Swift code, or the bundled
CLI tools (`da3-tool`, `da3-streaming-tool`).

A SwiftUI macOS sample app that exercises every public pipeline lives in
[`Examples/DA3Demo/`](Examples/DA3Demo/) — open the xcodeproj, add the
parent repo as a local SwiftPM dependency once, and run.

### Single-image pipeline

```swift
import MLXDA3

let pipeline = try MLXDA3.fromPretrained("/path/to/model.safetensors", configName: "da3-large")
let prediction = pipeline(cgImage)        // CGImage -> DepthAnything3Prediction
let depth = prediction.depth              // optional MLXArray
```

### Loop detection (SALAD VPR)

```swift
import MLXDA3Streaming

var cfg = LoopDetector.Config()
cfg.imageSize = (height: 336, width: 336)
cfg.batchSize = 32
cfg.similarityThreshold = 0.85
cfg.topK = 5

let detector = try LoopDetector.fromPretrained(
    "/path/to/dino_salad.safetensors", config: cfg
)
let (descriptors, loops) = detector.detect(images: cgImages)
// loops: [LoopClosure(a: Int, b: Int, similarity: Float)]
```

The SALAD weights ship as a PyTorch `.ckpt`; convert to safetensors first:

```bash
uv run --script Scripts/convert_salad_to_safetensors.py \
    --in /path/to/dino_salad.ckpt \
    --out /path/to/dino_salad.safetensors
```

There's also a `da3-loop-tool` CLI that mirrors the python `LoopDetector`'s
output format, plus a `--output-safetensors` flag that dumps descriptors +
loop pairs for parity testing.

### End-to-end loop-closed streaming

`StreamingPipeline.predict(images:loopConstraints:)` accepts external Sim(3)
measurements between non-adjacent chunk pairs and runs `Sim3LoopOptimizer`
to refine the chunk-pair Sim(3) chain before pose accumulation. Composition:

```swift
import MLXDA3Streaming

// 1. Run streaming on all frames (no loops yet).
let streamingPipeline = try MLXDA3Streaming.fromPretrained(da3WeightsPath, configName: "da3-giant")
let initialPrediction = streamingPipeline.predict(images: cgImages)

// 2. Run SALAD loop detection on the same frames.
let detector = try LoopDetector.fromPretrained(saladWeightsPath)
let (_, loops) = detector.detect(images: cgImages)

// 3. For each loop pair (frame i, frame j), find which chunks they belong to
//    and compute a Sim(3) measurement between those chunks.
var loopConstraints: [LoopConstraint] = []
for loop in loops {
    guard let chunkA = chunkIndexFor(frame: loop.a, ranges: initialPrediction.chunkRanges),
          let chunkB = chunkIndexFor(frame: loop.b, ranges: initialPrediction.chunkRanges),
          chunkA != chunkB
    else { continue }
    if let constraint = streamingPipeline.computeLoopMeasurement(
        chunkA: initialPrediction.perChunk[chunkA],
        framesA: framesForChunk(chunkA, images: cgImages, ranges: initialPrediction.chunkRanges),
        chunkAIdx: chunkA,
        chunkB: initialPrediction.perChunk[chunkB],
        framesB: framesForChunk(chunkB, images: cgImages, ranges: initialPrediction.chunkRanges),
        chunkBIdx: chunkB
    ) {
        loopConstraints.append(constraint)
    }
}

// 4. Re-run streaming with loop constraints.
let refinedPrediction = streamingPipeline.predict(images: cgImages, loopConstraints: loopConstraints)
```

Helper `chunkIndexFor(frame:ranges:)` finds the chunk a given frame belongs to;
`framesForChunk(_:images:ranges:)` slices CGImages by chunk range.

When `loopConstraints` is empty, the pipeline behaves identically to a
non-loop-closed run — the existing `StreamingParityFixtureTests` validates
this.

### Streaming pipeline (multi-frame, in-memory)

```swift
import MLXDA3Streaming

var cfg = StreamingPipeline.Config()
cfg.chunkSize = 4
cfg.overlap = 2
cfg.resolution = 504

let pipeline = try MLXDA3Streaming.fromPretrained(
    "/path/to/model.safetensors", configName: "da3-giant", config: cfg
)
let pred = pipeline.predict(images: cgImages)   // [CGImage] -> StreamingPrediction
// pred.cameraPosesC2W  : [N, 4, 4] per-frame world poses
// pred.intrinsicsK     : [N, 3, 3] per-frame intrinsics
// pred.pairwiseSim3    : per-chunk-pair (s, R, t)
// pred.cumulativeSim3  : chunk-i → chunk-0 transforms
// pred.perChunk        : raw per-chunk model outputs (depth, conf, ray, ray_conf, ...)
```

For file outputs (camera_poses.txt, intrinsic.txt, per-chunk PLY +
combined PLY), use the `da3-streaming-tool` CLI below — internally it
calls into the same `StreamingPipeline` and adds the file IO layer.

## da3-streaming-tool — quick start

Extract some frames from a video and run the streaming pipeline:

```bash
mkdir -p /tmp/da3_frames
ffmpeg -i input.mp4 -vf "fps=5,scale=512:-1" /tmp/da3_frames/frame_%06d.png

.xcdd/Build/Products/Release/da3-streaming-tool \
  --model da3-giant \
  --weights /path/to/model.safetensors \
  --image-dir /tmp/da3_frames \
  --output-dir /tmp/da3_out \
  --chunk-size 8 \
  --overlap 4
```

Outputs (matches the Python reference's "basic outputs"):

- `camera_poses.txt` — one line per frame, 16 numbers (row-major c2w 4×4)
- `intrinsic.txt` — one line per frame, `fx fy cx cy`
- `pcd/<i>_pcd.ply` — per-chunk binary PLY with confidence-thresholded points
- `pcd/combined_pcd.ply` — merged point cloud across all chunks

### Memory and chunk size

Apple Silicon's unified memory budget is much smaller than the A100 the Python reference targets. The Python defaults (`chunk_size=120`, `overlap=60`) will OOM on a Mac. Defaults here are conservative: `chunk_size=8, overlap=4`. Tune up gradually for your machine.

### CLI flags

```
--model           Model config name (default: da3-giant)
--weights         Path to model.safetensors
--image-dir       Directory of input images (jpg/png, sorted)
--output-dir      Output directory
--chunk-size      Frames per chunk (default: 8)
--overlap         Overlap frames between chunks (default: 4)
--resolution      Processing resolution (default: 518)
--dtype           float16 | float32 (default: float16)
--limit           Cap input frame count (0 = all)
--pcd-conf-coef   PLY conf threshold = mean(conf) * coef (default: 0.75)
```

## da3-tool — single-image / multi-view

```bash
.xcdd/Build/Products/Release/da3-tool \
  --model da3-large \
  --weights /path/to/model.safetensors \
  --input image.png \
  --output depth.bin
```

Saves the depth map as raw float32 (or `.ply` for an unprojected point cloud).

## How the streaming pipeline works

Mirrors the non-loop path of Python's `DA3_Streaming.process_long_sequence`:

1. **Chunk windows** — overlapping windows over the sorted image list (`chunk_size - overlap` stride).
2. **Per-chunk inference** — multi-view forward pass produces depth, depth confidence, and per-patch ray maps for each frame in the chunk.
3. **Ray → camera pose** — geometric port of `get_extrinsic_from_camray`: weighted RANSAC homography on the ray map → QL decomposition → per-view extrinsics (w2c) and intrinsics. No learned camera decoder, so we skip those weights.
4. **Chunk-pair Sim(3) alignment** — for each adjacent chunk pair, gather pixels with both confidences above threshold, weight by `sqrt(c1*c2)`, run IRLS with Huber re-weighting. Yields `(s, R, t)` mapping chunk *i+1* → chunk *i*.
5. **Accumulate** — fold per-pair transforms into prefix transforms (chunk *i* → chunk *0*).
6. **Per-frame outputs** — apply the cumulative Sim(3) to each frame's c2w, write camera_poses.txt + intrinsic.txt; per-chunk world-space point clouds written as binary PLY, then merged.

### Where this differs from the Python reference

| Area | Python | Swift |
|---|---|---|
| Camera estimation | Default: learned `cam_dec` decoder. Optional: `use_ray_pose=True` geometric path | Always geometric ray-pose (cam_dec weights skipped) |
| Backbone ref-view selection | `saddle_balanced` for streaming | Default backbone behavior (port deferred) |
| IRLS conf threshold heuristic | `min(median(c1), median(c2)) * 0.1` | `min(mean(c1), mean(c2)) * 0.1` (MLX has no median) |
| SVD/QR | GPU (CUDA / Triton) | CPU stream (MLX-Swift has no GPU SVD/QR yet) |
| Per-chunk predictions | Spilled to `.npy` between phases | Held in RAM (works for short sequences) |
| Point-cloud subsample | Reservoir sampling at `sample_ratio<1.0` | Always `sample_ratio=1.0` (no subsample) |
| Loop closure / SALAD detector | Implemented | Not ported (Phase 2) |

## Tests

```bash
xcodebuild -scheme MLXDA3-Package -destination 'platform=macOS' \
  -derivedDataPath .xcdd test
```

Two parity fixture tests live under `Tests/MLXDA3Tests/`:

| Test | What it covers | Fixture generator |
|---|---|---|
| `ParityFixtureTests` | Single-image DA3 forward pass vs torch reference | `Scripts/generate_fixtures.py` |
| `StreamingParityFixtureTests` | End-to-end `StreamingPipeline` (poses + intrinsics + sim3) vs `da3_streaming.DA3_Streaming` | `Scripts/generate_streaming_parity_fixture.py` |
| `LoopDetectionFixtureTests` | SALAD `LoopDetector` (descriptors + loop pairs) vs python `loop_utils.LoopDetector` | `Scripts/generate_loop_detection_fixture.py` |
| `Sim3LieGroupTests` | Sim(3) Exp/Log/Compose/Inverse identities (no python dep) | (none — pure Swift) |
| `Sim3LoopOptimizerTests` | Sim(3) LM smoke (loop-closure cost reduces ≥5×) | (none — pure Swift) |
| `Sim3LoopOptimizerFixtureTests` | Sim(3) LM trajectory parity vs python `Sim3LoopOptimizer` on a noisy 8-pose ring | `Scripts/generate_sim3_loop_fixture.py` |
| `LoopClosureParityFixtureTests` | End-to-end loop-enabled streaming (detector + measurement + optimizer + pose accumulation) vs python with `loop_enable=True` on 42 robot_unitree frames | `Scripts/generate_streaming_parity_fixture.py --enable-loop-closure` |

Each test `XCTSkip`s if its fixture or weights aren't present. The fastest
way to run the suite locally is via the convenience script (which sources
your `.claude/local-runners/.env`):

```bash
# Copy the example .env once and edit it with your local paths
cp docs/local-runners.env.example .claude/local-runners/.env
$EDITOR .claude/local-runners/.env

# Then:
bash .claude/local-runners/test.sh                 # full suite
bash .claude/local-runners/test.sh ParityFixtureTests   # one suite
bash .claude/local-runners/regen-fixtures.sh       # rebuild every parity fixture (~80 min)
bash .claude/local-runners/bench.sh                # streaming benchmarks (Swift + Python)
```

Or run by hand: generate the fixture and pass weights via env var:

```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run --script \
    Scripts/generate_streaming_parity_fixture.py \
    --image-dir tmp/da3_frames \
    --out Tests/Fixtures \
    --weights /path/to/python/Depth-Anything-3/da3_streaming/weights/model.safetensors \
    --weights-config /path/to/python/Depth-Anything-3/da3_streaming/weights/config.json \
    --chunk-size 4 --overlap 2 --limit 8

TEST_RUNNER_DA3_STREAMING_WEIGHTS=/path/to/model.safetensors \
xcodebuild -scheme MLXDA3-Package -destination 'platform=macOS' \
  -derivedDataPath .xcdd test \
  -only-testing:MLXDA3Tests/StreamingParityFixtureTests
```

Streaming-test tolerances are deliberately loose to absorb the residual
fp32 matmul reduction-order drift between MLX and PyTorch (documented in
`tasks/todo.md`); the substantive porting bugs (pos-embed interpolation
mode) are already fixed and verified to 1.5e-6 vs torch.

## Benchmarks

8-frame fixture, `chunk_size=4`, `overlap=2`, 1 warmup + 3–5 measured
iterations. Apple Silicon (M-series); see `Benchmarks/README.md` for
exact hardware footnote when committing numbers from a fresh machine.

| Backend | dtype | mean_s | median_s | min_s | max_s | load_s |
|---|---|---|---|---|---|---|
| mlx-swift | fp16 | 2.65 | 2.65 | 2.57 | 2.71 | 0.42 |
| torch (CPU) | fp32 | 241.54 | 241.75 | 241.00 | 241.85 | 4.24 |

Swift (mlx-swift, fp16, GPU via Metal) is **~91× faster per iteration**
than torch on CPU for the streaming pipeline on this fixture. Apples-to-
apples vs torch MPS isn't covered here (the python `da3_streaming`
reference doesn't run on Metal without engineering work — pypose's
graph-based ops have CUDA-only paths). The CPU number is what's
realistically reproducible on a Mac without a GPU passthrough.

Reproduction:

```bash
# Swift (after building da3-streaming-bench)
.xcdd/Build/Products/release/da3-streaming-bench \
  --model da3-giant \
  --weights /path/to/model.safetensors \
  --image-dir tmp/da3_frames \
  --chunk-size 4 --overlap 2 --limit 8 --warmup 1 --iterations 5

# Python
KMP_DUPLICATE_LIB_OK=TRUE uv run --script Benchmarks/torch_da3_streaming_bench.py \
  --image-dir tmp/da3_frames \
  --weights /path/to/python/Depth-Anything-3/da3_streaming/weights/model.safetensors \
  --weights-config /path/to/python/Depth-Anything-3/da3_streaming/weights/config.json \
  --chunk-size 4 --overlap 2 --limit 8 --warmup 1 --iterations 5
```

Hardware footnote and full results table are in
[Benchmarks/README.md](Benchmarks/README.md).

## Project layout

```
mlx-swift-da3/
├── Package.swift
├── Sources/
│   ├── MLXDA3/                 # core model
│   └── MLXDA3Streaming/        # chunked streaming pipeline
│       ├── ImageDirectory.swift
│       ├── MultiViewPreprocessor.swift
│       ├── ChunkIndex.swift
│       ├── ChunkPredictions.swift
│       ├── StreamingInference.swift
│       ├── RayPose.swift              # ray → extrinsics + intrinsics
│       ├── Sim3Alignment.swift        # IRLS robust Sim(3)
│       ├── PointMaps.swift            # depth → world points + apply_sim3
│       ├── ChunkAlignment.swift       # weighted_align_point_maps
│       ├── PLYWriter.swift            # binary PLY save + merge
│       ├── CameraPosesIO.swift        # camera_poses.txt + intrinsic.txt
│       ├── StreamingOrchestrator.swift  # file-IO orchestrator (CLI uses this)
│       ├── StreamingPipeline.swift      # high-level in-memory API
│       └── loop/                        # Phase 2: loop closure
│           ├── Salad.swift              # SALAD VPR model (DinoV2-B/14 + aggregator)
│           ├── SaladWeightLoading.swift
│           ├── LoopDetector.swift       # batching + brute-force cosine matching + NMS
│           ├── Sim3LieGroup.swift       # Sim(3) Exp/Log/multiply/inverse (Sophus closed form)
│           └── Sim3LoopOptimizer.swift  # LM pose-graph optimizer with numerical Jacobian
├── Tests/
│   ├── Fixtures/                # parity fixtures (.safetensors + .json)
│   └── MLXDA3Tests/
│       ├── ForwardTests.swift
│       ├── ParityFixtureTests.swift            # single-image parity
│       └── StreamingParityFixtureTests.swift   # streaming parity
├── Scripts/
│   ├── generate_fixtures.py
│   └── generate_streaming_parity_fixture.py
├── Benchmarks/
│   ├── README.md
│   ├── torch_da3_bench.py
│   └── torch_da3_streaming_bench.py
├── Tools/
│   ├── da3-tool/               # single-image CLI
│   ├── da3-bench/              # benchmark CLI
│   └── da3-streaming-tool/     # streaming pipeline CLI
├── Tests/MLXDA3Tests/
├── Benchmarks/
├── PLAN.md                     # original DA3 model port plan
├── tasks/todo.md               # streaming port plan + status
└── README.md
```

## Acknowledgements

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — original model and streaming pipeline (ByteDance Seed)
- [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) — DA3-Streaming is built on VGGT-Long's chunking/alignment ideas
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — MLX framework for Swift
