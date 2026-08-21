# mlx-swift-da3

Swift port of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for Apple Silicon, built on [mlx-swift](https://github.com/ml-explore/mlx-swift).

Two products:

- **`MLXDA3`** — the DA3 model (DinoV2 backbone + DPT/DualDPT head) and a `da3-tool` CLI for single-image / multi-view inference.
- **`MLXDA3Streaming`** — port of the [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming) chunked inference pipeline (chunk alignment + Sim(3) pose-graph refinement) and a `da3-streaming-tool` CLI that produces camera poses + a combined point cloud from a directory of frames.
SALAD visual place recognition (used for loop detection) lives in its own repository,
[mlx-swift-salad](https://github.com/mnmly/mlx-swift-salad), because it is **GPL-3.0** unlike the
Apache-2.0 code here. `MLXDA3` and `MLXDA3Streaming` do not depend on it; only `da3-loop-tool`,
the `DA3Demo` example app, and the test target link it. See [Licence](#licence).

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
| `ref_view_strategy` (`first` / `middle` / `saddle_balanced` / `saddle_sim_range`) | ✅ parity-tested vs python selector; default `saddle_balanced` as upstream |
| Input preprocessing (`upper_bound_resize` + cv2 INTER_AREA/INTER_CUBIC chain) | ✅ parity-tested (mean 0.05 uint8 levels vs cv2) |
| Numerical parity check vs Python reference | ✅ per-stage fixtures; see [Tests](#tests) |

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

Loop closure additionally requires the SALAD descriptor weights, and the converter
lives in the [mlx-swift-salad](https://github.com/mnmly/mlx-swift-salad) package
(GPL-3.0, which is why it is not vendored here):

```bash
git clone https://github.com/mnmly/mlx-swift-salad.git ../mlx-swift-salad
```

The upstream release ships the weights as a PyTorch `.ckpt` saved on CUDA — convert to
MLX-loadable safetensors first:

```bash
uv run --script ../mlx-swift-salad/Scripts/convert_salad_to_safetensors.py \
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
import MLXSALAD

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
uv run --script ../mlx-swift-salad/Scripts/convert_salad_to_safetensors.py \
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

// 2. Run SALAD loop detection on the same frames (needs `import MLXSALAD`).
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
--resolution      Processing resolution (default: 504, matches python `process_res`)
--dtype           float16 | float32 (default: float16)
--ref-view-strategy  first | middle | saddle_balanced | saddle_sim_range
                     (default: saddle_balanced, matching python)
--limit           Cap input frame count (0 = all)
--pcd-conf-coef   PLY conf threshold = mean(conf) * coef (default: 0.75)
```

## da3-video — locked-off footage to an RGBD video

For video whose camera never moves — a locked-off plate, an animated still — the
useful pipeline is not the streaming one. There is no parallax for multi-view
attention to exploit, and per-frame pose estimation only injects jitter into a scene
that cannot have moved.

`da3-video --static-camera` instead estimates the geometry **once**: sample frames
across the clip, take the per-pixel median of inverse depth (the model's frame-to-frame
variation on static geometry is pure noise — measured at 18.7% per-pixel spread and
30.2% global scale drift on a 25 s 4K clip, which reads as the whole point cloud
breathing), guided-upsample it to source resolution, and stack it under every colour
frame.

```bash
.xcdd/Build/Products/Release/da3-video \
  --input clip.mov --output clip.rgbd.mp4 \
  --model da3-large --weights /path/to/model.safetensors \
  --resolution 2072 --static-camera --samples 51 \
  --near-percentile 99.5 --far-percentile 0.5
```

Output is one HEVC file at twice the source height — colour on top, inverse depth
normalized to 0...1 below — plus a `<clip>.stereodepth.json` sidecar describing the
mapping. A 752-frame 4K clip takes ~90 s end to end.

### `--depth-sequence`: per-frame depth without the breathing

A locked-off camera does not mean a locked-off *scene*. If subjects move — boats
crossing water, a figure walking — a single static map pins them to whatever depth
the background had there. But naively inferring every frame brings back the ~30%
global scale drift that makes the whole cloud surge toward the viewer.

`--depth-sequence` keeps both. For each frame it estimates depth, divides out a
robust global scale measured against the static map (the correction lands in
0.98…1.04 — small, precisely because the geometry really is fixed), and then blends:
the static map where the frame agrees with it, the frame's own depth where it does
not. Background locked, movers correct.

Output is a separate **10-bit** depth movie (`<clip>.depth.mov`, HEVC Main10, luma
only) — 4× the codes of an 8-bit band, and a fraction of the size since depth
compresses far better than colour. On a 752-frame 4K clip: ~7 MB.

    --depth-stride N   infer every Nth frame and hold between (depth drifts slowly)
    --agree-below      residual under which the static map is kept outright
    --disagree-above   residual over which the frame's own depth wins

One packing detail worth knowing if you write a consumer: `420YpCbCr10BiPlanar`
keeps its 10-bit samples **left-aligned** in each 16-bit word. Bind the luma plane as
`.r16Unorm` and you get code/1024 in 0...1 directly. Write them right-aligned and you
silently throw away six bits — a 512-step ramp comes back with 14 distinct values.

### `--depth-safetensors`: bring your own depth

DA3 is one estimator, not the only one. `--depth-safetensors` skips inference entirely
and reads an `(N, H, W)` float32 tensor of **inverse** depth, keeping everything after
it — the median, the percentile normalization, the guided upsample, and all three
band encodings. Written for [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
via its Swift port's `vda infer --save-raw`, whose temporal motion modules produce a
far steadier sequence than per-frame DA3 on the same clip:

```bash
vda infer --weights weights/vitl --input clip.mov \
    --input-size 924 --max-res 1920 --save-raw --output-dir out

.xcdd/Build/Products/Release/da3-video \
  --input clip.mov --output clip.rgbd.mp4 \
  --depth-safetensors out/clip_depth.safetensors \
  --static-camera --depth-sequence \
  --near-percentile 99.5 --far-percentile 0.5
```

Pass exactly one of `--weights` or `--depth-safetensors`. Values are taken to be 1/Z
already, so no reciprocal is applied.

With precomputed depth, `--depth-sequence` **skips the stabilization pass**. A video
model has already done that work, so reconciling against a static median is redundant
— and ill-posed here: these models pin the sky to exactly 0, and the robust
frame-to-clip scale is a median over `frame / static`, which collapses to 0 once more
than half the pixels are 0 on both sides. `--agree-below` / `--disagree-above` /
`--depth-stride` therefore do nothing in this combination.

That exact-zero far field is also a bonus downstream: it lands on code 0, so a
consumer culling at zero drops the sky for free, the way block matching used to.

### `--depth-image`: don't store a constant as a video

With a locked-off camera the depth band is *identical in every frame*, so a video of
it is N copies of one image, quantized to 8 bits and put through chroma subsampling
and a lossy codec for nothing. `--depth-image` writes a single 16-bit PNG instead and
leaves the colour clip untouched.

Measured on a 4K painting, counting the distinct codes the *subject* actually
receives (its depth sits in the bottom 6.5% of the range, so it never sees most of an
8-bit ramp):

| storage | subject codes | size |
|---|---|---|
| stacked 8-bit video band | 75 | 79 MB |
| same, 10-bit | 299 | ~79 MB |
| **16-bit still** | **28,803** | **7.7 MB** |

The sidecar then names `depthImage` and sets `video` to the untouched source, so a
consumer samples colour from the original clip and depth from the still. For an 8-bit
band that must stay a video, `--band-gamma 2.2` buys ~1.8× the effective codes; a
consumer that already applies a power curve to the sampled value undoes it by setting
that exponent.

Knobs worth knowing:

| flag | why |
|---|---|
| `--resolution` | see the table under [Benchmarks](#benchmarks): 2072 is the practical ceiling, past which the ViT's geometry degrades |
| `--near-percentile` | inverse depth mapped to white. Lower it (90, 80) to spend more of the 8-bit band on the mid-field instead of the nearest surface |
| `--guide-epsilon` | guided-filter regularization in normalized units. 1e-3 keeps edges crisp; too small transfers the *texture* of the guide into depth |
| `--no-guide` | plain resize instead of the edge-aware upsample |
| `--samples` | frames medianed. More is smoother and costs one forward pass each |
| `--depth-safetensors` | use a precomputed `(N, H, W)` inverse-depth tensor instead of running DA3 |

## da3-tool — single-image / multi-view

```bash
.xcdd/Build/Products/Release/da3-tool \
  --model da3-large \
  --weights /path/to/model.safetensors \
  --input image.png \
  --output depth.bin
```

Saves the depth map as raw float32 (or `.ply` for an unprojected point cloud).

### Batch mode

Swap `--input`/`--output` for `--input-dir`/`--output-dir` to run a whole
directory through one model load — the right shape for per-frame monocular depth
over a video, where re-launching the process per frame would spend more time
mmapping weights than inferring.

```bash
ffmpeg -i input.mov -vf "scale=2072:-2" -q:v 2 /tmp/frames/frame_%06d.jpg

.xcdd/Build/Products/Release/da3-tool \
  --model da3-large \
  --weights /path/to/model.safetensors \
  --input-dir /tmp/frames \
  --output-dir /tmp/depth \
  --resolution 1036
```

Writes one raw float32 `<stem>.bin` per input frame plus a `manifest.json`
recording `frames`, `width`, `height`, `resolution`, and `model` — so readers
don't have to guess the map dimensions. Every frame must produce the same
output size (the tool errors out rather than silently writing ragged maps), so
mixed-aspect input directories are rejected.

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
| Backbone ref-view selection | `saddle_balanced` when `S >= 3` | Same (`RefViewStrategy`, configurable per pipeline) |
| Input resize | PIL decode → cv2 `INTER_AREA`/`INTER_CUBIC`, two stages | Same chain, resampled in MLX; residual ≈0.05 uint8 levels (cv2's 11-bit fixed-point weights vs float32) |
| IRLS conf threshold heuristic | `min(median(c1), median(c2)) * 0.1` | Same (CPU-side median; MLX has no median op) |
| SVD/QR | GPU (CUDA / Triton) | CPU stream (MLX-Swift has no GPU SVD/QR yet). The DLT solve uses QR + a 9×9 SVD instead of a full SVD of the tall matrix — same right singular vectors, ~90× faster |
| RANSAC sampling | `np.random` (process-global) | seeded per view (`RayPose.randomSeed`) — runs are byte-reproducible |
| Per-chunk predictions | Spilled to `.npy` between phases | Held in RAM (works for short sequences) |
| Point-cloud subsample | Reservoir sampling at `sample_ratio<1.0` | Always `sample_ratio=1.0` (no subsample) |
| Loop closure / SALAD detector | Implemented | Ported in the separate [mlx-swift-salad](https://github.com/mnmly/mlx-swift-salad) package (GPL-3.0 — see [Licence](#licence)) |

## Tests

```bash
xcodebuild -scheme MLXDA3-Package -destination 'platform=macOS' \
  -derivedDataPath .xcdd test
```

Two parity fixture tests live under `Tests/MLXDA3Tests/`:

| Test | What it covers | Fixture generator |
|---|---|---|
| `ParityFixtureTests` | Single-image DA3 forward pass vs torch reference | `Scripts/generate_fixtures.py` |
| `PreprocessParityTests` | Image preprocessing vs python `InputProcessor` (resize chain + normalization) | `Scripts/generate_preprocess_fixture.py` |
| `ReferenceViewSelectionTests` | `select_reference_view` / `reorder_by_reference` / `restore_original_order` for all four strategies | `Scripts/generate_ref_view_fixture.py` |
| `BackboneRefViewTests` | Backbone forward is permutation-equivariant with content-based ref-view selection (no weights needed) | (none — pure Swift) |
| `StreamingParityFixtureTests` | End-to-end `StreamingPipeline` (poses + intrinsics + sim3) vs `da3_streaming.DA3_Streaming` | `Scripts/generate_streaming_parity_fixture.py` |
| `Sim3LieGroupTests` | Sim(3) Exp/Log/Compose/Inverse identities (no python dep) | (none — pure Swift) |
| `Sim3LoopOptimizerTests` | Sim(3) LM smoke (loop-closure cost reduces ≥5×) | (none — pure Swift) |
| `Sim3LoopOptimizerFixtureTests` | Sim(3) LM trajectory parity vs python `Sim3LoopOptimizer` on a noisy 8-pose ring | `Scripts/generate_sim3_loop_fixture.py` |
| `LoopClosureParityFixtureTests` | End-to-end loop-enabled streaming (detector + measurement + optimizer + pose accumulation) vs python with `loop_enable=True` on 42 robot_unitree frames | `Scripts/generate_streaming_parity_fixture.py --enable-loop-closure` |

Each test `XCTSkip`s if its fixture or weights aren't present — with everything in place
the suite is 51 tests, 0 skipped. The fastest
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

### Measured agreement with the Python reference

8-frame fixture, fp32 on both sides, DA3NESTED-GIANT-LARGE-1.1, both running
`ref_view_strategy=saddle_balanced`:

| Quantity | Agreement |
|---|---|
| Preprocessed model input | mean 0.0009, max 0.035 (normalized units) |
| Reference-view selection | exact (index + reorder/restore to 1e-6) |
| Camera poses (c2w) | max 0.42, mean 0.082 |
| Principal point `cx, cy` | within 11 px |
| Focal `fx, fy` | ~13% low — the open gap, see `tasks/todo.md` |

Streaming-test tolerances sit just above these numbers so a regression fails.
The residual focal-length error comes from backbone numerical drift (fp32 matmul
reduction order between MLX and PyTorch compounding over 40 layers, then amplified
by a ReLU in the DPT aux pyramid) — it is *not* preprocessing, which now matches
cv2 to ~0.05 uint8 levels.

## Benchmarks

**Build Release for inference** — a Debug build is several times slower and its numbers
are meaningless.

8-frame fixture, `chunk_size=4`, `overlap=2`, fp16, warmup + 5 measured iterations.

| Backend | dtype | median_s | notes |
|---|---|---|---|
| mlx-swift | fp16 | 1.81 | this port, quiet machine |
| torch (CPU) | fp32 | 241.75 | python reference, no Metal path |

The python `da3_streaming` reference has no working Metal path (pypose's graph ops are
CUDA-only), so CPU is what's actually reproducible on a Mac.

### Where the time goes

`da3-streaming-bench --profile` prints a per-phase breakdown:

| phase | share |
|---|---|
| backbone (DINOv2-giant, 4 views × 721 tokens) | ~55% |
| DPT head | ~25% |
| ray → pose (RANSAC + QL) | ~8% |
| chunk Sim(3) alignment | ~4% |
| preprocessing, point maps, pose accumulation | <1% |

The backbone runs at roughly 20 TFLOP/s fp16 — near this hardware's roofline — so the
remaining headroom is in quantization, not in restructuring the graph.

### Process resolution has a ceiling

DINOv2 was trained around 518 px (~1.4k patches). Raising `--resolution` buys real
detail up to a point, then the position-embedding extrapolation and attention
statistics go far enough out of distribution that geometry collapses — the depth
range compresses toward a single plane. Measured on a 4K painting, one frame:

| `--resolution` | patches | depth range | structure vs 1036 | verdict |
|---|---|---|---|---|
| 1036 | 3.0k | 0.082 – 2.878 | 1.000 | clean, soft detail |
| **2072** | 12.3k | 0.162 – 2.155 | 0.984 | **best** — more detail, structure intact |
| 2380 | 16.4k | 0.346 – 1.706 | 0.949 | drifting |
| 2912 | 24.3k | 0.533 – 1.257 | 0.905 | blobs |
| 3836 (4K) | 42.2k | 0.359 – 1.367 | 0.749 | broken |

4K runs without OOM (13 s/frame, 1.3 GB) — it is the *output* that is unusable, not the
compute. For a 4K deliverable, run at 2072 and upsample the depth with `da3-video`'s
guided filter, which takes its edges from the full-resolution colour.

### Memory

`--memory` prints MLX's counters each iteration. `active` stays flat across repeated
in-process runs (no leak); the large `cache` number is MLX's reusable buffer pool, not
live memory. For a long-lived process (GUI, server) bound it:

| `--cache-limit-mb` | cache held | median_s |
|---|---|---|
| unbounded | 14.2 GB | 2.56 |
| 6000 | 6.0 GB | 2.80 |
| 3000 | 3.0 GB | 2.94 |

(measured on a busy machine; the ratios are what matter — ~9% slower to give back 8 GB.)

### Determinism

RANSAC sampling is seeded (`RayPose.randomSeed`), so the same frames produce
byte-identical poses, intrinsics, and point clouds on every run. Change the seed to draw
a different sample.

Reproduction:

```bash
# Swift (after building da3-streaming-bench Release)
.xcdd/Build/Products/Release/da3-streaming-bench \
  --model da3-giant \
  --weights /path/to/model.safetensors \
  --image-dir tmp/da3_frames \
  --chunk-size 4 --overlap 2 --limit 8 --warmup 1 --iterations 5 \
  --profile --memory

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
│   ├── MLXDA3Streaming/        # chunked streaming pipeline
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
│   │   └── loop/                        # Phase 2: loop closure
│   │       ├── Sim3LieGroup.swift       # Sim(3) Exp/Log/multiply/inverse (Sophus closed form)
│   │       └── Sim3LoopOptimizer.swift  # LM pose-graph optimizer with numerical Jacobian
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

## Licence

Apache-2.0 (see [`LICENSE`](LICENSE)), matching upstream Depth Anything 3. Everything in
this repository is Apache-2.0.

SALAD place recognition is a port of [serizba/salad](https://github.com/serizba/salad) and is
therefore **GPL-3.0**; it lives in a separate repository,
[mlx-swift-salad](https://github.com/mnmly/mlx-swift-salad), and is pulled in only by
`da3-loop-tool`, the `DA3Demo` example app, and the test target. Nothing in `MLXDA3` or
`MLXDA3Streaming` links it: the streaming pipeline takes loop constraints as plain data, so
you can supply them from any detector. Depend on `MLXSALAD` only if your product can comply
with GPL-3.0.

Full attribution for every ported component is in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md). No model weights are redistributed here.

## Acknowledgements

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — original model and streaming pipeline (ByteDance Seed)
- [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) — DA3-Streaming is built on VGGT-Long's chunking/alignment ideas
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — MLX framework for Swift
