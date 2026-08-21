# DA3Demo — SwiftUI macOS sample app

Single-window SwiftUI app that drives every public pipeline in mlx-swift-da3.
Two modes via a segmented control at the top:

**Single image** (`DepthAnything3Pipeline`):
- Pick an image + DA3 weights (`da3-large` is the typical config name)
- Optionally tweak the model variant + processing resolution
- Output: a turbo-colormap depth heatmap rendered side-by-side with the input

**Video streaming** (`StreamingPipeline` + Phase 2 loop closure):
- Pick a video (mp4 / mov / etc.) + DA3 weights (`da3-giant` for the
  multi-view nested checkpoint)
- Optional: pick `dino_salad.safetensors` and toggle "Enable loop closure"
- Adjust fps / max frames / chunk-size / overlap
- Output: 3D camera trajectory + frustums (RealityKit, drag to orbit)
- "Save PLY + poses…" writes `pcd/*.ply` + `combined_pcd.ply` +
  `camera_poses.txt` + `intrinsic.txt`

APIs exercised across both modes:

- `DepthAnything3Pipeline.fromPretrained(...)` (single image)
- `StreamingPipeline.predict(images:)`
- `LoopDetector.detect(images:)` (SALAD VPR)
- `StreamingPipeline.computeLoopMeasurement(...)`
- `StreamingPipeline.predict(images:loopConstraints:)`
- `PLYWriter.saveConfidentPointCloud(...)` + `PLYWriter.mergePLYFiles(...)`
- `CameraPosesIO.savePoses(...)`

## One-time setup: wire the local SwiftPM dependency

The xcodeproj as committed is a stock SwiftUI template. To make it import
`MLXDA3` and `MLXDA3Streaming`, add the parent repo as a local SwiftPM
package:

1. Open `DA3Demo.xcodeproj`
2. **File → Add Package Dependencies…**
3. Click **Add Local…** (bottom-left)
4. Choose the parent directory: `mlx-swift-da3/` (two levels up from this
   xcodeproj)
5. In the next sheet, tick both **MLXDA3** and **MLXDA3Streaming**
   libraries and add them to the **DA3Demo** target.
6. Build (`⌘B`). The first build will resolve mlx-swift, swift-transformers,
   etc., which takes ~30s.

If Xcode complains about the deployment target, set the project to macOS 14+
(MLX requires Metal 3).

## Where to get weights

- **DA3 weights**: same `model.safetensors` used by the CLI tools — see the
  parent README's "Setup" section.
- **SALAD weights**: convert the upstream `.ckpt` first:

  ```bash
  uv run --script ../../../mlx-swift-salad/Scripts/convert_salad_to_safetensors.py \
      --in /path/to/dino_salad.ckpt \
      --out /path/to/dino_salad.safetensors
  ```

## Notes / limitations

- After a successful run, a RealityKit pose viewer appears in the bottom
  half of the window — drag to orbit, scroll to zoom. Each frame's camera
  is drawn as a hue-graded frustum (apex = camera centre, opening toward
  +Z); consecutive frames are connected by a white trajectory line; world
  axes are RGB at the centroid.
- The demo reloads the model in `savePLY` to recompute per-chunk world
  points, since the published `StreamingPrediction` only retains poses +
  intrinsics (per-chunk depth/conf is large). Not a concern for short
  clips; for longer videos cache `prediction.perChunk` upstream.
- Frame extraction is single-threaded `AVAssetImageGenerator`. For >100
  frames consider extracting upstream with `ffmpeg`.
- The pose viewer does not render the point cloud (yet) — only camera
  trajectory + frustums. The PLY is on disk after "Save PLY + poses…";
  open it in Meshlab / Blender / Cloud Compare for the dense view.

## Files

```
DA3Demo/
├── DA3DemoApp.swift     # SwiftUI App entry
├── ContentView.swift    # mode picker + form + status UI + viewers
├── PipelineRunner.swift # @MainActor ObservableObject, runs both modes
├── FrameExtractor.swift # AVAssetImageGenerator wrapper (streaming mode)
├── DepthMapView.swift   # turbo-colormap heatmap (single-image mode)
└── PoseSceneView.swift  # RealityKit pose viewer (streaming mode)
```
