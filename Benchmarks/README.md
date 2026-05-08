# Benchmarks

Side-by-side timing of the `mlx-swift-da3` ports against their python torch
references. Use the same input frames, same iteration count, same backend
dtype, same chunk/overlap config — anything else makes the comparison
meaningless.

## Streaming pipeline

End-to-end `da3_streaming` run on a fixed set of frames.

### Reproduction

Generate the 8-frame fixture and time both backends:

```bash
# Swift (release build)
xcodebuild -scheme da3-streaming-bench -destination 'platform=macOS' \
  -configuration release -derivedDataPath .xcdd build

.xcdd/Build/Products/release/da3-streaming-bench \
  --model da3-giant \
  --weights /path/to/model.safetensors \
  --image-dir tmp/da3_frames \
  --chunk-size 4 --overlap 2 --limit 8 \
  --warmup 1 --iterations 5

# Python (CPU/MPS)
KMP_DUPLICATE_LIB_OK=TRUE uv run --script torch_da3_streaming_bench.py \
  --image-dir tmp/da3_frames \
  --weights /path/to/python/Depth-Anything-3/da3_streaming/weights/model.safetensors \
  --weights-config /path/to/python/Depth-Anything-3/da3_streaming/weights/config.json \
  --device cpu \
  --chunk-size 4 --overlap 2 --limit 8 \
  --warmup 1 --iterations 5
```

Both benches synchronise before and after each timed iteration (Swift
`MLX.eval(...)`, Python `torch.mps.synchronize()` / `torch.cuda.synchronize()`).

### Hardware

Numbers in the package README's Benchmarks table are from Apple Silicon
M-series on macOS Sonoma+. Document the actual chip / RAM / OS version
when committing benchmark numbers from a fresh machine, since per-iteration
timings vary 2-3× across M1 → M3 generations.

Latest reference run (8-frame fixture, chunk=4, overlap=2):

| Backend | dtype | iter mean (s) | model load (s) |
|---|---|---|---|
| mlx-swift | fp16 | 2.65 | 0.42 |
| torch (CPU) | fp32 | 241.54 | 4.24 |

### Caveats

- Python's `da3_streaming.DA3_Streaming` writes per-chunk `.npy` spill files
  to disk between phases. Swift keeps everything in RAM. The bench uses a
  fresh temp dir per iteration to neutralise OS file cache.
- Python forces `use_ray_pose=True` via monkey-patch so both pipelines run
  the same algorithm. Without that patch python defaults to the learned
  `cam_dec` decoder, which Swift does not port.
- Python defaults to `bf16` autocast on capable GPUs; the `--device cpu`
  path used here runs in fp32. Swift defaults to fp16 — measure both
  fp16 and fp32 if you care about precision/speed tradeoffs.

## Single-image pipeline

See `torch_da3_bench.py` and the `da3-bench` Swift tool. Results in the
package README "Benchmarks" section.
