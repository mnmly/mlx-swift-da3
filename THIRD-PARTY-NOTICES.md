# Third-party notices

`mlx-swift-da3` is a Swift port of existing research code. This file records what each
part is derived from and under which licence.

## Apache-2.0 — the default for this repository

`LICENSE` (Apache-2.0) covers the code in this repository except where noted below.

| Target / files | Derived from | Upstream licence |
|---|---|---|
| `Sources/MLXDA3` | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — © 2025 ByteDance Ltd. and/or its affiliates | Apache-2.0 |
| `Sources/MLXDA3/DinoVisionTransformer.swift`, `Block.swift`, `Attention.swift`, `FFN.swift`, `Embeddings.swift`, `RoPE2D.swift` | [DINOv2](https://github.com/facebookresearch/dinov2) — © Meta Platforms, Inc. and affiliates, as vendored by Depth Anything 3 | Apache-2.0 |
| `Sources/MLXDA3Streaming` | `da3_streaming/` in Depth Anything 3 (chunked streaming pipeline, Sim(3) alignment, pose-graph optimiser) | Apache-2.0 |

Runtime dependencies:

- [mlx-swift](https://github.com/ml-explore/mlx-swift) — MIT
- [swift-transformers](https://github.com/huggingface/swift-transformers) — Apache-2.0
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) — Apache-2.0

## GPL-3.0 — `MLXDA3SALAD` (loop-closure place recognition)

`Sources/MLXDA3SALAD` (`Salad.swift`, `SaladWeightLoading.swift`, `LoopDetector.swift`) is a
port of [serizba/salad](https://github.com/serizba/salad) ("Optimal Transport Aggregation for
Visual Place Recognition", Izquierdo & Civera, CVPR 2024), which is licensed **GPL-3.0**. The
port is a derivative work and is therefore covered by GPL-3.0, not by this repository's
Apache-2.0 licence.

It is isolated in its own SwiftPM target and product for that reason:

- `MLXDA3` and `MLXDA3Streaming` do **not** depend on it. The streaming pipeline accepts
  loop constraints as plain data (`StreamingPipeline.predict(images:loopConstraints:)`), so
  any place-recognition front end can supply them.
- Only `da3-loop-tool` and the `DA3Demo` example app link it.

If you distribute a product that links `MLXDA3SALAD`, that product must comply with
GPL-3.0. To avoid this, depend on `MLXDA3` / `MLXDA3Streaming` only and supply loop
constraints from your own detector.

The SALAD weights (`dino_salad.ckpt`, from the serizba/salad releases) are likewise
governed by that project's terms and are not redistributed here.

## Model weights

No checkpoints are redistributed in this repository. DA3 weights come from the
[depth-anything](https://huggingface.co/depth-anything) Hugging Face organisation and remain
subject to their own model licences.
