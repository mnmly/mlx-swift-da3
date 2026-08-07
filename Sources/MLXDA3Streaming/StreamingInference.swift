import CoreGraphics
import Foundation
import MLX
import MLXDA3

/// Run multi-view inference over a list of images and return raw predictions.
///
/// Slice 1A: returns depth/conf/ray/rayConf and processed images. Camera estimation
/// (intrinsics + extrinsics) is added in slice 1B.
public struct StreamingInference {
    public let model: DepthAnything3
    public let preprocessor: MultiViewPreprocessor
    public let dtype: DType
    /// Mirrors python `ref_view_strategy` (default `saddle_balanced`).
    public let refViewStrategy: RefViewStrategy

    public init(
        model: DepthAnything3,
        preprocessor: MultiViewPreprocessor = MultiViewPreprocessor(),
        dtype: DType = .float16,
        refViewStrategy: RefViewStrategy = .saddleBalanced
    ) {
        self.model = model
        self.preprocessor = preprocessor
        self.dtype = dtype
        self.refViewStrategy = refViewStrategy
    }

    /// Run inference on a chunk of images (shape `[1, S, H, W, 3]` internally).
    public func predict(images: [CGImage], outputs: DA3Outputs = .all, dumpInputPath: String? = nil, dumpBackbonePath: String? = nil) -> ChunkPredictions {
        let batch = DA3Profiler.measure("preprocess", sync: { MLX.eval($0.input) }) {
            preprocessor.processBatch(images, dtype: dtype)
        }
        let input = batch.input
        let h = batch.height
        let w = batch.width
        if let path = dumpInputPath {
            try? NpyWriter.writeFloat32(input.asType(.float32), to: path)
        }
        if let path = dumpBackbonePath {
            // Run backbone in isolation and dump last feature stage.
            let (feats, _) = model.backbone(input, refViewStrategy: refViewStrategy)
            let last = feats[feats.count - 1].0  // [B, S, N, C] patch tokens after norm
            try? NpyWriter.writeFloat32(last.asType(.float32), to: path)
        }

        // Matches python predictions.processed_images.
        let processed = batch.processedImages

        let raw = DA3Profiler.measure("model.forward", sync: { MLX.eval($0) }) {
            model(input, outputs: outputs, refViewStrategy: refViewStrategy)
        }
        eval(raw)

        // Squeeze leading B=1 batch axis only. Trailing channel dim handling depends on
        // each output's actual shape (DPT may already collapse the 1-ch dim).
        func dropBatch(_ a: MLXArray) -> MLXArray {
            // [1, S, ...] -> [S, ...]; tolerate [S, ...] passthrough.
            return a.dim(0) == 1 ? a.squeezed(axis: 0) : a
        }
        // depth: [1, S, H, W] (DPT swift output) -> [S, H, W]
        guard let depthRaw = raw["depth"] else {
            fatalError("Model did not return 'depth' output")
        }
        let depth = dropBatch(depthRaw.asType(.float32))

        // conf: depth_conf - 1.0 (matches python convention)
        guard let confRaw = raw["depth_conf"] else {
            fatalError("Model did not return 'depth_conf' output")
        }
        let conf = dropBatch(confRaw.asType(.float32) - 1.0)

        // ray: [1, S, Hp, Wp, 6] (or already squeezed). Drop B and any trailing 1-axis on conf.
        let ray = raw["ray"].map { dropBatch($0.asType(.float32)) }
        let rayConf = raw["ray_conf"].map { rc -> MLXArray in
            var a = dropBatch(rc.asType(.float32))
            // If trailing dim is 1 (conf channel), drop it.
            if a.ndim >= 1, a.dim(a.ndim - 1) == 1 {
                a = a.squeezed(axis: -1)
            }
            return a
        }

        eval(depth, conf)
        if let r = ray { eval(r) }
        if let rc = rayConf { eval(rc) }

        // Camera estimation from rays (mirrors python use_ray_pose=True path).
        var intrinsics: MLXArray? = nil
        var extrinsics: MLXArray? = nil
        if let r = ray, let rc = rayConf {
            let (extW2C, intr) = DA3Profiler.measure(
                "raypose", sync: { MLX.eval($0.0, $0.1) }
            ) {
                RayPose.cameraInfoFromRays(
                    ray: r, rayConf: rc,
                    imageHeight: h, imageWidth: w
                )
            }
            extrinsics = extW2C
            intrinsics = intr
        }

        return ChunkPredictions(
            processedImages: processed,
            depth: depth,
            conf: conf,
            ray: ray,
            rayConf: rayConf,
            intrinsics: intrinsics,
            extrinsics: extrinsics
        )
    }
}
