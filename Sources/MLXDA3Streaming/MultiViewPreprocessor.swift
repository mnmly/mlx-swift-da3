import CoreGraphics
import Foundation
import MLX
import MLXDA3

/// Batch preprocessing for multi-view inference, mirroring python `InputProcessor`.
///
/// Each view is resized independently (see `DA3ImagePreprocessing`); if the resulting
/// sizes differ, all views are center-cropped to the smallest H/W exactly as python's
/// `_unify_batch_shapes` does.
public struct MultiViewPreprocessor {
    public let processRes: Int
    public let patchSize: Int

    public init(processRes: Int = 518, patchSize: Int = 14) {
        self.processRes = processRes
        self.patchSize = patchSize
    }

    /// Normalized model input plus the uint8 images the model actually saw.
    public struct Batch {
        /// `[1, S, H, W, 3]` normalized, in the requested dtype.
        public let input: MLXArray
        /// `[S, H, W, 3]` uint8 — python `predictions.processed_images`.
        public let processedImages: MLXArray
        public let height: Int
        public let width: Int
    }

    /// Target (width, height) this preprocessor will produce for a given image.
    public func targetSize(for image: CGImage) -> (width: Int, height: Int) {
        DA3ImagePreprocessing.targetSize(
            width: image.width, height: image.height,
            processRes: processRes, patchSize: patchSize
        )
    }

    /// Preprocess a batch of images.
    public func processBatch(_ images: [CGImage], dtype: DType = .float16) -> Batch {
        precondition(!images.isEmpty, "MultiViewPreprocessor requires at least one image")

        var views = images.map {
            DA3ImagePreprocessing.processedRGB(
                $0, processRes: processRes, patchSize: patchSize
            )
        }

        let height = views.map { $0.dim(0) }.min()!
        let width = views.map { $0.dim(1) }.min()!
        views = views.map { view -> MLXArray in
            let h = view.dim(0)
            let w = view.dim(1)
            if h == height && w == width { return view }
            let top = (h - height) / 2
            let left = (w - width) / 2
            return view[top ..< (top + height), left ..< (left + width), 0...]
        }

        let rgb = stacked(views, axis: 0) // [S, H, W, 3], 0...255
        let input = DA3ImagePreprocessing.normalize(rgb)
            .expandedDimensions(axis: 0)
            .asType(dtype)
        return Batch(
            input: input,
            processedImages: rgb.asType(.uint8),
            height: height,
            width: width
        )
    }

    /// Convenience overload for callers that only need the model input.
    public func process(_ images: [CGImage], dtype: DType = .float16) -> (input: MLXArray, height: Int, width: Int) {
        let batch = processBatch(images, dtype: dtype)
        return (batch.input, batch.height, batch.width)
    }
}
