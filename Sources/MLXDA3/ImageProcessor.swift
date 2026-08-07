import CoreGraphics
import Foundation
import MLX

/// Preprocess a single image for DA3 inference.
///
/// Thin wrapper over `DA3ImagePreprocessing`, which mirrors the python
/// `InputProcessor` resize chain (`upper_bound_resize` + patch-multiple rounding).
public struct ImageProcessor {
    public let processRes: Int
    public let patchSize: Int

    public init(processRes: Int = 518, patchSize: Int = 14) {
        self.processRes = processRes
        self.patchSize = patchSize
    }

    /// Target (width, height) this processor will produce for a given image.
    public func targetSize(for image: CGImage) -> (width: Int, height: Int) {
        DA3ImagePreprocessing.targetSize(
            width: image.width, height: image.height,
            processRes: processRes, patchSize: patchSize
        )
    }

    /// - Returns: `[1, H, W, 3]` float32, ImageNet-normalized.
    public func callAsFunction(_ image: CGImage) -> MLXArray {
        let rgb = DA3ImagePreprocessing.processedRGB(
            image, processRes: processRes, patchSize: patchSize
        )
        return DA3ImagePreprocessing.normalize(rgb).expandedDimensions(axis: 0)
    }
}
