import CoreGraphics
import Foundation
import MLX

// ImageNet normalization constants
private let imagenetMean: [Float] = [0.485, 0.456, 0.406]
private let imagenetStd: [Float] = [0.229, 0.224, 0.225]

/// Preprocess images for DA3 inference.
///
/// Resizes to a resolution divisible by patch_size (14),
/// normalizes with ImageNet statistics, returns NHWC array.
public struct ImageProcessor {
    public let processRes: Int
    public let patchSize: Int

    public init(processRes: Int = 518, patchSize: Int = 14) {
        self.processRes = processRes
        self.patchSize = patchSize
    }

    /// Round size to nearest multiple of patchSize.
    private func makeDivisible(_ size: Int) -> Int {
        max((size / patchSize) * patchSize, patchSize)
    }

    /// Process a CGImage into a normalized MLXArray.
    ///
    /// - Parameter image: RGB CGImage
    /// - Returns: MLXArray `[1, H', W', 3]` normalized and resized
    public func callAsFunction(_ image: CGImage) -> MLXArray {
        let origW = image.width
        let origH = image.height

        // Resize maintaining aspect ratio
        let scale = Float(processRes) / Float(max(origW, origH))
        let newW = makeDivisible(Int(Float(origW) * scale))
        let newH = makeDivisible(Int(Float(origH) * scale))

        // Render CGImage to float32 RGBA buffer, then extract RGB
        let bytesPerPixel = 4
        let bytesPerRow = newW * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: newH * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &pixelData,
                  width: newW,
                  height: newH,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              )
        else {
            fatalError("Failed to create CGContext for image processing")
        }

        // High-quality interpolation for resize
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: newW, height: newH))

        // Convert RGBA UInt8 -> RGB Float32 [0, 1] and normalize
        var floatData = [Float](repeating: 0, count: newH * newW * 3)
        for y in 0 ..< newH {
            for x in 0 ..< newW {
                let srcIdx = y * bytesPerRow + x * bytesPerPixel
                let dstIdx = (y * newW + x) * 3
                for c in 0 ..< 3 {
                    let pixel = Float(pixelData[srcIdx + c]) / 255.0
                    floatData[dstIdx + c] = (pixel - imagenetMean[c]) / imagenetStd[c]
                }
            }
        }

        // Create MLXArray [1, H, W, 3]
        return MLXArray(floatData, [1, newH, newW, 3])
    }
}
