import CoreGraphics
import Foundation
import MLX

private let imagenetMean: [Float] = [0.485, 0.456, 0.406]
private let imagenetStd: [Float] = [0.229, 0.224, 0.225]

/// Preprocess a batch of images into a single `[1, S, H, W, 3]` MLXArray
/// where (H, W) is determined from the first image and reused for all subsequent
/// views. All views are forced to that resolution so they can be stacked.
public struct MultiViewPreprocessor {
    public let processRes: Int
    public let patchSize: Int

    public init(processRes: Int = 518, patchSize: Int = 14) {
        self.processRes = processRes
        self.patchSize = patchSize
    }

    private func makeDivisible(_ size: Int) -> Int {
        max((size / patchSize) * patchSize, patchSize)
    }

    /// Compute (W, H) target rounded to patch multiples from the first image's aspect ratio.
    public func targetSize(for image: CGImage) -> (width: Int, height: Int) {
        let scale = Float(processRes) / Float(max(image.width, image.height))
        let w = makeDivisible(Int(Float(image.width) * scale))
        let h = makeDivisible(Int(Float(image.height) * scale))
        return (w, h)
    }

    /// Render a single CGImage to a normalized RGB float32 buffer of shape (H, W, 3).
    private func renderNormalized(_ image: CGImage, width: Int, height: Int) -> [Float] {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &pixelData,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              )
        else {
            fatalError("Failed to create CGContext")
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var floatData = [Float](repeating: 0, count: height * width * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let srcIdx = y * bytesPerRow + x * bytesPerPixel
                let dstIdx = (y * width + x) * 3
                for c in 0 ..< 3 {
                    let pixel = Float(pixelData[srcIdx + c]) / 255.0
                    floatData[dstIdx + c] = (pixel - imagenetMean[c]) / imagenetStd[c]
                }
            }
        }
        return floatData
    }

    /// Render a single CGImage to raw uint8 RGB (H, W, 3) — for "processed_images" output.
    public func renderRGBUInt8(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &pixelData,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              )
        else {
            fatalError("Failed to create CGContext")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [UInt8](repeating: 0, count: height * width * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let srcIdx = y * bytesPerRow + x * bytesPerPixel
                let dstIdx = (y * width + x) * 3
                rgb[dstIdx] = pixelData[srcIdx]
                rgb[dstIdx + 1] = pixelData[srcIdx + 1]
                rgb[dstIdx + 2] = pixelData[srcIdx + 2]
            }
        }
        return rgb
    }

    /// Preprocess a batch of images into `[1, S, H, W, 3]` MLXArray with the given dtype.
    /// (H, W) is derived from the first image; all images forced to that shape.
    public func process(_ images: [CGImage], dtype: DType = .float16) -> (input: MLXArray, height: Int, width: Int) {
        precondition(!images.isEmpty, "MultiViewPreprocessor.process requires at least one image")
        let (w, h) = targetSize(for: images[0])

        var combined = [Float]()
        combined.reserveCapacity(images.count * h * w * 3)
        for img in images {
            combined.append(contentsOf: renderNormalized(img, width: w, height: h))
        }
        // [S, H, W, 3] -> [1, S, H, W, 3]
        let arr = MLXArray(combined, [images.count, h, w, 3]).expandedDimensions(axis: 0)
        return (arr.asType(dtype), h, w)
    }
}
