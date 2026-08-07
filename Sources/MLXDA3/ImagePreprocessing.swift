import CoreGraphics
import Foundation
import MLX

/// Port of `src/depth_anything_3/utils/io/input_processor.py::InputProcessor`.
///
/// Python's pipeline is two sequential `cv2.resize` calls, not one:
///
///   1. `_resize_longest_side` — scale so `max(w, h) == process_res`, sizes via
///      `int(round(dim * scale))`.
///   2. `_make_divisible_by_resize` — round each side to the *nearest* multiple of
///      the patch size (ties round up), resizing again.
///
/// Each step picks `cv2.INTER_AREA` when shrinking and `cv2.INTER_CUBIC` when growing,
/// and each step round-trips through uint8. Matching that chain (rather than a single
/// CoreGraphics draw) is what keeps the model input within ~1 LSB of the reference.
public enum DA3ImagePreprocessing {
    public static let imagenetMean: [Float] = [0.485, 0.456, 0.406]
    public static let imagenetStd: [Float] = [0.229, 0.224, 0.225]

    public enum Interpolation {
        /// `cv2.INTER_AREA` — exact area-weighted average. Downscale only.
        case area
        /// `cv2.INTER_CUBIC` — Catmull-Rom-style bicubic, a = -0.75, replicated border.
        case cubic
    }

    // MARK: - Target size

    /// Python `nearest_multiple`: floor/ceil to `patch`, ties resolved upward.
    public static func nearestMultiple(_ x: Int, _ patch: Int) -> Int {
        let down = (x / patch) * patch
        let up = down + patch
        return (up - x) <= (x - down) ? up : down
    }

    /// Final (width, height) for `upper_bound_resize`, mirroring both python resize stages.
    public static func targetSize(
        width: Int, height: Int, processRes: Int, patchSize: Int
    ) -> (width: Int, height: Int) {
        var w = width
        var h = height
        let longest = max(w, h)
        if longest != processRes {
            let scale = Double(processRes) / Double(longest)
            // Python `int(round(...))` is banker's rounding.
            w = max(1, Int((Double(width) * scale).rounded(.toNearestOrEven)))
            h = max(1, Int((Double(height) * scale).rounded(.toNearestOrEven)))
        }
        return (max(1, nearestMultiple(w, patchSize)), max(1, nearestMultiple(h, patchSize)))
    }

    // MARK: - Resampling

    /// Resize `[H, W, C]` float32 (values in 0...255) with a separable filter.
    public static func resize(
        _ image: MLXArray, height dstH: Int, width dstW: Int, method: Interpolation
    ) -> MLXArray {
        let srcH = image.dim(0)
        let srcW = image.dim(1)
        let channels = image.dim(2)
        var out = image

        if dstH != srcH {
            let wy = weightMatrix(src: srcH, dst: dstH, method: method)
            out = matmul(wy, out.reshaped([srcH, srcW * channels]))
                .reshaped([dstH, srcW, channels])
        }
        if dstW != srcW {
            let wx = weightMatrix(src: srcW, dst: dstW, method: method)
            let columnMajor = out.transposed(1, 0, 2).reshaped([srcW, dstH * channels])
            out = matmul(wx, columnMajor)
                .reshaped([dstW, dstH, channels])
                .transposed(1, 0, 2)
        }
        return out
    }

    /// Every view in a batch resizes between the same two sizes, so the (dst × src)
    /// weight matrix is built once and reused.
    private struct WeightKey: Hashable {
        let src: Int
        let dst: Int
        let area: Bool
    }

    private static let weightCacheLock = NSLock()
    private static var weightCache: [WeightKey: MLXArray] = [:]

    private static func weightMatrix(src: Int, dst: Int, method: Interpolation) -> MLXArray {
        let key = WeightKey(src: src, dst: dst, area: method == .area)
        weightCacheLock.lock()
        defer { weightCacheLock.unlock() }
        if let cached = weightCache[key] { return cached }

        let values: [Float] = method == .area
            ? areaWeights(src: src, dst: dst)
            : cubicWeights(src: src, dst: dst)
        let matrix = MLXArray(values, [dst, src])
        eval(matrix)
        // Bound the cache: a session sees a handful of distinct sizes, but a long-lived
        // process fed arbitrary image sizes should not accumulate matrices forever.
        if weightCache.count >= 32 { weightCache.removeAll(keepingCapacity: true) }
        weightCache[key] = matrix
        return matrix
    }

    /// `[dst, src]` row-normalized overlap areas — the continuous form of `cv2.INTER_AREA`.
    private static func areaWeights(src: Int, dst: Int) -> [Float] {
        let scale = Double(src) / Double(dst)
        var w = [Float](repeating: 0, count: dst * src)
        for d in 0 ..< dst {
            let lo = Double(d) * scale
            let hi = lo + scale
            var sum: Float = 0
            var j = max(0, Int(lo.rounded(.down)))
            while j < src, Double(j) < hi {
                let overlap = min(hi, Double(j + 1)) - max(lo, Double(j))
                if overlap > 0 {
                    let v = Float(overlap)
                    w[d * src + j] = v
                    sum += v
                }
                j += 1
            }
            if sum > 0 {
                for j in 0 ..< src where w[d * src + j] != 0 { w[d * src + j] /= sum }
            }
        }
        return w
    }

    /// `[dst, src]` bicubic taps matching OpenCV's `interpolateCubic` (A = -0.75).
    private static func cubicWeights(src: Int, dst: Int) -> [Float] {
        let scale = Double(src) / Double(dst)
        let a = -0.75
        var w = [Float](repeating: 0, count: dst * src)
        for d in 0 ..< dst {
            let fx = (Double(d) + 0.5) * scale - 0.5
            let base = Int(fx.rounded(.down))
            let x = fx - Double(base)
            var c = [Double](repeating: 0, count: 4)
            c[0] = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
            c[1] = ((a + 2) * x - (a + 3)) * x * x + 1
            c[2] = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
            c[3] = 1 - c[0] - c[1] - c[2]
            for k in 0 ..< 4 {
                // cv2 clamps sample coordinates to the border (BORDER_REPLICATE).
                let idx = min(max(base - 1 + k, 0), src - 1)
                w[d * src + idx] += Float(c[k])
            }
        }
        return w
    }

    // MARK: - CGImage decoding

    /// Decode a `CGImage` to `[H, W, 3]` float32 in 0...255 at its native size.
    /// No resampling happens here — the resize is done afterwards so it matches cv2.
    public static func decodeRGB(_ image: CGImage) -> MLXArray {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &pixels,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  // noneSkipLast, not premultiplied: PIL's convert("RGB") drops alpha
                  // rather than compositing it.
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              )
        else {
            fatalError("Failed to create CGContext while decoding image")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let rgba = MLXArray(pixels, [height, width, 4])
        return rgba[0..., 0..., ..<3].asType(.float32)
    }

    // MARK: - Full pipeline

    /// Resize a decoded image to the model's processing resolution, mirroring the two
    /// python resize stages including the uint8 round-trip between them.
    ///
    /// - Parameter rgb: `[H, W, 3]` float32 in 0...255.
    /// - Returns: `[targetH, targetW, 3]` float32 in 0...255, integer-valued.
    public static func resizeToProcessingResolution(
        _ rgb: MLXArray, processRes: Int, patchSize: Int
    ) -> MLXArray {
        let srcH = rgb.dim(0)
        let srcW = rgb.dim(1)
        var out = rgb

        // Stage 1 — longest side to processRes.
        let longest = max(srcW, srcH)
        var stageW = srcW
        var stageH = srcH
        if longest != processRes {
            let scale = Double(processRes) / Double(longest)
            stageW = max(1, Int((Double(srcW) * scale).rounded(.toNearestOrEven)))
            stageH = max(1, Int((Double(srcH) * scale).rounded(.toNearestOrEven)))
            out = quantize(
                resize(out, height: stageH, width: stageW, method: scale > 1.0 ? .cubic : .area)
            )
        }

        // Stage 2 — each side to the nearest patch multiple.
        let finalW = max(1, nearestMultiple(stageW, patchSize))
        let finalH = max(1, nearestMultiple(stageH, patchSize))
        if finalW != stageW || finalH != stageH {
            let upscale = (finalW > stageW) || (finalH > stageH)
            out = quantize(
                resize(out, height: finalH, width: finalW, method: upscale ? .cubic : .area)
            )
        }
        return out
    }

    /// cv2 writes uint8: round half away from zero, clamp to 0...255.
    private static func quantize(_ x: MLXArray) -> MLXArray {
        clip(MLX.round(x), min: MLXArray(Float(0)), max: MLXArray(Float(255)))
    }

    /// ImageNet normalization on `[.., H, W, 3]` float32 in 0...255.
    public static func normalize(_ rgb: MLXArray) -> MLXArray {
        let mean = MLXArray(imagenetMean, [1, 1, 3])
        let std = MLXArray(imagenetStd, [1, 1, 3])
        return (rgb / 255.0 - mean) / std
    }

    /// Decode + resize a `CGImage` to `[H, W, 3]` float32 in 0...255 (integer-valued).
    public static func processedRGB(
        _ image: CGImage, processRes: Int, patchSize: Int
    ) -> MLXArray {
        resizeToProcessingResolution(
            decodeRGB(image), processRes: processRes, patchSize: patchSize
        )
    }
}
