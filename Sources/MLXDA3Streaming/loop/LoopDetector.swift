import CoreGraphics
import Foundation
import MLX
import MLXNN

/// Single detected loop closure: a pair of frame indices and the cosine
/// similarity between their SALAD descriptors.
public struct LoopClosure: Equatable {
    /// Lower-indexed frame (always `< b`).
    public let a: Int
    /// Higher-indexed frame.
    public let b: Int
    /// Cosine similarity in [-1, 1] (typically [0, 1] since descriptors are L2-normalised).
    public let similarity: Float

    public init(a: Int, b: Int, similarity: Float) {
        self.a = a
        self.b = b
        self.similarity = similarity
    }
}

/// Loop closure detection over a sequence of frames using SALAD VPR
/// descriptors. Mirrors `da3_streaming/loop_utils/loop_detector.py`'s
/// LoopDetector with two simplifications:
/// - **No faiss**: brute-force normalised matmul. Fine for sequences up to
///   a few thousand frames; faiss only matters at very large scale.
/// - **No CKPT load**: caller passes a `SaladModel` that's already been
///   loaded via `loadSaladModel(weightsURL:)`.
public struct LoopDetector {
    public struct Config {
        public var imageSize: (height: Int, width: Int) = (336, 336)
        public var batchSize: Int = 32
        public var similarityThreshold: Float = 0.85
        public var topK: Int = 5
        /// Minimum frame distance for a candidate to count as a loop. Mirrors
        /// `abs(i - neighbor_idx) > 10` in the python reference.
        public var minFrameDistance: Int = 10
        public var useNMS: Bool = true
        public var nmsThreshold: Int = 25

        public init() {}
    }

    public let model: SaladModel
    public let config: Config

    public init(model: SaladModel, config: Config = Config()) {
        self.model = model
        self.config = config
    }

    /// Run loop detection on a sequence of frames. Returns the descriptors
    /// (`[N, D]`) and the filtered loop pairs (sorted by descending similarity).
    public func detect(images: [CGImage]) -> (descriptors: MLXArray, loops: [LoopClosure]) {
        let descriptors = extractDescriptors(images: images)
        let loops = findLoopClosures(descriptors: descriptors)
        return (descriptors, loops)
    }

    /// Run SALAD over batches of frames and stack the L2-normalised descriptors.
    public func extractDescriptors(images: [CGImage]) -> MLXArray {
        var batches: [MLXArray] = []
        let h = config.imageSize.height
        let w = config.imageSize.width

        for start in stride(from: 0, to: images.count, by: config.batchSize) {
            let end = min(start + config.batchSize, images.count)
            let chunk = Array(images[start..<end])
            let batchTensor = preprocessBatch(chunk, h: h, w: w)  // [B, H, W, 3] float32
            let desc = model(batchTensor)                         // [B, D]
            eval(desc)
            batches.append(desc.asType(.float32))
        }

        return MLX.concatenated(batches, axis: 0)  // [N, D]
    }

    /// Brute-force top-K cosine similarity on already-normalised descriptors.
    public func findLoopClosures(descriptors: MLXArray) -> [LoopClosure] {
        let n = descriptors.dim(0)
        if n < 2 { return [] }

        // descriptors are already L2-normalised by SaladModel; cosine = inner product.
        let sim = descriptors.matmul(descriptors.transposed(axes: [1, 0]))  // [N, N]
        eval(sim)
        let simCPU = sim.asArray(Float.self)

        // For each i, take top-(K+1) (the +1 is i itself). Filter by frame distance + threshold.
        var pairs: [LoopClosure] = []
        let K = min(config.topK + 1, n)

        // Per-row top-K via partial sort (fast enough for n ≤ 1k)
        for i in 0..<n {
            let row = Array(simCPU[(i * n)..<((i + 1) * n)])
            // sort indices by descending similarity, take top-K
            var idx = Array(0..<n)
            idx.sort { row[$0] > row[$1] }
            for j in idx.prefix(K) {
                if j == i { continue }
                if abs(i - j) <= config.minFrameDistance { continue }
                let s = row[j]
                if s <= config.similarityThreshold { continue }
                let lo = min(i, j)
                let hi = max(i, j)
                pairs.append(LoopClosure(a: lo, b: hi, similarity: s))
            }
        }

        // Deduplicate (i,j) pairs (each loop appears twice from both rows)
        var dedup: [String: LoopClosure] = [:]
        for p in pairs {
            let key = "\(p.a)-\(p.b)"
            if let existing = dedup[key] {
                if p.similarity > existing.similarity { dedup[key] = p }
            } else {
                dedup[key] = p
            }
        }
        var unique = Array(dedup.values)
        unique.sort { $0.similarity > $1.similarity }

        if config.useNMS && config.nmsThreshold > 0 {
            unique = applyNMS(unique, threshold: config.nmsThreshold)
        }

        // Mirror python's _ensure_decending_order: store as (max(a,b), min(a,b)).
        // We already store as (lo, hi); python puts the larger first. Match that
        // by swapping for the test fixture's expected order.
        return unique.map { LoopClosure(a: $0.b, b: $0.a, similarity: $0.similarity) }
    }

    /// NMS: greedy filter that suppresses neighbouring (frame-distance) loop
    /// candidates around accepted ones. Mirrors `_apply_nms_filter`.
    private func applyNMS(_ loops: [LoopClosure], threshold: Int) -> [LoopClosure] {
        guard !loops.isEmpty else { return [] }
        let sorted = loops.sorted { $0.similarity > $1.similarity }
        var maxFrame = 0
        for p in loops { maxFrame = max(maxFrame, max(p.a, p.b)) }

        var suppressed = Set<Int>()
        var result: [LoopClosure] = []
        for p in sorted {
            // python uses (idx1, idx2, sim) where idx1 < idx2 — same as our (a, b)
            let idx1 = p.a
            let idx2 = p.b
            if suppressed.contains(idx1) || suppressed.contains(idx2) { continue }
            result.append(p)
            // suppress range1 = [max(0, idx1 - thr), min(idx1 + thr + 1, idx2))
            let s1 = max(0, idx1 - threshold)
            let e1 = min(idx1 + threshold + 1, idx2)
            for k in s1..<e1 { suppressed.insert(k) }
            // suppress range2 = [max(idx1+1, idx2 - thr), min(idx2 + thr + 1, max_frame + 1))
            let s2 = max(idx1 + 1, idx2 - threshold)
            let e2 = min(idx2 + threshold + 1, maxFrame + 1)
            for k in s2..<e2 { suppressed.insert(k) }
        }
        return result
    }

    // MARK: - Preprocessing

    /// Resize each `CGImage` to (h, w) with bilinear interp + ImageNet normalize,
    /// then stack into a `[B, H, W, 3]` float32 tensor.
    private func preprocessBatch(_ images: [CGImage], h: Int, w: Int) -> MLXArray {
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        var combined = [Float]()
        combined.reserveCapacity(images.count * h * w * 3)
        for img in images {
            let pixels = renderResized(img, width: w, height: h)
            for y in 0..<h {
                for x in 0..<w {
                    let src = (y * w + x) * 4  // RGBA
                    for c in 0..<3 {
                        let pixel = Float(pixels[src + c]) / 255.0
                        combined.append((pixel - mean[c]) / std[c])
                    }
                }
            }
        }
        return MLXArray(combined, [images.count, h, w, 3])
    }

    private func renderResized(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                  data: &pixelData,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              )
        else {
            fatalError("Failed to create CGContext for SALAD preprocess")
        }
        // PIL.Resize default is bilinear; CGContext .high is bicubic-ish.
        // For VPR descriptors a small interp difference is fine — the ranking
        // is robust to it.
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelData
    }
}

public extension LoopDetector {
    /// Convenience: load SALAD weights and build a detector in one call.
    static func fromPretrained(_ weightsPath: String, config: Config = Config(), dtype: DType = .float32) throws -> LoopDetector {
        let model = try loadSaladModel(weightsURL: URL(fileURLWithPath: weightsPath), dtype: dtype)
        return LoopDetector(model: model, config: config)
    }
}
