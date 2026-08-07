import Foundation
import MLX
import MLXDA3

/// Mirrors `weighted_align_point_maps` in `loop_utils/sim3utils.py`.
///
/// Aligns chunk2 → chunk1 via robust weighted Sim(3): given two overlapping
/// regions of point maps and confidences, weight each pixel pair by
/// `sqrt(conf1*conf2)`, then run IRLS.
public enum ChunkAlignment {

    public static let confThresholdScale: Float = 0.1

    /// pointMap{1,2}: `[N, H, W, 3]`, conf{1,2}: `[N, H, W]`. Returns `s, R, t` such
    /// that `s * R @ pm2 + t ≈ pm1`.
    public static func alignWeighted(
        pointMap1: MLXArray, conf1: MLXArray,
        pointMap2: MLXArray, conf2: MLXArray,
        irlsConfig: Sim3Alignment.Config = Sim3Alignment.Config()
    ) -> Sim3Alignment.Sim3 {

        let N1 = pointMap1.dim(0); let N2 = pointMap2.dim(0)
        let N = min(N1, N2)
        let H = min(pointMap1.dim(1), pointMap2.dim(1))
        let W = min(pointMap1.dim(2), pointMap2.dim(2))

        let conf1Slice = conf1[0..<N, 0..<H, 0..<W]
        let conf2Slice = conf2[0..<N, 0..<H, 0..<W]
        let pm1Slice = pointMap1[0..<N, 0..<H, 0..<W, 0...]
        let pm2Slice = pointMap2[0..<N, 0..<H, 0..<W, 0...]

        // Python: conf_threshold = min(median(conf1), median(conf2)) * 0.1
        let med1 = median(conf1Slice)
        let med2 = median(conf2Slice)
        let confThreshold = min(med1, med2) * confThresholdScale

        // Every term in the IRLS is multiplied by these weights, so pixels that fail
        // the confidence test contribute exactly nothing when their weight is zero.
        // Zeroing therefore gives the same answer as gathering the valid subset — and
        // keeps the whole thing on the GPU instead of round-tripping ~3M floats to
        // build a compacted copy.
        let total = N * H * W
        let valid = (conf1Slice .> confThreshold) & (conf2Slice .> confThreshold)
        let product = MLX.maximum(conf1Slice, 0) * MLX.maximum(conf2Slice, 0)
        let weights = MLX.which(valid, MLX.sqrt(product), MLXArray(Float(0)))
            .reshaped([total])

        let validCount = valid.asType(.int32).sum()
        eval(weights, validCount)
        let nValid = validCount.item(Int32.self)
        print(String(
            format: "[align] conf threshold = %.4f (med1=%.4f, med2=%.4f), matched %d point pairs",
            confThreshold, med1, med2, nValid
        ))
        if nValid == 0 {
            print("[align] WARNING: no matching point pairs — using identity")
            return .identity
        }

        // Rejected pixels must also be neutralized in the *coordinates*: an unprojected
        // point at a near-zero depth can be non-finite, and `0 * NaN` is still NaN, which
        // would poison the weighted means. Zeroing both weight and coordinate is what
        // makes this equivalent to dropping the pixel.
        let keep = valid.expandedDimensions(axis: -1)
        let src = MLX.which(keep, pm2Slice, MLXArray(Float(0))).reshaped([total, 3])
        let tgt = MLX.which(keep, pm1Slice, MLXArray(Float(0))).reshaped([total, 3])

        return Sim3Alignment.robustEstimate(
            src: src, target: tgt, initWeights: weights, config: irlsConfig
        )
    }

    /// `np.median` semantics: for an even count, the mean of the two middle values.
    /// MLX has no median op, so this sorts on the GPU and reads the middle back.
    static func median(_ a: MLXArray) -> Float {
        let count = a.size
        if count == 0 { return 0 }
        let sorted = MLX.sorted(a.reshaped([count]))
        let mid = count / 2
        let middle = count % 2 == 1
            ? sorted[mid]
            : (sorted[mid - 1] + sorted[mid]) / 2
        eval(middle)
        return middle.item(Float.self)
    }
}
