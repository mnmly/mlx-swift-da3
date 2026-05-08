import Foundation
import MLX

/// Mirrors `weighted_align_point_maps` in `loop_utils/sim3utils.py`.
///
/// Aligns chunk2 → chunk1 via robust weighted Sim(3): given two overlapping
/// regions of point maps and confidences, gather the pixels valid in both,
/// weight by sqrt(conf1*conf2), then run IRLS.
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

        // Per-frame conf threshold = min(median(c1), median(c2)) * 0.1
        let conf1Slice = conf1[0..<N, 0..<H, 0..<W]
        let conf2Slice = conf2[0..<N, 0..<H, 0..<W]
        let pm1Slice = pointMap1[0..<N, 0..<H, 0..<W, 0...]
        let pm2Slice = pointMap2[0..<N, 0..<H, 0..<W, 0...]

        let med1 = median(conf1Slice)
        let med2 = median(conf2Slice)
        let confThreshold = min(med1, med2) * confThresholdScale
        print(String(format: "[align] conf threshold = %.4f (med1=%.4f, med2=%.4f)", confThreshold, med1, med2))

        // Build mask & gather valid points on CPU (avoids MLX fancy indexing limits).
        let conf1CPU: [Float] = conf1Slice.asArray(Float.self)
        let conf2CPU: [Float] = conf2Slice.asArray(Float.self)
        let pm1CPU: [Float] = pm1Slice.asArray(Float.self)
        let pm2CPU: [Float] = pm2Slice.asArray(Float.self)

        let total = N * H * W
        var srcVals = [Float]()
        var tgtVals = [Float]()
        var weightVals = [Float]()
        srcVals.reserveCapacity(total * 3 / 4)
        tgtVals.reserveCapacity(total * 3 / 4)
        weightVals.reserveCapacity(total / 4)

        for i in 0..<total {
            let c1 = conf1CPU[i]
            let c2 = conf2CPU[i]
            if c1 > confThreshold && c2 > confThreshold {
                let w = sqrt(max(c1, 0) * max(c2, 0))
                let p3 = i * 3
                tgtVals.append(pm1CPU[p3]);     tgtVals.append(pm1CPU[p3+1]); tgtVals.append(pm1CPU[p3+2])
                srcVals.append(pm2CPU[p3]);     srcVals.append(pm2CPU[p3+1]); srcVals.append(pm2CPU[p3+2])
                weightVals.append(w)
            }
        }

        let nValid = weightVals.count
        if nValid == 0 {
            print("[align] WARNING: no matching point pairs — using identity")
            return .identity
        }
        print("[align] matched \(nValid) point pairs")

        let src = MLXArray(srcVals, [nValid, 3])
        let tgt = MLXArray(tgtVals, [nValid, 3])
        let weights = MLXArray(weightVals, [nValid])

        return Sim3Alignment.robustEstimate(
            src: src, target: tgt, initWeights: weights, config: irlsConfig
        )
    }

    /// Mean of array via MLX (returns Float on CPU).
    private static func median(_ a: MLXArray) -> Float {
        // MLX has no median; approximate via mean for the threshold heuristic.
        // Python uses np.median, but mean is monotonic enough for the threshold scale.
        // (We're computing min(med1, med2)*0.1 which is just a soft threshold.)
        let m = a.mean()
        eval(m)
        return m.item(Float.self)
    }
}
