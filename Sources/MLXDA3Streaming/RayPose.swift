import Foundation
import MLX
import MLXDA3

/// Geometric ray → camera pose estimation, ported from
/// `python/Depth-Anything-3/src/depth_anything_3/utils/ray_utils.py`.
///
/// Entry point: `RayPose.cameraInfoFromRays(ray:rayConf:height:width:)` which
/// turns the model's per-patch ray predictions into per-view extrinsics (w2c)
/// and intrinsics (pixels).
///
/// All routines accept the model's `ray` shape `[S, Hp, Wp, 6]` (origin xyz, dir xyz)
/// and `rayConf` shape `[S, Hp, Wp]`. We squeeze the python "B" dimension since
/// the streaming pipeline always uses B=1.
public enum RayPose {

    // MARK: - Tunables (mirror python `get_params_for_ransac`)

    public static let nIter: Int = 100
    public static let numSampleForRansac: Int = 8
    public static let sampleRatio: Float = 0.3
    public static let reprojThreshold: Float = 0.2
    public static let zThreshold: Float = 1e-4
    public static let maxInlierNum: Int = 8000

    /// Seed for RANSAC sampling. Sampling is seeded per view, so the same frames
    /// produce the same cameras — and therefore the same point cloud — on every run.
    /// Change it to explore a different sampling draw.
    public nonisolated(unsafe) static var randomSeed: UInt64 = 0x5EED_DA3_5EED

    // MARK: - Public entry point

    /// Compute per-view extrinsics (w2c, `[S, 3, 4]`) and intrinsics (pixels, `[S, 3, 3]`).
    public static func cameraInfoFromRays(
        ray: MLXArray,         // [S, Hp, Wp, 6]
        rayConf: MLXArray,     // [S, Hp, Wp]
        imageHeight: Int,
        imageWidth: Int
    ) -> (extrinsicsW2C: MLXArray, intrinsics: MLXArray) {

        let S = ray.dim(0)
        let Hp = ray.dim(1)
        let Wp = ray.dim(2)

        // Build canonical camera-plane unprojection (B=1, S, Hp, Wp, 3).
        // python `unproject_depth(ixt_normalized=True, depth=1)` with K = [[1,0,1],[0,1,1],[0,0,1]]
        // reduces to (xLin - 1, yLin - 1, 1) per pixel where xLin=linspace(dx, 2-dx).
        let identityCamPlane = identityCameraPlane(Hp: Hp, Wp: Wp, S: S)  // [S, Hp, Wp, 3]

        // Flatten for the homography RANSAC: each view gets (N, 3) source / (N, 3) target.
        let N = Hp * Wp
        let camrayFlat = ray.reshaped([S, N, 6]).asType(.float32)
        let rayDir = camrayFlat[0..., 0..., 0..<3]  // [S, N, 3]
        let identityFlat = identityCamPlane.reshaped([S, N, 3]).asType(.float32)
        let confFlat = rayConf.reshaped([S, N]).asType(.float32)

        // Compute R, focal, principal point per view.
        let (R, focalRaw, ppRaw) = optimalRotationIntrinsicsBatch(
            raysOrigin: identityFlat,
            raysTarget: rayDir,
            weights: confFlat
        )

        // T = weighted mean of camray[..., 3:6] (ray origins).
        let rayOrigin = camrayFlat[0..., 0..., 3..<6]  // [S, N, 3]
        let confExp = confFlat.expandedDimensions(axis: -1)  // [S, N, 1]
        let numerator = (rayOrigin * confExp).sum(axis: 1)   // [S, 3]
        let denom = confFlat.sum(axis: 1, keepDims: true)    // [S, 1]
        let T = numerator / (denom + 1e-12)                  // [S, 3]

        // Per-python: returned focal = 1/raw_f, pp = raw_pp + 1.
        let focalAdj = 1.0 / focalRaw                        // [S, 2]
        let ppAdj = ppRaw + 1.0                              // [S, 2]

        // Build c2w = [R | T], (S, 4, 4).
        let c2w = makeC2W(R: R, T: T)                        // [S, 4, 4]

        // w2c = inverse of affine c2w.
        let w2c = affineInverse(c2w)                         // [S, 4, 4]
        let extrinsicsW2C = w2c[0..., 0..<3, 0...]           // [S, 3, 4]

        // Build pixel-space intrinsics K (S, 3, 3).
        let intrinsics = pixelIntrinsics(
            focal: focalAdj, principalPoint: ppAdj,
            imageHeight: imageHeight, imageWidth: imageWidth
        )

        eval(extrinsicsW2C, intrinsics)
        return (extrinsicsW2C, intrinsics)
    }

    // MARK: - Identity camera plane (replaces unproject_depth ixt_normalized)

    /// Returns (S, Hp, Wp, 3) where each pixel is (x_lin - 1, y_lin - 1, 1) with
    /// `x_lin = linspace(dx, 2-dx, Wp)`, dx = 1/Wp. This is the python `I_cam_plane_unproj`
    /// for ixt_normalized=True with cx=cy=1, fx=fy=1, depth=1.
    static func identityCameraPlane(Hp: Int, Wp: Int, S: Int) -> MLXArray {
        let dx = 1.0 / Float(Wp)
        let dy = 1.0 / Float(Hp)
        let xLin = MLXArray.linspace(Float(dx), Float(2.0 - dx), count: Wp).asType(.float32) - 1.0
        let yLin = MLXArray.linspace(Float(dy), Float(2.0 - dy), count: Hp).asType(.float32) - 1.0
        let xGrid = MLX.broadcast(xLin.reshaped([1, Wp]), to: [Hp, Wp])
        let yGrid = MLX.broadcast(yLin.reshaped([Hp, 1]), to: [Hp, Wp])
        let ones = MLXArray.ones([Hp, Wp], dtype: .float32)
        let pix = MLX.stacked([xGrid, yGrid, ones], axis: -1)            // (Hp, Wp, 3)
        return MLX.broadcast(pix.expandedDimensions(axis: 0), to: [S, Hp, Wp, 3])
    }

    // MARK: - compute_optimal_rotation_intrinsics_batch

    /// Returns (R [S, 3, 3], focal [S, 2], principalPoint [S, 2]).
    static func optimalRotationIntrinsicsBatch(
        raysOrigin: MLXArray,    // (S, N, 3)
        raysTarget: MLXArray,    // (S, N, 3)
        weights: MLXArray        // (S, N)
    ) -> (R: MLXArray, focal: MLXArray, principalPoint: MLXArray) {
        let S = raysOrigin.dim(0)

        // Per python: divide x,y by z where |z| > z_threshold; mark bad pts with weight=0.
        let zo = raysOrigin[0..., 0..., 2]    // (S, N)
        let zt = raysTarget[0..., 0..., 2]
        let zMask = (MLX.abs(zo) .> Float(zThreshold)) & (MLX.abs(zt) .> Float(zThreshold))  // (S, N) bool

        let zoSafe = MLX.which(zMask, zo, MLXArray(Float(1.0)))
        let ztSafe = MLX.which(zMask, zt, MLXArray(Float(1.0)))

        // (S, N, 2)
        let originXY = raysOrigin[0..., 0..., 0..<2] / zoSafe.expandedDimensions(axis: -1)
        let targetXY = raysTarget[0..., 0..., 0..<2] / ztSafe.expandedDimensions(axis: -1)

        // Zero out weights where z-mask fails.
        let weightsMasked = MLX.which(zMask, weights, MLXArray(0.0, dtype: .float32))

        // RANSAC homography per-view, then QL-decompose.
        var Rs = [MLXArray]()
        var fs = [MLXArray]()
        var pps = [MLXArray]()
        Rs.reserveCapacity(S); fs.reserveCapacity(S); pps.reserveCapacity(S)

        for s in 0..<S {
            let src = originXY[s]                  // (N, 2)
            let dst = targetXY[s]                  // (N, 2)
            let w = weightsMasked[s]               // (N,)
            var H = ransacHomography(
                srcPts: src, dstPts: dst, weights: w,
                seed: randomSeed &+ UInt64(s)
            )

            // Sign flip if det < 0 (per python).
            let detH = det3x3(H)
            let need = (detH .< Float(0)).asType(.float32)
            H = MLX.which(need .> Float(0.5), -H, H)

            let (R, L) = qlDecomposition(H)
            // L = L / L[2][2]
            let L22 = L[2, 2]
            let Ln = L / L22

            let f = MLX.stacked([Ln[0, 0], Ln[1, 1]], axis: 0)
            let pp = MLX.stacked([Ln[2, 0], Ln[2, 1]], axis: 0)
            Rs.append(R); fs.append(f); pps.append(pp)
        }

        let R = MLX.stacked(Rs, axis: 0)         // (S, 3, 3)
        let f = MLX.stacked(fs, axis: 0)         // (S, 2)
        let pp = MLX.stacked(pps, axis: 0)       // (S, 2)
        return (R, f, pp)
    }

    // MARK: - QL decomposition via QR with permutation

    /// Returns (Q, L) such that A = Q @ L (left-triangular L).
    static func qlDecomposition(_ A: MLXArray) -> (MLXArray, MLXArray) {
        // P = anti-diagonal permutation matrix
        let P = MLXArray(
            [Float(0), 0, 1,
             0, 1, 0,
             1, 0, 0],
            [3, 3]
        )
        let aTilde = A.matmul(P)
        let (qTilde, rTilde) = DA3Profiler.measure("  raypose.qr", sync: { MLX.eval($0.0, $0.1) }) {
            MLX.qr(aTilde, stream: .cpu)
        }
        var Q = qTilde.matmul(P)
        var L = P.matmul(rTilde).matmul(P)

        // Normalize signs from diag(L)
        let d0 = L[0, 0]; let d1 = L[1, 1]; let d2 = L[2, 2]
        let sign0 = MLX.sign(d0); let sign1 = MLX.sign(d1); let sign2 = MLX.sign(d2)

        // Q[:, i] *= sign(d[i])
        let qCols = MLX.stacked([Q[0..., 0] * sign0, Q[0..., 1] * sign1, Q[0..., 2] * sign2], axis: -1)
        Q = qCols
        // L[i, :] *= sign(d[i])
        let lRows = MLX.stacked([L[0] * sign0, L[1] * sign1, L[2] * sign2], axis: 0)
        L = lRows
        return (Q, L)
    }

    // MARK: - RANSAC weighted homography (single view, N points)

    /// Returns 3×3 H. Mirrors `ransac_find_homography_weighted_fast` but per-view (no batch).
    static func ransacHomography(
        srcPts: MLXArray,    // (N, 2)
        dstPts: MLXArray,    // (N, 2)
        weights: MLXArray,   // (N,)
        seed: UInt64
    ) -> MLXArray {
        let N = srcPts.dim(0)
        precondition(N >= 4, "Need at least 4 points")

        let nSample = max(numSampleForRansac, Int(Float(N) * sampleRatio))
        let weightsCPU: [Float] = DA3Profiler.measure("  raypose.pull.weights") {
            weights.asArray(Float.self)
        }

        // Top-K candidate indices by weight (descending).
        let candidate: [Int] = topKDescending(weightsCPU, k: min(nSample, N))

        // Generate `nIter` random samples of size `numSampleForRansac` from candidate.
        var rng = SplitMix64(seed: seed)
        var sampleIdx = [[Int]]()
        sampleIdx.reserveCapacity(nIter)
        for _ in 0..<nIter {
            sampleIdx.append(randomChoice(candidate, k: numSampleForRansac, rng: &rng))
        }

        // Build batch of Hs: gather sampled src/dst/w into (nIter, k, 2) etc.
        let (srcCPU, dstCPU): ([Float], [Float]) = DA3Profiler.measure("  raypose.pull.pts") {
            (srcPts.asArray(Float.self), dstPts.asArray(Float.self))
        }
        let k = numSampleForRansac
        var srcBatch = [Float](repeating: 0, count: nIter * k * 2)
        var dstBatch = [Float](repeating: 0, count: nIter * k * 2)
        var wBatch = [Float](repeating: 0, count: nIter * k)
        for it in 0..<nIter {
            for j in 0..<k {
                let idx = sampleIdx[it][j]
                srcBatch[(it * k + j) * 2 + 0] = srcCPU[idx * 2 + 0]
                srcBatch[(it * k + j) * 2 + 1] = srcCPU[idx * 2 + 1]
                dstBatch[(it * k + j) * 2 + 0] = dstCPU[idx * 2 + 0]
                dstBatch[(it * k + j) * 2 + 1] = dstCPU[idx * 2 + 1]
                wBatch[it * k + j] = weightsCPU[idx]
            }
        }
        let src = MLXArray(srcBatch, [nIter, k, 2])
        let dst = MLXArray(dstBatch, [nIter, k, 2])
        let w = MLXArray(wBatch, [nIter, k])

        // Batched weighted least-squares homography: (nIter, 3, 3)
        let HBatch = weightedHomographyBatch(srcPts: src, dstPts: dst, weights: w)

        // Score each H by inlier weight against the full N points.
        let srcHomo = MLX.concatenated([srcPts, MLXArray.ones([N, 1], dtype: .float32)], axis: 1)  // (N, 3)
        // proj = src @ H^T  → (nIter, N, 3). Need broadcast over nIter.
        let HBatchT = HBatch.transposed(axes: [0, 2, 1])                        // (nIter, 3, 3)
        let srcHomoExp = MLX.broadcast(srcHomo.expandedDimensions(axis: 0), to: [nIter, N, 3])
        let proj = srcHomoExp.matmul(HBatchT)                                    // (nIter, N, 3)
        let projXY = proj[0..., 0..., 0..<2] / (proj[0..., 0..., 2..<3] + Float(1e-12))
        let dstExp = MLX.broadcast(dstPts.expandedDimensions(axis: 0), to: [nIter, N, 2])
        let err = MLX.sqrt(((projXY - dstExp) ** 2).sum(axis: 2))                // (nIter, N)
        let inlier = (err .< Float(reprojThreshold)).asType(.float32)            // (nIter, N)
        let weightsExp = MLX.broadcast(weights.expandedDimensions(axis: 0), to: [nIter, N])
        let score = (inlier * weightsExp).sum(axis: 1)                           // (nIter,)

        let scoreCPU: [Float] = DA3Profiler.measure("  raypose.score", sync: { _ in }) {
            eval(score, inlier)
            return score.asArray(Float.self)
        }
        var bestI = 0
        var bestS: Float = -1
        for i in 0..<nIter {
            if scoreCPU[i] > bestS { bestS = scoreCPU[i]; bestI = i }
        }

        // Refit using inliers of best iteration. Cap inlier count at maxInlierNum.
        let inlierMaskCPU: [Float] = DA3Profiler.measure("  raypose.pull.inliers") {
            inlier[bestI].asArray(Float.self)
        }
        var inlierIdx = [Int]()
        inlierIdx.reserveCapacity(N)
        for i in 0..<N where inlierMaskCPU[i] > 0.5 { inlierIdx.append(i) }
        if inlierIdx.count < 4 {
            // Degenerate; fall back to identity-ish — return current best H.
            return HBatch[bestI]
        }
        if inlierIdx.count > maxInlierNum {
            // Random downsample
            inlierIdx.shuffle(using: &rng)
            inlierIdx = Array(inlierIdx.prefix(maxInlierNum))
        }
        var inlierSrc = [Float](repeating: 0, count: inlierIdx.count * 2)
        var inlierDst = [Float](repeating: 0, count: inlierIdx.count * 2)
        var inlierW = [Float](repeating: 0, count: inlierIdx.count)
        for (j, idx) in inlierIdx.enumerated() {
            inlierSrc[j * 2 + 0] = srcCPU[idx * 2 + 0]
            inlierSrc[j * 2 + 1] = srcCPU[idx * 2 + 1]
            inlierDst[j * 2 + 0] = dstCPU[idx * 2 + 0]
            inlierDst[j * 2 + 1] = dstCPU[idx * 2 + 1]
            inlierW[j] = weightsCPU[idx]
        }
        let srcIn = MLXArray(inlierSrc, [inlierIdx.count, 2])
        let dstIn = MLXArray(inlierDst, [inlierIdx.count, 2])
        let wIn = MLXArray(inlierW, [inlierIdx.count])
        return weightedHomographySingle(srcPts: srcIn, dstPts: dstIn, weights: wIn)
    }

    // MARK: - Weighted least-squares homography (single & batched)

    /// Single: (N, 2), (N, 2), (N,) → (3, 3).
    static func weightedHomographySingle(
        srcPts: MLXArray, dstPts: MLXArray, weights: MLXArray
    ) -> MLXArray {
        let N = srcPts.dim(0)
        let w = MLX.sqrt(weights).expandedDimensions(axis: 1)  // (N, 1)
        let x = srcPts[0..., 0..<1]                            // (N, 1)
        let y = srcPts[0..., 1..<2]
        let u = dstPts[0..., 0..<1]
        let v = dstPts[0..., 1..<2]
        let zeros = MLXArray.zeros([N, 1], dtype: .float32)

        let A1 = MLX.concatenated([
            -x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w
        ], axis: 1)
        let A2 = MLX.concatenated([
            zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w
        ], axis: 1)
        let A = MLX.concatenated([A1, A2], axis: 0)            // (2N, 9)

        let Vh = DA3Profiler.measure("  raypose.svd.single", sync: { MLX.eval($0) }) {
            rightSingularVectors(A)
        }
        var H = Vh[8].reshaped([3, 3])
        H = H / H[2, 2]
        return H
    }

    /// Batch: (B, N, 2), (B, N, 2), (B, N) → (B, 3, 3).
    static func weightedHomographyBatch(
        srcPts: MLXArray, dstPts: MLXArray, weights: MLXArray
    ) -> MLXArray {
        let B = srcPts.dim(0)
        let N = srcPts.dim(1)
        let w = MLX.sqrt(weights).expandedDimensions(axis: 2)  // (B, N, 1)
        let x = srcPts[0..., 0..., 0..<1]                      // (B, N, 1)
        let y = srcPts[0..., 0..., 1..<2]
        let u = dstPts[0..., 0..., 0..<1]
        let v = dstPts[0..., 0..., 1..<2]
        let zeros = MLXArray.zeros([B, N, 1], dtype: .float32)

        let A1 = MLX.concatenated([
            -x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w
        ], axis: 2)
        let A2 = MLX.concatenated([
            zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w
        ], axis: 2)
        let A = MLX.concatenated([A1, A2], axis: 1)            // (B, 2N, 9)

        let Vh = DA3Profiler.measure("  raypose.svd.batch", sync: { MLX.eval($0) }) {
            rightSingularVectors(A)
        }
        var H = Vh[0..., 8, 0...].reshaped([B, 3, 3])
        let H22 = H[0..., 2..<3, 2..<3]                        // (B, 1, 1)
        H = H / H22
        return H
    }

    // MARK: - C2W / W2C builders

    /// Build (S, 4, 4) c2w from (S, 3, 3) R and (S, 3) T.
    static func makeC2W(R: MLXArray, T: MLXArray) -> MLXArray {
        let S = R.dim(0)
        let Tcol = T.expandedDimensions(axis: -1)            // (S, 3, 1)
        let top = MLX.concatenated([R, Tcol], axis: -1)      // (S, 3, 4)
        var bottomData = [Float](repeating: 0, count: 4)
        bottomData[3] = 1
        let bottom = MLX.broadcast(
            MLXArray(bottomData, [1, 4]).expandedDimensions(axis: 0),
            to: [S, 1, 4]
        )
        return MLX.concatenated([top, bottom], axis: 1)      // (S, 4, 4)
    }

    /// Affine inverse of a stack of 4x4 SE(3)-ish matrices: (S, 4, 4) → (S, 4, 4).
    static func affineInverse(_ M: MLXArray) -> MLXArray {
        let R = M[0..., 0..<3, 0..<3]                        // (S, 3, 3)
        let t = M[0..., 0..<3, 3..<4]                        // (S, 3, 1)
        let Rt = R.transposed(axes: [0, 2, 1])               // (S, 3, 3)
        let tInv = -(Rt.matmul(t))                           // (S, 3, 1)
        let top = MLX.concatenated([Rt, tInv], axis: -1)     // (S, 3, 4)
        let S = M.dim(0)
        var bottomData = [Float](repeating: 0, count: 4)
        bottomData[3] = 1
        let bottom = MLX.broadcast(
            MLXArray(bottomData, [1, 4]).expandedDimensions(axis: 0),
            to: [S, 1, 4]
        )
        return MLX.concatenated([top, bottom], axis: 1)
    }

    /// Build pixel-space intrinsics from focal and principalPoint per python convention:
    ///   fx = focal[0] / 2 * width, fy = focal[1] / 2 * height
    ///   cx = pp[0] * width * 0.5, cy = pp[1] * height * 0.5
    static func pixelIntrinsics(
        focal: MLXArray, principalPoint: MLXArray, imageHeight: Int, imageWidth: Int
    ) -> MLXArray {
        let S = focal.dim(0)
        let W = Float(imageWidth)
        let H = Float(imageHeight)

        let fx = focal[0..., 0] / 2.0 * W   // (S,)
        let fy = focal[0..., 1] / 2.0 * H
        let cx = principalPoint[0..., 0] * W * 0.5
        let cy = principalPoint[0..., 1] * H * 0.5

        // Build (S, 3, 3) intrinsics by composing rows.
        let zeros = MLXArray.zeros([S], dtype: .float32)
        let ones = MLXArray.ones([S], dtype: .float32)
        let row0 = MLX.stacked([fx, zeros, cx], axis: -1)       // (S, 3)
        let row1 = MLX.stacked([zeros, fy, cy], axis: -1)
        let row2 = MLX.stacked([zeros, zeros, ones], axis: -1)
        return MLX.stacked([row0, row1, row2], axis: 1)         // (S, 3, 3)
    }

    // MARK: - Helpers

    /// Right singular vectors (`Vᵀ`, shape `[..., n, n]`) of a tall matrix `[..., m, n]`.
    ///
    /// The DLT solve only needs `Vᵀ`, never `U`. LAPACK's `gesdd` — what MLX calls —
    /// materializes the full `m × m` `U`, which for the refit system (m ≈ 2·720, n = 9)
    /// costs two orders of magnitude more than the answer is worth.
    ///
    /// `A = QR` with orthonormal `Q` gives `AᵀA = RᵀR`, so `A` and the small `n × n` `R`
    /// have identical right singular vectors. MLX's `qr` is the reduced form (`Q` is
    /// `m × min(m, n)`), so this is exact, not an approximation — and unlike forming
    /// `AᵀA` directly it does not square the condition number.
    static func rightSingularVectors(_ A: MLXArray) -> MLXArray {
        let (_, R) = MLX.qr(A, stream: .cpu)
        return MLX.svd(R, stream: .cpu).2
    }

    /// Determinant of a 3×3 MLXArray.
    private static func det3x3(_ A: MLXArray) -> MLXArray {
        let a = A[0, 0], b = A[0, 1], c = A[0, 2]
        let d = A[1, 0], e = A[1, 1], f = A[1, 2]
        let g = A[2, 0], h = A[2, 1], i = A[2, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// Reproducible RNG (Vigna's SplitMix64) so RANSAC draws are stable across runs.
    struct SplitMix64: RandomNumberGenerator {
        private var state: UInt64
        init(seed: UInt64) { self.state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private static func topKDescending(_ values: [Float], k: Int) -> [Int] {
        let indexed = values.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        return sorted.prefix(k).map { $0.0 }
    }

    private static func randomChoice(
        _ pool: [Int], k: Int, rng: inout SplitMix64
    ) -> [Int] {
        precondition(pool.count >= k)
        var copy = pool
        for i in 0..<k {
            let j = i + Int.random(in: 0..<(copy.count - i), using: &rng)
            copy.swapAt(i, j)
        }
        return Array(copy.prefix(k))
    }
}
