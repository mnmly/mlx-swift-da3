import Foundation
import MLX

/// Weighted Sim(3) alignment via IRLS, ported from
/// `python/.../loop_utils/alignment_torch.py`. All linalg here runs on the CPU
/// stream because MLX-Swift's GPU backend lacks SVD.
public enum Sim3Alignment {

    public enum Method: String {
        case sim3
        case se3
    }

    public struct Config {
        public var method: Method = .sim3
        public var delta: Float = 0.1
        public var maxIters: Int = 5
        public var tol: Float = 1e-9
        public init() {}
    }

    /// Result: source @ s*R^T + t ≈ target. (Mirrors python convention.)
    public struct Sim3 {
        public var s: Float
        public var R: [Float]   // 3×3 row-major
        public var t: [Float]   // 3
        public init(s: Float, R: [Float], t: [Float]) {
            self.s = s; self.R = R; self.t = t
        }
        public static let identity = Sim3(
            s: 1.0,
            R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            t: [0, 0, 0]
        )
    }

    // MARK: - Weighted estimate (one iteration, single-shot)

    /// Single weighted Sim(3) / SE(3) estimate. Returns (s, R, t) with
    ///   transformed = s * (R @ src) + t  (sim3)
    ///   transformed =     (R @ src) + t  (se3)
    public static func weightedEstimate(
        src: MLXArray, target: MLXArray, weights: MLXArray, method: Method
    ) -> Sim3 {
        // src, target: (N, 3), weights: (N,). All float32.
        let totalW = weights.sum()
        eval(totalW)
        let totalCPU: Float = totalW.item(Float.self)
        if totalCPU < 1e-6 {
            return .identity
        }
        let nw = (weights / totalW).asType(.float32)             // (N,)
        let nwCol = nw.expandedDimensions(axis: -1)              // (N, 1)
        let muSrc = (nwCol * src).sum(axis: 0)                   // (3,)
        let muTgt = (nwCol * target).sum(axis: 0)                // (3,)

        let srcCenter = src - muSrc
        let tgtCenter = target - muTgt

        let s: Float
        let weightedSrc: MLXArray
        if method == .sim3 {
            let scaleSrc = MLX.sqrt((nw * (srcCenter * srcCenter).sum(axis: 1)).sum())
            let scaleTgt = MLX.sqrt((nw * (tgtCenter * tgtCenter).sum(axis: 1)).sum())
            let sArr = scaleTgt / scaleSrc
            eval(sArr)
            s = sArr.item(Float.self)
            weightedSrc = (Float(s) * srcCenter) * MLX.sqrt(nw).expandedDimensions(axis: -1)
        } else {
            s = 1.0
            weightedSrc = srcCenter * MLX.sqrt(nw).expandedDimensions(axis: -1)
        }
        let weightedTgt = tgtCenter * MLX.sqrt(nw).expandedDimensions(axis: -1)

        let H = weightedSrc.transposed().matmul(weightedTgt)     // (3, 3) on default stream
        // SVD on CPU stream
        let (U, _, Vh) = MLX.svd(H, stream: .cpu)
        // R = Vh^T @ U^T
        var Rmat = Vh.transposed().matmul(U.transposed())        // (3, 3)
        eval(Rmat)
        // Reflection fix
        let detR = det3x3(Rmat)
        eval(detR)
        if detR.item(Float.self) < 0 {
            // Flip last row of Vh and recompute
            let signs = MLXArray([Float(1), 1, -1])              // (3,)
            let VhFixed = Vh * signs.expandedDimensions(axis: -1)
            Rmat = VhFixed.transposed().matmul(U.transposed())
            eval(Rmat)
        }
        let muSrcCPU: [Float] = muSrc.asArray(Float.self)
        let muTgtCPU: [Float] = muTgt.asArray(Float.self)
        let Rcpu: [Float] = Rmat.asArray(Float.self)

        // t = mu_tgt - s * R @ mu_src   (sim3 path)
        // t = mu_tgt -     R @ mu_src   (se3)
        let scaleForT: Float = method == .sim3 ? s : 1.0
        var t = [Float](repeating: 0, count: 3)
        for i in 0..<3 {
            let rmu = Rcpu[i*3+0] * muSrcCPU[0] + Rcpu[i*3+1] * muSrcCPU[1] + Rcpu[i*3+2] * muSrcCPU[2]
            t[i] = muTgtCPU[i] - scaleForT * rmu
        }
        return Sim3(s: s, R: Rcpu, t: t)
    }

    // MARK: - Robust IRLS

    /// IRLS (iteratively re-weighted least-squares) Sim(3) with Huber re-weighting.
    public static func robustEstimate(
        src: MLXArray,           // (N, 3) float32
        target: MLXArray,        // (N, 3) float32
        initWeights: MLXArray,   // (N,) float32
        config: Config = Config()
    ) -> Sim3 {
        var transform = weightedEstimate(src: src, target: target, weights: initWeights, method: config.method)
        var prevError: Float = .infinity

        for iter in 0..<config.maxIters {
            // Apply current transform: transformed = s*(src @ R^T) + t
            let RArr = MLXArray(transform.R, [3, 3])
            let tArr = MLXArray(transform.t, [3])
            let transformed = Float(transform.s) * src.matmul(RArr.transposed()) + tArr  // (N, 3)
            let residuals = MLX.sqrt(((target - transformed) * (target - transformed)).sum(axis: 1))
            // Huber weights: w = 1 if r <= delta else delta/r
            let onesArr = MLXArray.ones(residuals.shape, dtype: .float32)
            let huberFar = Float(config.delta) / (residuals + Float(1e-12))
            let huber = MLX.which(residuals .< Float(config.delta), onesArr, huberFar)
            let combinedW = initWeights * huber
            let combinedSum = combinedW.sum() + 1e-12
            let normW = combinedW / combinedSum

            let newTransform = weightedEstimate(src: src, target: target, weights: normW, method: config.method)

            // Convergence checks
            let r2 = residuals * residuals
            let halfRsq = r2 * Float(0.5)
            let linearTerm = (residuals - Float(0.5 * Double(config.delta))) * Float(config.delta)
            let huberLossArr = MLX.which(residuals .< Float(config.delta), halfRsq, linearTerm)
            let curError = (huberLossArr * initWeights).sum()
            eval(curError)
            let curErrorCPU = curError.item(Float.self)

            let paramChange = abs(newTransform.s - transform.s)
                + dist(newTransform.t, transform.t)
            // approximate rotation angle change: arccos((trace(R_new R^T) - 1) / 2)
            let trace = traceProduct(newTransform.R, transform.R)
            let cosAngle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
            let rotAngle = acos(cosAngle)

            transform = newTransform

            let converged = (paramChange < config.tol && rotAngle < (Float.pi / 1800.0)) ||
                            (abs(prevError - curErrorCPU) < config.tol * prevError)
            prevError = curErrorCPU
            if converged {
                break
            }
            _ = iter
        }
        return transform
    }

    // MARK: - Helpers

    private static func det3x3(_ A: MLXArray) -> MLXArray {
        let a = A[0, 0], b = A[0, 1], c = A[0, 2]
        let d = A[1, 0], e = A[1, 1], f = A[1, 2]
        let g = A[2, 0], h = A[2, 1], i = A[2, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    private static func dist(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<a.count { s += (a[i] - b[i]) * (a[i] - b[i]) }
        return sqrt(s)
    }

    private static func traceProduct(_ A: [Float], _ B: [Float]) -> Float {
        // trace(A @ B^T) = sum_ij A[i,j] * B[i,j]
        var s: Float = 0
        for i in 0..<9 { s += A[i] * B[i] }
        return s
    }
}
