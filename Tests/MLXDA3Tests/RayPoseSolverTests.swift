import Foundation
import MLX
import XCTest

@testable import MLXDA3Streaming

/// The DLT solve takes the right singular vectors of a tall `[m, 9]` matrix. We get
/// them from `qr` + a 9×9 `svd` rather than a full SVD of the tall matrix, because
/// LAPACK's `gesdd` materializes the whole `m × m` `U` we never look at. These tests
/// pin that the shortcut agrees with the direct SVD.
final class RayPoseSolverTests: XCTestCase {

    private func randomTall(rows: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        return MLXRandom.normal([rows, 9]).asType(.float32)
    }

    /// Compare null vectors up to sign (SVD sign convention is arbitrary).
    private func assertSameNullVector(
        _ a: MLXArray, _ b: MLXArray, accuracy: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let av: [Float] = a.asArray(Float.self)
        let bv: [Float] = b.asArray(Float.self)
        let dot = zip(av, bv).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let sign: Float = dot < 0 ? -1 : 1
        for i in 0 ..< av.count {
            XCTAssertEqual(av[i], sign * bv[i], accuracy: accuracy, "component \(i)", file: file, line: line)
        }
    }

    func testMatchesFullSVDOnTallMatrix() {
        for (rows, seed) in [(16, 1), (144, 2), (1440, 3)] {
            let A = randomTall(rows: rows, seed: UInt64(seed))
            let viaQR = RayPose.rightSingularVectors(A)[8]
            let viaSVD = MLX.svd(A, stream: .cpu).2[8]
            eval(viaQR, viaSVD)
            assertSameNullVector(viaQR, viaSVD, accuracy: 1e-4)
        }
    }

    func testMatchesFullSVDOnBatch() {
        MLXRandom.seed(7)
        let A = MLXRandom.normal([32, 16, 9]).asType(.float32)
        let viaQR = RayPose.rightSingularVectors(A)
        let viaSVD = MLX.svd(A, stream: .cpu).2
        eval(viaQR, viaSVD)
        for b in 0 ..< 32 {
            assertSameNullVector(viaQR[b, 8], viaSVD[b, 8], accuracy: 1e-4)
        }
    }

    /// The vector we take must actually be in the null space of a rank-deficient system,
    /// which is the situation the DLT solve is always in.
    func testSolvesRankDeficientSystem() {
        MLXRandom.seed(11)
        let nullVector = MLXRandom.normal([9]).asType(.float32)
        let unit = nullVector / MLX.sqrt((nullVector * nullVector).sum())
        // Rows orthogonal to `unit`: A = B - (B·unit)unitᵀ, so A @ unit == 0.
        let B = MLXRandom.normal([200, 9]).asType(.float32)
        let A = B - matmul(matmul(B, unit.reshaped([9, 1])), unit.reshaped([1, 9]))

        let recovered = RayPose.rightSingularVectors(A)[8]
        eval(recovered)
        assertSameNullVector(recovered, unit, accuracy: 1e-3)

        // The shortcut must solve the system at least as well as the full SVD does;
        // the absolute floor here is float32 noise in `A` itself, not the solver.
        let residual = abs(matmul(A, recovered.reshaped([9, 1]))).max()
        let svdVector = MLX.svd(A, stream: .cpu).2[8]
        let svdResidual = abs(matmul(A, svdVector.reshaped([9, 1]))).max()
        eval(residual, svdResidual)
        XCTAssertLessThanOrEqual(
            residual.item(Float.self), svdResidual.item(Float.self) * 1.5 + 1e-6,
            "QR shortcut should not be less accurate than the full SVD"
        )
    }

    func testRansacSamplingIsDeterministic() {
        MLXRandom.seed(3)
        let src = MLXRandom.normal([64, 2]).asType(.float32)
        let dst = MLXRandom.normal([64, 2]).asType(.float32)
        let weights = MLXRandom.uniform(low: 0.1, high: 1.0, [64]).asType(.float32)
        eval(src, dst, weights)

        let first = RayPose.ransacHomography(srcPts: src, dstPts: dst, weights: weights, seed: 42)
        let second = RayPose.ransacHomography(srcPts: src, dstPts: dst, weights: weights, seed: 42)
        eval(first, second)
        XCTAssertEqual(abs(first - second).max().item(Float.self), 0,
                       "same seed must give byte-identical homographies")
    }
}
