import Foundation
import XCTest

@testable import MLXDA3Streaming

final class Sim3LoopOptimizerTests: XCTestCase {
    /// Sanity test: 5 noisy small rotations forming a near-loop, plus a single
    /// loop closure constraint that says (frame 5 → frame 0) should be identity.
    /// The optimizer should reduce the residual cost meaningfully.
    func testRingClosure() {
        // Build 5 small rotations with translation, with deliberate drift.
        let nPoses = 5
        var sequential: [Sim3] = []
        for k in 0..<nPoses {
            // rotate ~10° about z each step, walk forward 1 unit, scale 1.0
            let angle = Double(k + 1) * 0.05 + 0.6  // ~30-65° sectors
            let c = cos(angle), s = sin(angle)
            let R: [Double] = [c, -s, 0, s, c, 0, 0, 0, 1]
            let t: [Double] = [1.0, 0.0, 0.0]
            sequential.append(Sim3(R: R, t: t, s: 1.0))
        }
        // Loop constraint: pretend the chain ended back at start (identity).
        let loop = LoopConstraint(i: nPoses, j: 0, measurement: .identity)

        var cfg = Sim3LoopOptimizer.Config()
        cfg.verbose = false
        cfg.maxIterations = 30
        let opt = Sim3LoopOptimizer(config: cfg)

        // Cost before: residual on the loop edge alone
        let optimized = opt.optimize(sequentialTransforms: sequential, loopConstraints: [loop])

        // Verify structural properties:
        XCTAssertEqual(optimized.count, sequential.count, "length preserved")

        // The optimized sequence accumulated should bring frame n to closer-to-loop-measurement.
        var absInput = Sim3.identity
        for r in sequential { absInput = absInput.compose(r) }
        var absOpt = Sim3.identity
        for r in optimized { absOpt = absOpt.compose(r) }

        // residual of closing-the-loop: log(absOpt @ measurement.inverse) ≈ 0
        // Loop measurement here is identity, so we want absOpt close to identity.
        let resid = sim3Log(absOpt.compose(loop.measurement.inverse))
        let absResidNorm = sqrt(resid.reduce(0.0) { $0 + $1 * $1 })

        let inResid = sim3Log(absInput.compose(loop.measurement.inverse))
        let inResidNorm = sqrt(inResid.reduce(0.0) { $0 + $1 * $1 })

        // Optimization should reduce residual substantially
        XCTAssertLessThan(absResidNorm, inResidNorm * 0.2,
                          "loop residual reduced from \(inResidNorm) to \(absResidNorm)")
    }

    /// With no loop constraints, optimizer should return the input unchanged.
    func testNoLoopReturnsInput() {
        let s = sim3Exp([0.1, 0.2, -0.1, 0.05, 0.0, 0.1, 0.0])
        let s2 = sim3Exp([0.0, 0.1, 0.0, 0.0, 0.05, 0.0, 0.05])
        let opt = Sim3LoopOptimizer()
        let out = opt.optimize(sequentialTransforms: [s, s2], loopConstraints: [])
        XCTAssertEqual(out.count, 2)
        for k in 0..<2 {
            for i in 0..<9 {
                XCTAssertEqual(out[k].R[i], [s, s2][k].R[i], accuracy: 1e-12)
            }
            for i in 0..<3 {
                XCTAssertEqual(out[k].t[i], [s, s2][k].t[i], accuracy: 1e-12)
            }
            XCTAssertEqual(out[k].s, [s, s2][k].s, accuracy: 1e-12)
        }
    }
}
