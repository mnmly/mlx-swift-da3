import Foundation
import MLX
import XCTest

@testable import MLXDA3Streaming

/// `ChunkAlignment` weights rejected pixels to zero instead of gathering the valid
/// subset, which keeps the work on the GPU. These tests pin that this is equivalent
/// to gathering — including the case that isn't obvious: a rejected pixel whose
/// unprojected coordinate is non-finite, where `0 * NaN` would otherwise poison the
/// weighted means.
final class ChunkAlignmentTests: XCTestCase {

    /// Reference implementation: gather the valid pixels and align only those.
    private func alignByGathering(
        pointMap1: MLXArray, conf1: MLXArray,
        pointMap2: MLXArray, conf2: MLXArray
    ) -> Sim3Alignment.Sim3 {
        let conf1CPU: [Float] = conf1.asArray(Float.self)
        let conf2CPU: [Float] = conf2.asArray(Float.self)
        let pm1CPU: [Float] = pointMap1.asArray(Float.self)
        let pm2CPU: [Float] = pointMap2.asArray(Float.self)

        let threshold = min(
            ChunkAlignment.median(conf1), ChunkAlignment.median(conf2)
        ) * ChunkAlignment.confThresholdScale

        var src: [Float] = [], tgt: [Float] = [], w: [Float] = []
        for i in 0 ..< conf1CPU.count where conf1CPU[i] > threshold && conf2CPU[i] > threshold {
            tgt.append(contentsOf: pm1CPU[(i * 3) ..< (i * 3 + 3)])
            src.append(contentsOf: pm2CPU[(i * 3) ..< (i * 3 + 3)])
            w.append((max(conf1CPU[i], 0) * max(conf2CPU[i], 0)).squareRoot())
        }
        guard !w.isEmpty else { return .identity }
        return Sim3Alignment.robustEstimate(
            src: MLXArray(src, [w.count, 3]),
            target: MLXArray(tgt, [w.count, 3]),
            initWeights: MLXArray(w, [w.count])
        )
    }

    /// Synthetic overlap: a known Sim(3) between two point maps, a band of
    /// low-confidence pixels, and non-finite coordinates hidden inside that band.
    private func makeCase(poisonRejected: Bool) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let views = 2, height = 12, width = 16
        let count = views * height * width
        var target = [Float](), source = [Float](), c1 = [Float](), c2 = [Float]()
        // Known transform: scale 1.7, yaw 0.3 rad, translation (0.4, -0.2, 1.1).
        let scale: Float = 1.7, yaw: Float = 0.3
        for i in 0 ..< count {
            let x = Float(i % width) * 0.11 - 0.7
            let y = Float((i / width) % height) * 0.09 - 0.4
            let z = 1.0 + Float(i % 7) * 0.13
            // src -> tgt : tgt = s * R(yaw) * src + t
            let rx = cos(yaw) * x - sin(yaw) * z
            let rz = sin(yaw) * x + cos(yaw) * z
            target.append(contentsOf: [scale * rx + 0.4, scale * y - 0.2, scale * rz + 1.1])

            let rejected = (i % 5) == 0
            if rejected && poisonRejected {
                source.append(contentsOf: [Float.nan, Float.infinity, -Float.infinity])
            } else {
                source.append(contentsOf: [x, y, z])
            }
            let conf: Float = rejected ? 0.0001 : Float(1 + i % 3)
            c1.append(conf)
            c2.append(conf)
        }
        return (
            MLXArray(target, [views, height, width, 3]), MLXArray(c1, [views, height, width]),
            MLXArray(source, [views, height, width, 3]), MLXArray(c2, [views, height, width])
        )
    }

    private func assertMatchesGather(poisonRejected: Bool, file: StaticString = #filePath, line: UInt = #line) {
        let (pm1, conf1, pm2, conf2) = makeCase(poisonRejected: poisonRejected)
        let fast = ChunkAlignment.alignWeighted(
            pointMap1: pm1, conf1: conf1, pointMap2: pm2, conf2: conf2
        )
        let reference = alignByGathering(
            pointMap1: pm1, conf1: conf1, pointMap2: pm2, conf2: conf2
        )
        XCTAssertEqual(fast.s, reference.s, accuracy: 1e-4, "scale", file: file, line: line)
        for i in 0 ..< 9 {
            XCTAssertEqual(fast.R[i], reference.R[i], accuracy: 1e-4, "R[\(i)]", file: file, line: line)
        }
        for i in 0 ..< 3 {
            XCTAssertEqual(fast.t[i], reference.t[i], accuracy: 1e-4, "t[\(i)]", file: file, line: line)
        }
        XCTAssertTrue(fast.s.isFinite && fast.R.allSatisfy(\.isFinite) && fast.t.allSatisfy(\.isFinite),
                      "result must be finite", file: file, line: line)
    }

    func testMatchesGatherImplementation() {
        assertMatchesGather(poisonRejected: false)
    }

    func testRejectedPixelsWithNonFiniteCoordinatesDoNotLeak() {
        assertMatchesGather(poisonRejected: true)
    }

    func testRecoversKnownTransform() {
        let (pm1, conf1, pm2, conf2) = makeCase(poisonRejected: true)
        let fit = ChunkAlignment.alignWeighted(
            pointMap1: pm1, conf1: conf1, pointMap2: pm2, conf2: conf2
        )
        XCTAssertEqual(fit.s, 1.7, accuracy: 1e-3)
        XCTAssertEqual(fit.t[0], 0.4, accuracy: 1e-2)
        XCTAssertEqual(fit.t[2], 1.1, accuracy: 1e-2)
    }

    func testMedianMatchesNumpySemantics() {
        // Even count: mean of the two middle values.
        XCTAssertEqual(ChunkAlignment.median(MLXArray([Float(4), 1, 3, 2])), 2.5, accuracy: 1e-6)
        // Odd count: the middle value.
        XCTAssertEqual(ChunkAlignment.median(MLXArray([Float(5), 1, 3])), 3.0, accuracy: 1e-6)
    }
}
