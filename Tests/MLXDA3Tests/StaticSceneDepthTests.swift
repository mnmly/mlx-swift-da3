import Foundation
import MLX
import XCTest

@testable import MLXDA3

final class StaticSceneDepthTests: XCTestCase {

    func testTemporalMedianMatchesNumpySemantics() {
        // Odd count -> middle value.
        let odd = [MLXArray([Float(3), 9]), MLXArray([Float(1), 7]), MLXArray([Float(2), 8])]
        let m1 = StaticSceneDepth.temporalMedian(odd)
        eval(m1)
        XCTAssertEqual(m1.asArray(Float.self), [2, 8])

        // Even count -> mean of the two middle values.
        let even = [MLXArray([Float(4)]), MLXArray([Float(1)]), MLXArray([Float(3)]), MLXArray([Float(2)])]
        let m2 = StaticSceneDepth.temporalMedian(even)
        eval(m2)
        XCTAssertEqual(m2.item(Float.self), 2.5, accuracy: 1e-6)
    }

    /// The whole point of the median: outlier frames must not move the result.
    func testMedianRejectsOutlierFrames() {
        let truth = MLXArray.ones([8, 8]) * 2.0
        var maps = (0 ..< 9).map { _ in truth }
        maps[3] = truth * 40      // a frame where the model blew up
        maps[7] = truth * 0.01    // and one where it collapsed
        let median = StaticSceneDepth.temporalMedian(maps)
        eval(median)
        XCTAssertEqual(abs(median - truth).max().item(Float.self), 0, accuracy: 1e-6)
    }

    func testPercentile() {
        let x = MLXArray((0 ... 100).map { Float($0) })
        XCTAssertEqual(StaticSceneDepth.percentile(x, 0.0), 0, accuracy: 1e-5)
        XCTAssertEqual(StaticSceneDepth.percentile(x, 0.5), 50, accuracy: 1e-5)
        XCTAssertEqual(StaticSceneDepth.percentile(x, 1.0), 100, accuracy: 1e-5)
    }

    /// Percentile clipping exists so a handful of very near pixels cannot crush the
    /// rest of the scene into a few codes.
    func testNormalizeSpendsRangeOnTheBulkNotTheOutliers() {
        var values = (0 ..< 999).map { Float($0) / 998 }  // bulk in 0...1
        values.append(500)                               // one absurdly near pixel
        let inv = MLXArray(values)

        let clipped = StaticSceneDepth.normalize(
            inverseDepth: inv, nearPercentile: 99, farPercentile: 1
        )
        eval(clipped.map)
        // The bulk should still span most of the output range.
        let bulk = clipped.map[0 ..< 999]
        XCTAssertGreaterThan(bulk.max().item(Float.self) - bulk.min().item(Float.self), 0.9)
        XCTAssertEqual(clipped.map.max().item(Float.self), 1.0, accuracy: 1e-6)
        XCTAssertEqual(clipped.map.min().item(Float.self), 0.0, accuracy: 1e-6)

        // Without clipping, the single outlier would take the whole range.
        let unclipped = StaticSceneDepth.normalize(
            inverseDepth: inv, nearPercentile: 100, farPercentile: 0
        )
        eval(unclipped.map)
        let squashed = unclipped.map[0 ..< 999]
        XCTAssertLessThan(squashed.max().item(Float.self), 0.01)
    }

    func testInverseDepthGuardsZero() {
        let d = MLXArray([Float(0), 0.5, 2])
        let inv = StaticSceneDepth.inverseDepth(d, floor: 1e-3)
        eval(inv)
        let v = inv.asArray(Float.self)
        XCTAssertEqual(v[0], 1000, accuracy: 1e-3)
        XCTAssertEqual(v[1], 2, accuracy: 1e-6)
        XCTAssertEqual(v[2], 0.5, accuracy: 1e-6)
        XCTAssertTrue(v.allSatisfy(\.isFinite))
    }

    func testRelativeSpreadIsZeroForIdenticalFrames() {
        let maps = (0 ..< 5).map { _ in MLXArray.ones([4, 4]) * 3.0 }
        let median = StaticSceneDepth.temporalMedian(maps)
        XCTAssertEqual(StaticSceneDepth.relativeSpread(maps, median: median), 0, accuracy: 1e-6)
    }
}

final class GuidedFilterTests: XCTestCase {

    func testBoxMeanIsExactIncludingEdges() {
        let x = MLXArray((0 ..< 25).map { Float($0) }, [5, 5])
        let mean = GuidedFilter.boxMean(x, radius: 1)
        eval(mean)
        let got = mean.asArray(Float.self)

        // Reference: same window, clipped at the border, divided by the true count.
        for row in 0 ..< 5 {
            for col in 0 ..< 5 {
                var sum: Float = 0
                var count: Float = 0
                for dy in -1 ... 1 {
                    for dx in -1 ... 1 {
                        let r = row + dy, c = col + dx
                        guard r >= 0, r < 5, c >= 0, c < 5 else { continue }
                        sum += Float(r * 5 + c)
                        count += 1
                    }
                }
                XCTAssertEqual(got[row * 5 + col], sum / count, accuracy: 1e-4,
                               "box mean at (\(row), \(col))")
            }
        }
    }

    func testBoxMeanPreservesConstants() {
        let x = MLXArray.ones([16, 24]) * 7.0
        let mean = GuidedFilter.boxMean(x, radius: 3)
        eval(mean)
        XCTAssertEqual(abs(mean - 7.0).max().item(Float.self), 0, accuracy: 1e-4)
    }

    /// A guide that is an exact affine function of the map must be reproduced
    /// exactly — that is the model the filter fits.
    func testUpsampleRecoversAffineRelationship() {
        let height = 64, width = 96
        var guideValues = [Float]()
        for r in 0 ..< height {
            for c in 0 ..< width {
                guideValues.append(c < width / 2 ? 0.2 : 0.8)  // a hard edge
            }
        }
        let guide = MLXArray(guideValues, [height, width])
        // Low-res map that is 0.5·guide + 0.1, sampled at quarter resolution.
        let low = GuidedFilter.resample(guide * 0.5 + 0.1, height: height / 4, width: width / 4)

        let up = GuidedFilter.upsample(lowRes: low, guide: guide, radius: 4, epsilon: 1e-6)
        eval(up)
        let expected = guide * 0.5 + 0.1

        // Away from the edge the fit should be near-exact.
        let interior = abs(up - expected)[8 ..< (height - 8), 8 ..< (width / 2 - 8)]
        XCTAssertLessThan(interior.max().item(Float.self), 0.02)
    }

    /// The upsampled map must land on the guide's edge, not the low-res one.
    func testUpsampleTransfersEdgeLocation() {
        let height = 64, width = 64
        var guideValues = [Float](repeating: 0, count: height * width)
        for r in 0 ..< height {
            for c in 0 ..< width where c >= 40 { guideValues[r * width + c] = 1 }
        }
        let guide = MLXArray(guideValues, [height, width])
        let low = GuidedFilter.resample(guide, height: 8, width: 8)

        let up = GuidedFilter.upsample(lowRes: low, guide: guide, radius: 2, epsilon: 1e-6)
        eval(up)
        let row = up[height / 2]
        // Step should be sharp across columns 39 -> 41, which a plain 8x upsample
        // of the low-res map could not produce.
        XCTAssertLessThan(row[38].item(Float.self), 0.25)
        XCTAssertGreaterThan(row[42].item(Float.self), 0.75)
    }
}
