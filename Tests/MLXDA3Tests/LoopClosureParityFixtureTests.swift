import CoreGraphics
import Foundation
import MLX
import XCTest

@testable import MLXDA3
import MLXSALAD
@testable import MLXDA3Streaming

/// End-to-end parity test for the loop-closure-enabled streaming path.
///
/// Composes:
///   1. `StreamingPipeline.predict(images:)`  — initial chunk-pair Sim(3)s
///   2. `LoopDetector.detect(images:)`        — SALAD VPR loop pair search
///   3. `StreamingPipeline.computeLoopMeasurement(...)` — chunk-pair Sim(3)
///      measurements derived from re-running DA3 on combined frames
///   4. `StreamingPipeline.predict(images:loopConstraints:)` — refined run
///
/// Compares the final camera poses + intrinsics against a python fixture
/// generated with `Scripts/generate_streaming_parity_fixture.py
/// --enable-loop-closure`.
///
/// Note about the bundled fixture (robot_unitree, 42 frames):
/// The robot is moving forward — SALAD detects no loop closures. Both
/// python and swift paths therefore exit early when zero loops are
/// detected and produce the same output as the non-loop path. This is a
/// **plumbing-parity** test: it verifies the loop-enabled control flow
/// matches between the two pipelines on real input data.
///
/// Coverage of the actual optimizer + measurement code paths comes from
/// the unit fixtures `Sim3LoopOptimizerFixtureTests` and
/// `LoopDetectionFixtureTests` (now in the mlx-swift-salad package). End-to-end coverage *with* detected loops
/// requires KITTI-style data with real loop closures (not bundled with
/// either repo).
final class LoopClosureParityFixtureTests: XCTestCase {
    func testRobotUnitreeLoopEnabledParity() throws {
        let env = ProcessInfo.processInfo.environment
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixtureDir = env["DA3_STREAMING_FIXTURE_DIR"]
            ?? packageRoot.appendingPathComponent("Tests/Fixtures").path
        let imageDir = env["DA3_LOOP_FIXTURE_IMAGES"]
            ?? packageRoot.appendingPathComponent("tmp/da3_robot_frames").path
        let weightsPath = env["DA3_STREAMING_WEIGHTS"] ?? ""
        let saladPath = env["DA3_SALAD_WEIGHTS"] ?? ""
        let fixtureURL = URL(fileURLWithPath: fixtureDir)
            .appendingPathComponent("da3_streaming_loop_fixture.safetensors")

        guard FileManager.default.fileExists(atPath: fixtureURL.path),
              FileManager.default.fileExists(atPath: weightsPath),
              FileManager.default.fileExists(atPath: saladPath),
              FileManager.default.fileExists(atPath: imageDir)
        else {
            throw XCTSkip("loop-enabled fixture, weights, or input frames not found. Generate with: Scripts/generate_streaming_parity_fixture.py --enable-loop-closure --salad-weights ...")
        }

        let fixture = try loadArrays(url: fixtureURL)
        guard let pyPoses = fixture["output.camera_poses_c2w"],
              let pyIntr  = fixture["output.intrinsics_pixel"]
        else {
            // Either the fixture hasn't been regenerated with loop_closure
            // enabled, or it has the older streaming-only schema. Skip
            // gracefully so the developer regenerates as needed.
            throw XCTSkip("fixture missing camera_poses or intrinsics; regenerate with --enable-loop-closure.")
        }

        // Optional loop_frame_pairs (only present in loop-enabled fixtures)
        let pyLoopPairs = fixture["output.loop_frame_pairs"]

        // Build streaming pipeline
        var cfg = StreamingPipeline.Config()
        cfg.chunkSize = 8
        cfg.overlap = 4
        cfg.resolution = 504
        cfg.dtype = .float32
        cfg.verbose = false

        let model = try DepthAnything3.fromPretrained(weightsPath, configName: "da3-giant", dtype: cfg.dtype)
        let pipeline = StreamingPipeline(model: model, config: cfg)

        // Load images
        let imagePaths = try ImageDirectory.listImagePaths(in: imageDir).prefix(42)
        let images: [CGImage] = try imagePaths.map { try ImageDirectory.loadCGImage(path: $0) }

        // 1. Initial streaming pass (no loops)
        let initial = pipeline.predict(images: images)

        // 2. SALAD loop detection
        var detCfg = LoopDetector.Config()
        detCfg.imageSize = (height: 336, width: 336)
        detCfg.batchSize = 32
        detCfg.similarityThreshold = 0.85
        detCfg.topK = 5
        detCfg.minFrameDistance = 10
        detCfg.useNMS = true
        detCfg.nmsThreshold = 25
        let detector = try LoopDetector.fromPretrained(saladPath, config: detCfg, dtype: .float32)
        let (_, swLoops) = detector.detect(images: images)

        // Verify swift's detected loop pair count matches python's. Both should
        // typically be zero on robot_unitree (forward motion).
        if let pyPairs = pyLoopPairs {
            let pyCount = pyPairs.dim(0)
            // Allow a small tolerance — the documented 0.985 per-frame cosine
            // sim between swift and python descriptors means a candidate near
            // the threshold can flip side. ±2 pairs is reasonable for a 42-frame
            // sequence at threshold 0.85.
            XCTAssertLessThanOrEqual(
                abs(swLoops.count - pyCount), 2,
                "loop pair count mismatch: swift=\(swLoops.count) python=\(pyCount)"
            )
        }

        // 3. Compute Sim(3) measurement per loop pair (skipped when empty)
        var loopConstraints: [LoopConstraint] = []
        for loop in swLoops {
            let chunkAIdx = chunkIndexFor(frame: loop.a, ranges: initial.chunkRanges)
            let chunkBIdx = chunkIndexFor(frame: loop.b, ranges: initial.chunkRanges)
            guard let cA = chunkAIdx, let cB = chunkBIdx, cA != cB else { continue }
            let framesA = framesFor(chunk: cA, images: images, ranges: initial.chunkRanges)
            let framesB = framesFor(chunk: cB, images: images, ranges: initial.chunkRanges)
            if let constraint = pipeline.computeLoopMeasurement(
                chunkA: initial.perChunk[cA], framesA: framesA, chunkAIdx: cA,
                chunkB: initial.perChunk[cB], framesB: framesB, chunkBIdx: cB
            ) {
                loopConstraints.append(constraint)
            }
        }

        // 4. Refined run with loop constraints
        let refined = pipeline.predict(images: images, loopConstraints: loopConstraints)

        // Compare against python: poses + intrinsics
        try assertClose(actual: refined.cameraPosesC2W, expected: pyPoses,
                        key: "camera_poses_c2w", atol: 2.0, meanAtol: 0.5)
        let kFlat = swiftIntrinsicsToFxFyCxCy(refined.intrinsicsK)
        try assertClose(actual: kFlat, expected: pyIntr,
                        key: "intrinsics_pixel", atol: 400.0, meanAtol: 60.0)
    }

    /// Find the chunk index that contains a given frame.
    private func chunkIndexFor(frame: Int, ranges: [(Int, Int)]) -> Int? {
        for (idx, range) in ranges.enumerated() {
            if frame >= range.0 && frame < range.1 { return idx }
        }
        return nil
    }

    /// Slice the original image array to those that belong to a chunk.
    private func framesFor(chunk: Int, images: [CGImage], ranges: [(Int, Int)]) -> [CGImage] {
        let r = ranges[chunk]
        return Array(images[r.0..<r.1])
    }

    /// Convert `[N, 3, 3]` intrinsics to `[N, 4]` `[fx fy cx cy]` to match the fixture.
    private func swiftIntrinsicsToFxFyCxCy(_ k: MLXArray) -> MLXArray {
        let n = k.dim(0)
        var flat = [Float](repeating: 0, count: n * 4)
        let kCPU = k.asArray(Float.self)
        for i in 0..<n {
            let base = i * 9
            flat[i * 4 + 0] = kCPU[base + 0]
            flat[i * 4 + 1] = kCPU[base + 4]
            flat[i * 4 + 2] = kCPU[base + 2]
            flat[i * 4 + 3] = kCPU[base + 5]
        }
        return MLXArray(flat, [n, 4])
    }

    private func assertClose(
        actual: MLXArray, expected: MLXArray, key: String,
        atol: Float, meanAtol: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let af = actual.asType(.float32)
        let ef = expected.asType(.float32)
        eval(af, ef)
        XCTAssertEqual(af.shape, ef.shape, "shape mismatch for \(key)", file: file, line: line)
        guard af.shape == ef.shape else { return }

        let a = af.asArray(Float.self)
        let e = ef.asArray(Float.self)
        var maxAbs: Float = 0
        var sumAbs: Double = 0
        var n = 0
        for (av, ev) in zip(a, e) where av.isFinite && ev.isFinite {
            let d = abs(av - ev)
            maxAbs = max(maxAbs, d)
            sumAbs += Double(d)
            n += 1
        }
        let meanAbs = Float(sumAbs / Double(max(n, 1)))
        if maxAbs > atol || meanAbs > meanAtol {
            XCTFail("\(key): maxAbs=\(maxAbs) (atol=\(atol)), meanAbs=\(meanAbs) (meanAtol=\(meanAtol))",
                    file: file, line: line)
        }
    }
}
