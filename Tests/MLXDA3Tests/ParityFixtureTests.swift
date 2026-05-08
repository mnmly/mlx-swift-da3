import Foundation
import MLX
import XCTest

@testable import MLXDA3

final class ParityFixtureTests: XCTestCase {
    func testDA3LargeTorchForwardFixture() throws {
        let env = ProcessInfo.processInfo.environment
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let defaultFixtureDir = packageRoot.appendingPathComponent("Tests/Fixtures").path
        let fixtureDir = env["DA3_FIXTURE_DIR"] ?? defaultFixtureDir
        let weightsPath = env["DA3_WEIGHTS"] ?? "\(NSHomeDirectory())/.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c/model.safetensors"

        let fixtureURL = URL(fileURLWithPath: fixtureDir)
            .appendingPathComponent("da3_large_forward.safetensors")
        guard FileManager.default.fileExists(atPath: fixtureURL.path),
              FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("DA3 parity fixture or weights not found. Generate with Scripts/generate_fixtures.py, or set DA3_FIXTURE_DIR and DA3_WEIGHTS.")
        }

        let fixture = try loadArrays(url: fixtureURL)

        guard let input = fixture["input_nhwc"] else {
            XCTFail("Fixture missing input_nhwc")
            return
        }

        let model = try DepthAnything3.fromPretrained(
            weightsPath,
            configName: "da3-large",
            dtype: .float16
        )
        let outputs = model(input.asType(.float16))
        eval(outputs)

        try assertClose(outputs, fixture, key: "depth", atol: 8e-2, rtol: 8e-2, meanAtol: 2e-2)
        try assertClose(outputs, fixture, key: "depth_conf", atol: 3e-1, rtol: 2.5e-1, meanAtol: 8e-2)
    }

    private func assertClose(
        _ actualOutputs: [String: MLXArray],
        _ fixture: [String: MLXArray],
        key: String,
        atol: Float,
        rtol: Float,
        meanAtol: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        guard let actual = actualOutputs[key] else {
            XCTFail("Missing Swift output \(key)", file: file, line: line)
            return
        }
        guard let expected = fixture["output.\(key)"] else {
            XCTFail("Missing fixture output.\(key)", file: file, line: line)
            return
        }

        let actualFloat = actual.asType(.float32)
        let expectedFloat = expected.asType(.float32)
        eval(actualFloat, expectedFloat)

        XCTAssertEqual(actualFloat.shape, expectedFloat.shape, "shape mismatch for \(key)", file: file, line: line)
        guard actualFloat.shape == expectedFloat.shape else { return }

        let a = actualFloat.asArray(Float.self)
        let e = expectedFloat.asArray(Float.self)
        var maxAbs: Float = 0
        var maxRel: Float = 0
        var meanAbs: Float = 0

        for (av, ev) in zip(a, e) {
            let absDiff = abs(av - ev)
            let relDiff = absDiff / max(abs(ev), 1e-6)
            maxAbs = max(maxAbs, absDiff)
            maxRel = max(maxRel, relDiff)
            meanAbs += absDiff
        }
        meanAbs /= Float(max(a.count, 1))

        XCTAssertLessThanOrEqual(
            maxAbs,
            atol,
            "\(key) maxAbs=\(maxAbs) meanAbs=\(meanAbs) maxRel=\(maxRel)",
            file: file,
            line: line
        )
        XCTAssertLessThanOrEqual(
            maxRel,
            rtol,
            "\(key) maxRel=\(maxRel) meanAbs=\(meanAbs) maxAbs=\(maxAbs)",
            file: file,
            line: line
        )
        XCTAssertLessThanOrEqual(
            meanAbs,
            meanAtol,
            "\(key) meanAbs=\(meanAbs) maxAbs=\(maxAbs) maxRel=\(maxRel)",
            file: file,
            line: line
        )
    }
}
