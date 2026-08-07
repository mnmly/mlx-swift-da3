import Foundation
import MLX
import XCTest

@testable import MLXDA3

/// Parity against python `model/reference_view_selector.py`.
/// Fixture: `Scripts/generate_ref_view_fixture.py`.
final class ReferenceViewSelectionTests: XCTestCase {

    private struct Meta: Decodable {
        let views: Int
        let tokens: Int
        let channels: Int
        let selected: [String: [Int]]
    }

    func testMatchesPythonSelector() throws {
        let fixtureDir = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["DA3_FIXTURE_DIR"]
                ?? URL(fileURLWithPath: #filePath)
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .appendingPathComponent("Tests/Fixtures").path
        )
        let tensorsURL = fixtureDir.appendingPathComponent("ref_view_fixture.safetensors")
        let metaURL = fixtureDir.appendingPathComponent("ref_view_fixture.json")
        guard FileManager.default.fileExists(atPath: tensorsURL.path),
              FileManager.default.fileExists(atPath: metaURL.path)
        else {
            throw XCTSkip("Ref-view fixture not found. Generate with Scripts/generate_ref_view_fixture.py")
        }

        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let fixture = try loadArrays(url: tensorsURL)
        let x = try XCTUnwrap(fixture["x"])

        for strategy in RefViewStrategy.allCases {
            let expectedIndex = try XCTUnwrap(meta.selected[strategy.rawValue]?.first)
            let idx = ReferenceViewSelection.select(x, strategy: strategy)
            eval(idx)
            XCTAssertEqual(
                idx.item(Int32.self), Int32(expectedIndex),
                "selected view for \(strategy.rawValue)"
            )

            let reordered = ReferenceViewSelection.reorder(x, referenceIndices: idx)
            let expectedReordered = try XCTUnwrap(fixture["reordered.\(strategy.rawValue)"])
            XCTAssertLessThan(
                abs(reordered - expectedReordered).max().item(Float.self), 1e-6,
                "reorder for \(strategy.rawValue)"
            )

            let restored = ReferenceViewSelection.restore(reordered, referenceIndices: idx)
            XCTAssertLessThan(
                abs(restored - x).max().item(Float.self), 1e-6,
                "restore for \(strategy.rawValue)"
            )
        }
    }

    func testSelectionSkippedBelowThreshold() {
        // Python only reorders at S >= THRESH_FOR_REF_SELECTION (3).
        XCTAssertEqual(ReferenceViewSelection.viewThreshold, 3)
        let x = MLXArray.zeros([1, 1, 4, 8])
        let idx = ReferenceViewSelection.select(x, strategy: .saddleBalanced)
        eval(idx)
        XCTAssertEqual(idx.item(Int32.self), 0)
    }
}
