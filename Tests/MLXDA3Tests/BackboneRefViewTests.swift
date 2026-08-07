import Foundation
import MLX
import XCTest

@testable import MLXDA3

/// The backbone promotes a content-selected reference view to index 0 and restores the
/// original ordering before emitting features. Two consequences are testable without
/// checkpoint weights:
///
///  * output shape is unchanged by the reordering, and
///  * with a content-based strategy the whole forward pass is permutation-equivariant —
///    shuffling the input views shuffles the outputs identically. That only holds if
///    reorder and restore are exact inverses wired in the right places.
final class BackboneRefViewTests: XCTestCase {

    private func makeBackbone() -> DinoVisionTransformer {
        DinoVisionTransformer(
            imgSize: 28,
            patchSize: 14,
            inChannels: 3,
            embedDim: 32,
            depth: 4,
            numHeads: 2,
            mlpRatio: 2.0,
            initValues: 1.0,
            altStart: 2,
            qknormStart: -1,
            ropeStart: -1,
            catToken: false,
            outLayers: [3]
        )
    }

    func testReferenceViewSelectionIsPermutationEquivariant() {
        MLXRandom.seed(0)
        let backbone = makeBackbone()
        let views = 4
        let x = MLXRandom.normal([1, views, 28, 28, 3])

        let permutation = [2, 0, 3, 1]
        let permuted = concatenated(
            permutation.map { x[0..., $0 ..< ($0 + 1)] }, axis: 1
        )

        let (base, _) = backbone(x, refViewStrategy: .saddleBalanced)
        let (shuffled, _) = backbone(permuted, refViewStrategy: .saddleBalanced)
        XCTAssertEqual(base.count, 1)
        XCTAssertEqual(base[0].0.shape, [1, views, 4, 32])

        for (position, source) in permutation.enumerated() {
            let expected = base[0].0[0..., source ..< (source + 1)]
            let actual = shuffled[0].0[0..., position ..< (position + 1)]
            let diff = abs(actual - expected).max().item(Float.self)
            XCTAssertLessThan(diff, 1e-3, "view \(source) moved to slot \(position)")
        }
    }

    func testTwoViewInputSkipsReordering() {
        MLXRandom.seed(0)
        let backbone = makeBackbone()
        // Below THRESH_FOR_REF_SELECTION the strategy must not change anything.
        let x = MLXRandom.normal([1, 2, 28, 28, 3])
        let (balanced, _) = backbone(x, refViewStrategy: .saddleBalanced)
        let (first, _) = backbone(x, refViewStrategy: .first)
        XCTAssertLessThan(
            abs(balanced[0].0 - first[0].0).max().item(Float.self), 1e-6
        )
    }
}
