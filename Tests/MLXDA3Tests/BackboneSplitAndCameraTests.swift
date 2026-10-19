import MLX
import MLXNN
import XCTest

@testable import MLXDA3

/// Fixture-free checks for the split-forward backbone API, feature export and the camera
/// encoder/decoder helpers.
final class BackboneSplitAndCameraTests: XCTestCase {
    /// A small multi-view backbone with the same layout rules as DA3 (alt/rope/qknorm start).
    private func smallBackbone() -> DinoVisionTransformer {
        MLXRandom.seed(0)
        let vit = DinoVisionTransformer(
            imgSize: 56, patchSize: 14, embedDim: 64, depth: 8, numHeads: 4,
            altStart: 3, qknormStart: 3, ropeStart: 3, catToken: true, outLayers: [2, 5, 7])
        // Non-trivial weights everywhere (LayerScale defaults to 1, camera token to zeros).
        vit.update(parameters: vit.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        eval(vit)
        return vit
    }

    func testSplitRunMatchesSinglePass() {
        let vit = smallBackbone()
        let x = MLXRandom.normal([1, 4, 56, 70, 3])
        let (full, fullAux) = vit(x, exportFeatLayers: [2, 6])

        var state = vit.prepare(x)
        vit.run(&state, through: 4, exportFeatLayers: [2, 6])
        vit.run(&state, through: vit.blockCount, exportFeatLayers: [2, 6])
        let (split, splitAux) = vit.finish(state)

        XCTAssertEqual(full.count, 3)
        XCTAssertEqual(fullAux.count, 2)
        for (a, b) in zip(full, split) {
            XCTAssertEqual(abs(a.0 - b.0).max().item(Float.self), 0)
            XCTAssertEqual(abs(a.1 - b.1).max().item(Float.self), 0)
        }
        for (a, b) in zip(fullAux, splitAux) {
            XCTAssertEqual(abs(a - b).max().item(Float.self), 0)
        }
        // Exported features are normed patch tokens [B, S, N, D].
        XCTAssertEqual(fullAux[0].shape, [1, 4, 20, 64])
    }

    /// Resuming on a second backbone that shares the prefix blocks equals that backbone's
    /// own full pass.
    func testResumeOnSecondBackboneWithSharedPrefix() {
        let first = smallBackbone()
        let second = smallBackbone()
        // Same weights as `first`, except blocks 4..7.
        let flat = first.parameters().flattened().map { key, value -> (String, MLXArray) in
            let block = key.split(separator: ".").dropFirst().first.flatMap { Int($0) }
            if key.hasPrefix("blocks."), let block, block >= 4 { return (key, value + 0.01) }
            return (key, value)
        }
        second.update(parameters: ModuleParameters.unflattened(flat))
        eval(second)

        let x = MLXRandom.normal([1, 3, 56, 56, 3])
        var state = first.prepare(x)
        first.run(&state, through: 4)
        second.run(&state, through: second.blockCount)
        let (resumed, _) = second.finish(state)
        let (reference, _) = second(x)
        for (a, b) in zip(resumed, reference) {
            XCTAssertLessThan(abs(a.0 - b.0).max().item(Float.self), 1e-5)
        }
    }

    func testExportOrderRestoration() {
        let vit = smallBackbone()
        let x = MLXRandom.normal([1, 4, 56, 56, 3])
        var reordered = vit.prepare(x)
        vit.run(&reordered, through: vit.blockCount, exportFeatLayers: [6], restoreExportOrder: false)
        var restored = vit.prepare(x)
        vit.run(&restored, through: vit.blockCount, exportFeatLayers: [6], restoreExportOrder: true)
        guard let idx = restored.referenceIndices else { return XCTFail("reference view selection did not run") }
        let back = ReferenceViewSelection.restore(reordered.exported[0], referenceIndices: idx)
        XCTAssertEqual(abs(back - restored.exported[0]).max().item(Float.self), 0)
    }

    func testPoseEncodingRoundTrip() {
        // Rotation about an oblique axis, translation, distinct fx / fy.
        let angle: Float = 0.7
        let axis = [Float(1), 2, 3].map { $0 / Float(14).squareRoot() }
        let (c, s) = (cos(angle), sin(angle))
        let (x, y, z) = (axis[0], axis[1], axis[2])
        let R: [Float] = [
            c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s,
            y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s,
            z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c),
        ]
        let c2w = MLXArray([R[0], R[1], R[2], 0.3, R[3], R[4], R[5], -0.2, R[6], R[7], R[8], 1.5, 0, 0, 0, 1], [1, 1, 4, 4])
        let K = MLXArray([Float(300), 0, 140, 0, 320, 112, 0, 0, 1], [1, 1, 3, 3])
        let encoding = PoseEncoding.encode(c2w: c2w, intrinsics: K, height: 224, width: 280)
        XCTAssertGreaterThanOrEqual(encoding[0, 0, 6].item(Float.self), 0, "quaternion is standardized (w >= 0)")
        let (decoded, decodedK) = PoseEncoding.decode(encoding, height: 224, width: 280)
        XCTAssertLessThan(abs(decoded - c2w[.ellipsis, ..<3, 0...]).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(decodedK - K).max().item(Float.self), 1e-3)
        XCTAssertLessThan(abs(affineInverse(affineInverse(c2w)) - c2w).max().item(Float.self), 1e-6)
    }

    func testCameraModuleKeyRemap() {
        XCTAssertNil(remapDA3NetKey("cam_dec.fc_t.weight"))
        XCTAssertEqual(remapDA3NetKey("cam_dec.backbone.2.weight", includeCameraModules: true), "cam_dec.backbone.1.weight")
        XCTAssertEqual(remapDA3NetKey("cam_dec.fc_fov.0.bias", includeCameraModules: true), "cam_dec.fc_fov.bias")
        XCTAssertEqual(remapDA3NetKey("cam_enc.trunk.0.attn.qkv.weight", includeCameraModules: true),
                       "cam_enc.trunk.0.attn.qkv.weight")
        let model = buildModel(configName: "da3-small", includeCameraModules: true)
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("cam_dec.backbone.1.weight"))
        XCTAssertTrue(keys.contains("cam_enc.trunk.3.mlp.fc2.bias"))
        XCTAssertNil(buildModel(configName: "da3mono-large", includeCameraModules: true).cameraDecoder)
    }
}
