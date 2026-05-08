import MLX
import MLXNN
import XCTest

@testable import MLXDA3

final class BuildModelTests: XCTestCase {

    // MARK: - Model Construction

    func testBuildDA3MonoLarge() {
        let model = buildModel(configName: "da3mono-large")
        XCTAssertNotNil(model)
    }

    func testBuildDA3Small() {
        let model = buildModel(configName: "da3-small")
        XCTAssertNotNil(model)
    }

    func testBuildDA3Base() {
        let model = buildModel(configName: "da3-base")
        XCTAssertNotNil(model)
    }

    func testBuildDA3Large() {
        let model = buildModel(configName: "da3-large")
        XCTAssertNotNil(model)
    }

    func testBuildDA3Giant() {
        let model = buildModel(configName: "da3-giant")
        XCTAssertNotNil(model)
    }
}

final class ComponentShapeTests: XCTestCase {

    // MARK: - PatchEmbed

    func testPatchEmbedShape() {
        let pe = PatchEmbed(imgSize: 518, patchSize: 14, inChannels: 3, embedDim: 384)
        let input = MLXArray.zeros([2, 518, 518, 3])
        let output = pe(input)
        eval(output)
        // 518/14 = 37 patches per side, 37*37 = 1369
        XCTAssertEqual(output.shape, [2, 1369, 384])
    }

    // MARK: - Embeddings

    func testEmbeddingsShape() {
        let emb = Embeddings(imgSize: 518, patchSize: 14, inChannels: 3, embedDim: 384)
        let input = MLXArray.zeros([2, 518, 518, 3])
        let output = emb(input)
        eval(output)
        // 1369 patches + 1 CLS = 1370
        XCTAssertEqual(output.shape, [2, 1370, 384])
    }

    // MARK: - Attention

    func testAttentionShape() {
        let attn = DA3Attention(dim: 384, numHeads: 6)
        let input = MLXArray.zeros([1, 100, 384])
        let output = attn(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 100, 384])
    }

    // MARK: - FFN

    func testMlpShape() {
        let mlp = Mlp(inFeatures: 384, hiddenFeatures: 1536)
        let input = MLXArray.zeros([1, 50, 384])
        let output = mlp(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 50, 384])
    }

    func testSwiGLUFFNShape() {
        let ffn = SwiGLUFFN(inFeatures: 1536, hiddenFeatures: 6144)
        let input = MLXArray.zeros([1, 50, 1536])
        let output = ffn(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 50, 1536])
    }

    // MARK: - Block

    func testBlockShape() {
        let block = DA3Block(dim: 384, numHeads: 6)
        let input = MLXArray.zeros([1, 100, 384])
        let output = block(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 100, 384])
    }

    // MARK: - RoPE2D

    func testRoPE2DShape() {
        let rope = RotaryPositionEmbedding2D(frequency: 100.0)
        // tokens: [B, num_heads, N, head_dim]
        let tokens = MLXArray.zeros([1, 6, 37 * 37, 64])
        // positions: [B, N, 2]
        let positions = MLXArray.zeros([1, 37 * 37, 2]).asType(.int32)
        let output = rope(tokens, positions: positions)
        eval(output)
        XCTAssertEqual(output.shape, [1, 6, 1369, 64])
    }

    // MARK: - ResidualConvUnit

    func testResidualConvUnitShape() {
        let rcu = ResidualConvUnit(features: 64)
        let input = MLXArray.zeros([1, 37, 37, 64])
        let output = rcu(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 37, 37, 64])
    }

    // MARK: - AuxHead

    func testAuxHeadShape() {
        let head = AuxHead(inCh: 128, midCh: 32, outDim: 7)
        let input = MLXArray.zeros([1, 10, 10, 128])
        let output = head(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 10, 10, 7])
    }
}

final class ImageProcessorTests: XCTestCase {

    func testMakeDivisible() {
        let proc = ImageProcessor(processRes: 518, patchSize: 14)
        // 518/14 = 37, so 518 is divisible
        let input = createTestCGImage(width: 518, height: 518)
        let output = proc(input)
        eval(output)
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1] % 14, 0)
        XCTAssertEqual(output.shape[2] % 14, 0)
        XCTAssertEqual(output.shape[3], 3)
    }

    func testNonSquareImage() {
        let proc = ImageProcessor(processRes: 518, patchSize: 14)
        let input = createTestCGImage(width: 800, height: 600)
        let output = proc(input)
        eval(output)
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1] % 14, 0)
        XCTAssertEqual(output.shape[2] % 14, 0)
        XCTAssertEqual(output.shape[3], 3)
        // Width should be larger since image is landscape
        XCTAssertGreaterThanOrEqual(output.shape[2], output.shape[1])
    }
}

final class WeightKeyRemapTests: XCTestCase {

    func testBackboneBlockKey() {
        let result = remapKey("model.backbone.pretrained.blocks.0.attn.qkv.weight")
        XCTAssertEqual(result, "backbone.blocks.0.attn.qkv.weight")
    }

    func testBackboneNormKey() {
        let result = remapKey("model.backbone.pretrained.norm.weight")
        XCTAssertEqual(result, "backbone.norm.weight")
    }

    func testBackboneClsToken() {
        let result = remapKey("model.backbone.pretrained.cls_token")
        XCTAssertEqual(result, "backbone.embeddings.cls_token")
    }

    func testBackbonePosEmbed() {
        let result = remapKey("model.backbone.pretrained.pos_embed")
        XCTAssertEqual(result, "backbone.embeddings.pos_embed")
    }

    func testBackbonePatchEmbed() {
        let result = remapKey("model.backbone.pretrained.patch_embed.proj.weight")
        XCTAssertEqual(result, "backbone.embeddings.patch_embed.proj.weight")
    }

    func testCameraTokenKey() {
        let result = remapKey("model.backbone.pretrained.camera_token")
        XCTAssertEqual(result, "backbone.camera_token")
    }

    func testHeadNormKey() {
        let result = remapKey("model.head.norm.weight")
        XCTAssertEqual(result, "head.norm.weight")
    }

    func testHeadProjectsKey() {
        let result = remapKey("model.head.projects.0.weight")
        XCTAssertEqual(result, "head.projects.0.weight")
    }

    func testResizeLayerRemap() {
        let result = remapKey("model.head.resize_layers.0.weight")
        XCTAssertEqual(result, "head.resize_layer0.weight")
    }

    func testResizeLayerIdentitySkipped() {
        let result = remapKey("model.head.resize_layers.2.weight")
        XCTAssertNil(result)
    }

    func testScratchLayerRn() {
        let result = remapKey("model.head.scratch.layer1_rn.weight")
        XCTAssertEqual(result, "head.scratch.layer1_rn.weight")
    }

    func testRefinenetKey() {
        let result = remapKey("model.head.scratch.refinenet3.resConfUnit2.conv1.weight")
        XCTAssertEqual(result, "head.refinenet3.resConfUnit2.conv1.weight")
    }

    func testOutputConv2Sequential() {
        let conv0 = remapKey("model.head.scratch.output_conv2.0.weight")
        XCTAssertEqual(conv0, "head.output_conv2_0.weight")

        let relu = remapKey("model.head.scratch.output_conv2.1.weight")
        XCTAssertNil(relu) // ReLU has no weights, but key would be skipped

        let conv2 = remapKey("model.head.scratch.output_conv2.2.weight")
        XCTAssertEqual(conv2, "head.output_conv2_1.weight")
    }

    func testCamEncSkipped() {
        let result = remapKey("model.cam_enc.layer.weight")
        XCTAssertNil(result)
    }

    func testNoModelPrefixSkipped() {
        let result = remapKey("some_other_key.weight")
        XCTAssertNil(result)
    }

    func testAuxOutputConv1Aux() {
        let result = remapKey("model.head.scratch.output_conv1_aux.0.1.weight")
        XCTAssertEqual(result, "head.output_conv1_aux.0.1.weight")
    }

    func testAuxOutputConv2Aux() {
        let conv0 = remapKey("model.head.scratch.output_conv2_aux.0.0.weight")
        XCTAssertEqual(conv0, "head.output_conv2_aux.0.conv0.weight")

        let ln = remapKey("model.head.scratch.output_conv2_aux.0.2.weight")
        XCTAssertEqual(ln, "head.output_conv2_aux.0.ln.weight")

        let conv1 = remapKey("model.head.scratch.output_conv2_aux.0.5.weight")
        XCTAssertEqual(conv1, "head.output_conv2_aux.0.conv1.weight")
    }
}

// MARK: - Test Helpers

func createTestCGImage(width: Int, height: Int) -> CGImage {
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    var data = [UInt8](repeating: 128, count: height * bytesPerRow)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let context = CGContext(
        data: &data,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    return context.makeImage()!
}
