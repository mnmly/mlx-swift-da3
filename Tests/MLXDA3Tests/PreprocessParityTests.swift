import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import MLXDA3
@testable import MLXDA3Streaming

/// Parity against python `InputProcessor` (`upper_bound_resize`).
/// Fixture: `Scripts/generate_preprocess_fixture.py`.
final class PreprocessParityTests: XCTestCase {

    private struct Meta: Decodable {
        let images: [String]
        let processRes: Int
        let patchSize: Int
        let height: Int
        let width: Int
        let imagenetMean: [Float]
        let imagenetStd: [Float]

        enum CodingKeys: String, CodingKey {
            case images
            case processRes = "process_res"
            case patchSize = "patch_size"
            case height
            case width
            case imagenetMean = "imagenet_mean"
            case imagenetStd = "imagenet_std"
        }
    }

    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testMultiViewPreprocessMatchesPython() throws {
        let fixtureDir = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["DA3_FIXTURE_DIR"]
                ?? packageRoot.appendingPathComponent("Tests/Fixtures").path
        )
        let tensorsURL = fixtureDir.appendingPathComponent("preprocess_fixture.safetensors")
        let metaURL = fixtureDir.appendingPathComponent("preprocess_fixture.json")
        guard FileManager.default.fileExists(atPath: tensorsURL.path),
              FileManager.default.fileExists(atPath: metaURL.path)
        else {
            throw XCTSkip("Preprocess fixture not found. Generate with Scripts/generate_preprocess_fixture.py")
        }

        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let images: [CGImage] = try meta.images.map { relative in
            let url = relative.hasPrefix("/")
                ? URL(fileURLWithPath: relative)
                : packageRoot.appendingPathComponent(relative)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                throw XCTSkip("Fixture source image missing: \(url.path)")
            }
            return image
        }

        let fixture = try loadArrays(url: tensorsURL)
        // cv2's uint8 resize output — the part of the chain that can actually diverge.
        let expectedRGB = try XCTUnwrap(fixture["processed_rgb"]).asType(.float32)
        // Python's ImageNet normalization, recomputed from that same reference image so
        // a drift in our constants or formula still fails here.
        XCTAssertEqual(meta.imagenetMean, DA3ImagePreprocessing.imagenetMean)
        XCTAssertEqual(meta.imagenetStd, DA3ImagePreprocessing.imagenetStd)
        let expectedInput =
            (expectedRGB / 255.0 - MLXArray(meta.imagenetMean, [1, 1, 3]))
            / MLXArray(meta.imagenetStd, [1, 1, 3])

        let preprocessor = MultiViewPreprocessor(
            processRes: meta.processRes, patchSize: meta.patchSize
        )
        let batch = preprocessor.processBatch(images, dtype: .float32)

        XCTAssertEqual(batch.height, meta.height)
        XCTAssertEqual(batch.width, meta.width)
        XCTAssertEqual(
            batch.processedImages.shape, expectedRGB.shape,
            "processed image shape must match python"
        )

        // cv2 resamples 8-bit images with 11-bit fixed-point weights; we resample in
        // float32. The residual is that quantization: ~0.05 uint8 levels on average,
        // never more than 2 levels on the sharpest pixels.
        let rgbDiff = abs(batch.processedImages.asType(.float32) - expectedRGB)
        let rgbMax = rgbDiff.max().item(Float.self)
        let rgbMean = rgbDiff.mean().item(Float.self)
        XCTAssertLessThanOrEqual(rgbMax, 2.0, "max uint8 drift vs cv2")
        XCTAssertLessThanOrEqual(rgbMean, 0.08, "mean uint8 drift vs cv2")

        let inputDiff = abs(batch.input.squeezed(axis: 0) - expectedInput)
        let inputMax = inputDiff.max().item(Float.self)
        let inputMean = inputDiff.mean().item(Float.self)
        XCTAssertLessThanOrEqual(inputMax, 0.04, "max normalized drift vs python")
        XCTAssertLessThanOrEqual(inputMean, 0.001, "mean normalized drift vs python")

        print(String(
            format: "[preprocess parity] rgb max=%.3f mean=%.4f | input max=%.4f mean=%.6f",
            rgbMax, rgbMean, inputMax, inputMean
        ))
    }

    func testSingleImageProcessorMatchesMultiView() throws {
        let image = makeGradientImage(width: 1280, height: 720)
        let single = ImageProcessor(processRes: 504, patchSize: 14)(image)
        let batch = MultiViewPreprocessor(processRes: 504, patchSize: 14)
            .processBatch([image], dtype: .float32)
        eval(single, batch.input)
        XCTAssertEqual(single.shape, [1, batch.height, batch.width, 3])
        XCTAssertLessThan(abs(single - batch.input.squeezed(axis: 0)).max().item(Float.self), 1e-5)
    }

    func testTargetSizeRoundsToNearestPatchMultiple() {
        // 1280x720 -> longest side 504 -> (504, 284) -> nearest multiples of 14 -> (504, 280)
        let a = DA3ImagePreprocessing.targetSize(width: 1280, height: 720, processRes: 504, patchSize: 14)
        XCTAssertEqual(a.width, 504)
        XCTAssertEqual(a.height, 280)

        // Rounding must go up when that is nearer: 289 -> 294, not 280.
        XCTAssertEqual(DA3ImagePreprocessing.nearestMultiple(289, 14), 294)
        XCTAssertEqual(DA3ImagePreprocessing.nearestMultiple(284, 14), 280)
        // Ties round up, matching python.
        XCTAssertEqual(DA3ImagePreprocessing.nearestMultiple(287, 14), 294)
    }

    private func makeGradientImage(width: Int, height: Int) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let i = (y * width + x) * 4
                pixels[i] = UInt8((x * 255) / max(1, width - 1))
                pixels[i + 1] = UInt8((y * 255) / max(1, height - 1))
                pixels[i + 2] = UInt8((x ^ y) & 0xFF)
                pixels[i + 3] = 255
            }
        }
        let context = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return context.makeImage()!
    }
}
