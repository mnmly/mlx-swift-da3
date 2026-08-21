import ArgumentParser
import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import MLX
import MLXDA3

/// Turn a video into a "colour over disparity" RGBD video plus a sidecar.
///
/// The output frame is twice the source height: colour on top, inverse depth
/// normalized to 0...1 on the bottom. That single-file layout is what a
/// displacement shader wants — one texture, two bands, no sync problem.
///
/// `--static-camera` is the mode for footage where the camera never moves. The
/// depth band is then estimated once, from the per-pixel median of a sample of
/// frames, and repeated for every output frame. See `StaticSceneDepth` for why
/// that beats inferring each frame.
@main
struct DA3Video: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "da3-video",
        abstract: "Video -> colour-over-disparity RGBD video using Depth Anything 3."
    )

    @Option(name: .shortAndLong, help: "Input video.")
    var input: String

    @Option(name: .shortAndLong, help: "Output .mp4 (default: <input>.rgbd.mp4 alongside the input).")
    var output: String?

    @Option(name: .shortAndLong, help: "Model config name.")
    var model: String = "da3-large"

    @Option(name: .shortAndLong, help: "Path to model.safetensors. Required unless --depth-safetensors is given.")
    var weights: String?

    @Option(name: .long, help: "Read depth from a precomputed (N, H, W) float32 safetensors instead of running DA3 — e.g. `vda infer --save-raw` from mlx-swift-VideoDepthAnything. Values are taken to be INVERSE depth already.")
    var depthSafetensors: String?

    @Option(name: .long, help: "Resolution DA3 runs at, long edge. Past ~2072 the ViT is far enough outside its trained token grid that geometry degrades.")
    var resolution: Int = 2072

    @Option(name: .long, help: "Weight dtype: float16 or float32.")
    var dtype: String = "float16"

    @Flag(name: .long, help: "Camera never moves: estimate one depth band from a median of sampled frames.")
    var staticCamera: Bool = false

    @Option(name: .long, help: "Frames sampled for the static-camera median.")
    var samples: Int = 51

    @Option(name: .long, help: "Inverse-depth percentile mapped to 1 (nearest).")
    var nearPercentile: Float = 99.5

    @Option(name: .long, help: "Inverse-depth percentile mapped to 0 (farthest).")
    var farPercentile: Float = 0.5

    @Option(name: .long, help: "Disparity range in source pixels recorded in the sidecar (drives the shader's depth mapping).")
    var disparityMax: Double = 101

    @Option(name: .long, help: "Guided-filter radius, in depth-map pixels, for the upsample to output resolution.")
    var guideRadius: Int = 8

    @Option(name: .long, help: "Guided-filter regularization, in normalized 0...1 units. Larger = smoother; too small transfers the guide's texture into depth.")
    var guideEpsilon: Float = 1e-3

    @Flag(name: .long, help: "Skip the guided upsample and use a plain resize.")
    var noGuide: Bool = false

    @Option(name: .long, help: "Output bitrate in Mbps (video output only).")
    var bitrateMbps: Int = 40

    @Option(name: .long, help: "Store the band as value^(1/gamma). The consumer raises the sampled value to the same power. 1 = off. Only worth it for an 8-bit video band.")
    var bandGamma: Float = 1.0

    @Flag(name: .long, help: "With --static-camera: write the depth map as a single 16-bit PNG instead of a video, and leave the colour clip untouched.")
    var depthImage: Bool = false

    @Flag(name: .long, help: "Per-frame depth, stabilized against the static map: background stays locked, moving subjects get their own depth. Writes a 10-bit depth movie.")
    var depthSequence: Bool = false

    @Option(name: .long, help: "With --depth-sequence: infer every Nth frame and hold in between. Depth drifts slowly, so 2-3 is usually indistinguishable and proportionally faster.")
    var depthStride: Int = 1

    @Option(name: .long, help: "Relative disagreement with the static map below which the static map is kept outright.")
    var agreeBelow: Float = 0.05

    @Option(name: .long, help: "Relative disagreement above which the frame's own depth is used outright.")
    var disagreeAbove: Float = 0.20

    @Option(name: .long, help: "Stop after N source frames (0 = all).")
    var frames: Int = 0

    /// Where each frame's inverse-depth map comes from.
    ///
    /// DA3 runs here, frame by frame. A video model with temporal modules
    /// (Video-Depth-Anything) instead runs its own sliding window over the whole
    /// clip ahead of time, so its output arrives as one tensor and this tool only
    /// normalizes, upsamples and encodes it. Both arrive as `[H, W]` of 1/Z.
    private enum DepthProvider {
        case model(DepthAnything3, ImageProcessor, DType)
        case precomputed(MLXArray)          // [N, H, W], already inverse depth

        /// `[H, W]` inverse depth for one frame. `image` is unused when the maps
        /// were computed elsewhere.
        func inverseMap(frame index: Int, image: CGImage) throws -> MLXArray {
            switch self {
            case let .model(da3, processor, dtype):
                let batch = processor(image).expandedDimensions(axis: 1).asType(dtype)
                guard let depth = da3(batch)["depth"] else {
                    throw VideoError("model produced no depth at frame \(index)")
                }
                return StaticSceneDepth.inverseDepth(depth.asType(.float32).squeezed())
            case let .precomputed(maps):
                guard index < maps.dim(0) else {
                    throw VideoError("frame \(index) is past the \(maps.dim(0)) precomputed maps")
                }
                return maps[index]
            }
        }

        /// Precomputed video-model depth is already temporally consistent, so
        /// reconciling it against a static median would be redundant. It is also
        /// ill-posed: these models pin the sky to exactly 0, and the robust
        /// frame-to-clip scale is a median over `frame / static`, which collapses to
        /// 0 once more than half the pixels are 0 on both sides.
        var needsStabilization: Bool {
            if case .model = self { return true }
            return false
        }
    }

    mutating func run() throws {
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: output ?? Self.defaultOutput(for: inputURL))
        let targetDtype: DType = dtype == "float32" ? .float32 : .float16

        let source = try VideoSource(url: inputURL)
        let frameCount = frames > 0 ? min(frames, source.frameCount) : source.frameCount
        print("""
        \(inputURL.lastPathComponent): \(source.width)x\(source.height), \
        \(frameCount) frames @ \(String(format: "%.3f", source.fps))
        """)

        guard staticCamera else {
            throw ValidationError(
                "only --static-camera is implemented; per-frame depth video is a separate mode"
            )
        }

        let processor = ImageProcessor(processRes: resolution)
        let depth: DepthProvider
        switch (weights, depthSafetensors) {
        case (nil, nil), (.some, .some):
            throw ValidationError("pass exactly one of --weights or --depth-safetensors")
        case let (_, .some(path)):
            let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            guard let maps = loaded["depth"] ?? loaded.sorted(by: { $0.key < $1.key }).first?.value
            else { throw ValidationError("no arrays in \(path)") }
            guard maps.ndim == 3 else {
                throw ValidationError("expected (N, H, W), got \(maps.shape)")
            }
            print("Precomputed depth: \(maps.dim(0)) maps at \(maps.dim(2))x\(maps.dim(1))"
                  + " (already inverse depth)")
            depth = .precomputed(maps.asType(.float32))
        case let (.some(path), _):
            print("Loading \(model)…")
            depth = .model(
                try loadModel(
                    configName: model,
                    weightsURL: URL(fileURLWithPath: try Self.resolveWeights(path)),
                    dtype: targetDtype
                ),
                processor, targetDtype
            )
        }

        // ---- 1. sample frames evenly across the clip -------------------------
        let stride = max(1, frameCount / max(samples, 1))
        let sampleIndices = Array(Swift.stride(from: 0, to: frameCount, by: stride))
        print("Sampling \(sampleIndices.count) frames (every \(stride))…")
        let sampled = try source.frames(at: Set(sampleIndices))

        // ---- 2. depth per sample, medianed into one static band --------------
        var inverseMaps: [MLXArray] = []
        inverseMaps.reserveCapacity(sampled.count)
        let started = CFAbsoluteTimeGetCurrent()
        for (i, image) in sampled.enumerated() {
            let map = try depth.inverseMap(frame: sampleIndices[i], image: image)
            eval(map)
            inverseMaps.append(map)
            if (i + 1) % 10 == 0 || i + 1 == sampled.count {
                let rate = Double(i + 1) / (CFAbsoluteTimeGetCurrent() - started)
                print(String(format: "  %d/%d  %.2f fps", i + 1, sampled.count, rate))
            }
        }

        let median = StaticSceneDepth.temporalMedian(inverseMaps)
        eval(median)
        let spread = StaticSceneDepth.relativeSpread(inverseMaps, median: median)
        print(String(
            format: "Medianed %d maps at %dx%d (per-frame spread was %.1f%%)",
            inverseMaps.count, median.dim(1), median.dim(0), spread * 100
        ))
        inverseMaps.removeAll()

        // ---- 3. normalize, then upsample to output resolution -----------------
        // Normalizing BEFORE the guided filter matters: the filter's `epsilon` is a
        // contrast threshold expressed in the units of the data, so a map still in
        // raw inverse-depth units (here ~0.5…4.7) against a 0…1 guide makes epsilon
        // effectively zero, and the filter transfers the guide's *texture* — brush
        // strokes — into the depth. In a shared 0…1 space epsilon means what it says.
        let normalized = StaticSceneDepth.normalize(
            inverseDepth: median, nearPercentile: nearPercentile, farPercentile: farPercentile
        )
        print(String(format: "Inverse depth %.4f (far) … %.4f (near)", normalized.far, normalized.near))

        var band: MLXArray
        if noGuide {
            band = GuidedFilter.resample(normalized.map, height: source.height, width: source.width)
        } else {
            let guide = Self.luma(of: sampled[sampled.count / 2],
                                  width: source.width, height: source.height)
            band = GuidedFilter.upsample(
                lowRes: normalized.map, guide: guide,
                radius: guideRadius, epsilon: guideEpsilon
            )
            print("Guided upsample -> \(source.width)x\(source.height) "
                  + "(radius \(guideRadius), epsilon \(guideEpsilon))")
        }
        band = clip(band, min: 0, max: 1)

        // A gamma curve buys effective codes when the band is stored in 8 bits and the
        // subject sits at one end of the range — the consumer raises the sampled value
        // back to `bandGamma`. It is pointless for the 16-bit still, which has codes to
        // spare.
        if bandGamma != 1 && !depthImage {
            band = pow(band, 1.0 / bandGamma)
            print("Band stored with gamma \(bandGamma); consumer must raise samples to \(bandGamma)")
        }
        eval(band)

        // ---- 4. write the depth, as a sequence, a still, or under the colour ----
        let depthOutput: String
        if depthSequence {
            depthOutput = try writeDepthSequence(
                source: source, frameCount: frameCount, depth: depth,
                staticMap: normalized.map, rawStatic: median,
                near: normalized.near, far: normalized.far, outputURL: outputURL
            )
        } else if depthImage {
            // With a locked-off camera the band is the same in every frame, so a video
            // of it is 752 copies of one image, quantized to 8 bits and put through
            // chroma subsampling and a lossy codec for no reason. A single 16-bit PNG is
            // exact, ~250x more codes, and leaves the colour clip untouched.
            let pngURL = outputURL
                .deletingPathExtension().deletingPathExtension()
                .appendingPathExtension("depth.png")
            try Self.writeGray16PNG(band, to: pngURL)
            depthOutput = pngURL.lastPathComponent
            print("Depth still -> \(depthOutput) (16-bit, \(source.width)x\(source.height))")
        } else {
            let bandBytes = Self.grayBytes(band, width: source.width, height: source.height)
            let writer = try VideoWriter(
                url: outputURL, width: source.width, height: source.height * 2,
                fps: source.fps, bitrate: bitrateMbps * 1_000_000
            )
            print("Writing \(outputURL.lastPathComponent) (\(source.width)x\(source.height * 2))…")
            try source.forEachFrame(limit: frameCount) { index, pixelBuffer in
                try writer.append(colour: pixelBuffer, depthBand: bandBytes, frameIndex: index)
                if (index + 1) % 100 == 0 {
                    print("  \(index + 1)/\(frameCount)")
                }
            }
            try writer.finish()
            depthOutput = outputURL.lastPathComponent
        }

        // ---- 5. sidecar -------------------------------------------------------
        let sidecar: [String: Any] = [
            "source": inputURL.lastPathComponent,
            // `video` is what the consumer samples: the stacked clip, or — for a depth
            // still — the untouched source, with the depth in `depthImage`.
            "video": (depthImage || depthSequence)
                ? inputURL.lastPathComponent : outputURL.lastPathComponent,
            "depthImage": depthImage ? depthOutput : "",
            "depthVideo": depthSequence ? depthOutput : "",
            "depthVideoBits": depthSequence ? 10 : 0,
            "bandGamma": depthImage ? 1.0 : Double(bandGamma),
            "bandBits": depthImage ? 16 : (depthSequence ? 10 : 8),
            "layout": depthImage ? "colour-plus-depth-still"
                : (depthSequence ? "colour-plus-depth-movie" : "colour-over-disparity"),
            "width": source.width,
            "height": source.height,
            "eyeWidth": source.width,
            "eyeHeight": source.height,
            "frames": frameCount,
            "fps": source.fps,
            "disparityMin": 0.0,
            "disparityMax": disparityMax,
            "depthSource": depthSafetensors == nil ? "depth-anything-3" : "precomputed",
            "depthModel": depthSafetensors.map { URL(fileURLWithPath: $0).lastPathComponent }
                ?? model,
            "depthResolution": depthSafetensors == nil ? resolution : median.dim(1),
            "inverseDepthNear": Double(normalized.near),
            "inverseDepthFar": Double(normalized.far),
            "nearPercentile": Double(nearPercentile),
            "farPercentile": Double(farPercentile),
            "staticCamera": !depthSequence,
            "staticSamples": sampleIndices.count,
            "guided": !noGuide,
        ]
        let sidecarURL = outputURL
            .deletingPathExtension()                     // <name>.rgbd
            .deletingPathExtension()                     // <name>
            .appendingPathExtension("stereodepth.json")
        let json = try JSONSerialization.data(
            withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys]
        )
        try json.write(to: sidecarURL)
        print("Sidecar -> \(sidecarURL.lastPathComponent)")
    }

    // MARK: - Per-frame depth, stabilized against the static map

    /// Walk every frame: infer its depth, reconcile it with the clip's static
    /// geometry, upsample it guided by that frame's own colour, and write it to a
    /// 10-bit depth movie.
    private func writeDepthSequence(
        source: VideoSource, frameCount: Int, depth: DepthProvider,
        staticMap: MLXArray, rawStatic: MLXArray,
        near: Float, far: Float, outputURL: URL
    ) throws -> String {
        let depthURL = outputURL
            .deletingPathExtension().deletingPathExtension()
            .appendingPathExtension("depth.mov")
        let writer = try DepthVideoWriter(
            url: depthURL, width: source.width, height: source.height,
            fps: source.fps, bitrate: bitrateMbps * 1_000_000
        )
        if depth.needsStabilization {
            print("Per-frame depth -> \(depthURL.lastPathComponent) "
                  + "(10-bit, stride \(depthStride), agree \(agreeBelow)…\(disagreeAbove))")
        } else {
            print("Per-frame depth -> \(depthURL.lastPathComponent) "
                  + "(10-bit, stride \(depthStride), stabilization skipped: "
                  + "precomputed depth is already temporally consistent)")
        }

        let context = CIContext(options: [.useSoftwareRenderer: false])
        let span = max(near - far, 1e-6)
        var held: MLXArray? = nil            // last inferred low-res map, for stride > 1
        var scales: [Float] = []
        var movingFractions: [Float] = []
        let started = CFAbsoluteTimeGetCurrent()

        try source.forEachFrame(limit: frameCount) { index, pixelBuffer in
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                throw VideoError("could not convert frame \(index)")
            }

            if index % depthStride == 0 || held == nil {
                held = try depth.inverseMap(frame: index, image: cgImage)
            }

            let frameMap: MLXArray
            if depth.needsStabilization {
                let reconciled = StaticSceneDepth.reconcile(
                    frame: held!, staticMap: rawStatic,
                    agreeBelow: agreeBelow, disagreeAbove: disagreeAbove
                )
                scales.append(reconciled.scale)
                movingFractions.append(reconciled.movingFraction)
                frameMap = reconciled.map
            } else {
                frameMap = held!
            }

            // One global mapping for the whole clip — a per-frame range would make the
            // depth breathe again, which is the thing this mode exists to remove.
            let lowRes = clip((frameMap - far) / span, min: 0, max: 1)
            var band: MLXArray
            if noGuide {
                band = GuidedFilter.resample(lowRes, height: source.height, width: source.width)
            } else {
                let guide = Self.luma(of: cgImage, width: source.width, height: source.height)
                band = GuidedFilter.upsample(
                    lowRes: lowRes, guide: guide, radius: guideRadius, epsilon: guideEpsilon
                )
            }
            band = clip(band, min: 0, max: 1)
            if bandGamma != 1 { band = pow(band, 1.0 / bandGamma) }
            try writer.append(Self.tenBitCodes(band), frameIndex: index)

            if (index + 1) % 25 == 0 || index + 1 == frameCount {
                let rate = Double(index + 1) / (CFAbsoluteTimeGetCurrent() - started)
                let eta = Double(frameCount - index - 1) / max(rate, 1e-6)
                print(String(format: "  %d/%d  %.2f fps  eta %.0fs",
                             index + 1, frameCount, rate, eta))
            }
        }
        try writer.finish()

        if !scales.isEmpty {
            let moving = movingFractions.reduce(0, +) / Float(movingFractions.count)
            print(String(
                format: "Scale correction %.3f…%.3f, %.1f%% of pixels took their own depth",
                scales.min()!, scales.max()!, moving * 100
            ))
        }
        _ = staticMap
        return depthURL.lastPathComponent
    }

    /// 0...1 map -> 10-bit codes.
    private static func tenBitCodes(_ map: MLXArray) -> [UInt16] {
        let quantized = clip(MLX.round(map * 1023), min: 0, max: 1023).asType(.uint16)
        eval(quantized)
        return quantized.asArray(UInt16.self)
    }

    // MARK: - Helpers

    private static func defaultOutput(for input: URL) -> String {
        input.deletingPathExtension().appendingPathExtension("rgbd.mp4").path
    }

    private static func resolveWeights(_ path: String) throws -> String {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
            throw ValidationError("weights not found: \(path)")
        }
        if !isDirectory.boolValue { return path }
        let candidate = (path as NSString).appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: candidate) else {
            throw ValidationError("no model.safetensors under \(path)")
        }
        return candidate
    }

    /// Rec.709 luma in 0...1 at the given size — the guided filter's guide.
    private static func luma(of image: CGImage, width: Int, height: Int) -> MLXArray {
        let rgb = DA3ImagePreprocessing.decodeRGB(image) // [H, W, 3] 0...255
        let resized = rgb.dim(0) == height && rgb.dim(1) == width
            ? rgb
            : DA3ImagePreprocessing.resize(rgb, height: height, width: width, method: .area)
        let weights = MLXArray([Float(0.2126), 0.7152, 0.0722], [1, 1, 3])
        return (resized * weights).sum(axis: -1) / 255.0
    }

    /// 0...1 map -> a 16-bit grayscale PNG. 65536 codes, no chroma subsampling, no
    /// YUV range squeeze, no lossy codec — the depth arrives as computed.
    private static func writeGray16PNG(_ map: MLXArray, to url: URL) throws {
        let height = map.dim(0)
        let width = map.dim(1)
        let quantized = clip(MLX.round(map * 65535), min: 0, max: 65535).asType(.uint16)
        eval(quantized)
        var samples: [UInt16] = quantized.asArray(UInt16.self)

        let data = samples.withUnsafeMutableBytes { raw in
            CFDataCreate(nil, raw.baseAddress!.assumingMemoryBound(to: UInt8.self), raw.count)!
        }
        guard let provider = CGDataProvider(data: data),
              let image = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 16, bitsPerPixel: 16,
                  bytesPerRow: width * 2,
                  space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Little.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false,
                  intent: .defaultIntent
              )
        else { throw VideoError("could not build 16-bit image") }

        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else { throw VideoError("could not create PNG at \(url.path)") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw VideoError("could not write \(url.lastPathComponent)")
        }
    }

    /// 0...1 map -> one byte per pixel.
    private static func grayBytes(_ map: MLXArray, width: Int, height: Int) -> [UInt8] {
        let scaled = clip(MLX.round(map * 255), min: 0, max: 255).asType(.uint8)
        eval(scaled)
        return scaled.asArray(UInt8.self)
    }
}
