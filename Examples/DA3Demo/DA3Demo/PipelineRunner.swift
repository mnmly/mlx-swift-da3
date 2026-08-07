import Combine
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXDA3
import MLXDA3SALAD
import MLXDA3Streaming
import SwiftUI

/// Holds pipeline configuration + run state for the demo UI.
///
/// Demonstrates every public Phase 1 + Phase 2 API:
/// - Single-image:  `DepthAnything3Pipeline(...)`
/// - Streaming:     `StreamingPipeline.predict(images:)`
/// - Loop closure:  `LoopDetector.detect(images:)`,
///                  `StreamingPipeline.computeLoopMeasurement(...)`,
///                  `StreamingPipeline.predict(images:loopConstraints:)`
/// - PLY save:      `PLYWriter.saveConfidentPointCloud(...)`,
///                  `PLYWriter.mergePLYFiles(...)`
@MainActor
final class PipelineRunner: ObservableObject {

    enum Mode: String, CaseIterable, Identifiable {
        case singleImage = "Single image"
        case streaming = "Video streaming"
        var id: Self { self }
    }

    enum RunState: Equatable {
        case idle
        case extractingFrames(Int)        // total found
        case loadingWeights
        case runningSingleImage
        case runningStreaming
        case detectingLoops
        case measuringLoops(Int, Int)     // i out of N
        case refining
        case writingPLY
        case done
        case failed(String)
    }

    // MARK: - Inputs (bound to UI form)

    @Published var mode: Mode = .singleImage
    @Published var imageURL: URL?
    @Published var videoURL: URL?
    @Published var da3WeightsURL: URL?
    @Published var saladWeightsURL: URL?
    @Published var enableLoopClosure: Bool = false
    @Published var fps: Double = 5.0
    @Published var maxFrames: Int = 32
    @Published var chunkSize: Int = 8
    @Published var overlap: Int = 4
    /// Single-image config: model variant config name (e.g. "da3-large", "da3-giant").
    @Published var singleImageConfig: String = "da3-large"
    @Published var processRes: Int = 518

    // MARK: - Outputs

    @Published var state: RunState = .idle
    @Published var frameCount: Int = 0
    @Published var detectedLoopCount: Int = 0
    @Published var poseCount: Int = 0
    @Published var lastError: String?
    @Published var finalPredictionPoses: MLXArray?
    @Published var finalPredictionIntrinsics: MLXArray?
    @Published var savedPLYPath: String?

    // Single-image outputs
    @Published var singleDepth: MLXArray?           // [H, W] float32
    @Published var singleSourceImage: NSImage?      // for side-by-side display
    @Published var singleProcessedHeight: Int = 0
    @Published var singleProcessedWidth: Int = 0

    /// Last set of frames decoded — kept around so PLY save can reuse them
    /// without re-decoding. Set to `nil` after a "Reset".
    private var lastFrames: [CGImage]?

    // MARK: - Run (dispatch)

    func run() async {
        switch mode {
        case .singleImage: await runSingleImage()
        case .streaming:   await runStreaming()
        }
    }

    private func runSingleImage() async {
        guard let imageURL, let da3WeightsURL else {
            state = .failed("Pick an image and DA3 weights first.")
            return
        }
        lastError = nil
        // Reset prior outputs so the depth viewer only shows current run.
        finalPredictionPoses = nil
        finalPredictionIntrinsics = nil
        singleDepth = nil
        singleSourceImage = NSImage(contentsOf: imageURL)

        do {
            state = .loadingWeights
            let pipeline = try DepthAnything3Pipeline.fromPretrained(
                da3WeightsURL.path,
                configName: singleImageConfig,
                processRes: processRes,
                dtype: .float16
            )

            // Decode the source image
            guard let cg = loadCGImage(at: imageURL) else {
                state = .failed("Could not decode image at \(imageURL.lastPathComponent)")
                return
            }

            state = .runningSingleImage
            let pred = pipeline(cg, outputs: [.depth])

            guard var depth = pred.depth else {
                state = .failed("Model did not return depth output.")
                return
            }
            // depth shape is typically [1, 1, H, W] or [1, S, H, W] — squeeze to [H, W].
            while depth.ndim > 2 && depth.dim(0) == 1 {
                depth = depth.squeezed(axis: 0)
            }
            eval(depth)
            singleDepth = depth.asType(.float32)
            singleProcessedHeight = pred.processedHeight
            singleProcessedWidth = pred.processedWidth
            state = .done
        } catch {
            state = .failed("\(error.localizedDescription)")
            lastError = "\(error)"
        }
    }

    private func runStreaming() async {
        guard let videoURL, let da3WeightsURL else {
            state = .failed("Pick a video and DA3 weights first.")
            return
        }
        if enableLoopClosure && saladWeightsURL == nil {
            state = .failed("Loop closure requires SALAD weights (.safetensors).")
            return
        }
        state = .extractingFrames(0)
        lastError = nil
        // Reset single-image outputs
        singleDepth = nil
        singleSourceImage = nil

        do {
            // 1. Decode video → CGImages
            let frames = try await FrameExtractor.extractFrames(
                from: videoURL, fps: fps, maxFrames: maxFrames
            )
            frameCount = frames.count
            state = .extractingFrames(frames.count)
            guard frames.count >= chunkSize else {
                state = .failed("Need ≥\(chunkSize) frames; got \(frames.count). Increase fps or pick a longer clip.")
                return
            }
            self.lastFrames = frames

            // 2. Load DA3 model + build StreamingPipeline
            state = .loadingWeights
            var cfg = StreamingPipeline.Config()
            cfg.chunkSize = chunkSize
            cfg.overlap = overlap
            cfg.resolution = 504
            cfg.dtype = .float16
            cfg.verbose = false

            let model = try DepthAnything3.fromPretrained(
                da3WeightsURL.path, configName: "da3-giant", dtype: cfg.dtype
            )
            let pipeline = StreamingPipeline(model: model, config: cfg)

            // 3. Initial streaming pass
            state = .runningStreaming
            let initial = pipeline.predict(images: frames)

            // 4. Optional loop closure
            var refined = initial
            if enableLoopClosure, let saladWeightsURL {
                state = .detectingLoops
                let detector = try LoopDetector.fromPretrained(saladWeightsURL.path)
                let (_, loops) = detector.detect(images: frames)
                detectedLoopCount = loops.count

                var constraints: [LoopConstraint] = []
                for (i, loop) in loops.enumerated() {
                    state = .measuringLoops(i + 1, loops.count)
                    guard let cA = chunkIndex(for: loop.a, in: initial.chunkRanges),
                          let cB = chunkIndex(for: loop.b, in: initial.chunkRanges),
                          cA != cB
                    else { continue }
                    let framesA = framesForChunk(cA, frames: frames, ranges: initial.chunkRanges)
                    let framesB = framesForChunk(cB, frames: frames, ranges: initial.chunkRanges)
                    if let constraint = pipeline.computeLoopMeasurement(
                        chunkA: initial.perChunk[cA], framesA: framesA, chunkAIdx: cA,
                        chunkB: initial.perChunk[cB], framesB: framesB, chunkBIdx: cB
                    ) {
                        constraints.append(constraint)
                    }
                }

                if !constraints.isEmpty {
                    state = .refining
                    refined = pipeline.predict(images: frames, loopConstraints: constraints)
                }
            }

            poseCount = refined.cameraPosesC2W.dim(0)
            finalPredictionPoses = refined.cameraPosesC2W
            finalPredictionIntrinsics = refined.intrinsicsK
            state = .done
        } catch {
            state = .failed("\(error.localizedDescription)")
            lastError = "\(error)"
        }
    }

    /// Save per-chunk + combined PLY files to a directory.
    func savePLY(to outputDir: URL) async {
        guard let frames = lastFrames,
              let _ = finalPredictionPoses
        else {
            state = .failed("No prediction in memory; run first.")
            return
        }
        state = .writingPLY
        do {
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
            let pcdDir = outputDir.appendingPathComponent("pcd")
            try FileManager.default.createDirectory(at: pcdDir, withIntermediateDirectories: true)

            // Re-run pipeline to get per-chunk world points (we don't store them
            // in the @Published prediction — the SwiftUI vm only keeps poses).
            // Demo simplification: use Swift API again. Real apps would cache.
            guard let videoURL, let da3WeightsURL else { return }
            var cfg = StreamingPipeline.Config()
            cfg.chunkSize = chunkSize
            cfg.overlap = overlap
            cfg.resolution = 504
            cfg.dtype = .float16
            let model = try DepthAnything3.fromPretrained(
                da3WeightsURL.path, configName: "da3-giant", dtype: cfg.dtype
            )
            let pipeline = StreamingPipeline(model: model, config: cfg)
            let pred = pipeline.predict(images: frames)

            // Write per-chunk PLYs (untransformed; chunk 0 in its own frame)
            for (i, chunk) in pred.perChunk.enumerated() {
                guard let intK = chunk.intrinsics, let extK = chunk.extrinsics else { continue }
                let pm = PointMaps.depthToWorldPoints(
                    depth: chunk.depth, intrinsics: intK, extrinsicsW2C: extK
                )
                eval(pm)
                let confMean = MLX.mean(chunk.conf).item(Float.self)
                let thresh = confMean * 0.75
                try PLYWriter.saveConfidentPointCloud(
                    points: pm, colors: chunk.processedImages, confs: chunk.conf,
                    confThreshold: thresh,
                    outputPath: pcdDir.appendingPathComponent("\(i)_pcd.ply").path
                )
            }
            let combined = pcdDir.appendingPathComponent("combined_pcd.ply").path
            try PLYWriter.mergePLYFiles(inputDir: pcdDir.path, outputPath: combined)
            savedPLYPath = combined

            // Also save camera poses + intrinsics for downstream tools.
            let posesInputs = CameraPosesIO.Inputs(
                chunkExtrinsicsW2C: pred.perChunk.map { $0.extrinsics ?? MLXArray.zeros([0, 3, 4], dtype: .float32) },
                chunkIntrinsics:    pred.perChunk.map { $0.intrinsics ?? MLXArray.zeros([0, 3, 3], dtype: .float32) },
                chunkRanges: pred.chunkRanges,
                cumulativeSim3: pred.cumulativeSim3,
                overlap: overlap,
                totalFrames: frames.count
            )
            try CameraPosesIO.savePoses(posesInputs, outputDir: outputDir.path)
            state = .done
        } catch {
            state = .failed("\(error)")
        }
    }

    // MARK: - Helpers

    private func chunkIndex(for frame: Int, in ranges: [(Int, Int)]) -> Int? {
        for (idx, r) in ranges.enumerated() where frame >= r.0 && frame < r.1 {
            return idx
        }
        return nil
    }

    private func framesForChunk(_ idx: Int, frames: [CGImage], ranges: [(Int, Int)]) -> [CGImage] {
        let r = ranges[idx]
        return Array(frames[r.0..<r.1])
    }

    private func loadCGImage(at url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(src) > 0
        else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}
