import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXDA3
import MLXDA3Streaming

@main
struct DA3StreamingBench: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "da3-streaming-bench",
        abstract: "Benchmark MLX Swift DA3 streaming pipeline (chunk inference + Sim(3) alignment + pose accumulation)."
    )

    @Option(name: .shortAndLong, help: "Model config name.")
    var model: String = "da3-giant"

    @Option(name: .shortAndLong, help: "Path to model.safetensors.")
    var weights: String

    @Option(name: .long, help: "Directory of input frames (jpg/png).")
    var imageDir: String

    @Option(name: .long, help: "Number of frames per chunk.")
    var chunkSize: Int = 4

    @Option(name: .long, help: "Number of overlap frames between chunks.")
    var overlap: Int = 2

    @Option(name: .long, help: "Limit number of frames (0 = all).")
    var limit: Int = 8

    @Option(name: .long, help: "Processing resolution.")
    var resolution: Int = 504

    @Option(name: .long, help: "Weight dtype: float16 or float32.")
    var dtype: String = "float16"

    @Option(name: .long, help: "Warmup iterations.")
    var warmup: Int = 1

    @Option(name: .long, help: "Measured iterations.")
    var iterations: Int = 5

    @Flag(name: .long, help: "Include image-load (CGImage decode) in measured time.")
    var includeLoad: Bool = false

    func run() throws {
        let targetDtype: DType = dtype == "float32" ? .float32 : .float16

        var paths = try ImageDirectory.listImagePaths(in: imageDir)
        if limit > 0, paths.count > limit { paths = Array(paths.prefix(limit)) }

        let loadStart = CFAbsoluteTimeGetCurrent()
        var cfg = StreamingPipeline.Config()
        cfg.chunkSize = chunkSize
        cfg.overlap = overlap
        cfg.resolution = resolution
        cfg.dtype = targetDtype
        cfg.verbose = false

        let weightsResolved = try resolveWeightsPath(weights)
        let loadedModel = try loadModel(configName: model, weightsURL: URL(fileURLWithPath: weightsResolved), dtype: targetDtype)
        let pipeline = StreamingPipeline(model: loadedModel, config: cfg)
        let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart

        // Pre-decode images outside the timed loop (unless --include-load).
        let preloaded: [CGImage] = try paths.map { try ImageDirectory.loadCGImage(path: $0) }

        func runOnce() throws -> StreamingPrediction {
            let images: [CGImage]
            if includeLoad {
                images = try paths.map { try ImageDirectory.loadCGImage(path: $0) }
            } else {
                images = preloaded
            }
            let pred = pipeline.predict(images: images)
            // Force materialization (matches PyTorch synchronize() in the reference bench).
            eval(pred.cameraPosesC2W, pred.intrinsicsK)
            return pred
        }

        for _ in 0..<max(0, warmup) {
            _ = try runOnce()
        }

        var times: [Double] = []
        for _ in 0..<max(1, iterations) {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try runOnce()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        let sorted = times.sorted()
        let median = sorted[sorted.count / 2]
        let minT = sorted.first ?? 0
        let maxT = sorted.last ?? 0

        print("backend=mlx-swift")
        print("pipeline=streaming")
        print("model=\(model)")
        print("dtype=\(dtype)")
        print("image_dir=\(imageDir)")
        print("frames=\(paths.count)")
        print("chunk_size=\(chunkSize)")
        print("overlap=\(overlap)")
        print("resolution=\(resolution)")
        print(String(format: "load_s=%.6f", loadSeconds))
        print("include_load=\(includeLoad)")
        print("warmup=\(warmup)")
        print("iterations=\(times.count)")
        print(String(format: "mean_s=%.6f", mean))
        print(String(format: "median_s=%.6f", median))
        print(String(format: "min_s=%.6f", minT))
        print(String(format: "max_s=%.6f", maxT))
    }
}

/// Accept either a direct .safetensors path or a directory containing it.
private func resolveWeightsPath(_ path: String) throws -> String {
    let fm = FileManager.default
    var isDir: ObjCBool = false
    guard fm.fileExists(atPath: path, isDirectory: &isDir) else {
        throw ValidationError("weights path not found: \(path)")
    }
    if !isDir.boolValue { return path }
    let candidate = (path as NSString).appendingPathComponent("model.safetensors")
    guard fm.fileExists(atPath: candidate) else {
        throw ValidationError("no model.safetensors under \(path)")
    }
    return candidate
}
