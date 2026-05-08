import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXDA3

@main
struct DA3Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "da3-bench",
        abstract: "Benchmark MLX Swift Depth Anything 3 inference."
    )

    @Option(name: .shortAndLong, help: "Model config name.")
    var model: String = "da3mono-large"

    @Option(name: .shortAndLong, help: "Path to model.safetensors or a directory containing it.")
    var weights: String

    @Option(name: .shortAndLong, help: "Input image path.")
    var input: String

    @Option(name: .long, help: "Processing resolution.")
    var resolution: Int = 518

    @Option(name: .long, help: "Weight dtype: float16 or float32.")
    var dtype: String = "float16"

    @Option(name: .long, help: "Warmup iterations.")
    var warmup: Int = 2

    @Option(name: .long, help: "Measured iterations.")
    var iterations: Int = 10

    @Flag(name: .long, help: "Include preprocessing in measured time.")
    var includePreprocess: Bool = false

    @Option(name: .long, help: "Comma-separated outputs: all, depth, depth_conf, ray, ray_conf, sky.")
    var outputs: String = "all"

    func run() throws {
        let targetDtype: DType = dtype == "float32" ? .float32 : .float16
        let requestedOutputs = try parseOutputs(outputs)
        guard let image = loadCGImage(path: input) else {
            throw ValidationError("Could not load image at \(input)")
        }

        let loadStart = CFAbsoluteTimeGetCurrent()
        let pipeline = try DepthAnything3Pipeline.fromPretrained(
            weights,
            configName: model,
            processRes: resolution,
            dtype: targetDtype
        )
        let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart

        let preprocessed = pipeline.preprocess(image)
        eval(preprocessed)

        for _ in 0..<max(0, warmup) {
            if includePreprocess {
                _ = pipeline(image, outputs: requestedOutputs)
            } else {
                _ = pipeline.predict(preprocessed, outputs: requestedOutputs)
            }
        }

        var times: [Double] = []
        for _ in 0..<max(1, iterations) {
            let start = CFAbsoluteTimeGetCurrent()
            if includePreprocess {
                _ = pipeline(image, outputs: requestedOutputs)
            } else {
                _ = pipeline.predict(preprocessed, outputs: requestedOutputs)
            }
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        let sorted = times.sorted()
        let median = sorted[sorted.count / 2]
        let minTime = sorted.first ?? 0
        let maxTime = sorted.last ?? 0

        print("backend=mlx-swift")
        print("model=\(model)")
        print("dtype=\(dtype)")
        print("input=\(input)")
        print("source_size=\(image.width)x\(image.height)")
        print("processed_shape=\(preprocessed.shape)")
        print(String(format: "load_s=%.6f", loadSeconds))
        print("include_preprocess=\(includePreprocess)")
        print("outputs=\(outputs)")
        print("warmup=\(warmup)")
        print("iterations=\(times.count)")
        print(String(format: "mean_s=%.6f", mean))
        print(String(format: "median_s=%.6f", median))
        print(String(format: "min_s=%.6f", minTime))
        print(String(format: "max_s=%.6f", maxTime))
    }
}

private func parseOutputs(_ value: String) throws -> DA3Outputs {
    var parsed: DA3Outputs = []
    for part in value.split(separator: ",") {
        switch part.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "all":
            parsed.formUnion(.all)
        case "depth":
            parsed.insert(.depth)
        case "depth_conf", "depth-confidence", "depthconfidence":
            parsed.insert(.depthConfidence)
        case "ray":
            parsed.insert(.ray)
        case "ray_conf", "ray-confidence", "rayconfidence":
            parsed.insert(.rayConfidence)
        case "sky":
            parsed.insert(.sky)
        case "":
            break
        default:
            throw ValidationError("Unknown output '\(part)'")
        }
    }
    return parsed.isEmpty ? .all : parsed
}

private func loadCGImage(path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          CGImageSourceGetCount(source) > 0
    else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}
