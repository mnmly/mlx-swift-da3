import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXSALAD
import MLXDA3Streaming

@main
struct DA3LoopTool: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "da3-loop-tool",
        abstract: "SALAD loop closure detection over a directory of frames."
    )

    @Option(name: .shortAndLong, help: "Path to dino_salad.safetensors (converted from dino_salad.ckpt).")
    var weights: String

    @Option(name: .long, help: "Directory of input frames (jpg/png).")
    var imageDir: String

    @Option(name: .long, help: "Output txt path. Format mirrors python `loop_closures.txt`.")
    var output: String?

    @Option(name: .long, help: "Output safetensors path with descriptors + loop list (for parity tests).")
    var outputSafetensors: String?

    @Option(name: .long, help: "Image size H,W (default 336,336).")
    var imageSize: String = "336,336"

    @Option(name: .long, help: "Batch size for descriptor extraction.")
    var batchSize: Int = 32

    @Option(name: .long, help: "Cosine similarity threshold for accepting a loop.")
    var similarityThreshold: Float = 0.85

    @Option(name: .long, help: "Top-K nearest neighbours per query frame.")
    var topK: Int = 5

    @Option(name: .long, help: "Minimum frame distance for a candidate to qualify.")
    var minFrameDistance: Int = 10

    @Flag(name: .long, help: "Apply Non-Maximum Suppression on the frame distance.")
    var noNMS: Bool = false

    @Option(name: .long, help: "NMS frame-distance threshold.")
    var nmsThreshold: Int = 25

    @Option(name: .long, help: "Limit number of frames (0 = all).")
    var limit: Int = 0

    func run() throws {
        let dims = imageSize.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) ?? 0 }
        guard dims.count == 2, dims[0] > 0, dims[1] > 0 else {
            throw ValidationError("--image-size must be 'H,W' with positive ints")
        }

        var paths = try ImageDirectory.listImagePaths(in: imageDir)
        if limit > 0, paths.count > limit { paths = Array(paths.prefix(limit)) }
        print("Found \(paths.count) frames")

        var cfg = LoopDetector.Config()
        cfg.imageSize = (height: dims[0], width: dims[1])
        cfg.batchSize = batchSize
        cfg.similarityThreshold = similarityThreshold
        cfg.topK = topK
        cfg.minFrameDistance = minFrameDistance
        cfg.useNMS = !noNMS
        cfg.nmsThreshold = nmsThreshold

        print("Loading SALAD weights...")
        let detector = try LoopDetector.fromPretrained(weights, config: cfg)

        print("Decoding \(paths.count) frames...")
        let images: [CGImage] = try paths.map { try ImageDirectory.loadCGImage(path: $0) }

        let t0 = CFAbsoluteTimeGetCurrent()
        let (descriptors, loops) = detector.detect(images: images)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "Detection done in %.2fs (n=%d, descriptor=%@)", dt, descriptors.dim(0), "\(descriptors.shape)"))
        print("Found \(loops.count) loop pairs")
        for (i, loop) in loops.prefix(10).enumerated() {
            print("  [\(i)] (\(loop.a), \(loop.b)) sim=\(String(format: "%.4f", loop.similarity))")
        }

        if let path = output {
            var text = "# Loop Detection Results (index1, index2, similarity)\n"
            if cfg.useNMS { text += "# NMS filtering applied, threshold: \(cfg.nmsThreshold)\n" }
            text += "\n# Loop pairs:\n"
            for loop in loops {
                text += "\(loop.a), \(loop.b), \(String(format: "%.4f", loop.similarity))\n"
            }
            text += "\n# Image path list:\n"
            for (i, p) in paths.enumerated() {
                text += "# \(i): \(p)\n"
            }
            try text.write(toFile: path, atomically: true, encoding: .utf8)
            print("Wrote \(loops.count) pairs → \(path)")
        }

        if let stPath = outputSafetensors {
            // Pack descriptors + loop pairs (a, b, sim) into safetensors for parity testing.
            var loopFlat = [Float](repeating: 0, count: loops.count * 3)
            for (i, l) in loops.enumerated() {
                loopFlat[i * 3 + 0] = Float(l.a)
                loopFlat[i * 3 + 1] = Float(l.b)
                loopFlat[i * 3 + 2] = l.similarity
            }
            let pairsArr = MLXArray(loopFlat, [loops.count, 3])
            try save(arrays: [
                "descriptors": descriptors,
                "loop_pairs": pairsArr,
            ], url: URL(fileURLWithPath: stPath))
            print("Wrote safetensors → \(stPath)")
        }
    }
}
