import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXDA3

@main
struct DA3Tool: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "da3-tool",
        abstract: "Depth Anything 3 inference tool",
        discussion: "Run monocular or multi-view depth estimation on images."
    )

    @Option(name: .shortAndLong, help: "Model config name (da3-small, da3-base, da3-large, da3-giant, da3mono-large)")
    var model: String = "da3mono-large"

    @Option(name: .shortAndLong, help: "Path to model.safetensors weights file")
    var weights: String

    @Option(name: .shortAndLong, help: "Input image path")
    var input: String

    @Option(name: .shortAndLong, help: "Output depth map path (raw float32 binary or .ply for point cloud)")
    var output: String = "depth_output.bin"


    @Option(name: .long, help: "Processing resolution (default 518)")
    var resolution: Int = 518

    @Option(name: .long, help: "Weight dtype: float16 or float32")
    var dtype: String = "float16"

    func run() throws {
        let targetDtype: DType = dtype == "float32" ? .float32 : .float16

        // Load model
        print("Building model '\(model)'...")
        let weightsURL = URL(fileURLWithPath: weights)
        let da3Model = try loadModel(configName: model, weightsURL: weightsURL, dtype: targetDtype)
        print("Model loaded.")

        // Load image
        print("Loading image: \(input)")
        guard let image = loadCGImage(path: input) else {
            print("Error: Could not load image at \(input)")
            throw ExitCode.failure
        }
        print("Image size: \(image.width) x \(image.height)")

        // Preprocess
        let processor = ImageProcessor(processRes: resolution)
        let inputTensor = processor(image) // [1, H', W', 3]
        let h = inputTensor.dim(1)
        let w = inputTensor.dim(2)
        print("Processed size: \(w) x \(h)")

        // Add view dimension: [1, 1, H', W', 3]
        let inputWithViews = inputTensor.expandedDimensions(axis: 1)
            .asType(targetDtype)

        // Inference
        print("Running inference...")
        let startTime = CFAbsoluteTimeGetCurrent()
        let results = da3Model(inputWithViews)
        eval(results)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print(String(format: "Inference time: %.2f s", elapsed))

        // Report results
        for (key, value) in results.sorted(by: { $0.key < $1.key }) {
            print("  \(key): \(value.shape) dtype=\(value.dtype)")
        }

        // Save depth map
        if let depth = results["depth"] {
            let depthFloat = depth.asType(.float32)
            eval(depthFloat)

            if output.hasSuffix("ply") {
                try savePLY(depthFloat, to: output, width: w, height: h)
            } else {
                let outputURL = URL(fileURLWithPath: output)
                let count = depthFloat.size
                let data = depthFloat.asData(access: .copy).data
                try data.write(to: outputURL)
                print("Saved depth map (\(count) floats) to \(output)")
            }
        }
    }
}

func savePLY(_ depth: MLXArray, to path: String, width: Int, height: Int) throws {
    let reshaped = depth.reshaped([height, width])
    reshaped.eval()

    var output = """
    ply
    format ascii 1.0
    element vertex \(depth.size)
    property float x
    property float y
    property float z
    end_header

    """

    let flatDepth = reshaped.asData(access: .copy).data
    let floats = reshaped.asArray(Float.self)

    for i in 0..<height {
        let rowOffset = i * width
        for j in 0..<width {
            output += "\(j) \(i) \(floats[rowOffset + j])\n"
        }
    }

    try output.write(toFile: path, atomically: true, encoding: .utf8)
    print("Saved PLY point cloud (\(depth.size) points) to \(path)")
}

/// Load a CGImage from a file path.
func loadCGImage(path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let source = CGImageSourceCreateWithURL(url, nil),
          CGImageSourceGetCount(source) > 0
    else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}
