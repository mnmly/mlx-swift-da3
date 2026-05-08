import CoreGraphics
import Foundation
import MLX

public struct DepthAnything3Prediction {
    public let outputs: [String: MLXArray]
    public let processedHeight: Int
    public let processedWidth: Int

    public init(outputs: [String: MLXArray], processedHeight: Int, processedWidth: Int) {
        self.outputs = outputs
        self.processedHeight = processedHeight
        self.processedWidth = processedWidth
    }

    public var depth: MLXArray? {
        outputs["depth"]
    }
}

public struct DepthAnything3Pipeline {
    public let model: DepthAnything3
    public let imageProcessor: ImageProcessor
    public let dtype: DType

    public init(
        model: DepthAnything3,
        imageProcessor: ImageProcessor = ImageProcessor(),
        dtype: DType = .float16
    ) {
        self.model = model
        self.imageProcessor = imageProcessor
        self.dtype = dtype
    }

    public func preprocess(_ image: CGImage) -> MLXArray {
        imageProcessor(image).expandedDimensions(axis: 1).asType(dtype)
    }

    public func predict(_ input: MLXArray, outputs requestedOutputs: DA3Outputs = .all) -> [String: MLXArray] {
        let outputs = requestedOutputs == .all ? model(input) : model(input, outputs: requestedOutputs)
        eval(outputs)
        return outputs
    }

    public func callAsFunction(
        _ image: CGImage,
        outputs requestedOutputs: DA3Outputs = .all
    ) -> DepthAnything3Prediction {
        let input = preprocess(image)
        let outputs = predict(input, outputs: requestedOutputs)
        return DepthAnything3Prediction(
            outputs: outputs,
            processedHeight: input.dim(2),
            processedWidth: input.dim(3)
        )
    }
}

public extension DepthAnything3Pipeline {
    static func fromPretrained(
        _ path: String,
        configName: String? = nil,
        processRes: Int = 518,
        dtype: DType = .float16
    ) throws -> DepthAnything3Pipeline {
        let model = try DepthAnything3.fromPretrained(path, configName: configName, dtype: dtype)
        return DepthAnything3Pipeline(
            model: model,
            imageProcessor: ImageProcessor(processRes: processRes),
            dtype: dtype
        )
    }
}

public enum MLXDA3 {
    public static func fromPretrained(
        _ path: String,
        configName: String? = nil,
        processRes: Int = 518,
        dtype: DType = .float16
    ) throws -> DepthAnything3Pipeline {
        try DepthAnything3Pipeline.fromPretrained(
            path,
            configName: configName,
            processRes: processRes,
            dtype: dtype
        )
    }
}
