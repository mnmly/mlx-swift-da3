import MLX
import MLXNN

/// Standard MLP with GELU activation.
public class Mlp: Module {
    @ModuleInfo(key: "fc1") private var fc1: Linear
    @ModuleInfo(key: "fc2") private var fc2: Linear
    private let act: GELU

    public init(
        inFeatures: Int,
        hiddenFeatures: Int? = nil,
        outFeatures: Int? = nil,
        bias: Bool = true
    ) {
        let out = outFeatures ?? inFeatures
        let hidden = hiddenFeatures ?? inFeatures
        self._fc1.wrappedValue = Linear(inFeatures, hidden, bias: bias)
        self.act = GELU()
        self._fc2.wrappedValue = Linear(hidden, out, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(act(fc1(x)))
    }
}

/// SwiGLU feed-forward network (used by ViT-Giant).
public class SwiGLUFFN: Module {
    @ModuleInfo(key: "w12") private var w12: Linear
    @ModuleInfo(key: "w3") private var w3: Linear

    public init(
        inFeatures: Int,
        hiddenFeatures: Int? = nil,
        outFeatures: Int? = nil,
        bias: Bool = true
    ) {
        let out = outFeatures ?? inFeatures
        let hidden = hiddenFeatures ?? inFeatures
        // Match SwiGLUFFNFused sizing: (2/3 * hidden) rounded to multiple of 8
        let adjustedHidden = ((Int(Float(hidden) * 2 / 3) + 7) / 8) * 8

        self._w12.wrappedValue = Linear(inFeatures, 2 * adjustedHidden, bias: bias)
        self._w3.wrappedValue = Linear(adjustedHidden, out, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (x1, x2) = w12(x).split(axis: -1)
        let hidden = silu(x1) * x2
        return w3(hidden)
    }
}
