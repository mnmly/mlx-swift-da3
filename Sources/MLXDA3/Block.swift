import MLX
import MLXNN

/// Per-dimension learnable scaling (stabilizes ViT training).
public class LayerScale: Module {
    @ParameterInfo(key: "gamma") private var gamma: MLXArray

    public init(_ dim: Int, initValues: Float = 1e-5) {
        self._gamma = ParameterInfo(wrappedValue: MLXArray.full([dim], values: MLXArray(initValues)), key: "gamma")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

/// Protocol for FFN modules callable with a single MLXArray.
protocol FFNLayer: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}
extension Mlp: FFNLayer {}
extension SwiGLUFFN: FFNLayer {}

/// Pre-norm transformer block.
public class DA3Block: Module {
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "attn") private var attn: DA3Attention
    @ModuleInfo(key: "ls1") private var ls1: LayerScale?
    @ModuleInfo(key: "norm2") private var norm2: LayerNorm
    @ModuleInfo(key: "mlp") private var mlp: FFNLayer
    @ModuleInfo(key: "ls2") private var ls2: LayerScale?

    public init(
        dim: Int,
        numHeads: Int,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        projBias: Bool = true,
        ffnBias: Bool = true,
        initValues: Float? = 1.0,
        qkNorm: Bool = false,
        rope: RotaryPositionEmbedding2D? = nil,
        useSwiglu: Bool = false,
        lnEps: Float = 1e-6
    ) {
        let mlpHidden = Int(Float(dim) * mlpRatio)

        let ffn: FFNLayer = useSwiglu
            ? SwiGLUFFN(inFeatures: dim, hiddenFeatures: mlpHidden, bias: ffnBias)
            : Mlp(inFeatures: dim, hiddenFeatures: mlpHidden, bias: ffnBias)

        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: lnEps)
        self._attn.wrappedValue = DA3Attention(
            dim: dim, numHeads: numHeads,
            qkvBias: qkvBias, projBias: projBias,
            qkNorm: qkNorm, rope: rope
        )
        self._ls1.wrappedValue = initValues.map { LayerScale(dim, initValues: $0) }
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: lnEps)
        self._mlp.wrappedValue = ffn
        self._ls2.wrappedValue = initValues.map { LayerScale(dim, initValues: $0) }
    }

    public func callAsFunction(
        _ x: MLXArray,
        pos: MLXArray? = nil,
        maxPosition: Int? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        // Attention residual
        let attnOut = attn(norm1(x), pos: pos, maxPosition: maxPosition, mask: mask)
        let attnScaled = ls1.map { $0(attnOut) } ?? attnOut
        var h = x + attnScaled

        // FFN residual
        let ffnOut = mlp.callAsFunction(norm2(h))
        let ffnScaled = ls2.map { $0(ffnOut) } ?? ffnOut
        h = h + ffnScaled

        return h
    }
}
