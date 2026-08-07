import MLX
import MLXNN

public class DA3Attention: Module {
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "qkv") private var qkv: Linear
    @ModuleInfo(key: "q_norm") private var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") private var kNorm: LayerNorm?
    @ModuleInfo(key: "proj") private var proj: Linear
    private let rope: RotaryPositionEmbedding2D?

    public init(
        dim: Int,
        numHeads: Int = 16,
        qkvBias: Bool = true,
        projBias: Bool = true,
        qkNorm: Bool = false,
        rope: RotaryPositionEmbedding2D? = nil
    ) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(self.headDim).squareRoot()

        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        self._qNorm.wrappedValue = qkNorm ? LayerNorm(dimensions: headDim) : nil
        self._kNorm.wrappedValue = qkNorm ? LayerNorm(dimensions: headDim) : nil
        self._proj.wrappedValue = Linear(dim, dim, bias: projBias)
        self.rope = rope
    }

    public func callAsFunction(
        _ x: MLXArray,
        pos: MLXArray? = nil,
        maxPosition: Int? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let c = x.dim(2)

        // Joint QKV projection: [B, N, 3*D]
        let qkvOut = qkv(x)

        // Reshape to [B, N, 3, num_heads, head_dim] then [3, B, num_heads, N, head_dim]
        let qkvReshaped = qkvOut.reshaped([b, n, 3, numHeads, headDim])
            .transposed(axes: [2, 0, 3, 1, 4])

        var q = qkvReshaped[0]
        var k = qkvReshaped[1]
        let v = qkvReshaped[2]

        // Optional QK normalization
        if let qn = qNorm { q = qn(q) }
        if let kn = kNorm { k = kn(k) }

        // Apply RoPE if available
        if let rope = rope, let pos = pos {
            q = rope(q, positions: pos, maxPosition: maxPosition)
            k = rope(k, positions: pos, maxPosition: maxPosition)
        }

        // Expand mask for multi-head: [B, N, N] -> [B, num_heads, N, N]
        let attnMask: MLXArray? = mask.map { m in
            MLX.repeated(m.expandedDimensions(axis: 1), count: numHeads, axis: 1)
        }

        // Scaled dot-product attention
        let out = scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: attnMask
        )

        // [B, num_heads, N, head_dim] -> [B, N, D]
        return proj(out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, c]))
    }
}
