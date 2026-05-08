import MLX
import MLXNN

/// Generates 2D spatial positions for patches in a grid.
public class PositionGetter {
    private var cache: [String: MLXArray] = [:]

    public init() {}

    public func callAsFunction(batchSize: Int, height: Int, width: Int) -> MLXArray {
        let key = "\(height)x\(width)"
        if cache[key] == nil {
            let y = MLXArray(0 ..< height)
            let x = MLXArray(0 ..< width)
            let yy = MLX.repeated(y, count: width, axis: 0)
            let xx = MLX.tiled(x, repetitions: [height])
            let positions = stacked([yy, xx], axis: -1) // [H*W, 2]
            cache[key] = positions
        }
        let pos = cache[key]!
        return broadcast(pos.expandedDimensions(axis: 0), to: [batchSize, pos.dim(0), 2])
    }
}

/// 2D Rotary Position Embedding — applies separate 1D RoPE to vertical and horizontal feature halves.
public class RotaryPositionEmbedding2D: Module {
    private let baseFrequency: Float
    private var frequencyCache: [String: (cos: MLXArray, sin: MLXArray)] = [:]

    public init(frequency: Float = 100.0) {
        self.baseFrequency = frequency
    }

    private func computeFreq(dim: Int, maxPos: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let key = "\(dim)x\(maxPos)x\(dtype)"
        if let cached = frequencyCache[key] {
            return cached
        }

        let exponents = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
        let invFreq = 1.0 / pow(MLXArray(baseFrequency), exponents)
        let positions = MLXArray(0 ..< maxPos).asType(.float32)
        // [maxPos, dim//2] then double to [maxPos, dim]
        let angles = outer(positions, invFreq)
        let anglesFull = concatenated([angles, angles], axis: -1)
        let cached = (cos: cos(anglesFull).asType(dtype), sin: sin(anglesFull).asType(dtype))
        frequencyCache[key] = cached
        return cached
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let half = d / 2
        let x1 = x[.ellipsis, ..<half]
        let x2 = x[.ellipsis, half...]
        return concatenated([-x2, x1], axis: -1)
    }

    private func apply1DRope(
        tokens: MLXArray,
        positions: MLXArray,
        cosComp: MLXArray,
        sinComp: MLXArray
    ) -> MLXArray {
        // positions: [B, N] → gather cos/sin: [B, N, D_half]
        let posFlat = positions.asType(.int32).reshaped([-1])
        let cosGathered = cosComp[posFlat].reshaped([positions.dim(0), positions.dim(1), -1])
        let sinGathered = sinComp[posFlat].reshaped([positions.dim(0), positions.dim(1), -1])
        // Broadcast to [B, 1, N, D_half] for multi-head
        let cosBcast = cosGathered.expandedDimensions(axis: 1)
        let sinBcast = sinGathered.expandedDimensions(axis: 1)
        return (tokens * cosBcast) + (rotateHalf(tokens) * sinBcast)
    }

    /// Apply 2D RoPE.
    /// - tokens: [B, num_heads, N, head_dim]
    /// - positions: [B, N, 2] with (y, x) coords
    public func callAsFunction(_ tokens: MLXArray, positions: MLXArray, maxPosition: Int? = nil) -> MLXArray {
        let featureDim = tokens.dim(-1) / 2
        let maxPos = maxPosition ?? (Int(positions.max().item(Int32.self)) + 1)
        let (cosComp, sinComp) = computeFreq(dim: featureDim, maxPos: maxPos, dtype: tokens.dtype)

        // Split features for vertical and horizontal
        let (vert, horiz) = tokens.split(axis: -1)
        let vertRotated = apply1DRope(tokens: vert, positions: positions[.ellipsis, 0], cosComp: cosComp, sinComp: sinComp)
        let horizRotated = apply1DRope(tokens: horiz, positions: positions[.ellipsis, 1], cosComp: cosComp, sinComp: sinComp)
        return concatenated([vertRotated, horizRotated], axis: -1)
    }
}
