import MLX

/// Port of `src/depth_anything_3/model/reference_view_selector.py`.
///
/// For multi-view input the backbone promotes one view to "reference" just before the
/// camera token is injected: the chosen view is moved to index 0, receives the
/// `camera_token[:, :1]` slice while the rest receive `camera_token[:, 1:]`, and the
/// original ordering is restored when features are collected.
public enum RefViewStrategy: String, Sendable, CaseIterable {
    case first
    case middle
    case saddleBalanced = "saddle_balanced"
    case saddleSimRange = "saddle_sim_range"
}

public enum ReferenceViewSelection {
    /// `THRESH_FOR_REF_SELECTION` (`utils/constants.py`) — selection is skipped below this.
    public static let viewThreshold = 3

    /// Select one reference view per batch element.
    ///
    /// - Parameter x: `[B, S, N, C]` tokens; token 0 of each view is its class token.
    /// - Returns: `[B]` int32 view indices.
    public static func select(_ x: MLXArray, strategy: RefViewStrategy) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)
        if s <= 1 { return MLXArray.zeros([b], type: Int32.self) }

        switch strategy {
        case .first:
            return MLXArray.zeros([b], type: Int32.self)
        case .middle:
            return MLXArray.full([b], values: MLXArray(Int32(s / 2)))
        case .saddleBalanced, .saddleSimRange:
            break
        }

        // Statistics only feed an argmin/argmax; run them in fp32 regardless of model dtype.
        let cls = x[0..., 0..., 0].asType(.float32) // [B, S, C]
        let clsNorm = sqrt((cls * cls).sum(axis: -1, keepDims: true))
        let unit = cls / clsNorm // [B, S, C]

        let sim = matmul(unit, unit.transposed(0, 2, 1)) // [B, S, S]
        let simNoDiag = sim - MLXArray.eye(s).expandedDimensions(axis: 0)

        if strategy == .saddleSimRange {
            let range = simNoDiag.max(axis: -1) - simNoDiag.min(axis: -1)
            return range.argMax(axis: 1).asType(.int32)
        }

        let simScore = simNoDiag.sum(axis: -1) / Float(s - 1) // [B, S]
        let featNorm = clsNorm.squeezed(axis: -1) // [B, S]
        // torch's Tensor.var defaults to Bessel-corrected variance.
        let featVar = unit.variance(axis: -1, ddof: 1) // [B, S]

        func normalizeMetric(_ m: MLXArray) -> MLXArray {
            let lo = m.min(axis: 1, keepDims: true)
            let hi = m.max(axis: 1, keepDims: true)
            return (m - lo) / (hi - lo + 1e-8)
        }

        let balance = abs(normalizeMetric(simScore) - 0.5)
            + abs(normalizeMetric(featNorm) - 0.5)
            + abs(normalizeMetric(featVar) - 0.5)
        return balance.argMin(axis: 1).asType(.int32)
    }

    /// Move the reference view to index 0, keeping the remaining views in order.
    /// `[0, 1, 2, 3, 4]` with ref 2 becomes `[2, 0, 1, 3, 4]`.
    public static func reorder(_ x: MLXArray, referenceIndices: MLXArray) -> MLXArray {
        permute(x, referenceIndices) { positions, ref in
            MLX.which(
                positions .== 0, ref,
                MLX.which(positions .<= ref, positions - 1, positions)
            )
        }
    }

    /// Inverse of `reorder`.
    public static func restore(_ x: MLXArray, referenceIndices: MLXArray) -> MLXArray {
        permute(x, referenceIndices) { positions, ref in
            MLX.which(
                positions .< ref, positions + 1,
                MLX.which(positions .== ref, MLXArray.zeros(like: positions), positions)
            )
        }
    }

    private static func permute(
        _ x: MLXArray,
        _ referenceIndices: MLXArray,
        _ indexMap: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let b = x.dim(0)
        let s = x.dim(1)
        if s <= 1 { return x }

        let positions = MLXArray(0 ..< Int32(s))
        if b == 1 {
            let gather = indexMap(positions, referenceIndices[0])
            return take(x, gather, axis: 1)
        }
        let permuted = (0 ..< b).map { i in
            take(x[i ..< (i + 1)], indexMap(positions, referenceIndices[i]), axis: 1)
        }
        return concatenated(permuted, axis: 0)
    }
}
