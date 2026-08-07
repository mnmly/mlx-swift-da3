import MLX

/// Depth for footage whose camera provably never moves — a locked-off plate, an
/// animated still.
///
/// Every frame then observes the *same* geometry, so the frame-to-frame variation
/// in the model's output is pure noise. Measured on a 4K animated painting: 18.7%
/// per-pixel spread and 30.2% global scale drift across a 25 s clip, which shows up
/// as the whole point cloud breathing toward and away from the viewer. A per-pixel
/// temporal median over a sample of frames removes both, and costs a fraction of
/// inferring every frame.
///
/// The median is taken on **inverse** depth (1/Z): that is the quantity stereo
/// disparity is linear in, so it is what downstream displacement wants, and it
/// compresses the far field where the model is least certain.
public enum StaticSceneDepth {

    /// Per-pixel median of a stack of maps.
    ///
    /// - Parameter maps: non-empty, all `[H, W]` and identically shaped.
    /// - Returns: `[H, W]` median. Even counts average the two middle values,
    ///   matching `np.median`.
    public static func temporalMedian(_ maps: [MLXArray]) -> MLXArray {
        precondition(!maps.isEmpty, "temporalMedian requires at least one map")
        if maps.count == 1 { return maps[0] }

        let stack = stacked(maps, axis: 0)
        let ordered = sorted(stack, axis: 0)
        let mid = maps.count / 2
        if maps.count % 2 == 1 { return ordered[mid] }
        return (ordered[mid - 1] + ordered[mid]) / 2
    }

    /// Spread of a stack about its median, as a fraction — a direct read on how much
    /// the model disagrees with itself on geometry that cannot have changed.
    public static func relativeSpread(_ maps: [MLXArray], median: MLXArray) -> Float {
        guard maps.count > 1 else { return 0 }
        let stack = stacked(maps, axis: 0)
        let deviation = stack - median.expandedDimensions(axis: 0)
        let rms = sqrt((deviation * deviation).mean(axis: 0))
        let ratio = (rms / MLX.maximum(abs(median), 1e-6)).mean()
        eval(ratio)
        return ratio.item(Float.self)
    }

    /// Value below which `fraction` of the data lies (`fraction` in 0...1).
    public static func percentile(_ x: MLXArray, _ fraction: Float) -> Float {
        let count = x.size
        let ordered = sorted(x.reshaped([count]))
        let index = min(max(Int((Float(count - 1) * fraction).rounded()), 0), count - 1)
        let value = ordered[index]
        eval(value)
        return value.item(Float.self)
    }

    /// Result of mapping inverse depth onto the 0...1 range a disparity band stores.
    public struct Normalization: Sendable {
        /// `[H, W]` in 0...1, where 1 is the nearest surface kept.
        public let map: MLXArray
        /// Inverse depth mapped to 1.
        public let near: Float
        /// Inverse depth mapped to 0.
        public let far: Float
    }

    /// Map inverse depth to 0...1 between two percentiles.
    ///
    /// Clipping at a percentile rather than the extremes is what keeps the content in
    /// range: a real scene's nearest surface can be far closer than anything of
    /// interest, and spending an 8-bit band on it collapses the whole subject into a
    /// handful of codes. Everything nearer than `nearPercentile` saturates to 1 and
    /// lands on a single plane.
    ///
    /// - Parameters:
    ///   - inverseDepth: `[H, W]` of 1/Z.
    ///   - nearPercentile: percentile mapped to 1 (e.g. 99.5).
    ///   - farPercentile: percentile mapped to 0 (e.g. 0.5).
    public static func normalize(
        inverseDepth: MLXArray,
        nearPercentile: Float,
        farPercentile: Float
    ) -> Normalization {
        let near = percentile(inverseDepth, nearPercentile / 100)
        let far = percentile(inverseDepth, farPercentile / 100)
        let span = max(near - far, 1e-6)
        let normalized = clip((inverseDepth - far) / span, min: 0, max: 1)
        return Normalization(map: normalized, near: near, far: far)
    }

    /// `1 / max(depth, floor)`, guarding the division for the zero-depth pixels the
    /// model emits where it has no opinion (sky, blown highlights).
    public static func inverseDepth(_ depth: MLXArray, floor: Float = 1e-6) -> MLXArray {
        1.0 / MLX.maximum(depth, floor)
    }
}
