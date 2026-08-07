import MLX

/// Edge-aware upsampling of a low-resolution map using a high-resolution guide,
/// via He et al.'s guided filter in its "fast" form.
///
/// Depth from a ViT is smooth by construction: at any process resolution its
/// output carries far less high-frequency detail than the colour frame does.
/// Bilinear upsampling to display resolution therefore leaves depth edges soft and
/// misaligned with the picture. The guided filter fits a local linear model
/// `q = a·I + b` between the guide and the map, so the upsampled result inherits
/// the guide's edges while keeping the map's values.
///
/// "Fast" means the per-pixel coefficients are solved at the *low* resolution and
/// only `a` and `b` are upsampled — two smooth fields that survive bilinear
/// interpolation — instead of filtering at full resolution.
public enum GuidedFilter {

    /// Box mean over a `(2·radius+1)²` window, with exact edge normalization.
    ///
    /// - Parameter x: `[H, W]` or `[H, W, C]`.
    static func boxMean(_ x: MLXArray, radius: Int) -> MLXArray {
        precondition(radius >= 0, "radius must be non-negative")
        if radius == 0 { return x }

        let hadChannel = x.ndim == 3
        let input = hadChannel ? x : x.expandedDimensions(axis: -1)
        let channels = input.dim(2)
        let nhwc = input.expandedDimensions(axis: 0) // [1, H, W, C]

        let k = 2 * radius + 1
        // Separable: a (k × 1) pass then a (1 × k) pass, per channel.
        let vertical = MLXArray.ones([channels, k, 1, 1])
        let horizontal = MLXArray.ones([channels, 1, k, 1])

        func boxSum(_ a: MLXArray) -> MLXArray {
            let c = a.dim(3)
            let v = conv2d(a, vertical[..<c], padding: [radius, 0], groups: c)
            return conv2d(v, horizontal[..<c], padding: [0, radius], groups: c)
        }

        // Dividing by the same box sum over an all-ones image makes the edges exact
        // (partial windows are normalized by their true element count).
        let counts = boxSum(MLXArray.ones([1, input.dim(0), input.dim(1), 1]))
        let mean = boxSum(nhwc) / counts
        let out = mean.squeezed(axis: 0)
        return hadChannel ? out : out.squeezed(axis: -1)
    }

    /// Upsample `lowRes` to the guide's resolution, taking edges from `guide`.
    ///
    /// - Parameters:
    ///   - lowRes: `[h, w]` map to upsample (e.g. inverse depth).
    ///   - guide: `[H, W]` single-channel guide at the target resolution, same
    ///     value range as itself matters only through `epsilon` — pass luma in 0...1.
    ///   - radius: window radius **in low-resolution pixels**.
    ///   - epsilon: regularization. Larger = smoother, less edge transfer. For a
    ///     guide in 0...1, 1e-4 is a reasonable default (≈ (0.01)² of contrast).
    /// - Returns: `[H, W]` upsampled map.
    public static func upsample(
        lowRes: MLXArray,
        guide: MLXArray,
        radius: Int = 8,
        epsilon: Float = 1e-4
    ) -> MLXArray {
        let targetH = guide.dim(0)
        let targetW = guide.dim(1)
        let lowH = lowRes.dim(0)
        let lowW = lowRes.dim(1)

        // Guide at the map's resolution, so the linear model is solved cheaply.
        let guideLow = resample(guide, height: lowH, width: lowW)

        let meanGuide = boxMean(guideLow, radius: radius)
        let meanMap = boxMean(lowRes, radius: radius)
        let meanGuideMap = boxMean(guideLow * lowRes, radius: radius)
        let meanGuideSq = boxMean(guideLow * guideLow, radius: radius)

        let covariance = meanGuideMap - meanGuide * meanMap
        let variance = meanGuideSq - meanGuide * meanGuide

        let a = covariance / (variance + epsilon)
        let b = meanMap - a * meanGuide

        // `a` and `b` are smooth, which is what makes upsampling them (rather than
        // the filtered result) sound.
        let aFull = resample(boxMean(a, radius: radius), height: targetH, width: targetW)
        let bFull = resample(boxMean(b, radius: radius), height: targetH, width: targetW)
        return aFull * guide + bFull
    }

    /// Resample an `[H, W]` map. Downscales with area weights (the guide must not
    /// alias when it is reduced to the map's grid) and upscales with triangle
    /// weights (the coefficient fields are smooth; cubic would overshoot).
    public static func resample(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        if x.dim(0) == height && x.dim(1) == width { return x }
        let shrinking = height < x.dim(0) || width < x.dim(1)
        let resized = DA3ImagePreprocessing.resize(
            x.expandedDimensions(axis: -1), height: height, width: width,
            method: shrinking ? .area : .bilinear
        )
        return resized.squeezed(axis: -1)
    }
}
