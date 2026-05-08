import Foundation
import MLX
import MLXNN

/// PyTorch-style bicubic resample (`nn.functional.interpolate(mode='bicubic',
/// align_corners=False, antialias=False)`) for 4D NHWC inputs.
///
/// Coordinate mapping: output[y, x] is sampled at input position
///   (y + 0.5) / sy - 0.5,  (x + 0.5) / sx - 0.5
/// where (sx, sy) = ((outW + offset) / inW, (outH + offset) / inH) for the
/// dinov2 `interpolate_offset=0.1` case.
///
/// MLX's `Upsample(.cubic(alignCorners: false))` uses a slightly different
/// start offset and doesn't match PyTorch on non-integer scales.
public func pytorchBicubicResample(_ x: MLXArray, outH: Int, outW: Int, offset: Float = 0.1) -> MLXArray {
    let B = x.dim(0), inH = x.dim(1), inW = x.dim(2), C = x.dim(3)
    // Per-axis scales
    let sy = (Float(outH) + offset) / Float(inH)
    let sx = (Float(outW) + offset) / Float(inW)

    // Compute float sample coordinates per output index using PyTorch's mapping
    var ys = [Float](repeating: 0, count: outH)
    for i in 0..<outH { ys[i] = (Float(i) + 0.5) / sy - 0.5 }
    var xs = [Float](repeating: 0, count: outW)
    for i in 0..<outW { xs[i] = (Float(i) + 0.5) / sx - 0.5 }

    // Cubic kernel (a = -0.75, matches PyTorch default for bicubic).
    func cubicWeights(_ t: Float) -> [Float] {
        // Returns weights for [floor-1, floor, floor+1, floor+2] given fractional t in [0,1).
        // Distances from sample points: |1+t|, |t|, |1-t|, |2-t|.
        let a: Float = -0.75
        // For |x| in [1, 2]:  W(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a
        // For |x| in [0, 1]:  W(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1
        let d_1 = 1 + t                              // in [1, 2)
        let d0 = t                                   // in [0, 1)
        let d1 = 1 - t                               // in (0, 1]
        let d2 = 2 - t                               // in (1, 2]
        let w_1 = a * d_1 * d_1 * d_1 - 5 * a * d_1 * d_1 + 8 * a * d_1 - 4 * a
        let w0 = (a + 2) * d0 * d0 * d0 - (a + 3) * d0 * d0 + 1
        let w1 = (a + 2) * d1 * d1 * d1 - (a + 3) * d1 * d1 + 1
        let w2 = a * d2 * d2 * d2 - 5 * a * d2 * d2 + 8 * a * d2 - 4 * a
        return [w_1, w0, w1, w2]
    }

    // Build per-output-index integer indices and weights for both axes.
    func build(_ coords: [Float], inSize: Int) -> (idx: [[Int]], wts: [[Float]]) {
        var idx = [[Int]]()
        var wts = [[Float]]()
        for c in coords {
            let f = floor(c)
            let t = c - f
            let base = Int(f)
            // Border-replicate (PyTorch default bicubic uses border reflection,
            // but for align_corners=False it uses zero-pad? Actually torch
            // bicubic_2d uses border replication for out-of-range.)
            let ii = [base - 1, base, base + 1, base + 2].map { max(0, min(inSize - 1, $0)) }
            idx.append(ii)
            wts.append(cubicWeights(t))
        }
        return (idx, wts)
    }

    let (yIdx, yWts) = build(ys, inSize: inH)
    let (xIdx, xWts) = build(xs, inSize: inW)

    // Output buffer: gather rows then convolve.
    // First, compute interpolation along H axis: for each output y, gather
    // 4 input rows at indices yIdx[y] and weight by yWts[y]. Result: [B, outH, inW, C].
    var hRows = [MLXArray]()
    for j in 0..<outH {
        let i0 = yIdx[j][0], i1 = yIdx[j][1], i2 = yIdx[j][2], i3 = yIdx[j][3]
        let w0 = yWts[j][0], w1 = yWts[j][1], w2 = yWts[j][2], w3 = yWts[j][3]
        let row = x[0..., i0..<(i0+1), 0..., 0...] * MLXArray(Float(w0))
            + x[0..., i1..<(i1+1), 0..., 0...] * MLXArray(Float(w1))
            + x[0..., i2..<(i2+1), 0..., 0...] * MLXArray(Float(w2))
            + x[0..., i3..<(i3+1), 0..., 0...] * MLXArray(Float(w3))
        hRows.append(row)
    }
    var hStack = MLX.concatenated(hRows, axis: 1)  // [B, outH, inW, C]

    // Now interpolate along W axis.
    var wCols = [MLXArray]()
    for j in 0..<outW {
        let i0 = xIdx[j][0], i1 = xIdx[j][1], i2 = xIdx[j][2], i3 = xIdx[j][3]
        let w0 = xWts[j][0], w1 = xWts[j][1], w2 = xWts[j][2], w3 = xWts[j][3]
        let col = hStack[0..., 0..., i0..<(i0+1), 0...] * MLXArray(Float(w0))
            + hStack[0..., 0..., i1..<(i1+1), 0...] * MLXArray(Float(w1))
            + hStack[0..., 0..., i2..<(i2+1), 0...] * MLXArray(Float(w2))
            + hStack[0..., 0..., i3..<(i3+1), 0...] * MLXArray(Float(w3))
        wCols.append(col)
    }
    return MLX.concatenated(wCols, axis: 2)  // [B, outH, outW, C]
}

public class PatchEmbed: Module {
    private let patchSize: Int

    @ModuleInfo(key: "proj") private var proj: Conv2d

    public init(
        imgSize: Int = 518,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 1024
    ) {
        self.patchSize = patchSize
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            bias: true
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let xProj = proj(x) // [B, H', W', D]
        let pH = h / patchSize
        let pW = w / patchSize
        return xProj.reshaped([b, pH * pW, -1]) // [B, N, D]
    }
}

public class Embeddings: Module {
    /// Diagnostic: dump the interpolated pos embedding when set.
    public static var _dumpPosEmbed: ((MLXArray) -> Void)?

    private let patchSize: Int
    private let embedDim: Int
    private let numPatches: Int

    @ModuleInfo(key: "patch_embed") private var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token") private var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") private var posEmbed: MLXArray

    public init(
        imgSize: Int = 518,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 1024
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numPatches = (imgSize / patchSize) * (imgSize / patchSize)

        self._patchEmbed.wrappedValue = PatchEmbed(
            imgSize: imgSize, patchSize: patchSize,
            inChannels: inChannels, embedDim: embedDim
        )
        self._clsToken = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, embedDim]), key: "cls_token")
        self._posEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([1, numPatches + 1, embedDim]), key: "pos_embed")
    }

    private func interpolatePosEncoding(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let npatch = x.dim(1) - 1
        let n = posEmbed.dim(1) - 1

        if npatch == n && w == h {
            return posEmbed
        }

        // pos_embed[:, :1] keeps batch dim — class position embedding
        let classPos = posEmbed[0..., ..<1]
        // pos_embed[:, 1:] — patch position embeddings
        let patchPos = posEmbed[0..., 1...]

        let dim = x.dim(-1)
        let w0 = w / patchSize
        let h0 = h / patchSize
        let m = Int(Float(n).squareRoot())

        // [1, M*M, D] -> [1, M, M, D]
        var patchPosGrid = patchPos.reshaped([1, m, m, dim])

        if h0 != m || w0 != m {
            // Mirror python (vision_transformer.py:interpolate_pos_encoding):
            //   mode="bicubic", align_corners=False (default), antialias=False,
            //   scale_factor = ((h0+0.1)/M, (w0+0.1)/M) when interpolate_offset=0.1
            // MLX's Upsample uses a slightly different sampling formula for
            //   align_corners=False that doesn't match PyTorch's
            //   `(i+0.5)/scale - 0.5` exactly when scale is non-integer.
            // We implement manually to match PyTorch.
            patchPosGrid = pytorchBicubicResample(
                patchPosGrid,
                outH: h0, outW: w0,
                offset: 0.1
            )
        }

        let patchPosFlat = patchPosGrid.reshaped([1, -1, dim])
        return concatenated([classPos, patchPosFlat], axis: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)

        let patches = patchEmbed(x) // [B, N, D]
        let clsTokens = broadcast(clsToken, to: [b, 1, embedDim])
        let tokens = concatenated([clsTokens, patches], axis: 1) // [B, 1+N, D]
        let pe = interpolatePosEncoding(tokens, h: h, w: w)
        Embeddings._dumpPosEmbed?(pe)
        return tokens + pe
    }
}
