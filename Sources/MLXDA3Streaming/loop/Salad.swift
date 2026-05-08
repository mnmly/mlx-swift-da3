import Foundation
import MLX
import MLXDA3
import MLXFast
import MLXNN

// SALAD VPR model — vanilla DinoV2-B/14 backbone + SALAD aggregator
// (Sinkhorn algorithm for Locally Aggregated Descriptors).
//
// Architecture mirrors `da3_streaming/loop_utils/salad/`:
//   model.backbone.model = facebook research DinoV2-B/14 (12 blocks, 768 dim)
//   model.aggregator     = SALAD: token_features + cluster_features + score + sinkhorn
//
// Output is an L2-normalised global descriptor of dim
//   token_dim + num_clusters * cluster_dim = 256 + 64 * 128 = 8448
//
// This is a separate, smaller backbone from `MLXDA3.DinoVisionTransformer`
// (which has DA3-specific alternating local/global attention + cam_token).
// Kept self-contained so it can be deleted/swapped without touching DA3.

// MARK: - PatchEmbed (NHWC Conv2d, then flatten spatial)

final class SaladPatchEmbed: Module {
    private let patchSize: Int
    @ModuleInfo(key: "proj") private var proj: Conv2d

    public init(patchSize: Int = 14, inChannels: Int = 3, embedDim: Int = 768) {
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
        // x: [B, H, W, 3] NHWC
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let xProj = proj(x)                          // [B, H/p, W/p, D]
        let pH = h / patchSize
        let pW = w / patchSize
        return xProj.reshaped([b, pH * pW, -1])      // [B, N, D]
    }
}

// MARK: - Embeddings (PatchEmbed + cls + pos_embed with bicubic interp)

final class SaladEmbeddings: Module {
    private let patchSize: Int
    private let embedDim: Int
    private let numPatches: Int

    @ModuleInfo(key: "patch_embed") private var patchEmbed: SaladPatchEmbed
    @ParameterInfo(key: "cls_token") private var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") private var posEmbed: MLXArray

    public init(imgSize: Int = 518, patchSize: Int = 14, inChannels: Int = 3, embedDim: Int = 768) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numPatches = (imgSize / patchSize) * (imgSize / patchSize)
        self._patchEmbed.wrappedValue = SaladPatchEmbed(patchSize: patchSize, inChannels: inChannels, embedDim: embedDim)
        self._clsToken = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, embedDim]), key: "cls_token")
        self._posEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([1, numPatches + 1, embedDim]), key: "pos_embed")
    }

    private func interpolatePos(_ tokensCount: Int, h: Int, w: Int) -> MLXArray {
        let n = posEmbed.dim(1) - 1
        if tokensCount - 1 == n && w == h {
            return posEmbed
        }
        let classPos = posEmbed[0..., ..<1]
        let patchPos = posEmbed[0..., 1...]
        let dim = posEmbed.dim(-1)
        let w0 = w / patchSize
        let h0 = h / patchSize
        let m = Int(Float(n).squareRoot())
        var grid = patchPos.reshaped([1, m, m, dim])
        if h0 != m || w0 != m {
            grid = pytorchBicubicResample(grid, outH: h0, outW: w0, offset: 0.1)
        }
        let flat = grid.reshaped([1, -1, dim])
        return MLX.concatenated([classPos, flat], axis: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, H, W, 3] NHWC -> [B, 1+N, D]
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let patches = patchEmbed(x)                                      // [B, N, D]
        let cls = MLX.broadcast(clsToken, to: [b, 1, embedDim])           // [B, 1, D]
        let tokens = MLX.concatenated([cls, patches], axis: 1)            // [B, 1+N, D]
        let pe = interpolatePos(tokens.dim(1), h: h, w: w)
        return tokens + pe
    }
}

// MARK: - Vanilla self-attention (no RoPE, no QK-norm)

final class SaladAttention: Module {
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "qkv") private var qkv: Linear
    @ModuleInfo(key: "proj") private var proj: Linear

    public init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(self.headDim).squareRoot()
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, N, D]
        let b = x.dim(0); let n = x.dim(1); let d = x.dim(2)
        let qkvOut = qkv(x).reshaped([b, n, 3, numHeads, headDim])
            .transposed(axes: [2, 0, 3, 1, 4])  // [3, B, H, N, head_dim]
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        return proj(out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, d]))
    }
}

// MARK: - LayerScale (per-dim learnable gamma)

final class SaladLayerScale: Module {
    @ParameterInfo(key: "gamma") private var gamma: MLXArray

    public init(_ dim: Int, initValue: Float = 1.0) {
        self._gamma = ParameterInfo(
            wrappedValue: MLXArray.full([dim], values: MLXArray(initValue)),
            key: "gamma"
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

// MARK: - MLP (fc1 -> gelu -> fc2)

final class SaladMlp: Module {
    @ModuleInfo(key: "fc1") private var fc1: Linear
    @ModuleInfo(key: "fc2") private var fc2: Linear
    private let act: GELU

    public init(inFeatures: Int, hiddenFeatures: Int) {
        self._fc1.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: true)
        self.act = GELU()
        self._fc2.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(act(fc1(x)))
    }
}

// MARK: - Pre-norm DinoV2 block

final class SaladBlock: Module {
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "attn") private var attn: SaladAttention
    @ModuleInfo(key: "ls1") private var ls1: SaladLayerScale
    @ModuleInfo(key: "norm2") private var norm2: LayerNorm
    @ModuleInfo(key: "mlp") private var mlp: SaladMlp
    @ModuleInfo(key: "ls2") private var ls2: SaladLayerScale

    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, lnEps: Float = 1e-6) {
        let mlpHidden = Int(Float(dim) * mlpRatio)
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: lnEps)
        self._attn.wrappedValue = SaladAttention(dim: dim, numHeads: numHeads)
        self._ls1.wrappedValue = SaladLayerScale(dim)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: lnEps)
        self._mlp.wrappedValue = SaladMlp(inFeatures: dim, hiddenFeatures: mlpHidden)
        self._ls2.wrappedValue = SaladLayerScale(dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + ls1(attn(norm1(x)))
        h = h + ls2(mlp(norm2(h)))
        return h
    }
}

// MARK: - DinoV2 backbone (vitb14)

public final class SaladDinoV2: Module {
    private let depth: Int
    private let embedDim: Int

    @ModuleInfo(key: "embeddings") private var embeddings: SaladEmbeddings
    @ModuleInfo(key: "blocks") private var blocks: [SaladBlock]
    @ModuleInfo(key: "norm") private var norm: LayerNorm

    public init(
        imgSize: Int = 518,
        patchSize: Int = 14,
        embedDim: Int = 768,
        depth: Int = 12,
        numHeads: Int = 12,
        mlpRatio: Float = 4.0
    ) {
        self.depth = depth
        self.embedDim = embedDim
        self._embeddings.wrappedValue = SaladEmbeddings(
            imgSize: imgSize, patchSize: patchSize, inChannels: 3, embedDim: embedDim
        )
        self._blocks.wrappedValue = (0..<depth).map { _ in
            SaladBlock(dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio)
        }
        self._norm.wrappedValue = LayerNorm(dimensions: embedDim)
    }

    /// Returns (`patchFeatures` `[B, Hp, Wp, D]`, `clsToken` `[B, D]`).
    public func callAsFunction(_ x: MLXArray) -> (patches: MLXArray, cls: MLXArray) {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        var tokens = embeddings(x)        // [B, 1+N, D]
        for blk in blocks { tokens = blk(tokens) }
        tokens = norm(tokens)
        let cls = tokens[0..., 0]         // [B, D]
        let patches = tokens[0..., 1...]  // [B, N, D]
        let pH = h / 14
        let pW = w / 14
        let pHWC = patches.reshaped([b, pH, pW, embedDim])
        return (pHWC, cls)
    }
}

// MARK: - SALAD aggregator

public final class SaladAggregator: Module {
    public let numClusters: Int
    public let clusterDim: Int
    public let tokenDim: Int

    // token_features = Linear -> ReLU -> Linear  (sequential idx 0, 2 in python)
    @ModuleInfo(key: "token_features_fc0") private var tokFc0: Linear
    @ModuleInfo(key: "token_features_fc2") private var tokFc2: Linear
    // cluster_features = Conv2d -> Dropout -> ReLU -> Conv2d  (idx 0, 3)
    @ModuleInfo(key: "cluster_features_conv0") private var clusFc0: Conv2d
    @ModuleInfo(key: "cluster_features_conv3") private var clusFc3: Conv2d
    // score = Conv2d -> Dropout -> ReLU -> Conv2d  (idx 0, 3)
    @ModuleInfo(key: "score_conv0") private var scoreC0: Conv2d
    @ModuleInfo(key: "score_conv3") private var scoreC3: Conv2d
    // dustbin scalar
    @ParameterInfo(key: "dust_bin") private var dustBin: MLXArray

    public init(
        numChannels: Int = 768,
        numClusters: Int = 64,
        clusterDim: Int = 128,
        tokenDim: Int = 256
    ) {
        self.numClusters = numClusters
        self.clusterDim = clusterDim
        self.tokenDim = tokenDim

        self._tokFc0.wrappedValue = Linear(numChannels, 512, bias: true)
        self._tokFc2.wrappedValue = Linear(512, tokenDim, bias: true)

        self._clusFc0.wrappedValue = Conv2d(
            inputChannels: numChannels, outputChannels: 512,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true
        )
        self._clusFc3.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: clusterDim,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true
        )

        self._scoreC0.wrappedValue = Conv2d(
            inputChannels: numChannels, outputChannels: 512,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true
        )
        self._scoreC3.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: numClusters,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true
        )

        self._dustBin = ParameterInfo(wrappedValue: MLXArray(Float(1.0)), key: "dust_bin")
    }

    /// Runs Sinkhorn matrix scaling on the augmented score matrix.
    /// `S`: `[B, m, n]`, dustbin score scalar appended as `S[:, m, :]`.
    private static func getMatchingProbs(_ s: MLXArray, dustbin: MLXArray, numIters: Int) -> MLXArray {
        let b = s.dim(0); let m = s.dim(1); let n = s.dim(2)
        // Augment: append dustbin row of shape [B, 1, n] filled with dustbin scalar
        let dustRow = MLX.broadcast(dustbin.reshaped([1, 1, 1]), to: [b, 1, n])
        let sAug = MLX.concatenated([s, dustRow], axis: 1)  // [B, m+1, n]

        let norm = -log(Float(n + m))  // scalar
        var logA = MLXArray([Float](repeating: norm, count: m + 1))
        // log_a[-1] += log(n - m)
        logA = logA.asType(.float32)
        var logAArr = logA.asArray(Float.self)
        logAArr[m] += log(Float(n - m))
        logA = MLXArray(logAArr, [m + 1])
        let logB = MLXArray([Float](repeating: norm, count: n))
        let logABroadcast = MLX.broadcast(logA.reshaped([1, m + 1]), to: [b, m + 1])
        let logBBroadcast = MLX.broadcast(logB.reshaped([1, n]), to: [b, n])

        // Sinkhorn iterations on M = sAug
        var u = MLXArray.zeros([b, m + 1])
        var v = MLXArray.zeros([b, n])
        let M = sAug
        for _ in 0..<numIters {
            // u = log_a - logsumexp(M + v[:, None, :], dim=2)
            let mPlusV = M + v.expandedDimensions(axis: 1)   // [B, m+1, n]
            u = logABroadcast - MLX.logSumExp(mPlusV, axis: 2)
            // v = log_b - logsumexp(M + u[:, :, None], dim=1)
            let mPlusU = M + u.expandedDimensions(axis: 2)   // [B, m+1, n]
            v = logBBroadcast - MLX.logSumExp(mPlusU, axis: 1)
        }
        let logP = M + u.expandedDimensions(axis: 2) + v.expandedDimensions(axis: 1)
        return logP - norm
    }

    /// `patches`: `[B, Hp, Wp, D]` (NHWC), `cls`: `[B, D]` -> descriptor `[B, tokenDim + m*l]`
    public func callAsFunction(patches: MLXArray, cls: MLXArray) -> MLXArray {
        let b = patches.dim(0)

        // cluster features: Conv2d(patches NHWC) -> ReLU -> Conv2d -> [B, Hp, Wp, l]
        var fNHWC = clusFc0(patches)
        fNHWC = relu(fNHWC)
        fNHWC = clusFc3(fNHWC)            // [B, Hp, Wp, l]
        // flatten spatial: [B, l, n] (matches python's [B, l, Hp*Wp])
        let l = fNHWC.dim(-1)
        let n = fNHWC.dim(1) * fNHWC.dim(2)
        // reshape to [B, n, l] then transpose to [B, l, n]
        let f = fNHWC.reshaped([b, n, l]).transposed(axes: [0, 2, 1])

        // score features: Conv2d -> ReLU -> Conv2d -> [B, Hp, Wp, m]
        var pNHWC = scoreC0(patches)
        pNHWC = relu(pNHWC)
        pNHWC = scoreC3(pNHWC)            // [B, Hp, Wp, m]
        let m = pNHWC.dim(-1)
        // reshape to [B, n, m] then transpose to [B, m, n]
        let p = pNHWC.reshaped([b, n, m]).transposed(axes: [0, 2, 1])

        // token features: Linear -> ReLU -> Linear -> [B, tokenDim]
        var t = tokFc0(cls)
        t = relu(t)
        t = tokFc2(t)

        // Sinkhorn on p
        var pSink = SaladAggregator.getMatchingProbs(p, dustbin: dustBin, numIters: 3)
        pSink = MLX.exp(pSink)
        // Drop the dustbin row: keep first m of m+1
        pSink = pSink[0..., ..<m, 0...]   // [B, m, n]

        // Multiply: f [B, l, n] x p [B, m, n] -> sum_n f[b,l,n] * p[b,m,n] -> [B, l, m]
        // python: f.unsqueeze(2).repeat(...,m,...) * p.unsqueeze(1).repeat(...,l,...).sum(dim=-1)
        // Equivalent: einsum bln, bmn -> blm
        let fExp = f.expandedDimensions(axis: 2)          // [B, l, 1, n]
        let pExp = pSink.expandedDimensions(axis: 1)       // [B, 1, m, n]
        let prod = fExp * pExp                             // [B, l, m, n]
        let sum = prod.sum(axis: -1)                       // [B, l, m]

        // L2-normalize sum along axis 1 (the l axis)
        let sumNorm = l2Normalize(sum, axis: 1)
        let sumFlat = sumNorm.reshaped([b, l * m])        // [B, l*m]

        let tNorm = l2Normalize(t, axis: -1)              // [B, tokenDim]

        let combined = MLX.concatenated([tNorm, sumFlat], axis: -1)  // [B, tokenDim + l*m]
        return l2Normalize(combined, axis: -1)
    }
}

private func l2Normalize(_ x: MLXArray, axis: Int) -> MLXArray {
    let sq = (x * x).sum(axis: axis, keepDims: true)
    return x / MLX.sqrt(sq + Float(1e-12))
}

// MARK: - Top-level SALAD model

public final class SaladModel: Module {
    @ModuleInfo(key: "backbone_model") private var backbone: SaladDinoV2
    @ModuleInfo(key: "aggregator") private var aggregator: SaladAggregator

    public override init() {
        self._backbone.wrappedValue = SaladDinoV2()
        self._aggregator.wrappedValue = SaladAggregator()
        super.init()
    }

    /// `x`: `[B, H, W, 3]` NHWC float, ImageNet-normalized. Returns `[B, 8448]` descriptors.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (patches, cls) = backbone(x)
        return aggregator(patches: patches, cls: cls)
    }
}
