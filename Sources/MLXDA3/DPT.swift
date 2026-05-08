import Foundation
import MLX
import MLXNN

private var posEmbedCache: [String: MLXArray] = [:]
private let posEmbedCacheLock = NSLock()

// MARK: - Positional Embedding Utilities

/// Create normalized UV grid of shape (H, W, 2).
func createUVGrid(width: Int, height: Int, aspectRatio: Float? = nil) -> MLXArray {
    let ar = aspectRatio ?? (Float(width) / Float(height))
    let diag = (ar * ar + 1.0).squareRoot()
    let spanX = ar / diag
    let spanY = 1.0 / diag

    let lx = -spanX * Float(width - 1) / Float(width)
    let rx = spanX * Float(width - 1) / Float(width)
    let ty = -spanY * Float(height - 1) / Float(height)
    let by = spanY * Float(height - 1) / Float(height)

    let xCoords = linspace(lx, rx, count: width)
    let yCoords = linspace(ty, by, count: height)

    // meshgrid -> (H, W) each
    let uu = broadcast(xCoords.expandedDimensions(axis: 0), to: [height, width])
    let vv = broadcast(yCoords.expandedDimensions(axis: 1), to: [height, width])
    return stacked([uu, vv], axis: -1) // [H, W, 2]
}

/// Generate 1D sincos positional embedding. Returns [M, D].
func makeSincosPosEmbed(embedDim: Int, pos: MLXArray, omega0: Float = 100.0) -> MLXArray {
    var omega = MLXArray(0 ..< (embedDim / 2)).asType(.float32)
    omega = omega / Float(embedDim / 2)
    omega = 1.0 / pow(MLXArray(omega0), omega) // [D/2]

    let posFlat = pos.reshaped([-1]).asType(.float32) // [M]
    let angles = outer(posFlat, omega) // [M, D/2]
    return concatenated([sin(angles), cos(angles)], axis: -1) // [M, D]
}

/// Convert (H, W, 2) position grid to (H, W, D) sincos embeddings.
func positionGridToEmbed(posGrid: MLXArray, embedDim: Int) -> MLXArray {
    let H = posGrid.dim(0)
    let W = posGrid.dim(1)
    let posFlat = posGrid.reshaped([-1, 2])

    let embX = makeSincosPosEmbed(embedDim: embedDim / 2, pos: posFlat[0..., 0]) // [H*W, D/2]
    let embY = makeSincosPosEmbed(embedDim: embedDim / 2, pos: posFlat[0..., 1]) // [H*W, D/2]
    let emb = concatenated([embX, embY], axis: -1) // [H*W, D]
    return emb.reshaped([H, W, embedDim])
}

/// Add UV positional embedding to feature map (NHWC).
func addPosEmbed(_ x: MLXArray, wImg: Int, hImg: Int, ratio: Float = 0.1) -> MLXArray {
    let ph = x.dim(1)
    let pw = x.dim(2)
    let C = x.dim(3)
    let key = "\(pw)x\(ph)x\(C)x\(wImg)x\(hImg)x\(ratio)x\(x.dtype)"

    posEmbedCacheLock.lock()
    let cached = posEmbedCache[key]
    posEmbedCacheLock.unlock()

    let peEmbed: MLXArray
    if let cached {
        peEmbed = cached
    } else {
        let pe = createUVGrid(width: pw, height: ph, aspectRatio: Float(wImg) / Float(hImg))
        peEmbed = (positionGridToEmbed(posGrid: pe, embedDim: C) * ratio).asType(x.dtype) // [ph, pw, C]
        posEmbedCacheLock.lock()
        posEmbedCache[key] = peEmbed
        posEmbedCacheLock.unlock()
    }

    return x + peEmbed.expandedDimensions(axis: 0)
}

// MARK: - Bilinear Interpolation

/// Bilinear interpolation for NHWC tensors.
func bilinearInterpolate(_ x: MLXArray, size: (Int, Int)) -> MLXArray {
    let (tH, tW) = size
    if x.dim(1) == tH && x.dim(2) == tW { return x }
    let upsample = Upsample(
        scaleFactor: [Float(tH) / Float(x.dim(1)), Float(tW) / Float(x.dim(2))],
        mode: .linear(alignCorners: true)
    )
    return upsample(x)
}

// MARK: - Activation

/// Apply named activation function.
func applyActivation(_ x: MLXArray, _ activation: String) -> MLXArray {
    switch activation.lowercased() {
    case "exp": return exp(x)
    case "expp1": return exp(x) + 1.0
    case "expm1": return exp(x) - 1.0
    case "relu": return relu(x)
    case "sigmoid": return sigmoid(x)
    case "softplus": return logAddExp(x, MLXArray.zeros(like: x))
    case "tanh": return tanh(x)
    default: return x // linear
    }
}

// MARK: - ResidualConvUnit

/// Lightweight residual convolution block for fusion.
public class ResidualConvUnit: Module {
    @ModuleInfo(key: "conv1") private var conv1: Conv2d
    @ModuleInfo(key: "conv2") private var conv2: Conv2d

    public init(features: Int) {
        self._conv1.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(x)
        out = conv1(out)
        out = relu(out)
        out = conv2(out)
        return out + x
    }
}

// MARK: - FeatureFusionBlock

/// Top-down fusion: optional residual merge + upsample + 1x1 projection.
public class FeatureFusionBlock: Module {
    private let hasResidual: Bool

    @ModuleInfo(key: "resConfUnit1") private var resConfUnit1: ResidualConvUnit?
    @ModuleInfo(key: "resConfUnit2") private var resConfUnit2: ResidualConvUnit
    @ModuleInfo(key: "out_conv") private var outConv: Conv2d

    public init(features: Int, hasResidual: Bool = true) {
        self.hasResidual = hasResidual
        self._resConfUnit1.wrappedValue = hasResidual ? ResidualConvUnit(features: features) : nil
        self._resConfUnit2.wrappedValue = ResidualConvUnit(features: features)
        self._outConv.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - top: Top branch feature map
    ///   - lateral: Optional lateral branch feature map
    ///   - size: Target spatial size (H, W) for upsampling; defaults to 2x if nil
    public func callAsFunction(_ top: MLXArray, _ lateral: MLXArray? = nil, size: (Int, Int)? = nil) -> MLXArray {
        var y = top
        if hasResidual, let lat = lateral, let rcu1 = resConfUnit1 {
            y = y + rcu1(lat)
        }
        y = resConfUnit2(y)

        if let sz = size {
            y = bilinearInterpolate(y, size: sz)
        } else {
            let h = y.dim(1)
            let w = y.dim(2)
            y = bilinearInterpolate(y, size: (h * 2, w * 2))
        }

        return outConv(y)
    }
}

// MARK: - Scratch

/// Stage adapters (3x3 conv) for each feature pyramid level.
public class Scratch: Module {
    @ModuleInfo(key: "layer1_rn") private var layer1Rn: Conv2d
    @ModuleInfo(key: "layer2_rn") private var layer2Rn: Conv2d
    @ModuleInfo(key: "layer3_rn") private var layer3Rn: Conv2d
    @ModuleInfo(key: "layer4_rn") private var layer4Rn: Conv2d

    public init(inChannels: [Int], outFeatures: Int) {
        self._layer1Rn.wrappedValue = Conv2d(inputChannels: inChannels[0], outputChannels: outFeatures, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: false)
        self._layer2Rn.wrappedValue = Conv2d(inputChannels: inChannels[1], outputChannels: outFeatures, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: false)
        self._layer3Rn.wrappedValue = Conv2d(inputChannels: inChannels[2], outputChannels: outFeatures, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: false)
        self._layer4Rn.wrappedValue = Conv2d(inputChannels: inChannels[3], outputChannels: outFeatures, kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: false)
    }

    public func apply(_ feats: [MLXArray]) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        (layer1Rn(feats[0]), layer2Rn(feats[1]), layer3Rn(feats[2]), layer4Rn(feats[3]))
    }
}

// MARK: - AuxHead

/// Auxiliary head: conv -> LN -> ReLU -> 1x1 conv.
public class AuxHead: Module {
    @ModuleInfo(key: "conv0") private var conv0: Conv2d
    @ModuleInfo(key: "ln") private var ln: LayerNorm
    @ModuleInfo(key: "conv1") private var conv1: Conv2d

    public init(inCh: Int, midCh: Int, outDim: Int = 7) {
        self._conv0.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: midCh, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        self._ln.wrappedValue = LayerNorm(dimensions: midCh)
        self._conv1.wrappedValue = Conv2d(inputChannels: midCh, outputChannels: outDim, kernelSize: .init(1), stride: .init(1), padding: .init(0))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv0(x)
        h = ln(h)
        h = relu(h)
        return conv1(h)
    }
}

// MARK: - DPT

/// Single-head DPT for dense depth prediction with optional sky head.
public class DPT: Module {
    private let patchSize: Int
    private let activation: String
    private let confActivation: String
    private let posEmbed: Bool
    private let downRatio: Int
    private let headMain: String
    private let skyName: String
    private let outDim: Int
    private let hasConf: Bool
    private let useSkyHead: Bool
    private let skyActivation: String

    @ModuleInfo(key: "norm") private var norm: LayerNorm?
    @ModuleInfo(key: "projects") private var projects: [Conv2d]
    @ModuleInfo(key: "resize_layer0") private var resizeLayer0: ConvTransposed2d
    @ModuleInfo(key: "resize_layer1") private var resizeLayer1: ConvTransposed2d
    @ModuleInfo(key: "resize_layer3") private var resizeLayer3: Conv2d
    @ModuleInfo(key: "scratch") private var scratch: Scratch
    @ModuleInfo(key: "refinenet1") private var refinenet1: FeatureFusionBlock
    @ModuleInfo(key: "refinenet2") private var refinenet2: FeatureFusionBlock
    @ModuleInfo(key: "refinenet3") private var refinenet3: FeatureFusionBlock
    @ModuleInfo(key: "refinenet4") private var refinenet4: FeatureFusionBlock
    @ModuleInfo(key: "output_conv1") private var outputConv1: Conv2d
    @ModuleInfo(key: "output_conv2_0") private var outputConv2_0: Conv2d
    @ModuleInfo(key: "output_conv2_ln") private var outputConv2Ln: LayerNorm?
    @ModuleInfo(key: "output_conv2_1") private var outputConv2_1: Conv2d
    @ModuleInfo(key: "sky_conv2_0") private var skyConv2_0: Conv2d?
    @ModuleInfo(key: "sky_conv2_ln") private var skyConv2Ln: LayerNorm?
    @ModuleInfo(key: "sky_conv2_1") private var skyConv2_1: Conv2d?

    public init(
        dimIn: Int,
        patchSize: Int = 14,
        outputDim: Int = 1,
        activation: String = "exp",
        confActivation: String = "expp1",
        features: Int = 256,
        outChannels: [Int] = [256, 512, 1024, 1024],
        posEmbed: Bool = false,
        downRatio: Int = 1,
        headName: String = "depth",
        useSkyHead: Bool = true,
        skyName: String = "sky",
        skyActivation: String = "relu",
        useLnForHeads: Bool = false,
        normType: String = "idt"
    ) {
        self.patchSize = patchSize
        self.activation = activation
        self.confActivation = confActivation
        self.posEmbed = posEmbed
        self.downRatio = downRatio
        self.headMain = headName
        self.skyName = skyName
        self.outDim = outputDim
        self.hasConf = outputDim > 1
        self.useSkyHead = useSkyHead
        self.skyActivation = skyActivation

        // Token pre-norm
        self._norm.wrappedValue = normType == "layer" ? LayerNorm(dimensions: dimIn) : nil

        // Per-stage 1x1 projection
        self._projects.wrappedValue = outChannels.map { oc in
            Conv2d(inputChannels: dimIn, outputChannels: oc, kernelSize: .init(1), stride: .init(1), padding: .init(0))
        }

        // Spatial resize layers
        self._resizeLayer0.wrappedValue = ConvTransposed2d(inputChannels: outChannels[0], outputChannels: outChannels[0], kernelSize: .init(4), stride: .init(4), padding: .init(0))
        self._resizeLayer1.wrappedValue = ConvTransposed2d(inputChannels: outChannels[1], outputChannels: outChannels[1], kernelSize: .init(2), stride: .init(2), padding: .init(0))
        self._resizeLayer3.wrappedValue = Conv2d(inputChannels: outChannels[3], outputChannels: outChannels[3], kernelSize: .init(3), stride: .init(2), padding: .init(1))

        self._scratch.wrappedValue = Scratch(inChannels: outChannels, outFeatures: features)

        // Fusion chain
        self._refinenet1.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet2.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet3.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet4.wrappedValue = FeatureFusionBlock(features: features, hasResidual: false)

        // Head neck + main head
        let hf1 = features
        let hf2 = 32
        self._outputConv1.wrappedValue = Conv2d(inputChannels: hf1, outputChannels: hf1 / 2, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        self._outputConv2_0.wrappedValue = Conv2d(inputChannels: hf1 / 2, outputChannels: hf2, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        self._outputConv2Ln.wrappedValue = useLnForHeads ? LayerNorm(dimensions: hf2) : nil
        self._outputConv2_1.wrappedValue = Conv2d(inputChannels: hf2, outputChannels: outputDim, kernelSize: .init(1), stride: .init(1), padding: .init(0))

        // Sky head
        if useSkyHead {
            self._skyConv2_0.wrappedValue = Conv2d(inputChannels: hf1 / 2, outputChannels: hf2, kernelSize: .init(3), stride: .init(1), padding: .init(1))
            self._skyConv2Ln.wrappedValue = useLnForHeads ? LayerNorm(dimensions: hf2) : nil
            self._skyConv2_1.wrappedValue = Conv2d(inputChannels: hf2, outputChannels: 1, kernelSize: .init(1), stride: .init(1), padding: .init(0))
        } else {
            self._skyConv2_0.wrappedValue = nil
            self._skyConv2Ln.wrappedValue = nil
            self._skyConv2_1.wrappedValue = nil
        }
    }

    private func resize(_ feats: [MLXArray]) -> [MLXArray] {
        [resizeLayer0(feats[0]), resizeLayer1(feats[1]), feats[2], resizeLayer3(feats[3])]
    }

    private func headForward(conv0: Conv2d, ln: LayerNorm?, conv1: Conv2d, _ x: MLXArray) -> MLXArray {
        var h = conv0(x)
        if let ln { h = ln(h) }
        h = relu(h)
        return conv1(h)
    }

    public func callAsFunction(
        feats: [(MLXArray, MLXArray)],
        H: Int, W: Int,
        outputs requestedOutputs: DA3Outputs = .all
    ) -> [String: MLXArray] {
        if requestedOutputs == .all {
            return callAsFunction(feats: feats, H: H, W: W)
        }

        let B = feats[0].0.dim(0)
        let S = feats[0].0.dim(1)
        let BS = B * S
        let C = feats[0].0.dim(-1)
        let ph = H / patchSize
        let pw = W / patchSize

        // Project and reshape to spatial
        var resized: [MLXArray] = []
        for i in 0 ..< 4 {
            var x = feats[i].0.reshaped([BS, feats[i].0.dim(2), C]) // [B*S, N, C]
            if let norm { x = norm(x) }
            x = x.reshaped([BS, ph, pw, C]) // [B*S, ph, pw, C] NHWC
            x = projects[i](x)
            if posEmbed { x = addPosEmbed(x, wImg: W, hImg: H) }
            resized.append(x)
        }

        resized = resize(resized)

        // Fusion pyramid
        let (l1Rn, l2Rn, l3Rn, l4Rn) = scratch.apply(resized)

        var out = refinenet4(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
        out = refinenet3(out, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
        out = refinenet2(out, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
        out = refinenet1(out, l1Rn)

        // Neck
        var fused = outputConv1(out)

        let hOut = ph * patchSize / downRatio
        let wOut = pw * patchSize / downRatio
        fused = bilinearInterpolate(fused, size: (hOut, wOut))
        if posEmbed { fused = addPosEmbed(fused, wImg: W, hImg: H) }

        var outs: [String: MLXArray] = [:]
        let wantsDepth = requestedOutputs.contains(.depth)
        let wantsDepthConf = requestedOutputs.contains(.depthConfidence)

        if wantsDepth || wantsDepthConf {
            let mainLogits = headForward(conv0: outputConv2_0, ln: outputConv2Ln, conv1: outputConv2_1, fused)

            if hasConf {
                if wantsDepth {
                    let pred = applyActivation(mainLogits[.ellipsis, ..<(outDim - 1)], activation)
                    outs[headMain] = pred.squeezed(axis: -1).reshaped([B, S, hOut, wOut])
                }
                if wantsDepthConf {
                    let conf = applyActivation(mainLogits[.ellipsis, (outDim - 1)], confActivation)
                    outs["\(headMain)_conf"] = conf.reshaped([B, S, hOut, wOut])
                }
            } else if wantsDepth {
                let pred = applyActivation(mainLogits, activation).squeezed(axis: -1)
                outs[headMain] = pred.reshaped([B, S, hOut, wOut])
            }
        }

        // Sky head
        if requestedOutputs.contains(.sky), useSkyHead, let sc0 = skyConv2_0, let sc1 = skyConv2_1 {
            let skyLogits = headForward(conv0: sc0, ln: skyConv2Ln, conv1: sc1, fused)
            outs[skyName] = applyActivation(skyLogits.squeezed(axis: -1), skyActivation)
                .reshaped([B, S, hOut, wOut])
        }

        return outs
    }

    public func callAsFunction(
        feats: [(MLXArray, MLXArray)],
        H: Int, W: Int
    ) -> [String: MLXArray] {
        let B = feats[0].0.dim(0)
        let S = feats[0].0.dim(1)
        let BS = B * S
        let C = feats[0].0.dim(-1)
        let ph = H / patchSize
        let pw = W / patchSize

        var resized: [MLXArray] = []
        for i in 0 ..< 4 {
            var x = feats[i].0.reshaped([BS, feats[i].0.dim(2), C])
            if let norm { x = norm(x) }
            x = x.reshaped([BS, ph, pw, C])
            x = projects[i](x)
            if posEmbed { x = addPosEmbed(x, wImg: W, hImg: H) }
            resized.append(x)
        }

        resized = resize(resized)

        let (l1Rn, l2Rn, l3Rn, l4Rn) = scratch.apply(resized)

        var out = refinenet4(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
        out = refinenet3(out, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
        out = refinenet2(out, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
        out = refinenet1(out, l1Rn)

        var fused = outputConv1(out)

        let hOut = ph * patchSize / downRatio
        let wOut = pw * patchSize / downRatio
        fused = bilinearInterpolate(fused, size: (hOut, wOut))
        if posEmbed { fused = addPosEmbed(fused, wImg: W, hImg: H) }

        let mainLogits = headForward(conv0: outputConv2_0, ln: outputConv2Ln, conv1: outputConv2_1, fused)

        var outs: [String: MLXArray] = [:]

        if hasConf {
            let pred = applyActivation(mainLogits[.ellipsis, ..<(outDim - 1)], activation)
            let conf = applyActivation(mainLogits[.ellipsis, (outDim - 1)], confActivation)
            outs[headMain] = pred.squeezed(axis: -1).reshaped([B, S, hOut, wOut])
            outs["\(headMain)_conf"] = conf.reshaped([B, S, hOut, wOut])
        } else {
            let pred = applyActivation(mainLogits, activation).squeezed(axis: -1)
            outs[headMain] = pred.reshaped([B, S, hOut, wOut])
        }

        if useSkyHead, let sc0 = skyConv2_0, let sc1 = skyConv2_1 {
            let skyLogits = headForward(conv0: sc0, ln: skyConv2Ln, conv1: sc1, fused)
            outs[skyName] = applyActivation(skyLogits.squeezed(axis: -1), skyActivation)
                .reshaped([B, S, hOut, wOut])
        }

        return outs
    }
}

// MARK: - DualDPT

/// Dual-head DPT: main depth head + auxiliary ray head.
public class DualDPT: Module {
    /// Diagnostic hooks: if non-nil, called with the input/output of the
    /// final aux head (`outputConv2Aux[-1]`). Used for python-parity dumps.
    public static var _dumpAuxLastInput: ((MLXArray) -> Void)?
    public static var _dumpAuxLogits: ((MLXArray) -> Void)?
    /// Per-level pyramid dumps. Called after each refinenet*Aux (before
    /// outputConv1Aux) and after outputConv1Aux. Index 0 = coarsest (level 4),
    /// index 3 = finest (level 1). Match python's `aux_list` ordering.
    public static var _dumpAuxPyrPre: ((Int, MLXArray) -> Void)?
    public static var _dumpAuxPyrPost: ((Int, MLXArray) -> Void)?
    /// Dumps of l1Rn..l4Rn (scratch outputs) for layout debugging.
    public static var _dumpScratch: ((Int, MLXArray) -> Void)?

    private let patchSize: Int
    private let activation: String
    private let confActivation: String
    private let posEmbed: Bool
    private let downRatio: Int
    private let auxLevels: Int
    private let auxOut1ConvNum: Int
    private let headMain: String
    private let headAux: String

    @ModuleInfo(key: "norm") private var norm: LayerNorm
    @ModuleInfo(key: "projects") private var projects: [Conv2d]
    @ModuleInfo(key: "resize_layer0") private var resizeLayer0: ConvTransposed2d
    @ModuleInfo(key: "resize_layer1") private var resizeLayer1: ConvTransposed2d
    @ModuleInfo(key: "resize_layer3") private var resizeLayer3: Conv2d
    @ModuleInfo(key: "scratch") private var scratch: Scratch

    // Main fusion chain
    @ModuleInfo(key: "refinenet1") private var refinenet1: FeatureFusionBlock
    @ModuleInfo(key: "refinenet2") private var refinenet2: FeatureFusionBlock
    @ModuleInfo(key: "refinenet3") private var refinenet3: FeatureFusionBlock
    @ModuleInfo(key: "refinenet4") private var refinenet4: FeatureFusionBlock

    // Main head
    @ModuleInfo(key: "output_conv1") private var outputConv1: Conv2d
    @ModuleInfo(key: "output_conv2_0") private var outputConv2_0: Conv2d
    @ModuleInfo(key: "output_conv2_1") private var outputConv2_1: Conv2d

    // Aux fusion chain
    @ModuleInfo(key: "refinenet1_aux") private var refinenet1Aux: FeatureFusionBlock
    @ModuleInfo(key: "refinenet2_aux") private var refinenet2Aux: FeatureFusionBlock
    @ModuleInfo(key: "refinenet3_aux") private var refinenet3Aux: FeatureFusionBlock
    @ModuleInfo(key: "refinenet4_aux") private var refinenet4Aux: FeatureFusionBlock

    // Aux pre-head (per level) — nested module arrays
    @ModuleInfo(key: "output_conv1_aux") private var outputConv1Aux: [[Conv2d]]

    // Aux final projection (per level)
    @ModuleInfo(key: "output_conv2_aux") private var outputConv2Aux: [AuxHead]

    public init(
        dimIn: Int,
        patchSize: Int = 14,
        outputDim: Int = 2,
        activation: String = "exp",
        confActivation: String = "expp1",
        features: Int = 256,
        outChannels: [Int] = [256, 512, 1024, 1024],
        posEmbed: Bool = true,
        downRatio: Int = 1,
        auxPyramidLevels: Int = 4,
        auxOut1ConvNum: Int = 5,
        headNames: (String, String) = ("depth", "ray")
    ) {
        self.patchSize = patchSize
        self.activation = activation
        self.confActivation = confActivation
        self.posEmbed = posEmbed
        self.downRatio = downRatio
        self.auxLevels = auxPyramidLevels
        self.auxOut1ConvNum = auxOut1ConvNum
        self.headMain = headNames.0
        self.headAux = headNames.1

        self._norm.wrappedValue = LayerNorm(dimensions: dimIn)

        self._projects.wrappedValue = outChannels.map { oc in
            Conv2d(inputChannels: dimIn, outputChannels: oc, kernelSize: .init(1), stride: .init(1), padding: .init(0))
        }

        self._resizeLayer0.wrappedValue = ConvTransposed2d(inputChannels: outChannels[0], outputChannels: outChannels[0], kernelSize: .init(4), stride: .init(4), padding: .init(0))
        self._resizeLayer1.wrappedValue = ConvTransposed2d(inputChannels: outChannels[1], outputChannels: outChannels[1], kernelSize: .init(2), stride: .init(2), padding: .init(0))
        self._resizeLayer3.wrappedValue = Conv2d(inputChannels: outChannels[3], outputChannels: outChannels[3], kernelSize: .init(3), stride: .init(2), padding: .init(1))

        self._scratch.wrappedValue = Scratch(inChannels: outChannels, outFeatures: features)

        // Main fusion
        self._refinenet1.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet2.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet3.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet4.wrappedValue = FeatureFusionBlock(features: features, hasResidual: false)

        // Main head
        let hf1 = features
        let hf2 = 32
        self._outputConv1.wrappedValue = Conv2d(inputChannels: hf1, outputChannels: hf1 / 2, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        self._outputConv2_0.wrappedValue = Conv2d(inputChannels: hf1 / 2, outputChannels: hf2, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        self._outputConv2_1.wrappedValue = Conv2d(inputChannels: hf2, outputChannels: outputDim, kernelSize: .init(1), stride: .init(1), padding: .init(0))

        // Aux fusion
        self._refinenet1Aux.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet2Aux.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet3Aux.wrappedValue = FeatureFusionBlock(features: features, hasResidual: true)
        self._refinenet4Aux.wrappedValue = FeatureFusionBlock(features: features, hasResidual: false)

        // Aux pre-head conv stacks (per level)
        self._outputConv1Aux.wrappedValue = (0 ..< auxPyramidLevels).map { _ in
            Self.makeAuxOut1Block(inCh: hf1, convNum: auxOut1ConvNum)
        }

        // Aux final projection (per level)
        self._outputConv2Aux.wrappedValue = (0 ..< auxPyramidLevels).map { _ in
            AuxHead(inCh: hf1 / 2, midCh: hf2, outDim: 7)
        }
    }

    private static func makeAuxOut1Block(inCh: Int, convNum: Int) -> [Conv2d] {
        var layers: [Conv2d] = []
        var ch = inCh
        for j in 0 ..< convNum {
            let outCh = (j % 2 == 0) ? inCh / 2 : inCh
            layers.append(Conv2d(inputChannels: ch, outputChannels: outCh, kernelSize: .init(3), stride: .init(1), padding: .init(1)))
            ch = outCh
        }
        return layers
    }

    private func applyAuxOut1(_ layers: [Conv2d], _ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }

    private func resize(_ feats: [MLXArray]) -> [MLXArray] {
        [resizeLayer0(feats[0]), resizeLayer1(feats[1]), feats[2], resizeLayer3(feats[3])]
    }

    public func callAsFunction(
        feats: [(MLXArray, MLXArray)],
        H: Int, W: Int,
        outputs requestedOutputs: DA3Outputs = .all
    ) -> [String: MLXArray] {
        if requestedOutputs == .all {
            return callAsFunction(feats: feats, H: H, W: W)
        }

        let B = feats[0].0.dim(0)
        let S = feats[0].0.dim(1)
        let BS = B * S
        let C = feats[0].0.dim(-1)
        let ph = H / patchSize
        let pw = W / patchSize

        // Project and reshape to spatial (NHWC)
        var resized: [MLXArray] = []
        for i in 0 ..< 4 {
            var x = feats[i].0.reshaped([BS, -1, C]) // [B*S, N, C]
            x = norm(x)
            x = x.reshaped([BS, ph, pw, C])
            x = projects[i](x)
            if posEmbed { x = addPosEmbed(x, wImg: W, hImg: H) }
            resized.append(x)
        }

        resized = resize(resized)

        // Scratch adapters
        let (l1Rn, l2Rn, l3Rn, l4Rn) = scratch.apply(resized)

        let hOut = ph * patchSize / downRatio
        let wOut = pw * patchSize / downRatio
        let wantsDepth = requestedOutputs.contains(.depth)
        let wantsDepthConf = requestedOutputs.contains(.depthConfidence)
        let wantsRay = requestedOutputs.contains(.ray)
        let wantsRayConf = requestedOutputs.contains(.rayConfidence)

        var outs: [String: MLXArray] = [:]
        let needsMain = wantsDepth || wantsDepthConf
        let needsAux = wantsRay || wantsRayConf

        if needsMain && needsAux {
            var out = refinenet4(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
            var auxOut = refinenet4Aux(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
            var auxList: [MLXArray] = []
            if auxLevels >= 4 { auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(0, auxOut) }

            out = refinenet3(out, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
            auxOut = refinenet3Aux(auxOut, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
            if auxLevels >= 3 { auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(1, auxOut) }

            out = refinenet2(out, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
            auxOut = refinenet2Aux(auxOut, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
            if auxLevels >= 2 { auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(2, auxOut) }

            out = refinenet1(out, l1Rn)
            auxOut = refinenet1Aux(auxOut, l1Rn)
            auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(3, auxOut)

            out = outputConv1(out)
            auxList = auxList.enumerated().map { i, a -> MLXArray in
                let r = applyAuxOut1(outputConv1Aux[i], a)
                DualDPT._dumpAuxPyrPost?(i, r)
                return r
            }

            var fusedMain = bilinearInterpolate(out, size: (hOut, wOut))
            if posEmbed { fusedMain = addPosEmbed(fusedMain, wImg: W, hImg: H) }

            var mainLogits = outputConv2_0(fusedMain)
            mainLogits = relu(mainLogits)
            mainLogits = outputConv2_1(mainLogits)

            if wantsDepth {
                let mainPred = applyActivation(mainLogits[.ellipsis, ..<(mainLogits.dim(-1) - 1)], activation)
                outs[headMain] = mainPred.squeezed(axis: -1).reshaped([B, S, hOut, wOut])
            }
            if wantsDepthConf {
                let mainConf = applyActivation(mainLogits[.ellipsis, (mainLogits.dim(-1) - 1)], confActivation)
                outs["\(headMain)_conf"] = mainConf.reshaped([B, S, hOut, wOut])
            }

            var lastAux = auxList[auxList.count - 1]
            if posEmbed { lastAux = addPosEmbed(lastAux, wImg: W, hImg: H) }
            DualDPT._dumpAuxLastInput?(lastAux)
            let auxLogits = outputConv2Aux[outputConv2Aux.count - 1](lastAux)
            DualDPT._dumpAuxLogits?(auxLogits)
            let auxH = auxLogits.dim(1)
            let auxW = auxLogits.dim(2)

            if wantsRay {
                let auxPred = auxLogits[.ellipsis, ..<(auxLogits.dim(-1) - 1)]
                outs[headAux] = auxPred.reshaped([B, S, auxH, auxW, -1])
            }
            if wantsRayConf {
                let auxConf = applyActivation(auxLogits[.ellipsis, (auxLogits.dim(-1) - 1)], confActivation)
                outs["\(headAux)_conf"] = auxConf.reshaped([B, S, auxH, auxW])
            }

            return outs
        }

        if needsMain {
            var out = refinenet4(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
            out = refinenet3(out, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
            out = refinenet2(out, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
            out = refinenet1(out, l1Rn)

            out = outputConv1(out)
            var fusedMain = bilinearInterpolate(out, size: (hOut, wOut))
            if posEmbed { fusedMain = addPosEmbed(fusedMain, wImg: W, hImg: H) }

            var mainLogits = outputConv2_0(fusedMain)
            mainLogits = relu(mainLogits)
            mainLogits = outputConv2_1(mainLogits)

            if wantsDepth {
                let mainPred = applyActivation(mainLogits[.ellipsis, ..<(mainLogits.dim(-1) - 1)], activation)
                outs[headMain] = mainPred.squeezed(axis: -1).reshaped([B, S, hOut, wOut])
            }
            if wantsDepthConf {
                let mainConf = applyActivation(mainLogits[.ellipsis, (mainLogits.dim(-1) - 1)], confActivation)
                outs["\(headMain)_conf"] = mainConf.reshaped([B, S, hOut, wOut])
            }
        }

        if needsAux {
            var auxOut = refinenet4Aux(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
            var auxList: [MLXArray] = []
            if auxLevels >= 4 { auxList.append(auxOut) }

            auxOut = refinenet3Aux(auxOut, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
            if auxLevels >= 3 { auxList.append(auxOut) }

            auxOut = refinenet2Aux(auxOut, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
            if auxLevels >= 2 { auxList.append(auxOut) }

            auxOut = refinenet1Aux(auxOut, l1Rn)
            auxList.append(auxOut)

            auxList = auxList.enumerated().map { i, a in applyAuxOut1(outputConv1Aux[i], a) }
            var lastAux = auxList[auxList.count - 1]
            if posEmbed { lastAux = addPosEmbed(lastAux, wImg: W, hImg: H) }
            DualDPT._dumpAuxLastInput?(lastAux)
            let auxLogits = outputConv2Aux[outputConv2Aux.count - 1](lastAux)
            DualDPT._dumpAuxLogits?(auxLogits)

            let auxH = auxLogits.dim(1)
            let auxW = auxLogits.dim(2)
            if wantsRay {
                let auxPred = auxLogits[.ellipsis, ..<(auxLogits.dim(-1) - 1)]
                outs[headAux] = auxPred.reshaped([B, S, auxH, auxW, -1])
            }
            if wantsRayConf {
                let auxConf = applyActivation(auxLogits[.ellipsis, (auxLogits.dim(-1) - 1)], confActivation)
                outs["\(headAux)_conf"] = auxConf.reshaped([B, S, auxH, auxW])
            }
        }

        return outs
    }

    public func callAsFunction(
        feats: [(MLXArray, MLXArray)],
        H: Int, W: Int
    ) -> [String: MLXArray] {
        let B = feats[0].0.dim(0)
        let S = feats[0].0.dim(1)
        let BS = B * S
        let C = feats[0].0.dim(-1)
        let ph = H / patchSize
        let pw = W / patchSize

        var resized: [MLXArray] = []
        for i in 0 ..< 4 {
            var x = feats[i].0.reshaped([BS, -1, C])
            x = norm(x)
            x = x.reshaped([BS, ph, pw, C])
            x = projects[i](x)
            if posEmbed { x = addPosEmbed(x, wImg: W, hImg: H) }
            resized.append(x)
        }

        resized = resize(resized)

        let (l1Rn, l2Rn, l3Rn, l4Rn) = scratch.apply(resized)
        DualDPT._dumpScratch?(0, l1Rn)
        DualDPT._dumpScratch?(1, l2Rn)
        DualDPT._dumpScratch?(2, l3Rn)
        DualDPT._dumpScratch?(3, l4Rn)

        var out = refinenet4(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
        var auxOut = refinenet4Aux(l4Rn, size: (l3Rn.dim(1), l3Rn.dim(2)))
        var auxList: [MLXArray] = []
        if auxLevels >= 4 { auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(0, auxOut) }

        out = refinenet3(out, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
        auxOut = refinenet3Aux(auxOut, l3Rn, size: (l2Rn.dim(1), l2Rn.dim(2)))
        if auxLevels >= 3 { auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(1, auxOut) }

        out = refinenet2(out, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
        auxOut = refinenet2Aux(auxOut, l2Rn, size: (l1Rn.dim(1), l1Rn.dim(2)))
        if auxLevels >= 2 { auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(2, auxOut) }

        out = refinenet1(out, l1Rn)
        auxOut = refinenet1Aux(auxOut, l1Rn)
        auxList.append(auxOut); DualDPT._dumpAuxPyrPre?(3, auxOut)

        out = outputConv1(out)
        auxList = auxList.enumerated().map { i, a -> MLXArray in
            let r = applyAuxOut1(outputConv1Aux[i], a)
            DualDPT._dumpAuxPyrPost?(i, r)
            return r
        }

        let hOut = ph * patchSize / downRatio
        let wOut = pw * patchSize / downRatio
        var fusedMain = bilinearInterpolate(out, size: (hOut, wOut))
        if posEmbed { fusedMain = addPosEmbed(fusedMain, wImg: W, hImg: H) }

        var mainLogits = outputConv2_0(fusedMain)
        mainLogits = relu(mainLogits)
        mainLogits = outputConv2_1(mainLogits)

        let mainPred = applyActivation(mainLogits[.ellipsis, ..<(mainLogits.dim(-1) - 1)], activation)
        let mainConf = applyActivation(mainLogits[.ellipsis, (mainLogits.dim(-1) - 1)], confActivation)

        var lastAux = auxList[auxList.count - 1]
        if posEmbed { lastAux = addPosEmbed(lastAux, wImg: W, hImg: H) }
        DualDPT._dumpAuxLastInput?(lastAux)
        let auxLogits = outputConv2Aux[outputConv2Aux.count - 1](lastAux)
        DualDPT._dumpAuxLogits?(auxLogits)

        let auxPred = auxLogits[.ellipsis, ..<(auxLogits.dim(-1) - 1)]
        let auxConf = applyActivation(auxLogits[.ellipsis, (auxLogits.dim(-1) - 1)], confActivation)
        let auxH = auxLogits.dim(1)
        let auxW = auxLogits.dim(2)

        return [
            headMain: mainPred.squeezed(axis: -1).reshaped([B, S, hOut, wOut]),
            "\(headMain)_conf": mainConf.reshaped([B, S, hOut, wOut]),
            headAux: auxPred.reshaped([B, S, auxH, auxW, -1]),
            "\(headAux)_conf": auxConf.reshaped([B, S, auxH, auxW]),
        ]
    }
}
