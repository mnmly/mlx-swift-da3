import MLX
import MLXNN

public struct DA3Outputs: OptionSet, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let depth = DA3Outputs(rawValue: 1 << 0)
    public static let depthConfidence = DA3Outputs(rawValue: 1 << 1)
    public static let ray = DA3Outputs(rawValue: 1 << 2)
    public static let rayConfidence = DA3Outputs(rawValue: 1 << 3)
    public static let sky = DA3Outputs(rawValue: 1 << 4)

    public static let depthOnly: DA3Outputs = [.depth]
    public static let depthWithConfidence: DA3Outputs = [.depth, .depthConfidence]
    public static let all: DA3Outputs = [.depth, .depthConfidence, .ray, .rayConfidence, .sky]
}

/// Depth Anything 3 — depth estimation from single or multi-view images.
///
/// Combines a DinoV2 backbone with a DPT/DualDPT decoder head.
/// For inference-only; does not include camera encoder/decoder or GS heads.
public class DepthAnything3: Module {
    public static let patchSize = 14

    @ModuleInfo(key: "backbone") public var backbone: DinoVisionTransformer
    @ModuleInfo(key: "head") private var head: Module

    private let headType: String

    public init(backbone: DinoVisionTransformer, head: DPT) {
        self.headType = "dpt"
        self._backbone.wrappedValue = backbone
        self._head.wrappedValue = head
    }

    public init(backbone: DinoVisionTransformer, head: DualDPT) {
        self.headType = "dualdpt"
        self._backbone.wrappedValue = backbone
        self._head.wrappedValue = head
    }

    /// Forward pass.
    ///
    /// - Parameter x: Input images `[B, S, H, W, C]` (NHWC) where S is number of views.
    ///   For single image: `[1, 1, H, W, 3]`
    /// - Returns: Dict with keys like "depth", "depth_conf", "ray", "ray_conf", "sky"
    public func callAsFunction(_ x: MLXArray) -> [String: MLXArray] {
        let H = x.dim(2)
        let W = x.dim(3)

        let (feats, _) = backbone(x)

        if let dualHead = head as? DualDPT {
            return dualHead.callAsFunction(feats: feats, H: H, W: W)
        } else if let dptHead = head as? DPT {
            return dptHead.callAsFunction(feats: feats, H: H, W: W)
        } else {
            fatalError("Unknown head type")
        }
    }

    public func callAsFunction(_ x: MLXArray, outputs requestedOutputs: DA3Outputs = .all) -> [String: MLXArray] {
        if requestedOutputs == .all {
            return callAsFunction(x)
        }

        let H = x.dim(2)
        let W = x.dim(3)

        // Backbone: extract multi-scale features
        let (feats, _) = backbone(x)

        // Head: decode features to depth
        if let dualHead = head as? DualDPT {
            return dualHead.callAsFunction(feats: feats, H: H, W: W, outputs: requestedOutputs)
        } else if let dptHead = head as? DPT {
            return dptHead.callAsFunction(feats: feats, H: H, W: W, outputs: requestedOutputs)
        } else {
            fatalError("Unknown head type")
        }
    }
}

// MARK: - Factory

/// Build a DepthAnything3 model from a config name.
///
/// - Parameter configName: One of "da3-small", "da3-base", "da3-large", "da3-giant", "da3mono-large"
/// - Returns: Uninitialized model (use `loadModel` for weights)
public func buildModel(configName: String) -> DepthAnything3 {
    let cfg = DA3Config.fromName(configName)
    let bb = cfg.backbone

    let backbone = DinoVisionTransformer(
        imgSize: 518,
        patchSize: 14,
        inChannels: 3,
        embedDim: bb.embedDim,
        depth: bb.depth,
        numHeads: bb.numHeads,
        mlpRatio: bb.mlpRatio,
        qkvBias: true,
        projBias: true,
        ffnBias: true,
        initValues: 1.0,
        useSwiglu: bb.useSwiglu,
        altStart: cfg.altStart,
        qknormStart: cfg.qknormStart,
        ropeStart: cfg.ropeStart,
        catToken: cfg.catToken,
        outLayers: cfg.outLayers
    )

    let hc = cfg.head

    if cfg.headType == "dualdpt" {
        let head = DualDPT(
            dimIn: hc.dimIn,
            patchSize: 14,
            outputDim: hc.outputDim,
            features: hc.features,
            outChannels: hc.outChannels,
            posEmbed: hc.posEmbed
        )
        return DepthAnything3(backbone: backbone, head: head)
    } else {
        let head = DPT(
            dimIn: hc.dimIn,
            patchSize: 14,
            outputDim: hc.outputDim,
            features: hc.features,
            outChannels: hc.outChannels,
            posEmbed: hc.posEmbed,
            useSkyHead: hc.useSkyHead,
            normType: hc.normType
        )
        return DepthAnything3(backbone: backbone, head: head)
    }
}
