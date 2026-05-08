import MLX

public struct BackboneConfig {
    public let embedDim: Int
    public let depth: Int
    public let numHeads: Int
    public let mlpRatio: Float
    public let useSwiglu: Bool
    
    public init(
        embedDim: Int,
        depth: Int,
        numHeads: Int,
        mlpRatio: Float = 4.0,
        useSwiglu: Bool = false
    ) {
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
        self.mlpRatio = mlpRatio
        self.useSwiglu = useSwiglu
    }
    
    public static let vits = BackboneConfig(embedDim: 384, depth: 12, numHeads: 6)
    public static let vitb = BackboneConfig(embedDim: 768, depth: 12, numHeads: 12)
    public static let vitl = BackboneConfig(embedDim: 1024, depth: 24, numHeads: 16)
    public static let vitg = BackboneConfig(embedDim: 1536, depth: 40, numHeads: 24, useSwiglu: true)
    
    public static func fromName(_ name: String) -> BackboneConfig {
        switch name.lowercased() {
        case "vits": return vits
        case "vitb": return vitb
        case "vitl": return vitl
        case "vitg": return vitg
        default:
            let available = ["vits", "vitb", "vitl", "vitg"].joined(separator: ", ")
            fatalError("Unknown backbone '\(name)'. Available: \(available)")
        }
    }
}

public struct HeadConfig {
    public let dimIn: Int
    public let features: Int
    public let outChannels: [Int]
    public let outputDim: Int
    public let posEmbed: Bool
    public let useSkyHead: Bool
    public let normType: String
    
    public init(
        dimIn: Int,
        features: Int,
        outChannels: [Int],
        outputDim: Int,
        posEmbed: Bool = true,
        useSkyHead: Bool = false,
        normType: String = "ln"
    ) {
        self.dimIn = dimIn
        self.features = features
        self.outChannels = outChannels
        self.outputDim = outputDim
        self.posEmbed = posEmbed
        self.useSkyHead = useSkyHead
        self.normType = normType
    }
}

public struct DA3Config {
    public let name: String
    public let backbone: BackboneConfig
    public let outLayers: [Int]
    public let altStart: Int
    public let qknormStart: Int
    public let ropeStart: Int
    public let catToken: Bool
    public let headType: String
    public let head: HeadConfig
    
    public init(
        name: String,
        backbone: BackboneConfig,
        outLayers: [Int],
        altStart: Int,
        qknormStart: Int,
        ropeStart: Int,
        catToken: Bool,
        headType: String,
        head: HeadConfig
    ) {
        self.name = name
        self.backbone = backbone
        self.outLayers = outLayers
        self.altStart = altStart
        self.qknormStart = qknormStart
        self.ropeStart = ropeStart
        self.catToken = catToken
        self.headType = headType
        self.head = head
    }
    
    public static let da3Small = DA3Config(
        name: "da3-small",
        backbone: .vits,
        outLayers: [5, 7, 9, 11],
        altStart: 4,
        qknormStart: 4,
        ropeStart: 4,
        catToken: true,
        headType: "dualdpt",
        head: HeadConfig(
            dimIn: 768,
            features: 64,
            outChannels: [48, 96, 192, 384],
            outputDim: 2
        )
    )
    
    public static let da3Base = DA3Config(
        name: "da3-base",
        backbone: .vitb,
        outLayers: [5, 7, 9, 11],
        altStart: 4,
        qknormStart: 4,
        ropeStart: 4,
        catToken: true,
        headType: "dualdpt",
        head: HeadConfig(
            dimIn: 1536,
            features: 128,
            outChannels: [96, 192, 384, 768],
            outputDim: 2
        )
    )
    
    public static let da3Large = DA3Config(
        name: "da3-large",
        backbone: .vitl,
        outLayers: [11, 15, 19, 23],
        altStart: 8,
        qknormStart: 8,
        ropeStart: 8,
        catToken: true,
        headType: "dualdpt",
        head: HeadConfig(
            dimIn: 2048,
            features: 256,
            outChannels: [256, 512, 1024, 1024],
            outputDim: 2
        )
    )
    
    public static let da3Giant = DA3Config(
        name: "da3-giant",
        backbone: .vitg,
        outLayers: [19, 27, 33, 39],
        altStart: 13,
        qknormStart: 13,
        ropeStart: 13,
        catToken: true,
        headType: "dualdpt",
        head: HeadConfig(
            dimIn: 3072,
            features: 256,
            outChannels: [256, 512, 1024, 1024],
            outputDim: 2
        )
    )
    
    public static let da3monoLarge = DA3Config(
        name: "da3mono-large",
        backbone: .vitl,
        outLayers: [4, 11, 17, 23],
        altStart: -1,
        qknormStart: -1,
        ropeStart: -1,
        catToken: false,
        headType: "dpt",
        head: HeadConfig(
            dimIn: 1024,
            features: 256,
            outChannels: [256, 512, 1024, 1024],
            outputDim: 1,
            posEmbed: false,
            useSkyHead: true,
            normType: "idt"
        )
    )
    
    public static func fromName(_ name: String) -> DA3Config {
        switch name.lowercased() {
        case "da3-small": return da3Small
        case "da3-base": return da3Base
        case "da3-large": return da3Large
        case "da3-giant": return da3Giant
        case "da3mono-large": return da3monoLarge
        default:
            let available = ["da3-small", "da3-base", "da3-large", "da3-giant", "da3mono-large"].joined(separator: ", ")
            fatalError("Unknown model '\(name)'. Available: \(available)")
        }
    }
}
