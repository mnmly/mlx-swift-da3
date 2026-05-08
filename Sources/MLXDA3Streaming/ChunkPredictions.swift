import Foundation
import MLX

/// Per-chunk model output, mirroring the Python `predictions` namespace.
///
/// Shapes (all CPU-resident MLXArrays at evaluation time):
/// - `processedImages`: `[S, H, W, 3]` UInt8
/// - `depth`: `[S, H, W]` Float32 (squeezed channel)
/// - `conf`: `[S, H, W]` Float32  (depth_conf - 1.0, matches Python)
/// - `ray`: `[S, Hp, Wp, 6]` Float32 (only present in slices that need it)
/// - `rayConf`: `[S, Hp, Wp]` Float32
/// - `intrinsics`: `[S, 3, 3]` Float32 (filled in slice 1B)
/// - `extrinsics`: `[S, 3, 4]` Float32 w2c (filled in slice 1B)
public struct ChunkPredictions {
    public var processedImages: MLXArray
    public var depth: MLXArray
    public var conf: MLXArray
    public var ray: MLXArray?
    public var rayConf: MLXArray?
    public var intrinsics: MLXArray?
    public var extrinsics: MLXArray?

    public var viewCount: Int { processedImages.dim(0) }
    public var height: Int { processedImages.dim(1) }
    public var width: Int { processedImages.dim(2) }

    public init(
        processedImages: MLXArray,
        depth: MLXArray,
        conf: MLXArray,
        ray: MLXArray? = nil,
        rayConf: MLXArray? = nil,
        intrinsics: MLXArray? = nil,
        extrinsics: MLXArray? = nil
    ) {
        self.processedImages = processedImages
        self.depth = depth
        self.conf = conf
        self.ray = ray
        self.rayConf = rayConf
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
    }
}
