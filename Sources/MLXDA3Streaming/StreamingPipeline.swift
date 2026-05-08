import CoreGraphics
import Foundation
import MLX
import MLXDA3

/// In-memory result of a streaming inference run.
///
/// Mirrors the high-level outputs of python `da3_streaming.py` but as
/// in-memory tensors instead of files on disk. File-IO callers should use
/// ``StreamingOrchestrator/run(imagePaths:outputDir:)`` which wraps this
/// prediction with `camera_poses.txt`, `intrinsic.txt`, and per-chunk PLY
/// writes.
public struct StreamingPrediction {
    /// Per-frame camera-to-world 4×4 poses, shape `[N, 4, 4]`.
    public let cameraPosesC2W: MLXArray
    /// Per-frame intrinsics, shape `[N, 3, 3]`.
    public let intrinsicsK: MLXArray
    /// Sim(3) transforms that map each chunk i (i≥1) into chunk-0's frame.
    /// Length = number of chunks - 1.
    public let cumulativeSim3: [Sim3Alignment.Sim3]
    /// Pair-wise Sim(3) transforms between adjacent chunks (i+1 → i).
    public let pairwiseSim3: [Sim3Alignment.Sim3]
    /// Chunk index ranges (start, end) for each chunk, half-open.
    public let chunkRanges: [(Int, Int)]
    /// Number of input frames.
    public let totalFrames: Int
    /// Processed image height (after preprocessing/resize).
    public let processedHeight: Int
    /// Processed image width.
    public let processedWidth: Int
    /// Per-chunk raw predictions retained for callers that want world points,
    /// confidence, or processed RGB frames. Lifetime managed by caller.
    public let perChunk: [ChunkPredictions]
}

/// High-level streaming pipeline. Mirrors ``DepthAnything3Pipeline`` for the
/// chunk-streaming use case (multi-chunk inference + Sim(3) alignment).
///
/// Use this in two ways:
/// - ``predict(images:)`` for in-memory results (e.g. parity tests, batching).
/// - Wrap with ``StreamingOrchestrator`` for file-based output (PLYs, txt).
public struct StreamingPipeline {

    public struct Config {
        public var chunkSize: Int = 8
        public var overlap: Int = 4
        public var resolution: Int = 504
        public var dtype: DType = .float16
        public var pcdSampleRatio: Float = 1.0
        public var pcdConfThresholdCoef: Float = 0.75
        public var irls: Sim3Alignment.Config = Sim3Alignment.Config()
        public var sim3Optimizer: Sim3LoopOptimizer.Config = Sim3LoopOptimizer.Config()
        public var verbose: Bool = false

        public init() {}
    }

    public let model: DepthAnything3
    public let config: Config

    public init(model: DepthAnything3, config: Config = Config()) {
        self.model = model
        self.config = config
    }

    /// Run the full chunk-streaming pipeline over the supplied frames and
    /// return the in-memory prediction (poses + intrinsics + per-chunk state).
    /// No file IO.
    ///
    /// - Parameter loopConstraints: optional external Sim(3) measurements
    ///   between non-adjacent chunk pairs (loop closures). If non-empty,
    ///   `Sim3LoopOptimizer` is run after the sequential alignment phase
    ///   to refine the chunk-pair Sim(3) chain before pose accumulation.
    public func predict(
        images: [CGImage],
        loopConstraints: [LoopConstraint] = []
    ) -> StreamingPrediction {
        let preproc = MultiViewPreprocessor(processRes: config.resolution)
        let inference = StreamingInference(model: model, preprocessor: preproc, dtype: config.dtype)

        let chunks = ChunkIndex.compute(
            numImages: images.count,
            chunkSize: config.chunkSize,
            overlap: config.overlap
        )
        if config.verbose {
            print("Processing \(images.count) images in \(chunks.count) chunks (size=\(config.chunkSize), overlap=\(config.overlap))")
        }

        var perChunk: [ChunkPredictions] = []
        var sim3Pairs: [Sim3Alignment.Sim3] = []  // length = chunks - 1

        var procH = 0
        var procW = 0

        for (chunkIdx, range) in chunks.enumerated() {
            if config.verbose {
                print("[chunk \(chunkIdx)/\(chunks.count - 1)] frames \(range.0)..<\(range.1)")
            }
            let chunkImages = Array(images[range.0..<range.1])
            let pred = inference.predict(images: chunkImages)
            procH = pred.height
            procW = pred.width
            perChunk.append(pred)

            if chunkIdx > 0 {
                let prev = perChunk[chunkIdx - 1]
                let cur = pred
                guard let prevExt = prev.extrinsics, let prevInt = prev.intrinsics,
                      let curExt = cur.extrinsics, let curInt = cur.intrinsics
                else {
                    sim3Pairs.append(.identity)
                    continue
                }
                let pmPrev = PointMaps.depthToWorldPoints(depth: prev.depth, intrinsics: prevInt, extrinsicsW2C: prevExt)
                let pmCur = PointMaps.depthToWorldPoints(depth: cur.depth, intrinsics: curInt, extrinsicsW2C: curExt)
                eval(pmPrev, pmCur)

                let O = config.overlap
                let pmPrevTail = pmPrev[(prev.viewCount - O)..<prev.viewCount]
                let pmCurHead = pmCur[0..<O]
                let confPrevTail = prev.conf[(prev.viewCount - O)..<prev.viewCount]
                let confCurHead = cur.conf[0..<O]

                let xform = ChunkAlignment.alignWeighted(
                    pointMap1: pmPrevTail, conf1: confPrevTail,
                    pointMap2: pmCurHead, conf2: confCurHead,
                    irlsConfig: config.irls
                )
                if config.verbose {
                    print(String(format: "  sim3 s=%.4f t=[%.3f, %.3f, %.3f]",
                                 xform.s, xform.t[0], xform.t[1], xform.t[2]))
                }
                sim3Pairs.append(xform)
            }
        }

        // Optional loop-closure refinement: run pose-graph LM to update the
        // chunk-pair Sim(3) chain using `loopConstraints` as extra edges in
        // the graph.
        var refinedPairs = sim3Pairs
        if !loopConstraints.isEmpty && !sim3Pairs.isEmpty {
            // Convert MLX Sim(3) (s, R, t) → Sim3 Lie type
            let seqInput: [Sim3] = sim3Pairs.map { sim3FromMLX($0) }
            let optimizer = Sim3LoopOptimizer(config: config.sim3Optimizer)
            let optimized = optimizer.optimize(
                sequentialTransforms: seqInput,
                loopConstraints: loopConstraints
            )
            refinedPairs = optimized.map { mlxFromSim3($0) }
            if config.verbose {
                print("Loop closure: optimised \(seqInput.count) sequential transforms with \(loopConstraints.count) loop edge(s)")
            }
        }

        let cumulative = PointMaps.accumulate(refinedPairs)

        let posesInputs = CameraPosesIO.Inputs(
            chunkExtrinsicsW2C: perChunk.map { $0.extrinsics ?? MLXArray.zeros([0, 3, 4], dtype: .float32) },
            chunkIntrinsics:    perChunk.map { $0.intrinsics ?? MLXArray.zeros([0, 3, 3], dtype: .float32) },
            chunkRanges: chunks,
            cumulativeSim3: cumulative,
            overlap: config.overlap,
            totalFrames: images.count
        )
        let poses = CameraPosesIO.computePoses(posesInputs)

        return StreamingPrediction(
            cameraPosesC2W: poses.cameraPosesC2W,
            intrinsicsK: poses.intrinsicsK,
            cumulativeSim3: cumulative,
            pairwiseSim3: sim3Pairs,
            chunkRanges: chunks,
            totalFrames: images.count,
            processedHeight: procH,
            processedWidth: procW,
            perChunk: perChunk
        )
    }
}

public extension StreamingPipeline {
    /// Compute a Sim(3) loop measurement between two chunks.
    ///
    /// Mirrors python's `get_loop_sim3_from_loop_predict`: runs DA3 on the
    /// combined `framesA + framesB` set, splits the loop output back into A
    /// and B halves, aligns each half with the corresponding original chunk's
    /// world-points, and composes the two transforms into a single chunk-A →
    /// chunk-B Sim(3) measurement suitable for `LoopConstraint`.
    ///
    /// - Parameters:
    ///   - chunkA, chunkB: per-chunk predictions from a previous `predict(...)`
    ///     run (use the corresponding entries from `prediction.perChunk`).
    ///   - framesA: the original CGImages that produced `chunkA`. Must match
    ///     `chunkA.viewCount`.
    ///   - framesB: same, for chunk B.
    ///   - chunkIdxA, chunkIdxB: positions in the optimizer's pose sequence
    ///     (typically the chunk indices in `prediction.chunkRanges`).
    func computeLoopMeasurement(
        chunkA: ChunkPredictions,
        framesA: [CGImage],
        chunkAIdx: Int,
        chunkB: ChunkPredictions,
        framesB: [CGImage],
        chunkBIdx: Int
    ) -> LoopConstraint? {
        precondition(framesA.count == chunkA.viewCount, "framesA / chunkA size mismatch")
        precondition(framesB.count == chunkB.viewCount, "framesB / chunkB size mismatch")
        guard let extA = chunkA.extrinsics, let intA = chunkA.intrinsics,
              let extB = chunkB.extrinsics, let intB = chunkB.intrinsics
        else { return nil }

        // Run DA3 on the combined frames.
        let preproc = MultiViewPreprocessor(processRes: config.resolution)
        let inference = StreamingInference(model: model, preprocessor: preproc, dtype: config.dtype)
        let combined = framesA + framesB
        let pred = inference.predict(images: combined)
        guard let loopExt = pred.extrinsics, let loopInt = pred.intrinsics else { return nil }

        // World points for each set.
        let pmLoop = PointMaps.depthToWorldPoints(
            depth: pred.depth, intrinsics: loopInt, extrinsicsW2C: loopExt
        )
        let pmA = PointMaps.depthToWorldPoints(
            depth: chunkA.depth, intrinsics: intA, extrinsicsW2C: extA
        )
        let pmB = PointMaps.depthToWorldPoints(
            depth: chunkB.depth, intrinsics: intB, extrinsicsW2C: extB
        )
        eval(pmLoop, pmA, pmB)

        // Slice loop output into A and B halves.
        let nA = framesA.count
        let nB = framesB.count
        let pmLoopA = pmLoop[0..<nA]
        let pmLoopB = pmLoop[nA..<(nA + nB)]
        let confLoopA = pred.conf[0..<nA]
        let confLoopB = pred.conf[nA..<(nA + nB)]

        // Align loop's A-half with chunkA's world points.
        let sA = ChunkAlignment.alignWeighted(
            pointMap1: pmA, conf1: chunkA.conf,
            pointMap2: pmLoopA, conf2: confLoopA,
            irlsConfig: config.irls
        )
        // Align loop's B-half with chunkB's world points.
        let sB = ChunkAlignment.alignWeighted(
            pointMap1: pmB, conf1: chunkB.conf,
            pointMap2: pmLoopB, conf2: confLoopB,
            irlsConfig: config.irls
        )

        // Compose: chunk-A → chunk-B Sim(3).
        // s_ab = s_b / s_a;  R_ab = R_b · R_aᵀ;  t_ab = t_b − s_ab · R_ab · t_a
        let s_ab = sB.s / max(sA.s, 1e-9)
        let R_aT: [Float] = transpose3x3(sA.R)
        let R_ab: [Float] = matmul3x3(sB.R, R_aT)
        let R_ab_t_a: [Float] = matvec3(R_ab, sA.t)
        let t_ab: [Float] = [
            sB.t[0] - s_ab * R_ab_t_a[0],
            sB.t[1] - s_ab * R_ab_t_a[1],
            sB.t[2] - s_ab * R_ab_t_a[2],
        ]

        let m = Sim3(
            R: R_ab.map { Double($0) },
            t: t_ab.map { Double($0) },
            s: max(Double(s_ab), 1e-9)
        )
        return LoopConstraint(i: chunkAIdx, j: chunkBIdx, measurement: m)
    }

    /// Convenience: load model from path and construct a pipeline with default
    /// streaming config.
    static func fromPretrained(
        _ path: String,
        configName: String? = nil,
        config: Config = Config()
    ) throws -> StreamingPipeline {
        let model = try DepthAnything3.fromPretrained(path, configName: configName, dtype: config.dtype)
        return StreamingPipeline(model: model, config: config)
    }
}

// MARK: - Sim3Alignment.Sim3 ↔ Sim3 (Lie group) converters

/// Bridge between the MLX-resident `Sim3Alignment.Sim3` (Float arrays) and
/// the CPU-side `Sim3` Lie-group type used by `Sim3LoopOptimizer`.
private func sim3FromMLX(_ x: Sim3Alignment.Sim3) -> Sim3 {
    let r = x.R.map { Double($0) }
    let t = x.t.map { Double($0) }
    return Sim3(R: r, t: t, s: max(Double(x.s), 1e-9))
}

private func mlxFromSim3(_ x: Sim3) -> Sim3Alignment.Sim3 {
    Sim3Alignment.Sim3(
        s: Float(x.s),
        R: x.R.map { Float($0) },
        t: x.t.map { Float($0) }
    )
}

private func transpose3x3(_ a: [Float]) -> [Float] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

private func matmul3x3(_ a: [Float], _ b: [Float]) -> [Float] {
    var out = [Float](repeating: 0, count: 9)
    for i in 0..<3 {
        for j in 0..<3 {
            var s: Float = 0
            for k in 0..<3 { s += a[i * 3 + k] * b[k * 3 + j] }
            out[i * 3 + j] = s
        }
    }
    return out
}

private func matvec3(_ a: [Float], _ v: [Float]) -> [Float] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

/// Top-level convenience namespace, mirroring ``MLXDA3``.
public enum MLXDA3Streaming {
    public static func fromPretrained(
        _ path: String,
        configName: String? = nil,
        config: StreamingPipeline.Config = StreamingPipeline.Config()
    ) throws -> StreamingPipeline {
        try StreamingPipeline.fromPretrained(path, configName: configName, config: config)
    }
}
