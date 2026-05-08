import CoreGraphics
import Foundation
import MLX
import MLXDA3

/// Phase 1 streaming orchestrator (no loop closure). Mirrors the non-loop path of
/// `DA3_Streaming.process_long_sequence` in the python reference.
public struct StreamingOrchestrator {

    public struct Config {
        public var chunkSize: Int = 8
        public var overlap: Int = 4
        public var resolution: Int = 504
        public var dtype: DType = .float16
        public var pcdSampleRatio: Float = 1.0
        public var pcdConfThresholdCoef: Float = 0.75
        public var irls: Sim3Alignment.Config = Sim3Alignment.Config()
        /// If non-nil, per-chunk depth/conf/intrinsics/extrinsics + sim3 list are
        /// dumped as .npy files into this directory for python parity diff.
        public var dumpDir: String? = nil
        public init() {}
    }

    public let model: DepthAnything3
    public let config: Config

    public init(model: DepthAnything3, config: Config = Config()) {
        self.model = model
        self.config = config
    }

    /// Run the full pipeline. Outputs:
    ///  - `<outputDir>/camera_poses.txt`
    ///  - `<outputDir>/intrinsic.txt`
    ///  - `<outputDir>/pcd/<i>_pcd.ply`
    ///  - `<outputDir>/pcd/combined_pcd.ply`
    public func run(imagePaths: [String], outputDir: String) throws {
        let fm = FileManager.default
        let pcdDir = (outputDir as NSString).appendingPathComponent("pcd")
        try fm.createDirectory(atPath: pcdDir, withIntermediateDirectories: true)
        if let dd = config.dumpDir {
            try fm.createDirectory(atPath: dd, withIntermediateDirectories: true)
        }

        let preproc = MultiViewPreprocessor(processRes: config.resolution)
        let inference = StreamingInference(model: model, preprocessor: preproc, dtype: config.dtype)

        // Debug dump: cameraToken values to verify weight loading.
        if let dd = config.dumpDir, let ct = model.backbone.cameraToken {
            try? NpyWriter.writeFloat32(ct.asType(.float32), to: (dd as NSString).appendingPathComponent("camera_token.npy"))
        }

        let chunks = ChunkIndex.compute(
            numImages: imagePaths.count,
            chunkSize: config.chunkSize,
            overlap: config.overlap
        )
        print("Processing \(imagePaths.count) images in \(chunks.count) chunks (size=\(config.chunkSize), overlap=\(config.overlap))")

        // Per-chunk predictions held until pose accumulation phase. For large runs
        // this is heavier than python (which spills .npy to disk between phases),
        // but for Phase 1 a few chunks at chunk_size=8 is fine.
        var perChunk: [ChunkPredictions] = []
        var sim3Pairs: [Sim3Alignment.Sim3] = []  // length = chunks - 1, transform mapping chunk_{i+1} → chunk_i

        for (chunkIdx, range) in chunks.enumerated() {
            print("[chunk \(chunkIdx)/\(chunks.count - 1)] frames \(range.0)..<\(range.1)")
            let images = try (range.0..<range.1).map { try ImageDirectory.loadCGImage(path: imagePaths[$0]) }
            let t0 = CFAbsoluteTimeGetCurrent()
            let inputDumpPath: String? = config.dumpDir.map { ($0 as NSString).appendingPathComponent("chunk\(chunkIdx)_input.npy") }
            let backboneDumpPath: String? = config.dumpDir.map { ($0 as NSString).appendingPathComponent("chunk\(chunkIdx)_backbone_last.npy") }
            if let dd = config.dumpDir {
                let auxInPath = (dd as NSString).appendingPathComponent("chunk\(chunkIdx)_aux_last_input.npy")
                let auxLogitsPath = (dd as NSString).appendingPathComponent("chunk\(chunkIdx)_aux_logits.npy")
                DualDPT._dumpAuxLastInput = { x in try? NpyWriter.writeFloat32(x.asType(.float32), to: auxInPath) }
                DualDPT._dumpAuxLogits = { x in try? NpyWriter.writeFloat32(x.asType(.float32), to: auxLogitsPath) }
                DualDPT._dumpAuxPyrPre = { i, x in
                    let p = (dd as NSString).appendingPathComponent("chunk\(chunkIdx)_aux_pyr_pre_\(i).npy")
                    try? NpyWriter.writeFloat32(x.asType(.float32), to: p)
                }
                DualDPT._dumpAuxPyrPost = { i, x in
                    let p = (dd as NSString).appendingPathComponent("chunk\(chunkIdx)_aux_pyr_post_\(i).npy")
                    try? NpyWriter.writeFloat32(x.asType(.float32), to: p)
                }
                DualDPT._dumpScratch = { i, x in
                    let p = (dd as NSString).appendingPathComponent("chunk\(chunkIdx)_scratch_l\(i+1)Rn.npy")
                    try? NpyWriter.writeFloat32(x.asType(.float32), to: p)
                }
            }
            let pred = inference.predict(images: images, dumpInputPath: inputDumpPath, dumpBackbonePath: backboneDumpPath)
            DualDPT._dumpAuxLastInput = nil
            DualDPT._dumpAuxLogits = nil
            DualDPT._dumpAuxPyrPre = nil
            DualDPT._dumpAuxPyrPost = nil
            DualDPT._dumpScratch = nil
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "  inference %.2fs, depth=%@, conf=%@", dt, "\(pred.depth.shape)", "\(pred.conf.shape)"))

            // Compute world points for this chunk in its own (unaligned) coordinate frame.
            // We retain depth/conf/processedImages in `pred`; world points are derived now.
            perChunk.append(pred)

            if let dd = config.dumpDir {
                try dumpChunk(pred, chunkIdx: chunkIdx, dir: dd)
            }

            if chunkIdx > 0 {
                let prev = perChunk[chunkIdx - 1]
                let cur = pred
                guard let prevExt = prev.extrinsics, let prevInt = prev.intrinsics,
                      let curExt = cur.extrinsics, let curInt = cur.intrinsics
                else {
                    print("  WARNING: missing camera info — skipping alignment for this chunk")
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
                print(String(format: "  sim3 s=%.4f t=[%.3f, %.3f, %.3f]",
                             xform.s, xform.t[0], xform.t[1], xform.t[2]))
                sim3Pairs.append(xform)
            }
        }

        if let dd = config.dumpDir {
            try dumpSim3List(sim3Pairs, dir: dd)
        }

        // Accumulate transforms: cumulative[i] maps chunk_{i+1} → chunk_0.
        let cumulative = PointMaps.accumulate(sim3Pairs)

        // Write per-chunk PLY clouds (transformed into chunk-0 frame).
        try writeChunkPLYs(perChunk: perChunk, cumulative: cumulative, pcdDir: pcdDir)

        // Camera poses + intrinsics.
        let posesInputs = CameraPosesIO.Inputs(
            chunkExtrinsicsW2C: perChunk.map { $0.extrinsics ?? MLXArray.zeros([0, 3, 4], dtype: .float32) },
            chunkIntrinsics:    perChunk.map { $0.intrinsics ?? MLXArray.zeros([0, 3, 3], dtype: .float32) },
            chunkRanges: chunks,
            cumulativeSim3: cumulative,
            overlap: config.overlap,
            totalFrames: imagePaths.count
        )
        try CameraPosesIO.savePoses(posesInputs, outputDir: outputDir)

        // Merge per-chunk PLYs.
        let combined = (pcdDir as NSString).appendingPathComponent("combined_pcd.ply")
        try PLYWriter.mergePLYFiles(inputDir: pcdDir, outputPath: combined)
    }

    private func writeChunkPLYs(
        perChunk: [ChunkPredictions], cumulative: [Sim3Alignment.Sim3], pcdDir: String
    ) throws {
        // Chunk 0: untransformed.
        let p0 = perChunk[0]
        guard let int0 = p0.intrinsics, let ext0 = p0.extrinsics else {
            throw NSError(domain: "Streaming", code: 1, userInfo: [NSLocalizedDescriptionKey: "chunk 0 missing camera info"])
        }
        let pm0 = PointMaps.depthToWorldPoints(depth: p0.depth, intrinsics: int0, extrinsicsW2C: ext0)
        eval(pm0)
        let conf0Mean = meanFloat(p0.conf)
        let confThresh0 = conf0Mean * config.pcdConfThresholdCoef
        try PLYWriter.saveConfidentPointCloud(
            points: pm0, colors: p0.processedImages, confs: p0.conf,
            confThreshold: confThresh0,
            outputPath: (pcdDir as NSString).appendingPathComponent("0_pcd.ply")
        )

        // Subsequent chunks.
        for i in 1..<perChunk.count {
            let p = perChunk[i]
            guard let int_i = p.intrinsics, let ext_i = p.extrinsics else { continue }
            let xform = cumulative[i - 1]
            var pm = PointMaps.depthToWorldPoints(depth: p.depth, intrinsics: int_i, extrinsicsW2C: ext_i)
            pm = PointMaps.applySim3(pm, transform: xform)
            eval(pm)
            let confMean = meanFloat(p.conf)
            let confThresh = confMean * config.pcdConfThresholdCoef
            try PLYWriter.saveConfidentPointCloud(
                points: pm, colors: p.processedImages, confs: p.conf,
                confThreshold: confThresh,
                outputPath: (pcdDir as NSString).appendingPathComponent("\(i)_pcd.ply")
            )
        }
    }

    private func dumpChunk(_ pred: ChunkPredictions, chunkIdx: Int, dir: String) throws {
        let p = { (name: String) -> String in
            (dir as NSString).appendingPathComponent("chunk\(chunkIdx)_\(name).npy")
        }
        try NpyWriter.writeFloat32(pred.depth, to: p("depth"))
        try NpyWriter.writeFloat32(pred.conf,  to: p("conf"))
        if let k = pred.intrinsics { try NpyWriter.writeFloat32(k, to: p("intrinsics")) }
        if let e = pred.extrinsics { try NpyWriter.writeFloat32(e, to: p("extrinsics")) }
        if let r = pred.ray { try NpyWriter.writeFloat32(r, to: p("ray")) }
        if let rc = pred.rayConf { try NpyWriter.writeFloat32(rc, to: p("rayconf")) }
    }

    private func dumpSim3List(_ pairs: [Sim3Alignment.Sim3], dir: String) throws {
        guard !pairs.isEmpty else { return }
        let n = pairs.count
        let s = MLXArray(pairs.map { $0.s }, [n])                               // (n,)
        let R = MLXArray(pairs.flatMap { $0.R }, [n, 3, 3])                     // (n,3,3)
        let t = MLXArray(pairs.flatMap { $0.t }, [n, 3])                        // (n,3)
        try NpyWriter.writeFloat32(s, to: (dir as NSString).appendingPathComponent("sim3_s.npy"))
        try NpyWriter.writeFloat32(R, to: (dir as NSString).appendingPathComponent("sim3_R.npy"))
        try NpyWriter.writeFloat32(t, to: (dir as NSString).appendingPathComponent("sim3_t.npy"))
    }

    private func meanFloat(_ a: MLXArray) -> Float {
        let m = a.mean()
        eval(m)
        return m.item(Float.self)
    }
}
