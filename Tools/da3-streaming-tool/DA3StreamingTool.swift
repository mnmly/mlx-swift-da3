import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXDA3
import MLXDA3Streaming

@main
struct DA3StreamingTool: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "da3-streaming-tool",
        abstract: "DA3 streaming pipeline (Phase 1 — chunked inference, no loop closure).",
        discussion: """
        Reads images from a directory, runs DA3 inference in chunks with overlap,
        aligns chunks via Sim(3), writes camera_poses.txt + intrinsic.txt + a
        combined point cloud (pcd/combined_pcd.ply).
        """
    )

    @Option(name: .shortAndLong, help: "Model config name")
    var model: String = "da3-giant"

    @Option(name: .shortAndLong, help: "Path to model.safetensors")
    var weights: String

    @Option(name: .long, help: "Directory of input images (jpg/png)")
    var imageDir: String

    @Option(name: .long, help: "Output directory")
    var outputDir: String

    @Option(name: .long, help: "Number of frames per chunk")
    var chunkSize: Int = 8

    @Option(name: .long, help: "Number of overlap frames between chunks")
    var overlap: Int = 4

    @Option(name: .long, help: "Processing resolution (matches python `process_res`; default 504)")
    var resolution: Int = 504

    @Option(name: .long, help: "Weight dtype: float16 or float32")
    var dtype: String = "float16"

    @Option(name: .long, help: "Limit number of images (0 = all)")
    var limit: Int = 0

    @Option(name: .long, help: "Point-cloud conf threshold coefficient (mean(conf) * coef)")
    var pcdConfCoef: Float = 0.75

    @Option(name: .long, help: "If set, dump per-chunk depth/conf/intrinsics/extrinsics + sim3 list as .npy here")
    var dumpDir: String?

    @Option(name: .long, help: "Diagnostic: instead of preprocessing images, load this .npy as model input ([1,S,3,H,W] NCHW float32, swift will permute to NHWC). Forces single-chunk mode.")
    var inputNpy: String?

    func run() throws {
        let targetDtype: DType = dtype == "float32" ? .float32 : .float16

        guard overlap < chunkSize else {
            print("Error: overlap (\(overlap)) must be < chunk-size (\(chunkSize))")
            throw ExitCode.failure
        }

        let fm = FileManager.default
        try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        var paths = try ImageDirectory.listImagePaths(in: imageDir)
        if limit > 0, paths.count > limit { paths = Array(paths.prefix(limit)) }
        print("Found \(paths.count) images")

        print("Building model '\(model)'...")
        let weightsURL = URL(fileURLWithPath: weights)
        let da3Model = try loadModel(configName: model, weightsURL: weightsURL, dtype: targetDtype)
        print("Model loaded.")

        var cfg = StreamingOrchestrator.Config()
        cfg.chunkSize = chunkSize
        cfg.overlap = overlap
        cfg.resolution = resolution
        cfg.dtype = targetDtype
        cfg.pcdConfThresholdCoef = pcdConfCoef
        cfg.dumpDir = dumpDir

        if let npyPath = inputNpy {
            // Diagnostic: bypass preprocessing, feed npy through model + RayPose, write rayconf.
            try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
            let raw = try NpyWriter.readFloat32(from: npyPath)  // [1, S, 3, H, W] NCHW float32
            // Permute to NHWC: [1, S, H, W, 3]
            let inputNHWC = raw.transposed(axes: [0, 1, 3, 4, 2]).asType(targetDtype)
            let h = inputNHWC.dim(2)
            let w = inputNHWC.dim(3)
            print("Loaded npy input: \(inputNHWC.shape) → running model directly")

            let outDirEarly = (outputDir as NSString).standardizingPath
            DinoVisionTransformer._dumpCamToken = { x in
                try? NpyWriter.writeFloat32(x.asType(.float32), to: (outDirEarly as NSString).appendingPathComponent("cam_token_applied.npy"))
            }
            DinoVisionTransformer._dumpTokensPostCam = { x in
                try? NpyWriter.writeFloat32(x.asType(.float32), to: (outDirEarly as NSString).appendingPathComponent("tokens_post_cam.npy"))
            }
            DinoVisionTransformer._dumpTokensAfterEmbed = { x in
                try? NpyWriter.writeFloat32(x.asType(.float32), to: (outDirEarly as NSString).appendingPathComponent("tokens_after_embed.npy"))
            }
            // Dump after blocks 0, 1, 2, 3, 5, 7, 10 to track drift growth.
            let blocksToDump: Set<Int> = [0, 1, 2, 3, 5, 7, 10]
            DinoVisionTransformer._dumpAfterBlock = { idx, x in
                if blocksToDump.contains(idx) {
                    try? NpyWriter.writeFloat32(x.asType(.float32), to: (outDirEarly as NSString).appendingPathComponent("tokens_after_block_\(idx).npy"))
                }
            }
            Embeddings._dumpPosEmbed = { x in
                try? NpyWriter.writeFloat32(x.asType(.float32), to: (outDirEarly as NSString).appendingPathComponent("pos_embed_interp.npy"))
            }

            let outs = da3Model(inputNHWC, outputs: .all)
            eval(outs)
            guard let rayRaw = outs["ray"], let rayConfRaw = outs["ray_conf"] else {
                fatalError("Missing ray/ray_conf outputs")
            }
            // Match StreamingInference squeezing
            var ray = rayRaw.asType(.float32)
            if ray.dim(0) == 1 { ray = ray.squeezed(axis: 0) }
            var rc = rayConfRaw.asType(.float32)
            if rc.dim(0) == 1 { rc = rc.squeezed(axis: 0) }
            if rc.ndim >= 1, rc.dim(rc.ndim - 1) == 1 { rc = rc.squeezed(axis: -1) }
            eval(ray, rc)

            let outDir = (outputDir as NSString).standardizingPath
            try NpyWriter.writeFloat32(ray, to: (outDir as NSString).appendingPathComponent("chunk0_ray.npy"))
            try NpyWriter.writeFloat32(rc, to: (outDir as NSString).appendingPathComponent("chunk0_rayconf.npy"))
            // Also run RayPose for intrinsics/extrinsics
            let (extW2C, intr) = RayPose.cameraInfoFromRays(ray: ray, rayConf: rc, imageHeight: h, imageWidth: w)
            try NpyWriter.writeFloat32(intr, to: (outDir as NSString).appendingPathComponent("chunk0_intrinsics.npy"))
            try NpyWriter.writeFloat32(extW2C, to: (outDir as NSString).appendingPathComponent("chunk0_extrinsics.npy"))
            print("Saved ray, rayconf, intrinsics, extrinsics to \(outDir)")
            return
        }

        let orchestrator = StreamingOrchestrator(model: da3Model, config: cfg)
        let t0 = CFAbsoluteTimeGetCurrent()
        try orchestrator.run(imagePaths: paths, outputDir: outputDir)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "Done in %.2fs", dt))
    }
}
