import Foundation
import MLX

/// Mirrors the python `save_camera_poses` (txt + per-frame intrinsic.txt).
///
/// Inputs:
/// - `chunkExtrinsicsW2C[i]` is the per-view w2c (S, 3, 4) for chunk i
/// - `chunkIntrinsics[i]` is per-view K (S, 3, 3)
/// - `chunkRanges[i]` is the (start, end) global frame index for chunk i
/// - `cumulativeSim3[i]` (i in 0..<chunks-1) maps chunk i+1 → chunk 0 (output of accumulate)
/// - `overlap` is the chunk overlap; `overlapStart` is python's `overlap_s` (we use 0)
public enum CameraPosesIO {

    public struct Inputs {
        public var chunkExtrinsicsW2C: [MLXArray]   // each (S, 3, 4)
        public var chunkIntrinsics: [MLXArray]      // each (S, 3, 3)
        public var chunkRanges: [(Int, Int)]
        public var cumulativeSim3: [Sim3Alignment.Sim3]  // length = chunks - 1
        public var overlap: Int
        public var overlapStart: Int = 0
        public var totalFrames: Int

        public init(
            chunkExtrinsicsW2C: [MLXArray],
            chunkIntrinsics: [MLXArray],
            chunkRanges: [(Int, Int)],
            cumulativeSim3: [Sim3Alignment.Sim3],
            overlap: Int,
            totalFrames: Int
        ) {
            self.chunkExtrinsicsW2C = chunkExtrinsicsW2C
            self.chunkIntrinsics = chunkIntrinsics
            self.chunkRanges = chunkRanges
            self.cumulativeSim3 = cumulativeSim3
            self.overlap = overlap
            self.totalFrames = totalFrames
        }
    }

    /// In-memory result returned by ``computePoses``.
    public struct Result {
        /// Per-frame camera-to-world poses, shape `[N, 4, 4]` row-major.
        public let cameraPosesC2W: MLXArray
        /// Per-frame intrinsic matrices, shape `[N, 3, 3]`.
        public let intrinsicsK: MLXArray

        public init(cameraPosesC2W: MLXArray, intrinsicsK: MLXArray) {
            self.cameraPosesC2W = cameraPosesC2W
            self.intrinsicsK = intrinsicsK
        }
    }

    /// Compute per-frame c2w poses + intrinsics from per-chunk extrinsics/intrinsics
    /// and the cumulative sim3 chain. Pure math, no file IO.
    public static func computePoses(_ inputs: Inputs) -> Result {
        let (allPoses, allIntr) = computeRaw(inputs)
        // Pack into flat float arrays for MLXArray construction. Frames with
        // no pose are filled with zeros.
        var posesFlat = [Float](repeating: 0, count: inputs.totalFrames * 16)
        var intrFlat = [Float](repeating: 0, count: inputs.totalFrames * 9)
        for i in 0..<inputs.totalFrames {
            if let p = allPoses[i] {
                posesFlat.replaceSubrange((i * 16)..<((i + 1) * 16), with: p)
            }
            if let k = allIntr[i] {
                intrFlat.replaceSubrange((i * 9)..<((i + 1) * 9), with: k)
            }
        }
        return Result(
            cameraPosesC2W: MLXArray(posesFlat, [inputs.totalFrames, 4, 4]),
            intrinsicsK: MLXArray(intrFlat, [inputs.totalFrames, 3, 3])
        )
    }

    public static func savePoses(_ inputs: Inputs, outputDir: String) throws {
        let (allPoses, allIntr) = computeRaw(inputs)
        try writePoseFiles(allPoses: allPoses, allIntr: allIntr, outputDir: outputDir)
    }

    private static func computeRaw(_ inputs: Inputs) -> ([[Float]?], [[Float]?]) {
        let overlapEnd = inputs.overlap - inputs.overlapStart
        var allPoses = [[Float]?](repeating: nil, count: inputs.totalFrames)
        var allIntr = [[Float]?](repeating: nil, count: inputs.totalFrames)

        // First chunk: no transform applied. Take frames [start, end - overlap_e).
        let firstRange = inputs.chunkRanges[0]
        let firstExtCPU: [Float] = inputs.chunkExtrinsicsW2C[0].asArray(Float.self)
        let firstIntCPU: [Float] = inputs.chunkIntrinsics[0].asArray(Float.self)
        let firstS = inputs.chunkExtrinsicsW2C[0].dim(0)
        let firstFrameEnd = firstRange.1 - overlapEnd
        for (i, idx) in (firstRange.0..<firstFrameEnd).enumerated() {
            // Build c2w from w2c[i]
            let w2c = sliceMatrix3x4(firstExtCPU, viewIndex: i, viewCount: firstS)
            let c2w = invertW2C(w2c)
            let c2w4x4 = expandTo4x4(c2w)
            allPoses[idx] = c2w4x4
            allIntr[idx] = sliceMatrix3x3(firstIntCPU, viewIndex: i, viewCount: firstS)
        }

        // Subsequent chunks: apply S = [[s*R | t]] to each c2w then normalize rotation by /s.
        for chunkIdx in 1..<inputs.chunkRanges.count {
            let range = inputs.chunkRanges[chunkIdx]
            let xform = inputs.cumulativeSim3[chunkIdx - 1]
            let extCPU: [Float] = inputs.chunkExtrinsicsW2C[chunkIdx].asArray(Float.self)
            let intCPU: [Float] = inputs.chunkIntrinsics[chunkIdx].asArray(Float.self)
            let S = inputs.chunkExtrinsicsW2C[chunkIdx].dim(0)

            let chunkRangeEnd = chunkIdx < inputs.chunkRanges.count - 1 ? range.1 - overlapEnd : range.1

            // Build S = [[s*R | t]; [0, 0, 0, 1]] (4×4)
            let SR = scaleAndRotate(xform.s, xform.R)
            let Smat = mat4FromBlock(SR, t: xform.t)

            for (i, idx) in stride(from: range.0 + inputs.overlapStart, to: chunkRangeEnd, by: 1).enumerated() {
                let w2c = sliceMatrix3x4(extCPU, viewIndex: i + inputs.overlapStart, viewCount: S)
                let c2w = expandTo4x4(invertW2C(w2c))
                var transformed = matmul4x4(Smat, c2w)                    // S @ c2w (left-multiply)
                // Normalize the rotation: divide top-left 3x3 by s.
                for r in 0..<3 {
                    for c in 0..<3 {
                        transformed[r * 4 + c] /= xform.s
                    }
                }
                allPoses[idx] = transformed
                allIntr[idx] = sliceMatrix3x3(intCPU, viewIndex: i + inputs.overlapStart, viewCount: S)
            }
        }

        return (allPoses, allIntr)
    }

    private static func writePoseFiles(
        allPoses: [[Float]?], allIntr: [[Float]?], outputDir: String
    ) throws {
        // Write camera_poses.txt: 16 numbers per line (row-major c2w 4x4)
        let posesPath = (outputDir as NSString).appendingPathComponent("camera_poses.txt")
        var posesText = ""
        for (i, p) in allPoses.enumerated() {
            guard let pose = p else {
                print("[poses] WARNING: frame \(i) has no pose; writing identity")
                let id = [Float](repeating: 0, count: 16)
                posesText += id.map { String($0) }.joined(separator: " ") + "\n"
                continue
            }
            posesText += pose.map { String($0) }.joined(separator: " ") + "\n"
        }
        try posesText.write(toFile: posesPath, atomically: true, encoding: .utf8)
        print("[poses] wrote \(allPoses.count) poses → \(posesPath)")

        // Write intrinsic.txt: fx fy cx cy per line
        let intrPath = (outputDir as NSString).appendingPathComponent("intrinsic.txt")
        var intrText = ""
        for k in allIntr {
            guard let K = k else {
                intrText += "0 0 0 0\n"
                continue
            }
            // K is row-major 3×3
            let fx = K[0], fy = K[4], cx = K[2], cy = K[5]
            intrText += "\(fx) \(fy) \(cx) \(cy)\n"
        }
        try intrText.write(toFile: intrPath, atomically: true, encoding: .utf8)
        print("[poses] wrote \(allIntr.count) intrinsics → \(intrPath)")
    }

    // MARK: - small linalg helpers (3×3 / 3×4 / 4×4 row-major)

    private static func sliceMatrix3x4(_ buf: [Float], viewIndex: Int, viewCount: Int) -> [Float] {
        let stride = 12
        return Array(buf[(viewIndex * stride)..<((viewIndex + 1) * stride)])
    }

    private static func sliceMatrix3x3(_ buf: [Float], viewIndex: Int, viewCount: Int) -> [Float] {
        let stride = 9
        return Array(buf[(viewIndex * stride)..<((viewIndex + 1) * stride)])
    }

    /// w2c (3×4 row-major) → c2w (3×4 row-major): [R^T | -R^T t]
    private static func invertW2C(_ w2c: [Float]) -> [Float] {
        let R: [Float] = [w2c[0], w2c[1], w2c[2], w2c[4], w2c[5], w2c[6], w2c[8], w2c[9], w2c[10]]
        let t: [Float] = [w2c[3], w2c[7], w2c[11]]
        // R^T (3×3 row-major)
        let RT: [Float] = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]
        let tInv: [Float] = [
            -(RT[0]*t[0] + RT[1]*t[1] + RT[2]*t[2]),
            -(RT[3]*t[0] + RT[4]*t[1] + RT[5]*t[2]),
            -(RT[6]*t[0] + RT[7]*t[1] + RT[8]*t[2])
        ]
        return [
            RT[0], RT[1], RT[2], tInv[0],
            RT[3], RT[4], RT[5], tInv[1],
            RT[6], RT[7], RT[8], tInv[2]
        ]
    }

    /// 3×4 → 4×4 (append last row [0, 0, 0, 1])
    private static func expandTo4x4(_ m34: [Float]) -> [Float] {
        var m = [Float](repeating: 0, count: 16)
        for i in 0..<3 { for j in 0..<4 { m[i * 4 + j] = m34[i * 4 + j] } }
        m[15] = 1
        return m
    }

    /// Scale a 3×3 R by s.
    private static func scaleAndRotate(_ s: Float, _ R: [Float]) -> [Float] {
        return R.map { $0 * s }
    }

    /// Build 4×4 [[A | t]; [0, 0, 0, 1]] from 3×3 A (row-major) and t (3,).
    private static func mat4FromBlock(_ A: [Float], t: [Float]) -> [Float] {
        return [
            A[0], A[1], A[2], t[0],
            A[3], A[4], A[5], t[1],
            A[6], A[7], A[8], t[2],
            0, 0, 0, 1
        ]
    }

    /// 4×4 row-major matmul.
    private static func matmul4x4(_ A: [Float], _ B: [Float]) -> [Float] {
        var C = [Float](repeating: 0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                var s: Float = 0
                for k in 0..<4 { s += A[i*4+k] * B[k*4+j] }
                C[i*4+j] = s
            }
        }
        return C
    }
}
