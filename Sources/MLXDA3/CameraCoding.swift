import MLX
import MLXNN

// Port of `depth_anything_3/model/{cam_enc,cam_dec}.py` and the pose-encoding helpers in
// `model/utils/transform.py`. All camera math runs in float32, as upstream disables
// autocast around both modules.

/// DA3's 9-channel camera pose encoding: `[T_c2w (3), quat_xyzw (4), fov_h, fov_w]`.
public enum PoseEncoding {
    /// Encode camera-to-world poses and pixel intrinsics.
    ///
    /// - Parameters:
    ///   - c2w: Camera-to-world poses `[B, S, 4, 4]` (or `[B, S, 3, 4]`).
    ///   - intrinsics: Pixel intrinsics `[B, S, 3, 3]`.
    ///   - height: Image height in pixels.
    ///   - width: Image width in pixels.
    /// - Returns: `[B, S, 9]` float32 encoding.
    public static func encode(c2w: MLXArray, intrinsics: MLXArray, height: Int, width: Int) -> MLXArray {
        let c2w = c2w.asType(.float32)
        let intrinsics = intrinsics.asType(.float32)
        let R = c2w[.ellipsis, ..<3, ..<3]
        let T = c2w[.ellipsis, ..<3, 3]
        let quat = matrixToQuaternion(R)
        let fovH = 2 * atan((Float(height) / 2) / intrinsics[.ellipsis, 1, 1])
        let fovW = 2 * atan((Float(width) / 2) / intrinsics[.ellipsis, 0, 0])
        return concatenated(
            [T, quat, fovH.expandedDimensions(axis: -1), fovW.expandedDimensions(axis: -1)],
            axis: -1
        )
    }

    /// Decode an encoding back to camera-to-world `[R|T]` and intrinsics.
    ///
    /// - Returns: `(c2w [B, S, 3, 4], intrinsics [B, S, 3, 3])`; the principal point is the
    ///   image centre, as upstream.
    public static func decode(_ encoding: MLXArray, height: Int, width: Int) -> (c2w: MLXArray, intrinsics: MLXArray) {
        let encoding = encoding.asType(.float32)
        let T = encoding[.ellipsis, ..<3]
        let quat = encoding[.ellipsis, 3 ..< 7]
        let fovH = encoding[.ellipsis, 7]
        let fovW = encoding[.ellipsis, 8]

        let R = quaternionToMatrix(quat)
        let c2w = concatenated([R, T.expandedDimensions(axis: -1)], axis: -1)

        let fy = (Float(height) / 2) / maximum(tan(fovH / 2), MLXArray(Float(1e-6)))
        let fx = (Float(width) / 2) / maximum(tan(fovW / 2), MLXArray(Float(1e-6)))
        let zero = MLXArray.zeros(like: fx)
        let one = MLXArray.ones(like: fx)
        let cx = full(fx.shape, values: MLXArray(Float(width) / 2))
        let cy = full(fx.shape, values: MLXArray(Float(height) / 2))
        let rows = stacked([fx, zero, cx, zero, fy, cy, zero, zero, one], axis: -1)
        return (c2w, rows.reshaped(fx.shape + [3, 3]))
    }

    /// Rotation matrices `[..., 3, 3]` to standardized scalar-last quaternions `[..., 4]`
    /// (pytorch3d `matrix_to_quaternion`, then `w >= 0`).
    public static func matrixToQuaternion(_ m: MLXArray) -> MLXArray {
        let m00 = m[.ellipsis, 0, 0], m01 = m[.ellipsis, 0, 1], m02 = m[.ellipsis, 0, 2]
        let m10 = m[.ellipsis, 1, 0], m11 = m[.ellipsis, 1, 1], m12 = m[.ellipsis, 1, 2]
        let m20 = m[.ellipsis, 2, 0], m21 = m[.ellipsis, 2, 1], m22 = m[.ellipsis, 2, 2]

        let squares = stacked([
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ], axis: -1)
        let qAbs = MLX.where(squares .> 0, sqrt(maximum(squares, 0)), MLXArray.zeros(like: squares))

        // Candidate quaternions (real part first), one row per pivot.
        let candidates = stacked([
            stacked([square(qAbs[.ellipsis, 0]), m21 - m12, m02 - m20, m10 - m01], axis: -1),
            stacked([m21 - m12, square(qAbs[.ellipsis, 1]), m10 + m01, m02 + m20], axis: -1),
            stacked([m02 - m20, m10 + m01, square(qAbs[.ellipsis, 2]), m12 + m21], axis: -1),
            stacked([m10 - m01, m20 + m02, m21 + m12, square(qAbs[.ellipsis, 3])], axis: -1),
        ], axis: -2)
        let scaled = candidates / (2 * maximum(qAbs, MLXArray(Float(0.1)))).expandedDimensions(axis: -1)

        // Pick the best-conditioned candidate (largest |q| component).
        let pivot = argMax(qAbs, axis: -1)
        let index = broadcast(pivot.reshaped(pivot.shape + [1, 1]), to: pivot.shape + [1, 4])
        let wxyz = takeAlong(scaled, index, axis: -2).squeezed(axis: -2)

        let xyzw = concatenated([wxyz[.ellipsis, 1 ..< 4], wxyz[.ellipsis, ..<1]], axis: -1)
        return MLX.where(xyzw[.ellipsis, 3 ..< 4] .< 0, -xyzw, xyzw)
    }

    /// Scalar-last quaternions `[..., 4]` to rotation matrices `[..., 3, 3]`.
    public static func quaternionToMatrix(_ q: MLXArray) -> MLXArray {
        let i = q[.ellipsis, 0], j = q[.ellipsis, 1], k = q[.ellipsis, 2], r = q[.ellipsis, 3]
        let twoS = 2 / (q * q).sum(axis: -1)
        let o = stacked([
            1 - twoS * (j * j + k * k),
            twoS * (i * j - k * r),
            twoS * (i * k + j * r),
            twoS * (i * j + k * r),
            1 - twoS * (i * i + k * k),
            twoS * (j * k - i * r),
            twoS * (i * k - j * r),
            twoS * (j * k + i * r),
            1 - twoS * (i * i + j * j),
        ], axis: -1)
        return o.reshaped(Array(q.shape.dropLast()) + [3, 3])
    }
}

/// Inverse of a batch of rigid `[..., 4, 4]` (or `[..., 3, 4]`) transforms
/// (python `affine_inverse`).
public func affineInverse(_ a: MLXArray) -> MLXArray {
    let R = a[.ellipsis, ..<3, ..<3]
    let T = a[.ellipsis, ..<3, 3...]
    let Rt = R.swappedAxes(-1, -2)
    let top = concatenated([Rt, -matmul(Rt, T)], axis: -1)
    if a.dim(-2) == 3 { return top }
    return concatenated([top, a[.ellipsis, 3..., 0...]], axis: -2)
}

/// DA3 camera encoder: turns known cameras into per-view conditioning tokens that replace
/// the backbone's camera token at `altStart` (python `CameraEnc`).
public class CameraEncoder: Module {
    @ModuleInfo(key: "pose_branch") private var poseBranch: Mlp
    @ModuleInfo(key: "token_norm") private var tokenNorm: LayerNorm
    @ModuleInfo(key: "trunk") private var trunk: [DA3Block]
    @ModuleInfo(key: "trunk_norm") private var trunkNorm: LayerNorm

    /// - Parameters:
    ///   - dimOut: Token width; the backbone embedding dimension.
    ///   - trunkDepth: Number of transformer blocks over the view axis.
    ///   - numHeads: Attention heads per trunk block.
    public init(dimOut: Int = 1024, trunkDepth: Int = 4, numHeads: Int = 16) {
        self._poseBranch.wrappedValue = Mlp(inFeatures: 9, hiddenFeatures: dimOut / 2, outFeatures: dimOut)
        // Upstream uses the default `nn.LayerNorm` (eps 1e-5) here, unlike the backbone.
        self._tokenNorm.wrappedValue = LayerNorm(dimensions: dimOut, eps: 1e-5)
        self._trunk.wrappedValue = (0 ..< trunkDepth).map { _ in
            DA3Block(dim: dimOut, numHeads: numHeads, initValues: 0.01, lnEps: 1e-5)
        }
        self._trunkNorm.wrappedValue = LayerNorm(dimensions: dimOut, eps: 1e-5)
    }

    /// - Parameters:
    ///   - w2c: World-to-camera extrinsics `[B, S, 4, 4]`.
    ///   - intrinsics: Pixel intrinsics `[B, S, 3, 3]`.
    ///   - height: Image height in pixels.
    ///   - width: Image width in pixels.
    /// - Returns: Camera tokens `[B, S, dimOut]`.
    public func callAsFunction(w2c: MLXArray, intrinsics: MLXArray, height: Int, width: Int) -> MLXArray {
        let encoding = PoseEncoding.encode(
            c2w: affineInverse(w2c.asType(.float32)), intrinsics: intrinsics, height: height, width: width
        )
        var tokens = tokenNorm(poseBranch(encoding))
        for block in trunk {
            tokens = block(tokens)
        }
        return trunkNorm(tokens)
    }
}

/// DA3 camera decoder: predicts a pose encoding from the last out-layer camera token
/// (python `CameraDec`).
public class CameraDecoder: Module {
    @ModuleInfo(key: "backbone") private var backbone: [Linear]
    @ModuleInfo(key: "fc_t") private var fcT: Linear
    @ModuleInfo(key: "fc_qvec") private var fcQvec: Linear
    @ModuleInfo(key: "fc_fov") private var fcFov: Linear

    /// - Parameter dimIn: Camera-token width (`2 * embedDim` with `catToken`).
    public init(dimIn: Int = 1536) {
        self._backbone.wrappedValue = [Linear(dimIn, dimIn), Linear(dimIn, dimIn)]
        self._fcT.wrappedValue = Linear(dimIn, 3)
        self._fcQvec.wrappedValue = Linear(dimIn, 4)
        self._fcFov.wrappedValue = Linear(dimIn, 2)
    }

    /// - Parameter cameraToken: `[B, S, dimIn]` raw camera token from the last out layer.
    /// - Returns: Pose encoding `[B, S, 9]` (float32).
    public func callAsFunction(_ cameraToken: MLXArray) -> MLXArray {
        var h = cameraToken
        for layer in backbone {
            h = relu(layer(h))
        }
        h = h.asType(.float32)
        return concatenated([fcT(h), fcQvec(h), relu(fcFov(h))], axis: -1)
    }
}
