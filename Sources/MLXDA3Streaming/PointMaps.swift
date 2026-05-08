import Foundation
import MLX

/// Per-frame depth → world-space point cloud, plus Sim(3) application.
///
/// Mirrors `depth_to_point_cloud_optimized_torch` and `apply_sim3_direct_torch`.
public enum PointMaps {

    /// Convert depth + intrinsics + extrinsics(w2c) to per-pixel world coordinates.
    ///
    /// - depth: `[N, H, W]` float32
    /// - intrinsics: `[N, 3, 3]` float32 (`[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`)
    /// - extrinsicsW2C: `[N, 3, 4]` float32 (world→camera)
    /// - Returns: `[N, H, W, 3]` float32 world coordinates.
    public static func depthToWorldPoints(
        depth: MLXArray, intrinsics: MLXArray, extrinsicsW2C: MLXArray
    ) -> MLXArray {
        let N = depth.dim(0)
        let H = depth.dim(1)
        let W = depth.dim(2)

        // Pixel grid (H, W, 3): (u, v, 1)
        let u = MLXArray.linspace(Float(0), Float(W - 1), count: W).asType(.float32)
        let v = MLXArray.linspace(Float(0), Float(H - 1), count: H).asType(.float32)
        let uGrid = MLX.broadcast(u.reshaped([1, W]), to: [H, W])
        let vGrid = MLX.broadcast(v.reshaped([H, 1]), to: [H, W])
        let ones = MLXArray.ones([H, W], dtype: .float32)
        let pixelHW3 = MLX.stacked([uGrid, vGrid, ones], axis: -1)            // (H, W, 3)

        // Closed-form K^-1 per view: [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]
        let fx = intrinsics[0..., 0, 0]                                       // (N,)
        let fy = intrinsics[0..., 1, 1]
        let cx = intrinsics[0..., 0, 2]
        let cy = intrinsics[0..., 1, 2]
        // camera_coords = K^-1 @ pixel  →  ((u - cx) / fx, (v - cy) / fy, 1)
        // Per-view broadcasting: (N, H, W, 3)
        let pixelExpanded = pixelHW3.expandedDimensions(axis: 0)              // (1, H, W, 3)
        let pixelN = MLX.broadcast(pixelExpanded, to: [N, H, W, 3])
        let pu = pixelN[0..., 0..., 0..., 0]                                  // (N, H, W)
        let pv = pixelN[0..., 0..., 0..., 1]
        let pone = pixelN[0..., 0..., 0..., 2]
        // Reshape fx etc. to (N, 1, 1) for broadcasting against (N, H, W).
        let fxR = fx.reshaped([N, 1, 1])
        let fyR = fy.reshaped([N, 1, 1])
        let cxR = cx.reshaped([N, 1, 1])
        let cyR = cy.reshaped([N, 1, 1])
        let camX = (pu - cxR) / fxR
        let camY = (pv - cyR) / fyR
        let camZ = pone
        let camCoordsUnit = MLX.stacked([camX, camY, camZ], axis: -1)          // (N, H, W, 3)
        let camCoords = camCoordsUnit * depth.expandedDimensions(axis: -1)     // (N, H, W, 3)

        // Convert w2c (3, 4) -> c2w (3, 4) without using inv (which is CPU-only).
        // w2c = [R | t]; c2w = [R^T | -R^T t]
        let R = extrinsicsW2C[0..., 0..<3, 0..<3]                              // (N, 3, 3)
        let t = extrinsicsW2C[0..., 0..<3, 3..<4]                              // (N, 3, 1)
        let RT = R.transposed(axes: [0, 2, 1])                                 // (N, 3, 3)
        let tInv = -RT.matmul(t)                                               // (N, 3, 1)

        // world = R^T @ cam + t_inv. Apply per-view via einsum-like batched matmul.
        // cam: (N, H*W, 3). transpose to (N, 3, H*W). Multiply by RT: (N, 3, 3) @ (N, 3, H*W).
        let camFlat = camCoords.reshaped([N, H * W, 3]).transposed(axes: [0, 2, 1])  // (N, 3, H*W)
        let worldFlat = RT.matmul(camFlat) + tInv                              // (N, 3, H*W)
        let world = worldFlat.transposed(axes: [0, 2, 1]).reshaped([N, H, W, 3])
        return world
    }

    /// Apply Sim(3) per-pixel: `points * s @ R^T + t`. Returns `[N, H, W, 3]`.
    public static func applySim3(
        _ points: MLXArray, transform: Sim3Alignment.Sim3
    ) -> MLXArray {
        let R = MLXArray(transform.R, [3, 3])
        let t = MLXArray(transform.t, [3])
        // points (N, H, W, 3) → flat (N, H*W, 3)
        let N = points.dim(0)
        let H = points.dim(1)
        let W = points.dim(2)
        let flat = points.reshaped([N, H * W, 3])
        let rotated = flat.matmul(R.transposed())                              // (N, H*W, 3)
        let scaled = Float(transform.s) * rotated + t
        return scaled.reshaped([N, H, W, 3])
    }

    /// Accumulate adjacent transforms into prefix transforms (chunk-i → chunk-0).
    /// Mirrors `accumulate_sim3_transforms`.
    public static func accumulate(_ transforms: [Sim3Alignment.Sim3]) -> [Sim3Alignment.Sim3] {
        guard !transforms.isEmpty else { return [] }
        var out: [Sim3Alignment.Sim3] = [transforms[0]]
        for i in 1..<transforms.count {
            let prev = out[i - 1]
            let next = transforms[i]
            // R_cum = R_prev @ R_next ; s_cum = s_prev * s_next ;
            // t_cum = s_prev * (R_prev @ t_next) + t_prev
            let Rcum = matmul3x3(prev.R, next.R)
            let scum = prev.s * next.s
            let RprevTnext = matVec3x3(prev.R, next.t)
            let tcum = [
                prev.s * RprevTnext[0] + prev.t[0],
                prev.s * RprevTnext[1] + prev.t[1],
                prev.s * RprevTnext[2] + prev.t[2]
            ]
            out.append(.init(s: scum, R: Rcum, t: tcum))
        }
        return out
    }

    // MARK: - small helpers

    private static func matmul3x3(_ A: [Float], _ B: [Float]) -> [Float] {
        var C = [Float](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var s: Float = 0
                for k in 0..<3 { s += A[i*3+k] * B[k*3+j] }
                C[i*3+j] = s
            }
        }
        return C
    }

    private static func matVec3x3(_ A: [Float], _ v: [Float]) -> [Float] {
        return [
            A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
            A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
            A[6] * v[0] + A[7] * v[1] + A[8] * v[2]
        ]
    }
}
