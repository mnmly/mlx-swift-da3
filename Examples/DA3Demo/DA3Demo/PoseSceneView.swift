import AppKit
import MLX
import RealityKit
import SwiftUI

/// RealityKit-based viewer for DA3 streaming output: each camera pose is
/// drawn as a small frustum (apex at the camera centre, opening toward +Z),
/// connected by trajectory line segments. Mouse drag = orbit, scroll = zoom.
///
/// Inputs:
/// - `cameraPosesC2W`: `[N, 4, 4]` MLXArray, row-major c2w matrices.
/// - `intrinsicsK`: optional `[N, 3, 3]` for sizing the frustum aspect ratio.
struct PoseSceneView: View {
    let cameraPosesC2W: MLXArray
    let intrinsicsK: MLXArray?

    var body: some View {
        RealityView { content in
            populate(content: content)
        } update: { content in
            // Strip and re-populate on prediction change.
            for entity in content.entities {
                content.remove(entity)
            }
            populate(content: content)
        }
        .realityViewCameraControls(.orbit)
    }

    // MARK: - Scene construction

    @MainActor
    private func populate(content: any RealityViewContentProtocol) {
        let n = cameraPosesC2W.dim(0)
        guard n > 0 else { return }
        let posesCPU = cameraPosesC2W.asType(.float32).asArray(Float.self)

        // Aspect ratio for frustum drawing — defaults to 16:10 if no intrinsics.
        var aspect: Float = 1.6
        if let K = intrinsicsK {
            let kCPU = K.asType(.float32).asArray(Float.self)
            if kCPU.count >= 9, kCPU[0] > 0 {
                aspect = max(0.5, min(3.0, kCPU[4] / kCPU[0]))
            }
        }

        // Compute scene scale so frustums are visible regardless of trajectory size.
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(n)
        var maxAbsT: Float = 0
        for k in 0..<n {
            let base = k * 16
            let t = SIMD3<Float>(posesCPU[base + 3], posesCPU[base + 7], posesCPU[base + 11])
            positions.append(t)
            maxAbsT = max(maxAbsT, max(abs(t.x), max(abs(t.y), abs(t.z))))
        }
        let frustumSize: Float = max(0.05, maxAbsT * 0.05)

        // Root for the whole pose viz so we can centre it.
        let centroid = positions.reduce(SIMD3<Float>(0, 0, 0), +) / Float(positions.count)
        let root = Entity()
        root.position = -centroid    // re-centre on origin

        // Per-frame frustum entity
        for k in 0..<n {
            let base = k * 16
            let R = simd_float3x3(
                SIMD3<Float>(posesCPU[base + 0], posesCPU[base + 4], posesCPU[base + 8]),
                SIMD3<Float>(posesCPU[base + 1], posesCPU[base + 5], posesCPU[base + 9]),
                SIMD3<Float>(posesCPU[base + 2], posesCPU[base + 6], posesCPU[base + 10])
            )
            let t = positions[k]

            // Build c2w transform. RealityKit Transform: scale * rotation + translation.
            var transform = Transform()
            transform.rotation = simd_quatf(R)
            transform.translation = t

            let hue = Float(k) / Float(max(n - 1, 1))
            let frustum = makeFrustumEntity(size: frustumSize, aspect: aspect, hue: hue)
            frustum.transform = transform
            root.addChild(frustum)
        }

        // Trajectory polyline (between consecutive frame positions).
        if positions.count >= 2 {
            let traj = makePolylineEntity(points: positions, color: .init(red: 0.9, green: 0.9, blue: 0.9, alpha: 1))
            root.addChild(traj)
        }

        // World axes (RGB = XYZ).
        let axisLen = frustumSize * 4
        root.addChild(makeAxisEntity(direction: SIMD3<Float>(axisLen, 0, 0), color: .init(red: 1.0, green: 0.2, blue: 0.2, alpha: 1)))
        root.addChild(makeAxisEntity(direction: SIMD3<Float>(0, axisLen, 0), color: .init(red: 0.2, green: 1.0, blue: 0.2, alpha: 1)))
        root.addChild(makeAxisEntity(direction: SIMD3<Float>(0, 0, axisLen), color: .init(red: 0.3, green: 0.5, blue: 1.0, alpha: 1)))

        content.add(root)
    }

    // MARK: - Geometry primitives

    /// Frustum: 4 lines from apex (origin) to base corners + 4 base outline.
    @MainActor
    private func makeFrustumEntity(size: Float, aspect: Float, hue: Float) -> Entity {
        let depth = size
        let halfW = size * 0.5 * aspect
        let halfH = size * 0.5
        let apex = SIMD3<Float>(0, 0, 0)
        let bl = SIMD3<Float>(-halfW, -halfH, depth)
        let br = SIMD3<Float>( halfW, -halfH, depth)
        let tr = SIMD3<Float>( halfW,  halfH, depth)
        let tl = SIMD3<Float>(-halfW,  halfH, depth)

        let nsCol = NSColor(hue: CGFloat(hue), saturation: 0.85, brightness: 1.0, alpha: 1.0)
        let color = colorFromNS(nsCol)

        let parent = Entity()
        for (a, b) in [(apex, bl), (apex, br), (apex, tr), (apex, tl),
                       (bl, br), (br, tr), (tr, tl), (tl, bl)] {
            parent.addChild(makeLineSegmentEntity(start: a, end: b, color: color, thickness: size * 0.025))
        }
        return parent
    }

    @MainActor
    private func makePolylineEntity(points: [SIMD3<Float>], color: SimpleMaterial.Color) -> Entity {
        let parent = Entity()
        for i in 0..<(points.count - 1) {
            parent.addChild(makeLineSegmentEntity(start: points[i], end: points[i + 1], color: color, thickness: 0.005))
        }
        return parent
    }

    @MainActor
    private func makeAxisEntity(direction: SIMD3<Float>, color: SimpleMaterial.Color) -> Entity {
        makeLineSegmentEntity(start: SIMD3<Float>(0, 0, 0), end: direction, color: color, thickness: 0.005)
    }

    /// RealityKit has no native line primitive — render lines as long thin
    /// boxes oriented from `start` to `end`.
    @MainActor
    private func makeLineSegmentEntity(
        start: SIMD3<Float>, end: SIMD3<Float>, color: SimpleMaterial.Color, thickness: Float
    ) -> Entity {
        let delta = end - start
        let length = simd_length(delta)
        if length < 1e-6 { return Entity() }
        let mesh = MeshResource.generateBox(size: SIMD3<Float>(thickness, thickness, length))
        var mat = SimpleMaterial()
        mat.color = .init(tint: color)
        mat.metallic = 0
        mat.roughness = 0.9
        let model = ModelEntity(mesh: mesh, materials: [mat])

        // Orient the box so that its +Z points from start → end.
        let dir = simd_normalize(delta)
        let zAxis = SIMD3<Float>(0, 0, 1)
        let dot = simd_dot(zAxis, dir)
        if abs(dot - 1.0) < 1e-6 {
            // already aligned
        } else if abs(dot + 1.0) < 1e-6 {
            // 180° — pick any perpendicular axis to flip around
            model.transform.rotation = simd_quatf(angle: .pi, axis: SIMD3<Float>(1, 0, 0))
        } else {
            let axis = simd_normalize(simd_cross(zAxis, dir))
            let angle = acos(max(-1, min(1, dot)))
            model.transform.rotation = simd_quatf(angle: angle, axis: axis)
        }
        model.transform.translation = (start + end) * 0.5
        return model
    }

    private func colorFromNS(_ ns: NSColor) -> SimpleMaterial.Color {
        let conv = ns.usingColorSpace(.deviceRGB) ?? ns
        return .init(red: conv.redComponent, green: conv.greenComponent, blue: conv.blueComponent, alpha: conv.alphaComponent)
    }
}
