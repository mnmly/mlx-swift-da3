import AppKit
import MLX
import SwiftUI

/// Renders a single-channel depth map as a colour heatmap (turbo / inferno
/// style). Auto-normalises to [min, max] of the depth values so any unit
/// range maps cleanly to a visible gradient.
struct DepthMapView: View {
    let depth: MLXArray  // [H, W] float32

    var body: some View {
        if let img = renderColorMap(depth: depth) {
            Image(nsImage: img)
                .interpolation(.high)
                .resizable()
                .aspectRatio(contentMode: .fit)
        } else {
            Text("Could not render depth map.")
                .foregroundStyle(.red)
        }
    }

    /// Convert `[H, W]` float depth → `NSImage` with a turbo-ish colormap.
    private func renderColorMap(depth: MLXArray) -> NSImage? {
        let h = depth.dim(0)
        let w = depth.dim(1)
        let flat = depth.asType(.float32).asArray(Float.self)
        guard flat.count == h * w, h > 0, w > 0 else { return nil }

        // Find min/max ignoring NaN.
        var lo: Float = .greatestFiniteMagnitude
        var hi: Float = -.greatestFiniteMagnitude
        for v in flat where v.isFinite {
            if v < lo { lo = v }
            if v > hi { hi = v }
        }
        let range = max(hi - lo, 1e-6)

        var rgba = [UInt8](repeating: 0, count: h * w * 4)
        for i in 0..<flat.count {
            let v = flat[i]
            let t: Float
            if !v.isFinite {
                t = 0
            } else {
                t = max(0, min(1, (v - lo) / range))
            }
            let (r, g, b) = turbo(t)
            rgba[i * 4 + 0] = UInt8(r * 255)
            rgba[i * 4 + 1] = UInt8(g * 255)
            rgba[i * 4 + 2] = UInt8(b * 255)
            rgba[i * 4 + 3] = 255
        }

        let bytesPerRow = w * 4
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(
                  width: w, height: h,
                  bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: bytesPerRow, space: cs,
                  bitmapInfo: info, provider: provider,
                  decode: nil, shouldInterpolate: true,
                  intent: .defaultIntent
              )
        else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: w, height: h))
    }

    /// Approx Google "Turbo" colormap (compact polynomial fit).
    /// `t ∈ [0, 1]` → (r, g, b) ∈ [0, 1]³.
    private func turbo(_ t: Float) -> (Float, Float, Float) {
        // Coefficients from the official Google Research turbo lookup (5th-degree fit).
        // https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a
        let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
        let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
        let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
        return (clamp01(r), clamp01(g), clamp01(b))
    }

    private func clamp01(_ x: Float) -> Float { max(0, min(1, x)) }
}
