import AVFoundation
import CoreGraphics
import Foundation

/// Extracts frames from a video URL at a given frame rate using
/// AVAssetImageGenerator. Designed for short clips (≤200 frames) — for
/// long videos, batch frame extraction or use `ffmpeg` upstream.
enum FrameExtractor {
    enum ExtractError: Error {
        case noVideoTrack
        case generationFailed(Error)
    }

    /// Extract evenly-spaced frames from a video.
    /// - Parameters:
    ///   - url: video file URL.
    ///   - fps: target frames per second.
    ///   - maxFrames: cap on returned frame count (sized for memory).
    static func extractFrames(
        from url: URL, fps: Double = 5.0, maxFrames: Int = 64
    ) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        let durationSec = CMTimeGetSeconds(duration)
        guard durationSec > 0 else { return [] }

        let totalFromFps = Int((durationSec * fps).rounded(.down))
        let nFrames = min(max(totalFromFps, 1), maxFrames)
        let step = durationSec / Double(nFrames)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let timeScale: CMTimeScale = 600
        let times: [CMTime] = (0..<nFrames).map { i in
            CMTime(seconds: Double(i) * step, preferredTimescale: timeScale)
        }

        var frames: [CGImage] = []
        frames.reserveCapacity(nFrames)
        for t in times {
            do {
                let cg = try generator.copyCGImage(at: t, actualTime: nil)
                frames.append(cg)
            } catch {
                // Skip individual failed frame requests; the rest still extract.
                continue
            }
        }
        return frames
    }
}
