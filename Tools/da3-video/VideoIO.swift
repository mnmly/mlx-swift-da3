import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation

/// Sequential BGRA reader over a video track.
struct VideoSource {
    let url: URL
    let width: Int
    let height: Int
    let fps: Double
    let frameCount: Int

    private let asset: AVAsset
    private let track: AVAssetTrack

    init(url: URL) throws {
        self.url = url
        let asset = AVURLAsset(url: url)
        guard let track = Self.loadTracks(asset).first else {
            throw VideoError("no video track in \(url.lastPathComponent)")
        }
        self.asset = asset
        self.track = track

        let size = track.naturalSize.applying(track.preferredTransform)
        self.width = Int(abs(size.width).rounded())
        self.height = Int(abs(size.height).rounded())
        self.fps = Double(track.nominalFrameRate)
        let duration = CMTimeGetSeconds(asset.duration)
        self.frameCount = max(1, Int((duration * Double(track.nominalFrameRate)).rounded()))
    }

    private static func loadTracks(_ asset: AVAsset) -> [AVAssetTrack] {
        let semaphore = DispatchSemaphore(value: 0)
        var result: [AVAssetTrack] = []
        Task {
            result = (try? await asset.loadTracks(withMediaType: .video)) ?? []
            semaphore.signal()
        }
        semaphore.wait()
        return result
    }

    private func makeReader() throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String:
                                kCVPixelFormatType_32BGRA]
        )
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw VideoError("could not start reading: \(reader.error?.localizedDescription ?? "?")")
        }
        return (reader, output)
    }

    /// Decode the requested frame indices as CGImages, in index order.
    ///
    /// Decoding sequentially and keeping the wanted frames beats seeking: these are
    /// long-GOP or intra codecs where a seek costs as much as a decode, and the
    /// sample set is spread across the whole clip anyway.
    func frames(at indices: Set<Int>) throws -> [CGImage] {
        let (reader, output) = try makeReader()
        defer { reader.cancelReading() }

        var result: [(Int, CGImage)] = []
        var index = 0
        let last = indices.max() ?? 0
        let context = CIContext(options: [.useSoftwareRenderer: false])

        while let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            guard indices.contains(index),
                  let buffer = CMSampleBufferGetImageBuffer(sample)
            else {
                if index > last { break }
                continue
            }
            let ci = CIImage(cvPixelBuffer: buffer)
            guard let cg = context.createCGImage(ci, from: ci.extent) else {
                throw VideoError("could not convert frame \(index)")
            }
            result.append((index, cg))
            if index >= last { break }
        }
        guard !result.isEmpty else { throw VideoError("decoded no frames") }
        return result.sorted { $0.0 < $1.0 }.map(\.1)
    }

    /// Walk every frame in order, handing the raw BGRA buffer to `body`.
    func forEachFrame(limit: Int, _ body: (Int, CVPixelBuffer) throws -> Void) throws {
        let (reader, output) = try makeReader()
        defer { reader.cancelReading() }

        var index = 0
        while index < limit, let sample = output.copyNextSampleBuffer() {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            try body(index, buffer)
            index += 1
        }
        if index < limit {
            throw VideoError("stream ended after \(index) frames, expected \(limit)")
        }
    }
}

/// Writes colour-over-depth frames as HEVC.
final class VideoWriter {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let width: Int
    private let height: Int
    private let fps: Double

    init(url: URL, width: Int, height: Int, fps: Double, bitrate: Int) throws {
        try? FileManager.default.removeItem(at: url)
        self.writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        self.width = width
        self.height = height
        self.fps = fps

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoMaxKeyFrameIntervalKey: 60,
            ],
        ]
        self.input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        self.adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        guard writer.canAdd(input) else { throw VideoError("cannot add writer input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw VideoError("could not start writing: \(writer.error?.localizedDescription ?? "?")")
        }
        writer.startSession(atSourceTime: .zero)
    }

    /// Compose one output frame: `colour` on top, `depthBand` (one byte per pixel,
    /// `width × height/2`) repeated across BGR on the bottom.
    func append(colour: CVPixelBuffer, depthBand: [UInt8], frameIndex: Int) throws {
        while !input.isReadyForMoreMediaData { usleep(2000) }

        guard let pool = adaptor.pixelBufferPool else { throw VideoError("no pixel buffer pool") }
        var maybeBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer) == kCVReturnSuccess,
              let buffer = maybeBuffer
        else { throw VideoError("could not allocate output buffer") }

        CVPixelBufferLockBaseAddress(buffer, [])
        CVPixelBufferLockBaseAddress(colour, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(colour, .readOnly)
            CVPixelBufferUnlockBaseAddress(buffer, [])
        }

        guard let dst = CVPixelBufferGetBaseAddress(buffer),
              let src = CVPixelBufferGetBaseAddress(colour)
        else { throw VideoError("null buffer base address") }

        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let srcStride = CVPixelBufferGetBytesPerRow(colour)
        let colourHeight = height / 2
        let rowBytes = min(dstStride, srcStride)

        for row in 0 ..< colourHeight {
            memcpy(dst.advanced(by: row * dstStride), src.advanced(by: row * srcStride), rowBytes)
        }

        // Depth band: replicate the byte across B, G, R so any channel reads it.
        depthBand.withUnsafeBufferPointer { band in
            for row in 0 ..< colourHeight {
                let out = dst.advanced(by: (colourHeight + row) * dstStride)
                    .assumingMemoryBound(to: UInt8.self)
                let line = band.baseAddress! + row * width
                for x in 0 ..< width {
                    let v = line[x]
                    let p = x * 4
                    out[p] = v; out[p + 1] = v; out[p + 2] = v; out[p + 3] = 255
                }
            }
        }

        let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps.rounded()))
        guard adaptor.append(buffer, withPresentationTime: time) else {
            throw VideoError("append failed at frame \(frameIndex): "
                + (writer.error?.localizedDescription ?? "?"))
        }
    }

    func finish() throws {
        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        if writer.status == .failed {
            throw VideoError("write failed: \(writer.error?.localizedDescription ?? "?")")
        }
    }
}

/// Writes a single-channel depth sequence as 10-bit HEVC.
///
/// Depth is a smooth gradient, which is exactly what an 8-bit band quantises into
/// visible planes. 10-bit luma quadruples the codes; because the signal is grey we
/// can author the YUV planes directly — chroma is a constant — and skip the RGB→YUV
/// conversion and its limited-range squeeze entirely.
///
/// Note the sample packing: `420YpCbCr10BiPlanar` keeps its 10-bit samples
/// **left-aligned** in each 16-bit word. Writing them right-aligned silently throws
/// away six bits (verified: a 512-step ramp came back with 14 distinct values).
final class DepthVideoWriter {
    static let fullRangeFormat = kCVPixelFormatType_420YpCbCr10BiPlanarFullRange

    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let width: Int
    private let height: Int
    private let fps: Double

    init(url: URL, width: Int, height: Int, fps: Double, bitrate: Int) throws {
        try? FileManager.default.removeItem(at: url)
        self.writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        self.width = width
        self.height = height
        self.fps = fps

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel,
                AVVideoAverageBitRateKey: bitrate,
                AVVideoMaxKeyFrameIntervalKey: 30,
            ],
        ]
        self.input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        self.adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Self.fullRangeFormat,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        guard writer.canAdd(input) else { throw VideoError("cannot add depth writer input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw VideoError("could not start depth writer: \(writer.error?.localizedDescription ?? "?")")
        }
        writer.startSession(atSourceTime: .zero)
    }

    /// `samples` is one 10-bit code (0...1023) per pixel, row-major.
    func append(_ samples: [UInt16], frameIndex: Int) throws {
        while !input.isReadyForMoreMediaData { usleep(2000) }

        guard let pool = adaptor.pixelBufferPool else { throw VideoError("no depth buffer pool") }
        var maybeBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer) == kCVReturnSuccess,
              let buffer = maybeBuffer
        else { throw VideoError("could not allocate depth buffer") }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)?
            .assumingMemoryBound(to: UInt16.self),
            let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)?
            .assumingMemoryBound(to: UInt16.self)
        else { throw VideoError("null depth plane") }

        let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) / 2
        samples.withUnsafeBufferPointer { src in
            for row in 0 ..< height {
                let line = src.baseAddress! + row * width
                let out = luma + row * lumaStride
                for x in 0 ..< width { out[x] = line[x] << 6 }
            }
        }
        // Neutral chroma, also left-aligned.
        let chromaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) / 2
        let neutral: UInt16 = 512 << 6
        for row in 0 ..< (height / 2) {
            let out = chroma + row * chromaStride
            for x in 0 ..< width { out[x] = neutral }
        }

        let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps.rounded()))
        guard adaptor.append(buffer, withPresentationTime: time) else {
            throw VideoError("depth append failed at \(frameIndex): "
                + (writer.error?.localizedDescription ?? "?"))
        }
    }

    func finish() throws {
        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        if writer.status == .failed {
            throw VideoError("depth write failed: \(writer.error?.localizedDescription ?? "?")")
        }
    }
}

struct VideoError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
