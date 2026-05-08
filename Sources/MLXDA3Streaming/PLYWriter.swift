import Foundation
import MLX

/// Binary little-endian PLY writer matching python `save_confident_pointcloud_batch` output.
public enum PLYWriter {

    /// Save a confidence-thresholded point cloud (binary PLY, no sampling — sample_ratio=1.0
    /// path of the python reference).
    ///
    /// - points: `[N, H, W, 3]` (or `[K, 3]` flat) Float32
    /// - colors: `[N, H, W, 3]` (or `[K, 3]` flat) UInt8
    /// - confs:  `[N, H, W]` (or `[K]` flat) Float32
    public static func saveConfidentPointCloud(
        points: MLXArray, colors: MLXArray, confs: MLXArray,
        confThreshold: Float, outputPath: String
    ) throws {
        let pts: MLXArray
        let cls: MLXArray
        let cfs: MLXArray
        if points.ndim == 4 {
            let N = points.dim(0), H = points.dim(1), W = points.dim(2)
            pts = points.reshaped([N * H * W, 3])
            cls = colors.reshaped([N * H * W, 3])
            cfs = confs.reshaped([N * H * W])
        } else {
            pts = points; cls = colors; cfs = confs
        }
        eval(pts, cls, cfs)

        let confCPU: [Float] = cfs.asArray(Float.self)
        let total = confCPU.count
        var keep = [Bool](repeating: false, count: total)
        var keepCount = 0
        for i in 0..<total {
            let c = confCPU[i]
            if c >= confThreshold && c > 1e-5 {
                keep[i] = true
                keepCount += 1
            }
        }

        let ptsCPU: [Float] = pts.asArray(Float.self)
        let clsCPU: [UInt8] = cls.asArray(UInt8.self)

        // Build raw PLY body. Each vertex = 3*float32 + 3*uint8 = 15 bytes.
        var body = Data()
        body.reserveCapacity(keepCount * 15)
        var floatBuf = [Float](repeating: 0, count: 3)
        var byteBuf = [UInt8](repeating: 0, count: 3)
        for i in 0..<total where keep[i] {
            floatBuf[0] = ptsCPU[i * 3]
            floatBuf[1] = ptsCPU[i * 3 + 1]
            floatBuf[2] = ptsCPU[i * 3 + 2]
            byteBuf[0] = clsCPU[i * 3]
            byteBuf[1] = clsCPU[i * 3 + 1]
            byteBuf[2] = clsCPU[i * 3 + 2]
            floatBuf.withUnsafeBytes { body.append(contentsOf: $0) }
            byteBuf.withUnsafeBytes { body.append(contentsOf: $0) }
        }

        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(keepCount)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        let url = URL(fileURLWithPath: outputPath)
        let fm = FileManager.default
        try fm.createDirectory(atPath: (outputPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        if fm.fileExists(atPath: outputPath) {
            try fm.removeItem(at: url)
        }
        fm.createFile(atPath: outputPath, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.write(contentsOf: header.data(using: .ascii)!)
        try handle.write(contentsOf: body)
    }

    /// Merge all `*_pcd.ply` files in `inputDir` into a single binary PLY.
    public static func mergePLYFiles(inputDir: String, outputPath: String) throws {
        let fm = FileManager.default
        let names = try fm.contentsOfDirectory(atPath: inputDir)
            .filter { $0.hasSuffix("_pcd.ply") }
            .sorted()
        if names.isEmpty {
            print("[merge] no PLY files in \(inputDir)")
            return
        }
        // Read each header to get vertex counts and body offsets.
        var totalVertices = 0
        var bodies: [(path: String, offset: Int)] = []
        for name in names {
            let path = (inputDir as NSString).appendingPathComponent(name)
            let url = URL(fileURLWithPath: path)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }

            // Read up to 4 KB and scan bytes for newlines to extract header lines.
            // The body after `end_header\n` is binary, so we can't decode it as ASCII.
            let chunk = try handle.read(upToCount: 4096) ?? Data()
            var vertexCount = 0
            var headerEnd = 0
            var lineStart = 0
            for i in 0..<chunk.count {
                if chunk[i] == 0x0A {  // '\n'
                    let lineBytes = chunk[lineStart..<i]
                    let line = String(decoding: lineBytes, as: UTF8.self)
                    if line.hasPrefix("element vertex") {
                        let parts = line.split(separator: " ")
                        if let last = parts.last, let n = Int(last) { vertexCount = n }
                    }
                    if line.hasPrefix("end_header") {
                        headerEnd = i + 1
                        break
                    }
                    lineStart = i + 1
                }
            }
            if headerEnd == 0 {
                throw NSError(domain: "PLY", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "could not find end_header in \(path)"
                ])
            }
            totalVertices += vertexCount
            bodies.append((path, headerEnd))
        }

        // Write merged file.
        let outURL = URL(fileURLWithPath: outputPath)
        try fm.createDirectory(atPath: (outputPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        if fm.fileExists(atPath: outputPath) { try fm.removeItem(at: outURL) }
        fm.createFile(atPath: outputPath, contents: nil)
        let outHandle = try FileHandle(forWritingTo: outURL)
        defer { try? outHandle.close() }

        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(totalVertices)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        try outHandle.write(contentsOf: header.data(using: .ascii)!)

        for body in bodies {
            let url = URL(fileURLWithPath: body.path)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(body.offset))
            while true {
                let buf = try handle.read(upToCount: 1 << 20) ?? Data()
                if buf.isEmpty { break }
                try outHandle.write(contentsOf: buf)
            }
        }
        print("[merge] wrote \(totalVertices) vertices → \(outputPath)")
    }
}
