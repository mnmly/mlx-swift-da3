import Foundation
import MLX

/// Minimal numpy .npy writer (v1.0, little-endian float32).
/// Used by the streaming tool's --dump-fixtures path to emit per-chunk
/// tensors in a format directly loadable via `np.load(path)` in the
/// python parity comparison script.
public enum NpyWriter {

    public static func writeFloat32(_ array: MLXArray, to path: String) throws {
        let arr = array.asType(.float32)
        eval(arr)
        let shape = arr.shape
        let flat = arr.asArray(Float.self)

        var data = Data()
        // Magic + version
        data.append(contentsOf: [0x93])
        data.append("NUMPY".data(using: .ascii)!)
        data.append(contentsOf: [0x01, 0x00])

        let shapeStr = shape.map { String($0) }.joined(separator: ", ")
        let shapeTuple = shape.count == 1 ? "(\(shape[0]),)" : "(\(shapeStr))"
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeTuple), }"
        // Pad header so total prefix (10 bytes) + header length is multiple of 64
        let prefixLen = 10
        let unpadded = prefixLen + header.count + 1  // +1 for trailing newline
        let padTo = ((unpadded + 63) / 64) * 64
        let padding = padTo - unpadded
        header.append(String(repeating: " ", count: padding))
        header.append("\n")

        let headerBytes = header.data(using: .ascii)!
        let headerLen = UInt16(headerBytes.count)
        data.append(UInt8(headerLen & 0xFF))
        data.append(UInt8((headerLen >> 8) & 0xFF))
        data.append(headerBytes)

        flat.withUnsafeBufferPointer { buf in
            data.append(buf.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: buf.count * 4) { ptr in
                Data(bytes: ptr, count: buf.count * 4)
            })
        }

        try data.write(to: URL(fileURLWithPath: path))
    }

    /// Minimal .npy reader: float32 only, parses shape + reads raw bytes.
    public static func readFloat32(from path: String) throws -> MLXArray {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        // Magic check
        guard data.count >= 10 && data[0] == 0x93 else {
            throw NSError(domain: "NpyWriter", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad npy magic"])
        }
        let major = data[6]
        let headerLenLo: UInt16
        let headerLenHi: UInt16
        let headerStart: Int
        if major == 1 {
            headerLenLo = UInt16(data[8])
            headerLenHi = UInt16(data[9]) << 8
            headerStart = 10
        } else {
            // v2/v3 use 4-byte header length, but unused here
            throw NSError(domain: "NpyWriter", code: 2, userInfo: [NSLocalizedDescriptionKey: "unsupported npy version \(major)"])
        }
        let headerLen = Int(headerLenLo | headerLenHi)
        let headerBytes = data.subdata(in: headerStart..<(headerStart + headerLen))
        let header = String(data: headerBytes, encoding: .ascii) ?? ""
        // Parse shape from a string like "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 4, 3, 280, 504), }"
        guard let shapeRange = header.range(of: "'shape':") else {
            throw NSError(domain: "NpyWriter", code: 3, userInfo: [NSLocalizedDescriptionKey: "no shape in header"])
        }
        let after = header[shapeRange.upperBound...]
        guard let openParen = after.firstIndex(of: "("),
              let closeParen = after.range(of: ")", range: openParen..<after.endIndex)?.lowerBound
        else { throw NSError(domain: "NpyWriter", code: 4, userInfo: [NSLocalizedDescriptionKey: "no shape parens"]) }
        let shapeStr = after[after.index(after: openParen)..<closeParen]
        let shape: [Int] = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        // Verify dtype
        guard header.contains("'<f4'") || header.contains("'>f4'") else {
            throw NSError(domain: "NpyWriter", code: 5, userInfo: [NSLocalizedDescriptionKey: "only float32 supported"])
        }
        let dataStart = headerStart + headerLen
        let count = shape.reduce(1, *)
        var floats = [Float](repeating: 0, count: count)
        let _ = floats.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: dataStart..<(dataStart + count * 4))
        }
        return MLXArray(floats, shape)
    }
}
