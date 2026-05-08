import Foundation

/// Compute overlapping chunk windows over an image list.
///
/// Mirrors `DA3_Streaming.get_chunk_indices` in the Python reference — for N <= chunkSize
/// returns a single window; otherwise stride = chunkSize - overlap, last window clamped.
public enum ChunkIndex {
    public static func compute(numImages: Int, chunkSize: Int, overlap: Int) -> [(start: Int, end: Int)] {
        precondition(chunkSize > 0, "chunkSize must be positive")
        precondition(overlap >= 0 && overlap < chunkSize, "overlap must be in [0, chunkSize)")

        if numImages <= chunkSize {
            return [(0, numImages)]
        }

        let step = chunkSize - overlap
        let numChunks = (numImages - overlap + step - 1) / step
        var out: [(Int, Int)] = []
        out.reserveCapacity(numChunks)
        for i in 0..<numChunks {
            let start = i * step
            let end = min(start + chunkSize, numImages)
            out.append((start, end))
        }
        return out
    }
}
