import CoreGraphics
import Foundation
import ImageIO

public enum ImageDirectoryError: Error {
    case directoryNotFound(String)
    case empty(String)
    case decodeFailed(String)
}

public enum ImageDirectory {
    public static let supportedExtensions: Set<String> = ["jpg", "jpeg", "png"]

    public static func listImagePaths(in directory: String) throws -> [String] {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: directory, isDirectory: &isDir), isDir.boolValue else {
            throw ImageDirectoryError.directoryNotFound(directory)
        }

        let contents = try fm.contentsOfDirectory(atPath: directory)
        let matches = contents.filter { name in
            let ext = (name as NSString).pathExtension.lowercased()
            return supportedExtensions.contains(ext)
        }.sorted()

        if matches.isEmpty {
            throw ImageDirectoryError.empty(directory)
        }

        return matches.map { (directory as NSString).appendingPathComponent($0) }
    }

    public static func loadCGImage(path: String) throws -> CGImage {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let source = CGImageSourceCreateWithURL(url, nil),
              CGImageSourceGetCount(source) > 0,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImageDirectoryError.decodeFailed(path)
        }
        return image
    }
}
