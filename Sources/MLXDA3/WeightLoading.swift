import Foundation
import MLX
import MLXNN

public enum DA3ModelLoadingError: Error, CustomStringConvertible {
    case missingWeights(String)

    public var description: String {
        switch self {
        case let .missingWeights(path):
            return "Could not find DA3 weights at '\(path)'"
        }
    }
}

// MARK: - Key Remapping

/// Split a string at the first occurrence of a separator.
private func splitFirst(_ s: String, separator: Character = ".") -> (String, String)? {
    guard let idx = s.firstIndex(of: separator) else { return nil }
    return (String(s[s.startIndex ..< idx]), String(s[s.index(after: idx)...]))
}

/// Map a PyTorch weight key to the corresponding MLX model key.
/// Returns nil for keys that should be skipped.
func remapKey(_ key: String) -> String? {
    // Strip "model." prefix
    guard key.hasPrefix("model.") else { return nil }
    var rest = String(key.dropFirst("model.".count))

    // Nested checkpoints (e.g. da3nested-giant-large) wrap weights as
    // `model.da3.<...>` (anyview / multi-view branch) and `model.da3_metric.<...>`
    // (mono branch). For Phase 1 streaming we only load the multi-view branch.
    if rest.hasPrefix("da3.") {
        rest = String(rest.dropFirst("da3.".count))
    } else if rest.hasPrefix("da3_metric.") {
        // Skip the metric branch entirely.
        return nil
    }

    // Skip camera encoder/decoder + GS head (not ported)
    if rest.hasPrefix("cam_enc.") || rest.hasPrefix("cam_dec.") {
        return nil
    }
    if rest.hasPrefix("gs_head.") || rest.hasPrefix("gs_adapter.") {
        return nil
    }

    // ---- Backbone keys ----
    if rest.hasPrefix("backbone.pretrained.") {
        rest = String(rest.dropFirst("backbone.pretrained.".count))

        if rest.hasPrefix("patch_embed.") { return "backbone.embeddings.\(rest)" }
        if rest == "cls_token" || rest == "pos_embed" { return "backbone.embeddings.\(rest)" }
        if rest == "camera_token" { return "backbone.\(rest)" }
        if rest.hasPrefix("blocks.") { return "backbone.\(rest)" }
        if rest.hasPrefix("norm.") { return "backbone.\(rest)" }
        return nil
    }

    // ---- Head keys ----
    if rest.hasPrefix("head.") {
        let headRest = String(rest.dropFirst("head.".count))

        if headRest.hasPrefix("norm.") { return rest }
        if headRest.hasPrefix("projects.") { return rest }

        // resize_layers.N.param -> resize_layerN.param
        if headRest.hasPrefix("resize_layers.") {
            let suffix = String(headRest.dropFirst("resize_layers.".count))
            guard let (idxStr, param) = splitFirst(suffix) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 2 { return nil } // Identity layer
            return "head.resize_layer\(idx).\(param)"
        }

        // Scratch layer adapters: scratch.layerN_rn.* -> head.scratch.layerN_rn.*
        if headRest.hasPrefix("scratch.layer") {
            let suffix = String(headRest.dropFirst("scratch.".count))
            return "head.scratch.\(suffix)"
        }

        // Aux output_conv1_aux (check before output_conv1 to avoid false match)
        if headRest.hasPrefix("scratch.output_conv1_aux.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv1_aux.".count))
            return "head.output_conv1_aux.\(suffix)"
        }

        // Aux output_conv2_aux: sequential [0]=conv, [2]=LN, [5]=conv
        if headRest.hasPrefix("scratch.output_conv2_aux.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv2_aux.".count))
            // suffix = "LEVEL.IDX.param"
            guard let (level, rest2) = splitFirst(suffix) else { return nil }
            guard let (idxStr, param) = splitFirst(rest2) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 0 { return "head.output_conv2_aux.\(level).conv0.\(param)" }
            if idx == 2 { return "head.output_conv2_aux.\(level).ln.\(param)" }
            if idx == 5 { return "head.output_conv2_aux.\(level).conv1.\(param)" }
            return nil
        }

        // Main output_conv1: scratch.output_conv1.* -> head.output_conv1.*
        if headRest.hasPrefix("scratch.output_conv1.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv1.".count))
            return "head.output_conv1.\(suffix)"
        }

        // Sky head: scratch.output_conv2_sky.N.* -> head.sky_conv2_N.*
        if headRest.hasPrefix("scratch.output_conv2_sky.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv2_sky.".count))
            guard let (idxStr, param) = splitFirst(suffix) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 0 { return "head.sky_conv2_0.\(param)" }
            if idx == 2 { return "head.sky_conv2_1.\(param)" }
            return nil
        }

        // Main output_conv2: sequential 0=conv, 1=ReLU, 2=conv
        if headRest.hasPrefix("scratch.output_conv2.") {
            let suffix = String(headRest.dropFirst("scratch.output_conv2.".count))
            guard let (idxStr, param) = splitFirst(suffix) else { return nil }
            guard let idx = Int(idxStr) else { return nil }
            if idx == 0 { return "head.output_conv2_0.\(param)" }
            if idx == 2 { return "head.output_conv2_1.\(param)" }
            return nil // ReLU
        }

        // Fusion blocks: scratch.refinenetN.* -> head.refinenetN.*
        if headRest.hasPrefix("scratch.refinenet") {
            let suffix = String(headRest.dropFirst("scratch.".count))
            return "head.\(suffix)"
        }

        // Passthrough for anything else under head that isn't scratch
        return headRest.hasPrefix("scratch.") ? nil : rest
    }

    return nil
}

// MARK: - Conv Transposition

/// Check if a 4D weight tensor needs NCHW -> NHWC transposition.
func isConvWeight(key: String, shape: [Int]) -> Bool {
    key.hasSuffix(".weight") && shape.count == 4
}

/// Check if key belongs to a ConvTranspose2d layer.
func isConvTranspose(key: String) -> Bool {
    key.contains("resize_layer0") || key.contains("resize_layer1")
}

/// Transpose Conv2d weight: PyTorch (O, I, kH, kW) -> MLX (O, kH, kW, I).
func transposeConv2d(_ value: MLXArray) -> MLXArray {
    value.transposed(axes: [0, 2, 3, 1])
}

/// Transpose ConvTranspose2d weight: PyTorch (I, O, kH, kW) -> MLX (O, kH, kW, I).
func transposeConvTranspose2d(_ value: MLXArray) -> MLXArray {
    value.transposed(axes: [1, 2, 3, 0])
}

// MARK: - Public API

/// Load PyTorch DA3 safetensors weights into an MLX model.
///
/// - Parameters:
///   - model: Built MLX model
///   - url: URL to model.safetensors file
///   - dtype: Target dtype (default .float16 for efficiency)
/// - Returns: Model with loaded weights
public func loadWeights(
    model: DepthAnything3,
    url: URL,
    dtype: DType = .float16
) throws -> DepthAnything3 {
    let weights = try loadArrays(url: url)

    var remapped: [(String, MLXArray)] = []
    var skippedCount = 0

    for (key, value) in weights {
        guard let newKey = remapKey(key) else {
            skippedCount += 1
            continue
        }

        var v = value

        // Conv weight transposition
        if isConvWeight(key: newKey, shape: v.shape) {
            if isConvTranspose(key: newKey) {
                v = transposeConvTranspose2d(v)
            } else {
                v = transposeConv2d(v)
            }
        }

        // Cast to target dtype
        if v.dtype == .float32 || v.dtype == .bfloat16 {
            v = v.asType(dtype)
        }

        remapped.append((newKey, v))
    }

    // Load into model
    let params = ModuleParameters.unflattened(remapped)
    try model.update(parameters: params, verify: .none)
    eval(model)

    print("Loaded \(remapped.count) weight tensors, skipped \(skippedCount)")
    return model
}

/// Build and load a DA3 model.
///
/// - Parameters:
///   - configName: Model config name (e.g., "da3-large")
///   - weightsURL: URL to model.safetensors
///   - dtype: Weight dtype
/// - Returns: Loaded DepthAnything3 model
public func loadModel(
    configName: String,
    weightsURL: URL,
    dtype: DType = .float16
) throws -> DepthAnything3 {
    let model = buildModel(configName: configName)
    return try loadWeights(model: model, url: weightsURL, dtype: dtype)
}

private func inferConfigName(from path: String) -> String? {
    let lowercasedPath = path.lowercased()
    let names = ["da3mono-large", "da3-giant", "da3-large", "da3-base", "da3-small"]
    return names.first { lowercasedPath.contains($0) }
}

private func resolveWeightsURL(_ path: String) throws -> URL {
    let url = URL(fileURLWithPath: path)
    var isDirectory: ObjCBool = false

    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
        throw DA3ModelLoadingError.missingWeights(path)
    }

    if isDirectory.boolValue {
        let weightsURL = url.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw DA3ModelLoadingError.missingWeights(weightsURL.path)
        }
        return shimExtensionIfNeeded(weightsURL)
    }
    return shimExtensionIfNeeded(url)
}

/// MLX's `loadArrays` switches on `url.pathExtension` and rejects anything
/// that isn't `safetensors` / `npy` / etc. This breaks for HuggingFace cache
/// files: `~/.cache/huggingface/.../snapshots/<hash>/model.safetensors`
/// is a POSIX symlink whose resolved target lives at `.../blobs/<sha>` with
/// no extension, and `NSOpenPanel` resolves the symlink for us before
/// returning.
///
/// Shim: if the URL has no `safetensors` extension but the underlying file
/// is a safetensors bundle, materialise a sibling symlink with the right
/// extension under `NSTemporaryDirectory()` and hand that to MLX.
private func shimExtensionIfNeeded(_ url: URL) -> URL {
    let ext = url.pathExtension
    if ext == "safetensors" || ext == "npy" {
        return url
    }
    // Resolve symlinks so we always operate on the canonical path.
    let resolved = url.resolvingSymlinksInPath()
    if resolved.pathExtension == "safetensors" || resolved.pathExtension == "npy" {
        return resolved
    }

    // Materialise a tmp symlink with .safetensors extension. We can't write
    // into the user's HF cache (it's owned by huggingface tooling) so we
    // create the alias under tmp keyed on a stable hash of the source path.
    let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("mlx-da3-weight-shim", isDirectory: true)
    try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

    var hasher = Hasher()
    hasher.combine(resolved.path)
    let key = String(UInt(bitPattern: hasher.finalize()), radix: 36)
    let aliased = tmpDir.appendingPathComponent("weights-\(key).safetensors")

    if !FileManager.default.fileExists(atPath: aliased.path) {
        try? FileManager.default.createSymbolicLink(at: aliased, withDestinationURL: resolved)
    }
    if FileManager.default.fileExists(atPath: aliased.path) {
        return aliased
    }
    // Fall back to the original URL if symlinking failed; MLX will then
    // surface the underlying error to the caller.
    return resolved
}

public extension DepthAnything3 {
    /// Load a DA3 model from a local `model.safetensors` file or a directory containing it.
    ///
    /// This mirrors the common `from_pretrained(...)` shape while keeping Swift naming.
    /// If `configName` is omitted, the loader tries to infer it from the path and falls
    /// back to `da3mono-large`.
    static func fromPretrained(
        _ path: String,
        configName: String? = nil,
        dtype: DType = .float16
    ) throws -> DepthAnything3 {
        let weightsURL = try resolveWeightsURL(path)
        let resolvedConfigName = configName ?? inferConfigName(from: path) ?? "da3mono-large"
        return try loadModel(configName: resolvedConfigName, weightsURL: weightsURL, dtype: dtype)
    }
}
