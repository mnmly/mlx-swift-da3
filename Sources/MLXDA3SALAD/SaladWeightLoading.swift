import Foundation
import MLX
import MLXNN

/// Remap a SALAD safetensors key into the Swift-side key path expected by
/// `SaladModel`. Returns `nil` to skip the key.
///
/// Python key conventions:
///   `backbone.model.cls_token`            -> swift `backbone_model.embeddings.cls_token`
///   `backbone.model.pos_embed`            -> swift `backbone_model.embeddings.pos_embed`
///   `backbone.model.patch_embed.proj.*`   -> swift `backbone_model.embeddings.patch_embed.proj.*`
///   `backbone.model.blocks.N.*`           -> swift `backbone_model.blocks.N.*`
///   `backbone.model.norm.*`               -> swift `backbone_model.norm.*`
///   `aggregator.token_features.{0,2}.*`   -> swift `aggregator.token_features_fc{0,2}.*`
///   `aggregator.cluster_features.{0,3}.*` -> swift `aggregator.cluster_features_conv{0,3}.*`
///   `aggregator.score.{0,3}.*`            -> swift `aggregator.score_conv{0,3}.*`
///   `aggregator.dust_bin`                 -> swift `aggregator.dust_bin`
private func remapSaladKey(_ key: String) -> String? {
    if key.hasPrefix("backbone.model.") {
        let rest = String(key.dropFirst("backbone.model.".count))
        if rest == "cls_token" { return "backbone_model.embeddings.cls_token" }
        if rest == "pos_embed" { return "backbone_model.embeddings.pos_embed" }
        if rest.hasPrefix("patch_embed.") {
            let suffix = String(rest.dropFirst("patch_embed.".count))
            return "backbone_model.embeddings.patch_embed.\(suffix)"
        }
        return "backbone_model.\(rest)"
    }
    if key.hasPrefix("aggregator.") {
        let rest = String(key.dropFirst("aggregator.".count))
        if rest == "dust_bin" { return "aggregator.dust_bin" }
        if rest.hasPrefix("token_features.0.") {
            let suffix = String(rest.dropFirst("token_features.0.".count))
            return "aggregator.token_features_fc0.\(suffix)"
        }
        if rest.hasPrefix("token_features.2.") {
            let suffix = String(rest.dropFirst("token_features.2.".count))
            return "aggregator.token_features_fc2.\(suffix)"
        }
        if rest.hasPrefix("cluster_features.0.") {
            let suffix = String(rest.dropFirst("cluster_features.0.".count))
            return "aggregator.cluster_features_conv0.\(suffix)"
        }
        if rest.hasPrefix("cluster_features.3.") {
            let suffix = String(rest.dropFirst("cluster_features.3.".count))
            return "aggregator.cluster_features_conv3.\(suffix)"
        }
        if rest.hasPrefix("score.0.") {
            let suffix = String(rest.dropFirst("score.0.".count))
            return "aggregator.score_conv0.\(suffix)"
        }
        if rest.hasPrefix("score.3.") {
            let suffix = String(rest.dropFirst("score.3.".count))
            return "aggregator.score_conv3.\(suffix)"
        }
        return nil
    }
    return nil
}

/// Load a SALAD safetensors checkpoint into a fresh `SaladModel`.
public func loadSaladModel(weightsURL: URL, dtype: DType = .float32) throws -> SaladModel {
    let model = SaladModel()

    let weights = try loadArrays(url: weightsURL)
    var remapped: [(String, MLXArray)] = []
    var skipped = 0

    for (key, value) in weights {
        guard let newKey = remapSaladKey(key) else {
            skipped += 1
            continue
        }
        var v = value
        // Conv2d weights: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in).
        // The keys we care about: patch_embed.proj.weight, cluster_features_convN.weight, score_convN.weight.
        if newKey.hasSuffix(".weight") && v.shape.count == 4 {
            v = v.transposed(axes: [0, 2, 3, 1])
        }
        if v.dtype == .float32 || v.dtype == .bfloat16 {
            v = v.asType(dtype)
        }
        remapped.append((newKey, v))
    }

    let params = ModuleParameters.unflattened(remapped)
    try model.update(parameters: params, verify: .none)
    eval(model)

    print("Loaded \(remapped.count) SALAD tensors, skipped \(skipped)")
    return model
}
