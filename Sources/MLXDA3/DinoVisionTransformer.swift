import MLX
import MLXNN

/// DA3's DinoV2 backbone with local/global attention and per-layer feature control.
public class DinoVisionTransformer: Module {
    /// Diagnostic hook: called once with the cam_token tensor [B, S, D] at altStart.
    public static var _dumpCamToken: ((MLXArray) -> Void)?
    /// Diagnostic hook: called once with tokens right AFTER cam_token injection.
    public static var _dumpTokensPostCam: ((MLXArray) -> Void)?
    /// Diagnostic hook: called once with tokens immediately after patch embedding.
    public static var _dumpTokensAfterEmbed: ((MLXArray) -> Void)?
    /// Diagnostic hook: called after each transformer block with (layer_idx, tokens).
    public static var _dumpAfterBlock: ((Int, MLXArray) -> Void)?
    private let embedDim: Int
    private let patchSize: Int
    private let altStart: Int
    private let catToken: Bool
    private let outLayers: [Int]
    private let depth: Int
    private let ropeStart: Int

    @ModuleInfo(key: "embeddings") private var embeddings: Embeddings
    private let rope: RotaryPositionEmbedding2D?
    private let positionGetter: PositionGetter?
    @ParameterInfo(key: "camera_token") public var cameraToken: MLXArray?
    @ModuleInfo(key: "blocks") private var blocks: [DA3Block]
    @ModuleInfo(key: "norm") private var norm: LayerNorm

    public init(
        imgSize: Int = 518,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 1024,
        depth: Int = 24,
        numHeads: Int = 16,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        projBias: Bool = true,
        ffnBias: Bool = true,
        initValues: Float = 1.0,
        useSwiglu: Bool = false,
        altStart: Int = -1,
        qknormStart: Int = -1,
        ropeStart: Int = -1,
        ropeFreq: Float = 100.0,
        catToken: Bool = true,
        outLayers: [Int]? = nil
    ) {
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.altStart = altStart
        self.catToken = catToken
        self.outLayers = outLayers ?? []
        self.depth = depth
        self.ropeStart = ropeStart

        self._embeddings.wrappedValue = Embeddings(
            imgSize: imgSize, patchSize: patchSize,
            inChannels: inChannels, embedDim: embedDim
        )

        // RoPE (only created if ropeStart != -1)
        if ropeStart != -1 {
            self.rope = RotaryPositionEmbedding2D(frequency: ropeFreq)
            self.positionGetter = PositionGetter()
        } else {
            self.rope = nil
            self.positionGetter = nil
        }

        // Camera token for multi-view (only if altStart != -1)
        if altStart != -1 {
            self._cameraToken = ParameterInfo(
                wrappedValue: MLXArray.zeros([1, 2, embedDim]),
                key: "camera_token"
            )
        } else {
            self._cameraToken = ParameterInfo(wrappedValue: nil, key: "camera_token")
        }

        // Build blocks with per-layer config
        let ropeRef = self.rope
        let blockArray = (0 ..< depth).map { i in
            DA3Block(
                dim: embedDim,
                numHeads: numHeads,
                mlpRatio: mlpRatio,
                qkvBias: qkvBias,
                projBias: projBias,
                ffnBias: ffnBias,
                initValues: initValues,
                qkNorm: qknormStart != -1 && i >= qknormStart,
                rope: (ropeStart != -1 && i >= ropeStart) ? ropeRef : nil,
                useSwiglu: useSwiglu
            )
        }
        self._blocks.wrappedValue = blockArray
        self._norm.wrappedValue = LayerNorm(dimensions: embedDim)
    }

    // MARK: - Private helpers

    /// Prepare position tensors for RoPE.
    private func prepareRope(B: Int, S: Int, H: Int, W: Int) -> (pos: MLXArray?, posNoDiff: MLXArray?) {
        guard let pg = positionGetter, rope != nil else {
            return (nil, nil)
        }

        let ph = H / patchSize
        let pw = W / patchSize

        // [B*S, ph*pw, 2] -> [B, S, ph*pw, 2]
        var pos = pg.callAsFunction(batchSize: B * S, height: ph, width: pw)
            .reshaped([B, S, ph * pw, 2])
        var posNoDiff = MLXArray.zeros(like: pos)

        // Offset by 1 for CLS token at position 0
        pos = pos + 1
        let clsPos = MLXArray.zeros([B, S, 1, 2])
        pos = concatenated([clsPos, pos], axis: 2)

        posNoDiff = posNoDiff + 1
        let clsPosNoDiff = MLXArray.zeros([B, S, 1, 2])
        posNoDiff = concatenated([clsPosNoDiff, posNoDiff], axis: 2)

        return (pos, posNoDiff)
    }

    /// Apply block with local or global attention pattern.
    /// - x: [B, S, N, C]
    /// - attnType: "local" (per-view) or "global" (cross-view)
    private func processAttention(
        _ x: MLXArray,
        block: DA3Block,
        attnType: String,
        pos: MLXArray?,
        maxPosition: Int?,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let N = x.dim(2)
        let C = x.dim(3)

        var xReshaped: MLXArray
        var posApplied: MLXArray? = nil

        if attnType == "local" {
            xReshaped = x.reshaped([B * S, N, C])
            if let p = pos {
                posApplied = p.reshaped([B * S, p.dim(2), p.dim(3)])
            }
        } else {
            // global: merge views into sequence
            xReshaped = x.reshaped([B, S * N, C])
            if let p = pos {
                posApplied = p.reshaped([B, S * p.dim(2), p.dim(3)])
            }
        }

        let result = block(xReshaped, pos: posApplied, maxPosition: maxPosition, mask: mask)

        // Reshape back to [B, S, N, C]
        if attnType == "local" {
            return result.reshaped([B, S, N, C])
        } else {
            return result.reshaped([B, S, N, C])
        }
    }

    // MARK: - Forward

    /// Forward pass extracting intermediate features.
    ///
    /// - Parameters:
    ///   - x: Input images `[B, S, H, W, C]` (NHWC) where S is num views
    ///   - camToken: Optional camera conditioning tokens `[B, S, D]`
    ///   - refViewStrategy: Reference-view selection strategy. Applied only when
    ///     `camToken` is nil and `S >= ReferenceViewSelection.viewThreshold`, matching
    ///     python `_get_intermediate_layers_not_chunked`.
    /// - Returns: Tuple of (features at out_layers, auxiliary features)
    public func callAsFunction(
        _ x: MLXArray,
        camToken: MLXArray? = nil,
        refViewStrategy: RefViewStrategy = .saddleBalanced
    ) -> ([(MLXArray, MLXArray)], [MLXArray]) {
        let B = x.dim(0)
        let S = x.dim(1)
        let H = x.dim(2)
        let W = x.dim(3)
        let C = x.dim(4)

        // Flatten views for patch embedding: [B*S, H, W, C]
        let xFlat = x.reshaped([B * S, H, W, C])
        var tokens = embeddings(xFlat) // [B*S, 1+N, D]
        tokens = tokens.reshaped([B, S, tokens.dim(1), embedDim]) // [B, S, 1+N, D]
        DinoVisionTransformer._dumpTokensAfterEmbed?(tokens)

        let (pos, posNoDiff) = prepareRope(B: B, S: S, H: H, W: W)
        let maxPosition = rope == nil ? nil : max(H / patchSize, W / patchSize) + 1

        var output: [(MLXArray, MLXArray)] = []
        var localX = tokens

        // Reference-view selection is only active for multi-view input without
        // caller-provided camera tokens.
        let selectsReferenceView =
            altStart != -1 && camToken == nil && S >= ReferenceViewSelection.viewThreshold
        var referenceIndices: MLXArray? = nil

        for i in 0 ..< depth {
            // Determine RoPE positions for this layer
            let gPos: MLXArray?
            let lPos: MLXArray?
            if i < ropeStart || rope == nil {
                gPos = nil
                lPos = nil
            } else {
                gPos = posNoDiff
                lPos = pos
            }

            // Promote the reference view to index 0, one block before the camera token
            // is injected (python: `i == self.alt_start - 1`).
            if selectsReferenceView && i == altStart - 1 {
                let idx = ReferenceViewSelection.select(tokens, strategy: refViewStrategy)
                referenceIndices = idx
                tokens = ReferenceViewSelection.reorder(tokens, referenceIndices: idx)
                localX = ReferenceViewSelection.reorder(localX, referenceIndices: idx)
            }

            // Inject camera tokens at altStart
            if altStart != -1 && i == altStart {
                if let cam = camToken {
                    // tokens[:, :, 0] = cam_token
                    // Python: tokens = tokens.at[:, :, 0].add(cam_token - tokens[:, :, 0])
                    let currentCls = tokens[0..., 0..., 0]  // [B, S, D]
                    tokens = tokens.at[0..., 0..., 0].add(cam - currentCls)
                } else {
                    // Use learned camera_token
                    let refTok = broadcast(cameraToken![0..., ..<1], to: [B, 1, embedDim]) // [B, 1, D]
                    let srcTok = broadcast(cameraToken![0..., 1...], to: [B, S - 1, embedDim]) // [B, S-1, D]
                    let cam = concatenated([refTok, srcTok], axis: 1) // [B, S, D]
                    DinoVisionTransformer._dumpCamToken?(cam)
                    // Replace CLS position with camera token
                    let camExpanded = cam.expandedDimensions(axis: 2) // [B, S, 1, D]
                    let patchTokens = tokens[0..., 0..., 1...] // [B, S, N, D]
                    tokens = concatenated([camExpanded, patchTokens], axis: 2)
                    DinoVisionTransformer._dumpTokensPostCam?(tokens)
                }
            }

            // Local/global attention pattern
            if altStart != -1 && i >= altStart && i % 2 == 1 {
                tokens = processAttention(tokens, block: blocks[i], attnType: "global", pos: gPos, maxPosition: maxPosition)
            } else {
                tokens = processAttention(tokens, block: blocks[i], attnType: "local", pos: lPos, maxPosition: maxPosition)
                localX = tokens
            }
            DinoVisionTransformer._dumpAfterBlock?(i, tokens)

            // Collect outputs at specified layers
            if outLayers.contains(i) {
                var outX: MLXArray
                if catToken {
                    // Concatenate local and global features along feature dim
                    outX = concatenated([localX, tokens], axis: -1)
                } else {
                    outX = tokens
                }
                if let idx = referenceIndices {
                    outX = ReferenceViewSelection.restore(outX, referenceIndices: idx)
                }
                // Camera token is position 0: [B, S, D] or [B, S, 2*D]
                let camTok = outX[0..., 0..., 0]
                output.append((outX, camTok))
            }
        }

        // Apply final norm to outputs
        var processed: [(MLXArray, MLXArray)] = []
        for (outX, camTok) in output {
            let normalized: MLXArray
            if outX.dim(-1) == embedDim {
                normalized = norm(outX)
            } else if outX.dim(-1) == embedDim * 2 {
                // Only norm the global (second half) features
                let firstHalf = outX[.ellipsis, ..<embedDim]
                let secondHalf = norm(outX[.ellipsis, embedDim...])
                normalized = concatenated([firstHalf, secondHalf], axis: -1)
            } else {
                normalized = outX
            }

            // Remove CLS token, keep only patch tokens
            let patchTokens = normalized[0..., 0..., 1...]
            processed.append((patchTokens, camTok))
        }

        return (processed, [])
    }
}
