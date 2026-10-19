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

    /// Intermediate state of a backbone forward pass, between two block ranges.
    ///
    /// Produced by ``prepare(_:)`` and advanced by ``run(_:through:camToken:refViewStrategy:exportFeatLayers:restoreExportOrder:)``.
    /// Because the state carries everything a block needs (tokens, the last local-attention
    /// output, RoPE positions, the reference-view permutation and the outputs collected so
    /// far), a pass can be split across two backbones that share their first blocks: run the
    /// shared prefix once, then resume a copy of the state on each backbone.
    public struct State {
        /// Current tokens `[B, S, 1+N, D]`.
        public var tokens: MLXArray
        /// Output of the most recent local-attention block `[B, S, 1+N, D]`.
        public var localTokens: MLXArray
        /// Index of the next block to run.
        public var nextBlock: Int
        /// Reference-view permutation `[B]`, once reference-view selection has run.
        public var referenceIndices: MLXArray?
        /// Raw (pre-norm) outputs at `outLayers`, in block order.
        public var outputs: [MLXArray] = []
        /// Raw (pre-norm) exported features, in block order.
        public var exported: [MLXArray] = []

        let pos: MLXArray?
        let posNoDiff: MLXArray?
        let maxPosition: Int?
    }

    /// Number of transformer blocks.
    public var blockCount: Int { depth }

    /// Patch-embed the views and build RoPE positions.
    ///
    /// - Parameter x: Input images `[B, S, H, W, C]` (NHWC).
    public func prepare(_ x: MLXArray) -> State {
        let B = x.dim(0)
        let S = x.dim(1)
        let H = x.dim(2)
        let W = x.dim(3)
        let C = x.dim(4)

        // Flatten views for patch embedding: [B*S, H, W, C]
        var tokens = embeddings(x.reshaped([B * S, H, W, C])) // [B*S, 1+N, D]
        tokens = tokens.reshaped([B, S, tokens.dim(1), embedDim]) // [B, S, 1+N, D]
        DinoVisionTransformer._dumpTokensAfterEmbed?(tokens)

        let (pos, posNoDiff) = prepareRope(B: B, S: S, H: H, W: W)
        let maxPosition = rope == nil ? nil : max(H / patchSize, W / patchSize) + 1
        return State(
            tokens: tokens, localTokens: tokens, nextBlock: 0,
            pos: pos, posNoDiff: posNoDiff, maxPosition: maxPosition
        )
    }

    /// Run blocks `state.nextBlock ..< end`.
    ///
    /// Matches python `_get_intermediate_layers_not_chunked`, including InfiniSplat's
    /// prefix-cache resume: running `0 ..< k` and then `k ..< depth` (on this or on another
    /// backbone with identical first `k` blocks) gives the same result as one full pass.
    ///
    /// - Parameters:
    ///   - state: State from ``prepare(_:)`` or a previous call.
    ///   - end: One past the last block to run.
    ///   - camToken: Optional camera conditioning tokens `[B, S, D]`. When set, reference-view
    ///     selection is skipped, as in python.
    ///   - refViewStrategy: Reference-view selection, applied at block `altStart - 1` when
    ///     `camToken` is nil and `S >= ReferenceViewSelection.viewThreshold`.
    ///   - exportFeatLayers: Blocks whose current tokens are also recorded in
    ///     ``State/exported`` (python `export_feat_layers`).
    ///   - restoreExportOrder: Put exported features back in input-view order after
    ///     reference-view reordering. Upstream DA3 leaves them reordered (`false`);
    ///     InfiniSplat's patched backbone restores them (`true`).
    public func run(
        _ state: inout State,
        through end: Int,
        camToken: MLXArray? = nil,
        refViewStrategy: RefViewStrategy = .saddleBalanced,
        exportFeatLayers: Set<Int> = [],
        restoreExportOrder: Bool = false
    ) {
        let B = state.tokens.dim(0)
        let S = state.tokens.dim(1)
        var tokens = state.tokens
        var localX = state.localTokens

        // Reference-view selection is only active for multi-view input without
        // caller-provided camera tokens.
        let selectsReferenceView =
            altStart != -1 && camToken == nil && S >= ReferenceViewSelection.viewThreshold

        for i in state.nextBlock ..< min(end, depth) {
            // Determine RoPE positions for this layer
            let gPos: MLXArray?
            let lPos: MLXArray?
            if i < ropeStart || rope == nil {
                gPos = nil
                lPos = nil
            } else {
                gPos = state.posNoDiff
                lPos = state.pos
            }

            // Promote the reference view to index 0, one block before the camera token
            // is injected (python: `i == self.alt_start - 1`).
            if selectsReferenceView && i == altStart - 1 {
                let idx = ReferenceViewSelection.select(tokens, strategy: refViewStrategy)
                state.referenceIndices = idx
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
                tokens = processAttention(tokens, block: blocks[i], attnType: "global", pos: gPos, maxPosition: state.maxPosition)
            } else {
                tokens = processAttention(tokens, block: blocks[i], attnType: "local", pos: lPos, maxPosition: state.maxPosition)
                localX = tokens
            }
            DinoVisionTransformer._dumpAfterBlock?(i, tokens)

            // Collect outputs at specified layers
            if outLayers.contains(i) {
                // Concatenate local and global features along feature dim
                var outX = catToken ? concatenated([localX, tokens], axis: -1) : tokens
                if let idx = state.referenceIndices {
                    outX = ReferenceViewSelection.restore(outX, referenceIndices: idx)
                }
                state.outputs.append(outX)
            }
            if exportFeatLayers.contains(i) {
                var auxX = tokens
                if restoreExportOrder, let idx = state.referenceIndices {
                    auxX = ReferenceViewSelection.restore(auxX, referenceIndices: idx)
                }
                state.exported.append(auxX)
            }
            state.nextBlock = i + 1
        }
        state.tokens = tokens
        state.localTokens = localX
    }

    /// Normalize the collected outputs and strip the camera/CLS token.
    ///
    /// - Returns: `(features, exported)`. `features` holds one `(patchTokens, camToken)` pair
    ///   per `outLayers` entry: patch tokens `[B, S, N, D·(1+catToken)]` (only the global half
    ///   normed) and the raw camera token `[B, S, D·(1+catToken)]`. `exported` holds normed
    ///   patch tokens `[B, S, N, D]` for each exported layer.
    public func finish(_ state: State) -> ([(MLXArray, MLXArray)], [MLXArray]) {
        var processed: [(MLXArray, MLXArray)] = []
        for outX in state.outputs {
            // Camera token is position 0: [B, S, D] or [B, S, 2*D]
            let camTok = outX[0..., 0..., 0]
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
            processed.append((normalized[0..., 0..., 1...], camTok))
        }
        let exported = state.exported.map { norm($0)[0..., 0..., 1...] }
        return (processed, exported)
    }

    /// Forward pass extracting intermediate features.
    ///
    /// - Parameters:
    ///   - x: Input images `[B, S, H, W, C]` (NHWC) where S is num views
    ///   - camToken: Optional camera conditioning tokens `[B, S, D]`
    ///   - refViewStrategy: Reference-view selection strategy. Applied only when
    ///     `camToken` is nil and `S >= ReferenceViewSelection.viewThreshold`, matching
    ///     python `_get_intermediate_layers_not_chunked`.
    ///   - exportFeatLayers: Blocks whose normed patch tokens are returned as auxiliary
    ///     features (python `export_feat_layers`).
    ///   - restoreExportOrder: See ``run(_:through:camToken:refViewStrategy:exportFeatLayers:restoreExportOrder:)``.
    /// - Returns: Tuple of (features at out_layers, auxiliary features)
    public func callAsFunction(
        _ x: MLXArray,
        camToken: MLXArray? = nil,
        refViewStrategy: RefViewStrategy = .saddleBalanced,
        exportFeatLayers: Set<Int> = [],
        restoreExportOrder: Bool = false
    ) -> ([(MLXArray, MLXArray)], [MLXArray]) {
        var state = prepare(x)
        run(
            &state, through: depth, camToken: camToken, refViewStrategy: refViewStrategy,
            exportFeatLayers: exportFeatLayers, restoreExportOrder: restoreExportOrder
        )
        return finish(state)
    }
}
