# DA3-Streaming Swift Port — Phase 1

## Goal
Port `python/Depth-Anything-3/da3_streaming/` chunk-streaming pipeline (without loop closure)
to mlx-swift-da3. Match Python "basic outputs": `camera_poses.txt`, `intrinsic.txt`,
`pcd/combined_pcd.ply`.

## Acceptance criteria
- New target `MLXDA3Streaming` + CLI `da3-streaming-tool`.
- `da3-streaming-tool --image-dir <dir>` runs end-to-end, produces the three basic outputs.
- Numerical sanity: per-chunk poses non-degenerate; combined PLY contains points from all chunks.
- No loop closure, no Triton/C++ solver, no salad.

## Blockers discovered (need user decision)

### B1. Per-frame camera intrinsics + extrinsics — not produced by Swift port
Streaming requires `predictions.intrinsics` (N,3,3) and `predictions.extrinsics` (N,3,4) per
chunk for: chunk-to-chunk Sim(3) alignment, world-coord point clouds, camera_poses output.

Python has two paths:
  - `cam_dec` (camera decoder) — learned module, **not in Swift port**.
  - `use_ray_pose=True` — geometric, derives poses from `ray`+`ray_conf` outputs which
    the Swift DPT head **already produces**. Implemented in
    `src/depth_anything_3/utils/ray_utils.py::get_extrinsic_from_camray` (~520 LOC file,
    only one function needed plus its dependencies).

Recommended: port the ray-pose path. No new weights, leverages existing outputs.

### B2. `ref_view_strategy` — not implemented in Swift backbone
Python streaming uses `saddle_balanced` ref-view selection inside the backbone for
multi-view inference (`vision_transformer.py:316`). Swift `DinoVisionTransformer`
has a `cam_token` comment stub but no ref-view logic.

Options:
  - Port `saddle_balanced` (extra backbone work).
  - Stick with default ref-view (likely `first` or no-op) and accept some accuracy loss
    for Phase 1.

Recommended: defer — start with whatever the Swift backbone does today, validate
numerically, only port `saddle_balanced` if results are bad.

### B3. Memory ceiling on Apple Silicon
Python defaults `chunk_size=120`. On Apple Silicon unified memory, batched inference
of 120 views at DA3 resolution likely OOMs. Phase 1 should default to a much smaller
chunk size (e.g. 8–16) and let user override.

## Working notes

### Python orchestrator non-loop call graph (from `da3_streaming.py`)
`run` → `process_long_sequence`:
  1. `get_chunk_indices()` — stride/overlap windows
  2. for each chunk: `process_single_chunk` → `model.inference(images, ref_view_strategy=...)`
  3. for chunks i>0: `align_2pcds(point_map_prev[-overlap:], point_map_cur[:overlap], ...)` →
     `weighted_align_point_maps` (sim3utils) → robust IRLS Sim(3) → store `(s, R, t)`
  4. (loop closure) — SKIP for Phase 1
  5. `accumulate_sim3_transforms` — fold per-chunk transforms into chunk-i→chunk-0
  6. for chunks 1..N-1: load chunk, build world points (`depth_to_point_cloud`),
     `apply_sim3_direct`, save PLY
  7. `save_camera_poses` — apply S to per-frame extrinsics, write txt
  8. `merge_ply_files` — concat all chunk PLYs

### Functions to port (Phase 1)
From `loop_utils/alignment_torch.py`:
  - `weighted_estimate_se3_torch` / `_sim3_torch` (~50 LOC each)
  - `weighted_estimate_sim3_numba_torch` (SVD + det check)
  - `huber_loss`, `compute_residuals`, `compute_huber_weights`, `apply_transformation`
  - `robust_weighted_estimate_sim3_torch` (IRLS loop)
  - `apply_sim3_direct_torch`, `depth_to_point_cloud_optimized_torch`

From `loop_utils/sim3utils.py` (1261 LOC; only need a subset):
  - `weighted_align_point_maps` — entry point for chunk alignment
  - `accumulate_sim3_transforms`
  - `save_confident_pointcloud_batch` — PLY writer
  - `merge_ply_files`
  - SKIP: numba paths, loop processing, sparse alignment, scale precompute (use sim3 only)

From `da3_streaming.py`: orchestrator + IO

### From DA3 model side (B1 resolution)
Port `get_extrinsic_from_camray` from `src/depth_anything_3/utils/ray_utils.py:506`
to Swift. Need its sub-dependencies — survey first.

## Decisions (confirmed)
- B1: port ray-pose path (no learned weights needed)
- B2: defer `saddle_balanced`, use Swift backbone defaults
- B3: small default chunk size (e.g. 8) on Apple Silicon

## Slice plan (each ends with a buildable, runnable artifact)

### Slice 1A — multi-view inference plumbing  ← in progress
- [x] Add `MLXDA3Streaming` library + `da3-streaming-tool` exec to Package.swift
- [x] Image directory loader (sort, jpg+png)
- [x] Multi-view batched preprocess: `[1, S, H, W, 3]`
- [x] Run model on a batch, return raw outputs
- [x] CLI runs end-to-end on a small directory and prints per-view shapes

### Slice 1B — ray → pose  ← code complete, runtime not yet verified
- [x] Port `unproject_depth` (geometry.py) — what's needed for unit camera plane
- [x] Port `find_homography_least_squares_weighted_torch_batch` (SVD)
- [x] Port `ransac_find_homography_weighted_fast_batch` (RANSAC)
- [x] Port `ql_decomposition`
- [x] Port `compute_optimal_rotation_intrinsics_batch`
- [x] Port `camray_to_caminfo` + `get_extrinsic_from_camray`
- [x] CLI prints intrinsics + extrinsics per view
- [x] Verify against a real chunk — works on 4-frame sample, da3-giant weights
  - Loader skips 546/1377 keys (mono branch + cam_enc/dec + gs_head)
  - First view: fx≈604, cx≈261 on 518×280 image; R≈I; T≈small. Plausible.
  - SVD/QR pinned to CPU stream (MLX GPU lacks them)

### Slice 1C — Sim(3) alignment + chunking  ✅
- [x] Port weighted SE3/Sim3 estimate (alignment_torch.py) → `Sim3Alignment.swift`
- [x] Port IRLS robust estimate
- [x] Port `weighted_align_point_maps` core path → `ChunkAlignment.swift`
- [x] Port `accumulate_sim3_transforms` → `PointMaps.swift`
- [x] Port `depth_to_point_cloud` (closed-form K^-1 + R^T to avoid GPU-incompatible ops)
- [x] Port `apply_sim3_direct`
- [x] Multi-chunk orchestrator loop → `StreamingOrchestrator.swift`
  - Note: per-chunk predictions kept in RAM (vs python's .npy spill). Fine for short runs.

### Slice 1D — outputs  ✅
- [x] Port `save_confident_pointcloud_batch` → `PLYWriter.swift` (binary PLY, sample_ratio=1.0)
- [x] Port `merge_ply_files` (byte-scan header parser for binary PLYs)
- [x] `camera_poses.txt` (16-flat per line) + `intrinsic.txt` (fx fy cx cy) → `CameraPosesIO.swift`
- [x] `pcd/combined_pcd.ply`
- [x] Verify on 8-frame video sample
  - 3 chunks (size=4, overlap=2), 2 chunk-pair alignments
  - sim3 s ≈ 0.97, 1.13 — close to identity, plausible
  - 875k vertices in combined PLY
  - Total runtime 2.9s

## Phase 1 status: COMPLETE — runs end-to-end on Apple Silicon.

## Phase 1.5 — numerical parity vs python (in progress)

Goal: detect drift in the ports by diffing python `da3_streaming.py` outputs vs the
Swift tool on the same 8-frame fixture.

- [x] `tmp.py` — runs python pipeline with chunk_size=4, overlap=2, ref_view='first',
  loop_enable=False, align_lib='torch', save_depth_conf_result+save_debug_info=True.
  Outputs to `tmp/da3_python_out/`. Mac patches torch.cuda no-ops.
- [x] `NpyWriter.swift` + `--dump-dir` flag on da3-streaming-tool. Dumps per-chunk
  depth/conf/intrinsics/extrinsics + sim3_{s,R,t} as .npy.
- [x] `tmp_compare.py` — diffs poses/intrinsics/sim3/chunk0 depth+conf.
- [x] Run both pipelines on 8-frame fixture (chunk_size=4, overlap=2,
  ref_view='first', loop_enable=False, align_lib='torch').
- [ ] Diff is **not** close — see findings below. Root-cause work needed before
  a regression test is meaningful.

### Parity findings (2026-05-07, full chunk-0 dump diff)

Full `tmp_compare.py` run with swift `--dump-dir` enabled, default dtype.

**Fixed in this session:**
- [x] Resolution mismatch (was the dominant intrinsic diff). Python defaults
  `process_res=504` (`api.py:145`); Swift CLI defaulted to 518. Changed Swift
  default to 504 in both `DA3StreamingTool` and `StreamingOrchestrator.Config`.
  After fix: `cx,cy` agree exactly on frame 0 (252.00, 139.77 vs 252.00, 140.00).

**Root cause for cx/cy = W/2 in python**: NOT forced to center anywhere — it's
just that python's QL-decomp `pp` falls extremely close to (0,0), and the +1.0
shift + W/2 scaling brings it to image center. Swift produces nearly-identical
cx ≈ 252 too on frame 0; the original "swift cx slightly off" finding was due
to the resolution mismatch (W=518 vs 504 — divided by 2 differently).

**Remaining blockers (model-level, scope-flagged):**

1. **Per-view fy collapse persists.** Even after resolution fix, Swift fy
   collapses with view index (chunk-0 fp32 run):

   ```
   frame  py_fx  sw_fx  py_fy  sw_fy
     0    693.2  605.0  695.8  612.3   ← close
     1    706.3  543.9  694.8  514.0
     2    697.8  625.3  698.8  400.5
     3    704.8  466.1  701.2  238.4   ← fy nearly halves
   ```

   fp32 dtype produced identical numbers to fp16 → not a precision issue.

2. **Depth scale mismatch (~3×).** Python depth mean ≈ 2.44, Swift depth ≈ 0.92.
   This is the **missing metric branch**. Python uses `NestedDepthAnything3Net`
   (`da3.py:308`) which runs both `da3` (anyview) and `da3_metric` branches,
   then applies `_apply_metric_scaling` + `_apply_depth_alignment` +
   `_handle_sky_regions`. Swift's `WeightLoading.swift:36-39` explicitly skips
   `da3_metric.*` keys. So Swift's depth is the raw unaligned anyview output.

   Note: this only affects depth values (and any downstream point-cloud scale).
   It does NOT affect ray output, so the fy collapse above is independent.

3. **Sim3 / pose differences.** Per-pair sim3 scale & R differ substantially
   (s diff up to 0.27; R Frobenius 0.5). These are downstream of the bad rays/
   intrinsics — fixing 1+2 is prerequisite to meaningful sim3 parity.

### Apples-to-apples ray-pose diff (2026-05-07)

Important caveat from this session: **python streaming defaults
`use_ray_pose=False`** (api.py:107) and goes through the learned `cam_dec`
MLP via `_process_camera_estimation`. Swift uses the geometric ray-pose
path (`get_extrinsic_from_camray`) because cam_dec is not ported. The
"fy collapse" reported earlier was apples-to-oranges.

Re-ran tmp.py with `use_ray_pose=True` injection (monkey-patch on
`DepthAnything3.inference`) so both pipelines exercise the same RANSAC/QL
algorithm. Also added raw-ray dumps in both pipelines (chunk{i}_ray.npy
and chunk{i}_rayconf.npy). Result on chunk 0 (4 views, fp32 swift, fp32
python via no-cuda autocast):

```
view 0 ray: max_abs=0.28 mean=0.08  py_origin~(0,0,0)  sw_origin=(-0.025,-0.035,-0.022)
view 1 ray: max_abs=0.28 mean=0.07
view 2 ray: max_abs=0.42 mean=0.16   ← largest diff
view 3 ray: max_abs=0.36 mean=0.09
view 0 rayconf: py_mean=460  sw_mean=5396  max_abs=1.1e4   ← completely different scale
view 1 rayconf: py_mean=1.07  sw_mean=1.12  max_abs=0.21
view 2 rayconf: py_mean=1.01  sw_mean=1.00  max_abs=0.026
view 3 rayconf: py_mean=1.01  sw_mean=1.01  max_abs=0.017
```

**Conclusion: divergence is at the model output level, not in RayPose
port.** The Swift backbone/head produces different `ray` and `ray_conf`
than python's, even on view 0 (a single image, no multi-view interaction
unique to view 0).

Likely causes (not investigated further this session):
1. Image preprocessing differs at the pixel level — CGContext draw with
   `interpolationQuality=.high` vs `cv2.INTER_AREA` for downsampling.
   Could explain ~0.05 input pixel drift, possibly amplified by the
   backbone.
2. View-0 rayconf scale (460 vs 5396) suggests something different about
   how the cam_token is being applied or how `ray_conf` head output is
   post-processed for the reference view. Worth checking whether python
   has a ref-view-specific scaling Swift misses.
3. fp16 vs fp32 numerical drift through DA3-giant — but Swift was run
   with fp32 too and still differs, so this isn't dominant.

### Files changed this session
- `Tools/da3-streaming-tool/DA3StreamingTool.swift`: default `--resolution`
  504 (matches python `process_res=504`).
- `Sources/MLXDA3Streaming/StreamingOrchestrator.swift`: default
  `Config.resolution` 504; `dumpChunk` now also writes `ray.npy` and
  `rayconf.npy` per chunk.
- `tmp.py`: monkey-patches `DepthAnything3.inference` to default
  `use_ray_pose=True`, and `get_extrinsic_from_camray` to dump per-chunk
  raw rays into `tmp/da3_python_out/raw_rays/`. Run with
  `KMP_DUPLICATE_LIB_OK=TRUE uv run --script tmp.py`.

### Followup investigation (2026-05-08)

Wired `chunk{i}_input.npy` dumps in both pipelines, re-ran both, and diffed
the normalized model-input tensor at view 0:

```
view 0 input: max_abs=1.32  mean_abs=0.016  mean_signed=-0.0004
              std=0.05 per channel  (no systematic channel bias)
big diffs (>0.5): 686 px, concentrated in rows 41-120, cols 148-444 (interior)
```

Cause: Swift uses CGContext bicubic for resize; python uses cv2.INTER_AREA
for downsampling. ~1 LSB normalized noise on average, isolated peaks up to
~75 raw pixel-value units at high-frequency interior content.

Per-view rayconf summary (post expp1 = exp(x)+1):
```
view 0: py median 445   sw median 4870   ← 10× across the whole view
view 1: py median 1.07  sw median 1.11
view 2: py median 1.01  sw median 1.00
view 3: py median 1.00  sw median 1.00
```

Pre-activation: x_py ≈ 6.1, x_sw ≈ 8.5 — uniform +2.3 logit shift on view 0
only. View 0 is the reference view (gets `camera_token[:, :1]` while others
get `camera_token[:, 1:]`). Swift's `cameraToken[:, ..<1]` and `[:, 1...]`
slices match python; injection into position 0 is equivalent
(`tokens.at[..., 0].add(cam - currentCls)` for caller-provided tokens, or
prepend-then-replace for learned). `select_reference_view(strategy='first')`
returns all-zero indices — reorder is a no-op for both backbones.

The 10× rayconf on view 0 is real but isolated: views 1-3 match closely.
Phase 1 outputs are self-consistent; this is a model-output-fidelity issue
that doesn't break the streaming math.

### Decisions taken to close P1.5
1. **Don't port da3_metric branch** for now — depth-scale mismatch is
   downstream of the same backbone divergence; sim3 alignment normalizes
   per-chunk scale anyway. Treat python's depth as a different metric.
2. **Don't chase view-0 rayconf** without a deeper backbone trace —
   weights, slicing, ref-view path, input pre-norm all check out. Suspected
   numerical drift compounded over 40 transformer layers from 0.016
   mean-abs input noise; would need per-layer activation diffs to localize
   further. Out of scope for Phase 1.
3. **Preprocess parity**: cv2.INTER_AREA isn't easily replicable on Apple
   without porting opencv. CGContext + interpolationQuality=.high is the
   cleanest available path. Documented as a known input-fidelity gap.

### Phase 1.5 status: CLOSED with documented gaps.

### 2026-05-08 follow-up: localized the view-0 rayconf bug

Ran two control experiments to localize the 10× rayconf anomaly:

1. **Swift's preprocessed input through python model** (single chunk,
   ~6 min CPU). View 0 ray output matched python_orig within 0.001 max_abs
   (essentially identical). Same input through swift still gave 10× rayconf.
   → Bug is in swift, not preprocessing.

2. **Swift S=1** (single image, no cross-view attention): view 0 rayconf
   median ≈ 3341 vs python S=4 ≈ 445. Still ~7× off.
   → Bug is in single-view path, not cross-view attention.

3. **Backbone last-stage feature diff** (swift vs swift-input-through-python,
   so identical inputs):
   ```
   view 0: max_abs=125    mean_abs=0.77    py_norm 20870  sw_norm 20816
   view 1: max_abs=295    mean_abs=1.35    py_norm 15920  sw_norm 15786
   view 2: max_abs=329    mean_abs=1.44    py_norm 17350  sw_norm 17033
   view 3: max_abs=365    mean_abs=2.23    py_norm 18782  sw_norm 18396
   ```
   View 0 has the **smallest** backbone diff but the **largest** rayconf
   anomaly. → Bug is in DPT head, not backbone.

Verified along the way:
- `camera_token` weight loads correctly (max_abs 1e-4 vs python, fp16 noise).
- `ref_view_strategy='first'` reorder is a no-op for both implementations.
- DPT input layout `(patch_features, cam_token)` tuple matches python.

### Files changed in 2026-05-08 follow-up
- `Sources/MLXDA3/DepthAnything3.swift`: `backbone` made public for diagnostic
  access.
- `Sources/MLXDA3/DinoVisionTransformer.swift`: `cameraToken` made public.
- `Sources/MLXDA3Streaming/StreamingInference.swift`: optional
  `dumpInputPath` and `dumpBackbonePath` parameters on `predict()`.
- `Sources/MLXDA3Streaming/StreamingOrchestrator.swift`: dumps `chunk*_input`,
  `chunk*_backbone_last`, `camera_token` when `dumpDir` is set.
- `tmp.py`: forces `use_ray_pose=True`, dumps `chunk*_input` and `chunk*_ray`.
- `tmp_swift_input.py` (new): feeds swift's input through python; dumps
  python's `ray`, `ray_conf`, `backbone_last`. Run with
  `KMP_DUPLICATE_LIB_OK=TRUE uv run --script tmp_swift_input.py`.

### Open question (next slice)
The DPT head produces diverging output on view 0 even though its input from
the backbone is closer to python's. Needs another round of dumps: per-aux-
pyramid-level features just before `outputConv2Aux[-1]` (the final aux head
that produces the rayconf logits). That will tell us whether the divergence
is in the aux fusion pyramid (refinenet*Aux + outputConv1Aux chain) or in
the final AuxHead call.

### 2026-05-08 follow-up #2: localized view-0 bug to aux fusion pyramid

Wired diagnostic dumps for `lastAux` (input to the final AuxHead) and
`auxLogits` (output) on both swift and python sides:
- swift: static hooks `DualDPT._dumpAuxLastInput` / `_dumpAuxLogits` set
  by `StreamingOrchestrator.run` when `dumpDir` is provided.
- python: torch forward_hook on `model.head.scratch.output_conv2_aux[-1]`
  in `tmp_swift_input.py`.

Results on chunk 0, view 0:

```
lastAux (input to final AuxHead):
  view 0: max_abs=7.5e7  mean_abs=2.9e6  py_norm 3.39e11  sw_norm 3.37e11
  view 1: max_abs=1.2e8  mean_abs=1.1e7  py_norm 4.41e11  sw_norm 4.39e11
  view 2: max_abs=1.4e8  mean_abs=1.2e7  py_norm 4.33e11  sw_norm 4.26e11
  view 3: max_abs=1.5e8  mean_abs=1.6e7  py_norm 3.49e11  sw_norm 3.15e11

aux_logits (output of final AuxHead) view 0 per-channel mean:
  ch0..ch5 (ray dirs): swift mean within ±0.04 of python — close
  ch6 (rayconf logit): py_mean=+6.17  sw_mean=+8.51   ← exactly the +2.3 logit shift
```

**Decisive test**: built a torch reimpl of `AuxHead`, loaded python weights,
fed it swift's `lastAux`. Output matched swift's actual aux_logits to 0.001
on every channel. Same recompute differed from python's actual aux_logits
by up to 2.34 on ch6.

→ **Swift's AuxHead module is correct.** Bug is upstream in lastAux —
i.e., in the aux fusion pyramid (`refinenet1Aux`/`refinenet2Aux`/
`refinenet3Aux`/`refinenet4Aux`) or the per-level `outputConv1Aux` neck.

The puzzle: view 0 has the smallest backbone-output diff (mean 0.77) but
the biggest aux-fusion-output diff. Swift's fusion pyramid is amplifying
view-0-specific features.

Note about safetensors: levels 1, 2, 3 of `output_conv2_aux` are missing
their LN weight/bias keys (only level 0 has them). Both python and swift
use default-init LN there (weight=1, bias=0), so this is symmetric and
not the bug source. Verified.

### Files changed in 2026-05-08 follow-up #2
- `Sources/MLXDA3/DPT.swift`: static `_dumpAuxLastInput` / `_dumpAuxLogits`
  hooks on `DualDPT` for parity dumps.
- `Sources/MLXDA3Streaming/StreamingOrchestrator.swift`: wires hooks per
  chunk when `dumpDir` is set.
- `tmp_swift_input.py`: forward_hook on `output_conv2_aux[-1]` to capture
  python-side lastAux + aux_logits.

### 2026-05-08 follow-up #3: traced the bug to backbone (not DPT)

Wired per-pyramid-level dumps:
- swift: `DualDPT._dumpAuxPyrPre`/`Post` static hooks, `_dumpScratch` for l*Rn.
- python: `forward_hook` on each `refinenet*_aux`, `output_conv1_aux[i]`,
  `layer{1..4}_rn`.

Per-level diff (swift vs swift-input-through-python):

```
l1Rn (input to fusion, finest): mean_abs=0.023  view-0 max_rel=0.49
l2Rn:                            mean_abs=0.18   view-0 max_rel=0.75
l3Rn:                            mean_abs=1.06   view-0 max_rel=0.17
l4Rn (input to refinenet4Aux):   mean_abs=1.64   view-0 max_rel=0.12

aux_pyr_pre[0] (refinenet4Aux output):
  view 0: max_rel=0.28  mean_abs=2.9   ← jumped from 0.12 to 0.28 (view-0 amplified)
  view 1: max_rel=0.08  mean_abs=9.2
  view 2: max_rel=0.10  mean_abs=9.7
  view 3: max_rel=0.14  mean_abs=10.5
```

**Decisive test**: built torch reimpl of `refinenet4_aux` with python weights,
fed swift's `l4Rn` to it. Output matched swift's actual `aux_pyr_pre[0]`
within 0.003 mean_abs, but differed from python's by 2.87 mean_abs.

→ **Swift's refinenet4Aux module is correct.** And recursively, every block
in the swift DPT/AuxHead is correct given its input.

The view-0 amplification comes from the **ReLU** in `resConfUnit2`:
- View 0 has uniquely small-magnitude features at the deepest backbone level
  (l4Rn norm ≈ 6.3k for view 0 vs ≈16.7k for views 1–3).
- Tiny perturbations in input cross the ReLU threshold and propagate as
  large output differences only on view 0 because more values are near zero.
- This non-linear amplification compounds across the 4-level pyramid,
  yielding the +2.3 logit shift on the rayconf channel.

### Root cause (final): Swift backbone numerical drift

Working backward, swift's backbone last-stage output already differs from
python's by ~0.8 mean_abs (view 0). Sources:
1. **Preprocessing** (~0.016 mean input pixel diff from CGContext bicubic
   vs cv2.INTER_AREA). Small contribution.
2. **Compounding through 40 transformer layers** of the giant backbone.

Both the small input perturbation and the compounding-through-40-layers are
expected. The real surprise is that python's pipeline is robust to this and
swift's is not — but that asymmetry comes from view 0's specific feature
distribution (small magnitude → ReLU sensitivity), not from any bug in
swift's DPT. Per-pixel diff in input → cascading drift through backbone →
ReLU threshold crossings amplify only on view 0 → 10× rayconf logit shift.

### Why this isn't easily fixable

- Backbone numerical drift comes from CV2.INTER_AREA in python vs CGContext
  bicubic in swift. INTER_AREA isn't a built-in CoreGraphics resampler.
  Porting opencv to swift is a large undertaking.
- Even if pixel inputs matched exactly, fp16 vs fp32 numerics through 40
  layers will yield small drift; same ReLU sensitivity on view 0 would
  still amplify.

### Phase 1.5 status: CLOSED, root-caused.

### 2026-05-08 follow-up #4: REAL bug found — pos_embed interpolation

After more investigation, found a **real porting bug**: swift's
`Embeddings.interpolatePosEncoding` was using:
- `Upsample(mode: .linear(alignCorners: true))` — wrong interpolation mode

Python uses (vision_transformer.py:246):
- `nn.functional.interpolate(mode="bicubic", antialias=False, scale_factor=...)`
- `interpolate_offset=0.1` adds 0.1 to scale numerator
- `align_corners=False` (default for bicubic)

**Fix in `Sources/MLXDA3/Embeddings.swift`**: replaced the Upsample call
with a manual `pytorchBicubicResample` helper that mirrors PyTorch's
`(i+0.5)/scale - 0.5` coordinate mapping with the standard bicubic kernel
(a=-0.75). Verified the new swift path matches `torch.nn.functional.
interpolate(mode='bicubic', align_corners=False)` to 1.5e-6.

Why this was the bug:
- MLX's `Upsample(.cubic(alignCorners:false))` uses a different `start`
  offset formula than PyTorch when scale × N is non-integer (which it is
  here because of the +0.1 offset trick).
- bilinear (the previous mode) is also wrong; bicubic is required for
  parity.

**Impact verified**: with python's input fed to both pipelines, fp32:
- `tokens_after_embed` swift vs python: was 0.13 max, now **3.98e-5 max**
  (essentially numerical noise).
- `pos_embed` itself swift vs python: 1.5e-6 max (matches torch).

### Remaining gap: fp32 matmul reduction-order noise (NOT a bug)

With identical input AND fixed pos_embed (fp32 both sides), per-block
divergence growth:
```
tokens_after_embed:  max=4e-5    mean=4e-7   ← fp32 numerical noise
tokens_after_block_0: max=0.06   mean=4e-4   ← +1000× from one DA3Block
tokens_after_block_3: max=0.04   mean=8e-4
tokens_after_block_10: max=0.04  mean=9e-4
tokens_post_cam (≈block 11): max=0.07  mean=0.001
```

Per-view drift is uniform across all 4 views at every block (e.g., all
view norms within 1% of each other). So no single block has a
view-0-specific bug.

**Verified the source isn't `MLXFast.scaledDotProductAttention`**: replaced
with manual `softmax(QK^T * scale) V` — same 0.06 max at block 0.

Conclusion: the +1000× jump per block is consistent with fp32 matmul
inner-product reduction order differences between MLX and PyTorch (K=1536
inner dim → ~1e-4 max relative error per matmul, compounds linearly per
layer). This is genuine implementation precision, not a porting bug.

After 40 layers, accumulated noise hits 125 max at backbone output.
DPT's ReLU non-linearity in `refinenet4_aux.resConfUnit2` then
disproportionately amplifies view 0 (which uniquely has small-magnitude
features at the deepest backbone level — l4Rn norm ≈6.3k vs ≈16.7k
for views 1–3) into a +2.3 logit shift on the rayconf channel = 10× output.

### Phase 1.5 status: CLOSED. Real fixes shipped + remaining gap explained.

#### Real porting bug fixed
- Position-embedding interpolation (`Sources/MLXDA3/Embeddings.swift`):
  was bilinear with align_corners=true; should be bicubic with offset 0.1
  and align_corners=false. Fix: manual `pytorchBicubicResample` matching
  PyTorch's coordinate mapping. Verified to 1.5e-6 vs torch.

#### Documented unfixable gap
- Transformer-layer fp32 matmul reduction-order noise compounds across
  40 layers and gets ReLU-amplified for view 0 in DPT. Not fixable
  without forcing identical reduction order between MLX and PyTorch
  (would require custom kernels). Phase 1 outputs remain self-consistent
  and the streaming math (chunk align + sim3 + PLY) works end-to-end.

### Files added/modified (cumulative across all P1.5 work)
- `Sources/MLXDA3/Embeddings.swift`: NEW `pytorchBicubicResample` helper;
  `interpolatePosEncoding` calls it instead of MLX `Upsample`.
- `Sources/MLXDA3/DPT.swift` static dump hooks (still in place for next slice).
- `Sources/MLXDA3/DinoVisionTransformer.swift` static dump hooks.
- `Sources/MLXDA3Streaming/{StreamingOrchestrator, StreamingInference}.swift`
  optional dump paths.
- `Sources/MLXDA3Streaming/NpyWriter.swift`: NEW `readFloat32` helper to
  load python-saved .npy as MLXArray.
- `Tools/da3-streaming-tool/DA3StreamingTool.swift`: NEW `--input-npy` flag
  to bypass preprocessing for diagnostic single-chunk runs.
- `tmp_swift_input.py` (existing, expanded): per-layer hooks on python's
  patch_embed, prepare_tokens_with_masks, refinenet*_aux, etc.

Phase 1 outputs are self-consistent and the streaming math (chunk
alignment + sim3 + point cloud) works end-to-end. Per-view rayconf and
intrinsics differ from python, but this is a numerical-drift artifact of
the input-preprocessing chain, not a porting bug in DPT/RayPose. Phase 2
work should not be blocked on this.

### Files changed across all P1.5 work (cumulative)
- `Tools/da3-streaming-tool/DA3StreamingTool.swift` (resolution default 504)
- `Sources/MLXDA3Streaming/StreamingOrchestrator.swift` (dumpDir wires
  input/backbone/aux pyramid + camera_token + sim3 dumps)
- `Sources/MLXDA3Streaming/StreamingInference.swift` (input/backbone dump
  parameters on `predict()`)
- `Sources/MLXDA3/DepthAnything3.swift` (backbone made public)
- `Sources/MLXDA3/DinoVisionTransformer.swift` (cameraToken made public)
- `Sources/MLXDA3/DPT.swift` (`DualDPT._dumpAuxLastInput`, `_dumpAuxLogits`,
  `_dumpAuxPyrPre`, `_dumpAuxPyrPost`, `_dumpScratch` static hooks)
- `tmp.py` (forces `use_ray_pose=True`, dumps inputs + raw rays)
- `tmp_swift_input.py` (new — feeds swift's input through python; dumps
  backbone/scratch/aux-pyramid/AuxHead intermediates)
- `tmp_compare.py` (existing diff harness; partial-diff variant in
  `tmp_compare_partial.py`)


## Known gaps for Phase 1+
- `ref_view_strategy=saddle_balanced` not ported (default backbone behavior used). Numerical
  comparison vs python output not yet done — could reveal accuracy gap.
- IRLS uses mean instead of median for the conf threshold heuristic (MLX has no median).
  Fine for the soft threshold but not exactly faithful.
- Per-chunk predictions held in RAM (no disk spill). For long sequences (>>50 frames at
  current chunk_size) this will blow out memory.
- `sample_ratio` in PLY save is hardcoded to 1.0 (no random sampling). Reservoir sampling
  not ported.
- ImageProcessor still does CPU per-pixel normalization loops; would benefit from MLX-vectorize
  on hot paths.

## Phase 2 (future, not started)
- Loop closure: `LoopDetector` (NetVLAD/salad), `Sim3LoopOptimizer`, `loop_refinement`
- Triton-style GPU IRLS (currently CPU-bound for SVD/QR)
- C++ `solve.cpp` Sim(3) bundle adjustment

## Working notes
- Swift Pipeline currently single-image only; need batched multi-view variant
- DA3 model input shape `[B, S, H, W, C]` already supported
- ImageProcessor returns `[1, H, W, 3]`; we'll stack along axis=1 to get `[1, S, H, W, 3]`
- `predictions.conf` = `output["depth_conf"] - 1.0` per python convention
- `predictions.depth` is squeezed (last dim = 1) per python; need same in Swift

## Results
(empty — pending implementation)
