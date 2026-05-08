# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "addict",
#     "einops",
#     "evo",
#     "faiss-cpu",
#     "huggingface_hub",
#     "matplotlib",
#     "moviepy<2",
#     "numba",
#     "numpy",
#     "omegaconf",
#     "opencv-python",
#     "plyfile",
#     "pycolmap",
#     "pypose",
#     "pyyaml",
#     "safetensors",
#     "scikit-learn",
#     "scipy",
#     "torch",
#     "torchvision",
#     "tqdm",
#     "trimesh",
# ]
# ///
"""
Generate a Swift parity fixture for the streaming DA3 pipeline.

Runs python `da3_streaming.DA3_Streaming` end-to-end on a directory of frames
and serialises the user-visible outputs (camera poses, intrinsics, sim3 pairs,
chunk-0 depth/conf summaries) to a single .safetensors file plus a json
sidecar. The Swift parity test loads this fixture, runs StreamingPipeline on
the same input frames, and compares.

Usage (from this repo root):

    KMP_DUPLICATE_LIB_OK=TRUE uv run --script \
        Scripts/generate_streaming_parity_fixture.py \
        --image-dir tmp/da3_frames \
        --out Tests/Fixtures \
        --weights /path/to/python/Depth-Anything-3/da3_streaming/weights/model.safetensors \
        --weights-config /path/to/python/Depth-Anything-3/da3_streaming/weights/config.json \
        --chunk-size 4 --overlap 2 --limit 8

Notes:
- `da3_streaming` defaults `use_ray_pose=False` (cam_dec path). We force
  `use_ray_pose=True` via a monkey-patch so the fixture compares apples-to-
  apples against Swift (which doesn't port cam_dec).
- The same monkey-patches as `tmp.py` are applied (no-CUDA shims).
"""

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch

# Force pypose's torch.jit compilation BEFORE we monkey-patch torch.tensor etc.
import pypose  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory with input frames")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--weights", required=True, help="Path to model.safetensors")
    parser.add_argument("--weights-config", required=True, help="Path to weights config.json")
    parser.add_argument("--py-repo", default=None, help="Path to python Depth-Anything-3 repo (default: ../../python/Depth-Anything-3)")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--ref-view-strategy", default="first")
    parser.add_argument("--align-lib", default="torch")
    parser.add_argument("--enable-loop-closure", action="store_true",
                        help="Run python with loop_enable=True; emits a fixture suitable for the Swift loop-closure parity test.")
    parser.add_argument("--salad-weights", default=None,
                        help="Path to dino_salad.ckpt; required when --enable-loop-closure is set.")
    parser.add_argument("--loop-chunk-size", type=int, default=20)
    args = parser.parse_args()

    py_repo = args.py_repo or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "python", "Depth-Anything-3")
    )
    assert os.path.isdir(py_repo), f"python repo not found: {py_repo}"

    sys.path.insert(0, py_repo)
    sys.path.insert(0, os.path.join(py_repo, "src"))
    sys.path.insert(0, os.path.join(py_repo, "da3_streaming"))

    # SALAD ckpt was saved on CUDA — force map to CPU for torch.load callers.
    _orig_load = torch.load

    def _cpu_load(*a, **kw):
        kw.setdefault("map_location", "cpu")
        return _orig_load(*a, **kw)

    torch.load = _cpu_load

    # Stub triton (CUDA-only; we use align_lib='torch')
    import types as _t
    _at = _t.ModuleType("loop_utils.alignment_triton")

    def _no_triton(*a, **kw):
        raise RuntimeError("triton path stubbed; use align_lib='torch'")

    _at.robust_weighted_estimate_sim3_triton = _no_triton
    sys.modules["loop_utils.alignment_triton"] = _at

    # No-CUDA shims (Mac)
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = lambda: None  # type: ignore[attr-defined]
        torch.cuda.get_device_capability = lambda *a, **kw: (0, 0)  # type: ignore[assignment]

        @contextlib.contextmanager
        def _na(*a, **kw):
            yield

        torch.cuda.amp.autocast = _na  # type: ignore[assignment]
        torch.Tensor.cuda = lambda self, *a, **kw: self  # type: ignore[assignment]

        for _name in ("tensor", "zeros", "ones", "empty", "full", "arange", "eye", "from_numpy"):
            _orig = getattr(torch, _name)

            def _wrap(orig):
                def _f(*a, **kw):
                    d = kw.get("device")
                    if d is not None and (str(d).startswith("cuda") or (hasattr(d, "type") and d.type == "cuda")):
                        kw["device"] = "cpu"
                    return orig(*a, **kw)
                return _f

            setattr(torch, _name, _wrap(_orig))

    # Force ray-pose path so we compare the same algorithm as Swift.
    import depth_anything_3.api as _api_mod

    _orig_inference = _api_mod.DepthAnything3.inference

    def _wrapped_inference(self, image, *iargs, **ikwargs):
        ikwargs.setdefault("use_ray_pose", True)
        return _orig_inference(self, image, *iargs, **ikwargs)

    _api_mod.DepthAnything3.inference = _wrapped_inference

    # Monkey-patch python LoopDetector.find_loop_closures to use numpy
    # brute-force (no faiss) — faiss-cpu on macOS arm64 segfaults on some
    # descriptor inputs. This matches what swift's LoopDetector does, so the
    # loop pair output is also exactly comparable across the two pipelines.
    if args.enable_loop_closure:
        import loop_utils.loop_detector as _ld

        def _patched_find_loop_closures(self):
            if self.descriptors is None:
                self.extract_descriptors()
            desc = self.descriptors.numpy().astype("float32")
            n = desc.shape[0]
            sim = desc @ desc.T  # [n, n], cosine since L2-normalised
            K = min(self.top_k + 1, n)
            order = np.argsort(-sim, axis=1)[:, :K]
            pairs = []
            for i in range(n):
                for jj in range(K):
                    ni = int(order[i, jj])
                    if ni == i:
                        continue
                    s = float(sim[i, ni])
                    if s > self.similarity_threshold and abs(i - ni) > 10:
                        a, b = (i, ni) if i < ni else (ni, i)
                        pairs.append((a, b, s))
            # dedup + sort
            pairs = list({(a, b): (a, b, s) for a, b, s in pairs}.values())
            pairs.sort(key=lambda x: x[2], reverse=True)
            if self.use_nms and self.nms_threshold > 0:
                pairs = self._apply_nms_filter(pairs, self.nms_threshold)
            self.loop_closures = self._ensure_decending_order(pairs)
            return self.loop_closures

        _ld.LoopDetector.find_loop_closures = _patched_find_loop_closures

    # List frames and limit
    frames = sorted(
        p for p in os.listdir(args.image_dir)
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )[: args.limit]
    if not frames:
        raise RuntimeError(f"no frames found in {args.image_dir}")

    out_dir = os.path.abspath(args.out)
    work_dir = os.path.join(out_dir, "_streaming_work")
    os.makedirs(work_dir, exist_ok=True)

    if args.enable_loop_closure and not args.salad_weights:
        raise SystemExit("--salad-weights is required when --enable-loop-closure is set")

    cfg = {
        "Weights": {
            "DA3": args.weights,
            "DA3_CONFIG": args.weights_config,
            "SALAD": args.salad_weights or "",
        },
        "Model": {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "loop_chunk_size": args.loop_chunk_size,
            "loop_enable": bool(args.enable_loop_closure),
            "useDBoW": False,
            "delete_temp_files": False,
            "align_lib": args.align_lib,
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": args.ref_view_strategy,
            "ref_view_strategy_loop": args.ref_view_strategy,
            "depth_threshold": 15.0,
            "save_depth_conf_result": False,
            "save_debug_info": False,
            "Sparse_Align": {"keypoint_select": "orb", "keypoint_num": 5000},
            "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
            "Pointcloud_Save": {"sample_ratio": 1.0, "conf_threshold_coef": 0.75},
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                "lang_version": "python",
                "max_iterations": 30,
                "lambda_init": "1e-6",  # python uses eval() on this str
            },
        },
    }

    # The streaming module loads weights via relative paths inside da3_streaming/.
    os.chdir(os.path.join(py_repo, "da3_streaming"))
    from da3_streaming import DA3_Streaming  # noqa: E402

    streamer = DA3_Streaming(args.image_dir, work_dir, cfg)
    streamer.run()
    streamer.close()

    # Read back camera_poses.txt + intrinsic.txt
    poses_txt = os.path.join(work_dir, "camera_poses.txt")
    intr_txt = os.path.join(work_dir, "intrinsic.txt")
    poses = np.array(
        [list(map(float, line.split())) for line in open(poses_txt) if line.strip()],
        dtype=np.float32,
    ).reshape(-1, 4, 4)
    intr = np.array(
        [list(map(float, line.split())) for line in open(intr_txt) if line.strip()],
        dtype=np.float32,
    )  # (N, 4) [fx fy cx cy]

    # Sim3 transforms
    sim3 = streamer.sim3_list  # list of (s, R, t) per chunk pair
    sim3_s = (
        np.stack([np.asarray(x[0]).reshape(-1) for x in sim3]).astype(np.float32)
        if sim3 else np.zeros([0, 1], dtype=np.float32)
    )
    sim3_R = (
        np.stack([np.asarray(x[1]) for x in sim3]).astype(np.float32)
        if sim3 else np.zeros([0, 3, 3], dtype=np.float32)
    )
    sim3_t = (
        np.stack([np.asarray(x[2]).reshape(-1) for x in sim3]).astype(np.float32)
        if sim3 else np.zeros([0, 3], dtype=np.float32)
    )

    # Pack into safetensors
    from safetensors.torch import save_file

    fixture = {
        "output.camera_poses_c2w": torch.from_numpy(poses),         # (N, 4, 4)
        "output.intrinsics_pixel": torch.from_numpy(intr),          # (N, 4) fx fy cx cy
        "output.sim3_s":           torch.from_numpy(sim3_s),        # (P,) or (P, 1)
        "output.sim3_R":           torch.from_numpy(sim3_R),        # (P, 3, 3)
        "output.sim3_t":           torch.from_numpy(sim3_t),        # (P, 3)
    }

    # Loop closure outputs (only when loop_enable was True). The dumped
    # `loop_pairs` are FRAME-index pairs from SALAD detection; the dumped
    # `loop_sim3_*` are CHUNK-pair Sim(3) measurements computed from the
    # combined-frame inference (i, j here are *chunk indices*, not frame).
    if args.enable_loop_closure:
        loop_pairs_list = getattr(streamer, "loop_list", None) or []
        loop_pairs = (
            np.array(loop_pairs_list, dtype=np.int32) if loop_pairs_list
            else np.zeros([0, 2], dtype=np.int32)
        )
        loop_sim3 = getattr(streamer, "loop_sim3_list", None) or []
        # loop_sim3 entries: (chunk_a_idx, chunk_b_idx, (s_ab, R_ab, t_ab))
        if loop_sim3:
            loop_idx_pairs = np.array(
                [[int(item[0]), int(item[1])] for item in loop_sim3], dtype=np.int32
            )
            loop_s = np.array([float(item[2][0]) for item in loop_sim3], dtype=np.float32)
            loop_R = np.stack([np.asarray(item[2][1], dtype=np.float32) for item in loop_sim3])
            loop_t = np.stack([np.asarray(item[2][2], dtype=np.float32) for item in loop_sim3])
        else:
            loop_idx_pairs = np.zeros([0, 2], dtype=np.int32)
            loop_s = np.zeros([0], dtype=np.float32)
            loop_R = np.zeros([0, 3, 3], dtype=np.float32)
            loop_t = np.zeros([0, 3], dtype=np.float32)
        fixture["output.loop_frame_pairs"] = torch.from_numpy(loop_pairs)
        fixture["output.loop_chunk_pairs"] = torch.from_numpy(loop_idx_pairs)
        fixture["output.loop_sim3_s"] = torch.from_numpy(loop_s)
        fixture["output.loop_sim3_R"] = torch.from_numpy(loop_R)
        fixture["output.loop_sim3_t"] = torch.from_numpy(loop_t)
        # Also record chunk_indices (start, end) for each chunk so Swift can map
        # frame index -> chunk index without re-running ChunkIndex.compute.
        chunk_idx = np.array(streamer.chunk_indices, dtype=np.int32)  # [n_chunks, 2]
        fixture["output.chunk_indices"] = torch.from_numpy(chunk_idx)

    fixture_name = (
        "da3_streaming_loop_fixture" if args.enable_loop_closure
        else "da3_streaming_fixture"
    )
    out_safetensors = os.path.join(out_dir, fixture_name + ".safetensors")
    save_file(fixture, out_safetensors)

    metadata = {
        # Store basenames only — absolute paths leak the author's filesystem
        # layout. The fixture safetensors is the source of truth; the json
        # sidecar is just human-readable provenance.
        "image_dir": os.path.basename(os.path.abspath(args.image_dir).rstrip(os.sep)),
        "frames": frames,
        "frame_count": len(frames),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "ref_view_strategy": args.ref_view_strategy,
        "align_lib": args.align_lib,
        "use_ray_pose": True,
        "weights": os.path.basename(args.weights),
        "fixture_layout": {
            "camera_poses_c2w": "[N, 4, 4] float32 — per-frame c2w pose",
            "intrinsics_pixel": "[N, 4] float32 — fx fy cx cy per frame",
            "sim3_s": "[P,] float32 — sim3 scale per chunk pair",
            "sim3_R": "[P, 3, 3] float32 — sim3 rotation per chunk pair",
            "sim3_t": "[P, 3] float32 — sim3 translation per chunk pair",
        },
    }
    out_json = os.path.join(out_dir, "da3_streaming_fixture.json")
    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {out_safetensors} ({os.path.getsize(out_safetensors)} bytes)")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
