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
Benchmark python `da3_streaming` end-to-end.

Mirrors the protocol of `Tools/da3-streaming-bench/`: warmup iterations + N
measured iterations of the full streaming run on the same input frames.
Reports `mean_s / median_s / min_s / max_s`, `load_s`, and warmup count so
the Swift and python numbers can be compared side-by-side.

Usage:

    KMP_DUPLICATE_LIB_OK=TRUE uv run --script \
        Benchmarks/torch_da3_streaming_bench.py \
        --image-dir tmp/da3_frames \
        --weights /path/to/python/Depth-Anything-3/da3_streaming/weights/model.safetensors \
        --weights-config /path/to/python/Depth-Anything-3/da3_streaming/weights/config.json \
        --chunk-size 4 --overlap 2 --limit 8 --iterations 5

Notes:
- Forces `use_ray_pose=True` to match Swift's algorithm. Same monkey-patches
  as `Scripts/generate_streaming_parity_fixture.py`.
- Each iteration writes to a fresh temp dir to avoid file-cache effects.
"""

import argparse
import contextlib
import os
import shutil
import statistics
import sys
import tempfile
import time

import torch
import pypose  # noqa: F401  — force jit before patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights-config", required=True)
    parser.add_argument("--py-repo", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--ref-view-strategy", default="first")
    parser.add_argument("--align-lib", default="torch")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    py_repo = args.py_repo or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "python", "Depth-Anything-3")
    )
    sys.path.insert(0, py_repo)
    sys.path.insert(0, os.path.join(py_repo, "src"))
    sys.path.insert(0, os.path.join(py_repo, "da3_streaming"))

    # Stub triton (CUDA-only)
    import types as _t
    _at = _t.ModuleType("loop_utils.alignment_triton")

    def _no_triton(*a, **kw):
        raise RuntimeError("triton stubbed; use align_lib='torch'")

    _at.robust_weighted_estimate_sim3_triton = _no_triton
    sys.modules["loop_utils.alignment_triton"] = _at

    # No-CUDA shims for Mac CPU runs
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = lambda: None
        torch.cuda.get_device_capability = lambda *a, **kw: (0, 0)

        @contextlib.contextmanager
        def _na(*a, **kw):
            yield

        torch.cuda.amp.autocast = _na
        torch.Tensor.cuda = lambda self, *a, **kw: self
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

    import depth_anything_3.api as _api_mod
    _orig_inference = _api_mod.DepthAnything3.inference

    def _wrapped_inference(self, image, *iargs, **ikwargs):
        ikwargs.setdefault("use_ray_pose", True)
        return _orig_inference(self, image, *iargs, **ikwargs)

    _api_mod.DepthAnything3.inference = _wrapped_inference

    cfg = {
        "Weights": {"DA3": args.weights, "DA3_CONFIG": args.weights_config, "SALAD": ""},
        "Model": {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "loop_chunk_size": 20,
            "loop_enable": False,
            "useDBoW": False,
            "delete_temp_files": True,
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
            "SALAD": {"image_size": [336, 336], "batch_size": 32, "similarity_threshold": 0.85, "top_k": 5, "use_nms": True, "nms_threshold": 25},
            "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": 1e-6},
        },
    }

    os.chdir(os.path.join(py_repo, "da3_streaming"))
    from da3_streaming import DA3_Streaming

    # Cold-load timing
    load_start = time.perf_counter()
    work_dir = tempfile.mkdtemp(prefix="da3_bench_load_")
    streamer = DA3_Streaming(args.image_dir, work_dir, cfg)
    load_s = time.perf_counter() - load_start

    def synchronize() -> None:
        if args.device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        elif args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def run_once() -> None:
        nonlocal streamer
        # Reset chunk indices etc by re-instantiating from the loaded model.
        # (DA3_Streaming holds state across runs; cleanest is a fresh instance.)
        wd = tempfile.mkdtemp(prefix="da3_bench_iter_")
        s = DA3_Streaming(args.image_dir, wd, cfg)
        s.run()
        s.close()
        synchronize()
        shutil.rmtree(wd, ignore_errors=True)

    try:
        for _ in range(max(0, args.warmup)):
            run_once()

        times = []
        for _ in range(max(1, args.iterations)):
            synchronize()
            start = time.perf_counter()
            run_once()
            times.append(time.perf_counter() - start)
    finally:
        try:
            streamer.close()
        except Exception:
            pass
        shutil.rmtree(work_dir, ignore_errors=True)

    print("backend=torch")
    print("pipeline=streaming")
    print(f"weights={args.weights}")
    print(f"device={args.device}")
    print(f"image_dir={args.image_dir}")
    print(f"frames_limit={args.limit}")
    print(f"chunk_size={args.chunk_size}")
    print(f"overlap={args.overlap}")
    print(f"load_s={load_s:.6f}")
    print(f"warmup={args.warmup}")
    print(f"iterations={len(times)}")
    print(f"mean_s={statistics.fmean(times):.6f}")
    print(f"median_s={statistics.median(times):.6f}")
    print(f"min_s={min(times):.6f}")
    print(f"max_s={max(times):.6f}")


if __name__ == "__main__":
    main()
