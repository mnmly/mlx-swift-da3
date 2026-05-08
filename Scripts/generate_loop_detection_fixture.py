# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "addict",
#     "einops",
#     "faiss-cpu",
#     "huggingface_hub",
#     "numpy",
#     "omegaconf",
#     "opencv-python",
#     "packaging",
#     "Pillow",
#     "safetensors",
#     "scipy",
#     "torch",
#     "torchvision",
#     "tqdm",
# ]
# ///
"""
Generate a Swift parity fixture for the SALAD `LoopDetector`.

Runs python's `LoopDetector` on the same frames the swift `da3-loop-tool`
would run on, and saves descriptors + loop pairs into a single
`.safetensors` file plus a json sidecar.

Usage:

    KMP_DUPLICATE_LIB_OK=TRUE uv run --script \
        Scripts/generate_loop_detection_fixture.py \
        --image-dir tmp/da3_frames \
        --weights /path/to/python/Depth-Anything-3/da3_streaming/weights/dino_salad.ckpt \
        --out Tests/Fixtures \
        --image-size 336 336 \
        --batch-size 8 \
        --similarity-threshold 0.5 --top-k 3 \
        --min-frame-distance 1 --no-nms
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--weights", required=True, help="dino_salad.ckpt path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--py-repo", default=None, help="Path to Depth-Anything-3 python repo")
    parser.add_argument("--image-size", nargs=2, type=int, default=[336, 336])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-frame-distance", type=int, default=10)
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument("--nms-threshold", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    py_repo = args.py_repo or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "python", "Depth-Anything-3")
    )
    sys.path.insert(0, py_repo)
    sys.path.insert(0, os.path.join(py_repo, "src"))
    sys.path.insert(0, os.path.join(py_repo, "da3_streaming"))

    # ckpt was saved on CUDA — force map to CPU
    _orig_load = torch.load

    def _cpu_load(*a, **kw):
        kw.setdefault("map_location", "cpu")
        return _orig_load(*a, **kw)

    torch.load = _cpu_load

    # Patch LoopDetector to override min_frame_distance via monkey-patch — python
    # hard-codes `abs(i - j) > 10` in `find_loop_closures`. We re-implement
    # `find_loop_closures` here to accept a configurable min distance, mirroring
    # the swift `Config.minFrameDistance` knob.
    from loop_utils.loop_detector import LoopDetector

    cfg = {
        "Weights": {"SALAD": args.weights},
        "Loop": {
            "SALAD": {
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "similarity_threshold": args.similarity_threshold,
                "top_k": args.top_k,
                "use_nms": not args.no_nms,
                "nms_threshold": args.nms_threshold,
            }
        },
    }
    detector = LoopDetector(args.image_dir, output="/dev/null", config=cfg)

    # Optionally limit frames
    detector.get_image_paths()
    if args.limit > 0:
        detector.image_paths = detector.image_paths[: args.limit]

    detector.load_model()
    detector.extract_descriptors()  # populates detector.descriptors

    # Brute-force matching (faiss IndexFlatIP just does inner product anyway).
    descriptors = detector.descriptors  # torch tensor [N, D] L2-normalised
    print(f"descriptors: shape={tuple(descriptors.shape)} dtype={descriptors.dtype}")
    desc_np = descriptors.numpy().astype(np.float32)
    sim = desc_np @ desc_np.T  # [N, N]
    n = sim.shape[0]
    K = min(args.top_k + 1, n)
    # argsort descending per row
    order = np.argsort(-sim, axis=1)[:, :K]

    pairs = []
    for i in range(n):
        for jj in range(K):
            ni = int(order[i, jj])
            if ni == i:
                continue
            s = float(sim[i, ni])
            if s > args.similarity_threshold and abs(i - ni) > args.min_frame_distance:
                a, b = (i, ni) if i < ni else (ni, i)
                pairs.append((a, b, s))

    # dedup + sort
    pairs = list({(a, b): (a, b, s) for a, b, s in pairs}.values())
    pairs.sort(key=lambda x: x[2], reverse=True)

    if not args.no_nms and args.nms_threshold > 0:
        pairs = detector._apply_nms_filter(pairs, args.nms_threshold)

    # Mirror python's _ensure_decending_order: store as (max(a,b), min(a,b))
    pairs = [(max(a, b), min(a, b), s) for a, b, s in pairs]

    print(f"python found {len(pairs)} loop pairs")
    for i, (a, b, s) in enumerate(pairs[:10]):
        print(f"  ({a}, {b}) sim={s:.4f}")

    # Save fixture
    pairs_arr = np.array([[float(a), float(b), s] for a, b, s in pairs], dtype=np.float32)
    if pairs_arr.size == 0:
        pairs_arr = np.zeros((0, 3), dtype=np.float32)
    desc_arr = descriptors.detach().cpu().float().numpy()

    from safetensors.torch import save_file

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    fixture = {
        "descriptors": torch.from_numpy(desc_arr),
        "loop_pairs": torch.from_numpy(pairs_arr),
    }
    out_path = os.path.join(out_dir, "salad_loop_fixture.safetensors")
    save_file(fixture, out_path)

    metadata = {
        # Basenames only — absolute paths leak the author's filesystem layout.
        "image_dir": os.path.basename(os.path.abspath(args.image_dir).rstrip(os.sep)),
        "frames": [os.path.basename(str(p)) for p in detector.image_paths],
        "frame_count": len(detector.image_paths),
        "image_size": list(args.image_size),
        "batch_size": args.batch_size,
        "similarity_threshold": args.similarity_threshold,
        "top_k": args.top_k,
        "min_frame_distance": args.min_frame_distance,
        "use_nms": not args.no_nms,
        "nms_threshold": args.nms_threshold,
        "weights": os.path.basename(args.weights),
    }
    with open(os.path.join(out_dir, "salad_loop_fixture.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {out_path} (descriptors={desc_arr.shape}, pairs={pairs_arr.shape})")


if __name__ == "__main__":
    main()
