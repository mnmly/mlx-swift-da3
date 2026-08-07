# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "opencv-python",
#     "pillow",
#     "safetensors",
#     "torch",
#     "torchvision",
#     "imageio",
#     "tqdm",
# ]
# ///
"""
Generate a Swift parity fixture for DA3 image preprocessing.

Mirrors `src/depth_anything_3/utils/io/input_processor.py::InputProcessor` for
`process_res_method="upper_bound_resize"`: resize the longest side to
`process_res` (cv2 INTER_AREA when shrinking, INTER_CUBIC when growing), then
round each side to the nearest multiple of 14 with a second cv2 resize, then
ToTensor + ImageNet normalize.

If `--reference-repo` is given, the reimplementation is cross-checked against
the real `InputProcessor` from that checkout and the script fails on mismatch.

Usage (from repo root):

    uv run --script Scripts/generate_preprocess_fixture.py \
        --images tmp/da3_frames/frame_000001.png tmp/da3_frames/frame_000002.png \
        --process-res 504 \
        --out Tests/Fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

PATCH_SIZE = 14
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_longest_side(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest == target:
        return img
    scale = target / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return Image.fromarray(cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interp))


def make_divisible_by_resize(img: Image.Image, patch: int) -> Image.Image:
    w, h = img.size

    def nearest_multiple(x: int, p: int) -> int:
        down = (x // p) * p
        up = down + p
        return up if abs(up - x) <= abs(x - down) else down

    new_w = max(1, nearest_multiple(w, patch))
    new_h = max(1, nearest_multiple(h, patch))
    if new_w == w and new_h == h:
        return img
    upscale = (new_w > w) or (new_h > h)
    interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
    return Image.fromarray(cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interp))


def preprocess(path: Path, process_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_uint8 HWC, normalized CHW float32)."""
    img = Image.open(path).convert("RGB")
    img = resize_longest_side(img, process_res)
    img = make_divisible_by_resize(img, PATCH_SIZE)
    rgb = np.asarray(img).astype(np.uint8)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return rgb, normalized


def center_crop_to(arrs: list[np.ndarray], min_h: int, min_w: int) -> list[np.ndarray]:
    out = []
    for a in arrs:
        h, w = a.shape[:2]
        top = (h - min_h) // 2
        left = (w - min_w) // 2
        out.append(a[top : top + min_h, left : left + min_w])
    return out


def cross_check(paths: list[Path], process_res: int, repo: Path) -> None:
    sys.path.insert(0, str(repo / "src"))
    from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: E402

    reference, _, _ = InputProcessor()(
        [str(p) for p in paths],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        num_workers=1,
    )
    reference = reference.numpy()
    while reference.ndim > 4:
        reference = reference[0]
    mine = np.stack([preprocess(p, process_res)[1].transpose(2, 0, 1) for p in paths])
    if mine.shape != reference.shape:
        raise SystemExit(f"shape mismatch vs reference: {mine.shape} != {reference.shape}")
    err = float(np.abs(mine - reference).max())
    if err > 1e-5:
        raise SystemExit(f"reimplementation diverges from InputProcessor: max_abs={err}")
    print(f"cross-check vs reference InputProcessor: max_abs={err:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, type=Path)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--out", type=Path, default=Path("Tests/Fixtures"))
    ap.add_argument("--name", default="preprocess_fixture")
    ap.add_argument(
        "--reference-repo",
        type=Path,
        default=None,
        help="Depth-Anything-3 checkout; cross-checks against the real InputProcessor.",
    )
    args = ap.parse_args()

    if args.reference_repo is not None:
        cross_check(args.images, args.process_res, args.reference_repo)

    rgbs = []
    for path in args.images:
        rgb, _ = preprocess(path, args.process_res)
        rgbs.append(rgb)

    min_h = min(a.shape[0] for a in rgbs)
    min_w = min(a.shape[1] for a in rgbs)
    rgbs = center_crop_to(rgbs, min_h, min_w)

    args.out.mkdir(parents=True, exist_ok=True)
    # Only the uint8 resize result is stored: it is what cv2 actually produced and the
    # only part of the chain that can realistically diverge. ImageNet normalization is
    # three deterministic arithmetic ops, which the Swift test recomputes from this.
    tensors = {"processed_rgb": torch.from_numpy(np.stack(rgbs))}  # NHWC uint8
    save_file(tensors, str(args.out / f"{args.name}.safetensors"))

    meta = {
        "images": [str(p) for p in args.images],
        "process_res": args.process_res,
        "patch_size": PATCH_SIZE,
        "height": min_h,
        "width": min_w,
        "imagenet_mean": MEAN.tolist(),
        "imagenet_std": STD.tolist(),
    }
    (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out / args.name}.safetensors  shape=({len(rgbs)}, {min_h}, {min_w}, 3)")


if __name__ == "__main__":
    main()
