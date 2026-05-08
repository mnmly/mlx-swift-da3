from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Torch Depth Anything 3 inference.")
    parser.add_argument("--model-dir", required=True, help="HF repo id or local model directory.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, mps, cpu.")
    parser.add_argument("--resolution", type=int, default=518, help="Processing resolution.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--export-format", default="mini_npz")
    args = parser.parse_args()

    image_path = str(Path(args.input).expanduser())

    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model_dir).to(args.device)
    model.eval()
    load_s = time.perf_counter() - load_start

    def run_once() -> None:
        prediction = model.inference(
            image=[image_path],
            process_res=args.resolution,
            export_dir=None,
            export_format=args.export_format,
        )
        _ = prediction.depth
        synchronize(args.device)

    for _ in range(max(0, args.warmup)):
        run_once()

    times: list[float] = []
    for _ in range(max(1, args.iterations)):
        synchronize(args.device)
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)

    print("backend=torch")
    print(f"model_dir={args.model_dir}")
    print(f"device={args.device}")
    print(f"input={image_path}")
    print(f"resolution={args.resolution}")
    print(f"load_s={load_s:.6f}")
    print(f"warmup={args.warmup}")
    print(f"iterations={len(times)}")
    print(f"mean_s={statistics.fmean(times):.6f}")
    print(f"median_s={statistics.median(times):.6f}")
    print(f"min_s={min(times):.6f}")
    print(f"max_s={max(times):.6f}")


if __name__ == "__main__":
    main()
