from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Torch DA3 parity fixtures.")
    parser.add_argument("--model-dir", required=True, help="HF repo id or local snapshot directory.")
    parser.add_argument("--out", required=True, help="Output fixture directory.")
    parser.add_argument("--device", default="mps", help="Torch device: mps, cuda, or cpu.")
    parser.add_argument("--height", type=int, default=70, help="Input height, divisible by 14.")
    parser.add_argument("--width", type=int, default=98, help="Input width, divisible by 14.")
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 14 != 0 or args.width % 14 != 0:
        raise ValueError("--height and --width must be divisible by 14")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = DepthAnything3.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    # Deterministic already-normalized model input. Torch uses BCHW inside the
    # public forward; the Swift port uses BSHWC, so save both layouts.
    input_nchw = torch.randn(
        1, 1, 3, args.height, args.width,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        dtype=torch.float32,
    )
    input_nhwc = input_nchw.permute(0, 1, 3, 4, 2).contiguous()

    with torch.inference_mode():
        image = input_nchw.to(args.device)
        raw = model.forward(image, export_feat_layers=[])
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device.startswith("cuda"):
            torch.cuda.synchronize()

    tensors: dict[str, torch.Tensor] = {
        "input_nchw": input_nchw.cpu(),
        "input_nhwc": input_nhwc.cpu(),
    }

    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            tensors[f"output.{key}"] = value.detach().float().cpu()

    fixture_path = out_dir / "da3_large_forward.safetensors"
    save_file(tensors, str(fixture_path))

    metadata = {
        "model_dir": args.model_dir,
        "device": args.device,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "keys": sorted(tensors.keys()),
    }
    (out_dir / "da3_large_forward.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {fixture_path}")


if __name__ == "__main__":
    # Allows the script to run past duplicate OpenMP runtimes from optional deps
    # in the upstream DA3 package on macOS.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
