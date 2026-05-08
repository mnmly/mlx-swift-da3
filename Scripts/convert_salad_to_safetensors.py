# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "packaging",
#     "safetensors",
#     "torch",
# ]
# ///
"""
Convert the SALAD VPR `.ckpt` (PyTorch state_dict) to a `.safetensors` file
that Swift can load with `MLX.loadArrays(url:)`.

Usage:

    uv run --script Scripts/convert_salad_to_safetensors.py \
        --in /path/to/dino_salad.ckpt \
        --out /path/to/dino_salad.safetensors
"""

import argparse

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sd = torch.load(args.inp, map_location="cpu", weights_only=True)
    # Some checkpoints wrap state_dict; ours is already a flat OrderedDict.
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Drop mask_token (training-only, never used in inference).
    sd = {k: v for k, v in sd.items() if not k.endswith("mask_token")}

    # Make tensors contiguous + cast to fp32 (safetensors prefers contiguous).
    sd = {k: v.contiguous().to(torch.float32) if torch.is_tensor(v) else v for k, v in sd.items()}

    save_file(sd, args.out)
    print(f"Wrote {len(sd)} tensors → {args.out}")


if __name__ == "__main__":
    main()
