# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "safetensors",
#     "addict",
#     "einops",
#     "omegaconf",
#     "torch",
# ]
# ///
"""
Generate a Swift parity fixture for reference-view selection.

Imports the real `depth_anything_3.model.reference_view_selector` from a
Depth-Anything-3 checkout and records, for a deterministic random token
tensor, the selected index plus the reordered/restored tensors for every
strategy.

Usage (from repo root):

    uv run --script Scripts/generate_ref_view_fixture.py \
        --reference-repo /path/to/Depth-Anything-3 \
        --out Tests/Fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

STRATEGIES = ["first", "middle", "saddle_balanced", "saddle_sim_range"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("Tests/Fixtures"))
    ap.add_argument("--name", default="ref_view_fixture")
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=7)
    ap.add_argument("--channels", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, str(args.reference_repo / "src"))
    from depth_anything_3.model.reference_view_selector import (  # noqa: E402
        reorder_by_reference,
        restore_original_order,
        select_reference_view,
    )

    torch.manual_seed(0)
    # Per-view scaling makes the class-token statistics (norm / variance /
    # similarity) genuinely differ so the balanced strategy has to work for it.
    x = torch.randn(1, args.views, args.tokens, args.channels, dtype=torch.float32)
    for s in range(args.views):
        x[:, s] *= 1.0 + 0.35 * s

    tensors = {"x": x}
    meta: dict[str, object] = {
        "views": args.views,
        "tokens": args.tokens,
        "channels": args.channels,
        "selected": {},
    }
    for strategy in STRATEGIES:
        b_idx = select_reference_view(x, strategy=strategy)
        reordered = reorder_by_reference(x, b_idx)
        restored = restore_original_order(reordered, b_idx)
        tensors[f"reordered.{strategy}"] = reordered
        tensors[f"restored.{strategy}"] = restored
        meta["selected"][strategy] = [int(v) for v in b_idx]
        assert torch.allclose(restored, x), f"{strategy}: restore is not the inverse of reorder"

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / f"{args.name}.safetensors"))
    (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out / args.name}.safetensors  selected={meta['selected']}")


if __name__ == "__main__":
    main()
