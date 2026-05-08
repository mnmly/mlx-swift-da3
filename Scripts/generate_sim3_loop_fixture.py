# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "numpy",
#     "packaging",
#     "pypose",
#     "safetensors",
#     "scipy",
#     "torch",
# ]
# ///
"""
Generate a parity fixture for `Sim3LoopOptimizer` by running python's
`Sim3LoopOptimizer.optimize(...)` on a deterministic ring trajectory.

Usage:

    KMP_DUPLICATE_LIB_OK=TRUE uv run --script \
        Scripts/generate_sim3_loop_fixture.py \
        --out Tests/Fixtures \
        --n-poses 8 \
        --radius 3.0 \
        --rot-noise-deg 5

The fixture stores:
- `input.sequential_srt`  : [N, 8] = [s, R(9 row-major), t(3)] per sequential transform
- `input.loop_edges`      : [M, 2] int = (i, j)
- `input.loop_srt`        : [M, 8] = same layout as sequential
- `output.sequential_srt` : optimized sequential transforms
- json sidecar with config + RNG seed

The Swift test reads this and asserts swift's optimized output is close to
python's (loose tolerances — pypose's analytic Jacobian + scipy sparse solve
vs Swift's numerical Jacobian + dense LU will diverge by some amount).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import pypose as pp  # noqa: F401  — must precede our patches
from scipy.spatial.transform import Rotation as R


def srt_to_flat(srt):
    s, Rmat, t = srt
    return np.concatenate([np.array([s], dtype=np.float32), Rmat.astype(np.float32).flatten(), t.astype(np.float32)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--py-repo", default=None)
    parser.add_argument("--n-poses", type=int, default=8)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--rot-noise-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--lambda-init", type=float, default=1e-6)
    args = parser.parse_args()

    np.random.seed(args.seed)

    py_repo = args.py_repo or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "python", "Depth-Anything-3")
    )
    sys.path.insert(0, py_repo)
    sys.path.insert(0, os.path.join(py_repo, "src"))
    sys.path.insert(0, os.path.join(py_repo, "da3_streaming"))

    from loop_utils.sim3loop import Sim3LoopOptimizer

    # Build a noisy ring of N relative poses, mirroring sim3loop.create_ring_transforms.
    transforms = []
    angle_step = 2 * np.pi / args.n_poses
    for _ in range(args.n_poses):
        angle = angle_step
        Rz = R.from_euler("z", angle, degrees=False)
        noise = R.from_euler("xyz", np.random.normal(0.0, args.rot_noise_deg, 3), degrees=True)
        Rmat = (noise * Rz).as_matrix()
        t = np.array([args.radius * np.sin(angle), args.radius * (1 - np.cos(angle)), 0.0])
        s = np.random.uniform(0.8, 1.2)
        transforms.append((s, Rmat, t))

    # Loop constraint: closing the ring back to identity.
    loop_constraints = [(args.n_poses, 0, (1.0, np.eye(3), np.zeros(3)))]

    cfg = {
        "Loop": {
            "SIM3_Optimizer": {
                "lang_version": "python",
                "max_iterations": args.max_iterations,
                "lambda_init": str(args.lambda_init),
            }
        }
    }
    optimizer = Sim3LoopOptimizer(cfg, device="cpu")
    optimized = optimizer.optimize(transforms, loop_constraints, max_iterations=args.max_iterations, lambda_init=args.lambda_init)

    # Pack inputs
    seq_in = np.stack([srt_to_flat(t) for t in transforms])
    loop_pairs = np.array([[i, j] for i, j, _ in loop_constraints], dtype=np.int32)
    loop_srt = np.stack([srt_to_flat(srt) for _, _, srt in loop_constraints])
    seq_out = np.stack([srt_to_flat(t) for t in optimized])

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    from safetensors.torch import save_file

    fixture = {
        "input.sequential_srt":  torch.from_numpy(seq_in),
        "input.loop_edges":      torch.from_numpy(loop_pairs.astype(np.int64)),
        "input.loop_srt":        torch.from_numpy(loop_srt),
        "output.sequential_srt": torch.from_numpy(seq_out),
    }
    out_st = os.path.join(out_dir, "sim3_loop_fixture.safetensors")
    save_file(fixture, out_st)

    metadata = {
        "n_poses": args.n_poses,
        "radius": args.radius,
        "rot_noise_deg": args.rot_noise_deg,
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "lambda_init": args.lambda_init,
        "fixture_layout": {
            "sequential_srt": "[N, 8] = [s, R(9 row-major), t(3)] per sequential transform",
            "loop_edges": "[M, 2] int64 (i, j)",
            "loop_srt": "[M, 8] same layout as sequential",
        },
    }
    with open(os.path.join(out_dir, "sim3_loop_fixture.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {out_st}")
    print(f"  sequential_srt: {seq_in.shape}, loop_edges: {loop_pairs.shape}, output: {seq_out.shape}")


if __name__ == "__main__":
    main()
