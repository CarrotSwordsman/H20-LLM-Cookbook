#!/usr/bin/env python3
"""
Benchmark default-fallback config vs our tuned H20 config for the
fused MoE Triton kernel in **fp8_w8a8 (per-tensor)** mode, on the same
representative (E, N) shapes and batch sizes as the bf16 harness.

Output: markdown table suitable for posting to vLLM PR #44273 or
`benchmarks/results/fp8_w8a8_default_vs_tuned.md`.

Run from this directory:
    python compare_default_vs_tuned_fp8.py
"""

import json
import math
import sys
from pathlib import Path

# Reuse the existing fp8 benchmark harness in this directory.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune_moe_h20_fp8 import benchmark_one  # noqa: E402

import torch  # noqa: E402


# vLLM's default-fallback config (vllm/model_executor/layers/fused_moe/fused_moe.py:get_default_config).
# Note: BLOCK_SIZE_K=16 is forbidden on the fp8 path, but the default already
# uses 32 / 64, so it's safe.
def get_default_config(M: int, E: int) -> dict:
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
        }
    return {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
    }


# Same 6 representative shapes as the bf16 harness.
SHAPES = [
    # (E, N, K, topk, model_label)
    (8,   14336, 4096, 2, "Mixtral 8x7B"),
    (8,    7168, 4096, 2, "Mixtral 8x7B (alt)"),
    (64,   2560, 2048, 6, "Qwen MoE"),
    (64,   1280, 2048, 6, "Qwen MoE (alt)"),
    (128,  1024, 2048, 6, "DeepSeek-V2-Lite"),
    (128,   512, 2048, 6, "DeepSeek-V2-Lite (alt)"),
]
BATCH_SIZES = [1, 32, 128, 512, 2048, 4096]


def load_tuned_config(E: int, N: int) -> dict:
    """Load the tuned fp8_w8a8 config JSON we generated for H20."""
    cookbook_dir = HERE.parent.parent / "configs" / "fp8_w8a8"
    candidates = [
        cookbook_dir / f"E={E},N={N},device_name=NVIDIA_H20,dtype=fp8_w8a8.json",
        HERE.parent / "vllm" / "model_executor" / "layers" / "fused_moe" /
            "configs" / f"E={E},N={N},device_name=NVIDIA_H20,dtype=fp8_w8a8.json",
    ]
    for fname in candidates:
        if fname.exists():
            with open(fname) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No tuned H20 fp8_w8a8 config for (E={E}, N={N}); tried {candidates}")


def median_of(times, n=3):
    return sorted(times)[len(times) // 2]


def main():
    print(f"# fused MoE kernel (fp8_w8a8, per-tensor): "
          f"default-fallback vs tuned H20 config")
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# torch:  {torch.__version__}")
    import triton
    print(f"# triton: {triton.__version__}")
    print()

    # Markdown header
    print("| Shape (E, N, K, topk) | Batch | default (ms) | tuned (ms) | speedup |")
    print("|---|---:|---:|---:|---:|")

    summary = []  # for averaged speedup later
    for E, N, K, topk, label in SHAPES:
        try:
            tuned_map = load_tuned_config(E, N)
        except FileNotFoundError:
            print(f"# !! tuned JSON missing for (E={E}, N={N}), skipping")
            continue

        for bs in BATCH_SIZES:
            tuned_key = min(tuned_map.keys(), key=lambda k: abs(int(k) - bs))
            tuned = tuned_map[tuned_key]

            # ensure tuned has num_warps/num_stages keys (fields the harness expects)
            for k in ("num_warps", "num_stages"):
                if k not in tuned:
                    tuned[k] = 4 if k == "num_warps" else 3

            default = get_default_config(bs, E)

            # benchmark_one already does 3 internal warmups + 10-iter average;
            # we additionally run it 3 times and take median for stability.
            try:
                t_def = [benchmark_one(default, bs, E, N, K, topk) for _ in range(3)]
                t_default = median_of(t_def)

                t_tun = [benchmark_one(tuned, bs, E, N, K, topk) for _ in range(3)]
                t_tuned = median_of(t_tun)

                if t_default == float("inf") or t_tuned == float("inf"):
                    print(f"| ({E}, {N}, {K}, {topk}) | {bs} | OOM/fail | OOM/fail | — |")
                    continue
                speedup = t_default / t_tuned
                print(
                    f"| ({E}, {N}, {K}, {topk}) | {bs} | {t_default:.3f} | "
                    f"{t_tuned:.3f} | **{speedup:.2f}x** |"
                )
                summary.append((label, bs, speedup))
            except Exception as e:
                print(f"| ({E}, {N}, {K}, {topk}) | {bs} | error: {type(e).__name__} | — | — |")

            torch.cuda.empty_cache()

    if summary:
        print()
        print(f"# {len(summary)} data points; geomean speedup:")
        geo = math.exp(sum(math.log(s) for _, _, s in summary) / len(summary))
        print(f"# geomean = {geo:.2f}x")


if __name__ == "__main__":
    main()
