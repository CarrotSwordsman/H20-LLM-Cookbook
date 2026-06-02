#!/usr/bin/env python3
"""
MoE kernel tuning for NVIDIA H20 with fp8_w8a8 (per-tensor quantization).

Generates configs compatible with vllm's `fused_moe/configs/` directory format,
specifically `E={E},N={N},device_name=NVIDIA_H20,dtype=fp8_w8a8.json`
(no block_shape; matches the existing E=64,N={384,1536,3072} family on main).

Usage:
    # Single config:
    python tune_moe_h20_fp8.py --E 8 --N 14336 --K 4096 --top-k 2

    # All shapes that the bf16 PR #44152 added (the natural fp8 follow-up):
    python tune_moe_h20_fp8.py --all-missing
"""

import argparse
import json
import os
import time
from itertools import product
from typing import Any, Dict, List, Tuple

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel,
    moe_align_block_size,
)
from vllm import _custom_ops as ops

# fp8 dtype is e4m3fn on Hopper-class GPUs (H20).
FP8_DTYPE = torch.float8_e4m3fn

# Same batch-size grid as upstream benchmark_moe.py and our bf16 tuner.
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
               1024, 1536, 2048, 3072, 4096]

# Triton search space. Note: fp8 path forbids BLOCK_SIZE_K=16 (see
# benchmarks/kernels/benchmark_moe.py upstream; our space already starts at 64).
SEARCH_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [64, 128, 256],
    "GROUP_SIZE_M": [1, 16, 32, 64],
    "num_warps": [4, 8],
    "num_stages": [3, 4, 5],
}


def get_all_configs() -> List[Dict[str, int]]:
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def benchmark_one(config, num_tokens, E, N, K, top_k,
                  device="cuda") -> float:
    """Benchmark one config in fp8_w8a8 (per-tensor) mode.

    Pre-quantizes A and B once before the per-config loop, then directly
    invokes `fused_moe_kernel` (mirroring the bf16 tuner). This avoids the
    per-config `ops.scaled_fp8_quant` cost that caused 1000x slowdown when
    routing through `invoke_fused_moe_kernel`.
    """
    # Inputs (kept stable across this benchmark call):
    a_bf16 = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device)
    a_fp8, a_scale = ops.scaled_fp8_quant(a_bf16, None)  # dynamic per-tensor
    b_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)
    b_fp8 = b_bf16.to(FP8_DTYPE)
    b_scale = (torch.randn(E, dtype=torch.float32, device=device).abs() + 1e-3)

    score = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(score, dim=-1), top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    try:
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E)
        c = torch.zeros(num_tokens * top_k, N,
                        dtype=torch.bfloat16, device=device)

        grid = lambda META: (
            triton.cdiv(sorted_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        # Same 27-positional layout as the bf16 tuner; only A_scale/B_scale
        # are non-None and use_fp8_w8a8=True. All scale strides and
        # group_n/group_k are zeros (per-tensor quant, no block-quant).
        args = (
            a_fp8, b_fp8, c, a_scale, b_scale,
            topk_weights, sorted_ids, expert_ids, ntp,
            N, K, sorted_ids.shape[0], topk_ids.numel(),
            a_fp8.stride(0), a_fp8.stride(1),
            b_fp8.stride(0), b_fp8.stride(2), b_fp8.stride(1),
            c.stride(0), c.stride(1),
            0, 0,           # A_scale stride (per-tensor scalar)
            0, 0, 0,        # B_scale stride (per-expert 1D)
            0, 0,           # group_n, group_k (no block-quant)
        )
        kwargs = dict(
            MUL_ROUTED_WEIGHT=True,
            top_k=top_k,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            **config,
        )

        # Warmup (3) + bench (10), matching the bf16 tuner.
        for _ in range(3):
            fused_moe_kernel[grid](*args, **kwargs)
        torch.cuda.synchronize()
        n_iter = 10
        start = time.perf_counter()
        for _ in range(n_iter):
            fused_moe_kernel[grid](*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / n_iter * 1000
    except Exception:
        return float("inf")


def tune_batch_size(num_tokens, E, N, K, top_k, configs):
    best_t, best_cfg = float("inf"), None
    for i, cfg in enumerate(configs):
        t = benchmark_one(cfg, num_tokens, E, N, K, top_k)
        if t < best_t:
            best_t, best_cfg = t, cfg
        if (i + 1) % 200 == 0:
            print(f"      [{i+1}/{len(configs)}] best={best_t:.4f}ms")
    return best_t, best_cfg


def get_output_filename(E, N) -> str:
    return f"E={E},N={N},device_name=NVIDIA_H20,dtype=fp8_w8a8.json"


# Shapes covered by the bf16 PR #44152 — natural fp8 follow-up set.
DEFAULT_SHAPES = [
    # (E, N, K, topk, label)
    (8,    1792, 4096, 2, "Mixtral 8x7B (1792)"),
    (8,    2048, 4096, 2, "Mixtral 8x7B (2048)"),
    (8,    3584, 4096, 2, "Mixtral 8x7B (3584)"),
    (8,    4096, 4096, 2, "Mixtral 8x7B (4096)"),
    (8,    7168, 4096, 2, "Mixtral 8x7B (7168)"),
    (8,   14336, 4096, 2, "Mixtral 8x7B (14336)"),
    (64,    320, 2048, 6, "Qwen MoE (320)"),
    (64,    640, 2048, 6, "Qwen MoE (640)"),
    (64,   1280, 2048, 6, "Qwen MoE (1280)"),
    (64,   2560, 2048, 6, "Qwen MoE (2560)"),
    (128,   512, 2048, 6, "DeepSeek-V2-Lite (512)"),
    (128,  1024, 2048, 6, "DeepSeek-V2-Lite (1024)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--E", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)
    p.add_argument("--top-k", type=int)
    p.add_argument("--all-missing", action="store_true",
                   help="Tune all default fp8 shapes that don't already have a config file.")
    p.add_argument("--out-dir", default=str(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "vllm", "model_executor", "layers", "fused_moe", "configs")))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = get_all_configs()
    print(f"Search space: {len(configs)} configs x {len(BATCH_SIZES)} batches")
    print(f"Output dir:   {args.out_dir}\n")

    if args.all_missing:
        shapes = []
        for E, N, K, topk, label in DEFAULT_SHAPES:
            out = os.path.join(args.out_dir, get_output_filename(E, N))
            if os.path.exists(out):
                print(f"[skip] {get_output_filename(E, N)} already exists")
                continue
            shapes.append((E, N, K, topk, label))
        if not shapes:
            print("All targets already present. Nothing to do.")
            return
    else:
        if not all([args.E, args.N, args.K, args.top_k]):
            p.error("Need --E --N --K --top-k (or use --all-missing)")
        shapes = [(args.E, args.N, args.K, args.top_k,
                   f"E={args.E},N={args.N}")]

    for E, N, K, topk, label in shapes:
        print(f"\n=== Tuning {label}: E={E}, N={N}, K={K}, top_k={topk} ===")
        t0 = time.time()
        result = {}
        for bs in BATCH_SIZES:
            print(f"  batch={bs:>5}", end=" ", flush=True)
            t1 = time.time()
            best_t, best_cfg = tune_batch_size(bs, E, N, K, topk, configs)
            if best_cfg is None:
                print("(all OOM/fail)")
                continue
            result[str(bs)] = {k: best_cfg[k] for k in
                               ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                                "GROUP_SIZE_M", "num_warps", "num_stages"]}
            print(f": {best_t:.4f}ms (searched in {time.time()-t1:.1f}s) "
                  f"-> {result[str(bs)]}")
        out_path = os.path.join(args.out_dir, get_output_filename(E, N))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
        print(f"  Saved: {out_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
