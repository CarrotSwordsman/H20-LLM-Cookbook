#!/usr/bin/env python3
"""
MoE kernel tuning for NVIDIA H20.
Generates configs compatible with vllm main branch format.

Usage:
    # Single config:
    python tune_moe_h20.py --E 8 --N 14336 --K 4096 --top-k 2

    # All missing H20 configs:
    python tune_moe_h20.py --all-missing
"""

import argparse
import json
import os
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel,
    moe_align_block_size,
)
from vllm.platforms import current_platform

# Batch sizes to tune (matches vllm official)
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
               1024, 1536, 2048, 3072, 4096]

# Search space (matches vllm official benchmark_moe.py)
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
    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def benchmark_one(config, num_tokens, E, N, K, top_k, dtype, device="cuda"):
    """Benchmark one config. Returns time in ms or inf on failure."""
    a = torch.randn(num_tokens, K, dtype=dtype, device=device)
    b = torch.randn(E, N, K, dtype=dtype, device=device)
    score = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(score, dim=-1), top_k, dim=-1)

    try:
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E)
        c = torch.zeros(num_tokens * top_k, N, dtype=dtype, device=device)

        grid = lambda META: (
            triton.cdiv(sorted_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        args = (
            a, b, c, None, None,
            topk_weights, sorted_ids, expert_ids, ntp,
            N, K, sorted_ids.shape[0], topk_ids.numel(),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(2), b.stride(1),
            c.stride(0), c.stride(1),
            0, 0, 0, 0, 0, 0, 0,
        )
        kwargs = dict(
            MUL_ROUTED_WEIGHT=True,
            top_k=top_k,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            **config,
        )

        # Warmup (3 iterations)
        for _ in range(3):
            fused_moe_kernel[grid](*args, **kwargs)
        torch.cuda.synchronize()

        # Benchmark (10 iterations)
        n_iter = 10
        start = time.perf_counter()
        for _ in range(n_iter):
            fused_moe_kernel[grid](*args, **kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / n_iter * 1000

        return elapsed_ms
    except Exception:
        return float("inf")


def tune_batch_size(num_tokens, E, N, K, top_k, dtype, configs):
    """Find best config for given batch size."""
    best_time = float("inf")
    best_config = None

    for i, cfg in enumerate(configs):
        t = benchmark_one(cfg, num_tokens, E, N, K, top_k, dtype)
        if t < best_time:
            best_time = t
            best_config = cfg.copy()

        # Progress every 100 configs
        if (i + 1) % 200 == 0:
            print(f"      [{i+1}/{len(configs)}] best={best_time:.4f}ms")

    return best_config, best_time


def get_output_filename(E, N, dtype_str=None):
    device_name = current_platform.get_device_name().replace(" ", "_")
    dtype_part = "" if not dtype_str else f",dtype={dtype_str}"
    return f"E={E},N={N},device_name={device_name}{dtype_part}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--E", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype-str", type=str, default=None,
                        help="e.g. fp8_w8a8 (None=bf16)")
    parser.add_argument("--save-dir", type=str,
                        default="/apdcephfs_hzlf/share_1227201/mershi/vllm-pr/"
                                "vllm/model_executor/layers/fused_moe/configs/")
    parser.add_argument("--all-missing", action="store_true")
    args = parser.parse_args()

    if args.all_missing:
        # Key configs H200 has but H20 doesn't
        targets = [
            # Mixtral-8x7B
            (8, 14336, 4096, 2, None),
            (8, 7168, 4096, 2, None),
            (8, 3584, 4096, 2, None),
            (8, 1792, 4096, 2, None),
            (8, 4096, 4096, 2, None),
            (8, 2048, 4096, 2, None),
            # E=128 larger N
            (128, 1024, 4096, 2, None),
            (128, 512, 4096, 2, None),
            # E=64 missing
            (64, 1280, 4096, 2, None),
            (64, 2560, 4096, 2, None),
            (64, 320, 4096, 2, None),
            (64, 640, 4096, 2, None),
        ]
    else:
        if not args.E or not args.N:
            parser.error("Need --E and --N, or --all-missing")
        targets = [(args.E, args.N, args.K, args.top_k, args.dtype_str)]

    os.makedirs(args.save_dir, exist_ok=True)
    configs = get_all_configs()
    print(f"Search space: {len(configs)} configs per batch size")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Device: {current_platform.get_device_name()}")
    print()

    for E, N, K, top_k, dtype_str in targets:
        filename = get_output_filename(E, N, dtype_str)
        filepath = os.path.join(args.save_dir, filename)

        if os.path.exists(filepath):
            print(f"SKIP (exists): {filename}")
            continue

        print(f"{'='*60}")
        print(f"Tuning E={E}, N={N}, K={K}, top_k={top_k}")
        print(f"  -> {filename}")
        print(f"{'='*60}")

        results = {}
        total_start = time.time()

        for bs in BATCH_SIZES:
            torch.cuda.empty_cache()
            t0 = time.time()
            best_cfg, best_time = tune_batch_size(
                bs, E, N, K, top_k, torch.bfloat16, configs)
            elapsed = time.time() - t0
            results[str(bs)] = best_cfg
            print(f"  batch={bs:>5}: {best_time:.4f}ms "
                  f"(searched in {elapsed:.1f}s) -> {best_cfg}")

        total_elapsed = time.time() - total_start
        print(f"\n  Total time: {total_elapsed/60:.1f} min")

        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Saved: {filepath}\n")

    print("\nAll done!")


if __name__ == "__main__":
    main()
