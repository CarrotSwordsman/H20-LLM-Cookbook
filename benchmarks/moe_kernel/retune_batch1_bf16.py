#!/usr/bin/env python3
"""
Focused retune of batch=1 bf16 configs for the 3 H20 (E, N) shapes that
slightly regress vs vLLM's default-fallback in `bf16_default_vs_tuned.md`:

  - (E=64,  N=1280): 0.92x
  - (E=128, N=1024): 0.91x
  - (E=128, N=512):  0.94x

Root cause: the original `tune_moe_h20.py` evaluates each candidate config
with a *single* 10-iter mean (no repeat / median), which is noisy at batch=1
where kernel time is tens of microseconds. The compare harness uses
3 warmup + 10-iter mean x 3 trials, median -- a stricter measurement.

This script:
  1. Coarse-sweeps all 1152 candidates with single-trial 10-iter mean
     (matches the original tuner cost).
  2. Re-evaluates the top-20 with 5-trial median measurement (twice as
     stable as the compare harness's 3-trial median).
  3. Reports the best config + speedup vs default.
  4. Optionally patches the corresponding `configs/bf16/E=*,N=*,...json`
     batch="1" entry in place (`--write`).

Usage:
    python retune_batch1_bf16.py             # dry run (just print)
    python retune_batch1_bf16.py --write     # patch JSONs in place
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune_moe_h20 import benchmark_one, SEARCH_SPACE  # noqa: E402

# 3 H20 shapes that regressed at batch=1 (see bf16_default_vs_tuned.md).
SHAPES = [
    # (E, N, K, topk)
    (64,  1280, 2048, 6),
    (128, 1024, 2048, 6),
    (128,  512, 2048, 6),
]

# vLLM's get_default_config() M<=E branch (batch=1 hits this on every shape
# above since E in {64, 128}).
DEFAULT_M_LE_E = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,  "num_warps": 4, "num_stages": 3,
}

CONFIG_DIR = HERE.parent.parent / "configs" / "bf16"


def all_configs():
    keys = list(SEARCH_SPACE.keys())
    return [dict(zip(keys, combo)) for combo in product(*SEARCH_SPACE.values())]


def stable_median(cfg, M, E, N, K, topk, n_trials):
    samples = [
        benchmark_one(cfg, M, E, N, K, topk, dtype=torch.bfloat16)
        for _ in range(n_trials)
    ]
    samples = [s for s in samples if s != float("inf")]
    if not samples:
        return float("inf")
    return sorted(samples)[len(samples) // 2]


def find_best_batch1(E, N, K, topk):
    print(f"\n=== (E={E}, N={N}, K={K}, topk={topk}) batch=1 ===")

    # 1) Coarse sweep: all 1152 candidates, single-trial 10-iter mean.
    configs = all_configs()
    print(f"[coarse]   sweeping {len(configs)} candidates...")
    coarse = []
    for i, cfg in enumerate(configs):
        t = benchmark_one(cfg, 1, E, N, K, topk, dtype=torch.bfloat16)
        coarse.append((t, cfg))
        if (i + 1) % 200 == 0:
            best = min(coarse, key=lambda x: x[0])[0]
            print(f"           [{i+1}/{len(configs)}] best so far={best*1000:.2f}us")
    coarse.sort(key=lambda x: x[0])
    top_k = 20
    candidates = [c for _, c in coarse[:top_k]]
    print(f"[coarse]   top-1 candidate raw time = {coarse[0][0]*1000:.2f}us")

    # 2) Stable refinement on top-20 + the default config (so the comparison
    #    is apples-to-apples under the same measurement methodology).
    all_to_eval = candidates + [DEFAULT_M_LE_E]
    stable = []
    for cfg in all_to_eval:
        t = stable_median(cfg, 1, E, N, K, topk, n_trials=5)
        stable.append((t, cfg))

    t_default = next(t for t, c in stable if c == DEFAULT_M_LE_E)
    stable_candidates = [(t, c) for t, c in stable if c != DEFAULT_M_LE_E]
    stable_candidates.sort(key=lambda x: x[0])
    t_best, cfg_best = stable_candidates[0]

    speedup = t_default / t_best if t_best > 0 else 0
    print(f"[refined]  default        = {t_default*1000:.3f} us  cfg={DEFAULT_M_LE_E}")
    print(f"[refined]  best-of-tuned  = {t_best*1000:.3f} us  speedup vs default = {speedup:.2f}x")
    print(f"[refined]  new cfg        = {cfg_best}")

    # Show old cfg from JSON for diff.
    old_path = CONFIG_DIR / f"E={E},N={N},device_name=NVIDIA_H20.json"
    if old_path.exists():
        old = json.load(open(old_path))["1"]
        old_t = stable_median(old, 1, E, N, K, topk, n_trials=5)
        print(f"[refined]  old cfg        = {old_t*1000:.3f} us  cfg={old}")

    return cfg_best, t_best, t_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="patch JSONs in place")
    args = parser.parse_args()

    new_entries = {}
    for E, N, K, topk in SHAPES:
        cfg, t_best, t_default = find_best_batch1(E, N, K, topk)
        new_entries[(E, N)] = cfg

    print("\n\n=== SUMMARY ===")
    for (E, N), cfg in new_entries.items():
        print(f"(E={E:>3}, N={N:>5}) -> {cfg}")

    if args.write:
        for (E, N), cfg in new_entries.items():
            path = CONFIG_DIR / f"E={E},N={N},device_name=NVIDIA_H20.json"
            data = json.load(open(path))
            old = data.get("1")
            data["1"] = cfg
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
                f.write("\n")
            print(f"  wrote {path.name}: batch=1  {old}  ->  {cfg}")
    else:
        print("\n(dry run; pass --write to patch JSONs in place)")


if __name__ == "__main__":
    main()
