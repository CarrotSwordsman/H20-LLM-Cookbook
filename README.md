# H20 LLM Inference Cookbook

> A reproducible benchmark suite and tuning recipe collection for **NVIDIA H20**
> LLM inference. Currently focused on the Triton **fused MoE kernel** path used
> by [vLLM](https://github.com/vllm-project/vllm) and SGLang.

[![configs](https://img.shields.io/badge/tuned_configs-24-blue)](./configs)
[![upstream PRs](https://img.shields.io/badge/upstream_PRs-vllm%2344152%20%2B%20vllm%2344273-orange)](https://github.com/vllm-project/vllm/pulls?q=author%3ACarrotSwordsman+H20)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## TL;DR

- **24 tuned Triton fused-MoE configs** for H20: 12 `bf16` + 12 `fp8_w8a8` (per-tensor).
- Covers **Mixtral 8×7B** (E=8), **Qwen MoE** (E=64), and **DeepSeek-V2-Lite** (E=128) shapes that previously had no H20 entries upstream.
- **Geomean speedup `1.09×` (bf16) / `1.39×` (fp8_w8a8); peak `1.74×` (bf16) / `2.74×` (fp8_w8a8)** vs vLLM's default-fallback heuristic across 36 (shape × batch) cells, kernel-level.
- Configs upstream as **[vllm#44152](https://github.com/vllm-project/vllm/pull/44152) (bf16)** and **[vllm#44273](https://github.com/vllm-project/vllm/pull/44273) (fp8_w8a8)**.
- Standalone tuning scripts work on **CUDA 12.2 + torch 2.5.1 + Triton 3.1**, no full vLLM build required.

## Why H20 specifically?

H20 is the Hopper-class card most readily available in mainland China, but it has unusual ratios that make off-the-shelf configs from H100 / H200 a poor fit:

| Spec                          | H100 (SXM5) | H200 (SXM5) | **H20** (SXM5)   |
|-------------------------------|------------:|------------:|-----------------:|
| HBM capacity                  |     80 GB   |     141 GB  |       **96 GB**  |
| HBM bandwidth                 |   3.35 TB/s |    4.8 TB/s |     **4.0 TB/s** |
| BF16 dense TFLOPS (no sparsity) |    989    |       989   |       **148**    |
| FP8 dense TFLOPS (no sparsity)  |   1979    |      1979   |       **296**    |
| SM count                      |        132  |        132  |          **78**  |
| L2 cache                      |     50 MB   |       60 MB |        **60 MB** |

Net effect: H20 is **bandwidth-rich, compute-poor** relative to H100. The arithmetic-intensity sweet spot of the Triton fused-MoE kernel ends up at **different `BLOCK_SIZE_*` and `num_stages`** than what's tuned for H100/H200 — directly using H100/H200 JSONs leaves measurable performance on the table.

For more on the reasoning and tuning methodology, see [`reports/2026-06_h20_moe_tuning.md`](./reports/2026-06_h20_moe_tuning.md).

## What's inside

```
.
├── configs/
│   ├── bf16/                    # 12 H20 bf16 configs (mirrors vllm#44152)
│   └── fp8_w8a8/                # 12 H20 fp8_w8a8 configs (mirrors vllm#44273)
├── benchmarks/
│   ├── moe_kernel/
│   │   ├── tune_moe_h20.py                 # bf16 standalone tuner
│   │   ├── tune_moe_h20_fp8.py             # fp8_w8a8 (per-tensor) tuner
│   │   ├── compare_default_vs_tuned.py     # bf16 default-fallback vs tuned harness
│   │   └── compare_default_vs_tuned_fp8.py # fp8_w8a8 default-fallback vs tuned harness
│   └── results/
│       ├── bf16_default_vs_tuned.md        # 36-point bf16 perf table (geomean 1.09×)
│       └── fp8_w8a8_default_vs_tuned.md    # 36-point fp8 perf table (geomean 1.39×)
├── reports/
│   ├── 2026-06_h20_moe_tuning.md           # methodology + analysis writeup (English)
│   └── blog_h20_moe_zh.md                  # Chinese blog post
└── docs/
    └── h20_vs_h100_vs_h200.md              # spec & arch comparison
```

## Coverage matrix

| Family                | E   | N values                              | bf16 | fp8_w8a8 |
|-----------------------|----:|---------------------------------------|:----:|:--------:|
| Mixtral 8×7B          |   8 | 1792, 2048, 3584, 4096, 7168, 14336   | ✅   |  ✅      |
| Qwen MoE              |  64 | 320, 640, 1280, 2560                  | ✅   |  ✅      |
| DeepSeek-V2-Lite      | 128 | 512, 1024                             | ✅   |  ✅      |

All 24 configs were tuned over the full upstream search space (1152 candidates × 18 batch sizes per config) on a single H20 (96 GB HBM3).

## Headline performance numbers

Kernel-level, `fused_moe_kernel` direct call, 3 warmups + 10-iter mean × 3 medians.

### bf16

| Shape (E, N, K, topk)      | Best speedup | Where                  |
|----------------------------|-------------:|------------------------|
| (8, 7168, 4096, 2)         |   **1.74×**  | batch = 32             |
| (8, 14336, 4096, 2)        |   **1.60×**  | batch = 32             |
| (64, 1280, 2048, 6)        |   **1.51×**  | batch = 128            |
| (64, 2560, 2048, 6)        |   **1.47×**  | batch = 128            |
| (128, 1024, 2048, 6)       |   **1.27×**  | batch = 128            |

Full 36-point table: [`benchmarks/results/bf16_default_vs_tuned.md`](./benchmarks/results/bf16_default_vs_tuned.md).
**Geomean across all 36 cells: 1.09×.**

### fp8_w8a8 (per-tensor)

| Shape (E, N, K, topk)      | Best speedup | Where                  |
|----------------------------|-------------:|------------------------|
| (8, 7168, 4096, 2)         |   **2.74×**  | batch = 1              |
| (128, 1024, 2048, 6)       |   **2.59×**  | batch = 32             |
| (8, 14336, 4096, 2)        |   **2.48×**  | batch = 1              |
| (64, 2560, 2048, 6)        |   **2.48×**  | batch = 32             |
| (128, 512, 2048, 6)        |   **2.38×**  | batch = 128            |

Full 36-point table: [`benchmarks/results/fp8_w8a8_default_vs_tuned.md`](./benchmarks/results/fp8_w8a8_default_vs_tuned.md).
**Geomean across all 36 cells: 1.39×, with zero regressions (all ≥ 1.00×).**

> Speedups concentrate in **small-to-medium batch** regimes (1–512), where the
> default 2-branch heuristic in `vllm.model_executor.layers.fused_moe.fused_moe.get_default_config`
> is most under-tuned. Under fp8 the wins are larger and extend to large
> batches as well (≥ 2048 still 1.08–1.17×) — fp8 halves HBM traffic, so the
> kernel hasn't yet saturated H20's 4 TB/s HBM3 ceiling.

## Quickstart: reproduce on your H20

```bash
# 1. Activate any env with: torch ≥ 2.4, triton ≥ 3.0, vllm installed (any
#    version that exposes `fused_moe_kernel` + `moe_align_block_size`)
conda activate <env>

# 2. Run the bf16 tuner over all 12 shapes (~80 minutes total)
cd benchmarks/moe_kernel
python -u tune_moe_h20.py --all-missing --save-dir ../../configs/bf16

# 3. (Optional) Run fp8_w8a8 tuner (~40 minutes total)
python -u tune_moe_h20_fp8.py --all-missing --out-dir ../../configs/fp8_w8a8

# 4. Reproduce the perf table
python compare_default_vs_tuned.py
```

## Using these configs in vLLM

Drop the JSONs from [`configs/bf16/`](./configs/bf16) and
[`configs/fp8_w8a8/`](./configs/fp8_w8a8) into your local
`vllm/model_executor/layers/fused_moe/configs/` directory. vLLM picks them up
automatically via `try_get_optimal_moe_config()` based on your device's
`current_platform.get_device_name()`.

If you're on a recent vLLM main, both sets are upstream-pending in PRs
**#44152** and **#44273** — once merged, no manual copy is needed.

## Roadmap

- [x] H20 bf16 fused-MoE configs (12 shapes)
- [x] H20 fp8_w8a8 (per-tensor) fused-MoE configs (12 shapes)
- [x] bf16 default-vs-tuned 36-point benchmark
- [x] fp8_w8a8 default-vs-tuned 36-point benchmark
- [ ] H20 fp8_w8a8 + `block_shape=[128,128]` configs (DeepSeek-V3 family)
- [ ] vLLM serve end-to-end TTFT / ITL numbers on H20 (Mixtral, Qwen2-MoE)
- [ ] SGLang head-to-head on the same shapes

Issues / PRs welcome.

## Citation

If this work helps your research or production deployment, a star or a citation is appreciated:

```bibtex
@misc{h20_llm_cookbook_2026,
  title  = {H20 LLM Inference Cookbook: Tuned Triton Fused-MoE Configs and
            Benchmarks for NVIDIA H20},
  author = {shiyichuan},
  year   = {2026},
  url    = {https://github.com/CarrotSwordsman/H20-LLM-Cookbook}
}
```

## License

[MIT](./LICENSE).
