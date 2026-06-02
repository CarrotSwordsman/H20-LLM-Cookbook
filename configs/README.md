# Tuned configs for NVIDIA H20

Drop these JSONs into vLLM's
`vllm/model_executor/layers/fused_moe/configs/` directory and they'll be picked
up automatically by `try_get_optimal_moe_config()` based on
`current_platform.get_device_name()`.

## Layout

| Path                               | dtype       | quant scheme                | Count |
|------------------------------------|-------------|-----------------------------|------:|
| [`bf16/`](./bf16)                  | `bfloat16`  | none                        |   12  |
| [`fp8_w8a8/`](./fp8_w8a8)          | `fp8_e4m3`  | per-tensor (act + weight)   |   12  |

## Coverage (E, N) matrix

```
E\N    320 | 512 | 640 | 1024 | 1280 | 1792 | 2048 | 2560 | 3584 | 4096 | 7168 | 14336
  8                                    bf16   bf16   bf16          bf16   bf16   bf16
                                       fp8    fp8    fp8           fp8    fp8    fp8
 64    bf16        bf16         bf16                       bf16
       fp8         fp8          fp8                        fp8
128          bf16         bf16
             fp8          fp8
```

`K` and `top_k` values used for tuning:

| E    | K    | top_k | Reference model family |
|------|------|------:|------------------------|
|   8  | 4096 |     2 | Mixtral 8×7B           |
|  64  | 2048 |     6 | Qwen MoE               |
| 128  | 2048 |     6 | DeepSeek-V2-Lite       |

## Tuning methodology

All configs were tuned with:

- **Search space** (1152 combinations):
  - `BLOCK_SIZE_M ∈ {16, 32, 64, 128}`
  - `BLOCK_SIZE_N ∈ {32, 64, 128, 256}`
  - `BLOCK_SIZE_K ∈ {64, 128, 256}`
  - `GROUP_SIZE_M ∈ {1, 16, 32, 64}`
  - `num_warps ∈ {4, 8}`
  - `num_stages ∈ {3, 4, 5}`
- **Batch size grid**: `[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]`
- **Per-config metric**: 3 warmup iterations + 10-iter mean wall time, GPU-synced via `torch.cuda.synchronize()`. Best config per batch wins.

This matches the search space in `vllm/benchmarks/kernels/benchmark_moe.py`.

## Hardware

- NVIDIA H20 (96 GB HBM3, SXM5)
- CUDA 12.2 driver / 12.1 runtime
- torch 2.5.1+cu121
- Triton 3.1.0

## Upstream status

- bf16 — [vllm#44152](https://github.com/vllm-project/vllm/pull/44152) (open)
- fp8_w8a8 — [vllm#44273](https://github.com/vllm-project/vllm/pull/44273) (open)
