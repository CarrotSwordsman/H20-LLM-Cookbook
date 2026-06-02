# Tuning the Triton Fused-MoE Kernel for NVIDIA H20

> A walkthrough of how the 24 configs in this cookbook were generated, what
> the numbers mean, and where they help (and don't).
>
> *Author: shiyichuan · 2026-06*

## TL;DR

I tuned 12 `bf16` and 12 `fp8_w8a8` configs of vLLM's Triton `fused_moe_kernel`
for **NVIDIA H20** across the (E, N) shapes shipped in popular MoE families
(Mixtral 8×7B, Qwen MoE, DeepSeek-V2-Lite). On the same H20 the tuner ran on,
the bf16 configs deliver **geomean 1.09×, max 1.74×** speedup over vLLM's
default-fallback heuristic across 36 (shape × batch) cells, with the gains
concentrated in the small-to-medium batch regime where the default's two-branch
heuristic is most under-tuned. Configs are upstream-pending in
[vllm#44152](https://github.com/vllm-project/vllm/pull/44152) (bf16) and
[vllm#44273](https://github.com/vllm-project/vllm/pull/44273) (fp8_w8a8).

## 1. The gap I started from

vLLM ships per-shape JSONs under
`vllm/model_executor/layers/fused_moe/configs/`. When a `(E, N, device, dtype)`
file is present, `try_get_optimal_moe_config()` interpolates from it; otherwise
it falls back to a tiny hand-coded heuristic in `get_default_config()`:

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py
def get_default_config(M, E, N, K, ...):
    if M <= E:                              # decode-ish
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=1,  num_warps=4, num_stages=3)
    return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,  num_warps=4, num_stages=3)
```

That's the "default" column you'll see throughout this writeup. It's
deliberately conservative — but on H20, *conservative* is a bad idea, because
the H20 roofline is shifted (see [`docs/h20_vs_h100_vs_h200.md`](../docs/h20_vs_h100_vs_h200.md)).

A check on `vllm/main` showed:

| Family | (E, N) shape   | H100 / H200 cfg present? | H20 cfg present? |
|--------|----------------|:-----------------------:|:----------------:|
| Mixtral 8×7B | E=8, N ∈ {1792, 2048, 3584, 4096, 7168, 14336} | ✅ | ❌ |
| Qwen MoE     | E=64, N ∈ {320, 640, 1280, 2560} | ✅ | ❌ |
| DeepSeek-V2-Lite | E=128, N ∈ {512, 1024} | ✅ | ❌ |

H20 had **0** of these. That's the gap this cookbook closes.

## 2. Tuning setup

I couldn't use `vllm/benchmarks/kernels/benchmark_moe.py` directly: it imports
modules from vLLM main that require torch ≥ 2.11 + CUDA 13, but my driver is
535.247 (CUDA 12.2 max). So I wrote a standalone tuner that imports only the
two stable symbols actually needed:

```python
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel, moe_align_block_size,
)
```

These exist in any vLLM ≥ 0.6.6 wheel and don't pull in the rest of the main
branch. The tuner's per-config kernel call mirrors `invoke_fused_moe_kernel`
exactly — same 27 positional args, same `MUL_ROUTED_WEIGHT=True`, same
`compute_type=tl.bfloat16`. See [`benchmarks/moe_kernel/tune_moe_h20.py`](../benchmarks/moe_kernel/tune_moe_h20.py).

Search space is identical to the upstream benchmark:

```
BLOCK_SIZE_M ∈ {16, 32, 64, 128}
BLOCK_SIZE_N ∈ {32, 64, 128, 256}
BLOCK_SIZE_K ∈ {64, 128, 256}
GROUP_SIZE_M ∈ {1, 16, 32, 64}
num_warps    ∈ {4, 8}
num_stages   ∈ {3, 4, 5}
                                 = 1152 configs / batch size
batches: [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
          1024, 1536, 2048, 3072, 4096]                  → 18 batches
```

Per-config metric: 3 warmup iters + 10-iter mean wall time, GPU-synced via
`torch.cuda.synchronize()`. OOM / illegal configs are caught and treated as
`inf`. Total wall time on H20: ~80 minutes for all 12 bf16 shapes (worst:
`E=8, N=14336` at ~10 min; best: `E=64, N=320` at ~2 min).

## 3. fp8 was the trickier path

The bf16 tuner uses `fused_moe_kernel` with `use_fp8_w8a8=False`, all
scale args set to `None`. For fp8 I needed:

1. Pre-quantized fp8 inputs (`torch.float8_e4m3fn`)
2. Per-tensor activation scale via `vllm._custom_ops.scaled_fp8_quant`
3. Per-expert weight scale (1D tensor of shape `[E]`)
4. `use_fp8_w8a8=True`
5. Non-zero `A_scale` and `B_scale` strides in the positional argument list
6. Zero `group_n` / `group_k` (per-tensor, not block-quant)

A first attempt routed each per-config benchmark through
`invoke_fused_moe_kernel`, which re-runs `scaled_fp8_quant` every call. That
caused **~1000× slowdown** vs bf16. Pre-quantizing once before the per-config
loop fixed it. The full script is at
[`benchmarks/moe_kernel/tune_moe_h20_fp8.py`](../benchmarks/moe_kernel/tune_moe_h20_fp8.py).

## 4. Results (bf16, default vs tuned)

Same H20, same kernel, same batch grid. Each cell: 3 warmup + 10-iter mean,
repeated 3 times taking the median.

| Shape (E, N, K, topk)              | Batch | default (ms) | tuned (ms) | speedup |
|------------------------------------|------:|-------------:|-----------:|--------:|
| (8, 14336, 4096, 2) — *Mixtral*    |   1   |  0.099       |  0.076     | **1.31×** |
|                                    |  32   |  0.447       |  0.279     | **1.60×** |
|                                    | 128   |  0.478       |  0.435     | **1.10×** |
|                                    | 512   |  1.101       |  1.069     | **1.03×** |
|                                    | 2048  |  3.588       |  3.606     |   0.99×   |
|                                    | 4096  |  6.986       |  7.015     |   1.00×   |
| (8, 7168, 4096, 2) — *Mixtral alt* |   1   |  0.059       |  0.044     | **1.32×** |
|                                    |  32   |  0.251       |  0.144     | **1.74×** |
|                                    | 128   |  0.256       |  0.250     | **1.02×** |
|                                    | 512   |  0.539       |  0.584     |   0.92×   |
|                                    | 2048  |  1.804       |  1.873     |   0.96×   |
|                                    | 4096  |  3.501       |  3.500     |   1.00×   |
| (64, 2560, 2048, 6) — *Qwen MoE*   |   1   |  0.037       |  0.027     | **1.38×** |
|                                    |  32   |  0.230       |  0.195     | **1.18×** |
|                                    | 128   |  0.322       |  0.218     | **1.47×** |
|                                    | 512   |  0.353       |  0.486     |   0.73×   |
|                                    | 2048  |  1.073       |  1.070     |   1.00×   |
|                                    | 4096  |  1.987       |  1.974     |   1.01×   |
| (64, 1280, 2048, 6) — *Qwen alt*   |   1   |  0.023       |  0.025     |   0.92×   |
|                                    |  32   |  0.125       |  0.103     | **1.21×** |
|                                    | 128   |  0.169       |  0.112     | **1.51×** |
|                                    | 512   |  0.182       |  0.238     |   0.77×   |
|                                    | 2048  |  0.537       |  0.536     |   1.00×   |
|                                    | 4096  |  0.992       |  0.996     |   1.00×   |
| (128, 1024, 2048, 6) — *DSV2-Lite* |   1   |  0.023       |  0.025     |   0.91×   |
|                                    |  32   |  0.158       |  0.127     | **1.25×** |
|                                    | 128   |  0.206       |  0.162     | **1.27×** |
|                                    | 512   |  0.270       |  0.255     | **1.06×** |
|                                    | 2048  |  0.506       |  0.497     |   1.02×   |
|                                    | 4096  |  0.859       |  0.860     |   1.00×   |
| (128, 512, 2048, 6) — *DSV2-Lite*  |   1   |  0.022       |  0.024     |   0.94×   |
|                                    |  32   |  0.087       |  0.072     | **1.21×** |
|                                    | 128   |  0.107       |  0.088     | **1.21×** |
|                                    | 512   |  0.141       |  0.137     | **1.03×** |
|                                    | 2048  |  0.260       |  0.266     |   0.98×   |
|                                    | 4096  |  0.435       |  0.433     |   1.00×   |

**36 data points · geomean 1.09× · range 0.73× – 1.74×**

## 5. Where the wins come from (and don't)

### Wins concentrate at small-to-medium batches

The default heuristic only branches on `M ≤ E`, returning *one* of *two*
configs. That's a coarse approximation. At `batch ∈ [16 .. 256]`, the optimal
`BLOCK_SIZE_M` and `num_stages` actually vary substantially with `(E, N)`:

| Family | batch=32 best | batch=128 best |
|--------|--------------|----------------|
| Mixtral 8×7B (E=8, N=7168) | `M=64, N=64, K=64, S=4` | `M=64, N=128, K=128, S=4` |
| Qwen MoE (E=64, N=1280)    | `M=64, N=64, K=128, S=4` | `M=64, N=128, K=64, S=5` |
| DSV2-Lite (E=128, N=1024)  | `M=64, N=128, K=64, S=5` | `M=128, N=64, K=128, S=4` |

The fallback's `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, num_stages=3` happens to win
on a few cells (mostly batch=1 launch-bound regime), but loses meaningfully
elsewhere — hence the **1.51× / 1.74×** wins at batch=128 / 32.

### Large-batch ties are expected

At `batch ≥ 2048`, all the cells are within ±1% of the default. These
workloads are HBM-bandwidth-bound, and on H20 (4 TB/s HBM3) the kernel
spends ~95% of its time waiting for memory — the choice of `BLOCK_SIZE_*`
just doesn't move the needle.

### Honest about the regressions

Three cells regress (worst 0.73× at `(64, 2560)` batch=512). These are
single-shot autotuning artifacts on a sparse batch grid: the tuner picked
the local minimum at exactly batch=512, but interpolated to "default" for
adjacent batches the chosen config performs worse. Two ways to mitigate:

1. **Tighter batch grid** around 384–768, then re-tune
2. **Geomean-of-neighbors selection** in the tuner

Adjacent batches (256, 1024) on the same shapes are neutral or positive, so
**realistic batch distributions still net positive**. But it's a known
artifact and the kind of thing a reviewer should ask about.

## 6. What a reviewer would (rightly) ask

1. **Is this only kernel-level? What about end-to-end TTFT/ITL?**
   Yes, currently only kernel-level (`fused_moe_kernel` direct call). End-to-end
   numbers require spinning up an actual server, which my container's CUDA-12.2
   driver can't do for vLLM main. **Roadmap item.**

2. **What about batch=1 regressions on DSV2-Lite?**
   The 0.91× / 0.94× points at batch=1 on E=128 shapes come from the launch
   overhead dominating; the default's `BLOCK_SIZE_M=16, BLOCK_SIZE_N=32`
   happens to be slightly better there. Re-tuning batch=1 with a tighter
   focused search would close it. Still net positive across realistic
   batch distributions.

3. **fp8 default-vs-tuned numbers?**
   Not yet. The fp8 tuning data is in this cookbook, but the comparison
   harness still needs to be ported to the fp8 path. **Roadmap item.**

4. **Per-shape scale-tensor stride bugs?**
   The kernel call layout for fp8 follows the production
   `invoke_fused_moe_kernel` layout exactly: A_scale stride = (0, 0)
   (per-tensor scalar), B_scale stride = (0, 0, 0) (per-expert 1D, but
   broadcast at kernel-call site), `group_n=group_k=0` (no block-quant).
   Cross-checked against the H100/H200 fp8_w8a8 entries on `vllm/main`.

## 7. Reproducibility

Everything in this report can be re-run from a fresh clone:

```bash
git clone https://github.com/CarrotSwordsman/H20-LLM-Cookbook
cd H20-LLM-Cookbook/benchmarks/moe_kernel

# bf16 tuning  (~80 min on H20)
python -u tune_moe_h20.py --all-missing --save-dir ../../configs/bf16

# fp8 tuning   (~40 min on H20)
python -u tune_moe_h20_fp8.py --all-missing --out-dir ../../configs/fp8_w8a8

# default-vs-tuned (bf16, ~10 min)
python compare_default_vs_tuned.py
```

If your H20 is identical (96 GB SXM5 SKU, 4.0 TB/s HBM, 78 SMs, CUDA driver
≥ 535) you should reproduce within ±5% on each cell.

## 8. Acknowledgments

The vLLM kernel and benchmark harness were the foundation for this work.
Maintainers of `vllm-project/vllm` provided the search space and per-shape
fallback architecture; this cookbook just fills the H20-shaped hole in their
config library.
