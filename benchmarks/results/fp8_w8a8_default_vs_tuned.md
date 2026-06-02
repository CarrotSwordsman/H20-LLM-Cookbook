# fp8_w8a8 fused-MoE: default fallback vs tuned configs (NVIDIA H20)

Kernel-level benchmark of vLLM's `fused_moe_kernel` in **fp8_w8a8 (per-tensor)**
mode, comparing `get_default_config()` (vLLM's two-branch fallback heuristic)
against the tuned per-shape JSONs in [`../../configs/fp8_w8a8/`](../../configs/fp8_w8a8/).

> **Status:** template — populate by running
> `python ../moe_kernel/compare_default_vs_tuned_fp8.py` on H20 and pasting
> the table below. Mirrors the bf16 table at
> [`bf16_default_vs_tuned.md`](./bf16_default_vs_tuned.md).

## Setup

- **Hardware**: NVIDIA H20 (96 GB HBM3, SXM5)
- **Software**: torch 2.5.1+cu121, Triton 3.1.0, CUDA 12.1 runtime / 12.2 driver
- **Data type**: `fp8_w8a8` per-tensor (`torch.float8_e4m3fn` for both A and B);
  A scale via `vllm._custom_ops.scaled_fp8_quant` (dynamic per-tensor),
  B scale per-expert 1D, `MUL_ROUTED_WEIGHT=True`, no block-quant
  (`group_n=group_k=0`)
- **Per cell**: 3 internal warmups + 10-iter mean wall time, GPU-synced
  (`torch.cuda.synchronize()`), repeated 3 times taking the median
- **Batch-key selection**: `min(keys, key=lambda k: abs(int(k) - M))`
  (matches `try_get_optimal_moe_config`)
- **Reproduce**: `python ../moe_kernel/compare_default_vs_tuned_fp8.py`

## Results — 36 data points

<!-- BEGIN AUTO-GENERATED TABLE: paste output of compare_default_vs_tuned_fp8.py here -->

| Shape (E, N, K, topk) | Batch | default (ms) | tuned (ms) | speedup |
|---|---:|---:|---:|---:|
| (8, 14336, 4096, 2) | 1 | _TBD_ | _TBD_ | _TBD_ |
| (8, 14336, 4096, 2) | 32 | _TBD_ | _TBD_ | _TBD_ |
| (8, 14336, 4096, 2) | 128 | _TBD_ | _TBD_ | _TBD_ |
| (8, 14336, 4096, 2) | 512 | _TBD_ | _TBD_ | _TBD_ |
| (8, 14336, 4096, 2) | 2048 | _TBD_ | _TBD_ | _TBD_ |
| (8, 14336, 4096, 2) | 4096 | _TBD_ | _TBD_ | _TBD_ |
| (8, 7168, 4096, 2) | 1 | _TBD_ | _TBD_ | _TBD_ |
| (8, 7168, 4096, 2) | 32 | _TBD_ | _TBD_ | _TBD_ |
| (8, 7168, 4096, 2) | 128 | _TBD_ | _TBD_ | _TBD_ |
| (8, 7168, 4096, 2) | 512 | _TBD_ | _TBD_ | _TBD_ |
| (8, 7168, 4096, 2) | 2048 | _TBD_ | _TBD_ | _TBD_ |
| (8, 7168, 4096, 2) | 4096 | _TBD_ | _TBD_ | _TBD_ |
| (64, 2560, 2048, 6) | 1 | _TBD_ | _TBD_ | _TBD_ |
| (64, 2560, 2048, 6) | 32 | _TBD_ | _TBD_ | _TBD_ |
| (64, 2560, 2048, 6) | 128 | _TBD_ | _TBD_ | _TBD_ |
| (64, 2560, 2048, 6) | 512 | _TBD_ | _TBD_ | _TBD_ |
| (64, 2560, 2048, 6) | 2048 | _TBD_ | _TBD_ | _TBD_ |
| (64, 2560, 2048, 6) | 4096 | _TBD_ | _TBD_ | _TBD_ |
| (64, 1280, 2048, 6) | 1 | _TBD_ | _TBD_ | _TBD_ |
| (64, 1280, 2048, 6) | 32 | _TBD_ | _TBD_ | _TBD_ |
| (64, 1280, 2048, 6) | 128 | _TBD_ | _TBD_ | _TBD_ |
| (64, 1280, 2048, 6) | 512 | _TBD_ | _TBD_ | _TBD_ |
| (64, 1280, 2048, 6) | 2048 | _TBD_ | _TBD_ | _TBD_ |
| (64, 1280, 2048, 6) | 4096 | _TBD_ | _TBD_ | _TBD_ |
| (128, 1024, 2048, 6) | 1 | _TBD_ | _TBD_ | _TBD_ |
| (128, 1024, 2048, 6) | 32 | _TBD_ | _TBD_ | _TBD_ |
| (128, 1024, 2048, 6) | 128 | _TBD_ | _TBD_ | _TBD_ |
| (128, 1024, 2048, 6) | 512 | _TBD_ | _TBD_ | _TBD_ |
| (128, 1024, 2048, 6) | 2048 | _TBD_ | _TBD_ | _TBD_ |
| (128, 1024, 2048, 6) | 4096 | _TBD_ | _TBD_ | _TBD_ |
| (128, 512, 2048, 6) | 1 | _TBD_ | _TBD_ | _TBD_ |
| (128, 512, 2048, 6) | 32 | _TBD_ | _TBD_ | _TBD_ |
| (128, 512, 2048, 6) | 128 | _TBD_ | _TBD_ | _TBD_ |
| (128, 512, 2048, 6) | 512 | _TBD_ | _TBD_ | _TBD_ |
| (128, 512, 2048, 6) | 2048 | _TBD_ | _TBD_ | _TBD_ |
| (128, 512, 2048, 6) | 4096 | _TBD_ | _TBD_ | _TBD_ |

<!-- END AUTO-GENERATED TABLE -->

**Geomean speedup: `_TBD_×` across 36 points. Range: `_TBD_× – _TBD_×`.**

## Observations (to fill in after running)

Things to write up once numbers land — same skeleton as the bf16 analysis:

- Where do the wins concentrate? (Expected: small-to-medium batches, same as
  bf16 — the default heuristic is dtype-agnostic and equally under-tuned.)
- Are there any batch=1 regressions on E=128 (same launch-bound concern as
  bf16)?
- Do mid-batch single-shot autotuning artifacts (around 384–768) reproduce in
  fp8?
- **Cross-check**: do the (E, N, batch) cells where bf16 wins big also win big
  in fp8? If yes, that's evidence the default heuristic is the bottleneck, not
  the dtype.

## Roadmap

- [ ] Populate this table with H20 numbers (`compare_default_vs_tuned_fp8.py`)
- [ ] Tighter batch grid around 384–768 to remove mid-batch artifacts
- [ ] vLLM serve end-to-end TTFT / ITL on Mixtral-8×7B and Qwen2-MoE under fp8
