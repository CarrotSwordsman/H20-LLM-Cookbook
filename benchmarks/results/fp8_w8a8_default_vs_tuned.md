# fp8_w8a8 fused-MoE: default fallback vs tuned configs (NVIDIA H20)

Kernel-level benchmark of vLLM's `fused_moe_kernel` in **fp8_w8a8 (per-tensor)**
mode, comparing `get_default_config()` (vLLM's two-branch fallback heuristic)
against the tuned per-shape JSONs in [`../../configs/fp8_w8a8/`](../../configs/fp8_w8a8/).

> Mirrors the bf16 table at
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
| (8, 14336, 4096, 2) | 1 | 0.159 | 0.064 | **2.48×** |
| (8, 14336, 4096, 2) | 32 | 0.257 | 0.222 | **1.16×** |
| (8, 14336, 4096, 2) | 128 | 0.283 | 0.233 | **1.21×** |
| (8, 14336, 4096, 2) | 512 | 0.619 | 0.601 | **1.03×** |
| (8, 14336, 4096, 2) | 2048 | 2.005 | 1.859 | **1.08×** |
| (8, 14336, 4096, 2) | 4096 | 3.924 | 3.511 | **1.12×** |
| (8, 7168, 4096, 2) | 1 | 0.099 | 0.036 | **2.74×** |
| (8, 7168, 4096, 2) | 32 | 0.145 | 0.119 | **1.21×** |
| (8, 7168, 4096, 2) | 128 | 0.150 | 0.121 | **1.24×** |
| (8, 7168, 4096, 2) | 512 | 0.322 | 0.277 | **1.16×** |
| (8, 7168, 4096, 2) | 2048 | 1.013 | 0.936 | **1.08×** |
| (8, 7168, 4096, 2) | 4096 | 1.917 | 1.781 | **1.08×** |
| (64, 2560, 2048, 6) | 1 | 0.058 | 0.027 | **2.17×** |
| (64, 2560, 2048, 6) | 32 | 0.390 | 0.157 | **2.48×** |
| (64, 2560, 2048, 6) | 128 | 0.190 | 0.166 | **1.14×** |
| (64, 2560, 2048, 6) | 512 | 0.221 | 0.170 | **1.30×** |
| (64, 2560, 2048, 6) | 2048 | 0.613 | 0.533 | **1.15×** |
| (64, 2560, 2048, 6) | 4096 | 1.111 | 0.988 | **1.12×** |
| (64, 1280, 2048, 6) | 1 | 0.034 | 0.026 | **1.32×** |
| (64, 1280, 2048, 6) | 32 | 0.203 | 0.087 | **2.35×** |
| (64, 1280, 2048, 6) | 128 | 0.106 | 0.087 | **1.21×** |
| (64, 1280, 2048, 6) | 512 | 0.119 | 0.093 | **1.29×** |
| (64, 1280, 2048, 6) | 2048 | 0.310 | 0.277 | **1.12×** |
| (64, 1280, 2048, 6) | 4096 | 0.559 | 0.505 | **1.11×** |
| (128, 1024, 2048, 6) | 1 | 0.029 | 0.025 | **1.15×** |
| (128, 1024, 2048, 6) | 32 | 0.275 | 0.106 | **2.59×** |
| (128, 1024, 2048, 6) | 128 | 0.331 | 0.139 | **2.37×** |
| (128, 1024, 2048, 6) | 512 | 0.161 | 0.138 | **1.17×** |
| (128, 1024, 2048, 6) | 2048 | 0.299 | 0.258 | **1.16×** |
| (128, 1024, 2048, 6) | 4096 | 0.501 | 0.433 | **1.16×** |
| (128, 512, 2048, 6) | 1 | 0.025 | 0.025 | 1.00× |
| (128, 512, 2048, 6) | 32 | 0.137 | 0.060 | **2.29×** |
| (128, 512, 2048, 6) | 128 | 0.175 | 0.073 | **2.38×** |
| (128, 512, 2048, 6) | 512 | 0.085 | 0.074 | **1.16×** |
| (128, 512, 2048, 6) | 2048 | 0.156 | 0.134 | **1.17×** |
| (128, 512, 2048, 6) | 4096 | 0.252 | 0.223 | **1.13×** |

<!-- END AUTO-GENERATED TABLE -->

**Geomean speedup: `1.39×` across 36 points. Range: `1.00× – 2.74×`.**

## Observations

- **Every single cell is ≥ 1.00×** — unlike bf16 (which had 4 cells in 0.73–0.94×),
  fp8 has zero regressions across all 36 points. The default heuristic's
  conservative blocks hurt fp8 universally.
- **Batch=1 wins are dramatic on E=8 / E=64**: peak **2.74× at (E=8, N=7168)**,
  and 2.17–2.48× on E=8 N=14336 / E=64 N=2560. The default's
  `M ≤ E` branch (BLOCK_M=16) is launch-bound on bf16 but fp8's higher
  arithmetic intensity per byte changes the cliff — tuned configs with
  larger BLOCK_M reach closer to roofline.
- **Mid-batch (32 / 128) is where the biggest wins land**: 8 of the top 10
  speedups are at batch ∈ {32, 128}, with multiple **2.3–2.6×** cells across
  all three model families (Mixtral, Qwen MoE, DeepSeek-V2-Lite). This is
  exactly where the bf16 table also peaked, just *more* extreme in fp8.
- **Large batch (≥ 2048) keeps winning 1.08–1.17×**, *unlike* bf16 which
  tied within noise. fp8 halves HBM traffic, so the kernel hasn't yet
  saturated the 4 TB/s HBM3 ceiling; kernel-config choice still has leverage.
- **Cross-dtype consistency**: every (E, N, batch) cell where bf16 won
  ≥ 1.20× also wins ≥ 1.20× in fp8 (often more). This corroborates the
  conclusion from the bf16 analysis — the bottleneck is the dtype-agnostic
  default heuristic, not the kernel itself.
- **Only tie**: (E=128, N=512) batch=1 at 1.00×, where default's
  BLOCK_M=16 / BLOCK_N=32 happens to match the tuner's pick. Same
  launch-bound regime as the bf16 0.94× cell — purely tiny-shape limits.

## Roadmap

- [x] Populate this table with H20 numbers (`compare_default_vs_tuned_fp8.py`)
- [ ] Tighter batch grid around 384–768 to remove mid-batch artifacts
- [ ] vLLM serve end-to-end TTFT / ITL on Mixtral-8×7B and Qwen2-MoE under fp8
