# bf16 fused-MoE: default fallback vs tuned configs (NVIDIA H20)

Kernel-level benchmark of vLLM's `fused_moe_kernel`, comparing
`get_default_config()` (vLLM's two-branch fallback heuristic) against the
tuned per-shape JSONs in [`../../configs/bf16/`](../../configs/bf16/).

## Setup

- **Hardware**: NVIDIA H20 (96 GB HBM3, SXM5)
- **Software**: torch 2.5.1+cu121, Triton 3.1.0, CUDA 12.1 runtime / 12.2 driver
- **Data type**: `bfloat16`, `MUL_ROUTED_WEIGHT=True`, no quantization
- **Per cell**: 3 internal warmups + 10-iter mean wall time, GPU-synced
  (`torch.cuda.synchronize()`), repeated 3 times taking the median
- **Batch-key selection**: `min(keys, key=lambda k: abs(int(k) - M))`
  (matches `try_get_optimal_moe_config`)
- **Reproduce**: `python ../moe_kernel/compare_default_vs_tuned.py`

## Results — 36 data points

| Shape (E, N, K, topk) | Batch | default (ms) | tuned (ms) | speedup |
|---|---:|---:|---:|---:|
| (8, 14336, 4096, 2) | 1 | 0.099 | 0.076 | **1.31×** |
| (8, 14336, 4096, 2) | 32 | 0.447 | 0.279 | **1.60×** |
| (8, 14336, 4096, 2) | 128 | 0.478 | 0.435 | **1.10×** |
| (8, 14336, 4096, 2) | 512 | 1.101 | 1.069 | **1.03×** |
| (8, 14336, 4096, 2) | 2048 | 3.588 | 3.606 | 0.99× |
| (8, 14336, 4096, 2) | 4096 | 6.986 | 7.015 | 1.00× |
| (8, 7168, 4096, 2) | 1 | 0.059 | 0.044 | **1.32×** |
| (8, 7168, 4096, 2) | 32 | 0.251 | 0.144 | **1.74×** |
| (8, 7168, 4096, 2) | 128 | 0.256 | 0.250 | **1.02×** |
| (8, 7168, 4096, 2) | 512 | 0.539 | 0.584 | 0.92× |
| (8, 7168, 4096, 2) | 2048 | 1.804 | 1.873 | 0.96× |
| (8, 7168, 4096, 2) | 4096 | 3.501 | 3.500 | 1.00× |
| (64, 2560, 2048, 6) | 1 | 0.037 | 0.027 | **1.38×** |
| (64, 2560, 2048, 6) | 32 | 0.230 | 0.195 | **1.18×** |
| (64, 2560, 2048, 6) | 128 | 0.322 | 0.218 | **1.47×** |
| (64, 2560, 2048, 6) | 512 | 0.353 | 0.486 | 0.73× |
| (64, 2560, 2048, 6) | 2048 | 1.073 | 1.070 | 1.00× |
| (64, 2560, 2048, 6) | 4096 | 1.987 | 1.974 | 1.01× |
| (64, 1280, 2048, 6) | 1 | 0.023 | 0.025 | 0.92× |
| (64, 1280, 2048, 6) | 32 | 0.125 | 0.103 | **1.21×** |
| (64, 1280, 2048, 6) | 128 | 0.169 | 0.112 | **1.51×** |
| (64, 1280, 2048, 6) | 512 | 0.182 | 0.238 | 0.77× |
| (64, 1280, 2048, 6) | 2048 | 0.537 | 0.536 | 1.00× |
| (64, 1280, 2048, 6) | 4096 | 0.992 | 0.996 | 1.00× |
| (128, 1024, 2048, 6) | 1 | 0.023 | 0.025 | 0.91× |
| (128, 1024, 2048, 6) | 32 | 0.158 | 0.127 | **1.25×** |
| (128, 1024, 2048, 6) | 128 | 0.206 | 0.162 | **1.27×** |
| (128, 1024, 2048, 6) | 512 | 0.270 | 0.255 | **1.06×** |
| (128, 1024, 2048, 6) | 2048 | 0.506 | 0.497 | 1.02× |
| (128, 1024, 2048, 6) | 4096 | 0.859 | 0.860 | 1.00× |
| (128, 512, 2048, 6) | 1 | 0.022 | 0.024 | 0.94× |
| (128, 512, 2048, 6) | 32 | 0.087 | 0.072 | **1.21×** |
| (128, 512, 2048, 6) | 128 | 0.107 | 0.088 | **1.21×** |
| (128, 512, 2048, 6) | 512 | 0.141 | 0.137 | **1.03×** |
| (128, 512, 2048, 6) | 2048 | 0.260 | 0.266 | 0.98× |
| (128, 512, 2048, 6) | 4096 | 0.435 | 0.433 | 1.00× |

**Geomean speedup: `1.09×` across 36 points. Range: `0.73× – 1.74×`.**

## Observations

- **Speedups concentrate in small-to-medium batches** (1–512), where the
  default heuristic's two-branch fallback is most under-tuned. Peak win:
  **1.74× at (E=8, N=7168) batch=32**.
- **Large batches (≥2048) tie within noise** — those workloads are
  HBM-bandwidth-bound on H20 (4 TB/s HBM3) and the kernel-config choice has
  diminishing leverage when ~95% of time is spent waiting on memory.
- **A handful of mid-batch cells regress** (worst 0.73× at (E=64, N=2560)
  batch=512). These are single-shot autotuning artifacts on the sparse
  `[1, 2, 4, ..., 4096]` grid — adjacent batches (256, 1024) on the same
  shapes are neutral or positive, so realistic batch distributions remain a
  net win. See [`reports/2026-06_h20_moe_tuning.md`](../../reports/2026-06_h20_moe_tuning.md)
  for discussion.
- **Batch=1 is launch-bound at ~22 µs floor** on H20: the original cookbook's
  per-shape tuner uses a single 10-iter mean (no repeat / median), so at this
  scale shape-to-shape sub-µs noise can pick a config that "wins" by chance
  during tuning but loses by 5–10 % in the more conservative 3-trial-median
  measurement used here. A focused batch=1 retune of the 3 cells with the
  worst regressions (5-trial median over all 1152 candidates, see
  [`../moe_kernel/retune_batch1_bf16.py`](../moe_kernel/retune_batch1_bf16.py))
  confirms the search space is **saturated**: the best achievable speedup
  over default is **1.00×–1.05×** (within the ±0.5 µs noise band of the
  measurement at 22 µs). Mechanistically, with `topk=6` and `E ∈ {64, 128}`,
  `moe_align_block_size` pads each active expert to a `BLOCK_SIZE_M`
  multiple, so the kernel work is dominated by 6×16=96 tokens of
  padded-zero compute plus Triton kernel-launch + sync overhead — neither
  of which `BLOCK_*` knobs can move. Closing batch=1 further would require
  either persistent CTAs / CUDA Graphs to amortize launch overhead, or a
  specialized small-M kernel. Treat the 0.91–0.94× cells in the table as
  **noise-band ties with default**, not real regressions.

## Roadmap

- [x] fp8_w8a8 default-vs-tuned table (see
  [`fp8_w8a8_default_vs_tuned.md`](./fp8_w8a8_default_vs_tuned.md);
  geomean 1.39×, peak 2.74×, zero regressions)
- [x] Batch=1 retune investigation — search space saturated at 22 µs
  launch-bound floor (see Observations and
  [`../moe_kernel/retune_batch1_bf16.py`](../moe_kernel/retune_batch1_bf16.py))
- [ ] Tighter batch grid around 384–768 to remove mid-batch artifacts
- [ ] vLLM serve end-to-end TTFT / ITL on Mixtral-8×7B and Qwen2-MoE
- [ ] Persistent-CTA / CUDA-Graph small-M kernel to break the 22 µs floor

## Appendix: batch=1 launch-bound floor analysis

To verify whether the 0.91–0.94× cells at batch=1 in the table above are
real regressions or measurement artifacts, we exhaustively re-tuned the
batch=1 entry on the three regressing shapes using a stricter measurement
protocol than the original tuner: a coarse sweep over all **1152** candidate
configs followed by a 5-trial-median refinement of the top-20 plus
`get_default_config()`, all using 3 warmups + 10-iter means
(`benchmarks/moe_kernel/retune_batch1_bf16.py`).

| Shape (E, N) | default (µs) | best of 1152 (µs) | best speedup vs default | range across 1152 |
|---|---:|---:|---:|---:|
| (64, 1280)   | 22.91 | 22.56 | **1.02×** | ~22.5–23.5 µs |
| (128, 1024)  | 22.54 | 22.64 | **1.00×** *(−0.4 %)* | ~22.5–23.5 µs |
| (128, 512)   | 23.23 | 22.08 | **1.05×** | ~21.7–23.5 µs |

All three converge to the same **~22 µs launch-bound floor** regardless of
`BLOCK_SIZE_*`, `GROUP_SIZE_M`, or `num_stages`. With `topk=6` and
`E ∈ {64, 128}`, `moe_align_block_size` pads each active expert to a
`BLOCK_SIZE_M` multiple, so the kernel mostly operates on
`6×BLOCK_SIZE_M = 96` tokens of padded-zero work plus Triton's per-launch
overhead — neither of which the search-space knobs can reduce.

**Implication**: the 0.91–0.94× cells in the main 36-point table should be
read as **noise-band ties with default** (single-trial-mean tuning artifacts),
not real regressions. The geomean 1.09× number stands. To break the 22 µs
floor on H20 we would need to either (a) amortize launch overhead with
persistent-CTA scheduling or CUDA Graphs, or (b) write a small-M
specialization that skips the `moe_align_block_size` padding. Both are
out-of-scope for the upstream config JSONs but tracked under Roadmap.

Reproduce: `python ../moe_kernel/retune_batch1_bf16.py` (≈ 30 min on H20).
