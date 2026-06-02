# Fused MoE kernel benchmarks

Standalone tuning + comparison scripts for the Triton `fused_moe_kernel` path
used by vLLM and SGLang. They import vLLM's kernel module directly and bypass
the high-level `invoke_fused_moe_kernel` wrapper to keep per-config overhead
minimal.

## Files

| Script                            | Purpose                                                  |
|-----------------------------------|----------------------------------------------------------|
| `tune_moe_h20.py`                 | bf16 tuner over the full 1152-config × 18-batch space    |
| `tune_moe_h20_fp8.py`             | fp8_w8a8 (per-tensor act + per-expert weight) tuner      |
| `compare_default_vs_tuned.py`     | Compare vLLM's default-fallback heuristic vs tuned JSON  |

## Prerequisites

```bash
# Any conda env with these works:
torch  >= 2.4, < 2.11
triton >= 3.0
vllm   >= 0.6.6  # any version that exposes fused_moe_kernel + moe_align_block_size
```

You don't need to build vLLM from source — `pip install vllm` is enough.

> **Note**: If you're inside a clone of vLLM main, run these scripts from a
> directory that is **not** the repo root, otherwise Python will load the
> source tree's `vllm/` package which may be missing the `_C` extension.

## Usage

### Tune all bf16 shapes

```bash
python -u tune_moe_h20.py --all-missing --save-dir ../../configs/bf16
```

This iterates over the 12 default `(E, N, K, top_k)` combinations and writes
one JSON per shape. A full sweep on H20 takes ~80 minutes (worst shape ~10
min, smallest ~2 min).

### Tune a single bf16 shape

```bash
python -u tune_moe_h20.py --E 8 --N 14336 --K 4096 --top-k 2 \
    --save-dir ../../configs/bf16
```

### Tune fp8_w8a8

```bash
python -u tune_moe_h20_fp8.py --all-missing --out-dir ../../configs/fp8_w8a8
```

The fp8 tuner pre-quantizes A and B once per benchmark call (e4m3fn, per-tensor
activation scale via `vllm._custom_ops.scaled_fp8_quant`, per-expert weight
scale), then directly invokes `fused_moe_kernel` with `use_fp8_w8a8=True`. This
mirrors how `invoke_fused_moe_kernel` calls the kernel in production but
avoids the per-config quantization overhead that would otherwise dominate the
benchmark loop.

### Reproduce the default-vs-tuned table

```bash
python compare_default_vs_tuned.py
```

Output is markdown-formatted; appends a geomean speedup line at the end. See
[`../results/bf16_default_vs_tuned.md`](../results/bf16_default_vs_tuned.md)
for the H20 reference run.

## Search space and methodology

Identical to vLLM's official `benchmarks/kernels/benchmark_moe.py`:

```python
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
               1024, 1536, 2048, 3072, 4096]

SEARCH_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [64, 128, 256],
    "GROUP_SIZE_M": [1, 16, 32, 64],
    "num_warps":    [4, 8],
    "num_stages":   [3, 4, 5],
}
# 1152 configs / batch size, 18 batch sizes per (E, N) shape
```

Per-config metric: 3 warmups + 10-iter `time.perf_counter()` mean, GPU-synced
via `torch.cuda.synchronize()`. The minimum-time config wins per batch.

## Known issues

- **Large `(E=8, N=14336)` + `BLOCK_SIZE_M=128`** can OOM at batch ≥ 2048;
  the tuner catches the exception and treats those configs as `inf`.
- **Source-tree shadowing**: see prerequisites note above.
