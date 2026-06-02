# 给 H20 调一套属于自己的 MoE kernel：从 vLLM 的 default fallback 到 1.74× 的故事

> *作者：shiyichuan · 2026-06*
>
> *本文配套仓库：[CarrotSwordsman/H20-LLM-Cookbook](https://github.com/CarrotSwordsman/H20-LLM-Cookbook) · 已合并/待合并 PR：[vllm#44152 (bf16)](https://github.com/vllm-project/vllm/pull/44152) · [vllm#44273 (fp8_w8a8)](https://github.com/vllm-project/vllm/pull/44273)*

## 写在前面

如果你拿到一台 H20、装好 vLLM、跑一下 Mixtral 8×7B 或者 Qwen MoE，在小到中 batch
（比如 batch=32 / 128）下盯 nsys 或者 `--profile`，你大概率会看到一个尴尬的现象：
**fused MoE kernel 的耗时明显比理论 roofline 高**，甚至比同代的 H100 / H200
按算力比例缩放后还差。

不是 H20 不行，而是 **vLLM 没给 H20 调 kernel config**。它的 fallback 路径只是一个
两分支的 hand-coded heuristic，对一张 *和 H100 算力差 6.7×、但 HBM 带宽只差 1.2×*
的 GPU 来说，这个 fallback 偏保守得厉害。

这篇文章讲我怎么一步步把这件事填上、跑出 24 个 H20 专属配置、最高拿到 **1.74× kernel-level 加速**，
顺便把方法、过程踩的坑、还没解决的问题都写出来 —— 让你能在自己的 H20 上 1:1 复现，
也能在自己关心的形状上接着做。

## 1. 为什么 H20 需要单独调？—— 一张表说清楚

先看 spec：

| 规格               | H100 SXM5 | H200 SXM5 | **H20 SXM5**   |
|--------------------|----------:|----------:|---------------:|
| HBM 容量           |     80 GB |    141 GB |     **96 GB**  |
| HBM 带宽           |  3.35 TB/s|   4.8 TB/s|   **4.0 TB/s** |
| BF16 稠密 TFLOPS   |       989 |       989 |      **148**   |
| FP8 稠密 TFLOPS    |      1979 |      1979 |      **296**   |
| SM 数              |       132 |       132 |       **78**   |
| L2 缓存            |     50 MB |     60 MB |     **60 MB**  |

注意三个失配的地方：

1. **算力差 6.7×，带宽差 1.2×**：H20 是典型的"带宽富、算力穷"的卡。
   等价说法：roofline 上 *算术强度的拐点* 比 H100 低得多。同一段 GEMM，
   在 H100 上你想吃满 SM 用大 BLOCK_SIZE_M / num_stages 高 occupancy
   把搬运藏掉；在 H20 上同样的 config 反而会让 SM 闲下来等 HBM。
2. **SM 数只有 78**：很多在 H100 上靠"切得碎一点让 132 个 SM 都干活"获益的
   `BLOCK_SIZE_M=16` 切法，到 H20 上反而会 occupancy 不够。
3. **L2 60 MB / 78 SM ≈ 0.77 MB/SM**：高于 H100（0.38 MB/SM），
   也就是说 H20 上 *单个 SM 能复用更多片上缓存* —— 大 `BLOCK_SIZE_K` 更划算。

这三点合在一起，**直接搬 H100 / H200 的 fused-MoE config JSON 用是会丢性能的**。
而 vLLM 默认 fallback（下文会贴代码）只有两个分支，丢得更多。

## 2. fallback 长啥样？—— 两个写死的 dict

打开 `vllm/model_executor/layers/fused_moe/fused_moe.py`，
找到 `get_default_config`：

```python
def get_default_config(M, E, N, K, ...):
    if M <= E:                              # decode-ish
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=1,  num_warps=4, num_stages=3)
    return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,  num_warps=4, num_stages=3)
```

就这。**一个 `M ≤ E` 分支，一个不分**。这是 vLLM 在缺少 per-shape JSON 时
的最后兜底 —— 思路是"宁可慢也别 OOM"，所以 BLOCK 都偏小、`num_stages=3` 偏保守。

vLLM 的真实路径其实是：

```text
try_get_optimal_moe_config()
   ├─ 找到 vllm/model_executor/layers/fused_moe/configs/E={E},N={N},device_name={DEV}.json
   │     → 用 JSON 里的 per-batch 最优 config（这是希望走到的路径）
   └─ 找不到？fallback 到 get_default_config() 那两个 dict
```

我去 `vllm/main` 翻了一下，发现：

| 模型族              | (E, N) shape                                | H100/H200 有 cfg? | H20 有 cfg? |
|---------------------|---------------------------------------------|:------------------:|:-----------:|
| Mixtral 8×7B        | E=8, N ∈ {1792, 2048, 3584, 4096, 7168, 14336} | ✅ | ❌ |
| Qwen MoE            | E=64, N ∈ {320, 640, 1280, 2560}            | ✅ | ❌ |
| DeepSeek-V2-Lite    | E=128, N ∈ {512, 1024}                       | ✅ | ❌ |

H20 一个都没有。所有这些常见 MoE 模型在 H20 上都走 fallback。这就是 cookbook
要补的洞。

## 3. 我没法直接用 vLLM 自带的 tuner

vLLM 仓库里其实带了一个 `benchmarks/kernels/benchmark_moe.py`，
理论上 `--tune` 一下就能出 config。问题是：

- 它 import 了 `vllm.platforms.cuda` 等模块
- 这些模块在 main 上要求 `torch ≥ 2.11 + CUDA 13`
- 我手头容器是 **CUDA 12.2 driver / 12.1 runtime**，升不动

绕过去的办法是写一个 standalone tuner，**只 import 两个稳定符号**：

```python
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_moe_kernel,         # Triton kernel 本体
    moe_align_block_size,     # 把 expert routing 排好序
)
```

这两个符号从 vLLM 0.6.6 起就存在，不会拖进 main 上的所有依赖。然后我手动构造
27 个位置参数（和生产路径 `invoke_fused_moe_kernel` 完全一致），就能跑。
完整脚本在 [`benchmarks/moe_kernel/tune_moe_h20.py`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/benchmarks/moe_kernel/tune_moe_h20.py)。

搜索空间和上游 benchmark 一模一样，没动：

```
BLOCK_SIZE_M ∈ {16, 32, 64, 128}        # 4
BLOCK_SIZE_N ∈ {32, 64, 128, 256}       # 4
BLOCK_SIZE_K ∈ {64, 128, 256}           # 3
GROUP_SIZE_M ∈ {1, 16, 32, 64}          # 4
num_warps    ∈ {4, 8}                   # 2
num_stages   ∈ {3, 4, 5}                # 3
                            = 1152 个 config / batch
batches: [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
          1024, 1536, 2048, 3072, 4096]                    # 18 个 batch
```

每个 cell：3 次 warmup + 10 次主循环 wall time，`torch.cuda.synchronize()` 同步，
取均值。OOM / 非法 config 用 `try/except` 接住、记 `inf`。

12 个 bf16 形状跑完大约 **80 分钟**（最大 `E=8, N=14336` 大约 10 分钟，
最小 `E=64, N=320` 大约 2 分钟）。

## 4. fp8 是更难的那一段

bf16 跑通了之后，自然要复刻 fp8_w8a8 path。结果第一版直接路由到
`invoke_fused_moe_kernel`，**单 config 比 bf16 慢了大约 1000 倍**。

排查发现：`invoke_fused_moe_kernel` 内部每次都会调
`vllm._custom_ops.scaled_fp8_quant` 重新算 activation scale。在 1152-config × 18-batch
的 sweep 里，这个 quant 调用被反复执行，搜索时间从原本的 ~40 分钟膨胀到
跑不完。

修法是把 quant 提到 per-config 循环之外，输入数据预先量化好、scale 算一次：

```python
# benchmarks/moe_kernel/tune_moe_h20_fp8.py
a_bf16 = torch.randn(num_tokens, K, dtype=torch.bfloat16, device="cuda")
a_fp8, a_scale = ops.scaled_fp8_quant(a_bf16, None)  # 一次性
b_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
b_fp8 = b_bf16.to(torch.float8_e4m3fn)
b_scale = torch.randn(E, dtype=torch.float32, device="cuda").abs() + 1e-3

# 之后所有 config 都直接用 (a_fp8, b_fp8, a_scale, b_scale, ...)
```

剩下要小心的是 fp8 path 那 27 个位置参数里的 scale stride 布局：

- `A_scale` stride = `(0, 0)` —— per-tensor 标量
- `B_scale` stride = `(0, 0, 0)` —— per-expert 1D，但 kernel 内部按 0-stride 广播
- `group_n = group_k = 0` —— **per-tensor 量化，不走 block-quant 路径**
- `use_fp8_w8a8 = True`、`compute_type = tl.bfloat16`（accumulator 还是 bf16）

我对着 vLLM main 上 H100 / H200 的 fp8_w8a8 entry 一项一项核了一遍这些数字，
保证 kernel 调用 layout 和生产路径一致。完整脚本在
[`tune_moe_h20_fp8.py`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/benchmarks/moe_kernel/tune_moe_h20_fp8.py)。

## 5. 结果：bf16 默认 vs 调优，36 个数据点

同一台 H20、同一个 kernel、同一组 batch grid。每个 cell 3 warmup + 10-iter mean，
重复 3 次取中位数。

完整 36 行表在仓库里
（[`benchmarks/results/bf16_default_vs_tuned.md`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/benchmarks/results/bf16_default_vs_tuned.md)），
这里只放头部最显眼的几行：

| Shape (E, N, K, topk)     | Batch | default (ms) | tuned (ms) | speedup |
|---------------------------|------:|-------------:|-----------:|--------:|
| (8, 7168, 4096, 2)        |  32   |  0.251       |  0.144     | **1.74×** |
| (8, 14336, 4096, 2)       |  32   |  0.447       |  0.279     | **1.60×** |
| (64, 1280, 2048, 6)       | 128   |  0.169       |  0.112     | **1.51×** |
| (64, 2560, 2048, 6)       | 128   |  0.322       |  0.218     | **1.47×** |
| (128, 1024, 2048, 6)      | 128   |  0.206       |  0.162     | **1.27×** |
| (8, 7168, 4096, 2)        | 4096  |  3.501       |  3.500     |   1.00×   |
| (64, 2560, 2048, 6)       | 512   |  0.353       |  0.486     |   0.73×   |

**36 个点 · 几何平均 1.09× · 范围 0.73× – 1.74×**。

## 6. 怎么解读这些数字？

### 收益集中在 small / medium batch

fallback 只用 `M ≤ E` 切两档。但实际上在 `batch ∈ [16 .. 256]` 这个区间，
不同 `(E, N)` 的最优 config 差异巨大：

| 模型族                         | batch=32 best                | batch=128 best                |
|--------------------------------|------------------------------|-------------------------------|
| Mixtral 8×7B (E=8, N=7168)     | `M=64, N=64, K=64, S=4`      | `M=64, N=128, K=128, S=4`     |
| Qwen MoE (E=64, N=1280)        | `M=64, N=64, K=128, S=4`     | `M=64, N=128, K=64, S=5`      |
| DSV2-Lite (E=128, N=1024)      | `M=64, N=128, K=64, S=5`     | `M=128, N=64, K=128, S=4`     |

注意 `BLOCK_SIZE_K=64 / 128` 和 `num_stages=4 / 5` —— 这两个值默认 fallback
里全是 32 / 3。这就是 H20 的"带宽富、算力穷"特性的直接体现：**更大的 K
块、更多的 pipeline stage 把 HBM 搬运藏到计算后面，是 H20 比 H100 更需要的**。

### Large batch 打平是预期内的

`batch ≥ 2048` 几乎全部和 default 在 ±1% 内。原因是这些 cell 已经被 HBM 带宽
卡死 —— 4 TB/s 的 HBM3 大约把 95% 的时间花在等数据上。Kernel config 怎么调
都搬不动剩下那 5%。

这是 cookbook 一个**重要的诚实声明**：H20 上要再榨大 batch 的性能，
应该看 attention path、KV cache、PagedAttention，**而不是 fused MoE kernel
config**。这件事大概率会让一些做 H20 部署的同学省下几天的弯路。

### 三个回归点：单点过拟合的代价

最差的 0.73× 落在 `(E=64, N=2560)` batch=512。原因是 batch grid 是稀疏的
`[1, 2, 4, ..., 4096]`，tuner 只看到了离散点；真实部署时 batch=512 这一
batch 用的是 batch=512 那个槽位的 config，但相邻的 batch=256 / 1024 在
**同一形状上是中性或正向的**，所以现实的 batch 分布下整体仍然是赚的。

要彻底修这个问题，两条路：

1. **加密 batch grid** —— 在 384 / 512 / 768 之间多采几个点重新跑
2. **邻居几何平均选择** —— tuner 选 config 时不只看当前 batch 的 best，
   而是看相邻 ±N 个 batch 的几何平均最优

后者算 cookbook 的一个 roadmap item。

## 7. fp8 那 12 个 config 呢？

仓库里 24 个 config 一半是 fp8_w8a8。但目前 README 头条数字（1.09× / 1.74×）
**只来自 bf16 的 default-vs-tuned 对比**。

原因很无聊：bf16 对比 harness 写完直接复用了，fp8 的对比 harness 还在路上。
fp8 跑出来的几何平均 / 最大值我估计会和 bf16 同向（fallback 是 dtype 无关的，
fp8 上同样欠调），但**没数字之前不上结论**。这是 cookbook 第二个 roadmap item。

> 更新：fp8 default-vs-tuned 脚本已经放进
> [`benchmarks/moe_kernel/compare_default_vs_tuned_fp8.py`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/benchmarks/moe_kernel/compare_default_vs_tuned_fp8.py)，
> 表格模板在
> [`benchmarks/results/fp8_w8a8_default_vs_tuned.md`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/benchmarks/results/fp8_w8a8_default_vs_tuned.md)，
> 等下一次 H20 时间窗口跑出来填进去。

## 8. 复现：你也能在自己的 H20 上跑

```bash
git clone https://github.com/CarrotSwordsman/H20-LLM-Cookbook
cd H20-LLM-Cookbook/benchmarks/moe_kernel

# bf16 调优  (~80 min on H20)
python -u tune_moe_h20.py --all-missing --save-dir ../../configs/bf16

# fp8 调优   (~40 min on H20)
python -u tune_moe_h20_fp8.py --all-missing --out-dir ../../configs/fp8_w8a8

# bf16 默认 vs 调优 对比 (~10 min)
python compare_default_vs_tuned.py
```

只要你的 H20 是 96 GB SXM5、HBM 4.0 TB/s、78 SM、CUDA driver ≥ 535，
每个 cell 应该都能复现到 ±5% 以内。

## 9. 最后想说的一点

这事的"含金量"其实不在于"调出了 1.74× 加速"，而在于：

- **fallback heuristic 是个有边界的近似**。它不是设计来拿冠军的，是设计来不出错的。
- **H20 这种"带宽富、算力穷"的 GPU 在国内有相当大的存量**，但社区对它的 kernel
  库覆盖明显不如 H100 / H200。任何一个用心做 sweep 的人都能在自己关心的形状上
  补出几个百分点。
- **大 batch 没收益是真相，不是失败**。把 roadmap 写诚实，比把表格 cherry-pick
  更长期受益。

如果你也在 H20 上做推理部署、有形状是 cookbook 还没覆盖的（比如更大的 expert
数、更长的 hidden）—— 欢迎开 issue / PR，我 merge。

---

*相关链接*

- 仓库：<https://github.com/CarrotSwordsman/H20-LLM-Cookbook>
- vLLM PR (bf16): <https://github.com/vllm-project/vllm/pull/44152>
- vLLM PR (fp8_w8a8): <https://github.com/vllm-project/vllm/pull/44273>
- 英文版 methodology writeup: [`reports/2026-06_h20_moe_tuning.md`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/reports/2026-06_h20_moe_tuning.md)
- H20 vs H100 vs H200 spec 对比：[`docs/h20_vs_h100_vs_h200.md`](https://github.com/CarrotSwordsman/H20-LLM-Cookbook/blob/main/docs/h20_vs_h100_vs_h200.md)
