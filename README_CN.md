# ⚡ LLM 性能建模工具

基于 Roofline 模型的大语言模型推理性能估算器，在浏览器中运行，无需后端。

**在线体验**: [https://joursbleu.github.io/llm-perf-model/](https://joursbleu.github.io/llm-perf-model/)

## 功能

- **Prefill 延迟估算** — 基于计算密集型分析，估算首 Token 时间 (TTFT)
- **Decode 吞吐估算** — 基于显存带宽瓶颈分析，估算每秒生成 Token 数
- **显存用量分析** — 模型权重 + KV Cache + 激活内存的可视化分解
- **Roofline 图表** — 展示 Prefill/Decode 在 Roofline 模型上的位置
- **多设备横向对比** — 所有兼容设备的 Decode 吞吐对比柱状图
- **量化支持** — FP16、INT8/FP8、INT4 (GPTQ/AWQ)、3-bit、2-bit
- **张量并行** — 1-8 卡并行的性能缩放估算

## 支持的模型（共 69 个）

| 模型家族 | 规格范围 | 架构 |
|----------|----------|------|
| LLaMA 3 / 3.1 / 3.2 / 3.3 | 1B - 405B | Dense |
| Qwen 3（含 Coder、Coder-Next） | 0.6B - 235B | Dense + MoE |
| Qwen 3.5 / 3.6 / 3.8（含 Flash-Next） | 0.8B - 180B | Dense + MoE，混合线性注意力 |
| Qwen 2.5 | 0.5B - 72B | Dense |
| Mistral / Mixtral | 7B - 141B | Dense + MoE |
| DeepSeek-V2 / V3 / R1（含蒸馏版） | 2.4B - 671B | MoE + MLA |
| Phi-3 / Phi-4 | 3.8B - 14B | Dense |
| MiniMax（Text-01、M2.1、M2.5） | 10.5B - 456B | MoE（Lightning Attention） |
| Gemma 1.1 / 2 | 2B - 27B | Dense |
| Gemma 3 / 3n / 4（含 E2B/E4B、26B-A4B） | 1B - 31B | Dense + MoE，全局/线性注意力 |

混合线性注意力模型（Qwen 3.5+、Gemma 3/4、Qwen3-Coder-Next）按 softmax（O(n²)）与线性（O(n)）注意力层的实际比例分别建模。

## 支持的设备（共 57 个）

| 类别 | 设备 |
|------|------|
| NVIDIA 数据中心 | B200, H200, H100, A100 (40/80GB), V100 (16/32GB), L40S, L4, P100, P40 |
| NVIDIA GeForce | RTX 50 (5090-5060), RTX 40 (4090-4060), RTX 30 (3090-3060), RTX 20 (2080 Ti-2080) |
| NVIDIA 边缘/计算 | DGX Spark, CMP 170HX 64GB |
| AMD 数据中心 | MI325X, MI300X, MI250X, MI100, MI50 |
| AMD Radeon | RX 9700 XT, RX 9070 XT, RX 7900 XTX, AI PRO R9700, W7900, Pro V620, Strix Halo APU |
| Apple Silicon | M4 Max, M4 Pro, M2 Ultra |
| Intel | Gaudi 3, Gaudi 2 |
| Google | TPU v6e (Trillium), TPU v5p |

## 建模方法

### Prefill（计算密集型）

Prefill 阶段需要并行处理整个 prompt，主要瓶颈是计算能力：

```
Prefill Time = (2 × 活跃参数量 × Prompt 长度) / (有效 TFLOPS)
```

- 每个 token 经过模型需要约 2×P 次浮点运算（乘法 + 加法）
- MoE 模型只计算活跃参数（如 DeepSeek-V3 用 37B 而非 671B）

### Decode（显存带宽瓶颈）

Decode 阶段逐 token 生成，每生成一个 token 需要从显存读取全部模型权重：

```
每 Token 时间 = 模型大小 (字节) / (有效显存带宽)
```

### KV Cache

```
KV Cache = 2 × 层数 × KV Head 数 × Head 维度 × 序列长度 × 元素字节数 × Batch Size
```

DeepSeek 的 MLA（Multi-head Latent Attention）使用压缩的 KV 表示，大幅降低 KV Cache 占用。

### 利用率假设

| 参数 | 值 | 说明 |
|------|-----|------|
| 计算利用率 | 55% | 包含 attention、layernorm 等非矩阵乘开销 |
| 带宽利用率 | 70% | 包含内存访问模式损失 |

> 这些是理论估算值。实际性能取决于推理框架（vLLM、TensorRT-LLM、llama.cpp 等）、调度策略和系统配置。

## 使用方法

直接用浏览器打开 `index.html`，无需任何构建步骤或服务器。

## License

MIT
