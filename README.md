# LLM Performance Model

[中文文档](README_CN.md)

⚡ A browser-based tool for estimating LLM inference performance across different models and hardware.

## Features

- **Prefill Latency** — Estimates time-to-first-token based on compute-bound analysis
- **Decode Throughput** — Estimates tokens/sec based on memory-bandwidth-bound analysis  
- **Memory Breakdown** — Model weights + KV cache + activation memory visualization
- **Roofline Analysis** — Visual roofline chart showing where prefill/decode operations fall
- **Multi-Device Comparison** — Side-by-side bar chart of all compatible devices
- **Quantization Support** — FP16, INT8/FP8, INT4 (GPTQ/AWQ), 3-bit, 2-bit
- **Tensor Parallel** — Multi-GPU scaling estimation

## Supported Models (69)

| Family | Sizes | Architecture |
|--------|-------|-------------|
| LLaMA 3 / 3.1 / 3.2 / 3.3 | 1B-405B | Dense |
| Qwen 3 (incl. Coder, Coder-Next) | 0.6B-235B | Dense + MoE |
| Qwen 3.5 / 3.6 / 3.8 (incl. Flash-Next) | 0.8B-180B | Dense + MoE, hybrid linear attention |
| Qwen 2.5 | 0.5B-72B | Dense |
| Mistral / Mixtral | 7B-141B | Dense + MoE |
| DeepSeek-V2 / V3 / R1 (incl. distills) | 2.4B-671B | MoE + MLA |
| Phi-3 / Phi-4 | 3.8B-14B | Dense |
| MiniMax (Text-01, M2.1, M2.5) | 10.5B-456B | MoE (Lightning Attention) |
| Gemma 1.1 / 2 | 2B-27B | Dense |
| Gemma 3 / 3n / 4 (incl. E2B/E4B, 26B-A4B) | 1B-31B | Dense + MoE, global/linear attention |

Hybrid linear-attention models (Qwen 3.5+, Gemma 3/4, Qwen3-Coder-Next) are modeled with the correct fraction of softmax (O(n²)) vs linear (O(n)) attention layers.

## Supported Devices (57)

- **NVIDIA Data Center**: B200, H200, H100, A100 (40/80GB), V100 (16/32GB), L40S, L4, P100, P40
- **NVIDIA GeForce**: RTX 50 (5090-5060), RTX 40 (4090-4060), RTX 30 (3090-3060), RTX 20 (2080 Ti-2080)
- **NVIDIA Edge/Compute**: DGX Spark, CMP 170HX 64GB
- **AMD Data Center**: MI325X, MI300X, MI250X, MI100, MI50
- **AMD Radeon**: RX 9700 XT, RX 9070 XT, RX 7900 XTX, AI PRO R9700, W7900, Pro V620, Strix Halo APU
- **Apple**: M4 Max, M4 Pro, M2 Ultra
- **Intel**: Gaudi 3, Gaudi 2
- **Google**: TPU v6e (Trillium), TPU v5p

## Methodology

**Prefill (Compute-Bound)**:
```
Prefill Time = (2 × Active Params × Prompt Length) / (Effective TFLOPS)
```

**Decode (Memory-Bandwidth-Bound)**:
```
Time per Token = Model Size (bytes) / (Effective Memory Bandwidth)
```

**Utilization Assumptions**:
- Compute utilization: 55% (accounts for attention, layernorm, non-matmul overhead)
- Bandwidth utilization: 70% (accounts for memory access pattern overhead)

These are theoretical estimates. Real-world performance depends on the inference framework (vLLM, TensorRT-LLM, llama.cpp, etc.), batching strategy, and system configuration.

## Usage

Just open `index.html` in a browser. No build step, no server required.

Or visit the live version: [https://joursbleu.github.io/llm-perf-model/](https://joursbleu.github.io/llm-perf-model/)

## License

MIT
