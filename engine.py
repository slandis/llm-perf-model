"""
LLM Inference Performance Model Engine

Roofline-based performance estimator for LLM prefill and decode phases.
Accounts for:
  - Linear layer FLOPs (2 * params * seq_len)
  - Attention FLOPs (O(n^2) for prefill, O(n) per decode step)
  - KV Cache memory (GQA / MLA aware)
  - Memory-bound vs Compute-bound analysis
  - MoE active parameter routing
  - Tensor Parallel scaling with communication overhead
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Model Definitions ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    params: float           # total parameters
    active_params: float    # active parameters (= params for dense, < params for MoE)
    layers: int
    heads: int              # num attention heads
    kv_heads: int           # num KV heads (GQA)
    hidden_dim: int
    head_dim: int
    inter_dim: int          # FFN intermediate dimension
    vocab_size: int
    max_ctx: int
    moe: bool = False
    num_experts: int = 0
    top_k: int = 0
    arch: str = ""
    mla_compressed_dim: Optional[int] = None  # DeepSeek MLA
    softmax_attn_layers: Optional[int] = None  # None=all softmax; N=only N layers use O(n²) softmax, rest use O(n) linear attn
    note: str = ""


MODELS = {
    # LLaMA family
    # Qwen 3 family
    "qwen3-0.6b": ModelConfig("Qwen3 0.6B", 0.75e9, 0.75e9, 28, 16, 8, 1024, 128, 3072, 151936, 32768, arch="Qwen3"),
    "qwen3-1.7b": ModelConfig("Qwen3 1.7B", 2.03e9, 2.03e9, 28, 16, 8, 2048, 128, 6144, 151936, 32768, arch="Qwen3"),
    "qwen3-4b": ModelConfig("Qwen3 4B", 4.02e9, 4.02e9, 36, 32, 8, 2560, 128, 9728, 151936, 32768, arch="Qwen3"),
    "qwen3-8b": ModelConfig("Qwen3 8B", 8.19e9, 8.19e9, 36, 32, 8, 4096, 128, 12288, 151936, 32768, arch="Qwen3"),
    "qwen3-14b": ModelConfig("Qwen3 14B", 14.77e9, 14.77e9, 40, 40, 8, 5120, 128, 17408, 151936, 32768, arch="Qwen3"),
    "qwen3-32b": ModelConfig("Qwen3 32B", 32.76e9, 32.76e9, 64, 64, 8, 5120, 128, 25600, 151936, 32768, arch="Qwen3"),
    "qwen3-30b-a3b": ModelConfig("Qwen3 30B-A3B", 30.5e9, 3.4e9, 48, 32, 4, 2048, 128, 768, 151936, 32768, True, 128, 8, arch="Qwen3", note="MoE"),
    "qwen3-235b-a22b": ModelConfig("Qwen3 235B-A22B", 235.1e9, 22.5e9, 94, 64, 4, 4096, 128, 1536, 151936, 32768, True, 128, 8, arch="Qwen3", note="MoE"),
    "qwen3-coder-30b-a3b": ModelConfig("Qwen3-Coder 30B-A3B", 30.5e9, 3.4e9, 48, 32, 4, 2048, 128, 768, 151936, 262144, True, 128, 8, arch="Qwen3", note="MoE coder"),
    "qwen3-coder-next": ModelConfig("Qwen3-Coder-Next 80B-A3B", 80.0e9, 3.9e9, 48, 16, 2, 2048, 256, 512, 151936, 262144, True, 512, 10, arch="Qwen3Next", softmax_attn_layers=12, note="MoE + hybrid linear attn"),

    # Qwen 3.5 / 3.6 / 3.8 family (hybrid linear + full attention)
    "qwen3.5-0.8b": ModelConfig("Qwen3.5 0.8B", 0.87e9, 0.87e9, 24, 16, 4, 896, 256, 4864, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=6, note="hybrid linear attn"),
    "qwen3.5-2b": ModelConfig("Qwen3.5 2B", 2.27e9, 2.27e9, 30, 16, 4, 1536, 256, 5632, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=8, note="hybrid linear attn"),
    "qwen3.5-4b": ModelConfig("Qwen3.5 4B", 4.66e9, 4.66e9, 32, 16, 4, 2560, 256, 9216, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=8, note="hybrid linear attn"),
    "qwen3.5-9b": ModelConfig("Qwen3.5 9B", 9.4e9, 9.4e9, 32, 16, 4, 4096, 256, 12288, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=8, note="hybrid linear attn"),
    "qwen3.5-27b": ModelConfig("Qwen3.5 27B", 27.8e9, 27.8e9, 64, 24, 4, 5120, 256, 17408, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=16, note="hybrid linear attn"),
    "qwen3.6-27b": ModelConfig("Qwen3.6 27B", 27.8e9, 27.8e9, 64, 24, 4, 5120, 256, 17408, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=16, note="hybrid linear attn"),
    "qwen3.6-35b-a3b": ModelConfig("Qwen3.6 35B-A3B", 36.0e9, 3.4e9, 40, 16, 2, 2048, 256, 512, 248320, 262144, True, 256, 8, arch="Qwen3.5MoE", softmax_attn_layers=10, note="MoE + hybrid linear attn"),
    "qwen3.8-27b": ModelConfig("Qwen3.8 27B", 27.8e9, 27.8e9, 64, 24, 4, 5120, 256, 17408, 248320, 262144, arch="Qwen3.5", softmax_attn_layers=16, note="hybrid linear attn"),
    "qwen3.8-flash-next": ModelConfig("Qwen3.8 Flash-Next 180B", 180.0e9, 95.0e9, 48, 24, 2, 2560, 256, 640, 248320, 262144, True, 512, 10, arch="Qwen4Exp", softmax_attn_layers=12, note="MoE + hybrid linear attn"),

    # LLaMA family
    "llama-3.2-1b": ModelConfig("LLaMA 3.2 1B", 1.24e9, 1.24e9, 16, 32, 8, 2048, 64, 8192, 128256, 131072, arch="LLaMA"),
    "llama-3.2-3b": ModelConfig("LLaMA 3.2 3B", 3.21e9, 3.21e9, 28, 24, 8, 3072, 128, 8192, 128256, 131072, arch="LLaMA"),
    "llama-3-8b": ModelConfig("LLaMA 3 8B", 8e9, 8e9, 32, 32, 8, 4096, 128, 14336, 128256, 8192, arch="LLaMA"),
    "llama-3.1-8b": ModelConfig("LLaMA 3.1 8B", 8e9, 8e9, 32, 32, 8, 4096, 128, 14336, 128256, 131072, arch="LLaMA"),
    "llama-3-70b": ModelConfig("LLaMA 3 70B", 70.6e9, 70.6e9, 80, 64, 8, 8192, 128, 28672, 128256, 8192, arch="LLaMA"),
    "llama-3.1-70b": ModelConfig("LLaMA 3.1 70B", 70.6e9, 70.6e9, 80, 64, 8, 8192, 128, 28672, 128256, 131072, arch="LLaMA"),
    "llama-3.3-70b": ModelConfig("LLaMA 3.3 70B", 70.6e9, 70.6e9, 80, 64, 8, 8192, 128, 28672, 128256, 131072, arch="LLaMA"),
    "llama-3.1-405b": ModelConfig("LLaMA 3.1 405B", 405e9, 405e9, 126, 128, 8, 16384, 128, 53248, 128256, 131072, arch="LLaMA"),

    # Qwen 2.5 family
    "qwen2.5-0.5b": ModelConfig("Qwen 2.5 0.5B", 0.49e9, 0.49e9, 24, 14, 2, 896, 64, 4864, 151936, 32768, arch="Qwen2"),
    "qwen2.5-1.5b": ModelConfig("Qwen 2.5 1.5B", 1.54e9, 1.54e9, 28, 12, 2, 1536, 128, 8960, 151936, 32768, arch="Qwen2"),
    "qwen2.5-3b": ModelConfig("Qwen 2.5 3B", 3.09e9, 3.09e9, 36, 16, 2, 2048, 128, 11008, 151936, 32768, arch="Qwen2"),
    "qwen2.5-7b": ModelConfig("Qwen 2.5 7B", 7.6e9, 7.6e9, 28, 28, 4, 3584, 128, 18944, 152064, 131072, arch="Qwen2"),
    "qwen2.5-14b": ModelConfig("Qwen 2.5 14B", 14.7e9, 14.7e9, 48, 40, 8, 5120, 128, 13824, 152064, 131072, arch="Qwen2"),
    "qwen2.5-32b": ModelConfig("Qwen 2.5 32B", 32.5e9, 32.5e9, 64, 40, 8, 5120, 128, 27648, 152064, 131072, arch="Qwen2"),
    "qwen2.5-72b": ModelConfig("Qwen 2.5 72B", 72.7e9, 72.7e9, 80, 64, 8, 8192, 128, 29568, 152064, 131072, arch="Qwen2"),

    # Mistral / Mixtral
    "mistral-7b": ModelConfig("Mistral 7B", 7.3e9, 7.3e9, 32, 32, 8, 4096, 128, 14336, 32000, 32768, arch="Mistral"),
    "mistral-small-24b": ModelConfig("Mistral Small 24B", 24e9, 24e9, 40, 32, 8, 5120, 128, 14336, 32768, 32768, arch="Mistral"),
    "mixtral-8x7b": ModelConfig("Mixtral 8x7B", 46.7e9, 12.9e9, 32, 32, 8, 4096, 128, 14336, 32000, 32768, True, 8, 2, arch="Mixtral"),
    "mixtral-8x22b": ModelConfig("Mixtral 8x22B", 141e9, 39e9, 56, 48, 8, 6144, 128, 16384, 32768, 65536, True, 8, 2, arch="Mixtral"),

    # DeepSeek
    "deepseek-v2-lite": ModelConfig("DeepSeek-V2-Lite", 15.7e9, 2.4e9, 27, 16, 16, 2048, 128, 10944, 102400, 32768, True, 64, 6, arch="DeepSeek", mla_compressed_dim=512),
    "deepseek-v2": ModelConfig("DeepSeek-V2", 236e9, 21e9, 60, 128, 128, 5120, 128, 12288, 102400, 131072, True, 160, 6, arch="DeepSeek", mla_compressed_dim=512, note="MLA + shared experts"),
    "deepseek-v3": ModelConfig("DeepSeek-V3", 671e9, 37e9, 61, 128, 128, 7168, 128, 18432, 129280, 131072, True, 256, 8, arch="DeepSeek", mla_compressed_dim=512, note="MLA + shared experts"),
    "deepseek-r1": ModelConfig("DeepSeek-R1", 671e9, 37e9, 61, 128, 128, 7168, 128, 18432, 129280, 131072, True, 256, 8, arch="DeepSeek", mla_compressed_dim=512, note="MLA + shared experts"),
    "deepseek-r1-7b": ModelConfig("DeepSeek-R1-Distill 7B", 7.6e9, 7.6e9, 28, 28, 4, 3584, 128, 18944, 152064, 131072, arch="Qwen2", note="Distilled from Qwen 2.5 7B"),
    "deepseek-r1-14b": ModelConfig("DeepSeek-R1-Distill 14B", 14.7e9, 14.7e9, 48, 40, 8, 5120, 128, 13824, 152064, 131072, arch="Qwen2", note="Distilled from Qwen 2.5 14B"),
    "deepseek-r1-32b": ModelConfig("DeepSeek-R1-Distill 32B", 32.5e9, 32.5e9, 64, 40, 8, 5120, 128, 27648, 152064, 131072, arch="Qwen2", note="Distilled from Qwen 2.5 32B"),
    "deepseek-r1-70b": ModelConfig("DeepSeek-R1-Distill 70B", 70.6e9, 70.6e9, 80, 64, 8, 8192, 128, 28672, 128256, 131072, arch="LLaMA", note="Distilled from LLaMA 3 70B"),

    # Phi
    "phi-3-mini": ModelConfig("Phi-3 Mini 3.8B", 3.8e9, 3.8e9, 32, 32, 32, 3072, 96, 8192, 32064, 128000, arch="Phi"),
    "phi-3-small": ModelConfig("Phi-3 Small 7B", 7.4e9, 7.4e9, 32, 32, 8, 4096, 128, 14336, 32064, 128000, arch="Phi"),
    "phi-3-medium": ModelConfig("Phi-3 Medium 14B", 14e9, 14e9, 40, 40, 10, 5120, 128, 17920, 32064, 128000, arch="Phi"),
    "phi-4": ModelConfig("Phi-4 14B", 14e9, 14e9, 40, 40, 10, 5120, 128, 17920, 100352, 16384, arch="Phi"),

    # MiniMax
    "minimax-text-01": ModelConfig("MiniMax-Text-01", 456e9, 45.9e9, 80, 64, 8, 6144, 128, 9216, 200064, 4096000, True, 32, 2, arch="MiniMax", softmax_attn_layers=10, note="Lightning Attention + MoE"),
    "minimax-m2.1": ModelConfig("MiniMax-M2.1", 229e9, 10.5e9, 62, 48, 8, 3072, 128, 1536, 200064, 196608, True, 256, 8, arch="MiniMax", note="MoE + MTP"),
    "minimax-m2.5": ModelConfig("MiniMax-M2.5", 229e9, 10.5e9, 62, 48, 8, 3072, 128, 1536, 200064, 196608, True, 256, 8, arch="MiniMax", note="MoE + MTP"),

    # Gemma 1.1
    "gemma-1.1-2b": ModelConfig("Gemma 1.1 2B", 2.6e9, 2.6e9, 26, 8, 4, 2048, 128, 16384, 256000, 8192, arch="Gemma"),
    "gemma-1.1-7b": ModelConfig("Gemma 1.1 7B", 8.1e9, 8.1e9, 30, 16, 8, 3072, 128, 24576, 256000, 8192, arch="Gemma"),

    # Gemma 2
    "gemma-2-2b": ModelConfig("Gemma 2 2B", 2.6e9, 2.6e9, 26, 8, 4, 2304, 256, 9216, 256000, 8192, arch="Gemma"),
    "gemma-2-9b": ModelConfig("Gemma 2 9B", 9.2e9, 9.2e9, 42, 16, 8, 3584, 256, 14336, 256000, 8192, arch="Gemma"),
    "gemma-2-27b": ModelConfig("Gemma 2 27B", 27.2e9, 27.2e9, 46, 32, 16, 4608, 128, 36864, 256000, 8192, arch="Gemma"),

    # Gemma 3 (hybrid sliding-window + global attention)
    "gemma-3-1b": ModelConfig("Gemma 3 1B", 1.0e9, 1.0e9, 26, 4, 1, 1152, 256, 6912, 262144, 32768, arch="Gemma3", softmax_attn_layers=26, note="all global attn"),
    "gemma-3-4b": ModelConfig("Gemma 3 4B", 4.4e9, 4.4e9, 34, 8, 4, 2560, 256, 10240, 262208, 131072, arch="Gemma3", softmax_attn_layers=6, note="global every 6th layer"),
    "gemma-3-12b": ModelConfig("Gemma 3 12B", 12.4e9, 12.4e9, 48, 16, 8, 3840, 256, 15360, 262208, 131072, arch="Gemma3", softmax_attn_layers=8, note="global every 6th layer"),
    "gemma-3-27b": ModelConfig("Gemma 3 27B", 27.4e9, 27.4e9, 62, 32, 16, 5376, 128, 21504, 262208, 131072, arch="Gemma3", softmax_attn_layers=11, note="global every 6th layer"),

    # Gemma 3n (effective params, PLE excluded)
    "gemma-3n-e2b": ModelConfig("Gemma 3n E2B", 2.7e9, 2.7e9, 30, 8, 2, 2048, 256, 8192, 262400, 32768, arch="Gemma3n", note="effective params"),
    "gemma-3n-e4b": ModelConfig("Gemma 3n E4B", 5.1e9, 5.1e9, 35, 8, 2, 2048, 256, 16384, 262400, 32768, arch="Gemma3n", note="effective params"),

    # Gemma 4 (hybrid sliding-window + global attention)
    "gemma-4-e2b": ModelConfig("Gemma 4 E2B", 5.1e9, 2.3e9, 35, 8, 1, 1536, 256, 6144, 262144, 131072, arch="Gemma4", softmax_attn_layers=7, note="effective params"),
    "gemma-4-e4b": ModelConfig("Gemma 4 E4B", 8.0e9, 4.5e9, 42, 8, 2, 2560, 256, 10240, 262144, 131072, arch="Gemma4", softmax_attn_layers=7, note="effective params"),
    "gemma-4-12b": ModelConfig("Gemma 4 12B", 11.95e9, 11.95e9, 48, 16, 8, 3840, 256, 15360, 262144, 262144, arch="Gemma4", softmax_attn_layers=8, note="global every 6th layer"),
    "gemma-4-31b": ModelConfig("Gemma 4 31B", 30.7e9, 30.7e9, 60, 32, 16, 5376, 256, 21504, 262144, 262144, arch="Gemma4", softmax_attn_layers=10, note="global every 6th layer"),
    "gemma-4-26b-a4b": ModelConfig("Gemma 4 26B-A4B", 25.2e9, 3.8e9, 30, 16, 8, 2816, 256, 704, 262144, 262144, True, 128, 8, arch="Gemma4", softmax_attn_layers=5, note="MoE + global every 6th layer"),
}


# ─── Device Definitions ──────────────────────────────────────────────────────

@dataclass
class DeviceConfig:
    name: str
    vram: float         # GB
    bw: float           # GB/s memory bandwidth
    fp16_tflops: float
    fp8_tflops: float
    int8_tflops: float
    vendor: str


DEVICES = {
    # NVIDIA Data Center
    "b200-sxm": DeviceConfig("NVIDIA B200 SXM", 192, 8000, 2250, 4500, 4500, "nvidia"),
    "h200-sxm": DeviceConfig("NVIDIA H200 SXM", 141, 4800, 989, 1979, 1979, "nvidia"),
    "h100-sxm": DeviceConfig("NVIDIA H100 SXM", 80, 3350, 989, 1979, 1979, "nvidia"),
    "a100-sxm-80": DeviceConfig("NVIDIA A100 80GB", 80, 2039, 312, 312, 624, "nvidia"),
    "a100-sxm-40": DeviceConfig("NVIDIA A100 40GB", 40, 1555, 312, 312, 624, "nvidia"),
    "l40s": DeviceConfig("NVIDIA L40S", 48, 864, 362, 733, 733, "nvidia"),
    "l4": DeviceConfig("NVIDIA L4", 24, 300, 121, 242, 242, "nvidia"),
    # NVIDIA Legacy Data Center (Pascal / Volta)
    "p100-16": DeviceConfig("NVIDIA P100 16GB", 16, 732, 101, 0, 202, "nvidia"),
    "p40": DeviceConfig("NVIDIA P40", 24, 347, 12, 0, 24, "nvidia"),
    "v100-sxm-16": DeviceConfig("NVIDIA V100 16GB SXM", 16, 900, 125, 0, 250, "nvidia"),
    "v100-sxm-32": DeviceConfig("NVIDIA V100 32GB SXM", 32, 900, 125, 0, 250, "nvidia"),
    "cmp170hx": DeviceConfig("NVIDIA CMP 170HX 64GB", 64, 1386, 174, 0, 48, "nvidia"),
    # NVIDIA Consumer / Edge
    "rtx-5090": DeviceConfig("RTX 5090", 32, 1792, 105, 210, 210, "nvidia"),
    "rtx-5080": DeviceConfig("RTX 5080", 16, 960, 44, 89, 89, "nvidia"),
    "rtx-5070-ti": DeviceConfig("RTX 5070 Ti", 16, 896, 37, 73, 73, "nvidia"),
    "rtx-5070": DeviceConfig("RTX 5070", 12, 497, 30, 59, 59, "nvidia"),
    "rtx-5060-ti": DeviceConfig("RTX 5060 Ti", 16, 448, 22, 44, 44, "nvidia"),
    "rtx-5060": DeviceConfig("RTX 5060", 8, 448, 14, 29, 29, "nvidia"),
    "rtx-4090": DeviceConfig("RTX 4090", 24, 1008, 165, 330, 330, "nvidia"),
    "rtx-4080-super": DeviceConfig("RTX 4080 Super", 16, 736, 97, 195, 195, "nvidia"),
    "rtx-4080": DeviceConfig("RTX 4080", 16, 717, 97, 194, 194, "nvidia"),
    "rtx-4070-ti-super": DeviceConfig("RTX 4070 Ti Super", 16, 672, 44, 87, 87, "nvidia"),
    "rtx-4070-ti": DeviceConfig("RTX 4070 Ti", 12, 672, 40, 80, 80, "nvidia"),
    "rtx-4070-super": DeviceConfig("RTX 4070 Super", 12, 504, 38, 75, 75, "nvidia"),
    "rtx-4070": DeviceConfig("RTX 4070", 12, 504, 40, 80, 80, "nvidia"),
    "rtx-4060-ti": DeviceConfig("RTX 4060 Ti", 16, 288, 22, 44, 44, "nvidia"),
    "rtx-4060": DeviceConfig("RTX 4060", 8, 272, 15, 30, 30, "nvidia"),
    "rtx-3090": DeviceConfig("RTX 3090", 24, 936, 71, 71, 142, "nvidia"),
    "rtx-3080-ti": DeviceConfig("RTX 3080 Ti", 12, 912, 35, 35, 70, "nvidia"),
    "rtx-3080": DeviceConfig("RTX 3080", 10, 912, 30, 30, 59, "nvidia"),
    "rtx-3070-ti": DeviceConfig("RTX 3070 Ti", 8, 608, 22, 22, 43, "nvidia"),
    "rtx-3070": DeviceConfig("RTX 3070", 8, 448, 20, 20, 41, "nvidia"),
    "rtx-3060-ti": DeviceConfig("RTX 3060 Ti", 8, 448, 16, 16, 32, "nvidia"),
    "rtx-3060": DeviceConfig("RTX 3060 12GB", 12, 360, 13, 13, 25, "nvidia"),
    "rtx-2080-ti": DeviceConfig("RTX 2080 Ti", 11, 616, 13.4, 0, 27, "nvidia"),
    "rtx-2080-super": DeviceConfig("RTX 2080 Super", 8, 616, 15.5, 0, 31, "nvidia"),
    "rtx-2080": DeviceConfig("RTX 2080", 8, 448, 14.8, 0, 30, "nvidia"),
    "dgx-spark": DeviceConfig("NVIDIA DGX Spark (GB10)", 128, 273, 209, 418, 418, "nvidia"),
    # AMD Data Center
    "mi300x": DeviceConfig("AMD MI300X", 192, 5300, 1307, 2614, 2614, "amd"),
    "mi325x": DeviceConfig("AMD MI325X", 256, 6000, 1307, 2614, 2614, "amd"),
    "mi250x": DeviceConfig("AMD MI250X", 128, 3277, 383, 383, 383, "amd"),
    "mi100": DeviceConfig("AMD MI100 32GB", 32, 1024, 13.8, 0, 13.8, "amd"),
    "mi50": DeviceConfig("AMD MI50 32GB", 32, 1024, 13.8, 0, 13.8, "amd"),
    # AMD Data Center / Pro (Vega 20)
    "pro-v620": DeviceConfig("AMD Radeon Pro V620 16GB", 16, 512, 13.8, 0, 13.8, "amd"),
    # AMD Consumer / Pro
    "rx-9700xt": DeviceConfig("AMD RX 9700 XT", 24, 864, 56, 112, 112, "amd"),
    "rx-9070xt": DeviceConfig("AMD RX 9070 XT", 16, 640, 49, 98, 98, "amd"),
    "ai-pro-r9700": DeviceConfig("AMD Radeon AI PRO R9700", 32, 645, 48, 96, 96, "amd"),
    "w7900": DeviceConfig("AMD W7900", 48, 864, 61, 61, 122, "amd"),
    "rx-7900xtx": DeviceConfig("AMD RX 7900 XTX", 24, 960, 123, 123, 123, "amd"),
    # AMD APU
    "strix-halo": DeviceConfig("AMD Ryzen AI Max 395 (Strix Halo)", 128, 256, 30, 30, 60, "amd"),
    # Apple Silicon
    "m4-max": DeviceConfig("Apple M4 Max", 128, 546, 54, 54, 54, "apple"),
    "m4-pro": DeviceConfig("Apple M4 Pro", 48, 273, 22, 22, 22, "apple"),
    "m2-ultra": DeviceConfig("Apple M2 Ultra", 192, 800, 27, 27, 27, "apple"),
    # Intel Gaudi
    "gaudi3": DeviceConfig("Intel Gaudi 3", 128, 3700, 900, 1835, 1835, "intel"),
    "gaudi2": DeviceConfig("Intel Gaudi 2", 96, 2460, 420, 865, 865, "intel"),
    # Google TPU
    "tpu-v6e": DeviceConfig("Google TPU v6e (Trillium)", 32, 1640, 918, 918, 1836, "google"),
    "tpu-v5p": DeviceConfig("Google TPU v5p", 95, 2765, 459, 459, 918, "google"),
}


# ─── Performance Engine ──────────────────────────────────────────────────────

@dataclass
class PerfConfig:
    """Tunable utilization parameters."""
    compute_util: float = 0.55      # matmul compute utilization
    bw_util: float = 0.70           # memory bandwidth utilization
    tp_comm_overhead: float = 0.05  # per-GPU communication overhead for TP > 1
    flash_attn: bool = True         # assume FlashAttention (reduces memory, IO-aware)


@dataclass
class PerfResult:
    # Identifiers
    model_name: str
    device_name: str
    quant_bits: int
    prompt_len: int
    output_len: int
    batch_size: int
    tp: int

    # Prefill
    prefill_linear_flops: float     # FLOPs for linear layers during prefill
    prefill_attn_flops: float       # FLOPs for attention during prefill
    prefill_total_flops: float
    prefill_time_ms: float
    prefill_tps: float              # tokens/sec during prefill

    # Decode
    decode_linear_flops: float      # per-step linear FLOPs
    decode_attn_flops: float        # per-step attention FLOPs (depends on current seq pos)
    decode_mem_bound_ms: float      # memory-bound time per token
    decode_compute_bound_ms: float  # compute-bound time per token
    decode_time_per_token_ms: float
    decode_tps: float               # tokens/sec decode
    decode_bottleneck: str          # "memory" or "compute"

    # Total
    total_time_ms: float
    total_output_tps: float         # output_len / total_time

    # Memory
    model_size_gb: float            # full model weights
    kv_cache_gb: float
    activation_gb: float
    total_memory_gb: float
    device_memory_gb: float
    fits_in_memory: bool

    # Roofline
    ridge_point: float              # FLOPs/Byte where compute = memory bound
    prefill_arith_intensity: float
    decode_arith_intensity: float

    # Device effective specs
    effective_tflops: float
    effective_bw_gbs: float

    # FlashAttention
    flash_attn: bool
    attn_score_memory_gb: float     # N×N attention matrix memory (0 with FlashAttn)


def get_device_tflops(device: DeviceConfig, quant_bits: int) -> float:
    """Select appropriate TFLOPS based on quantization and device capabilities."""
    if quant_bits <= 8:
        best = max(device.fp8_tflops, device.int8_tflops)
        if best > device.fp16_tflops:
            return best
    return device.fp16_tflops


def calc_kv_cache_gb(model: ModelConfig, seq_len: int, batch_size: int, quant_bits: int) -> float:
    """Calculate KV cache memory in GB."""
    if model.mla_compressed_dim:
        # DeepSeek MLA: compressed KV representation
        # layers * compressed_dim * seq_len * 2 bytes (FP16) * batch
        return (model.layers * model.mla_compressed_dim * seq_len * 2 * batch_size) / 1e9

    # How many layers use standard softmax attention vs linear attention
    softmax_layers = model.softmax_attn_layers if model.softmax_attn_layers is not None else model.layers
    linear_layers = model.layers - softmax_layers

    # Standard GQA KV cache: only softmax attention layers need per-token KV storage
    kv_bytes = max(quant_bits, 16) / 8  # KV cache usually FP16 minimum
    kv_cache = (2 * softmax_layers * model.kv_heads * model.head_dim * seq_len * kv_bytes * batch_size) / 1e9

    # Linear attention layers maintain a fixed-size state: heads * head_dim² * 2 bytes per layer
    linear_state = (linear_layers * model.heads * model.head_dim * model.head_dim * 2 * batch_size) / 1e9

    return kv_cache + linear_state


def calc_attention_flops_prefill(model: ModelConfig, seq_len: int, batch_size: int) -> float:
    """
    Attention FLOPs for prefill phase.
    Softmax attention (standard): O(n²) per layer — 4 * B * H * S² * D
    Linear attention (Lightning): O(n·d) per layer — 4 * B * H * S * D²
      (computes K^T@V then Q@state, each O(n*d²) per head)
    """
    softmax_layers = model.softmax_attn_layers if model.softmax_attn_layers is not None else model.layers
    linear_layers = model.layers - softmax_layers

    softmax_flops = 4 * batch_size * model.heads * seq_len * seq_len * model.head_dim * softmax_layers
    linear_flops = 4 * batch_size * model.heads * seq_len * model.head_dim * model.head_dim * linear_layers

    return softmax_flops + linear_flops


def calc_attention_flops_decode_step(model: ModelConfig, current_pos: int, batch_size: int) -> float:
    """
    Attention FLOPs for one decode step.
    Softmax: O(pos) per layer — 4 * B * H * pos * D (scan full KV cache)
    Linear:  O(d²) per layer  — 4 * B * H * D² (fixed-size state lookup)
    """
    softmax_layers = model.softmax_attn_layers if model.softmax_attn_layers is not None else model.layers
    linear_layers = model.layers - softmax_layers

    softmax_flops = 4 * batch_size * model.heads * current_pos * model.head_dim * softmax_layers
    linear_flops = 4 * batch_size * model.heads * model.head_dim * model.head_dim * linear_layers

    return softmax_flops + linear_flops


def estimate_performance(
    model_key: str,
    device_key: str,
    quant_bits: int = 4,
    prompt_len: int = 512,
    output_len: int = 256,
    batch_size: int = 1,
    tp: int = 1,
    config: Optional[PerfConfig] = None,
) -> PerfResult:
    """
    Main performance estimation function.

    Returns detailed PerfResult with prefill, decode, memory, and roofline analysis.
    """
    if config is None:
        config = PerfConfig()

    model = MODELS[model_key]
    device = DEVICES[device_key]

    # ── Effective device specs with TP ──
    tp_efficiency = 1.0 - config.tp_comm_overhead * max(0, tp - 1)  # diminishing returns
    tp_efficiency = max(tp_efficiency, 0.5)  # floor at 50%

    raw_tflops = get_device_tflops(device, quant_bits)
    effective_tflops = raw_tflops * tp * tp_efficiency * config.compute_util  # TFLOPS
    effective_bw = device.bw * tp * tp_efficiency * config.bw_util  # GB/s

    # ── Memory ──
    model_size_gb = (model.params * quant_bits / 8) / 1e9
    total_seq_len = prompt_len + output_len
    kv_cache_gb = calc_kv_cache_gb(model, total_seq_len, batch_size, quant_bits)
    # Activation memory: hidden states + residual
    activation_gb = (model.hidden_dim * prompt_len * batch_size * 4 * 2) / 1e9
    # Without FlashAttention: must materialize N×N attention score matrix in HBM
    # Peak per layer: batch * heads * seq_len^2 * 4 bytes (FP32 softmax)
    if config.flash_attn:
        attn_score_memory_gb = 0.0  # tiled in SRAM, never materialized
    else:
        attn_score_memory_gb = (batch_size * model.heads * prompt_len * prompt_len * 4) / 1e9
    activation_gb += attn_score_memory_gb
    total_memory_gb = model_size_gb + kv_cache_gb + activation_gb
    device_memory_gb = device.vram * tp
    fits_in_memory = total_memory_gb <= device_memory_gb

    # ── Prefill Phase ──
    # Linear layers: 2 * active_params * prompt_len * batch_size
    prefill_linear_flops = 2 * model.active_params * prompt_len * batch_size
    # Attention: O(n^2) per layer
    prefill_attn_flops = calc_attention_flops_prefill(model, prompt_len, batch_size)
    prefill_total_flops = prefill_linear_flops + prefill_attn_flops

    # Without FlashAttention, attention is IO-bound due to N×N HBM read/write.
    # Model as: linear layers run at full compute_util, attention portion has
    # reduced effective throughput due to HBM IO overhead.
    if config.flash_attn:
        # FlashAttention: fused kernel, attention runs at ~same utilization as matmul
        prefill_time_s = prefill_total_flops / (effective_tflops * 1e12)
    else:
        # Without FlashAttention: attention has extra HBM IO (read Q,K → write S → read S,V → write O)
        # Attention effective throughput ≈ 40% of peak (IO overhead dominates)
        attn_util_factor = 0.40 / config.compute_util  # ratio vs normal utilization
        prefill_linear_time_s = prefill_linear_flops / (effective_tflops * 1e12)
        prefill_attn_time_s = prefill_attn_flops / (effective_tflops * 1e12 * attn_util_factor)
        prefill_time_s = prefill_linear_time_s + prefill_attn_time_s

    prefill_time_ms = prefill_time_s * 1000
    prefill_tps = (prompt_len * batch_size) / prefill_time_s if prefill_time_s > 0 else 0

    # ── Decode Phase ──
    # Linear layers per decode step: 2 * active_params * batch_size
    decode_linear_flops = 2 * model.active_params * batch_size

    # For decode, attention cost varies with position. Use average position.
    avg_decode_pos = prompt_len + output_len / 2  # average KV length during decode
    decode_attn_flops = calc_attention_flops_decode_step(model, avg_decode_pos, batch_size)
    decode_total_flops_per_step = decode_linear_flops + decode_attn_flops

    # Memory-bound: read all active weights once per step
    active_model_bytes = model.active_params * quant_bits / 8
    # Also need to read KV cache for attention (proportional to avg_decode_pos)
    kv_read_bytes = calc_kv_cache_gb(model, int(avg_decode_pos), batch_size, quant_bits) * 1e9
    # Without FlashAttention: extra HBM traffic for attention score materialization
    if not config.flash_attn:
        # Must write and re-read attention scores: 2 * batch * heads * avg_pos * 4 bytes per layer
        attn_io_bytes = 2 * batch_size * model.heads * avg_decode_pos * 4 * model.layers
        kv_read_bytes += attn_io_bytes
    total_read_bytes = active_model_bytes + kv_read_bytes

    decode_mem_bound_s = total_read_bytes / (effective_bw * 1e9)
    decode_compute_bound_s = decode_total_flops_per_step / (effective_tflops * 1e12)

    decode_time_per_step_s = max(decode_mem_bound_s, decode_compute_bound_s)
    decode_time_per_token_ms = (decode_time_per_step_s / batch_size) * 1000
    decode_tps = 1000 / decode_time_per_token_ms if decode_time_per_token_ms > 0 else 0
    decode_bottleneck = "memory" if decode_mem_bound_s >= decode_compute_bound_s else "compute"

    # ── Total ──
    total_decode_ms = decode_time_per_token_ms * output_len
    total_time_ms = prefill_time_ms + total_decode_ms
    total_output_tps = output_len / (total_time_ms / 1000) if total_time_ms > 0 else 0

    # ── Roofline ──
    ridge_point = (effective_tflops * 1e12) / (effective_bw * 1e9)  # FLOPs/Byte
    # Prefill arithmetic intensity: total_flops / bytes_read
    prefill_bytes = active_model_bytes  # weights read once (assuming layers streamed)
    prefill_ai = prefill_total_flops / prefill_bytes if prefill_bytes > 0 else 0
    # Decode arithmetic intensity
    decode_ai = decode_total_flops_per_step / total_read_bytes if total_read_bytes > 0 else 0

    return PerfResult(
        model_name=model.name,
        device_name=device.name,
        quant_bits=quant_bits,
        prompt_len=prompt_len,
        output_len=output_len,
        batch_size=batch_size,
        tp=tp,
        # Prefill
        prefill_linear_flops=prefill_linear_flops,
        prefill_attn_flops=prefill_attn_flops,
        prefill_total_flops=prefill_total_flops,
        prefill_time_ms=prefill_time_ms,
        prefill_tps=prefill_tps,
        # Decode
        decode_linear_flops=decode_linear_flops,
        decode_attn_flops=decode_attn_flops,
        decode_mem_bound_ms=decode_mem_bound_s * 1000,
        decode_compute_bound_ms=decode_compute_bound_s * 1000,
        decode_time_per_token_ms=decode_time_per_token_ms,
        decode_tps=decode_tps,
        decode_bottleneck=decode_bottleneck,
        # Total
        total_time_ms=total_time_ms,
        total_output_tps=total_output_tps,
        # Memory
        model_size_gb=model_size_gb,
        kv_cache_gb=kv_cache_gb,
        activation_gb=activation_gb,
        total_memory_gb=total_memory_gb,
        device_memory_gb=device_memory_gb,
        fits_in_memory=fits_in_memory,
        # Roofline
        ridge_point=ridge_point,
        prefill_arith_intensity=prefill_ai,
        decode_arith_intensity=decode_ai,
        # Device
        effective_tflops=effective_tflops,
        effective_bw_gbs=effective_bw,
        # FlashAttention
        flash_attn=config.flash_attn,
        attn_score_memory_gb=attn_score_memory_gb,
    )


def compare_devices(
    model_key: str,
    quant_bits: int = 4,
    prompt_len: int = 512,
    output_len: int = 256,
    batch_size: int = 1,
    tp: int = 1,
) -> list[dict]:
    """Compare performance across all devices for a given model config."""
    model = MODELS[model_key]
    results = []
    for device_key in DEVICES:
        try:
            r = estimate_performance(model_key, device_key, quant_bits, prompt_len, output_len, batch_size, tp)
            dev = DEVICES[device_key]
            results.append({
                "device": r.device_name,
                "device_key": device_key,
                "vendor": dev.vendor,
                "vram_gb": dev.vram,
                "bw_gbs": dev.bw,
                "fits": r.fits_in_memory,
                "prefill_ms": round(r.prefill_time_ms, 2),
                "decode_tps": round(r.decode_tps, 1),
                "total_ms": round(r.total_time_ms, 2),
                "memory_gb": round(r.total_memory_gb, 1),
                "bottleneck": r.decode_bottleneck,
            })
        except Exception:
            pass
    # Sort: fits first, then by decode throughput
    results.sort(key=lambda x: (not x["fits"], -x["decode_tps"]))
    return results


def compare_models(
    device_key: str,
    quant_bits: int = 4,
    prompt_len: int = 512,
    output_len: int = 256,
    batch_size: int = 1,
    tp: int = 1,
) -> list[dict]:
    """Compare all models on a given device."""
    results = []
    for model_key in MODELS:
        try:
            r = estimate_performance(model_key, device_key, quant_bits, prompt_len, output_len, batch_size, tp)
            m = MODELS[model_key]
            results.append({
                "model": r.model_name,
                "model_key": model_key,
                "arch": m.arch,
                "params_b": round(m.params / 1e9, 2),
                "active_b": round(m.active_params / 1e9, 2) if m.moe else None,
                "moe": m.moe,
                "fits": r.fits_in_memory,
                "prefill_ms": round(r.prefill_time_ms, 2),
                "decode_tps": round(r.decode_tps, 1),
                "total_ms": round(r.total_time_ms, 2),
                "memory_gb": round(r.total_memory_gb, 1),
                "bottleneck": r.decode_bottleneck,
            })
        except Exception:
            pass
    # Sort: fits first, then by decode throughput
    results.sort(key=lambda x: (not x["fits"], -x["decode_tps"]))
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _format_num(n: float) -> str:
    if n >= 1e15:
        return f"{n/1e15:.1f} PFLOP"
    if n >= 1e12:
        return f"{n/1e12:.1f} TFLOP"
    if n >= 1e9:
        return f"{n/1e9:.1f} GFLOP"
    return f"{n:.0f} FLOP"


def _format_time(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f} μs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.2f} s"


def _format_gb(gb: float) -> str:
    if gb < 1:
        return f"{gb*1024:.0f} MB"
    return f"{gb:.1f} GB"


def print_result(r: PerfResult):
    """Pretty-print a PerfResult to console."""
    print(f"\n{'='*60}")
    print(f"  {r.model_name}  on  {r.device_name}")
    print(f"  {r.quant_bits}-bit | Prompt {r.prompt_len} | Output {r.output_len} | BS {r.batch_size} | TP {r.tp}")
    print(f"{'='*60}")

    print(f"\n  📊 Effective Device Specs")
    print(f"     Compute:  {r.effective_tflops:.0f} TFLOPS")
    print(f"     Memory BW: {r.effective_bw_gbs:.0f} GB/s")
    print(f"     Ridge Point: {r.ridge_point:.0f} FLOPs/Byte")

    print(f"\n  ⚡ Prefill (TTFT)")
    print(f"     Linear FLOPs:    {_format_num(r.prefill_linear_flops)}")
    print(f"     Attention FLOPs: {_format_num(r.prefill_attn_flops)} ({r.prefill_attn_flops/r.prefill_total_flops*100:.1f}%)")
    print(f"     Total FLOPs:     {_format_num(r.prefill_total_flops)}")
    print(f"     Latency:         {_format_time(r.prefill_time_ms)}")
    print(f"     Throughput:      {r.prefill_tps:.0f} tok/s")
    print(f"     Arith Intensity: {r.prefill_arith_intensity:.0f} FLOPs/Byte")

    print(f"\n  🔄 Decode")
    print(f"     Mem-bound:       {_format_time(r.decode_mem_bound_ms)}/step")
    print(f"     Compute-bound:   {_format_time(r.decode_compute_bound_ms)}/step")
    print(f"     Bottleneck:      {'📦 Memory' if r.decode_bottleneck == 'memory' else '⚡ Compute'}")
    print(f"     Per Token:       {_format_time(r.decode_time_per_token_ms)}")
    print(f"     Throughput:      {r.decode_tps:.1f} tok/s")
    print(f"     Arith Intensity: {r.decode_arith_intensity:.1f} FLOPs/Byte")

    print(f"\n  ⏱️  Total")
    print(f"     Prefill + Decode: {_format_time(r.total_time_ms)}")
    print(f"     Output Throughput: {r.total_output_tps:.1f} tok/s")

    print(f"\n  💾 Memory")
    print(f"     Weights:     {_format_gb(r.model_size_gb)}")
    print(f"     KV Cache:    {_format_gb(r.kv_cache_gb)}")
    print(f"     Activation:  {_format_gb(r.activation_gb)}")
    print(f"     Total:       {_format_gb(r.total_memory_gb)} / {_format_gb(r.device_memory_gb)}")
    print(f"     Status:      {'✅ Fits' if r.fits_in_memory else '❌ OOM'}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Inference Performance Estimator (Roofline-based)",
        epilog="Models: LLaMA, Qwen2.5/3/3.5/3.6/3.8, Gemma1.1/2/3/3n/4, Mistral, DeepSeek, Phi, MiniMax. "
               "Devices: NVIDIA (Pascal-Blackwell), AMD (Vega-CDNA3), Apple Silicon.",
    )
    parser.add_argument("-m", "--model", default="llama-3-8b", choices=list(MODELS.keys()), help="Model key (see --list-models)")
    parser.add_argument("-d", "--device", default="rtx-4090", choices=list(DEVICES.keys()), help="Device key (see --list-devices)")
    parser.add_argument("-q", "--quant", type=int, default=4, choices=[2, 3, 4, 8, 16], help="Quantization bits (2/3/4/8/16)")
    parser.add_argument("-p", "--prompt", type=int, default=512, help="Prompt length (tokens)")
    parser.add_argument("-o", "--output", type=int, default=256, help="Output length (tokens)")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
    parser.add_argument("-t", "--tp", type=int, default=1, help="Tensor parallel GPUs")
    parser.add_argument("--compare-devices", action="store_true", help="Compare all devices for the given model")
    parser.add_argument("--compare-models", action="store_true", help="Compare all models on the given device")
    parser.add_argument("--list-models", action="store_true", help="List all available model keys")
    parser.add_argument("--list-devices", action="store_true", help="List all available device keys")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Models:")
        for k, m in MODELS.items():
            moe_str = f" (MoE {m.num_experts}x top-{m.top_k}, active {m.active_params/1e9:.1f}B)" if m.moe else ""
            hybrid_str = f" [hybrid-attn]" if m.softmax_attn_layers is not None else ""
            print(f"  {k:25s}  {m.name:30s}  {m.params/1e9:>7.2f}B{moe_str}{hybrid_str}")
        exit()

    if args.list_devices:
        print("\nAvailable Devices:")
        for k, d in DEVICES.items():
            fp8_str = f" FP8:{d.fp8_tflops}T" if d.fp8_tflops else ""
            print(f"  {k:18s}  {d.name:38s}  {d.vram:>4}GB  BW:{d.bw:>5}GB/s  FP16:{d.fp16_tflops:>6}T{fp8_str}")
        exit()

    if args.compare_devices:
        results = compare_devices(args.model, args.quant, args.prompt, args.output, args.batch, args.tp)
        m = MODELS[args.model]
        print(f"\n{m.name} on all devices (q={args.quant}, p={args.prompt}, o={args.output}, bs={args.batch}, tp={args.tp})")
        print(f"\n{'Device':<32} {'VRAM':>5} {'Fits':>5} {'Prefill':>10} {'Decode':>9} {'Total':>10} {'Mem':>8} {'Bottleneck':>12}")
        print("-" * 100)
        for r in results:
            fit = "✅" if r["fits"] else "❌"
            print(f"  {r['device']:<32} {r['vram_gb']:>4}GB {fit:>5} {_format_time(r['prefill_ms']):>10} {r['decode_tps']:>7.1f}t/s {_format_time(r['total_ms']):>10} {r['memory_gb']:>6.1f}GB {r['bottleneck']:>12}")
        exit()

    if args.compare_models:
        results = compare_models(args.device, args.quant, args.prompt, args.output, args.batch, args.tp)
        d = DEVICES[args.device]
        print(f"\nAll models on {d.name} (q={args.quant}, p={args.prompt}, o={args.output}, bs={args.batch}, tp={args.tp})")
        print(f"\n{'Model':<32} {'Params':>8} {'Fits':>5} {'Prefill':>10} {'Decode':>9} {'Total':>10} {'Mem':>8} {'Bottleneck':>12}")
        print("-" * 105)
        for r in results:
            fit = "✅" if r["fits"] else "❌"
            active = f"/{r['active_b']:.1f}B" if r["active_b"] else ""
            print(f"  {r['model']:<32} {r['params_b']:>6.2f}B{active:<6} {fit:>5} {_format_time(r['prefill_ms']):>10} {r['decode_tps']:>7.1f}t/s {_format_time(r['total_ms']):>10} {r['memory_gb']:>6.1f}GB {r['bottleneck']:>12}")
        exit()

    r = estimate_performance(args.model, args.device, args.quant, args.prompt, args.output, args.batch, args.tp)
    print_result(r)
