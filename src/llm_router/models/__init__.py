"""Model-specific entry points, imported lazily to avoid loading ML runtimes."""

from importlib import import_module

__all__ = [
    "FastDLLMV2OnePointFiveBInference",
    "QwenOnePointFiveBInference",
    "QwenSevenB4BitInference",
]


def __getattr__(name: str):
    modules = {
        "FastDLLMV2OnePointFiveBInference": ".fast_dllm_v2_1_5b_inference",
        "QwenOnePointFiveBInference": ".qwen2_5_1_5b_inference",
        "QwenSevenB4BitInference": ".qwen2_5_7b_4bit_inference",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
