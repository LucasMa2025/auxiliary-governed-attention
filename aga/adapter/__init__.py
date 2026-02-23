"""
aga/adapter/ — LLM 适配器

提供:
  - LLMAdapter: 适配器抽象基类
  - HuggingFaceAdapter: HuggingFace Transformers 适配器
  - VLLMAdapter: vLLM 推理框架适配器
  - VLLMHookWorker: IBM vLLM-Hook 兼容的 AGA Worker
"""
from .base import LLMAdapter
from .huggingface import HuggingFaceAdapter
from .vllm import VLLMAdapter, VLLMHookWorker

__all__ = [
    "LLMAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "VLLMHookWorker",
]
