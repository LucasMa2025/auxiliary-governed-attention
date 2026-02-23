"""
AGA — Auxiliary Governed Attention

极简注意力治理插件，为冻结 LLM 提供推理时知识注入能力。

3 行集成:
    from aga import AGAPlugin, AGAConfig

    plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
    plugin.attach(model)
    model.generate(input_ids)  # AGA 自动工作

配置驱动:
    plugin = AGAPlugin.from_config("aga_config.yaml")
    plugin.attach(model)

知识管理:
    plugin.register("fact_001", key=key_tensor, value=value_tensor)
    plugin.load_from("knowledge.jsonl")
    plugin.unregister("fact_001")

诊断:
    print(plugin.get_diagnostics())
    print(plugin.get_audit_trail())
"""

__version__ = "4.4.0"
__author__ = "AGA Team"

from .plugin import AGAPlugin
from .config import AGAConfig
from .kv_store import KVStore
from .streaming import StreamingSession
from .adapter.base import LLMAdapter
from .adapter.huggingface import HuggingFaceAdapter
from .adapter.vllm import VLLMAdapter, VLLMHookWorker
from .retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult
from .retriever.null_retriever import NullRetriever
from .retriever.kv_store_retriever import KVStoreRetriever
from .types import (
    KnowledgeEntry,
    GateDiagnostics,
    ForwardResult,
    PluginDiagnostics,
)
from .exceptions import (
    AGAError,
    AttachError,
    KVStoreError,
    ConfigError,
    GateError,
    AdapterError,
    RetrieverError,
)

__all__ = [
    # 核心
    "AGAPlugin",
    "AGAConfig",
    "KVStore",
    "StreamingSession",
    # 适配器
    "LLMAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "VLLMHookWorker",
    # 召回器
    "BaseRetriever",
    "RetrievalQuery",
    "RetrievalResult",
    "NullRetriever",
    "KVStoreRetriever",
    # 类型
    "KnowledgeEntry",
    "GateDiagnostics",
    "ForwardResult",
    "PluginDiagnostics",
    # 异常
    "AGAError",
    "AttachError",
    "KVStoreError",
    "ConfigError",
    "GateError",
    "AdapterError",
    "RetrieverError",
]
