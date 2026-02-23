"""
aga/plugin.py — AGAPlugin 完整设计

源码映射:
  - attach/detach: 来自 core.py AGAManager.attach_to_model() (第 1190-1335 行)
  - register: 来自 core.py AuxiliaryGovernedAttention.inject_knowledge() (第 718-780 行)
  - forward 逻辑: 来自 core.py AuxiliaryGovernedAttention.forward() (第 458-585 行)

这是 AGA 的唯一入口类，提供:
  - 3 行集成: plugin = AGAPlugin(config) → plugin.attach(model) → model.generate()
  - 知识管理: register / unregister / load_from / clear
  - 配置驱动: AGAPlugin.from_config("aga.yaml")
  - 诊断查询: get_diagnostics / get_audit_trail
  - 配置驱动知识检索: retriever 标准协议

v4.4 变更:
  - 新增 BaseRetriever 标准召回器协议（配置驱动知识检索）
  - _decay_contexts 改为 threading.local() 实现线程隔离
  - get_active() 缓存优化
  - StreamingSession 层事件过滤
"""
import json
import time
import logging
import threading
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

import torch
import torch.nn as nn

from .config import AGAConfig
from .kv_store import KVStore
from .gate.entropy_gate import EntropyGateSystem
from .gate.decay import PersistenceDecay, DecayContext
from .operator.bottleneck_injector import BottleneckInjector
from .adapter.base import LLMAdapter
from .adapter.huggingface import HuggingFaceAdapter
from .retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult
from .retriever.null_retriever import NullRetriever
from .instrumentation import EventBus, ForwardMetrics, AuditLog
from .exceptions import AGAError, AttachError

logger = logging.getLogger(__name__)


class AGAPlugin:
    """
    AGA 注意力治理插件 — 唯一入口类

    使用方式（3 行集成）:
        plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
        plugin.attach(model, layer_indices=[-1, -2, -3])
        output = model.generate(input_ids)  # AGA 自动介入

    知识管理（推理前）:
        plugin.register("fact_001", key=k, value=v)
        plugin.load_from("knowledge.jsonl")
        plugin.unregister("fact_001")

    配置驱动（生产部署）:
        plugin = AGAPlugin.from_config("aga_config.yaml")
    """

    def __init__(self, config: AGAConfig = None, retriever: Optional[BaseRetriever] = None, **kwargs):
        self.config = config or AGAConfig(**kwargs)
        self.device = torch.device(self.config.device)

        # 核心组件
        self.store = KVStore(
            max_slots=self.config.max_slots,
            key_dim=self.config.bottleneck_dim,
            value_dim=self.config.hidden_dim,
            device=self.device,
        )
        self.gate_system = EntropyGateSystem(self.config).to(self.device)
        self.injector = BottleneckInjector(
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            value_bottleneck_dim=self.config.value_bottleneck_dim,
            top_k=self.config.gate2_top_k,
        ).to(self.device)
        self.decay = PersistenceDecay(self.config).to(self.device) if self.config.decay_enabled else None

        # 跨层衰减上下文管理 — 使用 threading.local() 实现线程隔离
        # 每个线程/请求维护独立的 decay context，避免并发请求间的状态污染
        self._decay_local = threading.local()

        # 召回器（配置驱动知识检索）
        self.retriever: BaseRetriever = retriever or self._create_retriever()

        # 内置埋点层（零外部依赖，始终启用）
        self.event_bus = EventBus(
            buffer_size=self.config.event_buffer_size,
            enabled=self.config.instrumentation_enabled,
        )
        self.forward_metrics = ForwardMetrics(self.event_bus)
        self.audit_log = AuditLog(
            event_bus=self.event_bus,
            log_level=self.config.audit_log_level,
        )

        # 模型适配器（attach 时初始化）
        self._adapter: Optional[LLMAdapter] = None
        self._hooks: List[Any] = []
        self._attached = False
        self._attached_model_name: Optional[str] = None
        self._first_hooked_layer: int = 0  # 第一个挂载层（召回器触发层）

        # Slot 治理状态
        self._retrieval_step_counter: int = 0  # forward 步数计数器
        self._last_retrieval_step: int = -999  # 上次召回的步数
        self._slot_change_counter: int = 0  # slot 变化计数
        self._slot_change_window: int = 0  # 变化窗口（forward 次数）

        # 可选: aga-knowledge 集成
        self._knowledge_manager = None

        # 可选: aga-observability 集成（自动检测）
        self._setup_observability()

        logger.info(
            f"AGAPlugin 初始化完成: "
            f"hidden_dim={self.config.hidden_dim}, "
            f"bottleneck_dim={self.config.bottleneck_dim}, "
            f"max_slots={self.config.max_slots}, "
            f"retriever={self.retriever}, "
            f"device={self.device}"
        )

    @classmethod
    def from_config(cls, config_source: Union[str, Path, Dict]) -> "AGAPlugin":
        """
        从 YAML / Dict / Path 创建插件

        Args:
            config_source: YAML 文件路径、字典或 Path 对象

        Returns:
            AGAPlugin 实例
        """
        if isinstance(config_source, dict):
            config = AGAConfig.from_dict(config_source)
        elif isinstance(config_source, (str, Path)):
            config = AGAConfig.from_yaml(str(config_source))
        else:
            raise ValueError(f"不支持的配置源类型: {type(config_source)}")

        plugin = cls(config)  # retriever 由 _create_retriever() 根据配置自动创建

        # 如果配置了 knowledge_sources，自动加载
        if config.knowledge_sources:
            plugin._auto_load_knowledge(config.knowledge_sources)

        return plugin

    # ========== 知识管理（管理面） ==========

    def register(
        self,
        id: str,
        key: torch.Tensor,
        value: torch.Tensor,
        reliability: float = 1.0,
        metadata: Optional[Dict] = None,
        pinned: Optional[bool] = None,
    ) -> bool:
        """
        注册单条知识到 GPU 常驻 KVStore

        这不是"注入"——知识只是被存储在 KVStore 中。
        真正的注入发生在推理时，由 AGA 的熵门控自动决定。

        Args:
            id: 知识唯一标识
            key: 检索键向量 [bottleneck_dim]
            value: 知识值向量 [hidden_dim]
            reliability: 可靠性分数 (0.0-1.0)
            metadata: 可选元数据
            pinned: 是否锁定（None=使用配置 pin_registered 的默认值）

        Returns:
            是否注册成功
        """
        # 范数裁剪
        if self.config.enable_norm_clipping:
            key = self._clip_norm(key, self.config.key_norm_target)
            value = self._clip_norm(value, self.config.value_norm_target)

        # 确定是否锁定
        should_pin = pinned if pinned is not None else self.config.pin_registered

        # 标记来源
        meta = dict(metadata) if metadata else {}
        meta.setdefault("source", "register")

        success = self.store.put(id, key, value, reliability, meta, pinned=should_pin)
        if success:
            self.audit_log.record("register", {
                "id": id,
                "reliability": reliability,
                "pinned": should_pin,
                "metadata": meta,
                "store_count": self.store.count,
            })
        return success

    def register_batch(self, entries: List[Dict]) -> int:
        """
        批量注册知识

        Args:
            entries: 知识条目列表，每个条目包含 id, key, value, reliability(可选), metadata(可选)

        Returns:
            成功注册的数量
        """
        count = 0
        for entry in entries:
            if self.register(
                id=entry["id"],
                key=entry["key"],
                value=entry["value"],
                reliability=entry.get("reliability", 1.0),
                metadata=entry.get("metadata"),
            ):
                count += 1

        self.audit_log.record("register_batch", {
            "total": len(entries),
            "success": count,
            "store_count": self.store.count,
        })
        return count

    def unregister(self, id: str) -> bool:
        """
        移除知识

        Args:
            id: 知识唯一标识

        Returns:
            是否移除成功
        """
        success = self.store.remove(id)
        if success:
            self.audit_log.record("unregister", {
                "id": id,
                "store_count": self.store.count,
            })
        return success

    def load_from(self, source: str, **kwargs) -> int:
        """
        从外部数据源加载知识

        支持的数据源类型:
          - "knowledge.jsonl" → JSONL 文件（内置）
          - 其他格式需要 aga-knowledge 包

        Args:
            source: 数据源路径或 URI

        Returns:
            加载的知识数量
        """
        if source.endswith(".jsonl"):
            count = self._load_from_jsonl(source)
            self.audit_log.record("load_from", {
                "source": source,
                "type": "jsonl",
                "count": count,
            })
            return count

        # 尝试使用 aga-knowledge 的配置适配器
        try:
            from aga_knowledge.config_adapter import KnowledgeSource
            ks = KnowledgeSource.from_uri(source, **kwargs)
            entries = ks.load()
            count = self.register_batch(entries)
            self.audit_log.record("load_from", {
                "source": source,
                "type": "aga-knowledge",
                "count": count,
            })
            return count
        except ImportError:
            raise ImportError(
                f"加载 '{source}' 需要 aga-knowledge 包。"
                f"请运行: pip install aga-knowledge"
            )

    def clear(self, namespace: Optional[str] = None):
        """
        清空知识

        Args:
            namespace: 如果指定，只清空该命名空间
        """
        self.store.clear(namespace)
        self.audit_log.record("clear", {
            "namespace": namespace,
            "store_count": self.store.count,
        })

    # ========== 模型集成 ==========

    def attach(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        adapter: Optional[LLMAdapter] = None,
    ):
        """
        挂载到模型（之后推理自动介入）

        Args:
            model: HuggingFace 模型实例
            layer_indices: 要挂载的层索引（负数表示从后往前）
                          默认: [-1, -2, -3]（最后 3 层）
            adapter: 自定义 LLM 适配器（默认自动检测）
        """
        if self._attached:
            raise AttachError("AGAPlugin 已挂载，请先 detach()")

        # 自动检测适配器
        self._adapter = adapter or HuggingFaceAdapter()
        layers = self._adapter.get_layers(model)

        if layer_indices is None:
            layer_indices = [-1, -2, -3]

        # 将负索引转换为正索引
        resolved = [i if i >= 0 else len(layers) + i for i in layer_indices]

        # 过滤有效索引
        valid_indices = [i for i in resolved if 0 <= i < len(layers)]

        if not valid_indices:
            raise AttachError(
                f"没有有效的层索引。模型共 {len(layers)} 层，"
                f"请求的索引: {layer_indices} → {resolved}"
            )

        # 记录第一个挂载层（召回器只在此层触发，避免多层重复检索）
        self._first_hooked_layer = min(valid_indices)

        # 为每个目标层注册 forward hook
        for layer_idx in valid_indices:
            hook = self._adapter.wrap_layer(
                model=model,
                layer_idx=layer_idx,
                aga_forward=self._create_layer_forward(layer_idx),
            )
            self._hooks.append(hook)

        self._attached = True
        self._attached_model_name = type(model).__name__

        # 预热召回器
        try:
            self.retriever.warmup()
        except Exception as e:
            logger.warning(f"召回器预热失败 (Fail-Open): {e}")

        self.audit_log.record("attach", {
            "model_type": self._attached_model_name,
            "layers": valid_indices,
            "hooks": len(self._hooks),
            "knowledge_count": self.store.count,
        })
        logger.info(
            f"AGA 已挂载到 {self._attached_model_name} 的 {len(self._hooks)} 层: "
            f"{valid_indices}, 知识槽位: {self.store.count}/{self.config.max_slots}"
        )

    def detach(self):
        """从模型卸载"""
        hooks_count = len(self._hooks)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attached = False

        # 关闭召回器
        try:
            self.retriever.shutdown()
        except Exception as e:
            logger.warning(f"召回器关闭失败: {e}")

        self.audit_log.record("detach", {
            "model_type": self._attached_model_name,
            "hooks_removed": hooks_count,
        })
        self._attached_model_name = None
        logger.info(f"AGA 已卸载，移除了 {hooks_count} 个 hooks")

    # ========== 流式生成 ==========

    def create_streaming_session(self, **kwargs) -> "StreamingSession":
        """
        创建流式生成会话

        在 LLM 的自回归生成过程中，AGA 通过 forward hook 自动在每个
        decode step 中评估和注入知识。StreamingSession 提供会话级管理：
        - 自动重置衰减上下文
        - 逐 token 实时诊断
        - 动态知识热更新
        - 会话统计摘要

        Args:
            **kwargs: 传递给 StreamingSession 的参数
                diagnostics_buffer_size: 诊断缓冲区大小（默认 1000）

        Returns:
            StreamingSession 实例

        使用方式:
            session = plugin.create_streaming_session()
            for step in generation:
                output = model.forward(token)
                diag = session.get_step_diagnostics()
            summary = session.get_session_summary()
            session.close()

            # 或使用 with 语句
            with plugin.create_streaming_session() as session:
                for step in generation:
                    output = model.forward(token)
                    diag = session.get_step_diagnostics()
        """
        from .streaming import StreamingSession
        return StreamingSession(plugin=self, **kwargs)

    # ========== 状态查询 ==========

    @property
    def knowledge_count(self) -> int:
        """当前知识数量"""
        return self.store.count

    @property
    def is_attached(self) -> bool:
        """是否已挂载"""
        return self._attached

    def get_diagnostics(self) -> Dict:
        """
        获取诊断信息（始终可用，不依赖外部包）

        Returns:
            包含运行状态、指标摘要的字典
        """
        result = {
            "attached": self._attached,
            "model_type": self._attached_model_name,
            "knowledge_count": self.store.count,
            "max_slots": self.config.max_slots,
            "utilization": self.store.utilization,
            "pinned_count": self.store.pinned_count,
            "unpinned_count": self.store.unpinned_count,
            "hooked_layers": len(self._hooks),
            "device": str(self.device),
            "retriever": repr(self.retriever),
            "retriever_stats": self.retriever.get_stats(),
            "slot_governance": {
                "retrieval_step_counter": self._retrieval_step_counter,
                "last_retrieval_step": self._last_retrieval_step,
                "slot_change_rate": (
                    self._slot_change_counter / max(self._slot_change_window, 1)
                ),
                "retriever_budget": self._compute_retriever_budget() if not isinstance(self.retriever, NullRetriever) else None,
            },
        }
        result.update(self.forward_metrics.get_summary())
        return result

    def get_audit_trail(self, limit: int = 100, operation: str = None) -> List[Dict]:
        """
        获取审计日志（始终可用，不依赖外部包）

        Args:
            limit: 返回数量
            operation: 过滤操作类型

        Returns:
            审计条目列表
        """
        return self.audit_log.query(limit=limit, operation=operation)

    def get_store_stats(self) -> Dict:
        """获取 KV 存储统计"""
        return self.store.get_stats()

    # ========== 内部方法 ==========

    def _create_layer_forward(self, layer_idx: int):
        """为指定层创建 AGA forward 函数"""

        def aga_forward(
            hidden_states: torch.Tensor,
            primary_attention_output: torch.Tensor,
            attention_weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            AGA 核心 forward — 每个 token 的每个挂载层都会调用

            技术依据: core.py 第 458-585 行 AuxiliaryGovernedAttention.forward()

            流程:
              1. 熵门控 → 判断是否需要介入
              2. 高熵时调用召回器 → 从外部知识库检索相关知识
              3. 召回结果自动注入 KVStore
              4. BottleneckInjector 在 KVStore 上执行注意力计算
              5. 门控融合 → 将辅助输出与原始输出融合
            """
            start_time = time.perf_counter()

            try:
                # 0. 检查是否有活跃知识（或有召回器可以获取知识）
                has_retriever = not isinstance(self.retriever, NullRetriever)
                if self.store.count == 0 and not has_retriever:
                    return primary_attention_output

                # 1. 熵门控（三段式）— 先判断是否需要介入
                gate, gate_diagnostics = self.gate_system(
                    hidden_states, layer_idx=layer_idx
                )

                # 步数计数（用于召回冷却期，在第一个挂载层计数）
                if layer_idx == self._first_hooked_layer:
                    self._retrieval_step_counter += 1

                # Early Exit: 低熵旁路
                gate_mean = gate_diagnostics.gate_mean
                if gate_diagnostics.early_exit or gate_mean < self.config.early_exit_threshold:
                    latency_us = (time.perf_counter() - start_time) * 1_000_000
                    self.forward_metrics.record(
                        aga_applied=False,
                        gate_mean=gate_mean,
                        entropy_mean=gate_diagnostics.entropy_mean,
                        layer_idx=layer_idx,
                        latency_us=latency_us,
                    )
                    return primary_attention_output

                # 2. 高熵触发 → 调用召回器检索外部知识
                #    经过 Slot 治理守卫（冷却期、预算、稳定性）
                if has_retriever and self._should_retrieve(gate_diagnostics, layer_idx):
                    query_projected = self.injector.q_proj(hidden_states)
                    self._retrieve_and_inject(
                        hidden_states=hidden_states,
                        query_projected=query_projected,
                        entropy=torch.tensor(gate_diagnostics.entropy_mean),
                        layer_idx=layer_idx,
                    )

                # 3. 获取活跃的 KV（可能包含刚召回注入的知识）
                keys, values, reliability = self.store.get_active()
                if keys.shape[0] == 0:
                    latency_us = (time.perf_counter() - start_time) * 1_000_000
                    self.forward_metrics.record(
                        aga_applied=False,
                        gate_mean=gate_mean,
                        entropy_mean=gate_diagnostics.entropy_mean,
                        layer_idx=layer_idx,
                        latency_us=latency_us,
                    )
                    return primary_attention_output

                # 确保数据类型匹配
                compute_dtype = hidden_states.dtype
                keys = keys.to(dtype=compute_dtype)
                values = values.to(dtype=compute_dtype)
                reliability = reliability.to(dtype=compute_dtype)

                # 4. Bottleneck 注入
                aux_output = self.injector(hidden_states, keys, values, reliability)

                # 5. 跨层衰减（使用线程隔离的上下文）
                if self.decay is not None:
                    decay_contexts = self._get_decay_contexts()
                    context = decay_contexts.get(layer_idx)
                    gate, context = self.decay(gate, context, layer_idx)
                    decay_contexts[layer_idx] = context

                # 6. 门控融合
                fused = primary_attention_output + gate.unsqueeze(-1) * aux_output

                # 7. 埋点
                latency_us = (time.perf_counter() - start_time) * 1_000_000
                self.forward_metrics.record(
                    aga_applied=True,
                    gate_mean=gate_mean,
                    entropy_mean=gate_diagnostics.entropy_mean,
                    layer_idx=layer_idx,
                    latency_us=latency_us,
                )

                return fused

            except Exception as e:
                # Fail-Open: 任何错误都回退到原始输出
                if self.config.fail_open:
                    logger.warning(
                        f"AGA forward 异常 (layer {layer_idx})，Fail-Open 回退: {e}",
                        exc_info=True,
                    )
                    return primary_attention_output
                raise AGAError(f"AGA forward 失败: {e}") from e

        return aga_forward

    def _setup_observability(self):
        """自动检测并集成 aga-observability（如果已安装）"""
        if not self.config.observability_enabled:
            return

        try:
            from aga_observability import setup_observability
            setup_observability(self.event_bus, self.config)
            logger.info("aga-observability 已自动集成")
        except ImportError:
            logger.debug("aga-observability 未安装，使用内置埋点")

    def _load_from_jsonl(self, path: str) -> int:
        """从 JSONL 文件加载知识（内置，不依赖 aga-knowledge）"""
        count = 0
        filepath = Path(path)
        if not filepath.exists():
            logger.warning(f"JSONL 文件不存在: {path}")
            return 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = torch.tensor(
                        entry["key"], dtype=torch.float16, device=self.device
                    )
                    value = torch.tensor(
                        entry["value"], dtype=torch.float16, device=self.device
                    )
                    if self.register(
                        id=entry["id"],
                        key=key,
                        value=value,
                        reliability=entry.get("reliability", 1.0),
                        metadata=entry.get("metadata"),
                    ):
                        count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"JSONL 第 {line_num} 行解析失败: {e}")

        logger.info(f"从 {path} 加载了 {count} 条知识")
        return count

    def _auto_load_knowledge(self, sources: List[Dict]):
        """从配置的 knowledge_sources 自动加载"""
        total = 0
        for source_config in sources:
            source_type = source_config.get("type", "")
            try:
                if source_type == "jsonl":
                    total += self._load_from_jsonl(source_config["path"])
                else:
                    from aga_knowledge.config_adapter import KnowledgeSource
                    ks = KnowledgeSource.from_config(source_config)
                    entries = ks.load()
                    total += self.register_batch(entries)
            except Exception as e:
                logger.error(f"加载知识源 {source_type} 失败: {e}")
        logger.info(f"自动加载完成，共 {total} 条知识")

    def _create_retriever(self) -> BaseRetriever:
        """根据配置创建召回器"""
        backend = self.config.retriever_backend

        if backend == "null" or not backend:
            return NullRetriever()

        if backend == "kv_store":
            from .retriever.kv_store_retriever import KVStoreRetriever
            return KVStoreRetriever(
                kv_store=self.store,
                default_top_k=self.config.retriever_top_k,
                min_similarity=self.config.retriever_min_score,
            )

        # 外部召回器：通过 entry_points 或动态导入
        try:
            # 尝试从 aga_knowledge 加载
            if backend == "chroma":
                from aga_knowledge.retrievers.chroma import ChromaRetriever
                return ChromaRetriever(
                    endpoint=self.config.retriever_endpoint,
                    collection=self.config.retriever_collection,
                    top_k=self.config.retriever_top_k,
                    **self.config.retriever_options,
                )
            elif backend == "milvus":
                from aga_knowledge.retrievers.milvus import MilvusRetriever
                return MilvusRetriever(
                    endpoint=self.config.retriever_endpoint,
                    collection=self.config.retriever_collection,
                    top_k=self.config.retriever_top_k,
                    **self.config.retriever_options,
                )
            elif backend == "elasticsearch":
                from aga_knowledge.retrievers.elasticsearch import ElasticsearchRetriever
                return ElasticsearchRetriever(
                    endpoint=self.config.retriever_endpoint,
                    index=self.config.retriever_collection,
                    top_k=self.config.retriever_top_k,
                    **self.config.retriever_options,
                )
            else:
                # 自定义后端：尝试动态导入
                # 格式: "mypackage.module:ClassName"
                if ":" in backend:
                    module_path, class_name = backend.rsplit(":", 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    retriever_cls = getattr(module, class_name)
                    return retriever_cls(
                        top_k=self.config.retriever_top_k,
                        **self.config.retriever_options,
                    )
                else:
                    logger.warning(
                        f"未知的召回器后端 '{backend}'，回退到 NullRetriever。"
                        f"支持的后端: null, kv_store, chroma, milvus, elasticsearch, "
                        f"或自定义 'module:ClassName' 格式"
                    )
                    return NullRetriever()
        except ImportError as e:
            logger.warning(
                f"召回器后端 '{backend}' 加载失败: {e}。"
                f"回退到 NullRetriever（Fail-Open）"
            )
            return NullRetriever()

    def _get_decay_contexts(self) -> Dict[int, DecayContext]:
        """
        获取当前线程的衰减上下文（线程隔离）

        每个线程维护独立的 decay context 字典，
        避免并发请求间的状态污染。
        """
        if not hasattr(self._decay_local, 'contexts'):
            self._decay_local.contexts = {}
        return self._decay_local.contexts

    def _should_retrieve(self, gate_diagnostics, layer_idx: int) -> bool:
        """
        判断是否应该调用召回器（Slot 治理守卫）

        检查条件:
          1. 只在第一个挂载层触发
          2. 冷却期内不召回
          3. KVStore 高利用率时提高熵阈值
          4. 稳定性检测（变化率过高时暂停）
        """
        # 条件 1: 只在第一个挂载层触发
        if layer_idx != self._first_hooked_layer:
            return False

        # 条件 2: 冷却期
        steps_since = self._retrieval_step_counter - self._last_retrieval_step
        if steps_since < self.config.retriever_cooldown_steps:
            return False

        # 条件 3: KVStore 高利用率时，只有极高熵才召回
        if self.store.utilization > 0.9:
            if gate_diagnostics.entropy_mean < self.config.tau_high * 0.8:
                return False

        # 条件 4: 稳定性检测
        if self._slot_change_window > 0:
            change_rate = self._slot_change_counter / self._slot_change_window
            if change_rate > self.config.slot_stability_threshold:
                logger.debug(
                    f"KVStore 变化率过高 ({change_rate:.1%})，暂停召回以稳定注入"
                )
                # 重置窗口
                self._slot_change_counter = 0
                self._slot_change_window = 0
                return False

        return True

    def _compute_retriever_budget(self) -> int:
        """
        计算召回器可用的 slot 预算

        预算 = min(显式预算, ratio * max_slots) - 已占用的召回器 slot 数
        """
        if self.config.retriever_slot_budget > 0:
            total_budget = self.config.retriever_slot_budget
        else:
            total_budget = int(self.config.max_slots * self.config.retriever_slot_ratio)

        # 统计当前召回器已占用的 slot 数
        retriever_slots = 0
        for kid in self.store.get_all_ids():
            meta = self.store.get_metadata(kid)
            if meta and meta.get("source") == "retriever":
                retriever_slots += 1

        return max(0, total_budget - retriever_slots)

    def _semantic_dedup(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        语义去重 — 过滤与 KVStore 中已有知识高度相似的召回结果

        步骤:
          1. 跳过已存在的知识（ID 去重）
          2. 与 KVStore 中已有 key 做余弦相似度检查
          3. 相似度 > threshold 的结果被过滤
        """
        import torch.nn.functional as F

        threshold = self.config.retriever_dedup_similarity
        deduped = []

        # 获取已有的 keys
        existing_keys, _, _ = self.store.get_active()
        has_existing = existing_keys.shape[0] > 0

        for result in results:
            # ID 去重
            if self.store.contains(result.id):
                continue

            # 语义去重
            if has_existing and threshold < 1.0:
                sim = F.cosine_similarity(
                    result.key.unsqueeze(0).float().to(existing_keys.device),
                    existing_keys.float(),
                    dim=-1,
                )
                if sim.max().item() > threshold:
                    continue

            deduped.append(result)

        return deduped

    def _retrieve_and_inject(
        self,
        hidden_states: torch.Tensor,
        query_projected: torch.Tensor,
        entropy: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """
        调用召回器检索知识并注入 KVStore（含 Slot 治理）

        仅在高熵触发时调用（由 aga_forward 中的门控逻辑决定）。
        召回结果经过 Slot 治理层后注入 KVStore，供 BottleneckInjector 使用。

        Slot 治理流程:
          1. 预算检查 → 召回器 slot 预算是否充足
          2. 召回 → 调用外部召回器
          3. 去重 → ID 去重 + 语义去重
          4. 优先级排序 → 按 score × reliability 排序
          5. 预算裁剪 → 只注入预算内的结果
          6. 写入 KVStore → 标记 source="retriever"（可被淘汰，不 pin）
          7. 记录变化 → 更新稳定性监控

        Args:
            hidden_states: 当前层的 hidden_states
            query_projected: q_proj 投影后的查询
            entropy: 当前位置的熵值
            layer_idx: 当前层索引
        """
        if isinstance(self.retriever, NullRetriever):
            return

        try:
            # 1. 预算检查
            budget = self._compute_retriever_budget()
            if budget <= 0:
                logger.debug("召回器 slot 预算已耗尽，跳过召回")
                return

            # 2. 召回
            effective_top_k = min(self.config.retriever_top_k, budget)
            query = RetrievalQuery(
                hidden_states=hidden_states.detach(),
                query_projected=query_projected.detach() if query_projected is not None else None,
                entropy=entropy.detach() if entropy is not None else None,
                layer_idx=layer_idx,
                top_k=effective_top_k,
            )

            results = self.retriever.retrieve(query)
            if not results:
                return

            # 3. 去重（ID + 语义）
            results = self._semantic_dedup(results)
            if not results:
                return

            # 4. 优先级排序（score × reliability 降序）
            results.sort(key=lambda r: r.score * r.reliability, reverse=True)

            # 5. 预算裁剪
            results = results[:budget]

            # 6. 写入 KVStore（标记 source="retriever"，不 pin）
            if self.config.retriever_auto_inject:
                before_count = self.store.count
                injected = 0
                for result in results:
                    meta = dict(result.metadata) if result.metadata else {}
                    meta["source"] = "retriever"
                    if self.store.put(
                        id=result.id,
                        key=result.key,
                        value=result.value,
                        reliability=result.reliability,
                        metadata=meta,
                        pinned=False,  # 召回知识不锁定，可被 LRU 淘汰
                    ):
                        injected += 1

                # 7. 记录变化（稳定性监控）
                after_count = self.store.count
                self._slot_change_counter += abs(after_count - before_count) + injected
                self._slot_change_window += 1
                self._last_retrieval_step = self._retrieval_step_counter

                self.event_bus.emit("retrieval", {
                    "layer_idx": layer_idx,
                    "results_count": len(results),
                    "injected_count": injected,
                    "budget_remaining": budget - injected,
                    "ids": [r.id for r in results[:injected]],
                    "scores": [r.score for r in results[:injected]],
                })

        except Exception as e:
            # Fail-Open: 召回失败不影响推理
            logger.warning(f"召回器调用失败 (Fail-Open): {e}", exc_info=True)

    @staticmethod
    def _clip_norm(tensor: torch.Tensor, target_norm: float) -> torch.Tensor:
        """范数裁剪"""
        norm = tensor.norm()
        if norm > 0 and norm != target_norm:
            tensor = tensor / (norm + 1e-8) * target_norm
        return tensor

    def reset_decay_contexts(self):
        """重置当前线程的衰减上下文（新推理请求时调用）"""
        if hasattr(self._decay_local, 'contexts'):
            self._decay_local.contexts.clear()

    def __repr__(self) -> str:
        return (
            f"AGAPlugin("
            f"hidden_dim={self.config.hidden_dim}, "
            f"slots={self.store.count}/{self.config.max_slots}, "
            f"attached={self._attached}, "
            f"device={self.device})"
        )
