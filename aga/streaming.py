"""
aga/streaming.py — StreamingSession 流式生成会话管理

设计要点:
  - 为 LLM 的自回归（token-by-token）生成过程提供会话级管理
  - 自动管理衰减上下文（每个新会话重置）
  - 提供逐 token 的实时诊断信息
  - 支持生成过程中的动态知识热更新
  - 会话结束时提供完整统计摘要

技术说明:
  AGA 的核心注入逻辑通过 register_forward_hook 自动工作，
  不需要特殊的流式 API。StreamingSession 提供的是会话管理
  和诊断能力的封装，使流式生成场景更易于使用和监控。

使用方式:
    session = plugin.create_streaming_session()
    for step in generation:
        output = model.forward(token)
        diag = session.get_step_diagnostics()
    summary = session.get_session_summary()
    session.close()
"""
import time
import logging
from typing import Dict, List, Optional, Any, Set
from collections import deque

import torch

logger = logging.getLogger(__name__)


class StreamingSession:
    """
    流式生成会话

    管理一次完整的自回归生成过程中的 AGA 状态，包括：
    - 衰减上下文的自动重置和管理
    - 逐步诊断信息的收集
    - 动态知识热更新
    - 会话级统计摘要

    注意: StreamingSession 不直接控制 AGA 的注入逻辑。
    AGA 的注入通过 forward hook 自动工作。
    StreamingSession 是一个观察者和管理者。
    """

    def __init__(
        self,
        plugin: "AGAPlugin",
        diagnostics_buffer_size: int = 1000,
        primary_layer_idx: Optional[int] = None,
    ):
        """
        初始化流式会话

        Args:
            plugin: AGAPlugin 实例
            diagnostics_buffer_size: 诊断缓冲区大小（保留最近 N 步的诊断）
            primary_layer_idx: 主监控层索引（只统计该层的事件，避免多层重复计数）
                              如果为 None，自动选择最后一个挂载层
        """
        self._plugin = plugin
        self._buffer_size = diagnostics_buffer_size

        # 确定主监控层（避免多层 hook 导致重复计数）
        if primary_layer_idx is not None:
            self._primary_layer_idx = primary_layer_idx
        elif plugin._hooks:
            # 默认选择最后一个挂载层（通常是最深层，信息最丰富）
            self._primary_layer_idx = None  # 延迟确定
            self._auto_detect_primary = True
        else:
            self._primary_layer_idx = None
            self._auto_detect_primary = False

        # 所有层的事件（用于详细诊断）
        self._all_layer_events: Dict[int, deque] = {}

        # 会话状态
        self._active = True
        self._start_time = time.time()
        self._step_count = 0

        # 诊断缓冲区（环形缓冲，只记录主监控层）
        self._step_diagnostics: deque = deque(maxlen=diagnostics_buffer_size)

        # 累计统计（只统计主监控层，避免重复计数）
        self._total_injection_count = 0
        self._total_gate_sum = 0.0
        self._total_entropy_sum = 0.0
        self._total_latency_sum = 0.0

        # 注册 forward 事件监听器
        self._subscriber_id = f"streaming_session_{id(self)}"
        self._plugin.event_bus.subscribe(
            "forward", self._on_forward_event, subscriber_id=self._subscriber_id
        )

        # 重置衰减上下文（新会话开始）
        self._plugin.reset_decay_contexts()

        # 审计记录
        self._plugin.audit_log.record("streaming_session_start", {
            "buffer_size": diagnostics_buffer_size,
            "knowledge_count": self._plugin.knowledge_count,
        })

        logger.info(
            f"StreamingSession 已创建: "
            f"knowledge={self._plugin.knowledge_count}, "
            f"buffer_size={diagnostics_buffer_size}"
        )

    def _on_forward_event(self, event):
        """
        处理 forward 事件（由 EventBus 回调）

        每次 AGA 的 forward 被调用时，ForwardMetrics 会发射一个 forward 事件。
        由于 AGA 可能挂载在多个层上，每个 decode step 会产生多个事件。
        StreamingSession 通过 primary_layer_idx 过滤，只对主监控层计数，
        避免 step_count 和 injection_count 被多层重复计算。

        所有层的事件仍然被记录在 _all_layer_events 中，供详细诊断使用。
        """
        if not self._active:
            return

        data = event.data if hasattr(event, 'data') else event
        if isinstance(data, dict):
            aga_applied = data.get("aga_applied", False)
            gate_mean = data.get("gate_mean", 0.0)
            entropy_mean = data.get("entropy_mean", 0.0)
            latency_us = data.get("latency_us", 0.0)
            layer_idx = data.get("layer_idx", -1)

            # 记录所有层的事件（详细诊断）
            if layer_idx not in self._all_layer_events:
                self._all_layer_events[layer_idx] = deque(maxlen=self._buffer_size)
            self._all_layer_events[layer_idx].append({
                "aga_applied": aga_applied,
                "gate_mean": gate_mean,
                "entropy_mean": entropy_mean,
                "latency_us": latency_us,
                "timestamp": time.time(),
            })

            # 自动检测主监控层（第一个收到的事件的最大 layer_idx）
            if hasattr(self, '_auto_detect_primary') and self._auto_detect_primary:
                if self._primary_layer_idx is None or layer_idx > self._primary_layer_idx:
                    self._primary_layer_idx = layer_idx

            # 只对主监控层计数（避免多层重复）
            is_primary = (
                self._primary_layer_idx is None  # 未设置主层时全部计数
                or layer_idx == self._primary_layer_idx
            )

            if is_primary:
                self._step_count += 1

                step_diag = {
                    "step": self._step_count,
                    "aga_applied": aga_applied,
                    "gate_mean": gate_mean,
                    "entropy_mean": entropy_mean,
                    "latency_us": latency_us,
                    "layer_idx": layer_idx,
                    "timestamp": time.time(),
                }
                self._step_diagnostics.append(step_diag)

                # 累计统计（只统计主监控层）
                if aga_applied:
                    self._total_injection_count += 1
                self._total_gate_sum += gate_mean
                self._total_entropy_sum += entropy_mean
                self._total_latency_sum += latency_us

    def get_step_diagnostics(self) -> Dict[str, Any]:
        """
        获取最近一步的诊断信息

        Returns:
            包含 step, aga_applied, gate_mean, entropy_mean, latency_us, layer_idx 的字典。
            如果没有诊断数据，返回空步骤信息。
        """
        if self._step_diagnostics:
            return dict(self._step_diagnostics[-1])
        return {
            "step": 0,
            "aga_applied": False,
            "gate_mean": 0.0,
            "entropy_mean": 0.0,
            "latency_us": 0.0,
            "layer_idx": -1,
            "timestamp": time.time(),
        }

    def get_recent_diagnostics(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近 N 步的诊断信息

        Args:
            n: 返回的步数

        Returns:
            诊断信息列表（最新的在最后）
        """
        items = list(self._step_diagnostics)
        return [dict(d) for d in items[-n:]]

    def get_session_summary(self) -> Dict[str, Any]:
        """
        获取会话统计摘要

        Returns:
            包含总步数、注入次数、注入率、平均门控值、平均熵值等的字典
        """
        elapsed = time.time() - self._start_time
        total = max(self._step_count, 1)

        return {
            "total_steps": self._step_count,
            "injection_count": self._total_injection_count,
            "injection_rate": self._total_injection_count / total,
            "bypass_count": self._step_count - self._total_injection_count,
            "bypass_rate": (self._step_count - self._total_injection_count) / total,
            "avg_gate_mean": self._total_gate_sum / total,
            "avg_entropy_mean": self._total_entropy_sum / total,
            "avg_latency_us": self._total_latency_sum / total,
            "total_latency_us": self._total_latency_sum,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self._step_count / max(elapsed, 1e-6),
            "knowledge_count": self._plugin.knowledge_count,
            "active": self._active,
        }

    def get_all_layer_diagnostics(self) -> Dict[int, List[Dict]]:
        """
        获取所有层的诊断信息（详细模式）

        与 get_step_diagnostics 不同，此方法返回所有挂载层的事件，
        用于分析各层的注入行为差异。

        Returns:
            {layer_idx: [event_dict, ...]}
        """
        return {
            layer_idx: [dict(e) for e in events]
            for layer_idx, events in self._all_layer_events.items()
        }

    def update_knowledge(
        self,
        id: str,
        key: torch.Tensor,
        value: torch.Tensor,
        reliability: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        在流式生成过程中动态更新知识

        Args:
            id: 知识唯一标识
            key: 检索键向量
            value: 知识值向量
            reliability: 可靠性分数
            metadata: 可选元数据

        Returns:
            是否更新成功
        """
        if not self._active:
            logger.warning("StreamingSession 已关闭，无法更新知识")
            return False

        success = self._plugin.register(
            id=id, key=key, value=value,
            reliability=reliability, metadata=metadata,
        )
        if success:
            self._plugin.audit_log.record("streaming_knowledge_update", {
                "id": id,
                "reliability": reliability,
                "step": self._step_count,
            })
        return success

    def remove_knowledge(self, id: str) -> bool:
        """
        在流式生成过程中移除知识

        Args:
            id: 知识唯一标识

        Returns:
            是否移除成功
        """
        if not self._active:
            logger.warning("StreamingSession 已关闭，无法移除知识")
            return False

        success = self._plugin.unregister(id)
        if success:
            self._plugin.audit_log.record("streaming_knowledge_remove", {
                "id": id,
                "step": self._step_count,
            })
        return success

    def reset_decay(self):
        """
        手动重置衰减上下文

        在某些场景下（如生成中途切换话题），可能需要手动重置衰减上下文
        以避免前一个话题的衰减状态影响新话题。
        """
        self._plugin.reset_decay_contexts()
        logger.debug("StreamingSession: 衰减上下文已重置")

    def close(self):
        """
        关闭会话

        - 取消事件订阅
        - 重置衰减上下文
        - 记录审计日志
        """
        if not self._active:
            return

        self._active = False

        # 取消事件订阅
        self._plugin.event_bus.unsubscribe(
            "forward", subscriber_id=self._subscriber_id
        )

        # 重置衰减上下文
        self._plugin.reset_decay_contexts()

        # 审计记录
        summary = self.get_session_summary()
        self._plugin.audit_log.record("streaming_session_end", {
            "total_steps": summary["total_steps"],
            "injection_count": summary["injection_count"],
            "injection_rate": summary["injection_rate"],
            "elapsed_seconds": summary["elapsed_seconds"],
        })

        logger.info(
            f"StreamingSession 已关闭: "
            f"steps={summary['total_steps']}, "
            f"injections={summary['injection_count']}, "
            f"rate={summary['injection_rate']:.2%}"
        )

    @property
    def is_active(self) -> bool:
        """会话是否活跃"""
        return self._active

    @property
    def step_count(self) -> int:
        """当前步数"""
        return self._step_count

    def __enter__(self):
        """支持 with 语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动关闭"""
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"StreamingSession("
            f"steps={self._step_count}, "
            f"injections={self._total_injection_count}, "
            f"active={self._active})"
        )
