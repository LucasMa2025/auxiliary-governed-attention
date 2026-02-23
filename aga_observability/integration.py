"""
aga_observability/integration.py — 自动集成入口

这是 aga-core 的 AGAPlugin._setup_observability() 调用的入口函数。
当 aga-observability 包被安装后，AGAPlugin 初始化时会自动调用:

    from aga_observability import setup_observability
    setup_observability(event_bus, config)

设计要点:
  - 单例模式: 同一进程中只创建一个 ObservabilityStack
  - Fail-Open: 任何异常不影响 aga-core 正常工作
  - 延迟绑定: plugin 实例在 setup 时可能还未完全初始化
"""
import logging
import threading
from typing import Optional

from .config import ObservabilityConfig
from .stack import ObservabilityStack

logger = logging.getLogger(__name__)

# 全局单例
_global_stack: Optional[ObservabilityStack] = None
_lock = threading.Lock()


def setup_observability(event_bus, aga_config=None) -> Optional[ObservabilityStack]:
    """
    自动集成入口 — 由 AGAPlugin._setup_observability() 调用

    Args:
        event_bus: aga-core 的 EventBus 实例
        aga_config: AGAConfig 实例（可选）

    Returns:
        ObservabilityStack 实例（如果创建成功）
    """
    global _global_stack

    with _lock:
        if _global_stack is not None:
            logger.debug("ObservabilityStack 已存在，跳过重复创建")
            return _global_stack

        try:
            # 从 AGAConfig 映射配置
            if aga_config is not None:
                obs_config = ObservabilityConfig.from_aga_config(aga_config)
            else:
                obs_config = ObservabilityConfig()

            if not obs_config.enabled:
                logger.debug("可观测性已禁用")
                return None

            # 创建并启动
            stack = ObservabilityStack(
                event_bus=event_bus,
                config=obs_config,
            )
            stack.start()

            _global_stack = stack
            logger.info("aga-observability 自动集成完成")
            return stack

        except Exception as e:
            # Fail-Open: 不影响 aga-core
            logger.warning(f"aga-observability 自动集成失败 (Fail-Open): {e}")
            return None


def get_global_stack() -> Optional[ObservabilityStack]:
    """获取全局 ObservabilityStack 实例"""
    return _global_stack


def bind_plugin(plugin) -> None:
    """
    绑定 AGAPlugin 到全局 ObservabilityStack

    通常在 AGAPlugin 完全初始化后调用。

    Args:
        plugin: AGAPlugin 实例
    """
    if _global_stack:
        _global_stack.bind_plugin(plugin)


def shutdown_observability() -> None:
    """关闭全局 ObservabilityStack"""
    global _global_stack

    with _lock:
        if _global_stack:
            _global_stack.shutdown()
            _global_stack = None
            logger.info("aga-observability 已关闭")
