"""
aga/distributed.py — 分布式支持（TP / 多实例）

提供 Tensor Parallelism 场景下的 KVStore 同步和参数广播。

设计原则:
  - TP 场景下每个 rank 维护完整的 KVStore 副本（Full Replica）
  - 知识更新由 rank 0 执行，然后广播到所有 rank
  - 学习参数（gate_system, injector）在初始化时从 rank 0 广播
  - 使用 torch.distributed 原语，不引入额外依赖

使用方式:
    # 在 TP 环境中初始化
    plugin = AGAPlugin(config)
    tp_manager = TPManager(plugin)
    tp_manager.broadcast_parameters()  # 同步学习参数

    # 注册知识（只在 rank 0 执行）
    if tp_manager.is_primary:
        plugin.register("fact_001", key=k, value=v)
    tp_manager.broadcast_knowledge()  # 同步到所有 rank

注意:
    此模块需要 torch.distributed 已初始化。
    如果未初始化，所有操作都是 no-op（单 GPU 兼容）。
"""
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """检查 torch.distributed 是否已初始化"""
    try:
        return torch.distributed.is_initialized()
    except Exception:
        return False


def get_rank() -> int:
    """获取当前 rank（未初始化时返回 0）"""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """获取 world size（未初始化时返回 1）"""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


class TPManager:
    """
    Tensor Parallelism 管理器

    在 TP 场景下管理 AGA 的状态同步：
    - KVStore 内容同步（Full Replica）
    - 学习参数同步（gate_system, injector, decay）
    - 知识注册/删除的协调

    设计:
      - rank 0 是 primary，负责知识管理
      - 其他 rank 通过 broadcast 接收更新
      - 所有 rank 独立执行 forward（使用相同的 KVStore 内容）
    """

    def __init__(self, plugin: "AGAPlugin", group=None):
        """
        Args:
            plugin: AGAPlugin 实例
            group: torch.distributed 进程组（默认使用全局组）
        """
        self._plugin = plugin
        self._group = group
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._enabled = is_distributed() and self._world_size > 1

        if self._enabled:
            logger.info(
                f"TPManager 初始化: rank={self._rank}/{self._world_size}, "
                f"primary={'是' if self.is_primary else '否'}"
            )
        else:
            logger.debug("TPManager: 非分布式环境，所有操作为 no-op")

    @property
    def is_primary(self) -> bool:
        """是否是主 rank（rank 0）"""
        return self._rank == 0

    @property
    def is_enabled(self) -> bool:
        """是否启用分布式"""
        return self._enabled

    def broadcast_parameters(self) -> None:
        """
        从 rank 0 广播学习参数到所有 rank

        包括:
        - EntropyGateSystem 的参数（gate_w1, gate_bias, uncertainty_proj）
        - BottleneckInjector 的参数（q_proj, value_down, value_up）
        - PersistenceDecay 的参数（adaptive_weight）
        """
        if not self._enabled:
            return

        # 广播所有可学习参数
        for name, param in self._plugin.gate_system.named_parameters():
            torch.distributed.broadcast(param.data, src=0, group=self._group)

        for name, param in self._plugin.injector.named_parameters():
            torch.distributed.broadcast(param.data, src=0, group=self._group)

        if self._plugin.decay is not None:
            for name, param in self._plugin.decay.named_parameters():
                torch.distributed.broadcast(param.data, src=0, group=self._group)

        logger.info(f"TPManager: 参数已从 rank 0 广播到 {self._world_size} 个 rank")

    def broadcast_knowledge(self) -> None:
        """
        从 rank 0 广播 KVStore 内容到所有 rank

        使用 broadcast 原语同步完整的 KVStore 张量。
        这是一个全量同步操作，适用于初始化或批量更新后。
        """
        if not self._enabled:
            return

        store = self._plugin.store

        # 广播 GPU 张量
        torch.distributed.broadcast(store.keys, src=0, group=self._group)
        torch.distributed.broadcast(store.values, src=0, group=self._group)
        torch.distributed.broadcast(store.reliability, src=0, group=self._group)
        torch.distributed.broadcast(store.active, src=0, group=self._group)

        # 广播 CPU 侧元数据（通过序列化）
        if self.is_primary:
            metadata = {
                "id_to_slot": store._id_to_slot,
                "slot_to_id": store._slot_to_id,
                "metadata": store._metadata,
                "access_order": list(store._access_order.keys()),
                "free_slots": store._free_slots,
            }
            metadata_list = [metadata]
        else:
            metadata_list = [None]

        torch.distributed.broadcast_object_list(metadata_list, src=0, group=self._group)

        # 非主 rank 更新 CPU 侧元数据
        if not self.is_primary:
            metadata = metadata_list[0]
            store._id_to_slot = metadata["id_to_slot"]
            store._slot_to_id = metadata["slot_to_id"]
            store._metadata = metadata["metadata"]
            store._access_order.clear()
            for key in metadata["access_order"]:
                store._access_order[key] = store._id_to_slot.get(key, 0)
            store._free_slots = metadata["free_slots"]

        # 使缓存失效
        store._active_cache_valid = False

        logger.info(
            f"TPManager: KVStore 已同步 "
            f"({store.count} 条知识, {self._world_size} 个 rank)"
        )

    def broadcast_single_entry(
        self,
        id: str,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        reliability: float = 1.0,
        metadata: Optional[Dict] = None,
        operation: str = "register",  # register / unregister
    ) -> bool:
        """
        广播单条知识更新

        比 broadcast_knowledge() 更高效，适用于增量更新。

        Args:
            id: 知识 ID
            key: 键向量（register 时必须）
            value: 值向量（register 时必须）
            reliability: 可靠性分数
            metadata: 元数据
            operation: 操作类型

        Returns:
            是否成功
        """
        if not self._enabled:
            # 单 GPU 模式直接操作
            if operation == "register" and key is not None and value is not None:
                return self._plugin.register(id, key, value, reliability, metadata)
            elif operation == "unregister":
                return self._plugin.unregister(id)
            return False

        # 广播操作信息
        op_info = [{
            "operation": operation,
            "id": id,
            "reliability": reliability,
            "metadata": metadata,
        }]
        torch.distributed.broadcast_object_list(op_info, src=0, group=self._group)

        op = op_info[0]

        if op["operation"] == "register":
            # 广播张量
            if self.is_primary:
                assert key is not None and value is not None
                key_broadcast = key.to(self._plugin.device)
                value_broadcast = value.to(self._plugin.device)
            else:
                key_broadcast = torch.zeros(
                    self._plugin.config.bottleneck_dim,
                    device=self._plugin.device, dtype=torch.float16,
                )
                value_broadcast = torch.zeros(
                    self._plugin.config.hidden_dim,
                    device=self._plugin.device, dtype=torch.float16,
                )

            torch.distributed.broadcast(key_broadcast, src=0, group=self._group)
            torch.distributed.broadcast(value_broadcast, src=0, group=self._group)

            return self._plugin.register(
                op["id"], key_broadcast, value_broadcast,
                op["reliability"], op["metadata"],
            )

        elif op["operation"] == "unregister":
            return self._plugin.unregister(op["id"])

        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取分布式状态统计"""
        return {
            "enabled": self._enabled,
            "rank": self._rank,
            "world_size": self._world_size,
            "is_primary": self.is_primary,
            "store_count": self._plugin.store.count,
        }
