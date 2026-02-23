"""
aga-knowledge Runtime 注册表

管理所有活跃的 AGA Runtime 实例。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class RuntimeInfo:
    """Runtime 实例信息"""
    instance_id: str
    namespaces: List[str]
    host: Optional[str] = None
    port: Optional[int] = None
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    messages_received: int = 0
    last_message_at: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "namespaces": self.namespaces,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "registered_at": self.registered_at,
            "messages_received": self.messages_received,
            "last_message_at": self.last_message_at,
            "metadata": self.metadata,
        }


class RuntimeRegistry:
    """
    Runtime 注册表

    跟踪所有活跃的 AGA Runtime 实例。

    功能：
    - 注册/注销 Runtime
    - 心跳检测
    - 获取活跃 Runtime 列表
    """

    def __init__(self, heartbeat_timeout: int = 90, cleanup_interval: int = 30):
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval
        self._runtimes: Dict[str, RuntimeInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("RuntimeRegistry started")

    async def stop(self):
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_inactive(self):
        now = time.time()
        inactive = [
            iid for iid, info in self._runtimes.items()
            if now - info.last_heartbeat > self.heartbeat_timeout
        ]
        for iid in inactive:
            logger.warning(f"Runtime {iid} inactive, removing")
            del self._runtimes[iid]

    def register(self, instance_id: str, namespaces: List[str],
                 host: Optional[str] = None, port: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        info = RuntimeInfo(
            instance_id=instance_id, namespaces=namespaces,
            host=host, port=port, metadata=metadata,
        )
        self._runtimes[instance_id] = info
        logger.info(f"Runtime registered: {instance_id}")

    def deregister(self, instance_id: str):
        if instance_id in self._runtimes:
            del self._runtimes[instance_id]

    def heartbeat(self, instance_id: str):
        if instance_id in self._runtimes:
            self._runtimes[instance_id].last_heartbeat = time.time()
            self._runtimes[instance_id].status = "active"

    def get_all_runtimes(self) -> List[RuntimeInfo]:
        return list(self._runtimes.values())

    def get_active_runtimes(self) -> List[RuntimeInfo]:
        return [r for r in self._runtimes.values() if r.status == "active"]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._runtimes)
        active = len([r for r in self._runtimes.values() if r.status == "active"])
        return {
            "total_runtimes": total,
            "active_runtimes": active,
            "inactive_runtimes": total - active,
            "runtimes": [r.to_dict() for r in self._runtimes.values()],
        }
