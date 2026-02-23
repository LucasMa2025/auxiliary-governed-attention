"""
aga/instrumentation/event_bus.py — 可插拔事件总线

设计要点:
  - 默认使用内存环形缓冲区，零外部依赖
  - aga-observability 安装后自动注册 Prometheus 消费者
  - 事件发射是同步的（<1μs），不影响推理延迟
  - 消费者处理是同步调用，消费者自行决定是否异步处理
"""
import time
import threading
import logging
from typing import Dict, Any, List, Callable, Optional
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """事件数据"""
    type: str  # "forward" | "register" | "unregister" | "attach" | ...
    timestamp: float  # time.time()
    data: Dict[str, Any]  # 事件数据
    source: str = "aga-core"  # 事件来源


class EventBus:
    """
    可插拔事件总线

    使用方式:
        # aga-core 内部（自动）
        bus = EventBus(buffer_size=10000)
        bus.emit("forward", {"gate_mean": 0.3, "aga_applied": True})

        # aga-observability 注册消费者
        bus.subscribe("forward", prometheus_forward_handler)
        bus.subscribe("*", audit_log_handler)  # 订阅所有事件

        # 查询最近事件（始终可用）
        recent = bus.query(event_type="forward", limit=100)
    """

    def __init__(self, buffer_size: int = 10000, enabled: bool = True):
        self._enabled = enabled
        self._buffer: deque = deque(maxlen=buffer_size)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

        # 统计
        self._total_emitted = 0
        self._total_consumed = 0

    def emit(self, event_type: str, data: Dict[str, Any]):
        """
        发射事件（同步，<1μs）

        这是热路径上的方法，必须极快。
        """
        if not self._enabled:
            return

        event = Event(
            type=event_type,
            timestamp=time.time(),
            data=data,
        )

        # 写入环形缓冲区
        self._buffer.append(event)
        self._total_emitted += 1

        # 通知订阅者
        handlers = self._subscribers.get(event_type, [])
        handlers = handlers + self._subscribers.get("*", [])  # 通配符订阅者
        for handler in handlers:
            try:
                handler(event)
                self._total_consumed += 1
            except Exception as e:
                logger.debug(f"Event handler error: {e}")

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        subscriber_id: Optional[str] = None,
    ):
        """
        订阅事件

        Args:
            event_type: 事件类型，"*" 表示订阅所有
            handler: 事件处理函数
            subscriber_id: 可选的订阅者 ID（用于按 ID 取消订阅）
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

            # 记录 subscriber_id → handler 映射
            if subscriber_id:
                if not hasattr(self, "_subscriber_ids"):
                    self._subscriber_ids: Dict[str, List[tuple]] = {}
                if subscriber_id not in self._subscriber_ids:
                    self._subscriber_ids[subscriber_id] = []
                self._subscriber_ids[subscriber_id].append((event_type, handler))

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable = None,
        subscriber_id: Optional[str] = None,
    ):
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 事件处理函数（与 subscribe 时相同）
            subscriber_id: 订阅者 ID（按 ID 取消所有该 ID 的订阅）
        """
        with self._lock:
            if subscriber_id and hasattr(self, "_subscriber_ids"):
                # 按 subscriber_id 取消所有订阅
                entries = self._subscriber_ids.pop(subscriber_id, [])
                for et, h in entries:
                    if et in self._subscribers:
                        self._subscribers[et] = [
                            x for x in self._subscribers[et] if x != h
                        ]
            elif handler and event_type in self._subscribers:
                self._subscribers[event_type] = [
                    h for h in self._subscribers[event_type] if h != handler
                ]

    def query(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[Dict]:
        """
        查询最近事件（始终可用，不依赖外部包）

        Args:
            event_type: 过滤事件类型
            limit: 返回数量
            since: 起始时间戳
        """
        events = list(self._buffer)
        if event_type:
            events = [e for e in events if e.type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return [
            {"type": e.type, "timestamp": e.timestamp, "data": e.data}
            for e in events[-limit:]
        ]

    def get_stats(self) -> Dict:
        """获取事件总线统计"""
        return {
            "total_emitted": self._total_emitted,
            "total_consumed": self._total_consumed,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self._buffer.maxlen,
            "subscribers": {k: len(v) for k, v in self._subscribers.items()},
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
