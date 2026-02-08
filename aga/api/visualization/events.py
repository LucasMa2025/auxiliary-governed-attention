"""
AGA 实时事件流 API

提供 WebSocket 实时事件推送。

版本: v1.0
"""

import logging
import asyncio
import json
import time
from typing import Set, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None  # type: ignore
    WebSocket = None  # type: ignore


# ==================== 事件类型 ====================

class EventType:
    """事件类型常量"""
    KNOWLEDGE_INJECT = "knowledge_inject"
    KNOWLEDGE_HIT = "knowledge_hit"
    KNOWLEDGE_TRANSITION = "knowledge_transition"
    ROUTE_DECISION = "route_decision"
    ALERT = "alert"
    SYSTEM = "system"


# ==================== 连接管理器 ====================

@dataclass
class ConnectionInfo:
    """连接信息"""
    websocket: Any  # WebSocket
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


class EventConnectionManager:
    """
    WebSocket 连接管理器
    
    管理多个客户端连接和事件订阅。
    """
    
    def __init__(self):
        self._connections: Dict[str, ConnectionInfo] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: Any, client_id: str) -> None:
        """接受新连接"""
        await websocket.accept()
        async with self._lock:
            self._connections[client_id] = ConnectionInfo(
                websocket=websocket,
                subscriptions=set(),
            )
        logger.info(f"Client {client_id} connected. Total: {len(self._connections)}")
    
    async def disconnect(self, client_id: str) -> None:
        """断开连接"""
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
        logger.info(f"Client {client_id} disconnected. Total: {len(self._connections)}")
    
    async def subscribe(self, client_id: str, event_types: List[str]) -> None:
        """订阅事件"""
        async with self._lock:
            if client_id in self._connections:
                self._connections[client_id].subscriptions.update(event_types)
                logger.debug(f"Client {client_id} subscribed to: {event_types}")
    
    async def unsubscribe(self, client_id: str, event_types: List[str]) -> None:
        """取消订阅"""
        async with self._lock:
            if client_id in self._connections:
                self._connections[client_id].subscriptions.difference_update(event_types)
                logger.debug(f"Client {client_id} unsubscribed from: {event_types}")
    
    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """广播事件到所有订阅的客户端"""
        message = json.dumps({
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": data,
        })
        
        disconnected = []
        async with self._lock:
            for client_id, conn in self._connections.items():
                if event_type in conn.subscriptions or "*" in conn.subscriptions:
                    try:
                        await conn.websocket.send_text(message)
                    except Exception as e:
                        logger.warning(f"Failed to send to {client_id}: {e}")
                        disconnected.append(client_id)
        
        # 清理断开的连接
        for client_id in disconnected:
            await self.disconnect(client_id)
    
    async def send_to(self, client_id: str, message: Dict[str, Any]) -> bool:
        """发送消息到指定客户端"""
        async with self._lock:
            if client_id in self._connections:
                try:
                    await self._connections[client_id].websocket.send_text(
                        json.dumps(message)
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
        return False
    
    def get_connection_count(self) -> int:
        """获取连接数"""
        return len(self._connections)
    
    def get_subscriptions(self, client_id: str) -> Set[str]:
        """获取客户端订阅"""
        if client_id in self._connections:
            return self._connections[client_id].subscriptions.copy()
        return set()


# 全局连接管理器
_event_manager = EventConnectionManager()


def get_event_manager() -> EventConnectionManager:
    """获取事件管理器实例"""
    return _event_manager


# ==================== 事件发布函数 ====================

async def publish_knowledge_inject(
    lu_id: str,
    namespace: str,
    lifecycle_state: str,
    **kwargs,
) -> None:
    """发布知识注入事件"""
    await _event_manager.broadcast(EventType.KNOWLEDGE_INJECT, {
        "lu_id": lu_id,
        "namespace": namespace,
        "lifecycle_state": lifecycle_state,
        **kwargs,
    })


async def publish_knowledge_hit(
    lu_id: str,
    namespace: str,
    hit_count: int,
    query_preview: Optional[str] = None,
    **kwargs,
) -> None:
    """发布知识命中事件"""
    await _event_manager.broadcast(EventType.KNOWLEDGE_HIT, {
        "lu_id": lu_id,
        "namespace": namespace,
        "hit_count": hit_count,
        "query_preview": query_preview,
        **kwargs,
    })


async def publish_knowledge_transition(
    lu_id: str,
    namespace: str,
    from_state: Optional[str],
    to_state: Optional[str],
    reason: str,
    **kwargs,
) -> None:
    """发布知识状态变更事件"""
    await _event_manager.broadcast(EventType.KNOWLEDGE_TRANSITION, {
        "lu_id": lu_id,
        "namespace": namespace,
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason,
        **kwargs,
    })


async def publish_route_decision(
    trace_id: str,
    gate0: str,
    gate1: str,
    gate2_selected: int,
    aga_applied: bool,
    latency_ms: float,
    **kwargs,
) -> None:
    """发布路由决策事件"""
    await _event_manager.broadcast(EventType.ROUTE_DECISION, {
        "trace_id": trace_id,
        "gate0": gate0,
        "gate1": gate1,
        "gate2_selected": gate2_selected,
        "aga_applied": aga_applied,
        "latency_ms": latency_ms,
        **kwargs,
    })


async def publish_alert(
    name: str,
    severity: str,
    summary: str,
    **kwargs,
) -> None:
    """发布告警事件"""
    await _event_manager.broadcast(EventType.ALERT, {
        "name": name,
        "severity": severity,
        "summary": summary,
        **kwargs,
    })


# ==================== API 路由 ====================

if HAS_FASTAPI:
    router = APIRouter(tags=["events"])
    
    @router.websocket("/ws/events")
    async def events_websocket(
        websocket: WebSocket,
        token: str = Query(None, description="认证 Token"),
    ):
        """
        实时事件流 WebSocket 端点
        
        客户端消息格式:
        ```json
        {
            "action": "subscribe" | "unsubscribe" | "ping",
            "events": ["knowledge_inject", "knowledge_hit", ...]
        }
        ```
        
        服务端推送格式:
        ```json
        {
            "event": "knowledge_hit",
            "timestamp": "2026-02-09T10:30:45.123Z",
            "data": { ... }
        }
        ```
        """
        import uuid
        
        # 简单认证 (生产环境应使用更安全的方式)
        # if token != "expected_token":
        #     await websocket.close(code=4001, reason="Unauthorized")
        #     return
        
        client_id = str(uuid.uuid4())[:8]
        
        try:
            await _event_manager.connect(websocket, client_id)
            
            # 发送欢迎消息
            await _event_manager.send_to(client_id, {
                "event": "system",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": {
                    "type": "connected",
                    "client_id": client_id,
                    "message": "Connected to AGA event stream",
                },
            })
            
            while True:
                try:
                    # 接收客户端消息
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    action = message.get("action")
                    
                    if action == "subscribe":
                        events = message.get("events", [])
                        await _event_manager.subscribe(client_id, events)
                        await _event_manager.send_to(client_id, {
                            "event": "system",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "data": {
                                "type": "subscribed",
                                "events": events,
                            },
                        })
                    
                    elif action == "unsubscribe":
                        events = message.get("events", [])
                        await _event_manager.unsubscribe(client_id, events)
                        await _event_manager.send_to(client_id, {
                            "event": "system",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "data": {
                                "type": "unsubscribed",
                                "events": events,
                            },
                        })
                    
                    elif action == "ping":
                        await _event_manager.send_to(client_id, {
                            "event": "system",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "data": {
                                "type": "pong",
                            },
                        })
                    
                    else:
                        await _event_manager.send_to(client_id, {
                            "event": "system",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "data": {
                                "type": "error",
                                "message": f"Unknown action: {action}",
                            },
                        })
                
                except json.JSONDecodeError:
                    await _event_manager.send_to(client_id, {
                        "event": "system",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "data": {
                            "type": "error",
                            "message": "Invalid JSON",
                        },
                    })
        
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            await _event_manager.disconnect(client_id)
    
    @router.get("/events/connections")
    async def get_event_connections() -> Dict[str, Any]:
        """
        获取当前 WebSocket 连接信息
        """
        return {
            "total_connections": _event_manager.get_connection_count(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    
    @router.post("/events/test")
    async def send_test_event(
        event_type: str = Query("system", description="事件类型"),
        message: str = Query("Test event", description="消息内容"),
    ) -> Dict[str, Any]:
        """
        发送测试事件 (仅用于调试)
        """
        await _event_manager.broadcast(event_type, {
            "type": "test",
            "message": message,
        })
        
        return {
            "success": True,
            "event_type": event_type,
            "connections_notified": _event_manager.get_connection_count(),
        }

else:
    router = None  # type: ignore


__all__ = [
    "router",
    "EventType",
    "EventConnectionManager",
    "get_event_manager",
    "publish_knowledge_inject",
    "publish_knowledge_hit",
    "publish_knowledge_transition",
    "publish_route_decision",
    "publish_alert",
]
