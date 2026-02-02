"""
AGA 同步消息协议

定义 Portal 与 Runtime 之间的消息格式。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import json
import uuid


class MessageType(str, Enum):
    """消息类型"""
    # 知识操作
    INJECT = "INJECT"           # 注入新知识
    UPDATE = "UPDATE"           # 更新知识
    QUARANTINE = "QUARANTINE"   # 隔离知识
    DELETE = "DELETE"           # 删除知识
    
    # 批量操作
    BATCH_INJECT = "BATCH_INJECT"
    BATCH_UPDATE = "BATCH_UPDATE"
    
    # 状态同步
    FULL_SYNC = "FULL_SYNC"     # 全量同步请求
    SYNC_RESPONSE = "SYNC_RESPONSE"  # 全量同步响应
    
    # 控制消息
    HEARTBEAT = "HEARTBEAT"
    ACK = "ACK"
    NACK = "NACK"
    
    # Runtime 注册
    RUNTIME_REGISTER = "RUNTIME_REGISTER"
    RUNTIME_DEREGISTER = "RUNTIME_DEREGISTER"
    RUNTIME_STATUS = "RUNTIME_STATUS"


@dataclass
class SyncMessage:
    """
    同步消息
    
    Portal 与 Runtime 之间的标准消息格式。
    
    示例:
        # 注入消息
        msg = SyncMessage(
            message_type=MessageType.INJECT,
            namespace="default",
            lu_id="knowledge_001",
            key_vector=[0.1, 0.2, ...],
            value_vector=[0.3, 0.4, ...],
            condition="当用户询问...",
            decision="应该回答...",
        )
        
        # 序列化
        json_str = msg.to_json()
        
        # 反序列化
        msg2 = SyncMessage.from_json(json_str)
    """
    # 消息标识
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    message_type: MessageType = MessageType.INJECT
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    
    # 目标
    namespace: str = "default"
    lu_id: Optional[str] = None
    
    # 向量数据（仅 INJECT 时包含）
    key_vector: Optional[List[float]] = None
    value_vector: Optional[List[float]] = None
    
    # 元数据
    condition: Optional[str] = None
    decision: Optional[str] = None
    
    # 状态
    lifecycle_state: Optional[str] = None  # probationary, confirmed, deprecated, quarantined
    trust_tier: Optional[str] = None       # s0_acceleration, s1_experience, s2_policy, s3_immutable
    
    # 来源
    source_instance: Optional[str] = None
    reason: Optional[str] = None
    
    # 批量数据（仅 BATCH_* 类型）
    batch_items: Optional[List[Dict[str, Any]]] = None
    
    # 扩展字段
    metadata: Optional[Dict[str, Any]] = None
    
    # 确认相关
    require_ack: bool = False
    correlation_id: Optional[str] = None  # 关联消息 ID
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value if isinstance(self.message_type, MessageType) else self.message_type,
            "timestamp": self.timestamp,
            "namespace": self.namespace,
            "lu_id": self.lu_id,
            "key_vector": self.key_vector,
            "value_vector": self.value_vector,
            "condition": self.condition,
            "decision": self.decision,
            "lifecycle_state": self.lifecycle_state,
            "trust_tier": self.trust_tier,
            "source_instance": self.source_instance,
            "reason": self.reason,
            "batch_items": self.batch_items,
            "metadata": self.metadata,
            "require_ack": self.require_ack,
            "correlation_id": self.correlation_id,
        }
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncMessage":
        """从字典创建"""
        message_type = data.get("message_type", "INJECT")
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        
        return cls(
            message_id=data.get("message_id", uuid.uuid4().hex),
            message_type=message_type,
            timestamp=data.get("timestamp", datetime.utcnow().timestamp()),
            namespace=data.get("namespace", "default"),
            lu_id=data.get("lu_id"),
            key_vector=data.get("key_vector"),
            value_vector=data.get("value_vector"),
            condition=data.get("condition"),
            decision=data.get("decision"),
            lifecycle_state=data.get("lifecycle_state"),
            trust_tier=data.get("trust_tier"),
            source_instance=data.get("source_instance"),
            reason=data.get("reason"),
            batch_items=data.get("batch_items"),
            metadata=data.get("metadata"),
            require_ack=data.get("require_ack", False),
            correlation_id=data.get("correlation_id"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "SyncMessage":
        """从 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))
    
    # ==================== 工厂方法 ====================
    
    @classmethod
    def inject(
        cls,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        condition: str,
        decision: str,
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        source_instance: Optional[str] = None,
        require_ack: bool = True,
    ) -> "SyncMessage":
        """创建注入消息"""
        return cls(
            message_type=MessageType.INJECT,
            namespace=namespace,
            lu_id=lu_id,
            key_vector=key_vector,
            value_vector=value_vector,
            condition=condition,
            decision=decision,
            lifecycle_state=lifecycle_state,
            trust_tier=trust_tier,
            source_instance=source_instance,
            require_ack=require_ack,
        )
    
    @classmethod
    def update_lifecycle(
        cls,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
        source_instance: Optional[str] = None,
        require_ack: bool = True,
    ) -> "SyncMessage":
        """创建生命周期更新消息"""
        return cls(
            message_type=MessageType.UPDATE,
            namespace=namespace,
            lu_id=lu_id,
            lifecycle_state=new_state,
            reason=reason,
            source_instance=source_instance,
            require_ack=require_ack,
        )
    
    @classmethod
    def quarantine(
        cls,
        lu_id: str,
        reason: str,
        namespace: str = "default",
        source_instance: Optional[str] = None,
        require_ack: bool = True,
    ) -> "SyncMessage":
        """创建隔离消息"""
        return cls(
            message_type=MessageType.QUARANTINE,
            namespace=namespace,
            lu_id=lu_id,
            lifecycle_state="quarantined",
            reason=reason,
            source_instance=source_instance,
            require_ack=require_ack,
        )
    
    @classmethod
    def delete(
        cls,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
        source_instance: Optional[str] = None,
    ) -> "SyncMessage":
        """创建删除消息"""
        return cls(
            message_type=MessageType.DELETE,
            namespace=namespace,
            lu_id=lu_id,
            reason=reason,
            source_instance=source_instance,
        )
    
    @classmethod
    def batch_inject(
        cls,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        source_instance: Optional[str] = None,
        require_ack: bool = True,
    ) -> "SyncMessage":
        """创建批量注入消息"""
        return cls(
            message_type=MessageType.BATCH_INJECT,
            namespace=namespace,
            batch_items=items,
            source_instance=source_instance,
            require_ack=require_ack,
        )
    
    @classmethod
    def full_sync_request(
        cls,
        namespace: str = "default",
        source_instance: Optional[str] = None,
    ) -> "SyncMessage":
        """创建全量同步请求"""
        return cls(
            message_type=MessageType.FULL_SYNC,
            namespace=namespace,
            source_instance=source_instance,
        )
    
    @classmethod
    def heartbeat(cls, instance_id: str) -> "SyncMessage":
        """创建心跳消息"""
        return cls(
            message_type=MessageType.HEARTBEAT,
            source_instance=instance_id,
        )
    
    @classmethod
    def runtime_register(
        cls,
        instance_id: str,
        namespaces: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SyncMessage":
        """创建 Runtime 注册消息"""
        return cls(
            message_type=MessageType.RUNTIME_REGISTER,
            source_instance=instance_id,
            metadata={
                "namespaces": namespaces,
                **(metadata or {}),
            },
        )


@dataclass
class SyncAck:
    """
    同步确认消息
    
    Runtime 处理消息后返回的确认。
    """
    message_id: str                    # 原消息 ID
    instance_id: str                   # Runtime 实例 ID
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "instance_id": self.instance_id,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "SyncAck":
        data = json.loads(json_str)
        return cls(
            message_id=data["message_id"],
            instance_id=data["instance_id"],
            success=data.get("success", True),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.utcnow().timestamp()),
            metadata=data.get("metadata"),
        )
