"""
AGA 持久化适配器抽象基类

定义统一的持久化接口，所有适配器必须实现。

版本: v3.0
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from ..types import LifecycleState


# ==================== 异常类 ====================

class PersistenceError(Exception):
    """持久化基础异常"""
    pass


class ConnectionError(PersistenceError):
    """连接错误"""
    pass


class SerializationError(PersistenceError):
    """序列化错误"""
    pass


# ==================== 数据类 ====================

@dataclass
class KnowledgeRecord:
    """
    知识记录（数据库层面）
    
    用于持久化层的数据传输对象。
    """
    slot_idx: int
    lu_id: str
    condition: str
    decision: str
    key_vector: List[float]
    value_vector: List[float]
    lifecycle_state: str
    namespace: str = "default"
    hit_count: int = 0
    consecutive_misses: int = 0
    version: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'slot_idx': self.slot_idx,
            'lu_id': self.lu_id,
            'condition': self.condition,
            'decision': self.decision,
            'key_vector': self.key_vector,
            'value_vector': self.value_vector,
            'lifecycle_state': self.lifecycle_state,
            'namespace': self.namespace,
            'hit_count': self.hit_count,
            'consecutive_misses': self.consecutive_misses,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeRecord":
        """从字典创建"""
        return cls(
            slot_idx=data['slot_idx'],
            lu_id=data['lu_id'],
            condition=data.get('condition', ''),
            decision=data.get('decision', ''),
            key_vector=data['key_vector'],
            value_vector=data['value_vector'],
            lifecycle_state=data['lifecycle_state'],
            namespace=data.get('namespace', 'default'),
            hit_count=data.get('hit_count', 0),
            consecutive_misses=data.get('consecutive_misses', 0),
            version=data.get('version', 1),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            metadata=data.get('metadata'),
        )
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeRecord":
        """从 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))


# ==================== 抽象基类 ====================

class PersistenceAdapter(ABC):
    """
    持久化适配器抽象基类
    
    定义统一的持久化接口，支持：
    - 同步和异步操作
    - 单条和批量操作
    - 生命周期管理
    - 统计和审计
    """
    
    # ==================== 连接管理 ====================
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        建立连接
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """检查连接状态"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        pass
    
    # ==================== 槽位操作 ====================
    
    @abstractmethod
    async def save_slot(self, namespace: str, record: KnowledgeRecord) -> bool:
        """
        保存槽位
        
        Args:
            namespace: 命名空间
            record: 知识记录
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    async def load_slot(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        """
        加载槽位
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
        
        Returns:
            知识记录或 None
        """
        pass
    
    @abstractmethod
    async def delete_slot(self, namespace: str, lu_id: str) -> bool:
        """
        删除槽位
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    async def slot_exists(self, namespace: str, lu_id: str) -> bool:
        """
        检查槽位是否存在
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
        
        Returns:
            是否存在
        """
        pass
    
    # ==================== 批量操作 ====================
    
    @abstractmethod
    async def save_batch(self, namespace: str, records: List[KnowledgeRecord]) -> int:
        """
        批量保存
        
        Args:
            namespace: 命名空间
            records: 知识记录列表
        
        Returns:
            成功保存的数量
        """
        pass
    
    @abstractmethod
    async def load_active_slots(self, namespace: str) -> List[KnowledgeRecord]:
        """
        加载活跃槽位（非 QUARANTINED）
        
        Args:
            namespace: 命名空间
        
        Returns:
            活跃知识记录列表
        """
        pass
    
    @abstractmethod
    async def load_all_slots(self, namespace: str) -> List[KnowledgeRecord]:
        """
        加载所有槽位
        
        Args:
            namespace: 命名空间
        
        Returns:
            所有知识记录列表
        """
        pass
    
    # ==================== 生命周期管理 ====================
    
    @abstractmethod
    async def update_lifecycle(
        self, 
        namespace: str, 
        lu_id: str, 
        new_state: LifecycleState
    ) -> bool:
        """
        更新生命周期状态
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
            new_state: 新状态
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    async def update_lifecycle_batch(
        self,
        namespace: str,
        updates: List[tuple]  # [(lu_id, new_state), ...]
    ) -> int:
        """
        批量更新生命周期
        
        Args:
            namespace: 命名空间
            updates: 更新列表 [(lu_id, new_state), ...]
        
        Returns:
            成功更新的数量
        """
        pass
    
    # ==================== 统计查询 ====================
    
    @abstractmethod
    async def get_slot_count(
        self, 
        namespace: str, 
        state: Optional[LifecycleState] = None
    ) -> int:
        """
        获取槽位数量
        
        Args:
            namespace: 命名空间
            state: 可选的状态过滤
        
        Returns:
            槽位数量
        """
        pass
    
    @abstractmethod
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            namespace: 命名空间
        
        Returns:
            统计信息字典
        """
        pass
    
    # ==================== 命中计数 ====================
    
    @abstractmethod
    async def increment_hit_count(
        self, 
        namespace: str, 
        lu_ids: List[str]
    ) -> bool:
        """
        批量增加命中计数
        
        Args:
            namespace: 命名空间
            lu_ids: 知识单元 ID 列表
        
        Returns:
            是否成功
        """
        pass
    
    # ==================== 同步方法（可选实现） ====================
    
    def save_slot_sync(self, namespace: str, record: KnowledgeRecord) -> bool:
        """同步保存槽位（默认实现）"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.save_slot(namespace, record))
    
    def load_slot_sync(self, namespace: str, lu_id: str) -> Optional[KnowledgeRecord]:
        """同步加载槽位（默认实现）"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.load_slot(namespace, lu_id))
    
    def load_active_slots_sync(self, namespace: str) -> List[KnowledgeRecord]:
        """同步加载活跃槽位（默认实现）"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.load_active_slots(namespace))
