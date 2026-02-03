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

class TrustTier(str):
    """
    信任层级
    
    定义知识的信任等级，用于访问控制和过滤。
    
    层级说明：
    - SYSTEM: 系统级，最高信任，核心规则
    - VERIFIED: 已验证，经过人工审核确认
    - STANDARD: 标准级，默认层级
    - EXPERIMENTAL: 实验性，尚在测试
    - UNTRUSTED: 不可信，需谨慎使用
    """
    SYSTEM = "system"
    VERIFIED = "verified"
    STANDARD = "standard"
    EXPERIMENTAL = "experimental"
    UNTRUSTED = "untrusted"


# 信任层级优先级（数值越大，信任度越高）
TRUST_TIER_PRIORITY = {
    TrustTier.SYSTEM: 100,
    TrustTier.VERIFIED: 80,
    TrustTier.STANDARD: 50,
    TrustTier.EXPERIMENTAL: 30,
    TrustTier.UNTRUSTED: 10,
}


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
    trust_tier: str = TrustTier.STANDARD  # 新增：信任层级
    
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
            'trust_tier': self.trust_tier,
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
            trust_tier=data.get('trust_tier', TrustTier.STANDARD),
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
    
    # ==================== 命名空间管理 ====================
    
    async def get_namespaces(self) -> List[str]:
        """
        获取所有命名空间
        
        Returns:
            命名空间列表
        """
        # 默认实现返回空列表，子类应覆盖
        return []
    
    # ==================== Portal 扩展方法 ====================
    
    async def save_knowledge(self, namespace: str, lu_id: str, data: Dict[str, Any]) -> bool:
        """
        保存知识（Portal 简化接口）
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
            data: 知识数据字典
        
        Returns:
            是否成功
        """
        record = KnowledgeRecord(
            slot_idx=data.get("slot_idx", -1),
            lu_id=lu_id,
            condition=data.get("condition", ""),
            decision=data.get("decision", ""),
            key_vector=data.get("key_vector", []),
            value_vector=data.get("value_vector", []),
            lifecycle_state=data.get("lifecycle_state", "probationary"),
            namespace=namespace,
            hit_count=data.get("hit_count", 0),
            metadata=data.get("metadata"),
        )
        return await self.save_slot(namespace, record)
    
    async def load_knowledge(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """
        加载知识（Portal 简化接口）
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
        
        Returns:
            知识数据字典或 None
        """
        record = await self.load_slot(namespace, lu_id)
        if record:
            return record.to_dict()
        return None
    
    async def delete_knowledge(self, namespace: str, lu_id: str) -> bool:
        """
        删除知识（Portal 简化接口）
        
        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
        
        Returns:
            是否成功
        """
        return await self.delete_slot(namespace, lu_id)
    
    async def query_knowledge(
        self,
        namespace: str,
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        min_trust_priority: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        查询知识列表（Portal 简化接口）
        
        Args:
            namespace: 命名空间
            lifecycle_states: 状态过滤
            trust_tiers: 层级过滤（精确匹配）
            min_trust_priority: 最低信任优先级（用于范围过滤）
            limit: 限制数量
            offset: 偏移量
        
        Returns:
            知识数据列表
        """
        # 默认实现：加载所有然后过滤
        records = await self.load_all_slots(namespace)
        
        # 生命周期状态过滤
        if lifecycle_states:
            records = [r for r in records if r.lifecycle_state in lifecycle_states]
        
        # 信任层级过滤
        if trust_tiers:
            records = [r for r in records if r.trust_tier in trust_tiers]
        
        # 最低信任优先级过滤
        if min_trust_priority is not None:
            records = [
                r for r in records 
                if TRUST_TIER_PRIORITY.get(r.trust_tier, 0) >= min_trust_priority
            ]
        
        # 分页
        records = records[offset:offset + limit]
        
        return [r.to_dict() for r in records]
    
    async def query_knowledge_by_trust(
        self,
        namespace: str,
        min_tier: str = TrustTier.STANDARD,
        include_experimental: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        按信任层级查询知识（便捷方法）
        
        Args:
            namespace: 命名空间
            min_tier: 最低信任层级
            include_experimental: 是否包含实验性知识
        
        Returns:
            符合条件的知识列表
        """
        min_priority = TRUST_TIER_PRIORITY.get(min_tier, 0)
        
        records = await self.load_active_slots(namespace)
        
        # 过滤
        filtered = []
        for r in records:
            tier_priority = TRUST_TIER_PRIORITY.get(r.trust_tier, 0)
            
            # 检查是否满足最低优先级
            if tier_priority >= min_priority:
                filtered.append(r)
            # 特殊处理实验性
            elif include_experimental and r.trust_tier == TrustTier.EXPERIMENTAL:
                filtered.append(r)
        
        return [r.to_dict() for r in filtered]
    
    async def update_trust_tier(
        self,
        namespace: str,
        lu_id: str,
        new_tier: str,
    ) -> bool:
        """
        更新知识的信任层级
        
        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            new_tier: 新的信任层级
        
        Returns:
            是否成功
        """
        # 验证信任层级
        if new_tier not in TRUST_TIER_PRIORITY:
            raise ValueError(f"Invalid trust tier: {new_tier}")
        
        # 加载记录
        record = await self.load_slot(namespace, lu_id)
        if not record:
            return False
        
        # 更新信任层级
        record.trust_tier = new_tier
        record.updated_at = datetime.now().isoformat()
        if record.version:
            record.version += 1
        
        # 保存
        return await self.save_slot(namespace, record)
    
    # ==================== 审计日志 ====================
    
    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        """
        保存审计日志
        
        Args:
            entry: 审计日志条目
        
        Returns:
            是否成功
        """
        # 默认实现：不保存
        return True
    
    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        查询审计日志
        
        Args:
            namespace: 命名空间过滤
            lu_id: 知识 ID 过滤
            limit: 限制数量
            offset: 偏移量
        
        Returns:
            审计日志列表
        """
        # 默认实现：返回空列表
        return []
    
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
