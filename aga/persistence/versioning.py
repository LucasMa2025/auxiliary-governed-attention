"""
AGA 知识版本控制模块

提供知识的版本管理功能：
- 版本历史记录
- 版本回滚
- 变更审计
- 差异比较

版本: v1.0
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeVersion:
    """知识版本"""
    version: int
    lu_id: str
    key_vector: List[float]
    value_vector: List[float]
    condition: str
    decision: str
    lifecycle_state: str
    created_at: datetime
    created_by: str
    change_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'lu_id': self.lu_id,
            'key_vector': self.key_vector,
            'value_vector': self.value_vector,
            'condition': self.condition,
            'decision': self.decision,
            'lifecycle_state': self.lifecycle_state,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'change_reason': self.change_reason,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeVersion':
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        
        return cls(
            version=data.get('version', 1),
            lu_id=data['lu_id'],
            key_vector=data.get('key_vector', []),
            value_vector=data.get('value_vector', []),
            condition=data.get('condition', ''),
            decision=data.get('decision', ''),
            lifecycle_state=data.get('lifecycle_state', 'probationary'),
            created_at=created_at,
            created_by=data.get('created_by', 'system'),
            change_reason=data.get('change_reason'),
            metadata=data.get('metadata', {}),
        )


@dataclass
class VersionDiff:
    """版本差异"""
    field: str
    old_value: Any
    new_value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'old_value': self.old_value,
            'new_value': self.new_value,
        }


class VersionedKnowledgeStore:
    """
    版本化知识存储
    
    提供知识的版本管理功能，支持版本历史、回滚和审计。
    
    使用示例：
        ```python
        store = VersionedKnowledgeStore(persistence_adapter)
        
        # 创建知识
        version = await store.create_knowledge(
            namespace="default",
            lu_id="rule_001",
            key_vector=[...],
            value_vector=[...],
            condition="当用户询问天气时",
            decision="提供天气信息",
            created_by="admin",
        )
        
        # 更新知识
        new_version = await store.update_knowledge(
            namespace="default",
            lu_id="rule_001",
            key_vector=[...],
            value_vector=[...],
            condition="当用户询问天气时",
            decision="提供详细天气预报",
            updated_by="admin",
            change_reason="增加详细信息",
        )
        
        # 查看历史
        history = await store.get_version_history("default", "rule_001")
        
        # 回滚
        await store.rollback("default", "rule_001", target_version=1)
        ```
    """
    
    def __init__(self, persistence_adapter):
        """
        初始化版本化存储
        
        Args:
            persistence_adapter: 持久化适配器
        """
        self.persistence = persistence_adapter
        
        # 版本历史缓存 (namespace -> lu_id -> [versions])
        self._version_cache: Dict[str, Dict[str, List[KnowledgeVersion]]] = {}
        
        # 配置
        self.max_versions_per_knowledge = 10  # 每个知识最多保留的版本数
    
    async def create_knowledge(
        self,
        namespace: str,
        lu_id: str,
        key_vector: List[float],
        value_vector: List[float],
        condition: str,
        decision: str,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeVersion:
        """
        创建新知识
        
        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            key_vector: Key 向量
            value_vector: Value 向量
            condition: 条件文本
            decision: 决策文本
            created_by: 创建者
            metadata: 元数据
        
        Returns:
            创建的版本
        """
        version = KnowledgeVersion(
            version=1,
            lu_id=lu_id,
            key_vector=key_vector,
            value_vector=value_vector,
            condition=condition,
            decision=decision,
            lifecycle_state='probationary',
            created_at=datetime.utcnow(),
            created_by=created_by,
            change_reason="Initial creation",
            metadata=metadata or {},
        )
        
        # 保存到持久化层
        await self._save_version(namespace, version)
        
        # 更新缓存
        self._add_to_cache(namespace, lu_id, version)
        
        logger.info(f"Created knowledge {lu_id} v1 in namespace {namespace}")
        return version
    
    async def update_knowledge(
        self,
        namespace: str,
        lu_id: str,
        key_vector: Optional[List[float]] = None,
        value_vector: Optional[List[float]] = None,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        updated_by: str = "system",
        change_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeVersion:
        """
        更新知识（创建新版本）
        
        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            key_vector: 新的 Key 向量（None 保持不变）
            value_vector: 新的 Value 向量（None 保持不变）
            condition: 新的条件文本（None 保持不变）
            decision: 新的决策文本（None 保持不变）
            lifecycle_state: 新的生命周期状态（None 保持不变）
            updated_by: 更新者
            change_reason: 变更原因
            metadata: 新的元数据（None 保持不变）
        
        Returns:
            新版本
        """
        # 获取当前版本
        current = await self.get_latest_version(namespace, lu_id)
        if current is None:
            raise ValueError(f"Knowledge {lu_id} not found in namespace {namespace}")
        
        # 创建新版本
        new_version = KnowledgeVersion(
            version=current.version + 1,
            lu_id=lu_id,
            key_vector=key_vector if key_vector is not None else current.key_vector,
            value_vector=value_vector if value_vector is not None else current.value_vector,
            condition=condition if condition is not None else current.condition,
            decision=decision if decision is not None else current.decision,
            lifecycle_state=lifecycle_state if lifecycle_state is not None else current.lifecycle_state,
            created_at=datetime.utcnow(),
            created_by=updated_by,
            change_reason=change_reason,
            metadata=metadata if metadata is not None else current.metadata,
        )
        
        # 保存
        await self._save_version(namespace, new_version)
        
        # 更新缓存
        self._add_to_cache(namespace, lu_id, new_version)
        
        logger.info(f"Updated knowledge {lu_id} to v{new_version.version} in namespace {namespace}")
        return new_version
    
    async def rollback(
        self,
        namespace: str,
        lu_id: str,
        target_version: int,
        rolled_back_by: str = "system",
    ) -> KnowledgeVersion:
        """
        回滚到指定版本
        
        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            target_version: 目标版本号
            rolled_back_by: 回滚操作者
        
        Returns:
            回滚后的新版本
        """
        # 获取目标版本
        target = await self.get_version(namespace, lu_id, target_version)
        if target is None:
            raise ValueError(f"Version {target_version} not found for {lu_id}")
        
        # 获取当前版本号
        current = await self.get_latest_version(namespace, lu_id)
        current_version = current.version if current else 0
        
        # 创建回滚版本
        rollback_version = KnowledgeVersion(
            version=current_version + 1,
            lu_id=lu_id,
            key_vector=target.key_vector,
            value_vector=target.value_vector,
            condition=target.condition,
            decision=target.decision,
            lifecycle_state=target.lifecycle_state,
            created_at=datetime.utcnow(),
            created_by=rolled_back_by,
            change_reason=f"Rollback to version {target_version}",
            metadata={
                **target.metadata,
                'rollback_from': current_version,
                'rollback_to': target_version,
            },
        )
        
        # 保存
        await self._save_version(namespace, rollback_version)
        
        # 更新缓存
        self._add_to_cache(namespace, lu_id, rollback_version)
        
        logger.info(f"Rolled back {lu_id} from v{current_version} to v{target_version} (new v{rollback_version.version})")
        return rollback_version
    
    async def get_latest_version(
        self,
        namespace: str,
        lu_id: str,
    ) -> Optional[KnowledgeVersion]:
        """获取最新版本"""
        history = await self.get_version_history(namespace, lu_id)
        if not history:
            return None
        return history[-1]
    
    async def get_version(
        self,
        namespace: str,
        lu_id: str,
        version: int,
    ) -> Optional[KnowledgeVersion]:
        """获取指定版本"""
        history = await self.get_version_history(namespace, lu_id)
        for v in history:
            if v.version == version:
                return v
        return None
    
    async def get_version_history(
        self,
        namespace: str,
        lu_id: str,
    ) -> List[KnowledgeVersion]:
        """获取版本历史"""
        # 检查缓存
        if namespace in self._version_cache and lu_id in self._version_cache[namespace]:
            return self._version_cache[namespace][lu_id]
        
        # 从持久化层加载
        history = await self._load_version_history(namespace, lu_id)
        
        # 更新缓存
        if namespace not in self._version_cache:
            self._version_cache[namespace] = {}
        self._version_cache[namespace][lu_id] = history
        
        return history
    
    async def compare_versions(
        self,
        namespace: str,
        lu_id: str,
        version1: int,
        version2: int,
    ) -> List[VersionDiff]:
        """
        比较两个版本的差异
        
        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            version1: 版本 1
            version2: 版本 2
        
        Returns:
            差异列表
        """
        v1 = await self.get_version(namespace, lu_id, version1)
        v2 = await self.get_version(namespace, lu_id, version2)
        
        if v1 is None or v2 is None:
            raise ValueError(f"Version not found")
        
        diffs = []
        
        # 比较各字段
        if v1.condition != v2.condition:
            diffs.append(VersionDiff('condition', v1.condition, v2.condition))
        
        if v1.decision != v2.decision:
            diffs.append(VersionDiff('decision', v1.decision, v2.decision))
        
        if v1.lifecycle_state != v2.lifecycle_state:
            diffs.append(VersionDiff('lifecycle_state', v1.lifecycle_state, v2.lifecycle_state))
        
        # 向量差异（使用余弦相似度）
        if v1.key_vector != v2.key_vector:
            import numpy as np
            sim = np.dot(v1.key_vector, v2.key_vector) / (
                np.linalg.norm(v1.key_vector) * np.linalg.norm(v2.key_vector) + 1e-10
            )
            diffs.append(VersionDiff('key_vector', f'similarity={sim:.4f}', 'changed'))
        
        if v1.value_vector != v2.value_vector:
            import numpy as np
            sim = np.dot(v1.value_vector, v2.value_vector) / (
                np.linalg.norm(v1.value_vector) * np.linalg.norm(v2.value_vector) + 1e-10
            )
            diffs.append(VersionDiff('value_vector', f'similarity={sim:.4f}', 'changed'))
        
        return diffs
    
    async def _save_version(self, namespace: str, version: KnowledgeVersion):
        """保存版本到持久化层"""
        # 保存当前版本到主存储
        from .base import KnowledgeRecord
        
        record = KnowledgeRecord(
            slot_idx=-1,  # 由 AGA 分配
            lu_id=version.lu_id,
            condition=version.condition,
            decision=version.decision,
            key_vector=version.key_vector,
            value_vector=version.value_vector,
            lifecycle_state=version.lifecycle_state,
            namespace=namespace,
            version=version.version,
            metadata={
                'version_info': version.to_dict(),
            },
        )
        
        await self.persistence.save_slot(namespace, record)
        
        # 保存版本历史（如果支持）
        if hasattr(self.persistence, 'save_audit_log'):
            await self.persistence.save_audit_log({
                'namespace': namespace,
                'lu_id': version.lu_id,
                'action': 'version_created',
                'new_state': version.lifecycle_state,
                'details': {
                    'version': version.version,
                    'created_by': version.created_by,
                    'change_reason': version.change_reason,
                },
            })
    
    async def _load_version_history(
        self,
        namespace: str,
        lu_id: str,
    ) -> List[KnowledgeVersion]:
        """从持久化层加载版本历史"""
        # 从审计日志加载历史
        if hasattr(self.persistence, 'query_audit_log'):
            logs = await self.persistence.query_audit_log(
                namespace=namespace,
                lu_id=lu_id,
                limit=self.max_versions_per_knowledge,
            )
            
            versions = []
            for log in logs:
                if log.get('action') == 'version_created':
                    details = log.get('details', {})
                    if 'version_info' in details:
                        versions.append(KnowledgeVersion.from_dict(details['version_info']))
            
            return sorted(versions, key=lambda v: v.version)
        
        # 回退：只返回当前版本
        record = await self.persistence.load_slot(namespace, lu_id)
        if record:
            metadata = record.metadata or {}
            if 'version_info' in metadata:
                return [KnowledgeVersion.from_dict(metadata['version_info'])]
            
            # 构造基本版本
            return [KnowledgeVersion(
                version=record.version or 1,
                lu_id=record.lu_id,
                key_vector=record.key_vector,
                value_vector=record.value_vector,
                condition=record.condition,
                decision=record.decision,
                lifecycle_state=record.lifecycle_state,
                created_at=datetime.fromisoformat(record.created_at) if record.created_at else datetime.utcnow(),
                created_by='system',
            )]
        
        return []
    
    def _add_to_cache(self, namespace: str, lu_id: str, version: KnowledgeVersion):
        """添加到缓存"""
        if namespace not in self._version_cache:
            self._version_cache[namespace] = {}
        if lu_id not in self._version_cache[namespace]:
            self._version_cache[namespace][lu_id] = []
        
        history = self._version_cache[namespace][lu_id]
        history.append(version)
        
        # 限制历史长度
        if len(history) > self.max_versions_per_knowledge:
            self._version_cache[namespace][lu_id] = history[-self.max_versions_per_knowledge:]
    
    def clear_cache(self, namespace: Optional[str] = None):
        """清除缓存"""
        if namespace:
            self._version_cache.pop(namespace, None)
        else:
            self._version_cache.clear()
