"""
aga-knowledge 知识版本控制模块

明文 KV 版本：提供知识的版本管理功能。
不包含向量数据，只管理 condition/decision 文本对的版本历史。

特性:
- 版本历史记录
- 版本回滚
- 变更审计
- 差异比较
"""

import json
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeVersion:
    """知识版本（明文 KV 版本，无向量字段）"""
    version: int
    lu_id: str
    condition: str
    decision: str
    lifecycle_state: str
    trust_tier: str
    created_at: datetime
    created_by: str
    change_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "lu_id": self.lu_id,
            "condition": self.condition,
            "decision": self.decision,
            "lifecycle_state": self.lifecycle_state,
            "trust_tier": self.trust_tier,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "change_reason": self.change_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeVersion":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        return cls(
            version=data.get("version", 1),
            lu_id=data["lu_id"],
            condition=data.get("condition", ""),
            decision=data.get("decision", ""),
            lifecycle_state=data.get("lifecycle_state", "probationary"),
            trust_tier=data.get("trust_tier", "standard"),
            created_at=created_at,
            created_by=data.get("created_by", "system"),
            change_reason=data.get("change_reason"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VersionDiff:
    """版本差异"""
    field: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


class VersionedKnowledgeStore:
    """
    版本化知识存储（明文 KV 版本）

    提供知识的版本管理功能，支持版本历史、回滚和审计。
    不包含向量数据，只管理 condition/decision 文本对。

    使用示例:
        ```python
        store = VersionedKnowledgeStore(persistence_adapter)

        # 创建知识
        version = await store.create_knowledge(
            namespace="default",
            lu_id="rule_001",
            condition="当用户询问天气时",
            decision="提供天气信息",
            created_by="admin",
        )

        # 更新知识
        new_version = await store.update_knowledge(
            namespace="default",
            lu_id="rule_001",
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

    def __init__(
        self,
        persistence_adapter,
        max_versions_per_knowledge: int = 10,
    ):
        """
        初始化版本化存储

        Args:
            persistence_adapter: 持久化适配器（PersistenceAdapter 实例）
            max_versions_per_knowledge: 每个知识最多保留的版本数
        """
        self.persistence = persistence_adapter
        self.max_versions_per_knowledge = max_versions_per_knowledge

        # 版本历史缓存 (namespace -> lu_id -> [versions])
        self._version_cache: Dict[str, Dict[str, List[KnowledgeVersion]]] = {}
        self._cache_lock = threading.RLock()

    async def create_knowledge(
        self,
        namespace: str,
        lu_id: str,
        condition: str,
        decision: str,
        created_by: str,
        trust_tier: str = "standard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeVersion:
        """
        创建新知识

        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            condition: 条件文本
            decision: 决策文本
            created_by: 创建者
            trust_tier: 信任层级
            metadata: 元数据

        Returns:
            创建的版本
        """
        version = KnowledgeVersion(
            version=1,
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            lifecycle_state="probationary",
            trust_tier=trust_tier,
            created_at=datetime.utcnow(),
            created_by=created_by,
            change_reason="Initial creation",
            metadata=metadata or {},
        )

        # 保存到持久化层
        await self._save_version(namespace, version)

        # 更新缓存
        self._add_to_cache(namespace, lu_id, version)

        logger.info(
            f"Created knowledge {lu_id} v1 in namespace {namespace}"
        )
        return version

    async def update_knowledge(
        self,
        namespace: str,
        lu_id: str,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        trust_tier: Optional[str] = None,
        updated_by: str = "system",
        change_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeVersion:
        """
        更新知识（创建新版本）

        Args:
            namespace: 命名空间
            lu_id: 知识 ID
            condition: 新的条件文本（None 保持不变）
            decision: 新的决策文本（None 保持不变）
            lifecycle_state: 新的生命周期状态（None 保持不变）
            trust_tier: 新的信任层级（None 保持不变）
            updated_by: 更新者
            change_reason: 变更原因
            metadata: 新的元数据（None 保持不变）

        Returns:
            新版本
        """
        current = await self.get_latest_version(namespace, lu_id)
        if current is None:
            raise ValueError(
                f"Knowledge {lu_id} not found in namespace {namespace}"
            )

        new_version = KnowledgeVersion(
            version=current.version + 1,
            lu_id=lu_id,
            condition=(
                condition if condition is not None else current.condition
            ),
            decision=(
                decision if decision is not None else current.decision
            ),
            lifecycle_state=(
                lifecycle_state
                if lifecycle_state is not None
                else current.lifecycle_state
            ),
            trust_tier=(
                trust_tier if trust_tier is not None else current.trust_tier
            ),
            created_at=datetime.utcnow(),
            created_by=updated_by,
            change_reason=change_reason,
            metadata=(
                metadata if metadata is not None else current.metadata
            ),
        )

        await self._save_version(namespace, new_version)
        self._add_to_cache(namespace, lu_id, new_version)

        logger.info(
            f"Updated knowledge {lu_id} to v{new_version.version} "
            f"in namespace {namespace}"
        )
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
        target = await self.get_version(namespace, lu_id, target_version)
        if target is None:
            raise ValueError(
                f"Version {target_version} not found for {lu_id}"
            )

        current = await self.get_latest_version(namespace, lu_id)
        current_version = current.version if current else 0

        rollback_version = KnowledgeVersion(
            version=current_version + 1,
            lu_id=lu_id,
            condition=target.condition,
            decision=target.decision,
            lifecycle_state=target.lifecycle_state,
            trust_tier=target.trust_tier,
            created_at=datetime.utcnow(),
            created_by=rolled_back_by,
            change_reason=f"Rollback to version {target_version}",
            metadata={
                **target.metadata,
                "rollback_from": current_version,
                "rollback_to": target_version,
            },
        )

        await self._save_version(namespace, rollback_version)
        self._add_to_cache(namespace, lu_id, rollback_version)

        logger.info(
            f"Rolled back {lu_id} from v{current_version} to "
            f"v{target_version} (new v{rollback_version.version})"
        )
        return rollback_version

    async def get_latest_version(
        self, namespace: str, lu_id: str
    ) -> Optional[KnowledgeVersion]:
        """获取最新版本"""
        history = await self.get_version_history(namespace, lu_id)
        if not history:
            return None
        return history[-1]

    async def get_version(
        self, namespace: str, lu_id: str, version: int
    ) -> Optional[KnowledgeVersion]:
        """获取指定版本"""
        history = await self.get_version_history(namespace, lu_id)
        for v in history:
            if v.version == version:
                return v
        return None

    async def get_version_history(
        self, namespace: str, lu_id: str
    ) -> List[KnowledgeVersion]:
        """获取版本历史"""
        with self._cache_lock:
            if (
                namespace in self._version_cache
                and lu_id in self._version_cache[namespace]
            ):
                return self._version_cache[namespace][lu_id]

        # 从持久化层加载
        history = await self._load_version_history(namespace, lu_id)

        # 更新缓存
        with self._cache_lock:
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
            raise ValueError("Version not found")

        diffs = []

        if v1.condition != v2.condition:
            diffs.append(
                VersionDiff("condition", v1.condition, v2.condition)
            )

        if v1.decision != v2.decision:
            diffs.append(
                VersionDiff("decision", v1.decision, v2.decision)
            )

        if v1.lifecycle_state != v2.lifecycle_state:
            diffs.append(
                VersionDiff(
                    "lifecycle_state",
                    v1.lifecycle_state,
                    v2.lifecycle_state,
                )
            )

        if v1.trust_tier != v2.trust_tier:
            diffs.append(
                VersionDiff("trust_tier", v1.trust_tier, v2.trust_tier)
            )

        if v1.metadata != v2.metadata:
            diffs.append(
                VersionDiff("metadata", v1.metadata, v2.metadata)
            )

        return diffs

    async def _save_version(
        self, namespace: str, version: KnowledgeVersion
    ):
        """保存版本到持久化层"""
        # 保存当前版本到主存储
        data = {
            "condition": version.condition,
            "decision": version.decision,
            "lifecycle_state": version.lifecycle_state,
            "trust_tier": version.trust_tier,
            "metadata": {
                "version_info": version.to_dict(),
            },
        }

        await self.persistence.save_knowledge(
            namespace, version.lu_id, data
        )

        # 保存版本历史到审计日志
        if hasattr(self.persistence, "save_audit_log"):
            await self.persistence.save_audit_log({
                "namespace": namespace,
                "lu_id": version.lu_id,
                "action": "version_created",
                "new_state": version.lifecycle_state,
                "details": json.dumps({
                    "version": version.version,
                    "created_by": version.created_by,
                    "change_reason": version.change_reason,
                    "version_info": version.to_dict(),
                }),
            })

    async def _load_version_history(
        self, namespace: str, lu_id: str
    ) -> List[KnowledgeVersion]:
        """从持久化层加载版本历史"""
        # 从审计日志加载历史
        if hasattr(self.persistence, "query_audit_log"):
            logs = await self.persistence.query_audit_log(
                namespace=namespace,
                lu_id=lu_id,
                limit=self.max_versions_per_knowledge,
            )

            versions = []
            for log_entry in logs:
                if log_entry.get("action") == "version_created":
                    details = log_entry.get("details", {})
                    # details 可能是 JSON 字符串
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    if (
                        isinstance(details, dict)
                        and "version_info" in details
                    ):
                        versions.append(
                            KnowledgeVersion.from_dict(
                                details["version_info"]
                            )
                        )

            if versions:
                return sorted(versions, key=lambda v: v.version)

        # 回退：只返回当前版本
        record = await self.persistence.load_knowledge(namespace, lu_id)
        if record:
            metadata = record.get("metadata") or {}
            if "version_info" in metadata:
                return [
                    KnowledgeVersion.from_dict(metadata["version_info"])
                ]

            # 构造基本版本
            return [
                KnowledgeVersion(
                    version=record.get("version", 1),
                    lu_id=record.get("lu_id", lu_id),
                    condition=record.get("condition", ""),
                    decision=record.get("decision", ""),
                    lifecycle_state=record.get(
                        "lifecycle_state", "probationary"
                    ),
                    trust_tier=record.get("trust_tier", "standard"),
                    created_at=(
                        datetime.fromisoformat(record["created_at"])
                        if record.get("created_at")
                        else datetime.utcnow()
                    ),
                    created_by="system",
                )
            ]

        return []

    def _add_to_cache(
        self,
        namespace: str,
        lu_id: str,
        version: KnowledgeVersion,
    ):
        """添加到缓存"""
        with self._cache_lock:
            if namespace not in self._version_cache:
                self._version_cache[namespace] = {}
            if lu_id not in self._version_cache[namespace]:
                self._version_cache[namespace][lu_id] = []

            history = self._version_cache[namespace][lu_id]
            history.append(version)

            # 限制历史长度
            if len(history) > self.max_versions_per_knowledge:
                self._version_cache[namespace][lu_id] = history[
                    -self.max_versions_per_knowledge:
                ]

    def clear_cache(self, namespace: Optional[str] = None):
        """清除缓存"""
        with self._cache_lock:
            if namespace:
                self._version_cache.pop(namespace, None)
            else:
                self._version_cache.clear()
