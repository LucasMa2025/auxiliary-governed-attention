"""
aga-knowledge 持久化适配器抽象基类

定义统一的持久化接口（明文 KV 版本）。
移除了向量化 KV 注入，保留明文 condition/decision 文本对。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..types import KnowledgeRecord, LifecycleState, TrustTier, TRUST_TIER_PRIORITY


class PersistenceAdapter(ABC):
    """
    持久化适配器抽象基类（明文 KV 版本）

    定义统一的持久化接口，支持：
    - 同步和异步操作
    - 单条和批量操作
    - 生命周期管理
    - 统计和审计
    - 命名空间隔离

    注意：此版本不包含向量数据，只存储明文 condition/decision。
    """

    # ==================== 连接管理 ====================

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接"""
        ...

    @abstractmethod
    async def disconnect(self):
        """断开连接"""
        ...

    @abstractmethod
    async def is_connected(self) -> bool:
        """检查连接状态"""
        ...

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...

    # ==================== 知识 CRUD ====================

    @abstractmethod
    async def save_knowledge(self, namespace: str, lu_id: str, data: Dict[str, Any]) -> bool:
        """
        保存知识

        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID
            data: 知识数据字典，包含:
                - condition: str — 触发条件描述
                - decision: str — 决策描述
                - lifecycle_state: str — 生命周期状态
                - trust_tier: str — 信任层级
                - metadata: dict — 扩展元数据

        Returns:
            是否成功
        """
        ...

    @abstractmethod
    async def load_knowledge(self, namespace: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """
        加载知识

        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID

        Returns:
            知识数据字典或 None
        """
        ...

    @abstractmethod
    async def delete_knowledge(self, namespace: str, lu_id: str) -> bool:
        """
        删除知识

        Args:
            namespace: 命名空间
            lu_id: 知识单元 ID

        Returns:
            是否成功
        """
        ...

    @abstractmethod
    async def knowledge_exists(self, namespace: str, lu_id: str) -> bool:
        """检查知识是否存在"""
        ...

    # ==================== 批量操作 ====================

    @abstractmethod
    async def save_batch(self, namespace: str, records: List[Dict[str, Any]]) -> int:
        """
        批量保存

        Args:
            namespace: 命名空间
            records: 知识数据列表

        Returns:
            成功保存的数量
        """
        ...

    @abstractmethod
    async def load_active_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        """加载活跃知识（非 QUARANTINED）"""
        ...

    @abstractmethod
    async def load_all_knowledge(self, namespace: str) -> List[Dict[str, Any]]:
        """加载所有知识"""
        ...

    # ==================== 查询 ====================

    @abstractmethod
    async def query_knowledge(
        self,
        namespace: str,
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        查询知识列表

        Args:
            namespace: 命名空间
            lifecycle_states: 状态过滤
            trust_tiers: 信任层级过滤
            limit: 限制数量
            offset: 偏移量

        Returns:
            知识数据列表
        """
        ...

    # ==================== 生命周期管理 ====================

    @abstractmethod
    async def update_lifecycle(
        self,
        namespace: str,
        lu_id: str,
        new_state: str,
    ) -> bool:
        """更新生命周期状态"""
        ...

    @abstractmethod
    async def update_trust_tier(
        self,
        namespace: str,
        lu_id: str,
        new_tier: str,
    ) -> bool:
        """更新信任层级"""
        ...

    # ==================== 统计 ====================

    @abstractmethod
    async def get_knowledge_count(
        self,
        namespace: str,
        state: Optional[str] = None,
    ) -> int:
        """获取知识数量"""
        ...

    @abstractmethod
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        ...

    @abstractmethod
    async def increment_hit_count(
        self,
        namespace: str,
        lu_ids: List[str],
    ) -> bool:
        """批量增加命中计数"""
        ...

    # ==================== 命名空间 ====================

    @abstractmethod
    async def get_namespaces(self) -> List[str]:
        """获取所有命名空间"""
        ...

    # ==================== 审计日志 ====================

    async def save_audit_log(self, entry: Dict[str, Any]) -> bool:
        """保存审计日志（默认不保存）"""
        return True

    async def query_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """查询审计日志（默认返回空）"""
        return []
