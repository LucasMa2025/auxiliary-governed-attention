"""
AGA API 客户端

为外部治理系统提供的 HTTP 客户端库。

用途说明：
=========

本模块提供同步和异步 HTTP 客户端，供**外部治理系统**调用 AGA API。

使用场景：
1. 持续学习系统：将 Learning Unit 写入 AGA
2. 监控系统：查询知识状态和统计
3. 治理控制台：管理知识生命周期
4. 自动化测试：测试 AGA API 功能

注意：
- 这是**外部系统**使用的客户端库
- AGA API 服务本身**不需要**此模块
- 此模块独立于 API 服务运行

架构位置：
=========

    ┌─────────────────────────────────────────────┐
    │          外部治理系统                        │
    │  ┌─────────────────────────────────────┐   │
    │  │     AGAClient / AsyncAGAClient      │   │  <-- 本模块
    │  └─────────────────────────────────────┘   │
    └─────────────────────────────────────────────┘
                        │ HTTP
                        ▼
    ┌─────────────────────────────────────────────┐
    │              AGA API 服务                    │
    │  ┌──────────┐  ┌───────────┐  ┌─────────┐  │
    │  │  路由层  │→│  服务层   │→│ 持久化  │  │
    │  └──────────┘  └───────────┘  └─────────┘  │
    └─────────────────────────────────────────────┘

使用示例：
=========

同步客户端:
    from aga.api import AGAClient
    
    # 创建客户端
    client = AGAClient("http://localhost:8081")
    
    # 健康检查
    health = client.health_check()
    print(f"Status: {health['status']}")
    
    # 注入知识
    result = client.inject_knowledge(
        lu_id="LU_001",
        condition="capital of France",
        decision="Paris",
        key_vector=[...],  # 64 维向量
        value_vector=[...],  # 4096 维向量
    )
    print(f"Injected to slot: {result['slot_idx']}")
    
    # 确认知识
    client.update_lifecycle("LU_001", "confirmed")
    
    # 查询知识
    knowledge = client.query_knowledge(namespace="default", limit=10)
    
    # 关闭客户端
    client.close()

异步客户端:
    from aga.api import AsyncAGAClient
    import asyncio
    
    async def main():
        async with AsyncAGAClient("http://localhost:8081") as client:
            # 健康检查
            health = await client.health_check()
            
            # 批量注入
            result = await client.batch_inject([
                {
                    "lu_id": "LU_001",
                    "condition": "...",
                    "decision": "...",
                    "key_vector": [...],
                    "value_vector": [...],
                },
                # ... more items
            ])
            print(f"Injected: {result['success_count']}/{result['total']}")
    
    asyncio.run(main())

作为上下文管理器:
    with AGAClient("http://localhost:8081") as client:
        client.inject_knowledge(...)
"""
from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class AGAClient:
    """
    AGA HTTP 同步客户端
    
    为外部治理系统提供的同步 HTTP 客户端。
    
    Attributes:
        base_url: API 基础 URL
        timeout: 请求超时时间（秒）
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        初始化客户端
        
        Args:
            base_url: API 基础 URL，如 "http://localhost:8081"
            timeout: 请求超时时间（秒）
            api_key: API 密钥（可选，用于认证）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        
        # 尝试使用 httpx，回退到 urllib
        try:
            import httpx
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                headers=self._get_headers(),
            )
            self._use_httpx = True
        except ImportError:
            self._client = None
            self._use_httpx = False
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """发送请求"""
        if self._use_httpx:
            if method == "GET":
                response = self._client.get(path, params=params)
            elif method == "POST":
                response = self._client.post(path, json=json_data, params=params)
            elif method == "DELETE":
                response = self._client.delete(path, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        else:
            import urllib.request
            import urllib.parse
            
            url = f"{self.base_url}{path}"
            if params:
                url += "?" + urllib.parse.urlencode(params)
            
            if json_data:
                data = json.dumps(json_data).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers=self._get_headers(),
                    method=method,
                )
            else:
                req = urllib.request.Request(
                    url,
                    headers=self._get_headers(),
                    method=method,
                )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
    
    def close(self):
        """关闭客户端"""
        if self._use_httpx and self._client:
            self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # ==================== 健康检查 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息，包含 status, version, uptime_seconds 等
        """
        return self._request("GET", "/health")
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取服务配置
        
        Returns:
            服务配置信息
        """
        return self._request("GET", "/config")
    
    # ==================== 命名空间 ====================
    
    def list_namespaces(self) -> List[str]:
        """
        列出所有命名空间
        
        Returns:
            命名空间列表
        """
        result = self._request("GET", "/namespaces")
        return result.get("namespaces", [])
    
    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        删除命名空间
        
        Args:
            namespace: 命名空间名称
        
        Returns:
            删除结果
        """
        return self._request("DELETE", f"/namespaces/{namespace}")
    
    # ==================== 知识管理 ====================
    
    def inject_knowledge(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        注入知识
        
        Args:
            lu_id: Learning Unit ID（唯一标识）
            condition: 触发条件描述
            decision: 决策/动作描述
            key_vector: 条件编码向量（维度=bottleneck_dim）
            value_vector: 决策编码向量（维度=hidden_dim）
            namespace: 命名空间（默认 "default"）
            lifecycle_state: 初始生命周期状态
            trust_tier: 信任层级
            metadata: 扩展元数据
        
        Returns:
            注入结果，包含 success, slot_idx, timestamp
        """
        return self._request("POST", "/knowledge/inject", {
            "lu_id": lu_id,
            "namespace": namespace,
            "condition": condition,
            "decision": decision,
            "key_vector": key_vector,
            "value_vector": value_vector,
            "lifecycle_state": lifecycle_state,
            "trust_tier": trust_tier,
            "metadata": metadata,
        })
    
    def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        批量注入知识
        
        Args:
            items: 知识列表，每项包含 lu_id, condition, decision, key_vector, value_vector
            namespace: 默认命名空间
            skip_duplicates: 是否跳过重复 LU ID
        
        Returns:
            批量结果，包含 total, success_count, failed_count, results
        """
        return self._request("POST", "/knowledge/inject/batch", {
            "items": items,
            "namespace": namespace,
            "skip_duplicates": skip_duplicates,
        })
    
    def get_knowledge(
        self,
        namespace: str,
        lu_id: str,
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        获取单条知识
        
        Args:
            namespace: 命名空间
            lu_id: Learning Unit ID
            include_vectors: 是否包含向量数据
        
        Returns:
            知识详情，不存在则返回 None
        """
        try:
            return self._request(
                "GET",
                f"/knowledge/{namespace}/{lu_id}",
                params={"include_vectors": include_vectors},
            )
        except Exception as e:
            if "404" in str(e):
                return None
            raise
    
    def query_knowledge(
        self,
        namespace: str = "default",
        lifecycle_states: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        查询知识列表
        
        Args:
            namespace: 命名空间
            lifecycle_states: 筛选状态列表
            limit: 返回数量限制
            offset: 偏移量
            include_vectors: 是否包含向量数据
        
        Returns:
            知识列表
        """
        return self._request("POST", "/knowledge/query", {
            "namespace": namespace,
            "lifecycle_states": lifecycle_states,
            "limit": limit,
            "offset": offset,
            "include_vectors": include_vectors,
        })
    
    def delete_knowledge(self, namespace: str, lu_id: str) -> Dict[str, Any]:
        """
        删除知识
        
        Args:
            namespace: 命名空间
            lu_id: Learning Unit ID
        
        Returns:
            删除结果
        """
        return self._request("DELETE", f"/knowledge/{namespace}/{lu_id}")
    
    # ==================== 生命周期管理 ====================
    
    def update_lifecycle(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        更新生命周期状态
        
        Args:
            lu_id: Learning Unit ID
            new_state: 新状态 (probationary, confirmed, deprecated, quarantined)
            namespace: 命名空间
            reason: 变更原因
        
        Returns:
            更新结果
        """
        return self._request("POST", "/lifecycle/update", {
            "lu_id": lu_id,
            "namespace": namespace,
            "new_state": new_state,
            "reason": reason,
        })
    
    def batch_update_lifecycle(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量更新生命周期状态
        
        Args:
            updates: 更新列表，每项包含 lu_id, namespace, new_state, reason
        
        Returns:
            批量结果
        """
        return self._request("POST", "/lifecycle/update/batch", {
            "updates": updates,
        })
    
    def quarantine_knowledge(
        self,
        lu_id: str,
        reason: str,
        namespace: str = "default",
        source_instance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        隔离知识
        
        立即将知识标记为 QUARANTINED 状态，不再参与推理。
        
        Args:
            lu_id: Learning Unit ID
            reason: 隔离原因
            namespace: 命名空间
            source_instance: 来源实例
        
        Returns:
            隔离结果
        """
        return self._request("POST", "/lifecycle/quarantine", {
            "lu_id": lu_id,
            "namespace": namespace,
            "reason": reason,
            "source_instance": source_instance,
        })
    
    # ==================== 槽位管理 ====================
    
    def find_free_slot(self, namespace: str = "default") -> Optional[int]:
        """
        查找空闲槽位
        
        Args:
            namespace: 命名空间
        
        Returns:
            空闲槽位索引，无空闲则返回 None
        """
        result = self._request("GET", f"/slots/{namespace}/free")
        return result.get("free_slot")
    
    def get_slot_info(self, namespace: str, slot_idx: int) -> Dict[str, Any]:
        """
        获取槽位信息
        
        Args:
            namespace: 命名空间
            slot_idx: 槽位索引
        
        Returns:
            槽位详细信息
        """
        return self._request("GET", f"/slots/{namespace}/{slot_idx}")
    
    # ==================== 统计 ====================
    
    def get_statistics(self, namespace: str = "default") -> Dict[str, Any]:
        """
        获取命名空间统计
        
        Args:
            namespace: 命名空间
        
        Returns:
            统计信息
        """
        return self._request("GET", f"/statistics/{namespace}")
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        获取所有命名空间统计
        
        Returns:
            所有命名空间的统计信息
        """
        return self._request("GET", "/statistics")
    
    def get_writer_statistics(self) -> Dict[str, Any]:
        """
        获取写入器统计
        
        Returns:
            写入器统计信息
        """
        return self._request("GET", "/statistics/writer")
    
    # ==================== 审计日志 ====================
    
    def get_audit_log(
        self,
        namespace: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        获取审计日志
        
        Args:
            namespace: 命名空间
            limit: 返回数量
            offset: 偏移量
        
        Returns:
            审计日志列表
        """
        return self._request(
            "GET",
            f"/audit/{namespace}",
            params={"limit": limit, "offset": offset},
        )


class AsyncAGAClient:
    """
    AGA HTTP 异步客户端
    
    为外部治理系统提供的异步 HTTP 客户端。
    适用于异步应用程序（如 FastAPI、aiohttp 应用）。
    
    Attributes:
        base_url: API 基础 URL
        timeout: 请求超时时间（秒）
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        初始化异步客户端
        
        Args:
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            api_key: API 密钥（可选）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def _ensure_client(self):
        """确保客户端已初始化"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers=self._get_headers(),
                )
            except ImportError:
                raise ImportError(
                    "httpx is required for async client. "
                    "Install with: pip install httpx"
                )
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        await self._ensure_client()
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """发送异步请求"""
        await self._ensure_client()
        
        if method == "GET":
            response = await self._client.get(path, params=params)
        elif method == "POST":
            response = await self._client.post(path, json=json_data, params=params)
        elif method == "DELETE":
            response = await self._client.delete(path, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return await self._request("GET", "/health")
    
    async def get_config(self) -> Dict[str, Any]:
        """获取服务配置"""
        return await self._request("GET", "/config")
    
    # ==================== 命名空间 ====================
    
    async def list_namespaces(self) -> List[str]:
        """列出所有命名空间"""
        result = await self._request("GET", "/namespaces")
        return result.get("namespaces", [])
    
    async def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """删除命名空间"""
        return await self._request("DELETE", f"/namespaces/{namespace}")
    
    # ==================== 知识管理 ====================
    
    async def inject_knowledge(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """注入知识"""
        return await self._request("POST", "/knowledge/inject", {
            "lu_id": lu_id,
            "namespace": namespace,
            "condition": condition,
            "decision": decision,
            "key_vector": key_vector,
            "value_vector": value_vector,
            "lifecycle_state": lifecycle_state,
            "trust_tier": trust_tier,
            "metadata": metadata,
        })
    
    async def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """批量注入知识"""
        return await self._request("POST", "/knowledge/inject/batch", {
            "items": items,
            "namespace": namespace,
            "skip_duplicates": skip_duplicates,
        })
    
    async def get_knowledge(
        self,
        namespace: str,
        lu_id: str,
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """获取单条知识"""
        try:
            return await self._request(
                "GET",
                f"/knowledge/{namespace}/{lu_id}",
                params={"include_vectors": include_vectors},
            )
        except Exception as e:
            if "404" in str(e):
                return None
            raise
    
    async def query_knowledge(
        self,
        namespace: str = "default",
        lifecycle_states: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """查询知识列表"""
        return await self._request("POST", "/knowledge/query", {
            "namespace": namespace,
            "lifecycle_states": lifecycle_states,
            "limit": limit,
            "offset": offset,
            "include_vectors": include_vectors,
        })
    
    async def delete_knowledge(self, namespace: str, lu_id: str) -> Dict[str, Any]:
        """删除知识"""
        return await self._request("DELETE", f"/knowledge/{namespace}/{lu_id}")
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """更新生命周期状态"""
        return await self._request("POST", "/lifecycle/update", {
            "lu_id": lu_id,
            "namespace": namespace,
            "new_state": new_state,
            "reason": reason,
        })
    
    async def batch_update_lifecycle(
        self,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """批量更新生命周期状态"""
        return await self._request("POST", "/lifecycle/update/batch", {
            "updates": updates,
        })
    
    async def quarantine_knowledge(
        self,
        lu_id: str,
        reason: str,
        namespace: str = "default",
        source_instance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """隔离知识"""
        return await self._request("POST", "/lifecycle/quarantine", {
            "lu_id": lu_id,
            "namespace": namespace,
            "reason": reason,
            "source_instance": source_instance,
        })
    
    # ==================== 槽位管理 ====================
    
    async def find_free_slot(self, namespace: str = "default") -> Optional[int]:
        """查找空闲槽位"""
        result = await self._request("GET", f"/slots/{namespace}/free")
        return result.get("free_slot")
    
    async def get_slot_info(self, namespace: str, slot_idx: int) -> Dict[str, Any]:
        """获取槽位信息"""
        return await self._request("GET", f"/slots/{namespace}/{slot_idx}")
    
    # ==================== 统计 ====================
    
    async def get_statistics(self, namespace: str = "default") -> Dict[str, Any]:
        """获取命名空间统计"""
        return await self._request("GET", f"/statistics/{namespace}")
    
    async def get_all_statistics(self) -> Dict[str, Any]:
        """获取所有命名空间统计"""
        return await self._request("GET", "/statistics")
    
    async def get_writer_statistics(self) -> Dict[str, Any]:
        """获取写入器统计"""
        return await self._request("GET", "/statistics/writer")
    
    # ==================== 审计日志 ====================
    
    async def get_audit_log(
        self,
        namespace: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """获取审计日志"""
        return await self._request(
            "GET",
            f"/audit/{namespace}",
            params={"limit": limit, "offset": offset},
        )
