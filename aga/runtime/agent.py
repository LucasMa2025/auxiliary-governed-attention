"""
AGA Runtime Agent

同步代理，管理 Runtime 的生命周期和同步。
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..config.runtime import RuntimeConfig
from ..sync import SyncSubscriber, SyncMessage, MessageType
from .cache import LocalCache
from .aga_runtime import AGARuntime, AGARuntimeLayer


class RuntimeAgent:
    """
    Runtime 同步代理
    
    负责：
    - 从 Portal 同步知识
    - 管理本地 AGA 模块
    - 上报状态
    
    使用示例：
    
    ```python
    from aga.runtime import RuntimeAgent
    from aga.config import RuntimeConfig
    
    # 创建配置
    config = RuntimeConfig.for_production(
        instance_id="runtime-001",
        portal_url="http://portal:8081",
        redis_host="localhost",
    )
    
    # 创建代理
    agent = RuntimeAgent(config)
    
    # 初始化
    await agent.initialize()
    
    # 附加到模型层
    aga_layer = agent.attach_to_layer(transformer_layer)
    
    # 启动同步
    await agent.start()
    
    # 使用（在推理循环中）
    output = aga_layer(hidden_states, attention_mask)
    
    # 停止
    await agent.stop()
    ```
    """
    
    def __init__(self, config: RuntimeConfig):
        """
        初始化代理
        
        Args:
            config: Runtime 配置
        """
        self.config = config
        self.instance_id = config.instance_id
        self.namespaces = config.namespaces
        
        # 组件（延迟初始化）
        self._subscriber: Optional[SyncSubscriber] = None
        self._cache: Optional[LocalCache] = None
        self._runtime: Optional[AGARuntime] = None
        
        # 状态
        self._initialized = False
        self._running = False
        self._start_time = time.time()
        
        # 统计
        self._stats = {
            "messages_received": 0,
            "inject_count": 0,
            "update_count": 0,
            "quarantine_count": 0,
            "errors": 0,
        }
        
        # 回调
        self._on_inject_callbacks: List[Callable] = []
        self._on_update_callbacks: List[Callable] = []
    
    async def initialize(self):
        """初始化代理"""
        if self._initialized:
            return
        
        # 创建缓存
        self._cache = LocalCache(
            max_slots=self.config.aga.num_slots,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        
        # 创建 Runtime
        self._runtime = AGARuntime(
            config=self.config.aga,
            cache=self._cache,
            namespace=self.config.namespace,
        )
        
        # 创建订阅器
        sync_config = self.config.sync
        
        # 验证 backend 配置
        valid_backends = {"redis", "kafka", "memory"}
        if sync_config.backend not in valid_backends:
            logger.warning(
                f"Unknown backend '{sync_config.backend}', falling back to 'memory'. "
                f"Valid options: {valid_backends}"
            )
            sync_config.backend = "memory"
        
        if sync_config.backend == "redis":
            self._subscriber = SyncSubscriber(
                backend_type="redis",
                channel=sync_config.redis_channel,
                instance_id=self.instance_id,
                host=sync_config.redis_host,
                port=sync_config.redis_port,
                db=sync_config.redis_db,
                password=sync_config.redis_password,
            )
        elif sync_config.backend == "kafka":
            self._subscriber = SyncSubscriber(
                backend_type="kafka",
                channel=sync_config.kafka_topic,
                instance_id=self.instance_id,
                bootstrap_servers=sync_config.kafka_bootstrap_servers,
            )
        else:
            self._subscriber = SyncSubscriber(
                backend_type="memory",
                channel=sync_config.redis_channel,
                instance_id=self.instance_id,
            )
        
        # 注册处理器
        self._subscriber.on_inject(self._handle_inject)
        self._subscriber.on_update(self._handle_update)
        self._subscriber.on_quarantine(self._handle_quarantine)
        self._subscriber.on_delete(self._handle_delete)
        self._subscriber.on_batch_inject(self._handle_batch_inject)
        self._subscriber.on_full_sync(self._handle_full_sync)
        
        self._initialized = True
        logger.info(f"RuntimeAgent initialized: {self.instance_id}")
    
    async def start(self):
        """启动同步"""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            return
        
        # 连接订阅器
        await self._subscriber.connect()
        await self._subscriber.start()
        
        # 注册到 Portal
        await self._register_to_portal()
        
        # 启动同步
        if self.config.sync.sync_on_start:
            await self._initial_sync()
        
        # 启动心跳
        asyncio.create_task(self._heartbeat_loop())
        
        self._running = True
        logger.info(f"RuntimeAgent started: {self.instance_id}")
    
    async def stop(self):
        """停止同步"""
        self._running = False
        
        if self._subscriber:
            try:
                await self._subscriber.stop()
                await self._subscriber.disconnect()
            except Exception as e:
                logger.warning(f"Error stopping subscriber: {e}")
        
        # 注销（允许失败）
        try:
            await self._deregister_from_portal()
        except Exception as e:
            logger.warning(f"Error deregistering from portal: {e}")
        
        logger.info(f"RuntimeAgent stopped: {self.instance_id}")
    
    # ==================== 消息处理 ====================
    
    async def _handle_inject(self, message: SyncMessage):
        """处理注入消息"""
        self._stats["messages_received"] += 1
        
        # 检查命名空间
        if message.namespace not in self.namespaces:
            return
        
        try:
            slot_idx = self._cache.add(
                lu_id=message.lu_id,
                key_vector=message.key_vector,
                value_vector=message.value_vector,
                namespace=message.namespace,
                condition=message.condition,
                decision=message.decision,
                lifecycle_state=message.lifecycle_state or "probationary",
                trust_tier=message.trust_tier,
            )
            
            if slot_idx is not None:
                self._stats["inject_count"] += 1
                logger.info(f"Injected {message.lu_id} to slot {slot_idx}")
                
                # 触发回调
                for callback in self._on_inject_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message.lu_id, slot_idx)
                        else:
                            callback(message.lu_id, slot_idx)
                    except Exception as e:
                        logger.error(f"Inject callback error: {e}")
            else:
                logger.warning(f"Failed to inject {message.lu_id}: cache full")
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Inject error: {e}")
    
    async def _handle_update(self, message: SyncMessage):
        """处理更新消息"""
        self._stats["messages_received"] += 1
        
        if message.namespace not in self.namespaces:
            return
        
        try:
            success = self._cache.update(
                lu_id=message.lu_id,
                lifecycle_state=message.lifecycle_state,
                trust_tier=message.trust_tier,
            )
            
            if success:
                self._stats["update_count"] += 1
                logger.info(f"Updated {message.lu_id} to {message.lifecycle_state}")
                
                # 触发回调
                for callback in self._on_update_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message.lu_id, message.lifecycle_state)
                        else:
                            callback(message.lu_id, message.lifecycle_state)
                    except Exception as e:
                        logger.error(f"Update callback error: {e}")
            else:
                logger.warning(f"Update failed: {message.lu_id} not found")
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Update error: {e}")
    
    async def _handle_quarantine(self, message: SyncMessage):
        """处理隔离消息"""
        self._stats["messages_received"] += 1
        
        if message.namespace not in self.namespaces:
            return
        
        try:
            success = self._cache.update(
                lu_id=message.lu_id,
                lifecycle_state="quarantined",
            )
            
            if success:
                self._stats["quarantine_count"] += 1
                logger.info(f"Quarantined {message.lu_id}: {message.reason}")
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Quarantine error: {e}")
    
    async def _handle_delete(self, message: SyncMessage):
        """处理删除消息"""
        self._stats["messages_received"] += 1
        
        if message.namespace not in self.namespaces:
            return
        
        try:
            self._cache.remove(message.lu_id)
            logger.info(f"Deleted {message.lu_id}")
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Delete error: {e}")
    
    async def _handle_batch_inject(self, message: SyncMessage):
        """处理批量注入消息"""
        self._stats["messages_received"] += 1
        
        if not message.batch_items:
            return
        
        for item in message.batch_items:
            ns = item.get("namespace", message.namespace)
            if ns not in self.namespaces:
                continue
            
            try:
                self._cache.add(
                    lu_id=item["lu_id"],
                    key_vector=item["key_vector"],
                    value_vector=item["value_vector"],
                    namespace=ns,
                    condition=item.get("condition"),
                    decision=item.get("decision"),
                    lifecycle_state=item.get("lifecycle_state", "probationary"),
                    trust_tier=item.get("trust_tier"),
                )
                self._stats["inject_count"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Batch inject item error: {e}")
    
    async def _handle_full_sync(self, message: SyncMessage):
        """处理全量同步请求"""
        # Portal 请求全量同步，从 Portal API 获取所有知识
        logger.info("Full sync requested")
        await self._initial_sync()
    
    # ==================== Portal 交互 ====================
    
    async def _register_to_portal(self):
        """注册到 Portal"""
        try:
            import httpx
            
            url = f"{self.config.sync.portal_url}/runtimes/register"
            params = {
                "instance_id": self.instance_id,
                "namespaces": ",".join(self.namespaces),
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, params=params, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Registered to Portal: {self.instance_id}")
                else:
                    logger.warning(f"Portal registration failed: {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Failed to register to Portal: {e}")
    
    async def _deregister_from_portal(self):
        """从 Portal 注销"""
        try:
            import httpx
            
            url = f"{self.config.sync.portal_url}/runtimes/{self.instance_id}"
            
            async with httpx.AsyncClient() as client:
                await client.delete(url, timeout=10)
                logger.info(f"Deregistered from Portal: {self.instance_id}")
                
        except Exception as e:
            logger.warning(f"Failed to deregister from Portal: {e}")
    
    async def _initial_sync(self):
        """从 Portal 初始同步"""
        max_retries = 3
        
        try:
            import httpx
            
            for namespace in self.namespaces:
                url = f"{self.config.sync.portal_url}/knowledge/{namespace}"
                params = {"include_vectors": "true", "limit": 1000}
                
                # 带重试的请求
                for attempt in range(max_retries):
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(url, params=params, timeout=30)
                            if response.status_code == 200:
                                data = response.json()
                                items = data.get("data", {}).get("items", [])
                                
                                loaded_count = 0
                                for item in items:
                                    result = self._cache.add(
                                        lu_id=item["lu_id"],
                                        key_vector=item["key_vector"],
                                        value_vector=item["value_vector"],
                                        namespace=namespace,
                                        condition=item.get("condition"),
                                        decision=item.get("decision"),
                                        lifecycle_state=item.get("lifecycle_state", "probationary"),
                                        trust_tier=item.get("trust_tier"),
                                    )
                                    if result is not None:
                                        loaded_count += 1
                                
                                logger.info(f"Initial sync: loaded {loaded_count}/{len(items)} items for {namespace}")
                                break  # 成功，跳出重试循环
                            else:
                                logger.warning(f"Initial sync failed for {namespace}: {response.status_code}")
                                break  # 非网络错误，不重试
                                
                    except httpx.RequestError as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # 指数退避
                            logger.warning(f"Initial sync attempt {attempt + 1} failed for {namespace}, retrying in {wait_time}s: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Initial sync failed for {namespace} after {max_retries} attempts: {e}")
                        
        except Exception as e:
            logger.warning(f"Initial sync failed: {e}")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(30)
                
                import httpx
                url = f"{self.config.sync.portal_url}/runtimes/heartbeat/{self.instance_id}"
                
                async with httpx.AsyncClient() as client:
                    await client.post(url, timeout=5)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat failed: {e}")
    
    # ==================== 模型集成 ====================
    
    def attach_to_layer(self, layer: "torch.nn.Module") -> "AGARuntimeLayer":
        """
        附加到 Transformer 层
        
        Args:
            layer: 原始 Transformer 层
        
        Returns:
            包装后的 AGA 层
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        return AGARuntimeLayer(layer, self._runtime)
    
    def get_runtime(self) -> AGARuntime:
        """获取 Runtime 模块"""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._runtime
    
    def get_cache(self) -> LocalCache:
        """获取缓存"""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._cache
    
    # ==================== 回调注册 ====================
    
    def on_inject(self, callback: Callable):
        """注册注入回调"""
        self._on_inject_callbacks.append(callback)
        return callback
    
    def on_update(self, callback: Callable):
        """注册更新回调"""
        self._on_update_callbacks.append(callback)
        return callback
    
    # ==================== 统计 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "instance_id": self.instance_id,
            "namespaces": self.namespaces,
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time,
            **self._stats,
            "cache": self._cache.get_stats() if self._cache else None,
            "runtime": self._runtime.get_stats() if self._runtime else None,
        }
