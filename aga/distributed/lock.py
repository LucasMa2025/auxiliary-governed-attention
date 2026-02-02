"""
AGA 分布式锁

提供分布式环境下的互斥访问控制。

版本: v3.0
"""
import asyncio
import time
import uuid
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DistributedLock:
    """
    分布式锁
    
    基于 Redis 实现的分布式锁，支持：
    - 自动续期
    - 可重入
    - 超时释放
    """
    
    def __init__(
        self,
        name: str,
        client,
        timeout: int = 30,
        retry_interval: float = 0.1,
        auto_extend: bool = True,
    ):
        """
        初始化分布式锁
        
        Args:
            name: 锁名称
            client: Redis 客户端
            timeout: 锁超时时间（秒）
            retry_interval: 重试间隔（秒）
            auto_extend: 是否自动续期
        """
        self.name = name
        self.client = client
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.auto_extend = auto_extend
        
        self._token = str(uuid.uuid4())
        self._acquired = False
        self._extend_task: Optional[asyncio.Task] = None
    
    @property
    def key(self) -> str:
        """锁的 Redis 键"""
        return f"aga:lock:{self.name}"
    
    async def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        获取锁
        
        Args:
            blocking: 是否阻塞等待
            timeout: 等待超时时间
        
        Returns:
            是否成功获取
        """
        start_time = time.time()
        
        while True:
            # 尝试获取锁
            acquired = await self.client.set(
                self.key,
                self._token,
                nx=True,
                ex=self.timeout,
            )
            
            if acquired:
                self._acquired = True
                
                # 启动自动续期
                if self.auto_extend:
                    self._extend_task = asyncio.create_task(self._auto_extend())
                
                logger.debug(f"Lock acquired: {self.name}")
                return True
            
            if not blocking:
                return False
            
            # 检查超时
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            await asyncio.sleep(self.retry_interval)
    
    async def release(self) -> bool:
        """
        释放锁
        
        Returns:
            是否成功释放
        """
        if not self._acquired:
            return False
        
        # 停止自动续期
        if self._extend_task:
            self._extend_task.cancel()
            self._extend_task = None
        
        # 使用 Lua 脚本确保原子性
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = await self.client.eval(script, 1, self.key, self._token)
            self._acquired = False
            logger.debug(f"Lock released: {self.name}")
            return result == 1
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False
    
    async def extend(self, additional_time: int = None) -> bool:
        """
        延长锁的持有时间
        
        Args:
            additional_time: 额外时间（秒），默认使用原始超时时间
        
        Returns:
            是否成功延长
        """
        if not self._acquired:
            return False
        
        ttl = additional_time or self.timeout
        
        # 使用 Lua 脚本确保原子性
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        
        try:
            result = await self.client.eval(script, 1, self.key, self._token, ttl)
            return result == 1
        except Exception as e:
            logger.error(f"Failed to extend lock: {e}")
            return False
    
    async def _auto_extend(self):
        """自动续期任务"""
        extend_interval = self.timeout / 3
        
        while self._acquired:
            await asyncio.sleep(extend_interval)
            
            if self._acquired:
                await self.extend()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.release()


class LockManager:
    """
    锁管理器
    
    管理多个分布式锁。
    """
    
    def __init__(
        self,
        client,
        default_timeout: int = 30,
        default_retry_interval: float = 0.1,
    ):
        """
        初始化锁管理器
        
        Args:
            client: Redis 客户端
            default_timeout: 默认超时时间
            default_retry_interval: 默认重试间隔
        """
        self.client = client
        self.default_timeout = default_timeout
        self.default_retry_interval = default_retry_interval
        
        self._locks: Dict[str, DistributedLock] = {}
    
    def get_lock(
        self,
        name: str,
        timeout: int = None,
        retry_interval: float = None,
        auto_extend: bool = True,
    ) -> DistributedLock:
        """
        获取或创建锁
        
        Args:
            name: 锁名称
            timeout: 超时时间
            retry_interval: 重试间隔
            auto_extend: 是否自动续期
        
        Returns:
            DistributedLock 实例
        """
        if name not in self._locks:
            self._locks[name] = DistributedLock(
                name=name,
                client=self.client,
                timeout=timeout or self.default_timeout,
                retry_interval=retry_interval or self.default_retry_interval,
                auto_extend=auto_extend,
            )
        return self._locks[name]
    
    async def acquire_lock(
        self,
        name: str,
        blocking: bool = True,
        timeout: float = None,
    ) -> bool:
        """
        获取锁
        
        Args:
            name: 锁名称
            blocking: 是否阻塞
            timeout: 等待超时
        
        Returns:
            是否成功
        """
        lock = self.get_lock(name)
        return await lock.acquire(blocking, timeout)
    
    async def release_lock(self, name: str) -> bool:
        """
        释放锁
        
        Args:
            name: 锁名称
        
        Returns:
            是否成功
        """
        if name in self._locks:
            return await self._locks[name].release()
        return False
    
    async def release_all(self):
        """释放所有锁"""
        for lock in self._locks.values():
            await lock.release()
