"""
Redis Mock 实现

模拟 Redis 客户端行为，用于测试。
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict
import threading


class MockRedisPubSub:
    """Mock Redis Pub/Sub"""
    
    def __init__(self, redis: 'MockRedis'):
        self._redis = redis
        self._subscriptions: Set[str] = set()
        self._pattern_subscriptions: Set[str] = set()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    def subscribe(self, *channels):
        """订阅频道"""
        for channel in channels:
            self._subscriptions.add(channel)
            self._redis._add_subscriber(channel, self)
    
    def psubscribe(self, *patterns):
        """模式订阅"""
        for pattern in patterns:
            self._pattern_subscriptions.add(pattern)
            self._redis._add_pattern_subscriber(pattern, self)
    
    def unsubscribe(self, *channels):
        """取消订阅"""
        for channel in channels:
            self._subscriptions.discard(channel)
            self._redis._remove_subscriber(channel, self)
    
    def punsubscribe(self, *patterns):
        """取消模式订阅"""
        for pattern in patterns:
            self._pattern_subscriptions.discard(pattern)
            self._redis._remove_pattern_subscriber(pattern, self)
    
    async def get_message(self, timeout: float = 1.0) -> Optional[Dict]:
        """获取消息"""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def _receive_message(self, channel: str, data: Any, pattern: Optional[str] = None):
        """接收消息（内部使用）"""
        msg = {
            "type": "pmessage" if pattern else "message",
            "channel": channel,
            "data": data,
        }
        if pattern:
            msg["pattern"] = pattern
        
        try:
            self._message_queue.put_nowait(msg)
        except:
            pass
    
    def close(self):
        """关闭"""
        for channel in list(self._subscriptions):
            self._redis._remove_subscriber(channel, self)
        for pattern in list(self._pattern_subscriptions):
            self._redis._remove_pattern_subscriber(pattern, self)


class MockRedis:
    """
    Mock Redis 客户端
    
    模拟 Redis 的基本操作，用于测试。
    支持：
    - 字符串操作 (get, set, delete)
    - 哈希操作 (hget, hset, hgetall)
    - 列表操作 (lpush, rpush, lpop, rpop, lrange)
    - 集合操作 (sadd, srem, smembers)
    - 发布订阅 (publish, subscribe)
    - 过期时间 (expire, ttl)
    """
    
    def __init__(self, fail_after: int = 0, latency_ms: float = 0):
        """
        初始化 Mock Redis
        
        Args:
            fail_after: 在多少次操作后开始失败（0 表示不失败）
            latency_ms: 模拟延迟（毫秒）
        """
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # 发布订阅
        self._subscribers: Dict[str, List[MockRedisPubSub]] = defaultdict(list)
        self._pattern_subscribers: Dict[str, List[MockRedisPubSub]] = defaultdict(list)
        
        # 故障注入
        self._fail_after = fail_after
        self._operation_count = 0
        self._latency_ms = latency_ms
        self._is_connected = True
    
    def _check_failure(self):
        """检查是否应该失败"""
        self._operation_count += 1
        if self._fail_after > 0 and self._operation_count > self._fail_after:
            raise ConnectionError("Mock Redis connection failed")
        if not self._is_connected:
            raise ConnectionError("Mock Redis not connected")
    
    def _simulate_latency(self):
        """模拟延迟"""
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000)
    
    def _check_expiry(self, key: str) -> bool:
        """检查键是否过期"""
        if key in self._expiry:
            if time.time() > self._expiry[key]:
                del self._data[key]
                del self._expiry[key]
                return True
        return False
    
    # ==================== 连接管理 ====================
    
    def disconnect(self):
        """断开连接（用于故障测试）"""
        self._is_connected = False
    
    def reconnect(self):
        """重新连接"""
        self._is_connected = True
    
    async def ping(self) -> bool:
        """Ping"""
        self._check_failure()
        return True
    
    async def close(self):
        """关闭连接"""
        pass
    
    # ==================== 字符串操作 ====================
    
    async def get(self, key: str) -> Optional[str]:
        """获取值"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            self._check_expiry(key)
            value = self._data.get(key)
            if isinstance(value, bytes):
                return value.decode()
            return value
    
    async def set(self, key: str, value: Any, ex: int = None, px: int = None) -> bool:
        """设置值"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            if isinstance(value, str):
                self._data[key] = value.encode() if isinstance(value, str) else value
            else:
                self._data[key] = value
            
            if ex:
                self._expiry[key] = time.time() + ex
            elif px:
                self._expiry[key] = time.time() + px / 1000
            
            return True
    
    async def delete(self, *keys) -> int:
        """删除键"""
        self._check_failure()
        self._simulate_latency()
        
        count = 0
        with self._lock:
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    self._expiry.pop(key, None)
                    count += 1
        return count
    
    async def exists(self, *keys) -> int:
        """检查键是否存在"""
        self._check_failure()
        self._simulate_latency()
        
        count = 0
        with self._lock:
            for key in keys:
                self._check_expiry(key)
                if key in self._data:
                    count += 1
        return count
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        self._check_failure()
        
        with self._lock:
            if key in self._data:
                self._expiry[key] = time.time() + seconds
                return True
            return False
    
    async def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        self._check_failure()
        
        with self._lock:
            if key not in self._data:
                return -2
            if key not in self._expiry:
                return -1
            remaining = self._expiry[key] - time.time()
            return max(0, int(remaining))
    
    # ==================== 哈希操作 ====================
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """获取哈希字段"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            self._check_expiry(name)
            hash_data = self._data.get(name, {})
            if isinstance(hash_data, dict):
                return hash_data.get(key)
            return None
    
    async def hset(self, name: str, key: str = None, value: Any = None, mapping: Dict = None) -> int:
        """设置哈希字段"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            if name not in self._data:
                self._data[name] = {}
            
            count = 0
            if key is not None and value is not None:
                if key not in self._data[name]:
                    count = 1
                self._data[name][key] = value
            
            if mapping:
                for k, v in mapping.items():
                    if k not in self._data[name]:
                        count += 1
                    self._data[name][k] = v
            
            return count
    
    async def hgetall(self, name: str) -> Dict:
        """获取所有哈希字段"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            self._check_expiry(name)
            return dict(self._data.get(name, {}))
    
    async def hdel(self, name: str, *keys) -> int:
        """删除哈希字段"""
        self._check_failure()
        
        count = 0
        with self._lock:
            if name in self._data and isinstance(self._data[name], dict):
                for key in keys:
                    if key in self._data[name]:
                        del self._data[name][key]
                        count += 1
        return count
    
    # ==================== 列表操作 ====================
    
    async def lpush(self, name: str, *values) -> int:
        """左侧插入"""
        self._check_failure()
        
        with self._lock:
            if name not in self._data:
                self._data[name] = []
            for value in reversed(values):
                self._data[name].insert(0, value)
            return len(self._data[name])
    
    async def rpush(self, name: str, *values) -> int:
        """右侧插入"""
        self._check_failure()
        
        with self._lock:
            if name not in self._data:
                self._data[name] = []
            self._data[name].extend(values)
            return len(self._data[name])
    
    async def lpop(self, name: str) -> Optional[str]:
        """左侧弹出"""
        self._check_failure()
        
        with self._lock:
            if name in self._data and self._data[name]:
                return self._data[name].pop(0)
            return None
    
    async def rpop(self, name: str) -> Optional[str]:
        """右侧弹出"""
        self._check_failure()
        
        with self._lock:
            if name in self._data and self._data[name]:
                return self._data[name].pop()
            return None
    
    async def lrange(self, name: str, start: int, end: int) -> List:
        """获取列表范围"""
        self._check_failure()
        
        with self._lock:
            if name not in self._data:
                return []
            if end == -1:
                return self._data[name][start:]
            return self._data[name][start:end + 1]
    
    async def llen(self, name: str) -> int:
        """获取列表长度"""
        self._check_failure()
        
        with self._lock:
            return len(self._data.get(name, []))
    
    # ==================== 集合操作 ====================
    
    async def sadd(self, name: str, *values) -> int:
        """添加集合成员"""
        self._check_failure()
        
        with self._lock:
            if name not in self._data:
                self._data[name] = set()
            count = 0
            for value in values:
                if value not in self._data[name]:
                    self._data[name].add(value)
                    count += 1
            return count
    
    async def srem(self, name: str, *values) -> int:
        """移除集合成员"""
        self._check_failure()
        
        count = 0
        with self._lock:
            if name in self._data:
                for value in values:
                    if value in self._data[name]:
                        self._data[name].discard(value)
                        count += 1
        return count
    
    async def smembers(self, name: str) -> Set:
        """获取集合成员"""
        self._check_failure()
        
        with self._lock:
            return set(self._data.get(name, set()))
    
    async def sismember(self, name: str, value: Any) -> bool:
        """检查是否是集合成员"""
        self._check_failure()
        
        with self._lock:
            return value in self._data.get(name, set())
    
    # ==================== 发布订阅 ====================
    
    def pubsub(self) -> MockRedisPubSub:
        """创建 Pub/Sub 对象"""
        return MockRedisPubSub(self)
    
    async def publish(self, channel: str, message: Any) -> int:
        """发布消息"""
        self._check_failure()
        
        count = 0
        
        # 直接订阅
        for subscriber in self._subscribers.get(channel, []):
            subscriber._receive_message(channel, message)
            count += 1
        
        # 模式订阅
        for pattern, subscribers in self._pattern_subscribers.items():
            if self._match_pattern(pattern, channel):
                for subscriber in subscribers:
                    subscriber._receive_message(channel, message, pattern)
                    count += 1
        
        return count
    
    def _add_subscriber(self, channel: str, subscriber: MockRedisPubSub):
        """添加订阅者"""
        self._subscribers[channel].append(subscriber)
    
    def _remove_subscriber(self, channel: str, subscriber: MockRedisPubSub):
        """移除订阅者"""
        if channel in self._subscribers:
            try:
                self._subscribers[channel].remove(subscriber)
            except ValueError:
                pass
    
    def _add_pattern_subscriber(self, pattern: str, subscriber: MockRedisPubSub):
        """添加模式订阅者"""
        self._pattern_subscribers[pattern].append(subscriber)
    
    def _remove_pattern_subscriber(self, pattern: str, subscriber: MockRedisPubSub):
        """移除模式订阅者"""
        if pattern in self._pattern_subscribers:
            try:
                self._pattern_subscribers[pattern].remove(subscriber)
            except ValueError:
                pass
    
    def _match_pattern(self, pattern: str, channel: str) -> bool:
        """匹配模式"""
        import fnmatch
        return fnmatch.fnmatch(channel, pattern)
    
    # ==================== 事务 ====================
    
    def pipeline(self) -> 'MockRedisPipeline':
        """创建管道"""
        return MockRedisPipeline(self)
    
    # ==================== 工具方法 ====================
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键"""
        self._check_failure()
        
        import fnmatch
        with self._lock:
            return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]
    
    async def flushdb(self):
        """清空数据库"""
        self._check_failure()
        
        with self._lock:
            self._data.clear()
            self._expiry.clear()
    
    def get_data(self) -> Dict:
        """获取所有数据（测试用）"""
        return dict(self._data)


class MockRedisPipeline:
    """Mock Redis Pipeline"""
    
    def __init__(self, redis: MockRedis):
        self._redis = redis
        self._commands: List[tuple] = []
    
    def get(self, key: str):
        self._commands.append(("get", key))
        return self
    
    def set(self, key: str, value: Any, ex: int = None):
        self._commands.append(("set", key, value, ex))
        return self
    
    def delete(self, *keys):
        self._commands.append(("delete", *keys))
        return self
    
    def hset(self, name: str, key: str, value: Any):
        self._commands.append(("hset", name, key, value))
        return self
    
    def hget(self, name: str, key: str):
        self._commands.append(("hget", name, key))
        return self
    
    async def execute(self) -> List:
        """执行管道"""
        results = []
        for cmd in self._commands:
            method = getattr(self._redis, cmd[0])
            result = await method(*cmd[1:])
            results.append(result)
        self._commands.clear()
        return results
