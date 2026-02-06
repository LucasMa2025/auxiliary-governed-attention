"""
Kafka Mock 实现

模拟 Kafka 客户端行为，用于测试。
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict
import threading
from dataclasses import dataclass


@dataclass
class MockKafkaMessage:
    """Mock Kafka 消息"""
    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    timestamp: float
    headers: List[tuple] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = []


class MockKafkaProducer:
    """
    Mock Kafka Producer
    
    模拟 aiokafka.AIOKafkaProducer
    """
    
    def __init__(self, kafka: 'MockKafka', fail_after: int = 0):
        self._kafka = kafka
        self._fail_after = fail_after
        self._send_count = 0
        self._started = False
    
    async def start(self):
        """启动生产者"""
        self._started = True
    
    async def stop(self):
        """停止生产者"""
        self._started = False
    
    async def send(
        self,
        topic: str,
        value: bytes,
        key: bytes = None,
        partition: int = None,
        headers: List[tuple] = None,
    ) -> asyncio.Future:
        """发送消息"""
        self._send_count += 1
        if self._fail_after > 0 and self._send_count > self._fail_after:
            raise Exception("Mock Kafka producer failed")
        
        if not self._started:
            raise Exception("Producer not started")
        
        return await self._kafka._produce(topic, value, key, partition, headers)
    
    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        key: bytes = None,
        partition: int = None,
        headers: List[tuple] = None,
    ):
        """发送消息并等待确认"""
        future = await self.send(topic, value, key, partition, headers)
        return future


class MockKafkaConsumer:
    """
    Mock Kafka Consumer
    
    模拟 aiokafka.AIOKafkaConsumer
    """
    
    def __init__(self, kafka: 'MockKafka', *topics, group_id: str = None):
        self._kafka = kafka
        self._topics = set(topics)
        self._group_id = group_id
        self._started = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._offsets: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    async def start(self):
        """启动消费者"""
        self._started = True
        for topic in self._topics:
            self._kafka._add_consumer(topic, self)
    
    async def stop(self):
        """停止消费者"""
        self._started = False
        for topic in self._topics:
            self._kafka._remove_consumer(topic, self)
    
    def subscribe(self, topics: List[str]):
        """订阅主题"""
        for topic in topics:
            self._topics.add(topic)
            if self._started:
                self._kafka._add_consumer(topic, self)
    
    def unsubscribe(self):
        """取消订阅"""
        for topic in self._topics:
            self._kafka._remove_consumer(topic, self)
        self._topics.clear()
    
    async def getone(self, timeout_ms: int = None) -> Optional[MockKafkaMessage]:
        """获取一条消息"""
        if not self._started:
            raise Exception("Consumer not started")
        
        try:
            timeout = timeout_ms / 1000 if timeout_ms else None
            return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def getmany(
        self,
        timeout_ms: int = None,
        max_records: int = None,
    ) -> Dict[str, List[MockKafkaMessage]]:
        """获取多条消息"""
        if not self._started:
            raise Exception("Consumer not started")
        
        result = defaultdict(list)
        count = 0
        max_records = max_records or 100
        
        try:
            timeout = timeout_ms / 1000 if timeout_ms else 0.1
            while count < max_records:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
                result[msg.topic].append(msg)
                count += 1
        except asyncio.TimeoutError:
            pass
        
        return dict(result)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> MockKafkaMessage:
        if not self._started:
            raise StopAsyncIteration
        
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            raise StopAsyncIteration
    
    def _receive_message(self, message: MockKafkaMessage):
        """接收消息（内部使用）"""
        try:
            self._message_queue.put_nowait(message)
        except:
            pass
    
    async def commit(self):
        """提交偏移量"""
        pass
    
    async def seek_to_beginning(self, *partitions):
        """重置到开始"""
        for tp in partitions:
            self._offsets[tp.topic][tp.partition] = 0
    
    async def seek_to_end(self, *partitions):
        """重置到末尾"""
        for tp in partitions:
            self._offsets[tp.topic][tp.partition] = self._kafka._get_end_offset(tp.topic, tp.partition)


class MockKafka:
    """
    Mock Kafka 集群
    
    模拟 Kafka 的基本操作，用于测试。
    """
    
    def __init__(self, fail_after: int = 0, latency_ms: float = 0):
        """
        初始化 Mock Kafka
        
        Args:
            fail_after: 在多少次操作后开始失败
            latency_ms: 模拟延迟
        """
        # 主题数据: {topic: {partition: [message, ...]}}
        self._topics: Dict[str, Dict[int, List[MockKafkaMessage]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # 消费者: {topic: [consumer, ...]}
        self._consumers: Dict[str, List[MockKafkaConsumer]] = defaultdict(list)
        
        # 配置
        self._fail_after = fail_after
        self._operation_count = 0
        self._latency_ms = latency_ms
        self._is_connected = True
        
        # 线程锁
        self._lock = threading.RLock()
    
    def _check_failure(self):
        """检查是否应该失败"""
        self._operation_count += 1
        if self._fail_after > 0 and self._operation_count > self._fail_after:
            raise Exception("Mock Kafka connection failed")
        if not self._is_connected:
            raise Exception("Mock Kafka not connected")
    
    def _simulate_latency(self):
        """模拟延迟"""
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000)
    
    # ==================== 连接管理 ====================
    
    def disconnect(self):
        """断开连接"""
        self._is_connected = False
    
    def reconnect(self):
        """重新连接"""
        self._is_connected = True
    
    # ==================== 生产者/消费者工厂 ====================
    
    def create_producer(self, fail_after: int = 0) -> MockKafkaProducer:
        """创建生产者"""
        return MockKafkaProducer(self, fail_after)
    
    def create_consumer(self, *topics, group_id: str = None) -> MockKafkaConsumer:
        """创建消费者"""
        return MockKafkaConsumer(self, *topics, group_id=group_id)
    
    # ==================== 内部方法 ====================
    
    async def _produce(
        self,
        topic: str,
        value: bytes,
        key: bytes = None,
        partition: int = None,
        headers: List[tuple] = None,
    ) -> MockKafkaMessage:
        """生产消息"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            if partition is None:
                partition = 0
            
            messages = self._topics[topic][partition]
            offset = len(messages)
            
            message = MockKafkaMessage(
                topic=topic,
                partition=partition,
                offset=offset,
                key=key,
                value=value,
                timestamp=time.time(),
                headers=headers or [],
            )
            
            messages.append(message)
            
            # 分发给消费者
            for consumer in self._consumers.get(topic, []):
                consumer._receive_message(message)
            
            return message
    
    def _add_consumer(self, topic: str, consumer: MockKafkaConsumer):
        """添加消费者"""
        with self._lock:
            if consumer not in self._consumers[topic]:
                self._consumers[topic].append(consumer)
    
    def _remove_consumer(self, topic: str, consumer: MockKafkaConsumer):
        """移除消费者"""
        with self._lock:
            try:
                self._consumers[topic].remove(consumer)
            except ValueError:
                pass
    
    def _get_end_offset(self, topic: str, partition: int) -> int:
        """获取末尾偏移量"""
        with self._lock:
            return len(self._topics[topic][partition])
    
    # ==================== 工具方法 ====================
    
    def get_messages(self, topic: str, partition: int = 0) -> List[MockKafkaMessage]:
        """获取主题消息（测试用）"""
        with self._lock:
            return list(self._topics[topic][partition])
    
    def clear_topic(self, topic: str):
        """清空主题（测试用）"""
        with self._lock:
            self._topics[topic] = defaultdict(list)
    
    def clear_all(self):
        """清空所有数据（测试用）"""
        with self._lock:
            self._topics.clear()
