"""
AGA Mock 对象

提供用于测试的 Mock 实现，模拟外部依赖。
"""
from .redis_mock import MockRedis, MockRedisPubSub
from .postgres_mock import MockPostgres
from .kafka_mock import MockKafka, MockKafkaProducer, MockKafkaConsumer
from .http_mock import MockHTTPClient, MockHTTPResponse

__all__ = [
    "MockRedis",
    "MockRedisPubSub",
    "MockPostgres",
    "MockKafka",
    "MockKafkaProducer",
    "MockKafkaConsumer",
    "MockHTTPClient",
    "MockHTTPResponse",
]
