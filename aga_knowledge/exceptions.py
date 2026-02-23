"""
aga-knowledge 异常定义
"""


class KnowledgeError(Exception):
    """知识管理基础异常"""
    pass


class PersistenceError(KnowledgeError):
    """持久化错误"""
    pass


class ConnectionError(PersistenceError):
    """连接错误"""
    pass


class SerializationError(PersistenceError):
    """序列化错误"""
    pass


class SyncError(KnowledgeError):
    """同步错误"""
    pass


class PortalError(KnowledgeError):
    """Portal 服务错误"""
    pass


class EncoderError(KnowledgeError):
    """编码器错误"""
    pass


class ChunkerError(KnowledgeError):
    """分片器错误"""
    pass


class ConfigAdapterError(KnowledgeError):
    """配置适配器错误"""
    pass


class ConfigError(KnowledgeError):
    """配置错误（包括对齐验证失败）"""
    pass


class ImageHandlingError(KnowledgeError):
    """图片处理错误"""
    pass


class RetrievalError(KnowledgeError):
    """检索错误"""
    pass
