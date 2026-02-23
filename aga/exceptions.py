"""
aga/exceptions.py — AGA 精简异常体系

所有 AGA 异常的基类和具体异常定义。
"""


class AGAError(Exception):
    """AGA 基础异常"""
    pass


class AttachError(AGAError):
    """模型挂载/卸载异常"""
    pass


class KVStoreError(AGAError):
    """KV 存储异常"""
    pass


class ConfigError(AGAError):
    """配置异常"""
    pass


class GateError(AGAError):
    """门控系统异常"""
    pass


class AdapterError(AGAError):
    """LLM 适配器异常"""
    pass


class RetrieverError(AGAError):
    """召回器异常"""
    pass
