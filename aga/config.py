"""
aga/config.py — AGAConfig 极简配置

源码映射:
  - 基础参数: 来自 core.py AGAConfig (第 101-130 行)
  - 门控参数: 来自 production/gate.py GateConfig
  - 衰减参数: 来自 decay.py PersistenceDecay
  合并 4 个配置类为 1 个，所有阈值外置支持运行时调节
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .exceptions import ConfigError


@dataclass
class AGAConfig:
    """
    AGA 配置 — 所有动态调节项外置

    使用方式:
        # 1. 直接创建
        config = AGAConfig(hidden_dim=4096)

        # 2. 从 YAML 加载
        config = AGAConfig.from_yaml("aga_config.yaml")

        # 3. 从字典创建
        config = AGAConfig.from_dict({"hidden_dim": 4096})
    """

    # === 模型维度（必须匹配目标模型） ===
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    num_heads: int = 32
    value_bottleneck_dim: int = 256  # value projection 瓶颈维度

    # === 容量 ===
    max_slots: int = 256  # 热知识槽位上限

    # === 设备 ===
    device: str = "cuda"

    # === 熵门控（完整三段式，全部外置） ===
    gate0_enabled: bool = True
    gate0_disabled_namespaces: List[str] = field(default_factory=list)

    gate1_enabled: bool = True
    gate1_uncertainty_source: str = "hidden_variance"

    gate2_top_k: int = 8

    tau_low: float = 0.5  # 低熵阈值（模型确信，禁止干预）
    tau_high: float = 2.0  # 高熵阈值（模型极度不确定，限制干预）
    max_gate: float = 0.8  # 最大门控值

    early_exit_enabled: bool = True
    early_exit_threshold: float = 0.05

    # === 衰减 ===
    decay_enabled: bool = True
    decay_strategy: str = "exponential"
    decay_gamma: float = 0.9
    decay_hard_reset_threshold: float = 3.0

    # === 安全 ===
    fail_open: bool = True  # 出错时回退到原始输出
    max_forward_timeout_ms: int = 50

    # === 范数控制 ===
    key_norm_target: float = 5.0
    value_norm_target: float = 3.0
    enable_norm_clipping: bool = True

    # === 埋点与审计（内置，零外部依赖） ===
    instrumentation_enabled: bool = True
    event_buffer_size: int = 10000  # 内存环形缓冲区大小
    audit_log_level: str = "INFO"  # 审计日志级别 (DEBUG/INFO/WARNING)
    audit_log_operations: List[str] = field(default_factory=lambda: [
        "register", "unregister", "attach", "detach", "clear", "load_from"
    ])

    # === 可观测性（aga-observability 配置，安装后生效） ===
    observability_enabled: bool = True  # 是否启用 aga-observability 自动集成
    prometheus_enabled: bool = True  # 启用 Prometheus 指标导出
    prometheus_port: int = 9090  # Prometheus HTTP 端点端口
    audit_storage_backend: str = "memory"  # memory / file / sqlite / postgresql
    audit_retention_days: int = 90  # 审计日志保留天数
    log_format: str = "json"  # json / text
    log_level: str = "INFO"

    # === 召回器配置（配置驱动知识检索） ===
    retriever_backend: str = "null"  # null / kv_store / chroma / milvus / elasticsearch / custom
    retriever_endpoint: str = ""  # 外部召回器连接地址
    retriever_collection: str = ""  # 知识集合/索引名称
    retriever_top_k: int = 5  # 每次召回的最大知识数
    retriever_min_score: float = 0.3  # 最小相关性阈值
    retriever_query_source: str = "q_proj"  # hidden_states / q_proj
    retriever_auto_inject: bool = True  # 召回结果是否自动注入 KVStore
    retriever_cache_ttl: int = 300  # 召回结果缓存 TTL（秒），0=不缓存
    retriever_timeout_ms: int = 10  # 召回超时（毫秒）
    retriever_options: Dict[str, Any] = field(default_factory=dict)  # 后端特定配置

    # === Slot 治理（防止 Slot Thrashing） ===
    pin_registered: bool = True  # register() 的知识自动锁定（不可被 LRU 淘汰）
    retriever_slot_ratio: float = 0.3  # 召回器最多占用 max_slots 的比例
    retriever_slot_budget: int = 0  # 召回器可用的 slot 预算（0=使用 ratio 计算）
    retriever_cooldown_steps: int = 5  # 召回冷却期（forward 步数）
    retriever_dedup_similarity: float = 0.95  # 语义去重阈值（余弦相似度）
    slot_stability_threshold: float = 0.5  # 稳定性阈值（每步最多变化比例）

    # === 知识源配置（配置驱动） ===
    knowledge_sources: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "AGAConfig":
        """从 YAML 文件加载配置"""
        try:
            import yaml
        except ImportError:
            raise ConfigError(
                "加载 YAML 配置需要 PyYAML。请运行: pip install pyyaml"
            )

        filepath = Path(path)
        if not filepath.exists():
            raise ConfigError(f"配置文件不存在: {path}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        aga_data = data.get("aga", data)
        return cls.from_dict(aga_data)

    @classmethod
    def from_dict(cls, data: Dict) -> "AGAConfig":
        """从字典创建配置"""
        data = dict(data)  # 避免修改原始字典

        # 展平嵌套的 gate 配置
        if "gate" in data:
            gate = data.pop("gate")
            if isinstance(gate, dict):
                for k, v in gate.items():
                    data[k] = v

        # 展平嵌套的 decay 配置
        if "decay" in data:
            decay = data.pop("decay")
            if isinstance(decay, dict):
                for k, v in decay.items():
                    data[f"decay_{k}"] = v

        # 展平嵌套的 instrumentation 配置
        if "instrumentation" in data:
            inst = data.pop("instrumentation")
            if isinstance(inst, dict):
                for k, v in inst.items():
                    data[k] = v

        # 展平嵌套的 observability 配置
        if "observability" in data:
            obs = data.pop("observability")
            if isinstance(obs, dict):
                for k, v in obs.items():
                    data[k] = v

        # 展平嵌套的 retriever 配置
        if "retriever" in data:
            ret = data.pop("retriever")
            if isinstance(ret, dict):
                for k, v in ret.items():
                    if k == "options":
                        data["retriever_options"] = v
                    else:
                        data[f"retriever_{k}"] = v

        # 展平嵌套的 slot_governance 配置
        if "slot_governance" in data:
            sg = data.pop("slot_governance")
            if isinstance(sg, dict):
                for k, v in sg.items():
                    # 直接映射到顶层字段
                    data[k] = v

        # 过滤有效字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []

        if self.hidden_dim <= 0:
            errors.append("hidden_dim 必须大于 0")
        if self.bottleneck_dim <= 0:
            errors.append("bottleneck_dim 必须大于 0")
        if self.bottleneck_dim >= self.hidden_dim:
            errors.append("bottleneck_dim 必须小于 hidden_dim")
        if self.max_slots <= 0:
            errors.append("max_slots 必须大于 0")
        if not (0 <= self.tau_low < self.tau_high):
            errors.append("需要 0 <= tau_low < tau_high")
        if not (0 < self.max_gate <= 1.0):
            errors.append("max_gate 必须在 (0, 1.0] 范围内")
        if self.gate2_top_k <= 0:
            errors.append("gate2_top_k 必须大于 0")
        if self.decay_gamma <= 0 or self.decay_gamma > 1.0:
            errors.append("decay_gamma 必须在 (0, 1.0] 范围内")

        return errors
