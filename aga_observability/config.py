"""
aga_observability/config.py — 可观测性配置

从 AGAConfig 中提取可观测性相关字段，提供独立的配置类。
支持直接创建或从 AGAConfig 自动映射。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class AlertRuleConfig:
    """告警规则配置"""
    name: str = ""
    metric: str = ""  # activation_rate / latency_p99 / gate_mean / entropy_mean / ...
    operator: str = ">"  # > / < / >= / <= / ==
    threshold: float = 0.0
    window_seconds: int = 60
    severity: str = "warning"  # info / warning / critical
    message: str = ""
    cooldown_seconds: int = 300  # 告警冷却期


@dataclass
class ObservabilityConfig:
    """
    可观测性配置

    可以从 AGAConfig 自动映射，也可以独立创建。
    """

    # === 总开关 ===
    enabled: bool = True

    # === Prometheus ===
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_prefix: str = "aga"  # 指标前缀
    prometheus_labels: Dict[str, str] = field(default_factory=dict)  # 全局标签

    # === 日志 ===
    log_enabled: bool = True
    log_format: str = "json"  # json / text
    log_level: str = "INFO"
    log_file: Optional[str] = None  # 日志文件路径（None=仅 stderr）
    log_max_bytes: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5

    # === 审计持久化 ===
    audit_storage_backend: str = "memory"  # memory / file / sqlite
    audit_storage_path: str = "aga_audit.db"  # file/sqlite 路径
    audit_retention_days: int = 90
    audit_flush_interval: int = 10  # 批量写入间隔（秒）
    audit_batch_size: int = 100  # 批量写入大小

    # === 告警 ===
    alert_enabled: bool = True
    alert_rules: List[AlertRuleConfig] = field(default_factory=list)
    alert_webhook_url: Optional[str] = None  # Webhook 通知地址
    alert_log_level: str = "WARNING"  # 告警日志级别

    # === 健康检查 ===
    health_enabled: bool = True
    health_port: int = 8080
    health_path: str = "/health"

    @classmethod
    def from_aga_config(cls, aga_config) -> "ObservabilityConfig":
        """
        从 AGAConfig 自动映射

        Args:
            aga_config: AGAConfig 实例

        Returns:
            ObservabilityConfig 实例
        """
        return cls(
            enabled=getattr(aga_config, "observability_enabled", True),
            prometheus_enabled=getattr(aga_config, "prometheus_enabled", True),
            prometheus_port=getattr(aga_config, "prometheus_port", 9090),
            log_format=getattr(aga_config, "log_format", "json"),
            log_level=getattr(aga_config, "log_level", "INFO"),
            audit_storage_backend=getattr(aga_config, "audit_storage_backend", "memory"),
            audit_retention_days=getattr(aga_config, "audit_retention_days", 90),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservabilityConfig":
        """从字典创建"""
        # 处理嵌套的 alert_rules
        if "alert_rules" in data:
            rules = data["alert_rules"]
            if isinstance(rules, list):
                data["alert_rules"] = [
                    AlertRuleConfig(**r) if isinstance(r, dict) else r
                    for r in rules
                ]

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
