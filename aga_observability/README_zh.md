# aga-observability

> **aga-core**（辅助注意力治理）的生产级可观测性附加包

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

## 概述

**aga-observability** 是 [aga-core](../aga/) 的配套可观测性包，为基于 AGA 的 LLM 推理系统提供全面的监控、告警、日志、审计和健康检查能力。

### 设计原则

| 原则          | 说明                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| **零侵入**    | 通过 `EventBus` 订阅事件，不修改 `aga-core` 源码                           |
| **自动集成**  | `pip install aga-observability` 后 `AGAPlugin` 自动检测并启用              |
| **Fail-Open** | 可观测性组件故障不影响 LLM 推理                                            |
| **配置驱动**  | 所有行为通过 `AGAConfig.observability_*` 字段或 `ObservabilityConfig` 控制 |

### 架构

```
┌─────────────────────────────────────────────────────────┐
│                      aga-core                           │
│  AGAPlugin → EventBus ─── emit("forward", data)         │
│                       ─── emit("retrieval", data)       │
│                       ─── emit("audit", data)           │
└────────────────────────────┬────────────────────────────┘
                             │ subscribe
┌────────────────────────────▼────────────────────────────┐
│                  aga-observability                      │
│                                                         │
│  ┌───────────────────┐  ┌──────────────────┐            │
│  │ PrometheusExporter│  │   LogExporter    │            │
│  │  Counter/Hist/    │  │  JSON / Text     │            │
│  │  Gauge → :9090    │  │  File + Rotation │            │
│  └───────────────────┘  └──────────────────┘            │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  AlertManager    │  │  AuditStorage    │             │
│  │  SLO/SLI 规则    │  │  File(JSONL)     │             │
│  │  Webhook/回调    │  │  SQLite          │             │
│  └──────────────────┘  └──────────────────┘             │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  HealthChecker   │  │ GrafanaDashboard │             │
│  │  HTTP :8080      │  │  JSON 生成器     │             │
│  │  K8s 探针        │  │  自动面板         │             │
│  └──────────────────┘  └──────────────────┘             │
│                                                         │
│  ObservabilityStack — 组件编排器                         │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
# 核心功能（日志、审计、告警、健康检查）
pip install aga-observability

# 包含 Prometheus 支持
pip install aga-observability[prometheus]

# 完整安装
pip install aga-observability[full]
```

### 自动集成（推荐）

安装后，`AGAPlugin` 会自动检测并启用 `aga-observability`：

```python
from aga import AGAPlugin, AGAConfig

config = AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,   # 默认: True
    prometheus_enabled=True,
    prometheus_port=9090,
)

plugin = AGAPlugin(config)
# aga-observability 已自动激活 — 无需额外代码！
```

### 手动集成

如需精细控制：

```python
from aga import AGAPlugin, AGAConfig
from aga_observability import ObservabilityStack, ObservabilityConfig

# 创建插件
plugin = AGAPlugin(AGAConfig(hidden_dim=4096))

# 创建可观测性配置
obs_config = ObservabilityConfig(
    prometheus_enabled=True,
    prometheus_port=9090,
    log_format="json",
    log_file="aga_events.log",
    audit_storage_backend="sqlite",
    audit_storage_path="aga_audit.db",
    alert_enabled=True,
    health_enabled=True,
    health_port=8080,
)

# 创建并启动
stack = ObservabilityStack(event_bus=plugin.event_bus, config=obs_config)
stack.start()
stack.bind_plugin(plugin)

# ... 推理 ...

# 关闭
stack.shutdown()
```

## 组件说明

### 1. PrometheusExporter — Prometheus 指标导出

将 AGA 指标通过 HTTP 端点导出到 Prometheus。

**指标清单：**

| 类型      | 指标名                                           | 说明                  |
| --------- | ------------------------------------------------ | --------------------- |
| Counter   | `aga_forward_total{layer, applied}`              | forward 调用总次数    |
| Counter   | `aga_retrieval_total{layer}`                     | 召回器调用总次数      |
| Counter   | `aga_retrieval_injected_total{layer}`            | 召回注入知识总数      |
| Counter   | `aga_audit_operations_total{operation, success}` | 审计操作总数          |
| Histogram | `aga_forward_latency_us{layer}`                  | forward 延迟分布 (μs) |
| Histogram | `aga_gate_value{layer}`                          | 门控值分布            |
| Histogram | `aga_entropy_value{layer}`                       | 熵值分布              |
| Histogram | `aga_retrieval_score`                            | 召回分数分布          |
| Gauge     | `aga_knowledge_count`                            | 当前知识数量          |
| Gauge     | `aga_knowledge_utilization`                      | KVStore 利用率        |
| Gauge     | `aga_knowledge_pinned_count`                     | 锁定知识数量          |
| Gauge     | `aga_activation_rate`                            | 激活率（滑动窗口）    |
| Gauge     | `aga_slot_change_rate`                           | Slot 变化率           |
| Info      | `aga_build`                                      | 构建信息              |

### 2. AlertManager — SLO/SLI 告警管理

基于事件流的实时告警系统，支持可配置规则。

**默认 SLO 规则：**

| 规则名                 | 监控指标                | 触发条件 | 告警级别 |
| ---------------------- | ----------------------- | -------- | -------- |
| `slo_latency_p99`      | `latency_p99`           | > 1000μs | WARNING  |
| `slo_latency_critical` | `latency_p99`           | > 5000μs | CRITICAL |
| `high_utilization`     | `knowledge_utilization` | > 95%    | WARNING  |
| `slot_thrashing`       | `slot_change_rate`      | > 0.5    | WARNING  |

**告警通道：** 日志输出、Webhook (HTTP POST)、自定义回调函数

### 3. LogExporter — 结构化日志导出

支持文件轮转的结构化事件日志。

-   **格式：** JSON（适合 ELK/Loki）或纯文本
-   **输出：** stderr + 可选文件（支持 rotation）
-   **事件：** 订阅所有 EventBus 事件（`*`）

### 4. AuditStorage — 审计日志持久化

知识操作的持久化审计追踪。

-   **File 后端：** JSONL 格式，支持轮转和批量写入
-   **SQLite 后端：** 带索引的数据库，支持保留策略
-   **特性：** 批量缓冲、异步刷新、自动清理过期数据

### 5. HealthChecker — 健康检查

全面的健康检查，支持 HTTP 端点。

**检查项：**

-   KVStore（利用率、数量、锁定数）
-   GateSystem（参数完整性、NaN 检测）
-   Retriever（类型、统计）
-   EventBus（缓冲区使用率）
-   模型挂载状态（Hook 数量）

**HTTP 端点：** `GET /health` → 200（healthy/degraded）或 503（unhealthy）

### 6. GrafanaDashboardGenerator — Grafana 面板生成

自动生成 Grafana Dashboard JSON，包含 5 组面板：

-   概览（激活率、知识数量、利用率、锁定数）
-   Forward 性能（延迟 P50/P95/P99、QPS）
-   门控与熵（热力图）
-   召回器（调用频率、注入数量、Slot 变化率）
-   审计（操作统计）

```python
stack.generate_dashboard("aga_dashboard.json")
# 将 JSON 文件导入 Grafana 即可
```

## 配置参考

### 通过 AGAConfig（自动映射）

```python
config = AGAConfig(
    # 可观测性总开关
    observability_enabled=True,

    # Prometheus
    prometheus_enabled=True,
    prometheus_port=9090,

    # 日志
    log_format="json",        # "json" 或 "text"
    log_level="INFO",

    # 审计
    audit_storage_backend="sqlite",  # "memory" / "file" / "sqlite"
    audit_retention_days=90,
)
```

### 通过 ObservabilityConfig（完整控制）

```python
from aga_observability import ObservabilityConfig, AlertRuleConfig

config = ObservabilityConfig(
    enabled=True,

    # Prometheus
    prometheus_enabled=True,
    prometheus_port=9090,
    prometheus_prefix="aga",
    prometheus_labels={"env": "production", "cluster": "gpu-01"},

    # 日志
    log_enabled=True,
    log_format="json",
    log_level="INFO",
    log_file="/var/log/aga/events.log",
    log_max_bytes=100 * 1024 * 1024,  # 100MB
    log_backup_count=5,

    # 审计
    audit_storage_backend="sqlite",
    audit_storage_path="/data/aga_audit.db",
    audit_retention_days=90,
    audit_flush_interval=10,
    audit_batch_size=100,

    # 告警
    alert_enabled=True,
    alert_webhook_url="https://hooks.slack.com/services/...",
    alert_rules=[
        AlertRuleConfig(
            name="custom_high_entropy",
            metric="entropy_mean",
            operator=">",
            threshold=5.0,
            severity="critical",
            message="熵值过高: {value:.2f}",
        ),
    ],

    # 健康检查
    health_enabled=True,
    health_port=8080,
    health_path="/health",
)
```

## 项目结构

```
aga_observability/
├── __init__.py              # 包入口，导出所有公开 API
├── config.py                # ObservabilityConfig + AlertRuleConfig
├── integration.py           # 自动集成入口（单例模式）
├── stack.py                 # ObservabilityStack — 组件编排器
├── prometheus_exporter.py   # Prometheus 指标导出
├── grafana_dashboard.py     # Grafana Dashboard JSON 生成
├── alert_manager.py         # SLO/SLI 告警管理
├── log_exporter.py          # 结构化日志导出（JSON/Text）
├── audit_storage.py         # 审计持久化（File/SQLite）
├── health.py                # 健康检查 + HTTP 端点
├── pyproject.toml           # 包配置
├── README_en.md             # 英文 README
├── README_zh.md             # 中文 README
└── docs/
    ├── user_manual_en.md    # 英文用户手册
    └── user_manual_zh.md    # 中文用户手册
```

## 依赖

| 依赖                                                         | 必需 | 用途                      |
| ------------------------------------------------------------ | ---- | ------------------------- |
| `aga-core >= 4.4.0`                                          | 是   | 核心 AGA 插件（EventBus） |
| `prometheus-client >= 0.17.0`                                | 可选 | Prometheus 指标导出       |
| Python 标准库（`sqlite3`, `json`, `logging`, `http.server`） | 是   | 审计、日志、健康检查      |

## 与 aga-core 的关系

`aga-observability` 是一个**纯附加**包：

-   **未安装时：** `aga-core` 使用内置的 `ForwardMetrics`（内存环形缓冲区）和 `AuditLog`（内存双端队列）正常工作。无外部指标导出、无持久化审计、无告警。
-   **安装后：** 所有 `EventBus` 事件被捕获并导出到 Prometheus、结构化日志、持久化审计存储，同时提供实时告警和健康检查。
-   **集成方式：** 零代码 — `AGAPlugin._setup_observability()` 在 import 时自动检测。

## 路线图

| 版本   | 功能                                                 | 状态      |
| ------ | ---------------------------------------------------- | --------- |
| v1.0.0 | Prometheus + Grafana + 告警 + 日志 + 审计 + 健康检查 | ✅ 已发布 |
| v1.1.0 | OpenTelemetry Traces 导出                            | 计划中    |
| v1.2.0 | 分布式指标聚合（多实例）                             | 计划中    |
| v1.3.0 | 异常检测（自动阈值）                                 | 计划中    |
| v2.0.0 | 实时仪表盘（WebSocket）                              | 计划中    |

## 许可证

MIT License
