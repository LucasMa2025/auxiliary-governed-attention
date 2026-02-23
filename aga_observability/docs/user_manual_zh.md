# aga-observability 用户手册

> 版本 1.0.0 | aga-core 生产级可观测性附加包

---

## 目录

1. [简介](#1-简介)
2. [安装与配置](#2-安装与配置)
3. [集成模式](#3-集成模式)
4. [组件详解](#4-组件详解)
    - [4.1 PrometheusExporter — Prometheus 指标导出](#41-prometheusexporter--prometheus-指标导出)
    - [4.2 AlertManager — SLO/SLI 告警管理](#42-alertmanager--slosli-告警管理)
    - [4.3 LogExporter — 结构化日志导出](#43-logexporter--结构化日志导出)
    - [4.4 AuditStorage — 审计日志持久化](#44-auditstorage--审计日志持久化)
    - [4.5 HealthChecker — 健康检查](#45-healthchecker--健康检查)
    - [4.6 GrafanaDashboardGenerator — Grafana 面板生成](#46-grafanadashboardgenerator--grafana-面板生成)
5. [配置参考](#5-配置参考)
6. [部署场景](#6-部署场景)
7. [API 参考](#7-api-参考)
8. [故障排查](#8-故障排查)
9. [常见问题](#9-常见问题)

---

## 1. 简介

### 什么是 aga-observability？

`aga-observability` 是 `aga-core` 的配套可观测性包，将 AGA 内置的轻量级埋点（`EventBus` 事件）转化为生产级可观测性能力：

-   **Prometheus 指标**：用于仪表盘和告警
-   **结构化日志**：用于集中式日志管理（ELK、Loki）
-   **持久化审计追踪**：用于合规和调试
-   **SLO/SLI 告警**：用于运维感知
-   **健康检查端点**：用于 Kubernetes 探针

### 何时需要它？

| 场景          | 内置功能（aga-core）                             | + aga-observability |
| ------------- | ------------------------------------------------ | ------------------- |
| 开发 / 研发   | `ForwardMetrics` 环形缓冲区、`AuditLog` 内存队列 | 非必需              |
| 预发布 / 测试 | 通过 `get_diagnostics()` 获取基本诊断            | 推荐                |
| 生产环境      | 运维能力不足                                     | **必需**            |
| 多实例部署    | 无跨实例可见性                                   | **必需**            |

### 工作原理

```
aga-core EventBus                    aga-observability
┌──────────────────┐                 ┌───────────────────┐
│ emit("forward")  │ ──subscribe──→  │ PrometheusExporter│
│ emit("retrieval")│ ──subscribe──→  │ LogExporter       │
│ emit("audit")    │ ──subscribe──→  │ AuditStorage      │
│                  │                 │ AlertManager      │
└──────────────────┘                 └───────────────────┘
```

`aga-core` 的 `EventBus` 发射三种事件类型：

-   **`forward`**：每次 AGA forward 传播时发射（层级指标）
-   **`retrieval`**：召回器被调用时发射
-   **`audit`**：知识操作时发射（register、unregister、load、clear）

`aga-observability` 订阅这些事件并路由到相应的后端。

---

## 2. 安装与配置

### 前置条件

-   Python >= 3.9
-   `aga-core >= 4.4.0` 已安装
-   （可选）`prometheus-client >= 0.17.0` 用于 Prometheus 支持

### 安装

```bash
# 最小安装（日志、审计、告警、健康检查）
pip install aga-observability

# 包含 Prometheus 支持
pip install aga-observability[prometheus]

# 完整安装（所有可选依赖）
pip install aga-observability[full]

# 开发环境
pip install aga-observability[dev]
```

### 验证安装

```python
import aga_observability
print(aga_observability.__version__)  # 1.0.0
```

---

## 3. 集成模式

### 模式一：自动集成（推荐）

安装包后，`AGAPlugin` 自动检测并启用：

```python
from aga import AGAPlugin, AGAConfig

config = AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,  # 默认值
)

plugin = AGAPlugin(config)
# 控制台输出: "aga-observability 已自动集成"
```

**内部机制：**

```python
# AGAPlugin.__init__() 内部:
def _setup_observability(self):
    if not self.config.observability_enabled:
        return
    try:
        from aga_observability import setup_observability
        setup_observability(self.event_bus, self.config)
    except ImportError:
        pass  # 包未安装，仅使用内置功能
```

### 模式二：手动集成

需要精细控制时使用：

```python
from aga_observability import ObservabilityStack, ObservabilityConfig

obs_config = ObservabilityConfig(
    prometheus_enabled=True,
    prometheus_port=9090,
    log_format="json",
    audit_storage_backend="sqlite",
    audit_storage_path="audit.db",
)

stack = ObservabilityStack(
    event_bus=plugin.event_bus,
    config=obs_config,
)
stack.start()
stack.bind_plugin(plugin)  # 启用 Gauge 更新和健康检查

# ... 推理 ...

stack.shutdown()
```

### 模式三：单独使用组件

只使用需要的组件：

```python
from aga_observability import PrometheusExporter, AlertManager

# 仅 Prometheus
exporter = PrometheusExporter(prefix="aga", port=9090)
exporter.subscribe(plugin.event_bus)
exporter.start_server()

# 仅告警
alert_mgr = AlertManager(use_defaults=True)
alert_mgr.subscribe(plugin.event_bus)
alert_mgr.add_callback(lambda alert: send_to_slack(alert.message))
```

---

## 4. 组件详解

### 4.1 PrometheusExporter — Prometheus 指标导出

#### 概述

将 EventBus 事件转换为 Prometheus 指标，通过 HTTP 端点暴露。

#### 指标清单

**Counter（单调递增）：**

| 指标名                         | 标签                   | 说明               |
| ------------------------------ | ---------------------- | ------------------ |
| `aga_forward_total`            | `layer`, `applied`     | forward 调用总次数 |
| `aga_retrieval_total`          | `layer`                | 召回器调用总次数   |
| `aga_retrieval_injected_total` | `layer`                | 召回注入知识总数   |
| `aga_audit_operations_total`   | `operation`, `success` | 审计操作总数       |

**Histogram（分布）：**

| 指标名                   | 标签    | 桶边界                                         | 说明              |
| ------------------------ | ------- | ---------------------------------------------- | ----------------- |
| `aga_forward_latency_us` | `layer` | 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 | forward 延迟 (μs) |
| `aga_gate_value`         | `layer` | 0.0 到 1.0（步长 0.1）                         | 门控值分布        |
| `aga_entropy_value`      | `layer` | 0.0, 0.5, 1.0, ..., 8.0                        | 熵值分布          |
| `aga_retrieval_score`    | —       | 0.0 到 1.0（步长 0.1）                         | 召回分数分布      |

**Gauge（当前值）：**

| 指标名                       | 说明                          |
| ---------------------------- | ----------------------------- |
| `aga_knowledge_count`        | KVStore 当前知识数量          |
| `aga_knowledge_utilization`  | KVStore 利用率 (0.0 - 1.0)    |
| `aga_knowledge_pinned_count` | 锁定知识数量                  |
| `aga_activation_rate`        | 滑动窗口激活率                |
| `aga_slot_change_rate`       | Slot 变化率（Thrashing 指标） |

**Info：**

| 指标名      | 说明                                                               |
| ----------- | ------------------------------------------------------------------ |
| `aga_build` | 构建信息（version, hidden_dim, bottleneck_dim, max_slots, device） |

#### 使用示例

```python
from aga_observability import PrometheusExporter

exporter = PrometheusExporter(
    prefix="aga",
    port=9090,
    labels={"env": "prod"},
)
exporter.subscribe(plugin.event_bus)
exporter.start_server()

# 手动更新 Gauge（ObservabilityStack 每 5 秒自动调用）
exporter.update_gauges(plugin)

# 访问: http://localhost:9090/metrics
```

#### Prometheus 抓取配置

```yaml
# prometheus.yml
scrape_configs:
    - job_name: "aga"
      static_configs:
          - targets: ["localhost:9090"]
      scrape_interval: 5s
```

### 4.2 AlertManager — SLO/SLI 告警管理

#### 概述

基于指标阈值的实时告警系统，支持冷却期。

#### 默认 SLO 规则

| 规则名                 | 监控指标                | 触发条件 | 告警级别 | 冷却期  |
| ---------------------- | ----------------------- | -------- | -------- | ------- |
| `slo_latency_p99`      | `latency_p99`           | > 1000μs | WARNING  | 5 分钟  |
| `slo_latency_critical` | `latency_p99`           | > 5000μs | CRITICAL | 1 分钟  |
| `high_utilization`     | `knowledge_utilization` | > 0.95   | WARNING  | 10 分钟 |
| `slot_thrashing`       | `slot_change_rate`      | > 0.5    | WARNING  | 2 分钟  |

#### 自定义规则

```python
from aga_observability import AlertManager, AlertRule, AlertSeverity

manager = AlertManager(use_defaults=True)

# 添加自定义规则
manager.add_rule(AlertRule(
    name="high_entropy_alert",
    metric="entropy_mean",
    operator=">",
    threshold=5.0,
    window_seconds=30,
    severity=AlertSeverity.CRITICAL,
    message="熵值严重偏高: {value:.2f} > {threshold:.1f}",
    cooldown_seconds=120,
))

# 移除默认规则
manager.remove_rule("slo_latency_p99")
```

#### 告警通道

```python
# 1. 日志输出（始终启用）
# 告警按相应级别记录（INFO/WARNING/CRITICAL）

# 2. Webhook
manager = AlertManager(webhook_url="https://hooks.slack.com/services/...")

# 3. 自定义回调
def my_handler(alert_event):
    print(f"告警: {alert_event.message}")
    # 发送到 PagerDuty、邮件等

manager.add_callback(my_handler)
```

#### 查询告警历史

```python
# 所有告警
alerts = manager.get_alerts()

# 按级别过滤
critical = manager.get_alerts(severity=AlertSeverity.CRITICAL)

# 按时间过滤
import time
recent = manager.get_alerts(since=time.time() - 3600)  # 最近一小时

# 统计信息
stats = manager.get_stats()
# {'rules_count': 5, 'total_alerts': 12, 'by_severity': {'warning': 8, 'critical': 4}, ...}
```

### 4.3 LogExporter — 结构化日志导出

#### 概述

将所有 EventBus 事件转换为结构化日志输出。

#### 日志格式

**JSON 格式**（推荐用于 ELK/Loki）：

```json
{
    "timestamp": "2026-02-23T10:30:00",
    "level": "DEBUG",
    "logger": "aga.observability",
    "message": "layer=12 applied=✓ gate=0.723 entropy=3.142 latency=45.2μs",
    "event_type": "forward",
    "event_source": "aga_forward",
    "event_timestamp": 1740300600.123,
    "data": {
        "layer_idx": 12,
        "aga_applied": true,
        "gate_mean": 0.723,
        "entropy_mean": 3.142,
        "latency_us": 45.2
    }
}
```

**Text 格式**（人类可读）：

```
2026-02-23T10:30:00 [DEBUG] forward: layer=12 applied=✓ gate=0.723 entropy=3.142 latency=45.2μs
```

#### 使用示例

```python
from aga_observability import LogExporter

exporter = LogExporter(
    format="json",
    level="INFO",           # DEBUG 级别可捕获 forward 事件
    file="aga_events.log",  # 可选文件输出
    max_bytes=100_000_000,  # 每个文件 100MB
    backup_count=5,         # 保留 5 个轮转文件
)
exporter.subscribe(plugin.event_bus)

# ... 推理 ...

exporter.shutdown()  # 刷新并关闭
```

#### 事件级别映射

| 事件类型        | 日志级别 | 原因                              |
| --------------- | -------- | --------------------------------- |
| `forward`       | DEBUG    | 高频事件，仅在 level=DEBUG 时记录 |
| `retrieval`     | INFO     | 重要运维事件                      |
| `audit`（成功） | INFO     | 正常操作                          |
| `audit`（失败） | WARNING  | 需要关注                          |

### 4.4 AuditStorage — 审计日志持久化

#### 概述

将审计事件持久化到外部存储，用于合规和调试。

#### File 后端（JSONL）

```python
from aga_observability import FileAuditStorage

storage = FileAuditStorage(
    path="audit/aga_audit.jsonl",
    flush_interval=10,    # 每 10 秒刷新
    batch_size=100,       # 或缓冲区达到 100 条时刷新
    max_file_size=100_000_000,  # 每个文件 100MB
    max_files=10,         # 保留 10 个轮转文件
)
storage.subscribe(plugin.event_bus)
```

**文件格式**（每行一个 JSON）：

```jsonl
{"timestamp": 1740300600.0, "operation": "register", "details": {"id": "fact_001", "reliability": 1.0}, "success": true, "error": null}
{"timestamp": 1740300601.0, "operation": "register", "details": {"id": "fact_002", "reliability": 0.9}, "success": true, "error": null}
```

#### SQLite 后端

```python
from aga_observability import SQLiteAuditStorage

storage = SQLiteAuditStorage(
    path="audit/aga_audit.db",
    flush_interval=10,
    batch_size=100,
)
storage.subscribe(plugin.event_bus)
```

**数据库表结构：**

```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    operation TEXT NOT NULL,
    details TEXT,           -- JSON 字符串
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- 在 timestamp 和 operation 上建有索引
```

#### 查询审计记录

```python
# 按操作类型查询
registers = storage.query(operation="register", limit=50)

# 按时间范围查询
import time
recent = storage.query(since=time.time() - 86400)  # 最近 24 小时

# 清理过期记录
deleted = storage.cleanup(retention_days=90)
```

### 4.5 HealthChecker — 健康检查

#### 概述

提供组件级健康状态检查，支持可选的 HTTP 端点用于 Kubernetes 探针。

#### 健康状态级别

| 状态        | HTTP 状态码 | 含义                         |
| ----------- | ----------- | ---------------------------- |
| `healthy`   | 200         | 所有组件正常                 |
| `degraded`  | 200         | 部分组件受限（如利用率过高） |
| `unhealthy` | 503         | 严重故障                     |

#### 组件检查项

| 组件          | Healthy      | Degraded      | Unhealthy    |
| ------------- | ------------ | ------------- | ------------ |
| `kv_store`    | 利用率 < 95% | 利用率 >= 95% | 异常         |
| `gate_system` | 参数正常     | —             | 参数包含 NaN |
| `retriever`   | 统计可访问   | 异常          | —            |
| `event_bus`   | 缓冲区 < 90% | 缓冲区 >= 90% | 异常         |
| `attachment`  | 模型已挂载   | 未挂载        | 异常         |

#### 使用示例

```python
from aga_observability import HealthChecker

checker = HealthChecker()
checker.bind_plugin(plugin)

# 编程式检查
result = checker.check()
print(result)
# {
#   "status": "healthy",
#   "components": {
#     "kv_store": {"status": "healthy", "count": 42, "utilization": 0.42, ...},
#     "gate_system": {"status": "healthy", "param_count": 12345},
#     "retriever": {"status": "healthy", "type": "NullRetriever"},
#     "event_bus": {"status": "healthy", "buffer_usage": 0.02, ...},
#     "attachment": {"status": "healthy", "attached": true, "hooks_count": 4},
#   },
#   "timestamp": 1740300600.0,
# }

# 启动 HTTP 端点
checker.start_server(port=8080, path="/health")
# 访问: GET http://localhost:8080/health
```

#### 自定义健康检查

```python
def check_gpu_memory():
    import torch
    free, total = torch.cuda.mem_get_info()
    usage = 1.0 - free / total
    if usage > 0.95:
        return {"status": "unhealthy", "gpu_memory_usage": usage}
    elif usage > 0.85:
        return {"status": "degraded", "gpu_memory_usage": usage}
    return {"status": "healthy", "gpu_memory_usage": usage}

checker.add_check("gpu_memory", check_gpu_memory)
```

#### Kubernetes 探针配置

```yaml
# deployment.yaml
livenessProbe:
    httpGet:
        path: /health
        port: 8080
    initialDelaySeconds: 30
    periodSeconds: 10

readinessProbe:
    httpGet:
        path: /health
        port: 8080
    initialDelaySeconds: 10
    periodSeconds: 5
```

### 4.6 GrafanaDashboardGenerator — Grafana 面板生成

#### 概述

自动生成完整的 Grafana Dashboard JSON，可直接导入。

#### 使用示例

```python
from aga_observability import GrafanaDashboardGenerator

gen = GrafanaDashboardGenerator(
    prefix="aga",
    datasource="Prometheus",
    title="AGA Observability Dashboard",
    refresh="5s",
)

# 保存到文件
gen.save("grafana_dashboard.json")

# 或获取字符串
json_str = gen.generate()

# 或获取字典
dashboard_dict = gen.to_dict()
```

#### Dashboard 面板

| 行           | 面板                                         | 说明           |
| ------------ | -------------------------------------------- | -------------- |
| 概览         | 激活率、知识数量、利用率、锁定数             | 关键指标一览   |
| Forward 性能 | 延迟 P50/P95/P99、QPS（applied vs bypassed） | 性能监控       |
| 门控与熵     | 门控值热力图、熵值热力图                     | 分布分析       |
| 召回器       | 调用频率、注入数量、Slot 变化率              | 召回器监控     |
| 审计         | 操作统计（register/unregister/load/clear）   | 审计追踪可视化 |

#### 导入 Grafana

1. 生成 JSON 文件
2. 打开 Grafana → Dashboards → Import
3. 上传 JSON 文件或粘贴内容
4. 选择 Prometheus 数据源
5. 点击 Import

---

## 5. 配置参考

### ObservabilityConfig 字段

| 字段                    | 类型 | 默认值           | 说明                                     |
| ----------------------- | ---- | ---------------- | ---------------------------------------- |
| `enabled`               | bool | `True`           | 总开关                                   |
| `prometheus_enabled`    | bool | `True`           | 启用 Prometheus 导出                     |
| `prometheus_port`       | int  | `9090`           | Prometheus HTTP 端口                     |
| `prometheus_prefix`     | str  | `"aga"`          | 指标名前缀                               |
| `prometheus_labels`     | Dict | `{}`             | 所有指标的全局标签                       |
| `log_enabled`           | bool | `True`           | 启用日志导出                             |
| `log_format`            | str  | `"json"`         | 日志格式：`"json"` 或 `"text"`           |
| `log_level`             | str  | `"INFO"`         | 日志级别                                 |
| `log_file`              | str  | `None`           | 日志文件路径（None = 仅 stderr）         |
| `log_max_bytes`         | int  | `104857600`      | 日志文件最大大小（100MB）                |
| `log_backup_count`      | int  | `5`              | 轮转文件数量                             |
| `audit_storage_backend` | str  | `"memory"`       | 后端：`"memory"` / `"file"` / `"sqlite"` |
| `audit_storage_path`    | str  | `"aga_audit.db"` | 审计存储路径                             |
| `audit_retention_days`  | int  | `90`             | 数据保留天数                             |
| `audit_flush_interval`  | int  | `10`             | 刷新间隔（秒）                           |
| `audit_batch_size`      | int  | `100`            | 批量写入大小                             |
| `alert_enabled`         | bool | `True`           | 启用告警                                 |
| `alert_rules`           | List | `[]`             | 自定义告警规则                           |
| `alert_webhook_url`     | str  | `None`           | Webhook 通知地址                         |
| `alert_log_level`       | str  | `"WARNING"`      | 告警日志级别                             |
| `health_enabled`        | bool | `True`           | 启用健康检查                             |
| `health_port`           | int  | `8080`           | 健康检查 HTTP 端口                       |
| `health_path`           | str  | `"/health"`      | 健康检查路径                             |

### AlertRuleConfig 字段

| 字段               | 类型  | 默认值      | 说明                                                  |
| ------------------ | ----- | ----------- | ----------------------------------------------------- |
| `name`             | str   | `""`        | 规则名称（唯一标识）                                  |
| `metric`           | str   | `""`        | 监控指标名                                            |
| `operator`         | str   | `">"`       | 比较运算符：`>`, `<`, `>=`, `<=`, `==`                |
| `threshold`        | float | `0.0`       | 阈值                                                  |
| `window_seconds`   | int   | `60`        | 评估窗口（秒）                                        |
| `severity`         | str   | `"warning"` | 级别：`"info"`, `"warning"`, `"critical"`             |
| `message`          | str   | `""`        | 消息模板（支持 `{value}`, `{threshold}`, `{metric}`） |
| `cooldown_seconds` | int   | `300`       | 冷却期（秒）                                          |

### 可用告警指标

| 指标名                   | 来源           | 聚合方式   |
| ------------------------ | -------------- | ---------- |
| `latency_p99`            | forward 事件   | 窗口内 P99 |
| `gate_mean`              | forward 事件   | 最新值     |
| `entropy_mean`           | forward 事件   | 最新值     |
| `activation_rate`        | forward 事件   | 窗口内均值 |
| `knowledge_utilization`  | plugin 指标    | 最新值     |
| `slot_change_rate`       | plugin 诊断    | 最新值     |
| `retrieval_failure_rate` | retrieval 事件 | 最新值     |

---

## 6. 部署场景

### 场景一：单 GPU，开发环境

```python
config = AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,
    log_format="text",  # 人类可读
)
plugin = AGAPlugin(config)
# 日志输出到 stderr，无 Prometheus，无持久化审计
```

### 场景二：单 GPU，生产环境

```python
from aga_observability import ObservabilityConfig

obs_config = ObservabilityConfig(
    prometheus_enabled=True,
    prometheus_port=9090,
    log_format="json",
    log_file="/var/log/aga/events.log",
    audit_storage_backend="sqlite",
    audit_storage_path="/data/aga_audit.db",
    alert_enabled=True,
    alert_webhook_url="https://hooks.slack.com/services/...",
    health_enabled=True,
    health_port=8080,
)
```

### 场景三：多实例（数据并行）

每个实例运行独立的 `ObservabilityStack`，使用 Prometheus 标签区分：

```python
import os

obs_config = ObservabilityConfig(
    prometheus_enabled=True,
    prometheus_port=9090 + int(os.environ.get("RANK", 0)),
    prometheus_labels={
        "instance": os.environ.get("HOSTNAME", "unknown"),
        "rank": os.environ.get("RANK", "0"),
    },
)
```

### 场景四：Kubernetes 部署

```yaml
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
    name: aga-observability-config
data:
    config.yaml: |
        observability:
          prometheus_enabled: true
          prometheus_port: 9090
          log_format: json
          audit_storage_backend: sqlite
          audit_storage_path: /data/audit.db
          health_enabled: true
          health_port: 8080
```

---

## 7. API 参考

### setup_observability()

```python
def setup_observability(event_bus, aga_config=None) -> Optional[ObservabilityStack]
```

自动集成入口。由 `AGAPlugin._setup_observability()` 调用。

**参数：**

-   `event_bus`：aga-core 的 EventBus 实例
-   `aga_config`：AGAConfig 实例（可选）

**返回：** `ObservabilityStack` 实例（成功时）或 `None`

### shutdown_observability()

```python
def shutdown_observability() -> None
```

关闭全局单例 Stack。

### ObservabilityStack

```python
class ObservabilityStack:
    def __init__(self, event_bus, config=None)
    def start(self) -> None                          # 启动所有组件
    def bind_plugin(self, plugin) -> None             # 绑定 AGAPlugin
    def shutdown(self) -> None                        # 关闭所有组件
    def get_stats(self) -> Dict[str, Any]             # 获取统计
    def generate_dashboard(self, path=None) -> str    # 生成 Grafana Dashboard
```

**属性：**

-   `prometheus: Optional[PrometheusExporter]`
-   `log_exporter: Optional[LogExporter]`
-   `audit_storage: Optional[AuditStorageBackend]`
-   `alert_manager: Optional[AlertManager]`
-   `health: Optional[HealthChecker]`
-   `dashboard_generator: Optional[GrafanaDashboardGenerator]`

### PrometheusExporter

```python
class PrometheusExporter:
    def __init__(self, prefix="aga", port=9090, labels=None, registry=None)
    def subscribe(self, event_bus) -> None             # 订阅事件
    def unsubscribe(self, event_bus) -> None            # 取消订阅
    def start_server(self) -> None                      # 启动 HTTP 端点
    def update_gauges(self, plugin) -> None             # 更新 Gauge 指标
    def set_build_info(self, info: Dict[str, str]) -> None  # 设置构建信息
    def shutdown(self) -> None                          # 关闭
    def get_stats(self) -> Dict[str, Any]               # 获取统计
```

### AlertManager

```python
class AlertManager:
    def __init__(self, rules=None, use_defaults=True, max_history=1000, webhook_url=None)
    def add_rule(self, rule: AlertRule) -> None                      # 添加规则
    def remove_rule(self, name: str) -> bool                         # 移除规则
    def add_callback(self, callback: Callable) -> None               # 添加回调
    def subscribe(self, event_bus) -> None                           # 订阅事件
    def unsubscribe(self, event_bus) -> None                         # 取消订阅
    def update_plugin_metrics(self, plugin) -> None                  # 更新 plugin 指标
    def get_alerts(self, severity=None, limit=100, since=None) -> List[Dict]  # 查询历史
    def get_stats(self) -> Dict[str, Any]                            # 获取统计
    def shutdown(self) -> None                                       # 关闭
```

### LogExporter

```python
class LogExporter:
    def __init__(self, format="json", level="INFO", file=None, max_bytes=..., backup_count=5)
    def subscribe(self, event_bus) -> None              # 订阅事件
    def unsubscribe(self, event_bus) -> None             # 取消订阅
    def get_stats(self) -> Dict[str, Any]                # 获取统计
    def shutdown(self) -> None                           # 关闭
```

### FileAuditStorage / SQLiteAuditStorage

```python
class FileAuditStorage(AuditStorageBackend):
    def __init__(self, path="aga_audit.jsonl", flush_interval=10, batch_size=100, ...)
    def subscribe(self, event_bus) -> None              # 订阅事件
    def unsubscribe(self, event_bus) -> None             # 取消订阅
    def store(self, entry: Dict) -> None                 # 存储单条
    def store_batch(self, entries: List[Dict]) -> None   # 批量存储
    def query(self, operation=None, since=None, limit=100) -> List[Dict]  # 查询
    def cleanup(self, retention_days: int) -> int        # 清理过期数据
    def shutdown(self) -> None                           # 关闭
    def get_stats(self) -> Dict[str, Any]                # 获取统计

class SQLiteAuditStorage(AuditStorageBackend):
    # 接口与 FileAuditStorage 相同
```

### HealthChecker

```python
class HealthChecker:
    def __init__(self)
    def bind_plugin(self, plugin) -> None                # 绑定 AGAPlugin
    def add_check(self, name: str, check_fn: callable) -> None  # 添加自定义检查
    def check(self) -> Dict[str, Any]                    # 执行健康检查
    def start_server(self, port=8080, path="/health") -> None  # 启动 HTTP 端点
    def shutdown(self) -> None                           # 关闭
```

### GrafanaDashboardGenerator

```python
class GrafanaDashboardGenerator:
    def __init__(self, prefix="aga", datasource="Prometheus", title="...", refresh="5s")
    def generate(self) -> str                            # 生成 JSON 字符串
    def to_dict(self) -> Dict                            # 生成 Python 字典
    def save(self, path: str) -> None                    # 保存到文件
```

---

## 8. 故障排查

### Prometheus 指标未显示

1. 确认 `prometheus-client` 已安装：`pip install prometheus-client`
2. 检查端口是否被占用：`netstat -tlnp | grep 9090`
3. 验证 `prometheus.yml` 中的抓取配置
4. 检查日志中是否有 `PrometheusExporter 创建失败` 消息

### 健康检查返回 503

1. 编程式调用 `checker.check()` 查看哪个组件不健康
2. 常见原因：
    - GateSystem 参数包含 NaN（训练不稳定）
    - KVStore 利用率 > 95%（增加 `max_slots` 或启用 Slot 治理）
    - 模型未挂载（先调用 `plugin.attach(model)`）

### 审计文件过大

1. 启用轮转：`max_file_size=100_000_000, max_files=10`
2. 定期清理：`storage.cleanup(retention_days=30)`
3. 考虑切换到 SQLite 后端以获得更好的查询性能

### 告警触发过于频繁

1. 增加规则的 `cooldown_seconds`
2. 调整 `threshold` 值
3. 增加 `window_seconds` 以获得更稳定的聚合

### 日志中出现 "aga-observability 自动集成失败"

1. 这是 Fail-Open 警告 — 推理不受影响
2. 查看完整错误消息了解详情
3. 常见原因：端口冲突、缺少依赖、权限问题

---

## 9. 常见问题

**Q: aga-observability 是否影响推理性能？**

A: 影响极小。事件处理是轻量级的（字典创建 + deque 追加）。Prometheus 指标更新使用原子操作。文件/SQLite 写入是批量异步的。热路径（forward 事件处理）增加的开销 < 1μs。

**Q: 可以不使用 Prometheus 吗？**

A: 可以。`prometheus-client` 是可选依赖。没有它时，`PrometheusExporter` 会被跳过，但日志、审计、告警和健康检查仍然正常工作。

**Q: 如何监控多个 AGA 实例？**

A: 每个实例运行独立的 `ObservabilityStack`，使用唯一的 Prometheus 标签（如 `instance`、`rank`）。使用 Prometheus 联邦或共享 Prometheus 服务器进行聚合。

**Q: 可以添加自定义指标吗？**

A: 目前自定义指标需要扩展 `PrometheusExporter`。自定义指标插件系统计划在 v1.2.0 中实现。

**Q: 审计数据库损坏怎么办？**

A: `SQLiteAuditStorage` 使用标准 SQLite。如果发生损坏，存储会记录警告并继续工作（Fail-Open）。可以删除数据库文件，它会被自动重建。

**Q: 可以使用 Prometheus 以外的时序数据库吗？**

A: 当前版本导出 Prometheus 格式。OpenTelemetry 导出（支持 Datadog、New Relic 等）计划在 v1.1.0 中实现。

---

_aga-observability v1.0.0 — AGA 驱动的 LLM 推理系统生产级可观测性_
