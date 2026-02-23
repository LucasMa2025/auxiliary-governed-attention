# aga-observability User Manual

> Version 1.0.0 | Production Observability Stack for aga-core

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation & Setup](#2-installation--setup)
3. [Integration Modes](#3-integration-modes)
4. [Component Guide](#4-component-guide)
    - [4.1 PrometheusExporter](#41-prometheusexporter)
    - [4.2 AlertManager](#42-alertmanager)
    - [4.3 LogExporter](#43-logexporter)
    - [4.4 AuditStorage](#44-auditstorage)
    - [4.5 HealthChecker](#45-healthchecker)
    - [4.6 GrafanaDashboardGenerator](#46-grafanadashboardgenerator)
5. [Configuration Reference](#5-configuration-reference)
6. [Deployment Scenarios](#6-deployment-scenarios)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [FAQ](#9-faq)

---

## 1. Introduction

### What is aga-observability?

`aga-observability` is a companion package for `aga-core` that transforms AGA's built-in instrumentation (lightweight `EventBus` events) into production-grade observability:

-   **Prometheus metrics** for dashboards and alerting
-   **Structured logs** for centralized log management (ELK, Loki)
-   **Persistent audit trails** for compliance and debugging
-   **SLO/SLI alerting** for operational awareness
-   **Health check endpoints** for Kubernetes probes

### When Do You Need It?

| Scenario                  | Built-in (aga-core)                                | + aga-observability |
| ------------------------- | -------------------------------------------------- | ------------------- |
| Development / R&D         | `ForwardMetrics` ring buffer, `AuditLog` in-memory | Not required        |
| Staging / Testing         | Basic diagnostics via `get_diagnostics()`          | Recommended         |
| Production                | Insufficient for ops                               | **Required**        |
| Multi-instance deployment | No cross-instance visibility                       | **Required**        |

### How It Works

```
aga-core EventBus                    aga-observability
┌──────────────────┐                 ┌───────────────────┐
│ emit("forward")  │ ──subscribe──→  │ PrometheusExporter│
│ emit("retrieval")│ ──subscribe──→  │ LogExporter       │
│ emit("audit")    │ ──subscribe──→  │ AuditStorage      │
│                  │                 │ AlertManager      │
└──────────────────┘                 └───────────────────┘
```

The `EventBus` in `aga-core` emits three event types:

-   **`forward`**: Emitted on every AGA forward pass (layer-level metrics)
-   **`retrieval`**: Emitted when the retriever is called
-   **`audit`**: Emitted on knowledge operations (register, unregister, load, clear)

`aga-observability` subscribes to these events and routes them to the appropriate backends.

---

## 2. Installation & Setup

### Prerequisites

-   Python >= 3.9
-   `aga-core >= 4.4.0` installed
-   (Optional) `prometheus-client >= 0.17.0` for Prometheus support

### Installation

```bash
# Minimal (logging, audit, alerting, health check)
pip install aga-observability

# With Prometheus
pip install aga-observability[prometheus]

# Full (all optional dependencies)
pip install aga-observability[full]

# Development
pip install aga-observability[dev]
```

### Verify Installation

```python
import aga_observability
print(aga_observability.__version__)  # 1.0.0
```

---

## 3. Integration Modes

### Mode 1: Automatic (Recommended)

Simply install the package. `AGAPlugin` detects it automatically:

```python
from aga import AGAPlugin, AGAConfig

config = AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,  # default
)

plugin = AGAPlugin(config)
# Console output: "aga-observability 已自动集成"
```

**What happens internally:**

```python
# Inside AGAPlugin.__init__():
def _setup_observability(self):
    if not self.config.observability_enabled:
        return
    try:
        from aga_observability import setup_observability
        setup_observability(self.event_bus, self.config)
    except ImportError:
        pass  # Package not installed, use built-in only
```

### Mode 2: Manual

For full control over component lifecycle:

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
stack.bind_plugin(plugin)  # Enables gauge updates and health checks

# ... use plugin for inference ...

stack.shutdown()
```

### Mode 3: Individual Components

Use only the components you need:

```python
from aga_observability import PrometheusExporter, AlertManager

# Prometheus only
exporter = PrometheusExporter(prefix="aga", port=9090)
exporter.subscribe(plugin.event_bus)
exporter.start_server()

# Alerting only
alert_mgr = AlertManager(use_defaults=True)
alert_mgr.subscribe(plugin.event_bus)
alert_mgr.add_callback(lambda alert: send_to_slack(alert.message))
```

---

## 4. Component Guide

### 4.1 PrometheusExporter

#### Overview

Converts EventBus events into Prometheus metrics, exposed via an HTTP endpoint.

#### Metrics

**Counters** (monotonically increasing):

| Metric                         | Labels                 | Description                    |
| ------------------------------ | ---------------------- | ------------------------------ |
| `aga_forward_total`            | `layer`, `applied`     | Total forward pass count       |
| `aga_retrieval_total`          | `layer`                | Total retrieval calls          |
| `aga_retrieval_injected_total` | `layer`                | Total knowledge items injected |
| `aga_audit_operations_total`   | `operation`, `success` | Total audit operations         |

**Histograms** (distribution):

| Metric                   | Labels  | Buckets                                        | Description                  |
| ------------------------ | ------- | ---------------------------------------------- | ---------------------------- |
| `aga_forward_latency_us` | `layer` | 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 | Forward latency (μs)         |
| `aga_gate_value`         | `layer` | 0.0 to 1.0 (step 0.1)                          | Gate value distribution      |
| `aga_entropy_value`      | `layer` | 0.0, 0.5, 1.0, ..., 8.0                        | Entropy distribution         |
| `aga_retrieval_score`    | —       | 0.0 to 1.0 (step 0.1)                          | Retrieval score distribution |

**Gauges** (current value):

| Metric                       | Description                            |
| ---------------------------- | -------------------------------------- |
| `aga_knowledge_count`        | Current knowledge count in KVStore     |
| `aga_knowledge_utilization`  | KVStore utilization (0.0 - 1.0)        |
| `aga_knowledge_pinned_count` | Number of pinned knowledge items       |
| `aga_activation_rate`        | Sliding window activation rate         |
| `aga_slot_change_rate`       | Slot change rate (thrashing indicator) |

**Info:**

| Metric      | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `aga_build` | Build info (version, hidden_dim, bottleneck_dim, max_slots, device) |

#### Usage

```python
from aga_observability import PrometheusExporter

exporter = PrometheusExporter(
    prefix="aga",
    port=9090,
    labels={"env": "prod"},
)
exporter.subscribe(plugin.event_bus)
exporter.start_server()

# Manually update gauges (called by ObservabilityStack every 5s)
exporter.update_gauges(plugin)

# Access: http://localhost:9090/metrics
```

#### Prometheus Scrape Config

```yaml
# prometheus.yml
scrape_configs:
    - job_name: "aga"
      static_configs:
          - targets: ["localhost:9090"]
      scrape_interval: 5s
```

### 4.2 AlertManager

#### Overview

Real-time alerting based on metric thresholds with cooldown periods.

#### Default SLO Rules

| Rule                   | Metric                  | Condition | Severity | Cooldown |
| ---------------------- | ----------------------- | --------- | -------- | -------- |
| `slo_latency_p99`      | `latency_p99`           | > 1000μs  | WARNING  | 5min     |
| `slo_latency_critical` | `latency_p99`           | > 5000μs  | CRITICAL | 1min     |
| `high_utilization`     | `knowledge_utilization` | > 0.95    | WARNING  | 10min    |
| `slot_thrashing`       | `slot_change_rate`      | > 0.5     | WARNING  | 2min     |

#### Custom Rules

```python
from aga_observability import AlertManager, AlertRule, AlertSeverity

manager = AlertManager(use_defaults=True)

# Add custom rule
manager.add_rule(AlertRule(
    name="high_entropy_alert",
    metric="entropy_mean",
    operator=">",
    threshold=5.0,
    window_seconds=30,
    severity=AlertSeverity.CRITICAL,
    message="Entropy critically high: {value:.2f} > {threshold:.1f}",
    cooldown_seconds=120,
))

# Remove a default rule
manager.remove_rule("slo_latency_p99")
```

#### Alert Channels

```python
# 1. Logging (always enabled)
# Alerts are logged at the appropriate level (INFO/WARNING/CRITICAL)

# 2. Webhook
manager = AlertManager(webhook_url="https://hooks.slack.com/services/...")

# 3. Custom callback
def my_handler(alert_event):
    print(f"ALERT: {alert_event.message}")
    # Send to PagerDuty, email, etc.

manager.add_callback(my_handler)
```

#### Query Alert History

```python
# All alerts
alerts = manager.get_alerts()

# Filter by severity
critical = manager.get_alerts(severity=AlertSeverity.CRITICAL)

# Filter by time
import time
recent = manager.get_alerts(since=time.time() - 3600)  # Last hour

# Statistics
stats = manager.get_stats()
# {'rules_count': 5, 'total_alerts': 12, 'by_severity': {'warning': 8, 'critical': 4}, ...}
```

### 4.3 LogExporter

#### Overview

Converts all EventBus events into structured log output.

#### Formats

**JSON format** (recommended for ELK/Loki):

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

**Text format** (human-readable):

```
2026-02-23T10:30:00 [DEBUG] forward: layer=12 applied=✓ gate=0.723 entropy=3.142 latency=45.2μs
```

#### Usage

```python
from aga_observability import LogExporter

exporter = LogExporter(
    format="json",
    level="INFO",           # DEBUG captures forward events
    file="aga_events.log",  # Optional file output
    max_bytes=100_000_000,  # 100MB per file
    backup_count=5,         # Keep 5 rotated files
)
exporter.subscribe(plugin.event_bus)

# ... inference ...

exporter.shutdown()  # Flush and close
```

#### Event Level Mapping

| Event Type        | Log Level | Rationale                                 |
| ----------------- | --------- | ----------------------------------------- |
| `forward`         | DEBUG     | High volume, only logged when level=DEBUG |
| `retrieval`       | INFO      | Important operational event               |
| `audit` (success) | INFO      | Normal operation                          |
| `audit` (failure) | WARNING   | Needs attention                           |

### 4.4 AuditStorage

#### Overview

Persists audit events to external storage for compliance and debugging.

#### File Backend (JSONL)

```python
from aga_observability import FileAuditStorage

storage = FileAuditStorage(
    path="audit/aga_audit.jsonl",
    flush_interval=10,    # Flush every 10 seconds
    batch_size=100,       # Or when buffer reaches 100 entries
    max_file_size=100_000_000,  # 100MB per file
    max_files=10,         # Keep 10 rotated files
)
storage.subscribe(plugin.event_bus)
```

**File format** (one JSON per line):

```jsonl
{"timestamp": 1740300600.0, "operation": "register", "details": {"id": "fact_001", "reliability": 1.0}, "success": true, "error": null}
{"timestamp": 1740300601.0, "operation": "register", "details": {"id": "fact_002", "reliability": 0.9}, "success": true, "error": null}
```

#### SQLite Backend

```python
from aga_observability import SQLiteAuditStorage

storage = SQLiteAuditStorage(
    path="audit/aga_audit.db",
    flush_interval=10,
    batch_size=100,
)
storage.subscribe(plugin.event_bus)
```

**Database schema:**

```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    operation TEXT NOT NULL,
    details TEXT,           -- JSON string
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Indexes on timestamp and operation
```

#### Querying Audit Records

```python
# Query by operation type
registers = storage.query(operation="register", limit=50)

# Query by time range
import time
recent = storage.query(since=time.time() - 86400)  # Last 24 hours

# Cleanup old records
deleted = storage.cleanup(retention_days=90)
```

### 4.5 HealthChecker

#### Overview

Provides component-level health status with an optional HTTP endpoint for Kubernetes probes.

#### Health Status Levels

| Status      | HTTP Code | Meaning                                          |
| ----------- | --------- | ------------------------------------------------ |
| `healthy`   | 200       | All components operational                       |
| `degraded`  | 200       | Some components limited (e.g., high utilization) |
| `unhealthy` | 503       | Critical failure                                 |

#### Component Checks

| Component     | Healthy           | Degraded           | Unhealthy         |
| ------------- | ----------------- | ------------------ | ----------------- |
| `kv_store`    | utilization < 95% | utilization >= 95% | Exception         |
| `gate_system` | Parameters OK     | —                  | NaN in parameters |
| `retriever`   | Stats accessible  | Exception          | —                 |
| `event_bus`   | Buffer < 90%      | Buffer >= 90%      | Exception         |
| `attachment`  | Model attached    | Not attached       | Exception         |

#### Usage

```python
from aga_observability import HealthChecker

checker = HealthChecker()
checker.bind_plugin(plugin)

# Programmatic check
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

# Start HTTP endpoint
checker.start_server(port=8080, path="/health")
# Access: GET http://localhost:8080/health
```

#### Custom Health Checks

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

#### Kubernetes Probe Configuration

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

### 4.6 GrafanaDashboardGenerator

#### Overview

Generates a complete Grafana Dashboard JSON that can be imported directly.

#### Usage

```python
from aga_observability import GrafanaDashboardGenerator

gen = GrafanaDashboardGenerator(
    prefix="aga",
    datasource="Prometheus",
    title="AGA Observability Dashboard",
    refresh="5s",
)

# Save to file
gen.save("grafana_dashboard.json")

# Or get as string
json_str = gen.generate()

# Or get as dict
dashboard_dict = gen.to_dict()
```

#### Dashboard Panels

| Row                 | Panels                                                | Description               |
| ------------------- | ----------------------------------------------------- | ------------------------- |
| Overview            | Activation Rate, Knowledge Count, Utilization, Pinned | Key metrics at a glance   |
| Forward Performance | Latency P50/P95/P99, QPS (applied vs bypassed)        | Performance monitoring    |
| Gate & Entropy      | Gate Value Heatmap, Entropy Value Heatmap             | Distribution analysis     |
| Retrieval           | Call Frequency, Injection Count, Slot Change Rate     | Retriever monitoring      |
| Audit               | Operation Statistics (register/unregister/load/clear) | Audit trail visualization |

#### Import to Grafana

1. Generate the JSON file
2. Open Grafana → Dashboards → Import
3. Upload the JSON file or paste the content
4. Select your Prometheus datasource
5. Click Import

---

## 5. Configuration Reference

### ObservabilityConfig Fields

| Field                   | Type | Default          | Description                                 |
| ----------------------- | ---- | ---------------- | ------------------------------------------- |
| `enabled`               | bool | `True`           | Master switch                               |
| `prometheus_enabled`    | bool | `True`           | Enable Prometheus export                    |
| `prometheus_port`       | int  | `9090`           | Prometheus HTTP port                        |
| `prometheus_prefix`     | str  | `"aga"`          | Metric name prefix                          |
| `prometheus_labels`     | Dict | `{}`             | Global labels for all metrics               |
| `log_enabled`           | bool | `True`           | Enable log export                           |
| `log_format`            | str  | `"json"`         | Log format: `"json"` or `"text"`            |
| `log_level`             | str  | `"INFO"`         | Log level                                   |
| `log_file`              | str  | `None`           | Log file path (None = stderr only)          |
| `log_max_bytes`         | int  | `104857600`      | Max log file size (100MB)                   |
| `log_backup_count`      | int  | `5`              | Rotated file count                          |
| `audit_storage_backend` | str  | `"memory"`       | Backend: `"memory"` / `"file"` / `"sqlite"` |
| `audit_storage_path`    | str  | `"aga_audit.db"` | Audit storage path                          |
| `audit_retention_days`  | int  | `90`             | Data retention period                       |
| `audit_flush_interval`  | int  | `10`             | Flush interval (seconds)                    |
| `audit_batch_size`      | int  | `100`            | Batch write size                            |
| `alert_enabled`         | bool | `True`           | Enable alerting                             |
| `alert_rules`           | List | `[]`             | Custom alert rules                          |
| `alert_webhook_url`     | str  | `None`           | Webhook URL for notifications               |
| `alert_log_level`       | str  | `"WARNING"`      | Alert log level                             |
| `health_enabled`        | bool | `True`           | Enable health check                         |
| `health_port`           | int  | `8080`           | Health check HTTP port                      |
| `health_path`           | str  | `"/health"`      | Health check path                           |

### AlertRuleConfig Fields

| Field              | Type  | Default     | Description                                                      |
| ------------------ | ----- | ----------- | ---------------------------------------------------------------- |
| `name`             | str   | `""`        | Rule name (unique identifier)                                    |
| `metric`           | str   | `""`        | Metric to monitor                                                |
| `operator`         | str   | `">"`       | Comparison: `>`, `<`, `>=`, `<=`, `==`                           |
| `threshold`        | float | `0.0`       | Threshold value                                                  |
| `window_seconds`   | int   | `60`        | Evaluation window                                                |
| `severity`         | str   | `"warning"` | Level: `"info"`, `"warning"`, `"critical"`                       |
| `message`          | str   | `""`        | Message template (supports `{value}`, `{threshold}`, `{metric}`) |
| `cooldown_seconds` | int   | `300`       | Cooldown between repeated alerts                                 |

### Available Alert Metrics

| Metric Name              | Source             | Aggregation    |
| ------------------------ | ------------------ | -------------- |
| `latency_p99`            | forward events     | P99 in window  |
| `gate_mean`              | forward events     | Latest value   |
| `entropy_mean`           | forward events     | Latest value   |
| `activation_rate`        | forward events     | Mean in window |
| `knowledge_utilization`  | plugin metrics     | Latest value   |
| `slot_change_rate`       | plugin diagnostics | Latest value   |
| `retrieval_failure_rate` | retrieval events   | Latest value   |

---

## 6. Deployment Scenarios

### Scenario 1: Single GPU, Development

```python
config = AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,
    log_format="text",  # Human-readable
)
plugin = AGAPlugin(config)
# Logs to stderr, no Prometheus, no persistent audit
```

### Scenario 2: Single GPU, Production

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

### Scenario 3: Multi-Instance (Data Parallelism)

Each instance runs its own `ObservabilityStack`. Use Prometheus labels to distinguish:

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

### Scenario 4: Kubernetes

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

## 7. API Reference

### setup_observability()

```python
def setup_observability(event_bus, aga_config=None) -> Optional[ObservabilityStack]
```

Auto-integration entry point. Called by `AGAPlugin._setup_observability()`.

### shutdown_observability()

```python
def shutdown_observability() -> None
```

Shuts down the global singleton stack.

### ObservabilityStack

```python
class ObservabilityStack:
    def __init__(self, event_bus, config=None)
    def start(self) -> None
    def bind_plugin(self, plugin) -> None
    def shutdown(self) -> None
    def get_stats(self) -> Dict[str, Any]
    def generate_dashboard(self, path=None) -> str
```

### PrometheusExporter

```python
class PrometheusExporter:
    def __init__(self, prefix="aga", port=9090, labels=None, registry=None)
    def subscribe(self, event_bus) -> None
    def unsubscribe(self, event_bus) -> None
    def start_server(self) -> None
    def update_gauges(self, plugin) -> None
    def set_build_info(self, info: Dict[str, str]) -> None
    def shutdown(self) -> None
    def get_stats(self) -> Dict[str, Any]
```

### AlertManager

```python
class AlertManager:
    def __init__(self, rules=None, use_defaults=True, max_history=1000, webhook_url=None)
    def add_rule(self, rule: AlertRule) -> None
    def remove_rule(self, name: str) -> bool
    def add_callback(self, callback: Callable[[AlertEvent], None]) -> None
    def subscribe(self, event_bus) -> None
    def unsubscribe(self, event_bus) -> None
    def update_plugin_metrics(self, plugin) -> None
    def get_alerts(self, severity=None, limit=100, since=None) -> List[Dict]
    def get_stats(self) -> Dict[str, Any]
    def shutdown(self) -> None
```

### LogExporter

```python
class LogExporter:
    def __init__(self, format="json", level="INFO", file=None, max_bytes=..., backup_count=5)
    def subscribe(self, event_bus) -> None
    def unsubscribe(self, event_bus) -> None
    def get_stats(self) -> Dict[str, Any]
    def shutdown(self) -> None
```

### FileAuditStorage / SQLiteAuditStorage

```python
class FileAuditStorage(AuditStorageBackend):
    def __init__(self, path="aga_audit.jsonl", flush_interval=10, batch_size=100, ...)
    def subscribe(self, event_bus) -> None
    def unsubscribe(self, event_bus) -> None
    def store(self, entry: Dict) -> None
    def store_batch(self, entries: List[Dict]) -> None
    def query(self, operation=None, since=None, limit=100) -> List[Dict]
    def cleanup(self, retention_days: int) -> int
    def shutdown(self) -> None
    def get_stats(self) -> Dict[str, Any]

class SQLiteAuditStorage(AuditStorageBackend):
    # Same interface as FileAuditStorage
```

### HealthChecker

```python
class HealthChecker:
    def __init__(self)
    def bind_plugin(self, plugin) -> None
    def add_check(self, name: str, check_fn: callable) -> None
    def check(self) -> Dict[str, Any]
    def start_server(self, port=8080, path="/health") -> None
    def shutdown(self) -> None
```

### GrafanaDashboardGenerator

```python
class GrafanaDashboardGenerator:
    def __init__(self, prefix="aga", datasource="Prometheus", title="...", refresh="5s")
    def generate(self) -> str       # JSON string
    def to_dict(self) -> Dict       # Python dict
    def save(self, path: str) -> None
```

---

## 8. Troubleshooting

### Prometheus metrics not appearing

1. Verify `prometheus-client` is installed: `pip install prometheus-client`
2. Check the port is not in use: `netstat -tlnp | grep 9090`
3. Verify scrape config in `prometheus.yml`
4. Check logs for `PrometheusExporter 创建失败` messages

### Health check returns 503

1. Run `checker.check()` programmatically to see which component is unhealthy
2. Common causes:
    - GateSystem parameters contain NaN (training instability)
    - KVStore utilization > 95% (increase `max_slots` or enable slot governance)
    - Model not attached (call `plugin.attach(model)` first)

### Audit file growing too large

1. Enable rotation: `max_file_size=100_000_000, max_files=10`
2. Run periodic cleanup: `storage.cleanup(retention_days=30)`
3. Consider switching to SQLite backend for better query performance

### Alerts firing too frequently

1. Increase `cooldown_seconds` on the rule
2. Adjust `threshold` values
3. Increase `window_seconds` for more stable aggregation

### "aga-observability 自动集成失败" in logs

1. This is a Fail-Open warning — inference is not affected
2. Check the full error message for details
3. Common causes: port conflicts, missing dependencies, permission issues

---

## 9. FAQ

**Q: Does aga-observability affect inference performance?**

A: Minimal impact. Event processing is lightweight (dict creation + deque append). Prometheus metric updates use atomic operations. File/SQLite writes are batched and asynchronous. The hot path (forward event handling) adds < 1μs overhead.

**Q: Can I use aga-observability without Prometheus?**

A: Yes. `prometheus-client` is an optional dependency. Without it, `PrometheusExporter` is skipped, but logging, auditing, alerting, and health checks still work.

**Q: How do I monitor multiple AGA instances?**

A: Each instance runs its own `ObservabilityStack` with unique Prometheus labels (e.g., `instance`, `rank`). Use Prometheus federation or a shared Prometheus server to aggregate.

**Q: Can I add custom metrics?**

A: Currently, custom metrics require extending `PrometheusExporter`. A plugin system for custom metrics is planned for v1.2.0.

**Q: What happens if the audit database is corrupted?**

A: `SQLiteAuditStorage` uses standard SQLite with WAL mode. If corruption occurs, the storage will log a warning and continue (Fail-Open). You can delete the database file and it will be recreated.

**Q: Can I use an external time-series database instead of Prometheus?**

A: The current version exports to Prometheus format. OpenTelemetry export (supporting Datadog, New Relic, etc.) is planned for v1.1.0.

---

_aga-observability v1.0.0 — Production observability for AGA-powered LLM inference_
