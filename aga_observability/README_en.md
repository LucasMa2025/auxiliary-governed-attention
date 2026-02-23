# aga-observability

> Production-grade observability stack for **aga-core** (Auxiliary Governed Attention)

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

## Overview

**aga-observability** is a companion package for [aga-core](../aga/) that provides comprehensive monitoring, alerting, logging, auditing, and health-checking capabilities for AGA-powered LLM inference systems.

### Design Principles

| Principle            | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Zero Intrusion**   | Integrates via `EventBus` subscription — no modifications to `aga-core` source code         |
| **Auto Integration** | `pip install aga-observability` → `AGAPlugin` detects and enables it automatically          |
| **Fail-Open**        | Observability failures never block or degrade LLM inference                                 |
| **Config-Driven**    | All behavior controlled through `AGAConfig.observability_*` fields or `ObservabilityConfig` |

### Architecture

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
│  ┌───────────────────┐  ┌──────────────────┐            │
│  │  AlertManager     │  │  AuditStorage    │            │
│  │  SLO/SLI Rules    │  │  File(JSONL)     │            │
│  │  Webhook/Callback │  │  SQLite          │            │
│  └───────────────────┘  └──────────────────┘            │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  HealthChecker   │  │ GrafanaDashboard │             │
│  │  HTTP :8080      │  │  JSON Generator  │             │
│  │  K8s Probes      │  │  Auto Panels     │             │
│  └──────────────────┘  └──────────────────┘             │
│                                                         │
│  ObservabilityStack — orchestrates all components       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Core (logging, auditing, alerting, health check)
pip install aga-observability

# With Prometheus support
pip install aga-observability[prometheus]

# Full installation
pip install aga-observability[full]
```

### Automatic Integration (Recommended)

Once installed, `AGAPlugin` automatically detects and enables `aga-observability`:

```python
from aga import AGAPlugin, AGAConfig

config = AGAConfig(
    hidden_dim=4096,
    observability_enabled=True,   # default: True
    prometheus_enabled=True,
    prometheus_port=9090,
)

plugin = AGAPlugin(config)
# aga-observability is now active — no extra code needed!
```

### Manual Integration

For fine-grained control:

```python
from aga import AGAPlugin, AGAConfig
from aga_observability import ObservabilityStack, ObservabilityConfig

# Create plugin
plugin = AGAPlugin(AGAConfig(hidden_dim=4096))

# Create observability config
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

# Create and start stack
stack = ObservabilityStack(event_bus=plugin.event_bus, config=obs_config)
stack.start()
stack.bind_plugin(plugin)

# ... inference ...

# Shutdown
stack.shutdown()
```

## Components

### 1. PrometheusExporter

Exports AGA metrics to Prometheus via HTTP endpoint.

**Metrics:**

| Type      | Metric                                           | Description                       |
| --------- | ------------------------------------------------ | --------------------------------- |
| Counter   | `aga_forward_total{layer, applied}`              | Total forward calls               |
| Counter   | `aga_retrieval_total{layer}`                     | Total retrieval calls             |
| Counter   | `aga_retrieval_injected_total{layer}`            | Total injected knowledge items    |
| Counter   | `aga_audit_operations_total{operation, success}` | Total audit operations            |
| Histogram | `aga_forward_latency_us{layer}`                  | Forward latency distribution (μs) |
| Histogram | `aga_gate_value{layer}`                          | Gate value distribution           |
| Histogram | `aga_entropy_value{layer}`                       | Entropy value distribution        |
| Histogram | `aga_retrieval_score`                            | Retrieval score distribution      |
| Gauge     | `aga_knowledge_count`                            | Current knowledge count           |
| Gauge     | `aga_knowledge_utilization`                      | KVStore utilization               |
| Gauge     | `aga_knowledge_pinned_count`                     | Pinned knowledge count            |
| Gauge     | `aga_activation_rate`                            | Activation rate (sliding window)  |
| Gauge     | `aga_slot_change_rate`                           | Slot change rate                  |
| Info      | `aga_build`                                      | Build information                 |

### 2. AlertManager

Real-time SLO/SLI alerting with configurable rules.

**Default SLO Rules:**

| Rule                   | Metric                  | Condition | Severity |
| ---------------------- | ----------------------- | --------- | -------- |
| `slo_latency_p99`      | `latency_p99`           | > 1000μs  | WARNING  |
| `slo_latency_critical` | `latency_p99`           | > 5000μs  | CRITICAL |
| `high_utilization`     | `knowledge_utilization` | > 95%     | WARNING  |
| `slot_thrashing`       | `slot_change_rate`      | > 0.5     | WARNING  |

**Alert Channels:** Logging, Webhook (HTTP POST), Custom Callbacks

### 3. LogExporter

Structured event logging with file rotation support.

-   **Formats:** JSON (for ELK/Loki) or plain text
-   **Output:** stderr + optional file with rotation
-   **Events:** Subscribes to all EventBus events (`*`)

### 4. AuditStorage

Persistent audit trail for knowledge operations.

-   **File Backend:** JSONL format with rotation and batch writes
-   **SQLite Backend:** Indexed database with retention policies
-   **Features:** Batch buffering, async flush, automatic cleanup

### 5. HealthChecker

Comprehensive health checking with HTTP endpoint.

**Checked Components:**

-   KVStore (utilization, count, pinned)
-   GateSystem (parameter integrity, NaN detection)
-   Retriever (type, stats)
-   EventBus (buffer usage)
-   Attachment status (model hooks)

**HTTP Endpoint:** `GET /health` → 200 (healthy/degraded) or 503 (unhealthy)

### 6. GrafanaDashboardGenerator

Auto-generates Grafana Dashboard JSON with 5 panel groups:

-   Overview (activation rate, knowledge count, utilization, pinned)
-   Forward Performance (latency P50/P95/P99, QPS)
-   Gate & Entropy (heatmaps)
-   Retrieval (call frequency, injection count, slot change rate)
-   Audit (operation statistics)

```python
stack.generate_dashboard("aga_dashboard.json")
# Import the JSON file into Grafana
```

## Configuration Reference

### Via AGAConfig (Automatic Mapping)

```python
config = AGAConfig(
    # Observability master switch
    observability_enabled=True,

    # Prometheus
    prometheus_enabled=True,
    prometheus_port=9090,

    # Logging
    log_format="json",        # "json" or "text"
    log_level="INFO",

    # Audit
    audit_storage_backend="sqlite",  # "memory" / "file" / "sqlite"
    audit_retention_days=90,
)
```

### Via ObservabilityConfig (Full Control)

```python
from aga_observability import ObservabilityConfig, AlertRuleConfig

config = ObservabilityConfig(
    enabled=True,

    # Prometheus
    prometheus_enabled=True,
    prometheus_port=9090,
    prometheus_prefix="aga",
    prometheus_labels={"env": "production", "cluster": "gpu-01"},

    # Logging
    log_enabled=True,
    log_format="json",
    log_level="INFO",
    log_file="/var/log/aga/events.log",
    log_max_bytes=100 * 1024 * 1024,  # 100MB
    log_backup_count=5,

    # Audit
    audit_storage_backend="sqlite",
    audit_storage_path="/data/aga_audit.db",
    audit_retention_days=90,
    audit_flush_interval=10,
    audit_batch_size=100,

    # Alerting
    alert_enabled=True,
    alert_webhook_url="https://hooks.slack.com/services/...",
    alert_rules=[
        AlertRuleConfig(
            name="custom_high_entropy",
            metric="entropy_mean",
            operator=">",
            threshold=5.0,
            severity="critical",
            message="Entropy too high: {value:.2f}",
        ),
    ],

    # Health Check
    health_enabled=True,
    health_port=8080,
    health_path="/health",
)
```

## Project Structure

```
aga_observability/
├── __init__.py              # Package entry, exports all public APIs
├── config.py                # ObservabilityConfig + AlertRuleConfig
├── integration.py           # Auto-integration entry (singleton)
├── stack.py                 # ObservabilityStack — component orchestrator
├── prometheus_exporter.py   # Prometheus metrics export
├── grafana_dashboard.py     # Grafana Dashboard JSON generator
├── alert_manager.py         # SLO/SLI alert management
├── log_exporter.py          # Structured log export (JSON/Text)
├── audit_storage.py         # Audit persistence (File/SQLite)
├── health.py                # Health checker + HTTP endpoint
├── pyproject.toml           # Package metadata
├── README_en.md             # English README
├── README_zh.md             # Chinese README
└── docs/
    ├── user_manual_en.md    # English user manual
    └── user_manual_zh.md    # Chinese user manual
```

## Dependencies

| Dependency                                                  | Required | Purpose                      |
| ----------------------------------------------------------- | -------- | ---------------------------- |
| `aga-core >= 4.4.0`                                         | Yes      | Core AGA plugin (EventBus)   |
| `prometheus-client >= 0.17.0`                               | Optional | Prometheus metrics export    |
| Python stdlib (`sqlite3`, `json`, `logging`, `http.server`) | Yes      | Audit, logging, health check |

## Relationship with aga-core

`aga-observability` is a **pure add-on** package:

-   **Without it:** `aga-core` works normally with built-in `ForwardMetrics` (in-memory ring buffer) and `AuditLog` (in-memory deque). No external metrics, no persistent audit, no alerting.
-   **With it:** All `EventBus` events are captured and exported to Prometheus, structured logs, persistent audit storage, with real-time alerting and health checking.
-   **Integration:** Zero-code — `AGAPlugin._setup_observability()` auto-detects the package on import.

## Roadmap

| Version | Feature                                                    | Status      |
| ------- | ---------------------------------------------------------- | ----------- |
| v1.0.0  | Prometheus + Grafana + Alerting + Logging + Audit + Health | ✅ Released |
| v1.1.0  | OpenTelemetry Traces export                                | Planned     |
| v1.2.0  | Distributed metrics aggregation (multi-instance)           | Planned     |
| v1.3.0  | Anomaly detection (auto-threshold)                         | Planned     |
| v2.0.0  | Real-time dashboard (WebSocket)                            | Planned     |

## License

MIT License
