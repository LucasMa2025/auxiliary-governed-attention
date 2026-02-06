# AGA 测试套件

## 目录结构

```
tests/
├── conftest.py           # 全局 fixtures 和配置
├── pytest.ini            # pytest 配置
├── unit/                 # 单元测试（纯函数 & 算子级）
│   ├── core/             # 核心模块测试
│   ├── entropy_gate/     # 熵门控测试
│   ├── decay/            # 衰减模块测试
│   └── compression/      # 压缩模块测试
├── component/            # 组件测试（模块级）
│   ├── persistence/      # 持久化适配器测试
│   ├── slot_pool/        # 槽位池测试
│   └── production_gate/  # 生产门控测试
├── integration/          # 集成测试
│   ├── test_single_node.py    # 单节点测试
│   └── test_multi_runtime.py  # 多 Runtime 测试（Mock）
├── fault/                # 故障注入测试
│   ├── test_redis_down.py         # Redis 故障测试
│   ├── test_network_partition.py  # 网络分区测试
│   └── test_stale_version.py      # 版本过期测试
├── performance/          # 性能测试
│   ├── latency.py        # 延迟测试
│   ├── memory_growth.py  # 内存增长测试
│   └── long_run.py       # 长期运行测试
├── mocks/                # Mock 对象
│   ├── redis_mock.py     # Redis Mock
│   ├── postgres_mock.py  # PostgreSQL Mock
│   ├── kafka_mock.py     # Kafka Mock
│   └── http_mock.py      # HTTP Mock
└── fixtures/             # 测试数据
```

## 运行测试

### 运行所有测试

```bash
python run_tests.py
# 或
python -m pytest tests/
```

### 运行特定测试套件

```bash
# 单元测试
python run_tests.py unit
python -m pytest tests/unit/ -m unit

# 组件测试
python run_tests.py component
python -m pytest tests/component/ -m component

# 集成测试
python run_tests.py integration
python -m pytest tests/integration/ -m integration

# 故障测试
python run_tests.py fault
python -m pytest tests/fault/ -m fault

# 性能测试
python run_tests.py performance
python -m pytest tests/performance/ -m performance
```

### 跳过慢速测试

```bash
python run_tests.py --fast
python -m pytest tests/ -m "not slow"
```

### 生成覆盖率报告

```bash
python run_tests.py --coverage
python -m pytest tests/ --cov=aga --cov-report=html
```

### 并行运行测试

```bash
python run_tests.py --parallel 4
python -m pytest tests/ -n 4
```

## 测试标记

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.component` - 组件测试
- `@pytest.mark.integration` - 集成测试
- `@pytest.mark.fault` - 故障注入测试
- `@pytest.mark.performance` - 性能测试
- `@pytest.mark.slow` - 慢速测试
- `@pytest.mark.asyncio` - 异步测试

## Mock 对象

测试套件提供了完整的 Mock 实现，用于模拟外部依赖：

### MockRedis

模拟 Redis 客户端，支持：
- 字符串操作 (get, set, delete)
- 哈希操作 (hget, hset, hgetall)
- 列表操作 (lpush, rpush, lpop, rpop)
- 集合操作 (sadd, srem, smembers)
- 发布订阅 (publish, subscribe)
- 故障注入 (fail_after, latency_ms)

### MockPostgres

模拟 PostgreSQL 客户端，支持：
- 基本 CRUD 操作
- 简单 SQL 解析
- 事务模拟
- 故障注入

### MockKafka

模拟 Kafka 客户端，支持：
- 生产者/消费者
- 主题管理
- 消息分发
- 故障注入

### MockHTTPClient

模拟 HTTP 客户端，支持：
- 路由匹配
- 请求记录
- 响应模拟
- 故障注入

## 测试 Fixtures

### 基础 Fixtures

- `device` - 测试设备（CPU）
- `hidden_dim` - 隐藏层维度（768）
- `bottleneck_dim` - 瓶颈层维度（64）
- `num_slots` - 槽位数量（32）

### 向量 Fixtures

- `random_key_vector` - 随机 key 向量
- `random_value_vector` - 随机 value 向量
- `random_hidden_states` - 随机隐藏状态

### 配置 Fixtures

- `aga_config` - AGA 配置
- `entropy_gate_config` - 熵门控配置
- `decay_config` - 衰减配置
- `slot_pool_config` - 槽位池配置

### 数据 Fixtures

- `sample_knowledge_record` - 示例知识记录
- `sample_knowledge_records` - 多个示例知识记录

## 注意事项

1. **CPU 环境**: 所有测试设计为在 CPU 环境下运行
2. **Mock 分布式**: 分布式测试使用 Mock 对象，不需要实际的 Redis/Kafka
3. **异步测试**: 需要安装 `pytest-asyncio`
4. **覆盖率**: 需要安装 `pytest-cov`
5. **并行测试**: 需要安装 `pytest-xdist`

## 依赖

```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
```
