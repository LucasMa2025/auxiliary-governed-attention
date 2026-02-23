# aga-knowledge 数据库 Schema

aga-knowledge v0.3.0 数据库架构文件。

## 与原 AGA Schema 的区别

| 特性 | 原 AGA (aga-core) | aga-knowledge |
|------|-------------------|---------------|
| 向量字段 | ✅ key_vector / value_vector | ❌ 不存储向量 |
| 槽位管理 | ✅ slot_idx | ❌ 不管理 GPU 槽位 |
| 版本历史 | ❌ | ✅ knowledge_versions |
| 文档管理 | ❌ | ✅ document_sources |
| 图片资产 | ❌ | ✅ image_assets |
| 编码器版本 | ❌ | ✅ encoder_versions |
| 治理投票 | ✅ governance_votes | ❌ 不需要 |
| 实例心跳 | ✅ instance_heartbeats | ❌ 不需要 |

> **设计原则**: aga-knowledge 只存储明文 condition/decision 文本对。向量化由 Encoder 在运行时处理，通过 BaseRetriever 协议与 aga-core 通讯。

## 文件说明

| 文件 | 用途 | 数据库 |
|------|------|--------|
| `postgresql_schema.sql` | 生产环境 | PostgreSQL 12+ |
| `sqlite_schema.sql` | 开发/测试环境 | SQLite 3.35+ |

## PostgreSQL 安装

```bash
# 创建数据库
createdb -U postgres aga_knowledge

# 执行 schema
psql -U postgres -d aga_knowledge -f postgresql_schema.sql

# 或使用连接字符串
psql "postgresql://user:password@localhost:5432/aga_knowledge" -f postgresql_schema.sql
```

## SQLite 安装

```bash
# 创建并初始化数据库
sqlite3 aga_knowledge.db < sqlite_schema.sql

# 或在 Python 中
python -c "
import sqlite3
with open('sqlite_schema.sql') as f:
    conn = sqlite3.connect('aga_knowledge.db')
    conn.executescript(f.read())
    conn.close()
"
```

## 表结构

### 核心表

#### `namespaces`

命名空间（租户/领域隔离）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | VARCHAR(255) | 命名空间名称（唯一） |
| description | TEXT | 描述 |
| metadata | JSONB | 扩展元数据 |

#### `knowledge`

知识表（明文 condition/decision 对，**不含向量字段**）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| namespace | VARCHAR(255) | 命名空间 |
| lu_id | VARCHAR(255) | Learning Unit ID |
| condition_text | TEXT | 触发条件描述 |
| decision_text | TEXT | 决策/知识内容 |
| lifecycle_state | VARCHAR(50) | 生命周期状态 |
| trust_tier | VARCHAR(50) | 信任层级 |
| hit_count | BIGINT | 命中次数 |
| version | INTEGER | 当前版本号 |
| encoder_version | VARCHAR(100) | 编码器版本标记 |
| source_document_id | VARCHAR(255) | 关联源文档 ID |
| metadata | JSONB | 扩展元数据 |

#### `knowledge_versions`

知识版本历史（支持回滚和审计）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| namespace | VARCHAR(255) | 命名空间 |
| lu_id | VARCHAR(255) | Learning Unit ID |
| version | INTEGER | 版本号 |
| condition_text | TEXT | 版本快照 - 条件 |
| decision_text | TEXT | 版本快照 - 决策 |
| created_by | VARCHAR(255) | 创建者 |
| change_reason | TEXT | 变更原因 |

#### `audit_logs`

审计日志

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| namespace | VARCHAR(255) | 命名空间 |
| lu_id | VARCHAR(255) | 相关 LU ID |
| action | VARCHAR(100) | 操作类型 |
| old_state | VARCHAR(50) | 旧状态 |
| new_state | VARCHAR(50) | 新状态 |
| reason | TEXT | 变更原因 |
| source | VARCHAR(100) | 来源（portal/api/sync） |
| operator | VARCHAR(255) | 操作者 |
| details | JSONB | 详细信息 |

### 文档管理表

#### `document_sources`

源文档管理（知识分片的来源）

#### `image_assets`

知识文档图片资产（与文档上下文对齐，通过 Portal 提供访问 URL）

### 编码器管理表

#### `encoder_versions`

编码器版本管理（投影层更新后触发重编码）

| 字段 | 类型 | 说明 |
|------|------|------|
| version_tag | VARCHAR(100) | 版本标签（唯一） |
| backend | VARCHAR(100) | 编码器后端 |
| key_dim | INTEGER | Key 维度（对齐 aga-core bottleneck_dim） |
| value_dim | INTEGER | Value 维度（对齐 aga-core hidden_dim） |
| key_norm_target | REAL | Key 范数目标 |
| value_norm_target | REAL | Value 范数目标 |
| projection_hash | VARCHAR(64) | 投影层权重 SHA256 |

## 生命周期状态

| 状态 | 值 | 说明 |
|------|-----|------|
| 试用期 | `probationary` | 新注入的知识 |
| 已确认 | `confirmed` | 验证通过 |
| 已弃用 | `deprecated` | 准备下线 |
| 已隔离 | `quarantined` | 不参与检索 |

## 信任层级

| 层级 | 值 | 说明 |
|------|-----|------|
| 系统级 | `system` | 核心系统知识 |
| 已验证 | `verified` | 经过验证的知识 |
| 标准 | `standard` | 默认层级 |
| 实验性 | `experimental` | 实验性知识 |
| 不可信 | `untrusted` | 待验证的知识 |

## 视图

| 视图 | 说明 |
|------|------|
| `active_knowledge` | 活跃知识（非隔离状态，供 Retriever 加载） |
| `namespace_stats` | 命名空间统计 |
| `recent_audit` | 近 7 天审计日志 |
| `document_chunk_stats` | 文档分片统计 |
| `active_alerts` | 活跃告警 |
| `pending_reencode` | 待重编码知识（编码器版本更新后） |

## PostgreSQL 存储过程

| 函数 | 说明 |
|------|------|
| `increment_hit_count()` | 批量增加命中计数 |
| `batch_quarantine()` | 批量隔离知识 |
| `get_namespace_statistics()` | 获取命名空间完整统计 |
| `rollback_knowledge()` | 回滚知识到指定版本 |
| `process_sync_queue()` | 处理同步队列（带行锁） |
| `mark_for_reencoding()` | 标记需要重编码的知识 |
| `record_alert()` | 记录系统告警 |

## 性能建议

### PostgreSQL

1. **连接池**: 使用 PgBouncer 或 asyncpg 内置连接池
2. **文本搜索**: 已启用 pg_trgm 三元组索引，支持模糊搜索
3. **分区**: 大规模部署时按命名空间分区
4. **真空**: 定期执行 `VACUUM ANALYZE`

```sql
-- 分区示例
CREATE TABLE knowledge_partitioned (
    LIKE knowledge INCLUDING ALL
) PARTITION BY LIST (namespace);

CREATE TABLE knowledge_default PARTITION OF knowledge_partitioned DEFAULT;
CREATE TABLE knowledge_domain_a PARTITION OF knowledge_partitioned
    FOR VALUES IN ('domain_a');
```

### SQLite

1. **WAL 模式**: 启用写前日志
2. **同步模式**: 根据需求调整

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB cache
PRAGMA temp_store = MEMORY;
```

## 迁移

从 SQLite 迁移到 PostgreSQL:

```bash
# 导出 SQLite 数据
sqlite3 aga_knowledge.db ".dump knowledge" > knowledge_data.sql

# 转换并导入 PostgreSQL
# (需要手动调整 SQL 语法差异)
```

## 备份

### PostgreSQL

```bash
pg_dump -U postgres aga_knowledge > aga_knowledge_backup.sql
```

### SQLite

```bash
sqlite3 aga_knowledge.db ".backup aga_knowledge_backup.db"
```
