# AGA 数据库 Schema

AGA v3.1 数据库架构文件。

## 文件说明

| 文件                    | 用途          | 数据库         |
| ----------------------- | ------------- | -------------- |
| `postgresql_schema.sql` | 生产环境      | PostgreSQL 12+ |
| `sqlite_schema.sql`     | 开发/测试环境 | SQLite 3.35+   |

## PostgreSQL 安装

```bash
# 创建数据库
createdb -U postgres aga

# 执行 schema
psql -U postgres -d aga -f postgresql_schema.sql

# 或使用连接字符串
psql "postgresql://user:password@localhost:5432/aga" -f postgresql_schema.sql
```

## SQLite 安装

```bash
# 创建并初始化数据库
sqlite3 aga_data.db < sqlite_schema.sql

# 或在 Python 中
python -c "
import sqlite3
with open('sqlite_schema.sql') as f:
    conn = sqlite3.connect('aga_data.db')
    conn.executescript(f.read())
    conn.close()
"
```

## 表结构

### 核心表

#### `namespaces`

命名空间（租户隔离）

| 字段        | 类型         | 说明                 |
| ----------- | ------------ | -------------------- |
| id          | INTEGER      | 主键                 |
| name        | VARCHAR(255) | 命名空间名称（唯一） |
| description | TEXT         | 描述                 |
| max_slots   | INTEGER      | 最大槽位数           |
| metadata    | JSONB        | 扩展元数据           |

#### `knowledge_slots`

知识槽位（核心表）

| 字段            | 类型         | 说明             |
| --------------- | ------------ | ---------------- |
| id              | BIGINT       | 主键             |
| namespace       | VARCHAR(255) | 命名空间         |
| slot_idx        | INTEGER      | 槽位索引         |
| lu_id           | VARCHAR(255) | Learning Unit ID |
| condition       | TEXT         | 触发条件         |
| decision        | TEXT         | 决策/动作        |
| key_vector      | JSONB        | 条件编码向量     |
| value_vector    | JSONB        | 决策编码向量     |
| lifecycle_state | VARCHAR(50)  | 生命周期状态     |
| trust_tier      | VARCHAR(50)  | 信任层级         |
| reliability     | DECIMAL      | 可靠性分数       |
| hit_count       | BIGINT       | 命中次数         |
| metadata        | JSONB        | 扩展元数据       |

#### `audit_logs`

审计日志

| 字段            | 类型         | 说明       |
| --------------- | ------------ | ---------- |
| id              | BIGINT       | 主键       |
| namespace       | VARCHAR(255) | 命名空间   |
| lu_id           | VARCHAR(255) | 相关 LU ID |
| action          | VARCHAR(100) | 操作类型   |
| old_state       | VARCHAR(50)  | 旧状态     |
| new_state       | VARCHAR(50)  | 新状态     |
| reason          | TEXT         | 变更原因   |
| source_instance | VARCHAR(255) | 来源实例   |
| details         | JSONB        | 详细信息   |
| created_at      | TIMESTAMP    | 创建时间   |

### 分布式表

#### `sync_states`

同步状态

#### `governance_votes`

治理投票（quorum 机制）

#### `propagation_queue`

传播队列

## 生命周期状态

| 状态   | 值             | 可靠性 | 说明         |
| ------ | -------------- | ------ | ------------ |
| 试用期 | `probationary` | 0.3    | 新注入的知识 |
| 已确认 | `confirmed`    | 1.0    | 验证通过     |
| 已弃用 | `deprecated`   | 0.1    | 准备下线     |
| 已隔离 | `quarantined`  | 0.0    | 不参与推理   |

## 信任层级

| 层级    | 值                | 说明             |
| ------- | ----------------- | ---------------- |
| S0 加速 | `s0_acceleration` | 可丢失的缓存知识 |
| S1 经验 | `s1_experience`   | 可回滚的经验知识 |
| S2 策略 | `s2_policy`       | 需审批的策略知识 |
| S3 禁止 | `s3_immutable`    | 只读的核心知识   |

## 视图

### `active_knowledge`

活跃知识（非隔离状态）

### `namespace_stats`

命名空间统计

### `recent_audit`

近 7 天审计日志

## 性能建议

### PostgreSQL

1. **连接池**: 使用 PgBouncer 或内置连接池
2. **向量搜索**: 考虑使用 pgvector 扩展
3. **分区**: 大规模部署时按命名空间分区
4. **真空**: 定期执行 VACUUM ANALYZE

```sql
-- 分区示例
CREATE TABLE knowledge_slots_partitioned (
    LIKE knowledge_slots INCLUDING ALL
) PARTITION BY LIST (namespace);

CREATE TABLE knowledge_slots_default PARTITION OF knowledge_slots_partitioned DEFAULT;
CREATE TABLE knowledge_slots_tenant_a PARTITION OF knowledge_slots_partitioned
    FOR VALUES IN ('tenant_a');
```

### SQLite

1. **WAL 模式**: 启用写前日志
2. **同步模式**: 根据需求调整

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB cache
```

## 迁移

从 SQLite 迁移到 PostgreSQL:

```bash
# 导出 SQLite 数据
sqlite3 aga_data.db ".dump knowledge_slots" > knowledge_data.sql

# 转换并导入 PostgreSQL
# (需要手动调整 SQL 语法差异)
```

## 备份

### PostgreSQL

```bash
pg_dump -U postgres aga > aga_backup.sql
```

### SQLite

```bash
sqlite3 aga_data.db ".backup aga_backup.db"
```
