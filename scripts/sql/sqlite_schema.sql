-- ============================================================
-- AGA (Auxiliary Governed Attention) SQLite Schema
-- Version: 3.1
-- 
-- 开发/测试环境数据库架构
-- 
-- 使用方式:
--   sqlite3 aga_data.db < sqlite_schema.sql
-- ============================================================

-- 启用外键约束
PRAGMA foreign_keys = ON;

-- ============================================================
-- 表定义
-- ============================================================

-- 命名空间表
CREATE TABLE IF NOT EXISTS namespaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    max_slots INTEGER DEFAULT 100,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- AGA 配置表
CREATE TABLE IF NOT EXISTS aga_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL UNIQUE,
    config TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 知识槽位表（核心表）
CREATE TABLE IF NOT EXISTS knowledge_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    slot_idx INTEGER NOT NULL,
    lu_id TEXT NOT NULL,
    condition TEXT,
    decision TEXT,
    -- 向量存储（JSON 格式）
    key_vector TEXT NOT NULL,
    value_vector TEXT NOT NULL,
    key_vector_dim INTEGER NOT NULL,
    value_vector_dim INTEGER NOT NULL,
    -- 生命周期
    lifecycle_state TEXT NOT NULL DEFAULT 'probationary',
    -- 信任层级
    trust_tier TEXT DEFAULT 's1_experience',
    -- 统计
    reliability REAL DEFAULT 0.3,
    hit_count INTEGER DEFAULT 0,
    miss_count INTEGER DEFAULT 0,
    consecutive_misses INTEGER DEFAULT 0,
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_hit_at TEXT,
    -- 扩展元数据
    metadata TEXT DEFAULT '{}',
    -- 约束
    UNIQUE(namespace, lu_id),
    UNIQUE(namespace, slot_idx)
);

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT,
    slot_idx INTEGER,
    action TEXT NOT NULL,
    old_state TEXT,
    new_state TEXT,
    reason TEXT,
    source_instance TEXT,
    operator TEXT,
    -- 详细信息
    details TEXT DEFAULT '{}',
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 同步状态表（用于分布式）
CREATE TABLE IF NOT EXISTS sync_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL UNIQUE,
    last_sync_at TEXT NOT NULL DEFAULT (datetime('now')),
    sync_version INTEGER DEFAULT 0,
    dirty_slots TEXT DEFAULT '[]',
    -- 分布式锁
    lock_holder TEXT,
    lock_acquired_at TEXT,
    lock_expires_at TEXT
);

-- 治理投票表（用于 quorum 机制）
CREATE TABLE IF NOT EXISTS governance_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT NOT NULL,
    vote_type TEXT NOT NULL,  -- quarantine, approve, reject
    source_instance TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT,
    -- 唯一约束
    UNIQUE(namespace, lu_id, vote_type, source_instance)
);

-- 传播队列表
CREATE TABLE IF NOT EXISTS propagation_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT NOT NULL,
    action TEXT NOT NULL,  -- inject, update, quarantine
    target_instances TEXT DEFAULT '[]',
    payload TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- 索引
-- ============================================================

-- knowledge_slots 索引
CREATE INDEX IF NOT EXISTS idx_knowledge_namespace ON knowledge_slots(namespace);
CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id ON knowledge_slots(lu_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_state ON knowledge_slots(namespace, lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_knowledge_trust ON knowledge_slots(namespace, trust_tier);
CREATE INDEX IF NOT EXISTS idx_knowledge_hit_count ON knowledge_slots(namespace, hit_count DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_updated ON knowledge_slots(updated_at DESC);

-- audit_logs 索引
CREATE INDEX IF NOT EXISTS idx_audit_namespace ON audit_logs(namespace);
CREATE INDEX IF NOT EXISTS idx_audit_lu_id ON audit_logs(namespace, lu_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);

-- governance_votes 索引
CREATE INDEX IF NOT EXISTS idx_votes_namespace_lu ON governance_votes(namespace, lu_id);
CREATE INDEX IF NOT EXISTS idx_votes_expires ON governance_votes(expires_at);

-- propagation_queue 索引
CREATE INDEX IF NOT EXISTS idx_propagation_status ON propagation_queue(status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_propagation_namespace ON propagation_queue(namespace, lu_id);

-- ============================================================
-- 触发器
-- ============================================================

-- 更新时间戳触发器（knowledge_slots）
CREATE TRIGGER IF NOT EXISTS update_knowledge_updated_at
    AFTER UPDATE ON knowledge_slots
    FOR EACH ROW
    WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE knowledge_slots 
    SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- 更新时间戳触发器（aga_configs）
CREATE TRIGGER IF NOT EXISTS update_configs_updated_at
    AFTER UPDATE ON aga_configs
    FOR EACH ROW
    WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE aga_configs 
    SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- 更新时间戳触发器（namespaces）
CREATE TRIGGER IF NOT EXISTS update_namespaces_updated_at
    AFTER UPDATE ON namespaces
    FOR EACH ROW
    WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE namespaces 
    SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- 审计日志触发器（INSERT）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_insert
    AFTER INSERT ON knowledge_slots
    FOR EACH ROW
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, slot_idx, action, new_state, details)
    VALUES (
        NEW.namespace, 
        NEW.lu_id, 
        NEW.slot_idx, 
        'create', 
        NEW.lifecycle_state,
        json_object('condition', NEW.condition, 'decision', NEW.decision)
    );
END;

-- 审计日志触发器（UPDATE lifecycle）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_lifecycle_update
    AFTER UPDATE OF lifecycle_state ON knowledge_slots
    FOR EACH ROW
    WHEN OLD.lifecycle_state != NEW.lifecycle_state
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, slot_idx, action, old_state, new_state)
    VALUES (
        NEW.namespace, 
        NEW.lu_id, 
        NEW.slot_idx, 
        'lifecycle_change', 
        OLD.lifecycle_state, 
        NEW.lifecycle_state
    );
END;

-- 审计日志触发器（DELETE）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_delete
    AFTER DELETE ON knowledge_slots
    FOR EACH ROW
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, slot_idx, action, old_state)
    VALUES (OLD.namespace, OLD.lu_id, OLD.slot_idx, 'delete', OLD.lifecycle_state);
END;

-- ============================================================
-- 视图
-- ============================================================

-- 活跃知识视图
CREATE VIEW IF NOT EXISTS active_knowledge AS
SELECT 
    namespace,
    slot_idx,
    lu_id,
    condition,
    decision,
    lifecycle_state,
    trust_tier,
    reliability,
    hit_count,
    created_at,
    updated_at
FROM knowledge_slots
WHERE lifecycle_state != 'quarantined';

-- 命名空间统计视图
CREATE VIEW IF NOT EXISTS namespace_stats AS
SELECT 
    namespace,
    COUNT(*) as total_slots,
    SUM(CASE WHEN lifecycle_state != 'quarantined' THEN 1 ELSE 0 END) as active_slots,
    SUM(CASE WHEN lifecycle_state = 'probationary' THEN 1 ELSE 0 END) as probationary_count,
    SUM(CASE WHEN lifecycle_state = 'confirmed' THEN 1 ELSE 0 END) as confirmed_count,
    SUM(CASE WHEN lifecycle_state = 'deprecated' THEN 1 ELSE 0 END) as deprecated_count,
    SUM(CASE WHEN lifecycle_state = 'quarantined' THEN 1 ELSE 0 END) as quarantined_count,
    COALESCE(SUM(hit_count), 0) as total_hits,
    COALESCE(AVG(reliability), 0) as avg_reliability
FROM knowledge_slots
GROUP BY namespace;

-- 近期审计视图
CREATE VIEW IF NOT EXISTS recent_audit AS
SELECT 
    id,
    namespace,
    lu_id,
    action,
    old_state,
    new_state,
    reason,
    source_instance,
    created_at
FROM audit_logs
WHERE created_at >= datetime('now', '-7 days')
ORDER BY created_at DESC;

-- ============================================================
-- 初始数据
-- ============================================================

-- 创建默认命名空间
INSERT OR IGNORE INTO namespaces (name, description, max_slots)
VALUES ('default', 'Default namespace', 100);

-- ============================================================
-- 实用 SQL 示例
-- ============================================================

-- 以下是一些常用的查询示例，不会自动执行

/*
-- 1. 获取命名空间统计
SELECT * FROM namespace_stats WHERE namespace = 'default';

-- 2. 查询活跃知识
SELECT * FROM active_knowledge 
WHERE namespace = 'default' 
ORDER BY hit_count DESC 
LIMIT 10;

-- 3. 获取近期审计日志
SELECT * FROM recent_audit 
WHERE namespace = 'default'
LIMIT 50;

-- 4. 批量更新生命周期状态
UPDATE knowledge_slots
SET lifecycle_state = 'confirmed',
    reliability = 1.0
WHERE namespace = 'default'
  AND lifecycle_state = 'probationary'
  AND hit_count >= 10;

-- 5. 隔离低可靠性知识
UPDATE knowledge_slots
SET lifecycle_state = 'quarantined',
    reliability = 0.0
WHERE namespace = 'default'
  AND consecutive_misses >= 5;

-- 6. 增加命中计数
UPDATE knowledge_slots
SET hit_count = hit_count + 1,
    last_hit_at = datetime('now'),
    consecutive_misses = 0
WHERE namespace = 'default'
  AND lu_id IN ('LU_001', 'LU_002');

-- 7. 清理过期投票
DELETE FROM governance_votes
WHERE expires_at < datetime('now');

-- 8. 获取待处理的传播任务
SELECT * FROM propagation_queue
WHERE status = 'pending'
  AND scheduled_at <= datetime('now')
  AND retry_count < max_retries
ORDER BY scheduled_at
LIMIT 100;

-- 9. 查询信任层级分布
SELECT 
    trust_tier,
    COUNT(*) as count,
    AVG(reliability) as avg_reliability
FROM knowledge_slots
WHERE namespace = 'default'
GROUP BY trust_tier;

-- 10. 查询最近修改的知识
SELECT * FROM knowledge_slots
WHERE namespace = 'default'
ORDER BY updated_at DESC
LIMIT 20;
*/

-- ============================================================
-- 完成
-- ============================================================

-- 检查表是否创建成功
SELECT 'AGA SQLite Schema v3.1 installed successfully' as message;
SELECT 'Tables: ' || group_concat(name, ', ') as tables 
FROM sqlite_master 
WHERE type = 'table' AND name NOT LIKE 'sqlite_%';
