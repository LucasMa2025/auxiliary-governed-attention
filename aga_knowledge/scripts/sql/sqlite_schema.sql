-- ============================================================
-- aga-knowledge SQLite Schema
-- Version: 0.3.0
--
-- 开发/测试环境数据库架构（明文 KV 版本）
--
-- 设计原则:
--   1. 与 PostgreSQL Schema 保持表结构一致
--   2. 适配 SQLite 语法（无 JSONB、无存储过程）
--   3. 使用触发器实现自动审计和版本记录
--   4. 适用于开发、测试和小规模单机部署
--
-- 使用方式:
--   sqlite3 aga_knowledge.db < sqlite_schema.sql
--
-- Python 使用:
--   import sqlite3
--   with open('sqlite_schema.sql') as f:
--       conn = sqlite3.connect('aga_knowledge.db')
--       conn.executescript(f.read())
--       conn.close()
-- ============================================================

-- 启用外键约束
PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. 核心表
-- ============================================================

-- 命名空间表
CREATE TABLE IF NOT EXISTS namespaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

-- 知识表（核心表 — 明文 KV，无向量字段）
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT NOT NULL,
    -- 明文知识对
    condition_text TEXT,                -- 触发条件描述
    decision_text TEXT,                 -- 决策/知识内容
    -- 生命周期
    lifecycle_state TEXT NOT NULL DEFAULT 'probationary',
    -- 信任层级
    trust_tier TEXT NOT NULL DEFAULT 'standard',
    -- 统计
    hit_count INTEGER DEFAULT 0,
    consecutive_misses INTEGER DEFAULT 0,
    -- 版本
    version INTEGER DEFAULT 1,
    -- 编码器版本标记（投影层更新后触发重编码）
    encoder_version TEXT,
    -- 来源文档关联
    source_document_id TEXT,
    chunk_index INTEGER,
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_hit_at TEXT,
    -- 扩展元数据
    metadata TEXT DEFAULT '{}',
    -- 约束
    UNIQUE(namespace, lu_id)
);

-- 知识版本历史表
CREATE TABLE IF NOT EXISTS knowledge_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    -- 版本快照
    condition_text TEXT,
    decision_text TEXT,
    lifecycle_state TEXT,
    trust_tier TEXT,
    metadata TEXT DEFAULT '{}',
    -- 变更信息
    created_by TEXT DEFAULT 'system',
    change_reason TEXT,
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- 约束
    UNIQUE(namespace, lu_id, version)
);

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT,
    action TEXT NOT NULL,
    old_state TEXT,
    new_state TEXT,
    reason TEXT,
    source TEXT DEFAULT 'portal',
    operator TEXT,
    -- 详细信息
    details TEXT DEFAULT '{}',
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- 2. 文档管理表
-- ============================================================

-- 源文档表
CREATE TABLE IF NOT EXISTS document_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    document_id TEXT NOT NULL,
    -- 文档信息
    title TEXT,
    source_type TEXT DEFAULT 'markdown',  -- markdown, html, text, pdf
    original_filename TEXT,
    -- 分片信息
    total_chunks INTEGER DEFAULT 0,
    -- 分片配置快照
    chunker_strategy TEXT,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    -- 状态
    status TEXT DEFAULT 'active',  -- active, archived, deleted
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- 元数据
    metadata TEXT DEFAULT '{}',
    -- 约束
    UNIQUE(namespace, document_id)
);

-- 图片资产表
CREATE TABLE IF NOT EXISTS image_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    -- 关联
    document_id TEXT,
    lu_id TEXT,
    -- 图片信息
    alt_text TEXT,
    original_src TEXT,
    portal_url TEXT NOT NULL,
    mime_type TEXT,
    file_size INTEGER DEFAULT 0,
    width INTEGER DEFAULT 0,
    height INTEGER DEFAULT 0,
    -- 存储路径
    storage_path TEXT,
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- 元数据
    metadata TEXT DEFAULT '{}',
    -- 约束
    UNIQUE(namespace, asset_id)
);

-- ============================================================
-- 3. 同步与分布式表
-- ============================================================

-- 同步状态表
CREATE TABLE IF NOT EXISTS sync_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL UNIQUE,
    last_sync_at TEXT NOT NULL DEFAULT (datetime('now')),
    sync_version INTEGER DEFAULT 0,
    dirty_knowledge TEXT DEFAULT '[]',
    -- 分布式锁
    lock_holder TEXT,
    lock_acquired_at TEXT,
    lock_expires_at TEXT
);

-- 同步消息队列表
CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    lu_id TEXT NOT NULL,
    action TEXT NOT NULL,  -- inject, update, delete, lifecycle_change
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
-- 4. 编码器管理表
-- ============================================================

-- 编码器版本表
CREATE TABLE IF NOT EXISTS encoder_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag TEXT NOT NULL UNIQUE,
    -- 编码器信息
    backend TEXT NOT NULL,               -- sentence_transformer, simple, custom
    model_name TEXT,
    -- 对齐参数快照
    key_dim INTEGER NOT NULL,
    value_dim INTEGER NOT NULL,
    key_norm_target REAL,
    value_norm_target REAL,
    -- 投影层信息
    projection_path TEXT,
    projection_hash TEXT,                -- 投影层权重的 SHA256
    -- 状态
    status TEXT DEFAULT 'active',        -- active, deprecated, archived
    -- 时间戳
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- 元数据
    metadata TEXT DEFAULT '{}'
);

-- ============================================================
-- 5. 告警表
-- ============================================================

-- 告警记录表
CREATE TABLE IF NOT EXISTS alert_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_name TEXT NOT NULL,
    severity TEXT NOT NULL,  -- info, warning, critical
    state TEXT NOT NULL,     -- pending, firing, resolved
    summary TEXT,
    description TEXT,
    component TEXT,          -- encoder, retriever, portal, sync
    labels TEXT DEFAULT '{}',
    annotations TEXT DEFAULT '{}',
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- 索引
-- ============================================================

-- knowledge 索引
CREATE INDEX IF NOT EXISTS idx_knowledge_namespace ON knowledge(namespace);
CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id ON knowledge(lu_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_state ON knowledge(namespace, lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_knowledge_tier ON knowledge(namespace, trust_tier);
CREATE INDEX IF NOT EXISTS idx_knowledge_hit_count ON knowledge(namespace, hit_count DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_updated ON knowledge(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_source_doc ON knowledge(namespace, source_document_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_encoder_ver ON knowledge(encoder_version);

-- knowledge_versions 索引
CREATE INDEX IF NOT EXISTS idx_versions_namespace_lu ON knowledge_versions(namespace, lu_id);
CREATE INDEX IF NOT EXISTS idx_versions_created ON knowledge_versions(created_at DESC);

-- audit_logs 索引
CREATE INDEX IF NOT EXISTS idx_audit_namespace ON audit_logs(namespace);
CREATE INDEX IF NOT EXISTS idx_audit_lu_id ON audit_logs(namespace, lu_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);

-- document_sources 索引
CREATE INDEX IF NOT EXISTS idx_documents_namespace ON document_sources(namespace);
CREATE INDEX IF NOT EXISTS idx_documents_status ON document_sources(namespace, status);

-- image_assets 索引
CREATE INDEX IF NOT EXISTS idx_images_namespace ON image_assets(namespace);
CREATE INDEX IF NOT EXISTS idx_images_document ON image_assets(namespace, document_id);
CREATE INDEX IF NOT EXISTS idx_images_lu_id ON image_assets(namespace, lu_id);

-- sync_queue 索引
CREATE INDEX IF NOT EXISTS idx_sync_queue_status ON sync_queue(status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_sync_queue_namespace ON sync_queue(namespace, lu_id);

-- alert_records 索引
CREATE INDEX IF NOT EXISTS idx_alert_name ON alert_records(alert_name);
CREATE INDEX IF NOT EXISTS idx_alert_severity ON alert_records(severity);
CREATE INDEX IF NOT EXISTS idx_alert_state ON alert_records(state);

-- ============================================================
-- 触发器
-- ============================================================

-- 更新时间戳触发器（knowledge）
CREATE TRIGGER IF NOT EXISTS update_knowledge_updated_at
    AFTER UPDATE ON knowledge
    FOR EACH ROW
    WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE knowledge
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

-- 更新时间戳触发器（document_sources）
CREATE TRIGGER IF NOT EXISTS update_documents_updated_at
    AFTER UPDATE ON document_sources
    FOR EACH ROW
    WHEN OLD.updated_at = NEW.updated_at
BEGIN
    UPDATE document_sources
    SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- 审计日志触发器（INSERT）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_insert
    AFTER INSERT ON knowledge
    FOR EACH ROW
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, action, new_state, details)
    VALUES (
        NEW.namespace,
        NEW.lu_id,
        'create',
        NEW.lifecycle_state,
        json_object(
            'condition', NEW.condition_text,
            'decision', substr(NEW.decision_text, 1, 200),
            'trust_tier', NEW.trust_tier
        )
    );
END;

-- 审计日志触发器（UPDATE lifecycle）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_lifecycle_update
    AFTER UPDATE OF lifecycle_state ON knowledge
    FOR EACH ROW
    WHEN OLD.lifecycle_state != NEW.lifecycle_state
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, action, old_state, new_state)
    VALUES (
        NEW.namespace,
        NEW.lu_id,
        'lifecycle_change',
        OLD.lifecycle_state,
        NEW.lifecycle_state
    );
END;

-- 审计日志触发器（UPDATE trust_tier）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_tier_update
    AFTER UPDATE OF trust_tier ON knowledge
    FOR EACH ROW
    WHEN OLD.trust_tier != NEW.trust_tier
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, action, old_state, new_state, details)
    VALUES (
        NEW.namespace,
        NEW.lu_id,
        'trust_tier_change',
        OLD.trust_tier,
        NEW.trust_tier,
        json_object('field', 'trust_tier')
    );
END;

-- 审计日志触发器（UPDATE content）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_content_update
    AFTER UPDATE ON knowledge
    FOR EACH ROW
    WHEN OLD.condition_text IS NOT NEW.condition_text
      OR OLD.decision_text IS NOT NEW.decision_text
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, action, details)
    VALUES (
        NEW.namespace,
        NEW.lu_id,
        'content_update',
        json_object(
            'old_version', OLD.version,
            'new_version', NEW.version
        )
    );
END;

-- 审计日志触发器（DELETE）
CREATE TRIGGER IF NOT EXISTS audit_knowledge_delete
    AFTER DELETE ON knowledge
    FOR EACH ROW
BEGIN
    INSERT INTO audit_logs (namespace, lu_id, action, old_state, details)
    VALUES (
        OLD.namespace,
        OLD.lu_id,
        'delete',
        OLD.lifecycle_state,
        json_object(
            'condition', OLD.condition_text,
            'decision', substr(OLD.decision_text, 1, 200)
        )
    );
END;

-- 知识版本自动记录触发器（INSERT）
CREATE TRIGGER IF NOT EXISTS auto_version_knowledge_insert
    AFTER INSERT ON knowledge
    FOR EACH ROW
BEGIN
    INSERT INTO knowledge_versions
        (namespace, lu_id, version, condition_text, decision_text,
         lifecycle_state, trust_tier, metadata, created_by)
    VALUES
        (NEW.namespace, NEW.lu_id, NEW.version,
         NEW.condition_text, NEW.decision_text,
         NEW.lifecycle_state, NEW.trust_tier,
         NEW.metadata, 'system');
END;

-- 知识版本自动记录触发器（UPDATE）
CREATE TRIGGER IF NOT EXISTS auto_version_knowledge_update
    AFTER UPDATE ON knowledge
    FOR EACH ROW
    WHEN OLD.condition_text IS NOT NEW.condition_text
      OR OLD.decision_text IS NOT NEW.decision_text
      OR OLD.lifecycle_state != NEW.lifecycle_state
      OR OLD.trust_tier != NEW.trust_tier
BEGIN
    INSERT INTO knowledge_versions
        (namespace, lu_id, version, condition_text, decision_text,
         lifecycle_state, trust_tier, metadata, created_by)
    VALUES
        (NEW.namespace, NEW.lu_id, NEW.version,
         NEW.condition_text, NEW.decision_text,
         NEW.lifecycle_state, NEW.trust_tier,
         NEW.metadata, 'system');

    -- 清理旧版本（保留最近 10 个）
    DELETE FROM knowledge_versions
    WHERE namespace = NEW.namespace
      AND lu_id = NEW.lu_id
      AND id NOT IN (
          SELECT id FROM knowledge_versions
          WHERE namespace = NEW.namespace AND lu_id = NEW.lu_id
          ORDER BY version DESC
          LIMIT 10
      );
END;

-- ============================================================
-- 视图
-- ============================================================

-- 活跃知识视图
CREATE VIEW IF NOT EXISTS active_knowledge AS
SELECT
    namespace,
    lu_id,
    condition_text,
    decision_text,
    lifecycle_state,
    trust_tier,
    hit_count,
    version,
    encoder_version,
    source_document_id,
    created_at,
    updated_at
FROM knowledge
WHERE lifecycle_state != 'quarantined';

-- 命名空间统计视图
CREATE VIEW IF NOT EXISTS namespace_stats AS
SELECT
    namespace,
    COUNT(*) as total_knowledge,
    SUM(CASE WHEN lifecycle_state != 'quarantined' THEN 1 ELSE 0 END) as active_knowledge,
    SUM(CASE WHEN lifecycle_state = 'probationary' THEN 1 ELSE 0 END) as probationary_count,
    SUM(CASE WHEN lifecycle_state = 'confirmed' THEN 1 ELSE 0 END) as confirmed_count,
    SUM(CASE WHEN lifecycle_state = 'deprecated' THEN 1 ELSE 0 END) as deprecated_count,
    SUM(CASE WHEN lifecycle_state = 'quarantined' THEN 1 ELSE 0 END) as quarantined_count,
    COALESCE(SUM(hit_count), 0) as total_hits,
    SUM(CASE WHEN encoder_version IS NULL THEN 1 ELSE 0 END) as pending_reencode
FROM knowledge
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
    operator,
    source,
    created_at
FROM audit_logs
WHERE created_at >= datetime('now', '-7 days')
ORDER BY created_at DESC;

-- 文档分片统计视图
CREATE VIEW IF NOT EXISTS document_chunk_stats AS
SELECT
    ds.namespace,
    ds.document_id,
    ds.title,
    ds.total_chunks,
    COUNT(k.id) as actual_chunks,
    SUM(CASE WHEN k.lifecycle_state = 'confirmed' THEN 1 ELSE 0 END) as confirmed_chunks,
    COALESCE(SUM(k.hit_count), 0) as total_hits,
    (SELECT COUNT(*) FROM image_assets ia
     WHERE ia.namespace = ds.namespace
       AND ia.document_id = ds.document_id) as image_count
FROM document_sources ds
LEFT JOIN knowledge k ON k.namespace = ds.namespace
    AND k.source_document_id = ds.document_id
WHERE ds.status = 'active'
GROUP BY ds.namespace, ds.document_id, ds.title, ds.total_chunks;

-- 活跃告警视图
CREATE VIEW IF NOT EXISTS active_alerts AS
SELECT
    id,
    alert_name,
    severity,
    summary,
    description,
    component,
    started_at,
    (julianday('now') - julianday(started_at)) * 86400 as duration_seconds
FROM alert_records
WHERE state = 'firing'
ORDER BY
    CASE severity
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        ELSE 3
    END,
    started_at DESC;

-- 待重编码知识视图
CREATE VIEW IF NOT EXISTS pending_reencode AS
SELECT
    namespace,
    lu_id,
    condition_text,
    decision_text,
    encoder_version,
    json_extract(metadata, '$.pending_reencode') as reencode_info,
    updated_at
FROM knowledge
WHERE encoder_version IS NULL
ORDER BY updated_at;

-- ============================================================
-- 初始数据
-- ============================================================

-- 创建默认命名空间
INSERT OR IGNORE INTO namespaces (name, description)
VALUES ('default', 'Default namespace');

-- ============================================================
-- 实用 SQL 示例
-- ============================================================

/*
-- 推荐的 PRAGMA 设置（应用启动时执行）
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB cache
PRAGMA temp_store = MEMORY;

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

-- 4. 批量确认知识（命中次数 >= 10）
UPDATE knowledge
SET lifecycle_state = 'confirmed',
    version = version + 1
WHERE namespace = 'default'
  AND lifecycle_state = 'probationary'
  AND hit_count >= 10;

-- 5. 隔离连续未命中的知识
UPDATE knowledge
SET lifecycle_state = 'quarantined'
WHERE namespace = 'default'
  AND consecutive_misses >= 5;

-- 6. 增加命中计数
UPDATE knowledge
SET hit_count = hit_count + 1,
    last_hit_at = datetime('now'),
    consecutive_misses = 0
WHERE namespace = 'default'
  AND lu_id IN ('LU_001', 'LU_002');

-- 7. 查看知识版本历史
SELECT * FROM knowledge_versions
WHERE namespace = 'default' AND lu_id = 'LU_001'
ORDER BY version DESC;

-- 8. 获取待处理的同步任务
SELECT * FROM sync_queue
WHERE status = 'pending'
  AND scheduled_at <= datetime('now')
  AND retry_count < max_retries
ORDER BY scheduled_at
LIMIT 100;

-- 9. 查询信任层级分布
SELECT
    trust_tier,
    COUNT(*) as count,
    SUM(hit_count) as total_hits
FROM knowledge
WHERE namespace = 'default'
GROUP BY trust_tier;

-- 10. 查看文档分片统计
SELECT * FROM document_chunk_stats
WHERE namespace = 'default';

-- 11. 标记需要重编码的知识
UPDATE knowledge
SET encoder_version = NULL,
    metadata = json_set(
        COALESCE(metadata, '{}'),
        '$.pending_reencode',
        json_object(
            'old_version', 'v1.0',
            'new_version', 'v1.1',
            'marked_at', datetime('now')
        )
    )
WHERE encoder_version = 'v1.0';

-- 12. 查看待重编码知识
SELECT * FROM pending_reencode
WHERE namespace = 'default';
*/

-- ============================================================
-- 完成
-- ============================================================

SELECT 'aga-knowledge SQLite Schema v0.3.0 installed successfully' as message;
SELECT 'Tables: ' || group_concat(name, ', ') as tables
FROM sqlite_master
WHERE type = 'table' AND name NOT LIKE 'sqlite_%';
