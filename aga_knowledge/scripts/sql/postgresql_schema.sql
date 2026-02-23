-- ============================================================
-- aga-knowledge PostgreSQL Schema
-- Version: 0.3.0
--
-- aga-knowledge 知识管理系统数据库架构（明文 KV 版本）
--
-- 设计原则:
--   1. 只存储明文 condition/decision 文本对，不含向量数据
--   2. 向量化由 Encoder 在运行时处理，存储在内存索引中
--   3. 通过 BaseRetriever 协议与 aga-core 通讯
--   4. 命名空间隔离，支持多租户
--   5. 完整的审计日志和版本历史
--
-- 与原 AGA Schema 的区别:
--   - 移除 key_vector / value_vector 字段（向量不持久化）
--   - 移除 slot_idx（aga-knowledge 不管理 GPU 槽位）
--   - 新增 knowledge_versions 表（版本历史）
--   - 新增 image_assets 表（多模态图片资产）
--   - 新增 document_sources 表（源文档管理）
--   - 信任层级改为 system/verified/standard/experimental/untrusted
--
-- 使用方式:
--   createdb -U postgres aga_knowledge
--   psql -U postgres -d aga_knowledge -f postgresql_schema.sql
-- ============================================================

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 用于文本搜索

-- ============================================================
-- 1. 核心表
-- ============================================================

-- 命名空间表
CREATE TABLE IF NOT EXISTS namespaces (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

COMMENT ON TABLE namespaces IS '命名空间（租户/领域隔离）';

-- 知识表（核心表 — 明文 KV，无向量字段）
CREATE TABLE IF NOT EXISTS knowledge (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    -- 明文知识对
    condition_text TEXT,                -- 触发条件描述
    decision_text TEXT,                 -- 决策/知识内容
    -- 生命周期
    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'probationary',
    -- 信任层级
    trust_tier VARCHAR(50) NOT NULL DEFAULT 'standard',
    -- 统计
    hit_count BIGINT DEFAULT 0,
    consecutive_misses INTEGER DEFAULT 0,
    -- 版本
    version INTEGER DEFAULT 1,
    -- 编码器版本标记（投影层更新后触发重编码）
    encoder_version VARCHAR(100),
    -- 来源文档关联
    source_document_id VARCHAR(255),
    chunk_index INTEGER,
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP WITH TIME ZONE,
    -- 扩展元数据
    metadata JSONB DEFAULT '{}'::jsonb,
    -- 约束
    UNIQUE(namespace, lu_id)
);

COMMENT ON TABLE knowledge IS 'aga-knowledge 知识表（明文 condition/decision 对）';
COMMENT ON COLUMN knowledge.condition_text IS '触发条件描述（用于检索匹配）';
COMMENT ON COLUMN knowledge.decision_text IS '决策/知识内容（注入到 aga-core 的语义信息）';
COMMENT ON COLUMN knowledge.lifecycle_state IS '生命周期: probationary, confirmed, deprecated, quarantined';
COMMENT ON COLUMN knowledge.trust_tier IS '信任层级: system, verified, standard, experimental, untrusted';
COMMENT ON COLUMN knowledge.encoder_version IS '编码此知识时使用的编码器版本，投影层更新后需重编码';
COMMENT ON COLUMN knowledge.source_document_id IS '关联的源文档 ID（如果从文档分片生成）';

-- 知识版本历史表
CREATE TABLE IF NOT EXISTS knowledge_versions (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL,
    -- 版本快照
    condition_text TEXT,
    decision_text TEXT,
    lifecycle_state VARCHAR(50),
    trust_tier VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    -- 变更信息
    created_by VARCHAR(255) DEFAULT 'system',
    change_reason TEXT,
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- 约束
    UNIQUE(namespace, lu_id, version)
);

COMMENT ON TABLE knowledge_versions IS '知识版本历史（支持回滚和审计）';

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    old_state VARCHAR(50),
    new_state VARCHAR(50),
    reason TEXT,
    source VARCHAR(100) DEFAULT 'portal',
    operator VARCHAR(255),
    -- 详细信息
    details JSONB DEFAULT '{}'::jsonb,
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE audit_logs IS '操作审计日志';

-- ============================================================
-- 2. 文档管理表
-- ============================================================

-- 源文档表
CREATE TABLE IF NOT EXISTS document_sources (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    -- 文档信息
    title TEXT,
    source_type VARCHAR(50) DEFAULT 'markdown',  -- markdown, html, text, pdf
    original_filename TEXT,
    -- 分片信息
    total_chunks INTEGER DEFAULT 0,
    -- 分片配置快照
    chunker_strategy VARCHAR(50),
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    -- 状态
    status VARCHAR(50) DEFAULT 'active',  -- active, archived, deleted
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- 元数据
    metadata JSONB DEFAULT '{}'::jsonb,
    -- 约束
    UNIQUE(namespace, document_id)
);

COMMENT ON TABLE document_sources IS '源文档管理（知识分片的来源）';

-- 图片资产表
CREATE TABLE IF NOT EXISTS image_assets (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    asset_id VARCHAR(255) NOT NULL,
    -- 关联
    document_id VARCHAR(255),
    lu_id VARCHAR(255),
    -- 图片信息
    alt_text TEXT,
    original_src TEXT,
    portal_url TEXT NOT NULL,
    mime_type VARCHAR(100),
    file_size BIGINT DEFAULT 0,
    width INTEGER DEFAULT 0,
    height INTEGER DEFAULT 0,
    -- 存储路径
    storage_path TEXT,
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- 元数据
    metadata JSONB DEFAULT '{}'::jsonb,
    -- 约束
    UNIQUE(namespace, asset_id)
);

COMMENT ON TABLE image_assets IS '知识文档图片资产（与文档上下文对齐）';

-- ============================================================
-- 3. 同步与分布式表
-- ============================================================

-- 同步状态表
CREATE TABLE IF NOT EXISTS sync_states (
    id SERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL UNIQUE,
    last_sync_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sync_version BIGINT DEFAULT 0,
    dirty_knowledge TEXT DEFAULT '[]',
    -- 分布式锁
    lock_holder VARCHAR(255),
    lock_acquired_at TIMESTAMP WITH TIME ZONE,
    lock_expires_at TIMESTAMP WITH TIME ZONE
);

COMMENT ON TABLE sync_states IS '知识同步状态（Portal → Runtime 同步）';

-- 同步消息队列表
CREATE TABLE IF NOT EXISTS sync_queue (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,  -- inject, update, delete, lifecycle_change
    payload JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE sync_queue IS '知识同步消息队列';

-- ============================================================
-- 4. 编码器管理表
-- ============================================================

-- 编码器版本表
CREATE TABLE IF NOT EXISTS encoder_versions (
    id SERIAL PRIMARY KEY,
    version_tag VARCHAR(100) NOT NULL UNIQUE,
    -- 编码器信息
    backend VARCHAR(100) NOT NULL,           -- sentence_transformer, simple, custom
    model_name VARCHAR(255),
    -- 对齐参数快照
    key_dim INTEGER NOT NULL,
    value_dim INTEGER NOT NULL,
    key_norm_target REAL,
    value_norm_target REAL,
    -- 投影层信息
    projection_path TEXT,
    projection_hash VARCHAR(64),             -- 投影层权重的 SHA256
    -- 状态
    status VARCHAR(50) DEFAULT 'active',     -- active, deprecated, archived
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- 元数据
    metadata JSONB DEFAULT '{}'::jsonb
);

COMMENT ON TABLE encoder_versions IS '编码器版本管理（投影层更新后触发重编码）';

-- ============================================================
-- 5. 告警表
-- ============================================================

-- 告警记录表
CREATE TABLE IF NOT EXISTS alert_records (
    id BIGSERIAL PRIMARY KEY,
    alert_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,  -- info, warning, critical
    state VARCHAR(50) NOT NULL,     -- pending, firing, resolved
    summary TEXT,
    description TEXT,
    component VARCHAR(100),         -- encoder, retriever, portal, sync
    labels JSONB DEFAULT '{}'::jsonb,
    annotations JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE alert_records IS '系统告警记录';

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

-- condition 文本搜索索引（pg_trgm 三元组索引）
CREATE INDEX IF NOT EXISTS idx_knowledge_condition_trgm
    ON knowledge USING gin(condition_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_decision_trgm
    ON knowledge USING gin(decision_text gin_trgm_ops);

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
CREATE INDEX IF NOT EXISTS idx_alert_created ON alert_records(created_at DESC);

-- ============================================================
-- 触发器函数
-- ============================================================

-- 更新时间戳触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 自动更新触发器
DROP TRIGGER IF EXISTS update_namespaces_updated_at ON namespaces;
CREATE TRIGGER update_namespaces_updated_at
    BEFORE UPDATE ON namespaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_knowledge_updated_at ON knowledge;
CREATE TRIGGER update_knowledge_updated_at
    BEFORE UPDATE ON knowledge
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_documents_updated_at ON document_sources;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON document_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 知识变更审计触发器函数
CREATE OR REPLACE FUNCTION log_knowledge_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (namespace, lu_id, action, new_state, details)
        VALUES (NEW.namespace, NEW.lu_id, 'create', NEW.lifecycle_state,
                jsonb_build_object(
                    'condition', NEW.condition_text,
                    'decision', LEFT(NEW.decision_text, 200),
                    'trust_tier', NEW.trust_tier
                ));
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.lifecycle_state != NEW.lifecycle_state THEN
            INSERT INTO audit_logs (namespace, lu_id, action, old_state, new_state)
            VALUES (NEW.namespace, NEW.lu_id, 'lifecycle_change',
                    OLD.lifecycle_state, NEW.lifecycle_state);
        END IF;
        IF OLD.trust_tier != NEW.trust_tier THEN
            INSERT INTO audit_logs (namespace, lu_id, action, old_state, new_state,
                                    details)
            VALUES (NEW.namespace, NEW.lu_id, 'trust_tier_change',
                    OLD.trust_tier, NEW.trust_tier,
                    jsonb_build_object('field', 'trust_tier'));
        END IF;
        IF OLD.condition_text IS DISTINCT FROM NEW.condition_text
           OR OLD.decision_text IS DISTINCT FROM NEW.decision_text THEN
            INSERT INTO audit_logs (namespace, lu_id, action, details)
            VALUES (NEW.namespace, NEW.lu_id, 'content_update',
                    jsonb_build_object(
                        'old_version', OLD.version,
                        'new_version', NEW.version
                    ));
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (namespace, lu_id, action, old_state, details)
        VALUES (OLD.namespace, OLD.lu_id, 'delete', OLD.lifecycle_state,
                jsonb_build_object(
                    'condition', OLD.condition_text,
                    'decision', LEFT(OLD.decision_text, 200)
                ));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 知识变更审计触发器
DROP TRIGGER IF EXISTS audit_knowledge_changes ON knowledge;
CREATE TRIGGER audit_knowledge_changes
    AFTER INSERT OR UPDATE OR DELETE ON knowledge
    FOR EACH ROW
    EXECUTE FUNCTION log_knowledge_change();

-- 知识版本自动记录触发器函数
CREATE OR REPLACE FUNCTION auto_version_knowledge()
RETURNS TRIGGER AS $$
BEGIN
    -- 仅在内容变更时记录版本
    IF TG_OP = 'INSERT' OR
       OLD.condition_text IS DISTINCT FROM NEW.condition_text OR
       OLD.decision_text IS DISTINCT FROM NEW.decision_text OR
       OLD.lifecycle_state IS DISTINCT FROM NEW.lifecycle_state OR
       OLD.trust_tier IS DISTINCT FROM NEW.trust_tier THEN

        INSERT INTO knowledge_versions
            (namespace, lu_id, version, condition_text, decision_text,
             lifecycle_state, trust_tier, metadata, created_by)
        VALUES
            (NEW.namespace, NEW.lu_id, NEW.version,
             NEW.condition_text, NEW.decision_text,
             NEW.lifecycle_state, NEW.trust_tier,
             NEW.metadata, COALESCE(current_setting('app.current_user', true), 'system'));

        -- 清理旧版本（保留最近 10 个）
        DELETE FROM knowledge_versions
        WHERE namespace = NEW.namespace
          AND lu_id = NEW.lu_id
          AND version NOT IN (
              SELECT version FROM knowledge_versions
              WHERE namespace = NEW.namespace AND lu_id = NEW.lu_id
              ORDER BY version DESC
              LIMIT 10
          );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 知识版本自动记录触发器
DROP TRIGGER IF EXISTS auto_version_on_knowledge_change ON knowledge;
CREATE TRIGGER auto_version_on_knowledge_change
    AFTER INSERT OR UPDATE ON knowledge
    FOR EACH ROW
    EXECUTE FUNCTION auto_version_knowledge();

-- ============================================================
-- 存储过程
-- ============================================================

-- 增加命中计数
CREATE OR REPLACE FUNCTION increment_hit_count(
    p_namespace VARCHAR(255),
    p_lu_ids VARCHAR(255)[]
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE knowledge
    SET hit_count = hit_count + 1,
        last_hit_at = CURRENT_TIMESTAMP,
        consecutive_misses = 0
    WHERE namespace = p_namespace
      AND lu_id = ANY(p_lu_ids);

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- 批量隔离
CREATE OR REPLACE FUNCTION batch_quarantine(
    p_namespace VARCHAR(255),
    p_lu_ids VARCHAR(255)[],
    p_reason TEXT DEFAULT 'batch quarantine'
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE knowledge
    SET lifecycle_state = 'quarantined'
    WHERE namespace = p_namespace
      AND lu_id = ANY(p_lu_ids)
      AND lifecycle_state != 'quarantined';

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    -- 记录审计
    INSERT INTO audit_logs (namespace, action, reason, details)
    VALUES (p_namespace, 'batch_quarantine', p_reason,
            jsonb_build_object('lu_ids', to_jsonb(p_lu_ids), 'count', updated_count));

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- 获取命名空间统计
CREATE OR REPLACE FUNCTION get_namespace_statistics(p_namespace VARCHAR(255))
RETURNS TABLE (
    total_knowledge BIGINT,
    active_knowledge BIGINT,
    probationary_count BIGINT,
    confirmed_count BIGINT,
    deprecated_count BIGINT,
    quarantined_count BIGINT,
    total_hits BIGINT,
    total_documents BIGINT,
    total_images BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*) FROM knowledge WHERE namespace = p_namespace)::BIGINT,
        (SELECT COUNT(*) FROM knowledge WHERE namespace = p_namespace
            AND lifecycle_state != 'quarantined')::BIGINT,
        (SELECT COUNT(*) FROM knowledge WHERE namespace = p_namespace
            AND lifecycle_state = 'probationary')::BIGINT,
        (SELECT COUNT(*) FROM knowledge WHERE namespace = p_namespace
            AND lifecycle_state = 'confirmed')::BIGINT,
        (SELECT COUNT(*) FROM knowledge WHERE namespace = p_namespace
            AND lifecycle_state = 'deprecated')::BIGINT,
        (SELECT COUNT(*) FROM knowledge WHERE namespace = p_namespace
            AND lifecycle_state = 'quarantined')::BIGINT,
        (SELECT COALESCE(SUM(hit_count), 0) FROM knowledge
            WHERE namespace = p_namespace)::BIGINT,
        (SELECT COUNT(*) FROM document_sources WHERE namespace = p_namespace
            AND status = 'active')::BIGINT,
        (SELECT COUNT(*) FROM image_assets WHERE namespace = p_namespace)::BIGINT;
END;
$$ LANGUAGE plpgsql;

-- 回滚知识到指定版本
CREATE OR REPLACE FUNCTION rollback_knowledge(
    p_namespace VARCHAR(255),
    p_lu_id VARCHAR(255),
    p_target_version INTEGER,
    p_rolled_back_by VARCHAR(255) DEFAULT 'system'
)
RETURNS BOOLEAN AS $$
DECLARE
    v_record RECORD;
    v_current_version INTEGER;
BEGIN
    -- 获取目标版本
    SELECT * INTO v_record
    FROM knowledge_versions
    WHERE namespace = p_namespace
      AND lu_id = p_lu_id
      AND version = p_target_version;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- 获取当前版本号
    SELECT version INTO v_current_version
    FROM knowledge
    WHERE namespace = p_namespace AND lu_id = p_lu_id;

    -- 设置当前操作者（供版本触发器使用）
    PERFORM set_config('app.current_user', p_rolled_back_by, true);

    -- 更新知识
    UPDATE knowledge
    SET condition_text = v_record.condition_text,
        decision_text = v_record.decision_text,
        lifecycle_state = v_record.lifecycle_state,
        trust_tier = v_record.trust_tier,
        version = v_current_version + 1,
        metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{rollback_info}',
            jsonb_build_object(
                'from_version', v_current_version,
                'to_version', p_target_version,
                'rolled_back_by', p_rolled_back_by
            )
        )
    WHERE namespace = p_namespace AND lu_id = p_lu_id;

    -- 记录审计
    INSERT INTO audit_logs (namespace, lu_id, action, reason, operator, details)
    VALUES (p_namespace, p_lu_id, 'rollback', 
            'Rollback to version ' || p_target_version,
            p_rolled_back_by,
            jsonb_build_object(
                'from_version', v_current_version,
                'to_version', p_target_version
            ));

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- 处理同步队列
CREATE OR REPLACE FUNCTION process_sync_queue(p_limit INTEGER DEFAULT 100)
RETURNS TABLE (
    id BIGINT,
    namespace VARCHAR(255),
    lu_id VARCHAR(255),
    action VARCHAR(50),
    payload JSONB
) AS $$
BEGIN
    RETURN QUERY
    UPDATE sync_queue sq
    SET status = 'processing',
        started_at = CURRENT_TIMESTAMP
    FROM (
        SELECT sq2.id
        FROM sync_queue sq2
        WHERE sq2.status = 'pending'
          AND sq2.scheduled_at <= CURRENT_TIMESTAMP
          AND sq2.retry_count < sq2.max_retries
        ORDER BY sq2.scheduled_at
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    ) sub
    WHERE sq.id = sub.id
    RETURNING sq.id, sq.namespace, sq.lu_id, sq.action, sq.payload;
END;
$$ LANGUAGE plpgsql;

-- 标记需要重编码的知识
CREATE OR REPLACE FUNCTION mark_for_reencoding(
    p_old_encoder_version VARCHAR(100),
    p_new_encoder_version VARCHAR(100)
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE knowledge
    SET encoder_version = NULL,
        metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{pending_reencode}',
            jsonb_build_object(
                'old_version', p_old_encoder_version,
                'new_version', p_new_encoder_version,
                'marked_at', CURRENT_TIMESTAMP
            )
        )
    WHERE encoder_version = p_old_encoder_version;

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    INSERT INTO audit_logs (namespace, action, details)
    VALUES ('_system', 'mark_reencoding',
            jsonb_build_object(
                'old_version', p_old_encoder_version,
                'new_version', p_new_encoder_version,
                'affected_count', updated_count
            ));

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- 记录告警
CREATE OR REPLACE FUNCTION record_alert(
    p_alert_name VARCHAR(255),
    p_severity VARCHAR(50),
    p_state VARCHAR(50),
    p_summary TEXT DEFAULT NULL,
    p_description TEXT DEFAULT NULL,
    p_component VARCHAR(100) DEFAULT NULL,
    p_labels JSONB DEFAULT NULL,
    p_annotations JSONB DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    alert_id BIGINT;
BEGIN
    INSERT INTO alert_records (
        alert_name, severity, state, summary, description,
        component, labels, annotations
    )
    VALUES (
        p_alert_name, p_severity, p_state, p_summary, p_description,
        p_component, COALESCE(p_labels, '{}'::jsonb),
        COALESCE(p_annotations, '{}'::jsonb)
    )
    RETURNING id INTO alert_id;

    RETURN alert_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 视图
-- ============================================================

-- 活跃知识视图
CREATE OR REPLACE VIEW active_knowledge AS
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

COMMENT ON VIEW active_knowledge IS '活跃知识视图（非隔离状态，供 Retriever 加载）';

-- 命名空间统计视图
CREATE OR REPLACE VIEW namespace_stats AS
SELECT
    namespace,
    COUNT(*) as total_knowledge,
    COUNT(*) FILTER (WHERE lifecycle_state != 'quarantined') as active_knowledge,
    COUNT(*) FILTER (WHERE lifecycle_state = 'probationary') as probationary_count,
    COUNT(*) FILTER (WHERE lifecycle_state = 'confirmed') as confirmed_count,
    COUNT(*) FILTER (WHERE lifecycle_state = 'deprecated') as deprecated_count,
    COUNT(*) FILTER (WHERE lifecycle_state = 'quarantined') as quarantined_count,
    COALESCE(SUM(hit_count), 0) as total_hits,
    COUNT(*) FILTER (WHERE encoder_version IS NULL) as pending_reencode
FROM knowledge
GROUP BY namespace;

COMMENT ON VIEW namespace_stats IS '命名空间统计视图';

-- 近期审计视图
CREATE OR REPLACE VIEW recent_audit AS
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
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY created_at DESC;

COMMENT ON VIEW recent_audit IS '近 7 天审计日志视图';

-- 文档分片统计视图
CREATE OR REPLACE VIEW document_chunk_stats AS
SELECT
    ds.namespace,
    ds.document_id,
    ds.title,
    ds.total_chunks,
    COUNT(k.id) as actual_chunks,
    COUNT(k.id) FILTER (WHERE k.lifecycle_state = 'confirmed') as confirmed_chunks,
    COALESCE(SUM(k.hit_count), 0) as total_hits,
    COUNT(ia.id) as image_count
FROM document_sources ds
LEFT JOIN knowledge k ON k.namespace = ds.namespace
    AND k.source_document_id = ds.document_id
LEFT JOIN image_assets ia ON ia.namespace = ds.namespace
    AND ia.document_id = ds.document_id
WHERE ds.status = 'active'
GROUP BY ds.namespace, ds.document_id, ds.title, ds.total_chunks;

COMMENT ON VIEW document_chunk_stats IS '文档分片统计视图';

-- 活跃告警视图
CREATE OR REPLACE VIEW active_alerts AS
SELECT
    id,
    alert_name,
    severity,
    summary,
    description,
    component,
    started_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)) as duration_seconds
FROM alert_records
WHERE state = 'firing'
ORDER BY
    CASE severity
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        ELSE 3
    END,
    started_at DESC;

COMMENT ON VIEW active_alerts IS '活跃告警视图';

-- 待重编码知识视图
CREATE OR REPLACE VIEW pending_reencode AS
SELECT
    namespace,
    lu_id,
    condition_text,
    decision_text,
    encoder_version,
    metadata->'pending_reencode' as reencode_info,
    updated_at
FROM knowledge
WHERE encoder_version IS NULL
ORDER BY updated_at;

COMMENT ON VIEW pending_reencode IS '待重编码知识视图（编码器版本更新后）';

-- ============================================================
-- 初始数据
-- ============================================================

-- 创建默认命名空间
INSERT INTO namespaces (name, description)
VALUES ('default', 'Default namespace')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- 权限
-- ============================================================

-- 创建只读角色
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'aga_knowledge_readonly') THEN
        CREATE ROLE aga_knowledge_readonly;
    END IF;
END
$$;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO aga_knowledge_readonly;
GRANT USAGE ON SCHEMA public TO aga_knowledge_readonly;

-- 创建读写角色（Portal API 使用）
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'aga_knowledge_readwrite') THEN
        CREATE ROLE aga_knowledge_readwrite;
    END IF;
END
$$;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO aga_knowledge_readwrite;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO aga_knowledge_readwrite;
GRANT USAGE ON SCHEMA public TO aga_knowledge_readwrite;

-- ============================================================
-- 完成
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE 'aga-knowledge PostgreSQL Schema v0.3.0 installed successfully';
    RAISE NOTICE 'Tables: namespaces, knowledge, knowledge_versions, audit_logs, document_sources, image_assets, sync_states, sync_queue, encoder_versions, alert_records';
    RAISE NOTICE 'Views: active_knowledge, namespace_stats, recent_audit, document_chunk_stats, active_alerts, pending_reencode';
END
$$;
