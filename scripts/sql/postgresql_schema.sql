-- ============================================================
-- AGA (Auxiliary Governed Attention) PostgreSQL Schema
-- Version: 3.1
-- 
-- 生产环境数据库架构
-- 
-- 使用方式:
--   psql -U postgres -d aga -f postgresql_schema.sql
-- ============================================================

-- 创建扩展（如果需要）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 用于文本搜索

-- ============================================================
-- 表定义
-- ============================================================

-- 命名空间表
CREATE TABLE IF NOT EXISTS namespaces (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    max_slots INTEGER DEFAULT 100,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

COMMENT ON TABLE namespaces IS 'AGA 命名空间（租户隔离）';

-- AGA 配置表
CREATE TABLE IF NOT EXISTS aga_configs (
    id SERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace)
);

COMMENT ON TABLE aga_configs IS 'AGA 实例配置';

-- 知识槽位表（核心表）
CREATE TABLE IF NOT EXISTS knowledge_slots (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    slot_idx INTEGER NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    condition TEXT,
    decision TEXT,
    -- 向量存储（JSON 格式，生产环境可考虑 pgvector）
    key_vector JSONB NOT NULL,
    value_vector JSONB NOT NULL,
    key_vector_dim INTEGER NOT NULL,
    value_vector_dim INTEGER NOT NULL,
    -- 生命周期
    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'probationary',
    -- 信任层级
    trust_tier VARCHAR(50) DEFAULT 's1_experience',
    -- 统计
    reliability DECIMAL(5, 4) DEFAULT 0.3,
    hit_count BIGINT DEFAULT 0,
    miss_count BIGINT DEFAULT 0,
    consecutive_misses INTEGER DEFAULT 0,
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP WITH TIME ZONE,
    -- 扩展元数据
    metadata JSONB DEFAULT '{}'::jsonb,
    -- 约束
    UNIQUE(namespace, lu_id),
    UNIQUE(namespace, slot_idx)
);

COMMENT ON TABLE knowledge_slots IS 'AGA 知识槽位';
COMMENT ON COLUMN knowledge_slots.lifecycle_state IS '生命周期状态: probationary, confirmed, deprecated, quarantined';
COMMENT ON COLUMN knowledge_slots.trust_tier IS '信任层级: s0_acceleration, s1_experience, s2_policy, s3_immutable';

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255),
    slot_idx INTEGER,
    action VARCHAR(100) NOT NULL,
    old_state VARCHAR(50),
    new_state VARCHAR(50),
    reason TEXT,
    source_instance VARCHAR(255),
    operator VARCHAR(255),
    -- 详细信息
    details JSONB DEFAULT '{}'::jsonb,
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE audit_logs IS '操作审计日志';

-- 同步状态表（用于分布式）
CREATE TABLE IF NOT EXISTS sync_states (
    id SERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL UNIQUE,
    last_sync_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sync_version BIGINT DEFAULT 0,
    dirty_slots JSONB DEFAULT '[]'::jsonb,
    -- 分布式锁
    lock_holder VARCHAR(255),
    lock_acquired_at TIMESTAMP WITH TIME ZONE,
    lock_expires_at TIMESTAMP WITH TIME ZONE
);

COMMENT ON TABLE sync_states IS '分布式同步状态';

-- 治理投票表（用于 quorum 机制）
CREATE TABLE IF NOT EXISTS governance_votes (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    vote_type VARCHAR(50) NOT NULL,  -- quarantine, approve, reject
    source_instance VARCHAR(255) NOT NULL,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    -- 唯一约束：每个实例对每个 LU 只能投一票
    UNIQUE(namespace, lu_id, vote_type, source_instance)
);

COMMENT ON TABLE governance_votes IS '治理投票记录（quorum 机制）';

-- 传播队列表
CREATE TABLE IF NOT EXISTS propagation_queue (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,  -- inject, update, quarantine
    target_instances JSONB DEFAULT '[]'::jsonb,
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

COMMENT ON TABLE propagation_queue IS '知识传播队列';

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

-- 条件文本搜索索引
CREATE INDEX IF NOT EXISTS idx_knowledge_condition_trgm 
    ON knowledge_slots USING gin(condition gin_trgm_ops);

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
-- 函数
-- ============================================================

-- 更新时间戳触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 自动更新触发器
DROP TRIGGER IF EXISTS update_namespaces_updated_at ON namespaces;
CREATE TRIGGER update_namespaces_updated_at
    BEFORE UPDATE ON namespaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_aga_configs_updated_at ON aga_configs;
CREATE TRIGGER update_aga_configs_updated_at
    BEFORE UPDATE ON aga_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_knowledge_slots_updated_at ON knowledge_slots;
CREATE TRIGGER update_knowledge_slots_updated_at
    BEFORE UPDATE ON knowledge_slots
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 记录审计日志函数
CREATE OR REPLACE FUNCTION log_knowledge_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (namespace, lu_id, slot_idx, action, new_state, details)
        VALUES (NEW.namespace, NEW.lu_id, NEW.slot_idx, 'create', NEW.lifecycle_state, 
                jsonb_build_object('condition', NEW.condition, 'decision', NEW.decision));
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.lifecycle_state != NEW.lifecycle_state THEN
            INSERT INTO audit_logs (namespace, lu_id, slot_idx, action, old_state, new_state)
            VALUES (NEW.namespace, NEW.lu_id, NEW.slot_idx, 'lifecycle_change', 
                    OLD.lifecycle_state, NEW.lifecycle_state);
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (namespace, lu_id, slot_idx, action, old_state)
        VALUES (OLD.namespace, OLD.lu_id, OLD.slot_idx, 'delete', OLD.lifecycle_state);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 知识变更审计触发器
DROP TRIGGER IF EXISTS audit_knowledge_changes ON knowledge_slots;
CREATE TRIGGER audit_knowledge_changes
    AFTER INSERT OR UPDATE OR DELETE ON knowledge_slots
    FOR EACH ROW
    EXECUTE FUNCTION log_knowledge_change();

-- 增加命中计数函数
CREATE OR REPLACE FUNCTION increment_hit_count(
    p_namespace VARCHAR(255),
    p_lu_ids VARCHAR(255)[]
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE knowledge_slots
    SET hit_count = hit_count + 1,
        last_hit_at = CURRENT_TIMESTAMP,
        consecutive_misses = 0
    WHERE namespace = p_namespace
      AND lu_id = ANY(p_lu_ids);
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- 批量隔离函数
CREATE OR REPLACE FUNCTION batch_quarantine(
    p_namespace VARCHAR(255),
    p_lu_ids VARCHAR(255)[],
    p_reason TEXT DEFAULT 'batch quarantine'
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE knowledge_slots
    SET lifecycle_state = 'quarantined',
        reliability = 0.0
    WHERE namespace = p_namespace
      AND lu_id = ANY(p_lu_ids)
      AND lifecycle_state != 'quarantined';
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    
    -- 记录审计
    INSERT INTO audit_logs (namespace, action, reason, details)
    VALUES (p_namespace, 'batch_quarantine', p_reason, 
            jsonb_build_object('lu_ids', p_lu_ids, 'count', updated_count));
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- 获取命名空间统计函数
CREATE OR REPLACE FUNCTION get_namespace_statistics(p_namespace VARCHAR(255))
RETURNS TABLE (
    total_slots BIGINT,
    active_slots BIGINT,
    probationary_count BIGINT,
    confirmed_count BIGINT,
    deprecated_count BIGINT,
    quarantined_count BIGINT,
    total_hits BIGINT,
    avg_reliability DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_slots,
        COUNT(*) FILTER (WHERE lifecycle_state != 'quarantined')::BIGINT as active_slots,
        COUNT(*) FILTER (WHERE lifecycle_state = 'probationary')::BIGINT as probationary_count,
        COUNT(*) FILTER (WHERE lifecycle_state = 'confirmed')::BIGINT as confirmed_count,
        COUNT(*) FILTER (WHERE lifecycle_state = 'deprecated')::BIGINT as deprecated_count,
        COUNT(*) FILTER (WHERE lifecycle_state = 'quarantined')::BIGINT as quarantined_count,
        COALESCE(SUM(hit_count), 0)::BIGINT as total_hits,
        COALESCE(AVG(reliability), 0)::DECIMAL as avg_reliability
    FROM knowledge_slots
    WHERE namespace = p_namespace;
END;
$$ LANGUAGE plpgsql;

-- 清理过期投票函数
CREATE OR REPLACE FUNCTION cleanup_expired_votes()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM governance_votes
    WHERE expires_at < CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 处理传播队列函数
CREATE OR REPLACE FUNCTION process_propagation_queue(p_limit INTEGER DEFAULT 100)
RETURNS TABLE (
    id BIGINT,
    namespace VARCHAR(255),
    lu_id VARCHAR(255),
    action VARCHAR(50),
    payload JSONB
) AS $$
BEGIN
    RETURN QUERY
    UPDATE propagation_queue pq
    SET status = 'processing',
        started_at = CURRENT_TIMESTAMP
    FROM (
        SELECT pq2.id
        FROM propagation_queue pq2
        WHERE pq2.status = 'pending'
          AND pq2.scheduled_at <= CURRENT_TIMESTAMP
          AND pq2.retry_count < pq2.max_retries
        ORDER BY pq2.scheduled_at
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    ) sub
    WHERE pq.id = sub.id
    RETURNING pq.id, pq.namespace, pq.lu_id, pq.action, pq.payload;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 视图
-- ============================================================

-- 活跃知识视图
CREATE OR REPLACE VIEW active_knowledge AS
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

COMMENT ON VIEW active_knowledge IS '活跃知识视图（非隔离状态）';

-- 命名空间统计视图
CREATE OR REPLACE VIEW namespace_stats AS
SELECT 
    namespace,
    COUNT(*) as total_slots,
    COUNT(*) FILTER (WHERE lifecycle_state != 'quarantined') as active_slots,
    COUNT(*) FILTER (WHERE lifecycle_state = 'probationary') as probationary_count,
    COUNT(*) FILTER (WHERE lifecycle_state = 'confirmed') as confirmed_count,
    COUNT(*) FILTER (WHERE lifecycle_state = 'deprecated') as deprecated_count,
    COUNT(*) FILTER (WHERE lifecycle_state = 'quarantined') as quarantined_count,
    COALESCE(SUM(hit_count), 0) as total_hits,
    COALESCE(AVG(reliability), 0) as avg_reliability
FROM knowledge_slots
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
    source_instance,
    created_at
FROM audit_logs
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY created_at DESC;

COMMENT ON VIEW recent_audit IS '近7天审计日志视图';

-- ============================================================
-- 初始数据
-- ============================================================

-- 创建默认命名空间
INSERT INTO namespaces (name, description, max_slots)
VALUES ('default', 'Default namespace', 100)
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- 权限（可根据需要调整）
-- ============================================================

-- 创建只读角色
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'aga_readonly') THEN
        CREATE ROLE aga_readonly;
    END IF;
END
$$;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO aga_readonly;
GRANT USAGE ON SCHEMA public TO aga_readonly;

-- 创建读写角色
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'aga_readwrite') THEN
        CREATE ROLE aga_readwrite;
    END IF;
END
$$;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO aga_readwrite;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO aga_readwrite;
GRANT USAGE ON SCHEMA public TO aga_readwrite;

-- ============================================================
-- 完成
-- ============================================================

-- 输出版本信息
DO $$
BEGIN
    RAISE NOTICE 'AGA PostgreSQL Schema v3.1 installed successfully';
    RAISE NOTICE 'Tables: namespaces, aga_configs, knowledge_slots, audit_logs, sync_states, governance_votes, propagation_queue';
    RAISE NOTICE 'Views: active_knowledge, namespace_stats, recent_audit';
END
$$;
