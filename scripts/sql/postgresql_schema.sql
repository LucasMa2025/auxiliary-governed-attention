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

-- 向量时钟表（用于因果一致性）
CREATE TABLE IF NOT EXISTS vector_clocks (
    id SERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    clocks JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace, lu_id)
);

COMMENT ON TABLE vector_clocks IS '向量时钟（因果一致性）';

-- 分区事件表
CREATE TABLE IF NOT EXISTS partition_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,  -- detected, suspected, healed
    affected_instances JSONB DEFAULT '[]'::jsonb,
    healthy_ratio DECIMAL(5, 4),
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE partition_events IS '网络分区事件记录';

-- 冲突记录表
CREATE TABLE IF NOT EXISTS conflict_records (
    id BIGSERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    lu_id VARCHAR(255) NOT NULL,
    conflict_type VARCHAR(50) NOT NULL,  -- concurrent_write, version_mismatch
    versions JSONB NOT NULL,  -- 冲突的版本列表
    resolution VARCHAR(50),  -- lww, fww, merge, manual
    resolved_value JSONB,
    resolved_by VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',  -- pending, resolved, escalated
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

COMMENT ON TABLE conflict_records IS '数据冲突记录';

-- 实例心跳表
CREATE TABLE IF NOT EXISTS instance_heartbeats (
    id SERIAL PRIMARY KEY,
    instance_id VARCHAR(255) NOT NULL UNIQUE,
    namespace VARCHAR(255) NOT NULL,
    host VARCHAR(255),
    port INTEGER,
    status VARCHAR(50) DEFAULT 'unknown',
    capabilities JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    vector_clock JSONB DEFAULT '{}'::jsonb,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE instance_heartbeats IS '实例心跳记录';

-- 告警记录表
CREATE TABLE IF NOT EXISTS alert_records (
    id BIGSERIAL PRIMARY KEY,
    alert_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,  -- info, warning, critical, page
    state VARCHAR(50) NOT NULL,  -- pending, firing, resolved
    summary TEXT,
    description TEXT,
    instance_id VARCHAR(255),
    labels JSONB DEFAULT '{}'::jsonb,
    annotations JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE alert_records IS '告警记录';

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

-- vector_clocks 索引
CREATE INDEX IF NOT EXISTS idx_vector_clocks_namespace ON vector_clocks(namespace);
CREATE INDEX IF NOT EXISTS idx_vector_clocks_updated ON vector_clocks(last_updated DESC);

-- partition_events 索引
CREATE INDEX IF NOT EXISTS idx_partition_events_type ON partition_events(event_type);
CREATE INDEX IF NOT EXISTS idx_partition_events_created ON partition_events(created_at DESC);

-- conflict_records 索引
CREATE INDEX IF NOT EXISTS idx_conflict_namespace ON conflict_records(namespace, lu_id);
CREATE INDEX IF NOT EXISTS idx_conflict_status ON conflict_records(status);
CREATE INDEX IF NOT EXISTS idx_conflict_created ON conflict_records(created_at DESC);

-- instance_heartbeats 索引
CREATE INDEX IF NOT EXISTS idx_heartbeat_namespace ON instance_heartbeats(namespace);
CREATE INDEX IF NOT EXISTS idx_heartbeat_status ON instance_heartbeats(status);
CREATE INDEX IF NOT EXISTS idx_heartbeat_last ON instance_heartbeats(last_heartbeat DESC);

-- alert_records 索引
CREATE INDEX IF NOT EXISTS idx_alert_name ON alert_records(alert_name);
CREATE INDEX IF NOT EXISTS idx_alert_severity ON alert_records(severity);
CREATE INDEX IF NOT EXISTS idx_alert_state ON alert_records(state);
CREATE INDEX IF NOT EXISTS idx_alert_created ON alert_records(created_at DESC);

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

-- ============================================================
-- v3.4.1 新增函数
-- ============================================================

-- 更新向量时钟函数
CREATE OR REPLACE FUNCTION update_vector_clock(
    p_namespace VARCHAR(255),
    p_lu_id VARCHAR(255),
    p_instance_id VARCHAR(255),
    p_remote_clocks JSONB DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    current_clocks JSONB;
    new_clocks JSONB;
    key TEXT;
    local_val INTEGER;
    remote_val INTEGER;
BEGIN
    -- 获取当前时钟
    SELECT clocks INTO current_clocks
    FROM vector_clocks
    WHERE namespace = p_namespace AND lu_id = p_lu_id;
    
    IF current_clocks IS NULL THEN
        current_clocks := '{}'::jsonb;
    END IF;
    
    -- 递增本实例时钟
    local_val := COALESCE((current_clocks->>p_instance_id)::INTEGER, 0) + 1;
    current_clocks := jsonb_set(current_clocks, ARRAY[p_instance_id], to_jsonb(local_val));
    
    -- 合并远程时钟
    IF p_remote_clocks IS NOT NULL THEN
        FOR key IN SELECT jsonb_object_keys(p_remote_clocks)
        LOOP
            remote_val := (p_remote_clocks->>key)::INTEGER;
            local_val := COALESCE((current_clocks->>key)::INTEGER, 0);
            IF remote_val > local_val THEN
                current_clocks := jsonb_set(current_clocks, ARRAY[key], to_jsonb(remote_val));
            END IF;
        END LOOP;
    END IF;
    
    -- 更新或插入
    INSERT INTO vector_clocks (namespace, lu_id, clocks, last_updated)
    VALUES (p_namespace, p_lu_id, current_clocks, CURRENT_TIMESTAMP)
    ON CONFLICT (namespace, lu_id) 
    DO UPDATE SET clocks = EXCLUDED.clocks, last_updated = EXCLUDED.last_updated;
    
    RETURN current_clocks;
END;
$$ LANGUAGE plpgsql;

-- 记录分区事件函数
CREATE OR REPLACE FUNCTION record_partition_event(
    p_event_type VARCHAR(50),
    p_affected_instances JSONB,
    p_healthy_ratio DECIMAL DEFAULT NULL,
    p_details JSONB DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    event_id BIGINT;
BEGIN
    INSERT INTO partition_events (event_type, affected_instances, healthy_ratio, details)
    VALUES (p_event_type, p_affected_instances, p_healthy_ratio, COALESCE(p_details, '{}'::jsonb))
    RETURNING id INTO event_id;
    
    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

-- 记录冲突函数
CREATE OR REPLACE FUNCTION record_conflict(
    p_namespace VARCHAR(255),
    p_lu_id VARCHAR(255),
    p_conflict_type VARCHAR(50),
    p_versions JSONB
)
RETURNS BIGINT AS $$
DECLARE
    conflict_id BIGINT;
BEGIN
    INSERT INTO conflict_records (namespace, lu_id, conflict_type, versions)
    VALUES (p_namespace, p_lu_id, p_conflict_type, p_versions)
    RETURNING id INTO conflict_id;
    
    RETURN conflict_id;
END;
$$ LANGUAGE plpgsql;

-- 解决冲突函数
CREATE OR REPLACE FUNCTION resolve_conflict(
    p_conflict_id BIGINT,
    p_resolution VARCHAR(50),
    p_resolved_value JSONB,
    p_resolved_by VARCHAR(255)
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE conflict_records
    SET resolution = p_resolution,
        resolved_value = p_resolved_value,
        resolved_by = p_resolved_by,
        status = 'resolved',
        resolved_at = CURRENT_TIMESTAMP
    WHERE id = p_conflict_id AND status = 'pending';
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- 更新实例心跳函数
CREATE OR REPLACE FUNCTION update_instance_heartbeat(
    p_instance_id VARCHAR(255),
    p_namespace VARCHAR(255),
    p_host VARCHAR(255) DEFAULT NULL,
    p_port INTEGER DEFAULT NULL,
    p_status VARCHAR(50) DEFAULT 'healthy',
    p_capabilities JSONB DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL,
    p_vector_clock JSONB DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO instance_heartbeats (
        instance_id, namespace, host, port, status, 
        capabilities, metadata, vector_clock, last_heartbeat
    )
    VALUES (
        p_instance_id, p_namespace, p_host, p_port, p_status,
        COALESCE(p_capabilities, '[]'::jsonb),
        COALESCE(p_metadata, '{}'::jsonb),
        COALESCE(p_vector_clock, '{}'::jsonb),
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (instance_id) 
    DO UPDATE SET 
        status = EXCLUDED.status,
        capabilities = COALESCE(EXCLUDED.capabilities, instance_heartbeats.capabilities),
        metadata = COALESCE(EXCLUDED.metadata, instance_heartbeats.metadata),
        vector_clock = COALESCE(EXCLUDED.vector_clock, instance_heartbeats.vector_clock),
        last_heartbeat = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- 检测分区实例函数
CREATE OR REPLACE FUNCTION detect_partitioned_instances(
    p_timeout_seconds INTEGER DEFAULT 30
)
RETURNS TABLE (
    instance_id VARCHAR(255),
    namespace VARCHAR(255),
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    seconds_since_heartbeat DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ih.instance_id,
        ih.namespace,
        ih.last_heartbeat,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ih.last_heartbeat)) as seconds_since_heartbeat
    FROM instance_heartbeats ih
    WHERE ih.last_heartbeat < CURRENT_TIMESTAMP - (p_timeout_seconds || ' seconds')::INTERVAL
      AND ih.status != 'partitioned';
END;
$$ LANGUAGE plpgsql;

-- 记录告警函数
CREATE OR REPLACE FUNCTION record_alert(
    p_alert_name VARCHAR(255),
    p_severity VARCHAR(50),
    p_state VARCHAR(50),
    p_summary TEXT DEFAULT NULL,
    p_description TEXT DEFAULT NULL,
    p_instance_id VARCHAR(255) DEFAULT NULL,
    p_labels JSONB DEFAULT NULL,
    p_annotations JSONB DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    alert_id BIGINT;
BEGIN
    INSERT INTO alert_records (
        alert_name, severity, state, summary, description,
        instance_id, labels, annotations
    )
    VALUES (
        p_alert_name, p_severity, p_state, p_summary, p_description,
        p_instance_id, COALESCE(p_labels, '{}'::jsonb), COALESCE(p_annotations, '{}'::jsonb)
    )
    RETURNING id INTO alert_id;
    
    RETURN alert_id;
END;
$$ LANGUAGE plpgsql;

-- 获取活跃告警函数
CREATE OR REPLACE FUNCTION get_active_alerts(
    p_severity VARCHAR(50) DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    alert_name VARCHAR(255),
    severity VARCHAR(50),
    summary TEXT,
    instance_id VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ar.id,
        ar.alert_name,
        ar.severity,
        ar.summary,
        ar.instance_id,
        ar.started_at
    FROM alert_records ar
    WHERE ar.state = 'firing'
      AND (p_severity IS NULL OR ar.severity = p_severity)
    ORDER BY ar.started_at DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- v3.4.1 新增视图
-- ============================================================

-- 实例健康状态视图
CREATE OR REPLACE VIEW instance_health AS
SELECT 
    instance_id,
    namespace,
    host,
    port,
    status,
    last_heartbeat,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_heartbeat)) as seconds_since_heartbeat,
    CASE 
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_heartbeat)) < 15 THEN 'healthy'
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_heartbeat)) < 30 THEN 'suspected'
        ELSE 'partitioned'
    END as computed_status
FROM instance_heartbeats;

COMMENT ON VIEW instance_health IS '实例健康状态视图';

-- 待解决冲突视图
CREATE OR REPLACE VIEW pending_conflicts AS
SELECT 
    id,
    namespace,
    lu_id,
    conflict_type,
    versions,
    created_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) as age_seconds
FROM conflict_records
WHERE status = 'pending'
ORDER BY created_at;

COMMENT ON VIEW pending_conflicts IS '待解决冲突视图';

-- 活跃告警视图
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    id,
    alert_name,
    severity,
    summary,
    description,
    instance_id,
    started_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)) as duration_seconds
FROM alert_records
WHERE state = 'firing'
ORDER BY 
    CASE severity 
        WHEN 'page' THEN 1 
        WHEN 'critical' THEN 2 
        WHEN 'warning' THEN 3 
        ELSE 4 
    END,
    started_at DESC;

COMMENT ON VIEW active_alerts IS '活跃告警视图';

-- 输出版本信息
DO $$
BEGIN
    RAISE NOTICE 'AGA PostgreSQL Schema v3.4.1 installed successfully';
    RAISE NOTICE 'Tables: namespaces, aga_configs, knowledge_slots, audit_logs, sync_states, governance_votes, propagation_queue, vector_clocks, partition_events, conflict_records, instance_heartbeats, alert_records';
    RAISE NOTICE 'Views: active_knowledge, namespace_stats, recent_audit, instance_health, pending_conflicts, active_alerts';
END
$$;
