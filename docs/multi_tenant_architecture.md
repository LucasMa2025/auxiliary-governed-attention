# AGA 多租户架构改造方案

## 1. 概述

### 1.1 目标

将 AGA 系统从单租户架构升级为支持多租户的 SaaS 级别服务，实现：

- **租户隔离**：知识、配置、资源完全隔离
- **API 认证授权**：完整的身份验证和权限控制
- **资源配额**：按租户分配槽位、API 调用等配额
- **计费基础**：支持按用量计费的基础设施
- **审计追踪**：完整的操作审计日志

### 1.2 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Gateway Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Rate Limit  │  │ Auth/JWT    │  │ Tenant      │  │ Request       │  │
│  │ Middleware  │  │ Validator   │  │ Resolver    │  │ Router        │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────┐
│                        Tenant Context Layer                              │
│  ┌─────────────────────────────────────┴──────────────────────────────┐ │
│  │                     TenantContext (Thread-Local)                    │ │
│  │  tenant_id | api_key_id | permissions | quotas | metadata          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────┐
│                         Service Layer                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Portal       │  │ Knowledge    │  │ Lifecycle    │  │ Sync        │ │
│  │ Service      │  │ Service      │  │ Service      │  │ Service     │ │
│  │ (scoped)     │  │ (scoped)     │  │ (scoped)     │  │ (scoped)    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────┐
│                      Data Isolation Layer                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Namespace Strategy                               │ │
│  │  namespace = "{tenant_id}:{app_namespace}"                         │ │
│  │  e.g., "tenant_abc:production", "tenant_xyz:staging"               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ PostgreSQL   │  │ Redis        │  │ Slot Pool    │  │ Audit Log   │ │
│  │ (per-tenant) │  │ (per-tenant) │  │ (per-tenant) │  │ (per-tenant)│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据模型设计

### 2.1 租户表 (tenants)

```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(50) NOT NULL UNIQUE,  -- URL 友好标识
    
    -- 状态
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active, suspended, deleted
    
    -- 配置
    config JSONB NOT NULL DEFAULT '{}',
    
    -- 配额
    max_slots INTEGER NOT NULL DEFAULT 1000,
    max_namespaces INTEGER NOT NULL DEFAULT 10,
    max_api_keys INTEGER NOT NULL DEFAULT 20,
    max_requests_per_minute INTEGER NOT NULL DEFAULT 1000,
    max_requests_per_day INTEGER NOT NULL DEFAULT 100000,
    
    -- 计费
    billing_plan VARCHAR(50) NOT NULL DEFAULT 'free',  -- free, starter, pro, enterprise
    billing_email VARCHAR(255),
    
    -- 元数据
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 索引
    CONSTRAINT tenants_status_check CHECK (status IN ('active', 'suspended', 'deleted'))
);

CREATE INDEX idx_tenants_slug ON tenants(slug);
CREATE INDEX idx_tenants_status ON tenants(status);
```

### 2.2 API 密钥表 (api_keys)

```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- 密钥
    key_prefix VARCHAR(8) NOT NULL,     -- 用于识别的前缀 (如 "aga_prod_")
    key_hash VARCHAR(64) NOT NULL,      -- SHA-256 哈希
    key_hint VARCHAR(8),                -- 最后4位，用于用户识别
    
    -- 名称和描述
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- 权限
    permissions JSONB NOT NULL DEFAULT '["read", "write"]',
    allowed_namespaces JSONB,           -- null = 所有命名空间
    allowed_ips JSONB,                  -- IP 白名单
    
    -- 状态
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active, revoked, expired
    
    -- 有效期
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- 速率限制（覆盖租户默认值）
    rate_limit_per_minute INTEGER,
    rate_limit_per_day INTEGER,
    
    -- 元数据
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100),
    
    -- 唯一约束
    UNIQUE(tenant_id, name)
);

CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_prefix_hash ON api_keys(key_prefix, key_hash);
CREATE INDEX idx_api_keys_status ON api_keys(status);
```

### 2.3 命名空间表 (namespaces)

```sql
CREATE TABLE namespaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- 标识
    name VARCHAR(100) NOT NULL,         -- 用户定义的名称
    full_name VARCHAR(200) NOT NULL,    -- tenant_id:name 格式
    
    -- 配置
    config JSONB NOT NULL DEFAULT '{}',
    
    -- 配额（覆盖租户默认值）
    max_slots INTEGER,
    
    -- 编码器配置
    encoder_type VARCHAR(50),
    encoder_config JSONB,
    encoder_signature JSONB,            -- 编码器签名，用于一致性检查
    
    -- 状态
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- 统计
    slot_count INTEGER NOT NULL DEFAULT 0,
    total_hits BIGINT NOT NULL DEFAULT 0,
    
    -- 元数据
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 唯一约束
    UNIQUE(tenant_id, name)
);

CREATE INDEX idx_namespaces_tenant ON namespaces(tenant_id);
CREATE INDEX idx_namespaces_full_name ON namespaces(full_name);
```

### 2.4 扩展知识槽位表 (aga_slots)

```sql
-- 扩展现有的 aga_slots 表
ALTER TABLE aga_slots ADD COLUMN IF NOT EXISTS tenant_id UUID;
ALTER TABLE aga_slots ADD COLUMN IF NOT EXISTS trust_tier VARCHAR(20) DEFAULT 'standard';
ALTER TABLE aga_slots ADD COLUMN IF NOT EXISTS created_by VARCHAR(100);
ALTER TABLE aga_slots ADD COLUMN IF NOT EXISTS api_key_id UUID;

-- 添加租户索引
CREATE INDEX IF NOT EXISTS idx_aga_slots_tenant ON aga_slots(tenant_id);
CREATE INDEX IF NOT EXISTS idx_aga_slots_tenant_namespace ON aga_slots(tenant_id, namespace);
```

### 2.5 审计日志表 (audit_logs)

```sql
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,
    
    -- 操作信息
    action VARCHAR(50) NOT NULL,        -- inject, update, quarantine, delete, etc.
    resource_type VARCHAR(50) NOT NULL, -- knowledge, namespace, api_key, etc.
    resource_id VARCHAR(100),
    
    -- 操作者
    api_key_id UUID,
    api_key_name VARCHAR(100),
    source_ip INET,
    user_agent TEXT,
    
    -- 变更详情
    old_value JSONB,
    new_value JSONB,
    
    -- 结果
    status VARCHAR(20) NOT NULL,        -- success, failed, denied
    error_message TEXT,
    
    -- 请求追踪
    request_id VARCHAR(100),
    trace_id VARCHAR(100),
    
    -- 时间
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 分区键
    partition_key DATE NOT NULL DEFAULT CURRENT_DATE
) PARTITION BY RANGE (partition_key);

-- 按月创建分区
CREATE TABLE audit_logs_2026_01 PARTITION OF audit_logs
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE audit_logs_2026_02 PARTITION OF audit_logs
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
-- ... 继续创建其他月份分区

CREATE INDEX idx_audit_tenant_time ON audit_logs(tenant_id, created_at);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
```

### 2.6 使用量统计表 (usage_stats)

```sql
CREATE TABLE usage_stats (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    
    -- 时间粒度
    stat_date DATE NOT NULL,
    stat_hour SMALLINT,  -- 0-23, null 表示日统计
    
    -- 计数
    api_calls BIGINT NOT NULL DEFAULT 0,
    inject_count BIGINT NOT NULL DEFAULT 0,
    query_count BIGINT NOT NULL DEFAULT 0,
    slot_seconds BIGINT NOT NULL DEFAULT 0,  -- 槽位使用时长
    
    -- 详细统计
    stats_detail JSONB,  -- 按 namespace、api_key 等维度的细分
    
    -- 唯一约束
    UNIQUE(tenant_id, stat_date, stat_hour)
);

CREATE INDEX idx_usage_tenant_date ON usage_stats(tenant_id, stat_date);
```

---

## 3. API 认证系统设计

### 3.1 认证流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                        API 认证流程                                   │
└──────────────────────────────────────────────────────────────────────┘

1. 请求到达
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  提取 API Key                                                         │
│  - Header: Authorization: Bearer aga_prod_xxxx...                    │
│  - Header: X-API-Key: aga_prod_xxxx...                               │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  验证 API Key                                                         │
│  1. 检查 key_prefix 格式                                              │
│  2. 计算 key_hash = SHA256(api_key)                                  │
│  3. 查询数据库: SELECT * FROM api_keys WHERE key_hash = ?            │
│  4. 验证状态: status = 'active'                                       │
│  5. 验证有效期: expires_at IS NULL OR expires_at > NOW()             │
│  6. 验证 IP 白名单 (如果配置)                                         │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  加载租户信息                                                         │
│  1. 查询 tenant: SELECT * FROM tenants WHERE id = api_key.tenant_id  │
│  2. 验证租户状态: status = 'active'                                   │
│  3. 加载配额信息                                                      │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  构建 TenantContext                                                   │
│  {                                                                    │
│    tenant_id: "...",                                                 │
│    tenant_slug: "...",                                               │
│    api_key_id: "...",                                                │
│    permissions: ["read", "write", "admin"],                          │
│    allowed_namespaces: ["production", "staging"],                    │
│    quotas: { max_slots: 1000, ... },                                │
│    rate_limits: { per_minute: 100, per_day: 10000 },                │
│  }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  速率限制检查                                                         │
│  1. 检查分钟级限制: Redis INCR + EXPIRE                              │
│  2. 检查日限制: Redis INCR + EXPIREAT                                │
│  3. 超限返回 429 Too Many Requests                                   │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  权限检查                                                             │
│  1. 检查操作权限: has_permission(action)                             │
│  2. 检查命名空间权限: is_namespace_allowed(namespace)                │
│  3. 无权限返回 403 Forbidden                                         │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
   继续处理请求 → Service Layer
```

### 3.2 API Key 格式

```
API Key 格式: {prefix}_{environment}_{random_string}

示例:
- aga_prod_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
- aga_test_xYz789AbCdEfGhIjKlMnOpQrStUvWx01

组成部分:
- prefix: "aga" (固定前缀)
- environment: "prod" | "test" | "dev" (环境标识)
- random_string: 32 字符随机字符串 (Base62 编码)

总长度: 3 + 1 + 4 + 1 + 32 = 41 字符
```

### 3.3 权限定义

```python
class Permission(str, Enum):
    """API 权限"""
    
    # 知识操作
    KNOWLEDGE_READ = "knowledge:read"       # 查询知识
    KNOWLEDGE_WRITE = "knowledge:write"     # 注入知识
    KNOWLEDGE_DELETE = "knowledge:delete"   # 删除知识
    
    # 生命周期操作
    LIFECYCLE_READ = "lifecycle:read"       # 查看状态
    LIFECYCLE_UPDATE = "lifecycle:update"   # 更新状态
    LIFECYCLE_QUARANTINE = "lifecycle:quarantine"  # 隔离
    
    # 命名空间操作
    NAMESPACE_READ = "namespace:read"
    NAMESPACE_CREATE = "namespace:create"
    NAMESPACE_UPDATE = "namespace:update"
    NAMESPACE_DELETE = "namespace:delete"
    
    # 管理操作
    ADMIN_API_KEYS = "admin:api_keys"       # 管理 API 密钥
    ADMIN_AUDIT = "admin:audit"             # 查看审计日志
    ADMIN_STATS = "admin:stats"             # 查看统计信息

# 预定义角色
ROLES = {
    "reader": [
        Permission.KNOWLEDGE_READ,
        Permission.LIFECYCLE_READ,
        Permission.NAMESPACE_READ,
    ],
    "writer": [
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_WRITE,
        Permission.LIFECYCLE_READ,
        Permission.LIFECYCLE_UPDATE,
        Permission.NAMESPACE_READ,
    ],
    "manager": [
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_WRITE,
        Permission.KNOWLEDGE_DELETE,
        Permission.LIFECYCLE_READ,
        Permission.LIFECYCLE_UPDATE,
        Permission.LIFECYCLE_QUARANTINE,
        Permission.NAMESPACE_READ,
        Permission.NAMESPACE_CREATE,
        Permission.NAMESPACE_UPDATE,
    ],
    "admin": [
        # 所有权限
        *Permission,
    ],
}
```

---

## 4. 核心代码实现

### 4.1 租户上下文 (TenantContext)

```python
# aga/multi_tenant/context.py

import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from contextvars import ContextVar


@dataclass
class TenantQuotas:
    """租户配额"""
    max_slots: int = 1000
    max_namespaces: int = 10
    max_api_keys: int = 20
    max_requests_per_minute: int = 1000
    max_requests_per_day: int = 100000


@dataclass
class RateLimits:
    """速率限制"""
    per_minute: int = 1000
    per_day: int = 100000


@dataclass
class TenantContext:
    """
    租户上下文
    
    包含当前请求的租户信息，通过 ContextVar 在请求处理过程中传递。
    """
    tenant_id: str
    tenant_slug: str
    tenant_name: str
    
    # API Key 信息
    api_key_id: str
    api_key_name: str
    
    # 权限
    permissions: Set[str] = field(default_factory=set)
    allowed_namespaces: Optional[List[str]] = None  # None = 所有
    
    # 配额和限制
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    rate_limits: RateLimits = field(default_factory=RateLimits)
    
    # 请求元数据
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # 额外数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        # 通配符权限
        if "*" in self.permissions:
            return True
        
        # 精确匹配
        if permission in self.permissions:
            return True
        
        # 前缀匹配 (如 knowledge:* 匹配 knowledge:read)
        prefix = permission.split(":")[0] + ":*"
        if prefix in self.permissions:
            return True
        
        return False
    
    def is_namespace_allowed(self, namespace: str) -> bool:
        """检查命名空间是否允许访问"""
        if self.allowed_namespaces is None:
            return True
        return namespace in self.allowed_namespaces
    
    def get_full_namespace(self, namespace: str) -> str:
        """
        获取完整的命名空间名称
        
        格式: {tenant_id}:{namespace}
        """
        return f"{self.tenant_id}:{namespace}"
    
    def strip_tenant_prefix(self, full_namespace: str) -> str:
        """从完整命名空间中移除租户前缀"""
        prefix = f"{self.tenant_id}:"
        if full_namespace.startswith(prefix):
            return full_namespace[len(prefix):]
        return full_namespace


# 全局上下文变量
_tenant_context: ContextVar[Optional[TenantContext]] = ContextVar(
    'tenant_context', 
    default=None
)


def get_current_tenant() -> Optional[TenantContext]:
    """获取当前租户上下文"""
    return _tenant_context.get()


def set_current_tenant(context: TenantContext) -> None:
    """设置当前租户上下文"""
    _tenant_context.set(context)


def clear_current_tenant() -> None:
    """清除当前租户上下文"""
    _tenant_context.set(None)


def require_tenant() -> TenantContext:
    """获取当前租户上下文，如果不存在则抛出异常"""
    ctx = get_current_tenant()
    if ctx is None:
        raise RuntimeError("No tenant context available")
    return ctx
```

### 4.2 API Key 管理器

```python
# aga/multi_tenant/api_keys.py

import hashlib
import secrets
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import logging

logger = logging.getLogger(__name__)


@dataclass
class APIKeyInfo:
    """API Key 信息"""
    id: str
    tenant_id: str
    name: str
    permissions: List[str]
    allowed_namespaces: Optional[List[str]]
    rate_limit_per_minute: Optional[int]
    rate_limit_per_day: Optional[int]
    expires_at: Optional[datetime]
    metadata: Optional[Dict[str, Any]]


class APIKeyManager:
    """
    API Key 管理器
    
    负责 API Key 的生成、验证和管理。
    """
    
    KEY_PREFIX = "aga"
    KEY_LENGTH = 32  # 随机部分长度
    CACHE_TTL = 300  # 缓存 5 分钟
    
    def __init__(
        self,
        db_session,
        redis_client=None,
        cache_enabled: bool = True,
    ):
        self.db = db_session
        self.redis = redis_client
        self.cache_enabled = cache_enabled and redis_client is not None
    
    def generate_api_key(
        self,
        tenant_id: str,
        name: str,
        environment: str = "prod",
        permissions: Optional[List[str]] = None,
        allowed_namespaces: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_day: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> tuple[str, APIKeyInfo]:
        """
        生成新的 API Key
        
        Returns:
            (raw_key, api_key_info)
            注意：raw_key 只返回一次，需要安全保存
        """
        # 生成随机 key
        random_part = secrets.token_urlsafe(self.KEY_LENGTH)[:self.KEY_LENGTH]
        raw_key = f"{self.KEY_PREFIX}_{environment}_{random_part}"
        
        # 计算哈希
        key_hash = self._hash_key(raw_key)
        key_hint = raw_key[-4:]
        key_prefix = f"{self.KEY_PREFIX}_{environment}_"
        
        # 计算过期时间
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # 默认权限
        if permissions is None:
            permissions = ["knowledge:read", "knowledge:write"]
        
        # 保存到数据库
        from sqlalchemy import text
        
        sql = text("""
        INSERT INTO api_keys (
            tenant_id, key_prefix, key_hash, key_hint, name,
            permissions, allowed_namespaces, allowed_ips,
            expires_at, rate_limit_per_minute, rate_limit_per_day,
            created_by
        ) VALUES (
            :tenant_id, :key_prefix, :key_hash, :key_hint, :name,
            :permissions, :allowed_namespaces, :allowed_ips,
            :expires_at, :rate_limit_per_minute, :rate_limit_per_day,
            :created_by
        ) RETURNING id
        """)
        
        import json
        result = self.db.execute(sql, {
            "tenant_id": tenant_id,
            "key_prefix": key_prefix,
            "key_hash": key_hash,
            "key_hint": key_hint,
            "name": name,
            "permissions": json.dumps(permissions),
            "allowed_namespaces": json.dumps(allowed_namespaces) if allowed_namespaces else None,
            "allowed_ips": json.dumps(allowed_ips) if allowed_ips else None,
            "expires_at": expires_at,
            "rate_limit_per_minute": rate_limit_per_minute,
            "rate_limit_per_day": rate_limit_per_day,
            "created_by": created_by,
        })
        
        key_id = result.fetchone()[0]
        self.db.commit()
        
        info = APIKeyInfo(
            id=str(key_id),
            tenant_id=tenant_id,
            name=name,
            permissions=permissions,
            allowed_namespaces=allowed_namespaces,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_day=rate_limit_per_day,
            expires_at=expires_at,
            metadata=None,
        )
        
        logger.info(f"Generated API key: {key_prefix}****{key_hint} for tenant {tenant_id}")
        
        return raw_key, info
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKeyInfo]:
        """
        验证 API Key
        
        Returns:
            APIKeyInfo 如果有效，否则 None
        """
        # 检查格式
        if not raw_key or not raw_key.startswith(f"{self.KEY_PREFIX}_"):
            return None
        
        # 尝试从缓存获取
        if self.cache_enabled:
            cached = self._get_from_cache(raw_key)
            if cached:
                return cached
        
        # 计算哈希
        key_hash = self._hash_key(raw_key)
        
        # 从数据库查询
        from sqlalchemy import text
        
        sql = text("""
        SELECT 
            ak.id, ak.tenant_id, ak.name, ak.permissions,
            ak.allowed_namespaces, ak.rate_limit_per_minute,
            ak.rate_limit_per_day, ak.expires_at, ak.metadata,
            t.status as tenant_status
        FROM api_keys ak
        JOIN tenants t ON ak.tenant_id = t.id
        WHERE ak.key_hash = :key_hash
          AND ak.status = 'active'
          AND t.status = 'active'
        """)
        
        result = self.db.execute(sql, {"key_hash": key_hash})
        row = result.fetchone()
        
        if not row:
            return None
        
        # 检查过期
        if row.expires_at and row.expires_at < datetime.utcnow():
            return None
        
        import json
        
        info = APIKeyInfo(
            id=str(row.id),
            tenant_id=str(row.tenant_id),
            name=row.name,
            permissions=json.loads(row.permissions) if row.permissions else [],
            allowed_namespaces=json.loads(row.allowed_namespaces) if row.allowed_namespaces else None,
            rate_limit_per_minute=row.rate_limit_per_minute,
            rate_limit_per_day=row.rate_limit_per_day,
            expires_at=row.expires_at,
            metadata=json.loads(row.metadata) if row.metadata else None,
        )
        
        # 更新最后使用时间（异步）
        self._update_last_used(row.id)
        
        # 缓存
        if self.cache_enabled:
            self._set_cache(raw_key, info)
        
        return info
    
    def revoke_api_key(self, key_id: str, tenant_id: str) -> bool:
        """撤销 API Key"""
        from sqlalchemy import text
        
        sql = text("""
        UPDATE api_keys 
        SET status = 'revoked', updated_at = NOW()
        WHERE id = :key_id AND tenant_id = :tenant_id
        """)
        
        result = self.db.execute(sql, {"key_id": key_id, "tenant_id": tenant_id})
        self.db.commit()
        
        # 清除缓存
        if self.cache_enabled:
            self._invalidate_cache_by_id(key_id)
        
        return result.rowcount > 0
    
    def _hash_key(self, raw_key: str) -> str:
        """计算 API Key 的 SHA-256 哈希"""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def _get_from_cache(self, raw_key: str) -> Optional[APIKeyInfo]:
        """从缓存获取"""
        if not self.redis:
            return None
        
        cache_key = f"aga:api_key:{self._hash_key(raw_key)}"
        data = self.redis.get(cache_key)
        
        if data:
            import json
            d = json.loads(data)
            return APIKeyInfo(**d)
        return None
    
    def _set_cache(self, raw_key: str, info: APIKeyInfo):
        """设置缓存"""
        if not self.redis:
            return
        
        import json
        cache_key = f"aga:api_key:{self._hash_key(raw_key)}"
        
        # 转换为可序列化格式
        data = {
            "id": info.id,
            "tenant_id": info.tenant_id,
            "name": info.name,
            "permissions": info.permissions,
            "allowed_namespaces": info.allowed_namespaces,
            "rate_limit_per_minute": info.rate_limit_per_minute,
            "rate_limit_per_day": info.rate_limit_per_day,
            "expires_at": info.expires_at.isoformat() if info.expires_at else None,
            "metadata": info.metadata,
        }
        
        self.redis.setex(cache_key, self.CACHE_TTL, json.dumps(data))
    
    def _invalidate_cache_by_id(self, key_id: str):
        """按 ID 清除缓存（需要遍历，较慢）"""
        # 实际实现中可以维护 key_id -> cache_key 的映射
        pass
    
    def _update_last_used(self, key_id: str):
        """更新最后使用时间"""
        from sqlalchemy import text
        
        sql = text("""
        UPDATE api_keys SET last_used_at = NOW() WHERE id = :key_id
        """)
        
        try:
            self.db.execute(sql, {"key_id": key_id})
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to update last_used_at: {e}")
```

### 4.3 认证中间件 (FastAPI)

```python
# aga/multi_tenant/middleware.py

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import time
import logging

from .context import TenantContext, set_current_tenant, clear_current_tenant, TenantQuotas, RateLimits
from .api_keys import APIKeyManager, APIKeyInfo

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    认证中间件
    
    处理 API Key 验证、租户解析、速率限制。
    """
    
    # 不需要认证的路径
    PUBLIC_PATHS = [
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]
    
    def __init__(
        self,
        app,
        api_key_manager: APIKeyManager,
        redis_client=None,
        db_session=None,
    ):
        super().__init__(app)
        self.api_key_manager = api_key_manager
        self.redis = redis_client
        self.db = db_session
    
    async def dispatch(self, request: Request, call_next):
        # 跳过公开路径
        if any(request.url.path.startswith(p) for p in self.PUBLIC_PATHS):
            return await call_next(request)
        
        try:
            # 1. 提取 API Key
            api_key = self._extract_api_key(request)
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="Missing API key",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # 2. 验证 API Key
            key_info = self.api_key_manager.validate_api_key(api_key)
            if not key_info:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired API key",
                )
            
            # 3. 加载租户信息
            tenant = await self._load_tenant(key_info.tenant_id)
            if not tenant:
                raise HTTPException(
                    status_code=403,
                    detail="Tenant not found or suspended",
                )
            
            # 4. 构建上下文
            context = self._build_context(request, key_info, tenant)
            
            # 5. 速率限制检查
            if self.redis:
                allowed, retry_after = await self._check_rate_limit(context)
                if not allowed:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={"Retry-After": str(retry_after)},
                    )
            
            # 6. 设置上下文
            set_current_tenant(context)
            
            # 7. 处理请求
            response = await call_next(request)
            
            # 8. 添加响应头
            response.headers["X-Tenant-ID"] = context.tenant_id
            response.headers["X-Request-ID"] = context.request_id or ""
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Auth middleware error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            clear_current_tenant()
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """提取 API Key"""
        # 1. Authorization: Bearer xxx
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]
        
        # 2. X-API-Key: xxx
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            return api_key_header
        
        # 3. Query parameter (不推荐，仅用于测试)
        api_key_param = request.query_params.get("api_key")
        if api_key_param:
            return api_key_param
        
        return None
    
    async def _load_tenant(self, tenant_id: str) -> Optional[dict]:
        """加载租户信息"""
        from sqlalchemy import text
        
        sql = text("""
        SELECT id, name, slug, status, config, 
               max_slots, max_namespaces, max_api_keys,
               max_requests_per_minute, max_requests_per_day,
               billing_plan, metadata
        FROM tenants WHERE id = :tenant_id AND status = 'active'
        """)
        
        result = self.db.execute(sql, {"tenant_id": tenant_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return {
            "id": str(row.id),
            "name": row.name,
            "slug": row.slug,
            "config": row.config,
            "quotas": {
                "max_slots": row.max_slots,
                "max_namespaces": row.max_namespaces,
                "max_api_keys": row.max_api_keys,
                "max_requests_per_minute": row.max_requests_per_minute,
                "max_requests_per_day": row.max_requests_per_day,
            },
            "billing_plan": row.billing_plan,
            "metadata": row.metadata,
        }
    
    def _build_context(
        self,
        request: Request,
        key_info: APIKeyInfo,
        tenant: dict,
    ) -> TenantContext:
        """构建租户上下文"""
        import uuid
        
        # 合并速率限制（API Key 级别覆盖租户级别）
        rate_per_min = key_info.rate_limit_per_minute or tenant["quotas"]["max_requests_per_minute"]
        rate_per_day = key_info.rate_limit_per_day or tenant["quotas"]["max_requests_per_day"]
        
        return TenantContext(
            tenant_id=tenant["id"],
            tenant_slug=tenant["slug"],
            tenant_name=tenant["name"],
            api_key_id=key_info.id,
            api_key_name=key_info.name,
            permissions=set(key_info.permissions),
            allowed_namespaces=key_info.allowed_namespaces,
            quotas=TenantQuotas(
                max_slots=tenant["quotas"]["max_slots"],
                max_namespaces=tenant["quotas"]["max_namespaces"],
                max_api_keys=tenant["quotas"]["max_api_keys"],
                max_requests_per_minute=rate_per_min,
                max_requests_per_day=rate_per_day,
            ),
            rate_limits=RateLimits(
                per_minute=rate_per_min,
                per_day=rate_per_day,
            ),
            request_id=request.headers.get("X-Request-ID") or str(uuid.uuid4()),
            source_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            metadata=tenant.get("metadata") or {},
        )
    
    async def _check_rate_limit(self, context: TenantContext) -> tuple[bool, int]:
        """
        检查速率限制
        
        Returns:
            (allowed, retry_after_seconds)
        """
        import time
        
        tenant_id = context.tenant_id
        now = int(time.time())
        current_minute = now // 60
        current_day = now // 86400
        
        # 分钟级限制
        minute_key = f"aga:rate:{tenant_id}:min:{current_minute}"
        minute_count = self.redis.incr(minute_key)
        if minute_count == 1:
            self.redis.expire(minute_key, 60)
        
        if minute_count > context.rate_limits.per_minute:
            retry_after = 60 - (now % 60)
            return False, retry_after
        
        # 日限制
        day_key = f"aga:rate:{tenant_id}:day:{current_day}"
        day_count = self.redis.incr(day_key)
        if day_count == 1:
            # 设置到次日 0 点过期
            self.redis.expireat(day_key, (current_day + 1) * 86400)
        
        if day_count > context.rate_limits.per_day:
            retry_after = (current_day + 1) * 86400 - now
            return False, retry_after
        
        return True, 0


# FastAPI 依赖
security = HTTPBearer(auto_error=False)


def require_permission(permission: str):
    """
    权限检查依赖
    
    Usage:
        @app.post("/knowledge/inject")
        async def inject(
            request: InjectRequest,
            _: None = Depends(require_permission("knowledge:write"))
        ):
            ...
    """
    async def check_permission():
        from .context import require_tenant
        
        ctx = require_tenant()
        if not ctx.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Missing permission: {permission}",
            )
        return ctx
    
    return Depends(check_permission)
```

### 4.4 租户感知的服务层

```python
# aga/multi_tenant/scoped_service.py

from typing import Optional, List, Dict, Any
from .context import require_tenant, TenantContext


class TenantScopedService:
    """
    租户感知的服务基类
    
    所有服务方法自动将命名空间限定到当前租户。
    """
    
    def __init__(self, base_service):
        self._base = base_service
    
    @property
    def context(self) -> TenantContext:
        return require_tenant()
    
    def _scope_namespace(self, namespace: str) -> str:
        """将命名空间限定到当前租户"""
        ctx = self.context
        
        # 检查命名空间权限
        if not ctx.is_namespace_allowed(namespace):
            raise PermissionError(f"Access to namespace '{namespace}' denied")
        
        # 返回完整命名空间
        return ctx.get_full_namespace(namespace)
    
    def _check_permission(self, permission: str):
        """检查权限"""
        if not self.context.has_permission(permission):
            raise PermissionError(f"Missing permission: {permission}")


class TenantScopedPortalService(TenantScopedService):
    """租户感知的 Portal 服务"""
    
    async def inject_knowledge(
        self,
        namespace: str,
        lu_id: str,
        condition: str,
        decision: str,
        lifecycle_state: str = "probationary",
        trust_tier: str = "standard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """注入知识（自动限定租户）"""
        self._check_permission("knowledge:write")
        
        scoped_namespace = self._scope_namespace(namespace)
        
        # 添加租户信息到元数据
        meta = metadata or {}
        meta["tenant_id"] = self.context.tenant_id
        meta["api_key_id"] = self.context.api_key_id
        meta["created_by"] = self.context.api_key_name
        
        return await self._base.inject_knowledge_text(
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            namespace=scoped_namespace,
            lifecycle_state=lifecycle_state,
            trust_tier=trust_tier,
            metadata=meta,
        )
    
    async def query_knowledge(
        self,
        namespace: str,
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """查询知识（自动限定租户）"""
        self._check_permission("knowledge:read")
        
        scoped_namespace = self._scope_namespace(namespace)
        
        result = await self._base.query_knowledge(
            namespace=scoped_namespace,
            lifecycle_states=lifecycle_states,
            trust_tiers=trust_tiers,
            limit=limit,
            offset=offset,
        )
        
        # 移除结果中的租户前缀
        for item in result.get("items", []):
            if "namespace" in item:
                item["namespace"] = self.context.strip_tenant_prefix(item["namespace"])
        
        return result
    
    async def quarantine_knowledge(
        self,
        namespace: str,
        lu_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """隔离知识"""
        self._check_permission("lifecycle:quarantine")
        
        scoped_namespace = self._scope_namespace(namespace)
        
        return await self._base.quarantine_knowledge(
            namespace=scoped_namespace,
            lu_id=lu_id,
            reason=reason,
            operator=self.context.api_key_name,
        )
    
    async def get_statistics(self, namespace: str) -> Dict[str, Any]:
        """获取统计信息"""
        self._check_permission("admin:stats")
        
        scoped_namespace = self._scope_namespace(namespace)
        
        stats = await self._base.get_statistics(scoped_namespace)
        stats["namespace"] = namespace  # 返回用户的命名空间名
        
        return stats
```

---

## 5. API 接口设计

### 5.1 租户管理 API

```yaml
# 仅限平台管理员

# 创建租户
POST /admin/tenants
Authorization: Bearer {admin_token}
{
  "name": "Acme Corp",
  "slug": "acme",
  "billing_email": "billing@acme.com",
  "billing_plan": "pro",
  "quotas": {
    "max_slots": 10000,
    "max_namespaces": 50
  }
}

# 响应
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Acme Corp",
    "slug": "acme",
    "status": "active",
    "created_at": "2026-02-03T10:00:00Z"
  }
}

# 获取租户列表
GET /admin/tenants?status=active&limit=20

# 更新租户
PATCH /admin/tenants/{tenant_id}
{
  "billing_plan": "enterprise",
  "quotas": { "max_slots": 50000 }
}

# 暂停租户
POST /admin/tenants/{tenant_id}/suspend

# 恢复租户
POST /admin/tenants/{tenant_id}/activate
```

### 5.2 API Key 管理

```yaml
# 创建 API Key
POST /api-keys
Authorization: Bearer {api_key}  # 需要 admin:api_keys 权限
{
  "name": "Production Key",
  "environment": "prod",
  "permissions": ["knowledge:read", "knowledge:write", "lifecycle:update"],
  "allowed_namespaces": ["production", "staging"],
  "expires_in_days": 365,
  "rate_limit_per_minute": 500
}

# 响应
{
  "success": true,
  "data": {
    "id": "key_123",
    "name": "Production Key",
    "key": "aga_prod_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456",  # 仅返回一次!
    "key_hint": "3456",
    "permissions": ["knowledge:read", "knowledge:write", "lifecycle:update"],
    "expires_at": "2027-02-03T10:00:00Z"
  },
  "warning": "Please save your API key securely. It will not be shown again."
}

# 列出 API Keys
GET /api-keys
Authorization: Bearer {api_key}

# 撤销 API Key
DELETE /api-keys/{key_id}
Authorization: Bearer {api_key}

# 响应
{
  "success": true,
  "message": "API key revoked"
}
```

### 5.3 知识管理 API (带认证)

```yaml
# 注入知识
POST /knowledge/inject-text
Authorization: Bearer aga_prod_xxxx...
X-Request-ID: req_123456
{
  "lu_id": "lu_001",
  "condition": "当用户询问天气时",
  "decision": "调用天气 API 获取实时数据",
  "namespace": "production",
  "lifecycle_state": "probationary",
  "trust_tier": "standard"
}

# 响应
{
  "success": true,
  "data": {
    "lu_id": "lu_001",
    "namespace": "production",
    "lifecycle_state": "probationary",
    "slot_idx": 42
  },
  "meta": {
    "request_id": "req_123456",
    "tenant_id": "550e8400-...",
    "latency_ms": 15.3
  }
}

# 查询知识
GET /knowledge/{namespace}?lifecycle_states=confirmed,probationary&limit=50
Authorization: Bearer aga_prod_xxxx...

# 响应
{
  "success": true,
  "data": {
    "items": [...],
    "total": 150,
    "limit": 50,
    "offset": 0
  }
}

# 隔离知识
POST /knowledge/{namespace}/{lu_id}/quarantine
Authorization: Bearer aga_prod_xxxx...
{
  "reason": "发现错误信息"
}

# 响应头示例
HTTP/1.1 200 OK
X-Tenant-ID: 550e8400-e29b-41d4-a716-446655440000
X-Request-ID: req_123456
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1706954400
```

### 5.4 审计日志 API

```yaml
# 查询审计日志
GET /audit-logs?action=inject&start_date=2026-02-01&end_date=2026-02-03
Authorization: Bearer {api_key}  # 需要 admin:audit 权限

# 响应
{
  "success": true,
  "data": {
    "items": [
      {
        "id": 12345,
        "action": "inject",
        "resource_type": "knowledge",
        "resource_id": "lu_001",
        "api_key_name": "Production Key",
        "source_ip": "192.168.1.100",
        "status": "success",
        "created_at": "2026-02-03T10:00:00Z"
      }
    ],
    "total": 1523,
    "has_more": true
  }
}
```

---

## 6. 部署架构

### 6.1 推荐的生产部署

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Load Balancer                                  │
│                      (Nginx / AWS ALB / Cloudflare)                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────────┐
        │                            │                                │
        ▼                            ▼                                ▼
┌───────────────┐          ┌───────────────┐              ┌───────────────┐
│  Portal API   │          │  Portal API   │              │  Portal API   │
│  Instance 1   │          │  Instance 2   │              │  Instance N   │
│  (Stateless)  │          │  (Stateless)  │              │  (Stateless)  │
└───────┬───────┘          └───────┬───────┘              └───────┬───────┘
        │                          │                              │
        └──────────────────────────┼──────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        │                          │                              │
        ▼                          ▼                              ▼
┌───────────────┐          ┌───────────────┐              ┌───────────────┐
│    Redis      │          │  PostgreSQL   │              │  Kafka/Redis  │
│   (Cache +    │          │   (Primary +  │              │   Pub/Sub     │
│  Rate Limit)  │          │   Replicas)   │              │  (Sync)       │
│   Cluster     │          │   Cluster     │              │               │
└───────────────┘          └───────────────┘              └───────────────┘
                                                                  │
                                                                  │
        ┌─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Runtime Layer (GPU Nodes)                        │
│                                                                          │
│  ┌───────────────┐  ┌───────────────┐          ┌───────────────┐       │
│  │ Runtime Agent │  │ Runtime Agent │   ...    │ Runtime Agent │       │
│  │ + AGA + LLM   │  │ + AGA + LLM   │          │ + AGA + LLM   │       │
│  │   (GPU 1)     │  │   (GPU 2)     │          │   (GPU N)     │       │
│  └───────────────┘  └───────────────┘          └───────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Kubernetes 部署示例

```yaml
# portal-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aga-portal
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aga-portal
  template:
    metadata:
      labels:
        app: aga-portal
    spec:
      containers:
      - name: portal
        image: aga/portal:v4.0.0
        ports:
        - containerPort: 8081
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aga-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: aga-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: aga-secrets
              key: jwt-secret
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: aga-portal
spec:
  selector:
    app: aga-portal
  ports:
  - port: 80
    targetPort: 8081
  type: ClusterIP
```

---

## 7. 安全考虑

### 7.1 API Key 安全

| 措施 | 描述 |
|------|------|
| **哈希存储** | 使用 SHA-256 哈希存储 API Key，即使数据库泄露也无法还原 |
| **传输加密** | 强制使用 HTTPS，API Key 只在 TLS 加密通道中传输 |
| **IP 白名单** | 支持按 API Key 配置 IP 白名单 |
| **有效期** | 支持设置 API Key 过期时间 |
| **最小权限** | 按需分配权限，不使用通配符权限 |
| **审计追踪** | 记录所有 API Key 使用情况 |
| **定期轮换** | 建议定期轮换 API Key |

### 7.2 数据隔离

| 层级 | 隔离方式 |
|------|----------|
| **命名空间隔离** | 所有数据存储使用 `{tenant_id}:{namespace}` 格式 |
| **查询过滤** | 所有数据库查询自动添加 `tenant_id` 条件 |
| **Redis 隔离** | 使用 Key 前缀 `aga:{tenant_id}:` |
| **运行时隔离** | 每个租户的知识加载到独立的命名空间 |

### 7.3 安全检查清单

- [ ] 所有 API 端点都需要认证
- [ ] API Key 使用安全的随机生成器
- [ ] 敏感数据加密存储
- [ ] 实施速率限制防止滥用
- [ ] 审计日志记录所有敏感操作
- [ ] 定期安全审计
- [ ] 实施 WAF 防护
- [ ] 监控异常访问模式

---

## 8. 计费与配额

### 8.1 计费计划示例

| Plan | 槽位数 | 命名空间 | API 调用/天 | 价格 |
|------|--------|----------|-------------|------|
| **Free** | 100 | 2 | 1,000 | $0/月 |
| **Starter** | 1,000 | 5 | 10,000 | $49/月 |
| **Pro** | 10,000 | 20 | 100,000 | $199/月 |
| **Enterprise** | 无限 | 无限 | 无限 | 联系销售 |

### 8.2 使用量统计

```python
# 每小时汇总使用量
async def aggregate_hourly_usage(tenant_id: str, hour: datetime):
    """汇总小时级使用量"""
    from sqlalchemy import text
    
    # 从审计日志汇总
    sql = text("""
    INSERT INTO usage_stats (tenant_id, stat_date, stat_hour, 
                             api_calls, inject_count, query_count)
    SELECT 
        :tenant_id,
        :stat_date,
        :stat_hour,
        COUNT(*),
        SUM(CASE WHEN action = 'inject' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action = 'query' THEN 1 ELSE 0 END)
    FROM audit_logs
    WHERE tenant_id = :tenant_id
      AND created_at >= :start_time
      AND created_at < :end_time
    ON CONFLICT (tenant_id, stat_date, stat_hour) 
    DO UPDATE SET
        api_calls = EXCLUDED.api_calls,
        inject_count = EXCLUDED.inject_count,
        query_count = EXCLUDED.query_count
    """)
    
    await db.execute(sql, {
        "tenant_id": tenant_id,
        "stat_date": hour.date(),
        "stat_hour": hour.hour,
        "start_time": hour,
        "end_time": hour + timedelta(hours=1),
    })
```

---

## 9. 迁移指南

### 9.1 从单租户迁移到多租户

#### 阶段 1：数据库迁移

```sql
-- 1. 创建默认租户
INSERT INTO tenants (id, name, slug, billing_plan)
VALUES ('00000000-0000-0000-0000-000000000001', 'Default Tenant', 'default', 'enterprise');

-- 2. 更新现有数据
UPDATE aga_slots SET tenant_id = '00000000-0000-0000-0000-000000000001';

-- 3. 更新命名空间（添加租户前缀）
UPDATE aga_slots SET namespace = '00000000-0000-0000-0000-000000000001:' || namespace;

-- 4. 创建默认 API Key
INSERT INTO api_keys (tenant_id, key_prefix, key_hash, name, permissions)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'aga_prod_',
    'your_hashed_key',
    'Migration Key',
    '["*"]'
);
```

#### 阶段 2：代码部署

1. 部署带认证中间件的新版本 Portal
2. 配置环境变量启用多租户模式
3. 验证现有客户端可以正常访问
4. 逐步迁移客户端使用新的认证方式

#### 阶段 3：租户隔离

1. 为新客户创建独立租户
2. 生成租户专用 API Key
3. 配置租户配额
4. 监控使用情况

---

## 10. 监控与运维

### 10.1 关键指标

| 指标 | 描述 | 告警阈值 |
|------|------|----------|
| `aga_tenant_api_calls_total` | 租户 API 调用总数 | N/A |
| `aga_tenant_rate_limit_hits` | 触发限速次数 | > 100/min |
| `aga_auth_failures_total` | 认证失败次数 | > 50/min |
| `aga_tenant_quota_usage_ratio` | 配额使用率 | > 0.9 |
| `aga_api_latency_p99` | API 延迟 P99 | > 500ms |

### 10.2 运维命令

```bash
# 列出所有租户
aga-cli tenants list

# 创建租户
aga-cli tenants create --name "New Corp" --slug "newcorp" --plan pro

# 暂停租户
aga-cli tenants suspend --id xxx

# 生成 API Key
aga-cli api-keys create --tenant xxx --name "Production"

# 查看使用量
aga-cli usage --tenant xxx --start 2026-02-01 --end 2026-02-03

# 导出审计日志
aga-cli audit export --tenant xxx --output audit.json
```

---

## 11. 附录

### 11.1 错误码参考

| 错误码 | HTTP Status | 描述 |
|--------|-------------|------|
| `AUTH_MISSING` | 401 | 缺少 API Key |
| `AUTH_INVALID` | 401 | API Key 无效 |
| `AUTH_EXPIRED` | 401 | API Key 已过期 |
| `TENANT_SUSPENDED` | 403 | 租户已暂停 |
| `PERMISSION_DENIED` | 403 | 权限不足 |
| `NAMESPACE_NOT_ALLOWED` | 403 | 无权访问该命名空间 |
| `RATE_LIMIT_EXCEEDED` | 429 | 超过速率限制 |
| `QUOTA_EXCEEDED` | 429 | 超过配额限制 |

### 11.2 SDK 示例

```python
# Python SDK 使用示例
from aga_client import AGAClient

client = AGAClient(
    base_url="https://api.aga.example.com",
    api_key="aga_prod_xxxxxxxx",
)

# 注入知识
result = client.knowledge.inject(
    namespace="production",
    lu_id="lu_001",
    condition="当用户询问天气时",
    decision="调用天气 API",
)

# 查询知识
items = client.knowledge.query(
    namespace="production",
    lifecycle_states=["confirmed"],
    limit=100,
)

# 隔离知识
client.knowledge.quarantine(
    namespace="production",
    lu_id="lu_001",
    reason="发现问题",
)
```

---

## 12. 更新日志

| 版本 | 日期 | 描述 |
|------|------|------|
| v1.0 | 2026-02-03 | 初始版本 |

---

**文档维护者**: AGA Team  
**最后更新**: 2026-02-03
