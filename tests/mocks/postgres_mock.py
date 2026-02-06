"""
PostgreSQL Mock 实现

模拟 PostgreSQL 客户端行为，用于测试。
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import threading
import re


class MockPostgresConnection:
    """Mock PostgreSQL 连接"""
    
    def __init__(self, pool: 'MockPostgres'):
        self._pool = pool
        self._in_transaction = False
    
    async def execute(self, query: str, *args) -> str:
        """执行查询"""
        return await self._pool.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict]:
        """获取多行"""
        return await self._pool.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict]:
        """获取单行"""
        return await self._pool.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """获取单值"""
        return await self._pool.fetchval(query, *args)
    
    def transaction(self):
        """开始事务"""
        return MockTransaction(self)
    
    async def close(self):
        """关闭连接"""
        pass


class MockTransaction:
    """Mock 事务"""
    
    def __init__(self, conn: MockPostgresConnection):
        self._conn = conn
    
    async def __aenter__(self):
        self._conn._in_transaction = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._conn._in_transaction = False
        if exc_type:
            # 回滚
            pass
        return False


class MockPostgres:
    """
    Mock PostgreSQL 客户端
    
    模拟 asyncpg 的基本操作，用于测试。
    支持：
    - 基本 CRUD 操作
    - 简单的 SQL 解析
    - 事务模拟
    """
    
    def __init__(self, fail_after: int = 0, latency_ms: float = 0):
        """
        初始化 Mock PostgreSQL
        
        Args:
            fail_after: 在多少次操作后开始失败
            latency_ms: 模拟延迟
        """
        # 表数据: {table_name: [row_dict, ...]}
        self._tables: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # 故障注入
        self._fail_after = fail_after
        self._operation_count = 0
        self._latency_ms = latency_ms
        self._is_connected = True
        
        # 自增 ID
        self._auto_increment: Dict[str, int] = defaultdict(int)
    
    def _check_failure(self):
        """检查是否应该失败"""
        self._operation_count += 1
        if self._fail_after > 0 and self._operation_count > self._fail_after:
            raise ConnectionError("Mock PostgreSQL connection failed")
        if not self._is_connected:
            raise ConnectionError("Mock PostgreSQL not connected")
    
    def _simulate_latency(self):
        """模拟延迟"""
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000)
    
    # ==================== 连接管理 ====================
    
    def disconnect(self):
        """断开连接"""
        self._is_connected = False
    
    def reconnect(self):
        """重新连接"""
        self._is_connected = True
    
    async def close(self):
        """关闭连接池"""
        pass
    
    def acquire(self):
        """获取连接"""
        return MockConnectionContext(self)
    
    # ==================== 查询操作 ====================
    
    async def execute(self, query: str, *args) -> str:
        """执行查询"""
        self._check_failure()
        self._simulate_latency()
        
        query_lower = query.lower().strip()
        
        with self._lock:
            if query_lower.startswith("insert"):
                return self._execute_insert(query, args)
            elif query_lower.startswith("update"):
                return self._execute_update(query, args)
            elif query_lower.startswith("delete"):
                return self._execute_delete(query, args)
            elif query_lower.startswith("create"):
                return self._execute_create(query)
            else:
                return "OK"
    
    async def fetch(self, query: str, *args) -> List[Dict]:
        """获取多行"""
        self._check_failure()
        self._simulate_latency()
        
        with self._lock:
            return self._execute_select(query, args)
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict]:
        """获取单行"""
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None
    
    async def fetchval(self, query: str, *args) -> Any:
        """获取单值"""
        row = await self.fetchrow(query, *args)
        if row:
            return list(row.values())[0]
        return None
    
    # ==================== SQL 解析（简化版）====================
    
    def _execute_insert(self, query: str, args: tuple) -> str:
        """执行 INSERT"""
        # 简化解析: INSERT INTO table (cols) VALUES ($1, $2, ...)
        match = re.search(r'insert\s+into\s+(\w+)\s*\(([^)]+)\)', query, re.IGNORECASE)
        if not match:
            return "INSERT 0 0"
        
        table = match.group(1)
        cols = [c.strip() for c in match.group(2).split(',')]
        
        row = {}
        for i, col in enumerate(cols):
            if i < len(args):
                row[col] = args[i]
        
        # 自增 ID
        if 'id' not in row:
            self._auto_increment[table] += 1
            row['id'] = self._auto_increment[table]
        
        self._tables[table].append(row)
        return f"INSERT 0 1"
    
    def _execute_update(self, query: str, args: tuple) -> str:
        """执行 UPDATE"""
        # 简化解析: UPDATE table SET col=$1 WHERE condition
        match = re.search(r'update\s+(\w+)\s+set\s+(.+?)\s+where\s+(.+)', query, re.IGNORECASE)
        if not match:
            return "UPDATE 0"
        
        table = match.group(1)
        set_clause = match.group(2)
        where_clause = match.group(3)
        
        # 解析 SET
        updates = {}
        for part in set_clause.split(','):
            if '=' in part:
                col, val = part.split('=', 1)
                col = col.strip()
                val = val.strip()
                if val.startswith('$'):
                    idx = int(val[1:]) - 1
                    if idx < len(args):
                        updates[col] = args[idx]
        
        # 解析 WHERE（简化：只支持 col = $n）
        where_col, where_val = self._parse_where(where_clause, args)
        
        count = 0
        for row in self._tables.get(table, []):
            if where_col and row.get(where_col) == where_val:
                row.update(updates)
                count += 1
        
        return f"UPDATE {count}"
    
    def _execute_delete(self, query: str, args: tuple) -> str:
        """执行 DELETE"""
        # 简化解析: DELETE FROM table WHERE condition
        match = re.search(r'delete\s+from\s+(\w+)\s+where\s+(.+)', query, re.IGNORECASE)
        if not match:
            return "DELETE 0"
        
        table = match.group(1)
        where_clause = match.group(2)
        
        where_col, where_val = self._parse_where(where_clause, args)
        
        if table not in self._tables:
            return "DELETE 0"
        
        original_len = len(self._tables[table])
        self._tables[table] = [
            row for row in self._tables[table]
            if not (where_col and row.get(where_col) == where_val)
        ]
        count = original_len - len(self._tables[table])
        
        return f"DELETE {count}"
    
    def _execute_select(self, query: str, args: tuple) -> List[Dict]:
        """执行 SELECT"""
        # 简化解析: SELECT cols FROM table WHERE condition
        match = re.search(r'select\s+(.+?)\s+from\s+(\w+)(?:\s+where\s+(.+))?', query, re.IGNORECASE)
        if not match:
            return []
        
        cols_str = match.group(1)
        table = match.group(2)
        where_clause = match.group(3)
        
        # 解析列
        if cols_str.strip() == '*':
            cols = None  # 所有列
        else:
            cols = [c.strip() for c in cols_str.split(',')]
        
        # 获取数据
        rows = self._tables.get(table, [])
        
        # 过滤
        if where_clause:
            where_col, where_val = self._parse_where(where_clause, args)
            if where_col:
                rows = [r for r in rows if r.get(where_col) == where_val]
        
        # 选择列
        if cols:
            rows = [{c: r.get(c) for c in cols} for r in rows]
        
        # 处理 LIMIT
        limit_match = re.search(r'limit\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            limit = int(limit_match.group(1))
            rows = rows[:limit]
        
        # 处理 OFFSET
        offset_match = re.search(r'offset\s+(\d+)', query, re.IGNORECASE)
        if offset_match:
            offset = int(offset_match.group(1))
            rows = rows[offset:]
        
        return rows
    
    def _execute_create(self, query: str) -> str:
        """执行 CREATE TABLE"""
        match = re.search(r'create\s+table\s+(?:if\s+not\s+exists\s+)?(\w+)', query, re.IGNORECASE)
        if match:
            table = match.group(1)
            if table not in self._tables:
                self._tables[table] = []
        return "CREATE TABLE"
    
    def _parse_where(self, where_clause: str, args: tuple) -> Tuple[Optional[str], Any]:
        """解析 WHERE 子句（简化版）"""
        # 支持: col = $n 或 col = 'value'
        match = re.search(r'(\w+)\s*=\s*\$(\d+)', where_clause)
        if match:
            col = match.group(1)
            idx = int(match.group(2)) - 1
            if idx < len(args):
                return col, args[idx]
        
        match = re.search(r"(\w+)\s*=\s*'([^']*)'", where_clause)
        if match:
            return match.group(1), match.group(2)
        
        return None, None
    
    # ==================== 工具方法 ====================
    
    def get_table(self, table: str) -> List[Dict]:
        """获取表数据（测试用）"""
        return list(self._tables.get(table, []))
    
    def clear_table(self, table: str):
        """清空表（测试用）"""
        self._tables[table] = []
    
    def clear_all(self):
        """清空所有数据（测试用）"""
        self._tables.clear()
        self._auto_increment.clear()
    
    def insert_row(self, table: str, row: Dict):
        """直接插入行（测试用）"""
        if 'id' not in row:
            self._auto_increment[table] += 1
            row['id'] = self._auto_increment[table]
        self._tables[table].append(row)


class MockConnectionContext:
    """Mock 连接上下文"""
    
    def __init__(self, pool: MockPostgres):
        self._pool = pool
        self._conn = MockPostgresConnection(pool)
    
    async def __aenter__(self) -> MockPostgresConnection:
        return self._conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
