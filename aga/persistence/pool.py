"""
AGA 数据库连接池管理

提供高效的数据库连接池管理，支持：
- SQLite 连接池
- 连接健康检查
- 自动重连
- 连接统计

版本: v1.0
"""
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Generator
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """连接池配置"""
    min_connections: int = 2          # 最小连接数
    max_connections: int = 10         # 最大连接数
    connection_timeout: float = 30.0  # 获取连接超时（秒）
    idle_timeout: float = 300.0       # 空闲连接超时（秒）
    max_lifetime: float = 3600.0      # 连接最大生命周期（秒）
    health_check_interval: float = 60.0  # 健康检查间隔（秒）
    retry_attempts: int = 3           # 重试次数
    retry_delay: float = 0.5          # 重试延迟（秒）


@dataclass
class PoolStats:
    """连接池统计"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    wait_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_acquire_time_ms: float = 0.0
    acquire_count: int = 0
    
    @property
    def avg_acquire_time_ms(self) -> float:
        return self.total_acquire_time_ms / max(1, self.acquire_count)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'idle_connections': self.idle_connections,
            'wait_count': self.wait_count,
            'timeout_count': self.timeout_count,
            'error_count': self.error_count,
            'avg_acquire_time_ms': self.avg_acquire_time_ms,
        }


class PooledConnection:
    """
    池化连接包装
    
    包装原始数据库连接，添加元数据和生命周期管理。
    """
    
    def __init__(self, conn: sqlite3.Connection, pool: 'ConnectionPool'):
        self.conn = conn
        self.pool = pool
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.use_count = 0
        self._in_use = False
    
    @property
    def age(self) -> float:
        """连接年龄（秒）"""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """空闲时间（秒）"""
        return time.time() - self.last_used_at
    
    def is_healthy(self) -> bool:
        """检查连接健康状态"""
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def is_expired(self, config: PoolConfig) -> bool:
        """检查连接是否过期"""
        # 超过最大生命周期
        if self.age > config.max_lifetime:
            return True
        # 空闲超时
        if not self._in_use and self.idle_time > config.idle_timeout:
            return True
        return False
    
    def mark_used(self):
        """标记为使用中"""
        self._in_use = True
        self.last_used_at = time.time()
        self.use_count += 1
    
    def mark_idle(self):
        """标记为空闲"""
        self._in_use = False
        self.last_used_at = time.time()
    
    def close(self):
        """关闭连接"""
        try:
            self.conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")


class ConnectionPool:
    """
    SQLite 连接池
    
    提供高效的连接复用和管理。
    
    使用示例：
        ```python
        pool = ConnectionPool("data.db", PoolConfig(max_connections=10))
        pool.initialize()
        
        # 使用连接
        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM table")
            results = cursor.fetchall()
        
        # 查看统计
        print(pool.get_stats())
        
        # 关闭池
        pool.close()
        ```
    """
    
    def __init__(self, db_path: str, config: Optional[PoolConfig] = None):
        """
        初始化连接池
        
        Args:
            db_path: 数据库文件路径
            config: 连接池配置
        """
        self.db_path = db_path
        self.config = config or PoolConfig()
        
        # 连接队列
        self._pool: queue.Queue[PooledConnection] = queue.Queue(maxsize=self.config.max_connections)
        self._all_connections: list[PooledConnection] = []
        
        # 状态
        self._lock = threading.RLock()
        self._initialized = False
        self._closed = False
        
        # 统计
        self._stats = PoolStats()
        
        # 健康检查线程
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
    
    def initialize(self) -> bool:
        """
        初始化连接池
        
        创建最小数量的连接。
        """
        with self._lock:
            if self._initialized:
                return True
            
            try:
                # 创建最小连接数
                for _ in range(self.config.min_connections):
                    conn = self._create_connection()
                    if conn:
                        self._pool.put(conn)
                        self._all_connections.append(conn)
                        self._stats.total_connections += 1
                        self._stats.idle_connections += 1
                
                self._initialized = True
                
                # 启动健康检查线程
                self._start_health_check()
                
                logger.info(f"Connection pool initialized with {self.config.min_connections} connections")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize connection pool: {e}")
                return False
    
    def _create_connection(self) -> Optional[PooledConnection]:
        """创建新连接"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=self.config.connection_timeout,
            )
            conn.row_factory = sqlite3.Row
            
            # 启用 WAL 模式提高并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            return PooledConnection(conn, self)
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self._stats.error_count += 1
            return None
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        获取连接（上下文管理器）
        
        自动归还连接到池中。
        
        Yields:
            sqlite3.Connection
        """
        conn = self._acquire()
        try:
            yield conn.conn
        finally:
            self._release(conn)
    
    def _acquire(self) -> PooledConnection:
        """获取连接"""
        start_time = time.time()
        
        with self._lock:
            self._stats.acquire_count += 1
        
        # 尝试从池中获取
        try:
            conn = self._pool.get(timeout=self.config.connection_timeout)
            
            # 检查连接健康
            if not conn.is_healthy() or conn.is_expired(self.config):
                conn.close()
                with self._lock:
                    if conn in self._all_connections:
                        self._all_connections.remove(conn)
                    self._stats.total_connections -= 1
                
                # 创建新连接
                conn = self._create_connection()
                if conn is None:
                    raise Exception("Failed to create new connection")
                
                with self._lock:
                    self._all_connections.append(conn)
                    self._stats.total_connections += 1
            
            conn.mark_used()
            
            with self._lock:
                self._stats.idle_connections -= 1
                self._stats.active_connections += 1
                self._stats.total_acquire_time_ms += (time.time() - start_time) * 1000
            
            return conn
            
        except queue.Empty:
            # 池为空，尝试创建新连接
            with self._lock:
                self._stats.wait_count += 1
                
                if len(self._all_connections) < self.config.max_connections:
                    conn = self._create_connection()
                    if conn:
                        conn.mark_used()
                        self._all_connections.append(conn)
                        self._stats.total_connections += 1
                        self._stats.active_connections += 1
                        self._stats.total_acquire_time_ms += (time.time() - start_time) * 1000
                        return conn
            
            # 重试
            for attempt in range(self.config.retry_attempts):
                time.sleep(self.config.retry_delay)
                try:
                    conn = self._pool.get(timeout=self.config.connection_timeout / 2)
                    conn.mark_used()
                    
                    with self._lock:
                        self._stats.idle_connections -= 1
                        self._stats.active_connections += 1
                        self._stats.total_acquire_time_ms += (time.time() - start_time) * 1000
                    
                    return conn
                except queue.Empty:
                    continue
            
            with self._lock:
                self._stats.timeout_count += 1
            
            raise TimeoutError(f"Failed to acquire connection after {self.config.connection_timeout}s")
    
    def _release(self, conn: PooledConnection):
        """归还连接"""
        conn.mark_idle()
        
        with self._lock:
            self._stats.active_connections -= 1
            self._stats.idle_connections += 1
        
        # 检查是否应该关闭
        if conn.is_expired(self.config) or self._closed:
            conn.close()
            with self._lock:
                if conn in self._all_connections:
                    self._all_connections.remove(conn)
                self._stats.total_connections -= 1
                self._stats.idle_connections -= 1
            return
        
        # 归还到池
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            # 池已满，关闭连接
            conn.close()
            with self._lock:
                if conn in self._all_connections:
                    self._all_connections.remove(conn)
                self._stats.total_connections -= 1
                self._stats.idle_connections -= 1
    
    def _start_health_check(self):
        """启动健康检查线程"""
        def health_check_loop():
            while not self._stop_health_check.wait(self.config.health_check_interval):
                self._perform_health_check()
        
        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="ConnectionPool-HealthCheck"
        )
        self._health_check_thread.start()
    
    def _perform_health_check(self):
        """执行健康检查"""
        with self._lock:
            # 检查所有空闲连接
            to_remove = []
            
            for conn in self._all_connections:
                if not conn._in_use:
                    if conn.is_expired(self.config) or not conn.is_healthy():
                        to_remove.append(conn)
            
            # 移除不健康的连接
            for conn in to_remove:
                conn.close()
                self._all_connections.remove(conn)
                self._stats.total_connections -= 1
                self._stats.idle_connections -= 1
            
            # 补充最小连接数
            while len(self._all_connections) < self.config.min_connections:
                new_conn = self._create_connection()
                if new_conn:
                    try:
                        self._pool.put_nowait(new_conn)
                        self._all_connections.append(new_conn)
                        self._stats.total_connections += 1
                        self._stats.idle_connections += 1
                    except queue.Full:
                        new_conn.close()
                        break
                else:
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        with self._lock:
            return {
                'db_path': self.db_path,
                'initialized': self._initialized,
                'closed': self._closed,
                'config': {
                    'min_connections': self.config.min_connections,
                    'max_connections': self.config.max_connections,
                },
                **self._stats.to_dict(),
            }
    
    def close(self):
        """关闭连接池"""
        with self._lock:
            self._closed = True
            self._stop_health_check.set()
            
            # 关闭所有连接
            for conn in self._all_connections:
                conn.close()
            
            self._all_connections.clear()
            self._stats.total_connections = 0
            self._stats.active_connections = 0
            self._stats.idle_connections = 0
        
        logger.info("Connection pool closed")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
