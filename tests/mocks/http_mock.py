"""
HTTP Mock 实现

模拟 HTTP 客户端行为，用于测试。
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
import re


@dataclass
class MockHTTPResponse:
    """Mock HTTP 响应"""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    content: bytes = b""
    _json: Any = None
    
    def json(self) -> Any:
        """解析 JSON"""
        if self._json is not None:
            return self._json
        return json.loads(self.content.decode())
    
    @property
    def text(self) -> str:
        """获取文本"""
        return self.content.decode()
    
    @property
    def ok(self) -> bool:
        """是否成功"""
        return 200 <= self.status_code < 300
    
    def raise_for_status(self):
        """检查状态码"""
        if not self.ok:
            raise HTTPError(f"HTTP {self.status_code}")


class HTTPError(Exception):
    """HTTP 错误"""
    pass


@dataclass
class MockHTTPRequest:
    """Mock HTTP 请求"""
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    json_data: Any = None
    content: bytes = None
    
    @property
    def path(self) -> str:
        """获取路径"""
        parsed = urlparse(self.url)
        return parsed.path
    
    @property
    def query_params(self) -> Dict[str, List[str]]:
        """获取查询参数"""
        parsed = urlparse(self.url)
        return parse_qs(parsed.query)


class MockHTTPClient:
    """
    Mock HTTP 客户端
    
    模拟 httpx.AsyncClient 的基本操作，用于测试。
    支持：
    - 路由匹配
    - 请求记录
    - 响应模拟
    - 故障注入
    """
    
    def __init__(self, base_url: str = "", fail_after: int = 0, latency_ms: float = 0):
        """
        初始化 Mock HTTP 客户端
        
        Args:
            base_url: 基础 URL
            fail_after: 在多少次请求后开始失败
            latency_ms: 模拟延迟
        """
        self._base_url = base_url
        self._fail_after = fail_after
        self._request_count = 0
        self._latency_ms = latency_ms
        self._is_connected = True
        
        # 路由处理器: [(method, pattern, handler), ...]
        self._routes: List[tuple] = []
        
        # 请求记录
        self._requests: List[MockHTTPRequest] = []
        
        # 默认响应
        self._default_response = MockHTTPResponse(
            status_code=404,
            content=b'{"error": "Not Found"}',
        )
    
    def _check_failure(self):
        """检查是否应该失败"""
        self._request_count += 1
        if self._fail_after > 0 and self._request_count > self._fail_after:
            raise ConnectionError("Mock HTTP connection failed")
        if not self._is_connected:
            raise ConnectionError("Mock HTTP not connected")
    
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
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
    
    # ==================== 路由注册 ====================
    
    def add_route(
        self,
        method: str,
        pattern: str,
        handler: Union[Callable, MockHTTPResponse, Dict],
    ):
        """
        添加路由
        
        Args:
            method: HTTP 方法 (GET, POST, etc.)
            pattern: URL 模式（支持正则）
            handler: 处理器（函数、响应对象或字典）
        """
        self._routes.append((method.upper(), pattern, handler))
    
    def get(self, pattern: str, handler: Union[Callable, MockHTTPResponse, Dict]):
        """添加 GET 路由"""
        self.add_route("GET", pattern, handler)
    
    def post(self, pattern: str, handler: Union[Callable, MockHTTPResponse, Dict]):
        """添加 POST 路由"""
        self.add_route("POST", pattern, handler)
    
    def put(self, pattern: str, handler: Union[Callable, MockHTTPResponse, Dict]):
        """添加 PUT 路由"""
        self.add_route("PUT", pattern, handler)
    
    def delete(self, pattern: str, handler: Union[Callable, MockHTTPResponse, Dict]):
        """添加 DELETE 路由"""
        self.add_route("DELETE", pattern, handler)
    
    def set_default_response(self, response: MockHTTPResponse):
        """设置默认响应"""
        self._default_response = response
    
    # ==================== 请求方法 ====================
    
    async def request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        json: Any = None,
        content: bytes = None,
        **kwargs,
    ) -> MockHTTPResponse:
        """发送请求"""
        self._check_failure()
        self._simulate_latency()
        
        # 构建完整 URL
        if not url.startswith("http"):
            url = self._base_url + url
        
        # 记录请求
        request = MockHTTPRequest(
            method=method.upper(),
            url=url,
            headers=headers or {},
            params=params or {},
            json_data=json,
            content=content,
        )
        self._requests.append(request)
        
        # 查找匹配的路由
        for route_method, pattern, handler in self._routes:
            if route_method != method.upper():
                continue
            
            # 匹配 URL
            if self._match_url(pattern, request.path):
                return self._handle_route(handler, request)
        
        return self._default_response
    
    async def get(
        self,
        url: str,
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        **kwargs,
    ) -> MockHTTPResponse:
        """GET 请求"""
        return await self.request("GET", url, headers=headers, params=params, **kwargs)
    
    async def post(
        self,
        url: str,
        headers: Dict[str, str] = None,
        json: Any = None,
        content: bytes = None,
        **kwargs,
    ) -> MockHTTPResponse:
        """POST 请求"""
        return await self.request("POST", url, headers=headers, json=json, content=content, **kwargs)
    
    async def put(
        self,
        url: str,
        headers: Dict[str, str] = None,
        json: Any = None,
        **kwargs,
    ) -> MockHTTPResponse:
        """PUT 请求"""
        return await self.request("PUT", url, headers=headers, json=json, **kwargs)
    
    async def delete(
        self,
        url: str,
        headers: Dict[str, str] = None,
        **kwargs,
    ) -> MockHTTPResponse:
        """DELETE 请求"""
        return await self.request("DELETE", url, headers=headers, **kwargs)
    
    # ==================== 内部方法 ====================
    
    def _match_url(self, pattern: str, path: str) -> bool:
        """匹配 URL"""
        # 将路径参数转换为正则
        regex_pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', pattern)
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.match(regex_pattern, path))
    
    def _handle_route(
        self,
        handler: Union[Callable, MockHTTPResponse, Dict],
        request: MockHTTPRequest,
    ) -> MockHTTPResponse:
        """处理路由"""
        if isinstance(handler, MockHTTPResponse):
            return handler
        
        if isinstance(handler, dict):
            return MockHTTPResponse(
                status_code=handler.get("status_code", 200),
                headers=handler.get("headers", {}),
                content=json.dumps(handler.get("json", {})).encode(),
                _json=handler.get("json"),
            )
        
        if callable(handler):
            result = handler(request)
            if isinstance(result, MockHTTPResponse):
                return result
            if isinstance(result, dict):
                return MockHTTPResponse(
                    status_code=result.get("status_code", 200),
                    headers=result.get("headers", {}),
                    content=json.dumps(result.get("json", {})).encode(),
                    _json=result.get("json"),
                )
        
        return self._default_response
    
    # ==================== 工具方法 ====================
    
    def get_requests(self) -> List[MockHTTPRequest]:
        """获取所有请求记录"""
        return list(self._requests)
    
    def get_last_request(self) -> Optional[MockHTTPRequest]:
        """获取最后一个请求"""
        return self._requests[-1] if self._requests else None
    
    def clear_requests(self):
        """清空请求记录"""
        self._requests.clear()
    
    def reset(self):
        """重置客户端"""
        self._requests.clear()
        self._request_count = 0
        self._is_connected = True
