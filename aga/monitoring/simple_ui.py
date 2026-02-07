"""
AGA 简单监控与知识注入 UI

提供基于 Flask 的轻量级 Web 界面：
1. 系统状态监控
2. 知识注入管理
3. 告警查看

使用固定口令认证，通过配置文件设置。

版本: v1.0
"""
import os
import json
import logging
import functools
from typing import Optional, Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template_string, request, jsonify, redirect, url_for, session
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    Flask = None


# ==================== 配置 ====================

class SimpleUIConfig:
    """简单 UI 配置"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8082,
        access_token: str = "aga-demo-token",  # 固定口令
        secret_key: str = "aga-secret-key-change-in-production",
        debug: bool = False,
    ):
        self.host = host
        self.port = port
        self.access_token = access_token
        self.secret_key = secret_key
        self.debug = debug
    
    @classmethod
    def from_env(cls) -> "SimpleUIConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("AGA_UI_HOST", "0.0.0.0"),
            port=int(os.getenv("AGA_UI_PORT", "8082")),
            access_token=os.getenv("AGA_UI_TOKEN", "aga-demo-token"),
            secret_key=os.getenv("AGA_UI_SECRET", "aga-secret-key-change-in-production"),
            debug=os.getenv("AGA_UI_DEBUG", "false").lower() == "true",
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "SimpleUIConfig":
        """从配置文件加载"""
        import yaml
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        ui_config = data.get("ui", {})
        return cls(
            host=ui_config.get("host", "0.0.0.0"),
            port=ui_config.get("port", 8082),
            access_token=ui_config.get("access_token", "aga-demo-token"),
            secret_key=ui_config.get("secret_key", "aga-secret-key-change-in-production"),
            debug=ui_config.get("debug", False),
        )


# ==================== HTML 模板 ====================

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGA 监控 - 登录</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-box {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            backdrop-filter: blur(10px);
        }
        h1 {
            color: #00d4ff;
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .subtitle {
            color: #888;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        input {
            width: 100%;
            padding: 14px 18px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            font-size: 16px;
            margin-bottom: 20px;
        }
        input:focus {
            outline: none;
            border-color: #00d4ff;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3);
        }
        .error {
            background: rgba(255, 77, 77, 0.2);
            border: 1px solid #ff4d4d;
            color: #ff4d4d;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>🔐 AGA 监控</h1>
        <p class="subtitle">Auxiliary Governed Attention</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <input type="password" name="token" placeholder="请输入访问口令" required autofocus>
            <button type="submit">登 录</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGA 监控仪表盘</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(90deg, #161b22 0%, #1a2332 100%);
            padding: 16px 24px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            color: #58a6ff;
            font-size: 20px;
        }
        .header-actions {
            display: flex;
            gap: 12px;
        }
        .header-actions a {
            color: #8b949e;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: background 0.2s;
        }
        .header-actions a:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        .header-actions a.active {
            background: #238636;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
        }
        .stat-card .label {
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .stat-card .value {
            font-size: 32px;
            font-weight: 600;
            color: #58a6ff;
        }
        .stat-card .value.success { color: #3fb950; }
        .stat-card .value.warning { color: #d29922; }
        .stat-card .value.danger { color: #f85149; }
        .section {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            margin-bottom: 24px;
        }
        .section-header {
            padding: 16px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .section-header h2 {
            font-size: 16px;
            color: #c9d1d9;
        }
        .section-body {
            padding: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }
        th {
            color: #8b949e;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-success { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
        .badge-warning { background: rgba(210, 153, 34, 0.2); color: #d29922; }
        .badge-danger { background: rgba(248, 81, 73, 0.2); color: #f85149; }
        .badge-info { background: rgba(88, 166, 255, 0.2); color: #58a6ff; }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-primary {
            background: #238636;
            color: #fff;
        }
        .btn-primary:hover {
            background: #2ea043;
        }
        .btn-danger {
            background: #da3633;
            color: #fff;
        }
        .btn-danger:hover {
            background: #f85149;
        }
        .form-group {
            margin-bottom: 16px;
        }
        .form-group label {
            display: block;
            color: #8b949e;
            font-size: 12px;
            margin-bottom: 6px;
        }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 10px 14px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
        }
        .form-group input:focus, .form-group textarea:focus {
            outline: none;
            border-color: #58a6ff;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .alert {
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 16px;
        }
        .alert-success {
            background: rgba(63, 185, 80, 0.1);
            border: 1px solid #3fb950;
            color: #3fb950;
        }
        .alert-error {
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid #f85149;
            color: #f85149;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #8b949e;
        }
        .refresh-btn {
            background: transparent;
            border: 1px solid #30363d;
            color: #8b949e;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
        }
        .refresh-btn:hover {
            background: rgba(255, 255, 255, 0.05);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 AGA 监控仪表盘</h1>
        <div class="header-actions">
            <a href="{{ url_for('dashboard') }}" class="{{ 'active' if active_tab == 'dashboard' else '' }}">仪表盘</a>
            <a href="{{ url_for('inject_page') }}" class="{{ 'active' if active_tab == 'inject' else '' }}">知识注入</a>
            <a href="{{ url_for('alerts_page') }}" class="{{ 'active' if active_tab == 'alerts' else '' }}">告警</a>
            <a href="{{ url_for('logout') }}">退出</a>
        </div>
    </div>
    
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    
    <script>
        // 自动刷新
        function autoRefresh() {
            setTimeout(() => {
                location.reload();
            }, 30000);
        }
        // autoRefresh();  // 取消注释启用自动刷新
    </script>
</body>
</html>
"""

DASHBOARD_CONTENT = """
{% extends "base" %}
{% block content %}
<div class="stats-grid">
    <div class="stat-card">
        <div class="label">服务状态</div>
        <div class="value {{ 'success' if stats.status == 'healthy' else 'danger' }}">
            {{ '运行中' if stats.status == 'healthy' else '异常' }}
        </div>
    </div>
    <div class="stat-card">
        <div class="label">活跃槽位</div>
        <div class="value">{{ stats.active_slots | default(0) }}</div>
    </div>
    <div class="stat-card">
        <div class="label">总槽位</div>
        <div class="value">{{ stats.total_slots | default(0) }}</div>
    </div>
    <div class="stat-card">
        <div class="label">总命中</div>
        <div class="value">{{ stats.total_hits | default(0) }}</div>
    </div>
    <div class="stat-card">
        <div class="label">注入次数</div>
        <div class="value">{{ stats.inject_count | default(0) }}</div>
    </div>
    <div class="stat-card">
        <div class="label">运行时间</div>
        <div class="value">{{ "%.1f" | format(stats.uptime_seconds / 3600) }}h</div>
    </div>
</div>

<div class="section">
    <div class="section-header">
        <h2>生命周期状态分布</h2>
        <button class="refresh-btn" onclick="location.reload()">🔄 刷新</button>
    </div>
    <div class="section-body">
        <table>
            <thead>
                <tr>
                    <th>状态</th>
                    <th>数量</th>
                    <th>占比</th>
                </tr>
            </thead>
            <tbody>
                {% for state, count in stats.state_distribution.items() %}
                <tr>
                    <td>
                        <span class="badge badge-{{ 'success' if state == 'confirmed' else 'warning' if state == 'probationary' else 'danger' }}">
                            {{ state }}
                        </span>
                    </td>
                    <td>{{ count }}</td>
                    <td>{{ "%.1f" | format(count / stats.total_slots * 100 if stats.total_slots > 0 else 0) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
{% endblock %}
"""

INJECT_CONTENT = """
{% extends "base" %}
{% block content %}
{% if message %}
<div class="alert alert-{{ message_type }}">{{ message }}</div>
{% endif %}

<div class="section">
    <div class="section-header">
        <h2>注入新知识</h2>
    </div>
    <div class="section-body">
        <form method="POST" action="{{ url_for('inject_knowledge') }}">
            <div class="form-row">
                <div class="form-group">
                    <label>LU ID *</label>
                    <input type="text" name="lu_id" placeholder="knowledge_001" required>
                </div>
                <div class="form-group">
                    <label>命名空间</label>
                    <input type="text" name="namespace" value="default">
                </div>
            </div>
            <div class="form-group">
                <label>触发条件 *</label>
                <textarea name="condition" rows="3" placeholder="当用户询问..." required></textarea>
            </div>
            <div class="form-group">
                <label>决策描述 *</label>
                <textarea name="decision" rows="3" placeholder="应该回答..." required></textarea>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>初始状态</label>
                    <select name="lifecycle_state">
                        <option value="probationary">试用期 (probationary)</option>
                        <option value="confirmed">已确认 (confirmed)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>信任层级</label>
                    <select name="trust_tier">
                        <option value="">未设置</option>
                        <option value="system">系统 (system)</option>
                        <option value="verified">已验证 (verified)</option>
                        <option value="user">用户 (user)</option>
                    </select>
                </div>
            </div>
            <button type="submit" class="btn btn-primary">注入知识</button>
        </form>
    </div>
</div>

<div class="section">
    <div class="section-header">
        <h2>最近注入的知识</h2>
    </div>
    <div class="section-body">
        {% if recent_knowledge %}
        <table>
            <thead>
                <tr>
                    <th>LU ID</th>
                    <th>命名空间</th>
                    <th>状态</th>
                    <th>条件</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                {% for k in recent_knowledge %}
                <tr>
                    <td>{{ k.lu_id }}</td>
                    <td>{{ k.namespace }}</td>
                    <td>
                        <span class="badge badge-{{ 'success' if k.lifecycle_state == 'confirmed' else 'warning' if k.lifecycle_state == 'probationary' else 'danger' }}">
                            {{ k.lifecycle_state }}
                        </span>
                    </td>
                    <td>{{ k.condition[:50] }}...</td>
                    <td>
                        <form method="POST" action="{{ url_for('quarantine_knowledge') }}" style="display:inline">
                            <input type="hidden" name="lu_id" value="{{ k.lu_id }}">
                            <input type="hidden" name="namespace" value="{{ k.namespace }}">
                            <button type="submit" class="btn btn-danger" onclick="return confirm('确定隔离此知识？')">隔离</button>
                        </form>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="empty-state">暂无知识记录</div>
        {% endif %}
    </div>
</div>
{% endblock %}
"""

ALERTS_CONTENT = """
{% extends "base" %}
{% block content %}
<div class="section">
    <div class="section-header">
        <h2>活跃告警</h2>
        <button class="refresh-btn" onclick="location.reload()">🔄 刷新</button>
    </div>
    <div class="section-body">
        {% if active_alerts %}
        <table>
            <thead>
                <tr>
                    <th>告警名称</th>
                    <th>严重级别</th>
                    <th>摘要</th>
                    <th>开始时间</th>
                </tr>
            </thead>
            <tbody>
                {% for alert in active_alerts %}
                <tr>
                    <td>{{ alert.name }}</td>
                    <td>
                        <span class="badge badge-{{ 'danger' if alert.severity == 'critical' else 'warning' if alert.severity == 'warning' else 'info' }}">
                            {{ alert.severity }}
                        </span>
                    </td>
                    <td>{{ alert.summary }}</td>
                    <td>{{ alert.started_at }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="empty-state">✅ 无活跃告警</div>
        {% endif %}
    </div>
</div>

<div class="section">
    <div class="section-header">
        <h2>告警历史</h2>
    </div>
    <div class="section-body">
        {% if alert_history %}
        <table>
            <thead>
                <tr>
                    <th>告警名称</th>
                    <th>严重级别</th>
                    <th>状态</th>
                    <th>时间</th>
                </tr>
            </thead>
            <tbody>
                {% for alert in alert_history %}
                <tr>
                    <td>{{ alert.name }}</td>
                    <td>
                        <span class="badge badge-{{ 'danger' if alert.severity == 'critical' else 'warning' if alert.severity == 'warning' else 'info' }}">
                            {{ alert.severity }}
                        </span>
                    </td>
                    <td>
                        <span class="badge badge-{{ 'success' if alert.state == 'resolved' else 'danger' }}">
                            {{ alert.state }}
                        </span>
                    </td>
                    <td>{{ alert.recorded_at }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="empty-state">暂无告警历史</div>
        {% endif %}
    </div>
</div>
{% endblock %}
"""


# ==================== Flask 应用 ====================

class SimpleMonitorUI:
    """
    简单监控 UI
    
    提供基于 Flask 的轻量级 Web 界面。
    
    使用示例:
    
    ```python
    from aga.monitoring import SimpleMonitorUI, SimpleUIConfig
    
    # 创建配置
    config = SimpleUIConfig(
        port=8082,
        access_token="my-secret-token",
    )
    
    # 创建 UI
    ui = SimpleMonitorUI(config)
    
    # 设置数据回调
    ui.set_stats_callback(lambda: portal_service.get_statistics())
    ui.set_inject_callback(lambda data: portal_service.inject_knowledge_text(**data))
    
    # 启动
    ui.run()
    ```
    """
    
    def __init__(self, config: Optional[SimpleUIConfig] = None):
        if not HAS_FLASK:
            raise ImportError("需要安装 Flask: pip install flask")
        
        self.config = config or SimpleUIConfig()
        self.app = Flask(__name__)
        self.app.secret_key = self.config.secret_key
        
        # 数据回调
        self._stats_callback: Optional[Callable] = None
        self._inject_callback: Optional[Callable] = None
        self._quarantine_callback: Optional[Callable] = None
        self._knowledge_list_callback: Optional[Callable] = None
        self._alert_manager = None
        
        # 注册路由
        self._register_routes()
    
    def set_stats_callback(self, callback: Callable[[], Dict[str, Any]]):
        """设置统计数据回调"""
        self._stats_callback = callback
    
    def set_inject_callback(self, callback: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """设置知识注入回调"""
        self._inject_callback = callback
    
    def set_quarantine_callback(self, callback: Callable[[str, str, str], Dict[str, Any]]):
        """设置隔离回调"""
        self._quarantine_callback = callback
    
    def set_knowledge_list_callback(self, callback: Callable[[str, int], list]):
        """设置知识列表回调"""
        self._knowledge_list_callback = callback
    
    def set_alert_manager(self, alert_manager):
        """设置告警管理器"""
        self._alert_manager = alert_manager
    
    def _require_auth(self, f):
        """认证装饰器"""
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("authenticated"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated
    
    def _register_routes(self):
        """注册路由"""
        
        @self.app.route("/login", methods=["GET", "POST"])
        def login():
            error = None
            if request.method == "POST":
                token = request.form.get("token", "")
                if token == self.config.access_token:
                    session["authenticated"] = True
                    return redirect(url_for("dashboard"))
                else:
                    error = "口令错误"
            return render_template_string(LOGIN_TEMPLATE, error=error)
        
        @self.app.route("/logout")
        def logout():
            session.pop("authenticated", None)
            return redirect(url_for("login"))
        
        @self.app.route("/")
        @self._require_auth
        def dashboard():
            stats = self._get_stats()
            content = render_template_string(
                DASHBOARD_TEMPLATE.replace("{% block content %}{% endblock %}", DASHBOARD_CONTENT),
                stats=stats,
                active_tab="dashboard",
                url_for=url_for,
            )
            return content
        
        @self.app.route("/inject")
        @self._require_auth
        def inject_page():
            recent = []
            if self._knowledge_list_callback:
                try:
                    recent = self._knowledge_list_callback("default", 10)
                except Exception as e:
                    logger.error(f"Failed to get knowledge list: {e}")
            
            message = request.args.get("message")
            message_type = request.args.get("type", "success")
            
            content = render_template_string(
                DASHBOARD_TEMPLATE.replace("{% block content %}{% endblock %}", INJECT_CONTENT),
                recent_knowledge=recent,
                message=message,
                message_type=message_type,
                active_tab="inject",
                url_for=url_for,
            )
            return content
        
        @self.app.route("/inject", methods=["POST"])
        @self._require_auth
        def inject_knowledge():
            if not self._inject_callback:
                return redirect(url_for("inject_page", message="注入功能未配置", type="error"))
            
            try:
                data = {
                    "lu_id": request.form.get("lu_id"),
                    "condition": request.form.get("condition"),
                    "decision": request.form.get("decision"),
                    "namespace": request.form.get("namespace", "default"),
                    "lifecycle_state": request.form.get("lifecycle_state", "probationary"),
                    "trust_tier": request.form.get("trust_tier") or None,
                }
                
                result = self._inject_callback(data)
                
                if result.get("success"):
                    return redirect(url_for("inject_page", message=f"知识 {data['lu_id']} 注入成功", type="success"))
                else:
                    return redirect(url_for("inject_page", message=result.get("error", "注入失败"), type="error"))
            
            except Exception as e:
                logger.error(f"Inject error: {e}")
                return redirect(url_for("inject_page", message=str(e), type="error"))
        
        @self.app.route("/quarantine", methods=["POST"])
        @self._require_auth
        def quarantine_knowledge():
            if not self._quarantine_callback:
                return redirect(url_for("inject_page", message="隔离功能未配置", type="error"))
            
            try:
                lu_id = request.form.get("lu_id")
                namespace = request.form.get("namespace", "default")
                
                result = self._quarantine_callback(lu_id, "UI 手动隔离", namespace)
                
                if result.get("success"):
                    return redirect(url_for("inject_page", message=f"知识 {lu_id} 已隔离", type="success"))
                else:
                    return redirect(url_for("inject_page", message=result.get("error", "隔离失败"), type="error"))
            
            except Exception as e:
                logger.error(f"Quarantine error: {e}")
                return redirect(url_for("inject_page", message=str(e), type="error"))
        
        @self.app.route("/alerts")
        @self._require_auth
        def alerts_page():
            active_alerts = []
            alert_history = []
            
            if self._alert_manager:
                active_alerts = self._alert_manager.get_active_alerts()
                alert_history = self._alert_manager.get_alert_history(limit=50)
            
            content = render_template_string(
                DASHBOARD_TEMPLATE.replace("{% block content %}{% endblock %}", ALERTS_CONTENT),
                active_alerts=active_alerts,
                alert_history=alert_history,
                active_tab="alerts",
                url_for=url_for,
            )
            return content
        
        # API 端点
        @self.app.route("/api/stats")
        @self._require_auth
        def api_stats():
            return jsonify(self._get_stats())
        
        @self.app.route("/api/health")
        def api_health():
            return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})
    
    def _get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        if self._stats_callback:
            try:
                stats = self._stats_callback()
                stats["status"] = "healthy"
                return stats
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {"status": "error", "error": str(e)}
        
        return {
            "status": "healthy",
            "active_slots": 0,
            "total_slots": 0,
            "total_hits": 0,
            "inject_count": 0,
            "uptime_seconds": 0,
            "state_distribution": {},
        }
    
    def run(self, **kwargs):
        """启动 UI 服务"""
        logger.info(f"Starting AGA Monitor UI on http://{self.config.host}:{self.config.port}")
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            **kwargs,
        )


# 导出
__all__ = [
    "SimpleUIConfig",
    "SimpleMonitorUI",
]
