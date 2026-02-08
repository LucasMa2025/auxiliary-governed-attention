#!/bin/bash
# ============================================================
# AGA 监控 UI 启动脚本 (Linux/macOS)
# ============================================================
#
# 使用方式:
#   ./scripts/start_monitor_ui.sh              # 默认启动
#   ./scripts/start_monitor_ui.sh --port 8082  # 指定端口
#
# ============================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 默认配置
HOST="0.0.0.0"
PORT="8082"
TOKEN="aga-monitor-2026"
PORTAL_URL="http://localhost:8081"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --portal)
            PORTAL_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "AGA 监控 UI 启动脚本"
            echo ""
            echo "使用方式:"
            echo "  $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --port PORT      监听端口 (默认: 8082)"
            echo "  --host HOST      监听地址 (默认: 0.0.0.0)"
            echo "  --token TOKEN    访问口令 (默认: aga-monitor-2026)"
            echo "  --portal URL     Portal API 地址 (默认: http://localhost:8081)"
            echo "  --help, -h       显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查依赖
echo -e "${BLUE}检查依赖...${NC}"
python3 -c "import flask" 2>/dev/null || {
    echo -e "${YELLOW}安装 Flask...${NC}"
    pip install flask flask-cors
}

# 设置环境变量
export AGA_UI_HOST="$HOST"
export AGA_UI_PORT="$PORT"
export AGA_UI_TOKEN="$TOKEN"
export AGA_PORTAL_URL="$PORTAL_URL"

# 显示配置
echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  AGA 监控 UI v3.5.0${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "  地址:       ${GREEN}http://$HOST:$PORT${NC}"
echo -e "  访问口令:   ${GREEN}$TOKEN${NC}"
echo -e "  Portal:     ${GREEN}$PORTAL_URL${NC}"
echo ""
echo -e "${BLUE}==========================================${NC}"
echo ""

# 启动
echo -e "${GREEN}启动监控 UI...${NC}"
python3 -c "
from aga.monitoring import SimpleMonitorUI, SimpleUIConfig

config = SimpleUIConfig(
    host='$HOST',
    port=$PORT,
    access_token='$TOKEN',
)

ui = SimpleMonitorUI(config)
ui.run()
"
