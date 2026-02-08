@echo off
REM ============================================================
REM AGA 监控 UI 启动脚本 (Windows)
REM ============================================================
REM
REM 使用方式:
REM   scripts\start_monitor_ui.bat              默认启动
REM   scripts\start_monitor_ui.bat --port 8082  指定端口
REM
REM ============================================================

setlocal enabledelayedexpansion

REM 默认配置
set HOST=0.0.0.0
set PORT=8082
set TOKEN=aga-monitor-2026
set PORTAL_URL=http://localhost:8081

REM 解析参数
:parse_args
if "%~1"=="" goto :start
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--token" (
    set TOKEN=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--portal" (
    set PORTAL_URL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help
shift
goto :parse_args

:help
echo AGA 监控 UI 启动脚本
echo.
echo 使用方式:
echo   %~nx0 [选项]
echo.
echo 选项:
echo   --port PORT      监听端口 (默认: 8082)
echo   --host HOST      监听地址 (默认: 0.0.0.0)
echo   --token TOKEN    访问口令 (默认: aga-monitor-2026)
echo   --portal URL     Portal API 地址 (默认: http://localhost:8081)
echo   --help, -h       显示帮助
goto :eof

:start
REM 检查依赖
echo 检查依赖...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo 安装 Flask...
    pip install flask flask-cors
)

REM 设置环境变量
set AGA_UI_HOST=%HOST%
set AGA_UI_PORT=%PORT%
set AGA_UI_TOKEN=%TOKEN%
set AGA_PORTAL_URL=%PORTAL_URL%

REM 显示配置
echo.
echo ==========================================
echo   AGA 监控 UI v3.5.0
echo ==========================================
echo.
echo   地址:       http://%HOST%:%PORT%
echo   访问口令:   %TOKEN%
echo   Portal:     %PORTAL_URL%
echo.
echo ==========================================
echo.

REM 启动
echo 启动监控 UI...
python -c "from aga.monitoring import SimpleMonitorUI, SimpleUIConfig; config = SimpleUIConfig(host='%HOST%', port=%PORT%, access_token='%TOKEN%'); ui = SimpleMonitorUI(config); ui.run()"
