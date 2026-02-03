@echo off
REM AGA Portal 启动脚本 (Windows)
REM
REM 使用方式:
REM   scripts\start_portal.bat                 开发模式
REM   scripts\start_portal.bat --prod          生产模式
REM

setlocal enabledelayedexpansion

REM 默认配置
set HOST=0.0.0.0
set PORT=8081
set WORKERS=1
set RELOAD=
set LOG_LEVEL=info
set PERSISTENCE=sqlite
set MESSAGING=memory
set REDIS_HOST=localhost

REM 解析参数
:parse_args
if "%~1"=="" goto :start
if "%~1"=="--prod" (
    set WORKERS=4
    set LOG_LEVEL=warning
    set MESSAGING=redis
    shift
    goto :parse_args
)
if "%~1"=="--dev" (
    set WORKERS=1
    set RELOAD=--reload
    set LOG_LEVEL=debug
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--redis" (
    set REDIS_HOST=%~2
    set MESSAGING=redis
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help

shift
goto :parse_args

:help
echo AGA Portal 启动脚本
echo.
echo 使用方式:
echo   %~nx0 [选项]
echo.
echo 选项:
echo   --dev              开发模式 (自动重载)
echo   --prod             生产模式 (多 worker)
echo   --host HOST        监听地址 (默认: 0.0.0.0)
echo   --port PORT        监听端口 (默认: 8081)
echo   --redis HOST       启用 Redis (消息同步)
echo   --help, -h         显示帮助
goto :eof

:start
REM 检查依赖
echo 检查依赖...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo 错误: 缺少依赖。请运行: pip install fastapi uvicorn
    exit /b 1
)

REM 显示配置
echo.
echo ==========================================
echo   AGA Portal 启动配置
echo ==========================================
echo   地址:     %HOST%:%PORT%
echo   进程数:   %WORKERS%
echo   日志级别: %LOG_LEVEL%
echo   持久化:   %PERSISTENCE%
echo   消息队列: %MESSAGING%
if "%MESSAGING%"=="redis" (
    echo   Redis:    %REDIS_HOST%
)
echo ==========================================
echo.

REM 构建命令
set CMD=python -m aga.portal.app
set CMD=%CMD% --host %HOST%
set CMD=%CMD% --port %PORT%
set CMD=%CMD% --workers %WORKERS%
set CMD=%CMD% --persistence %PERSISTENCE%
set CMD=%CMD% --messaging %MESSAGING%

if "%MESSAGING%"=="redis" (
    set CMD=%CMD% --redis-host %REDIS_HOST%
)

if not "%RELOAD%"=="" (
    set CMD=%CMD% %RELOAD%
)

REM 启动
echo 启动 Portal...
%CMD%
