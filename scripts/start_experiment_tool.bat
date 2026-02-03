@echo off
REM AGA Experiment Tool - Windows Launcher
REM 
REM Usage:
REM   start_experiment_tool.bat [port]

set PORT=%1
if "%PORT%"=="" set PORT=8765

echo ==========================================
echo AGA Experiment Tool Launcher v3.1
echo ==========================================
echo Port: %PORT%
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Show Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python Version: %PYTHON_VERSION%
echo.

REM Check dependencies
echo Checking dependencies...

pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask
)

pip show flask-cors >nul 2>&1
if errorlevel 1 (
    echo Installing Flask-CORS...
    pip install flask-cors
)

pip show torch >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch
)

pip show transformers >nul 2>&1
if errorlevel 1 (
    echo Installing Transformers...
    pip install transformers
)

pip show pyyaml >nul 2>&1
if errorlevel 1 (
    echo Installing PyYAML...
    pip install pyyaml
)

pip show aiosqlite >nul 2>&1
if errorlevel 1 (
    echo Installing aiosqlite...
    pip install aiosqlite
)

echo.
echo ==========================================
echo Starting AGA Experiment Tool...
echo ==========================================
echo Access at: http://localhost:%PORT%
echo Default password: aga_experiment_2026
echo.
echo Features:
echo   - Zero-training knowledge injection
echo   - Lifecycle management
echo   - Entropy gating
echo   - Multi-adapter persistence
echo.

cd %~dp0..
python -m aga_experiment_tool.app --port %PORT%

pause
