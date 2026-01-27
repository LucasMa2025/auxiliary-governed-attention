@echo off
REM AGA Experiment Tool - Windows Launcher
REM 
REM Usage:
REM   start_experiment_tool.bat [port]

set PORT=%1
if "%PORT%"=="" set PORT=8765

echo ==========================================
echo AGA Experiment Tool Launcher
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

REM Check dependencies
echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask
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

echo.
echo Starting AGA Experiment Tool...
echo Access at: http://localhost:%PORT%
echo Default password: aga_experiment_2026
echo.

cd %~dp0..
python -m aga_experiment_tool.app --port %PORT%

pause

