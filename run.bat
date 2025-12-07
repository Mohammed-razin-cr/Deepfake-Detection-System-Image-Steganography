@echo off
REM Deepfake Detection System - Windows Startup Script

setlocal enabledelayedexpansion

echo.
echo ================================================
echo   Deepfake Detection System - Startup Script
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

echo.
echo [2/5] Checking virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists
)

echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/5] Installing/updating dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!

echo.
echo [5/5] Starting Flask application...
echo.
echo ================================================
echo   Application starting on http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ================================================
echo.

cd web_app
python app.py

pause
