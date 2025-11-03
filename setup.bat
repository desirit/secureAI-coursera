@echo off
REM SecureAI Course - Complete Setup Script (Windows)
REM Run this to set up your environment from scratch

echo =========================================
echo SecureAI Course - Environment Setup
echo =========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
echo Python OK
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo venv directory already exists.
    set /p response="Remove it? (y/n): "
    if /i "%response%"=="y" (
        rmdir /s /q venv
        echo Removed old venv
    )
)

if not exist venv (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Using existing virtual environment
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo pip upgraded
echo.

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
echo Requirements installed
echo.

REM Check Ollama
echo Checking Ollama installation...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo Ollama is installed
    
    REM Check if model is available
    ollama list | findstr "llama2-uncensored" >nul
    if %errorlevel% equ 0 (
        echo llama2-uncensored model is installed
    ) else (
        echo WARNING: llama2-uncensored not found
        echo Run: ollama pull llama2-uncensored
    )
) else (
    echo WARNING: Ollama not installed
    echo Install from: https://ollama.ai
    echo Required for: m1-v2-1.py (Prompt Injection demo)
    echo Other demos work without Ollama
)
echo.

REM Test imports
echo Testing Python imports...
python -c "import streamlit, requests, pandas, numpy, plotly; from PIL import Image; print('All core dependencies import successfully')"
echo.

REM Summary
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Project structure:
echo    venv\              - Virtual environment
echo    m1-v2-1.py         - Prompt Injection (needs Ollama)
echo    m1-v2-2.py         - Model Extraction
echo    m1-v3-1.py         - STRIDE Threat Modeling
echo    m1-v3-2.py         - MITRE ATLAS Framework
echo    m2-v2.py           - Integration Testing
echo    m2-v3.py           - Adversarial Testing
echo    m3-v2.py           - CI/CD Security Gates
echo    m3-v3.py           - Monitoring Dashboard
echo.
echo Quick start:
echo    venv\Scripts\activate
echo    streamlit run m1-v2-1.py
echo.
echo For more info, see README.md
echo.
pause
