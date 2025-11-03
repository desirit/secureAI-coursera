@echo off
REM SecureAI Course - Demo Launcher (Windows)
REM Quick way to run any demo

REM Activate venv if it exists
if exist venv (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo WARNING: No virtual environment found. Run setup.bat first
    pause
    exit /b 1
)

:menu
cls
echo =========================================
echo    SecureAI Course - Demo Launcher
echo =========================================
echo.
echo Module 1: Understanding AI Threat Models
echo   1) m1-v2-1.py - Prompt Injection (needs Ollama)
echo   2) m1-v2-2.py - Model Extraction
echo   3) m1-v3-1.py - STRIDE Threat Modeling
echo   4) m1-v3-2.py - MITRE ATLAS Framework
echo.
echo Module 2: Creating Security Test Cases
echo   5) m2-v2.py - Integration Testing
echo   6) m2-v3.py - Adversarial Testing
echo.
echo Module 3: CI/CD Integration
echo   7) m3-v2.py - CI/CD Security Gates
echo   8) m3-v3.py - Monitoring Dashboard
echo.
echo   0) Exit
echo.
set /p choice="Select demo to run (1-8, 0 to exit): "

if "%choice%"=="1" goto demo1
if "%choice%"=="2" goto demo2
if "%choice%"=="3" goto demo3
if "%choice%"=="4" goto demo4
if "%choice%"=="5" goto demo5
if "%choice%"=="6" goto demo6
if "%choice%"=="7" goto demo7
if "%choice%"=="8" goto demo8
if "%choice%"=="0" goto exit
echo Invalid choice
pause
goto menu

:demo1
echo Launching Prompt Injection Demo...
streamlit run m1-v2-1.py
goto menu

:demo2
echo Launching Model Extraction Demo...
streamlit run m1-v2-2.py
goto menu

:demo3
echo Launching STRIDE Demo...
streamlit run m1-v3-1.py
goto menu

:demo4
echo Launching MITRE ATLAS Demo...
streamlit run m1-v3-2.py
goto menu

:demo5
echo Launching Integration Testing Demo...
streamlit run m2-v2.py
goto menu

:demo6
echo Launching Adversarial Testing Demo...
streamlit run m2-v3.py
goto menu

:demo7
echo Launching CI/CD Gates Demo...
streamlit run m3-v2.py
goto menu

:demo8
echo Launching Monitoring Dashboard...
streamlit run m3-v3.py
goto menu

:exit
echo Goodbye!
exit /b 0
