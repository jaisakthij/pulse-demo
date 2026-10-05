@echo off
REM PULSE Streamlit Launcher
REM Double-click this file to start the PULSE app

echo Starting PULSE Longitudinal Multimodal Demo...
echo.

cd /d "%~dp0"

REM Find Python
for /f "delims=" %%i in ('where python 2^>nul') do set PYTHON=%%i
if not defined PYTHON (
    echo Python not found in PATH. Trying Hermes Python...
    for /d %%d in ("C:\Users\jaisa\AppData\Local\hermes\tools\python*") do set PYTHON=%%d\python.exe
)

if not exist "%PYTHON%" (
    echo ERROR: Python not found. Please install Python.
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
echo.

REM Check dependencies
"%PYTHON%" -c "import streamlit, numpy, pandas, torch" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    "%PYTHON%" -m pip install -q -r requirements.txt
)

echo.
echo ========================================
echo   PULSE App Starting
echo   URL: http://127.0.0.1:8504
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

"%PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8504

pause
