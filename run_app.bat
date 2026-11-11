@echo off
setlocal enabledelayedexpansion
title RayXplain - Multi-Modal Pulmonary Diagnostic Assistant
color 0B

echo ==============================================================================
echo   RayXplain: Multi-Modal Pulmonary Disease Diagnostic Assistant
echo   Dual-Branch Deep Learning (ResNet-50 + Clinical MLP) with Grad-CAM XAI
echo ==============================================================================
echo.

:: 1. Environment & Python Verification
echo [1/4] Verifying Python Environment...
if exist ".venv\Scripts\activate.bat" (
    echo       Found virtual environment: .venv
    call .venv\Scripts\activate.bat
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo       Notice: .venv not found locally. Using system Python...
    set "PYTHON_EXE=python"
)

:: Validate Python availability
%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Python was not found on PATH or in .venv.
    echo Please install Python 3.11 and ensure it is added to your PATH.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('%PYTHON_EXE% -c "import sys; print(f'Python {sys.version.split()[0]} ({sys.executable})')"') do (
    echo       Using: %%i
)
echo.

:: 2. Hardware Acceleration Check (CUDA / CPU)
echo [2/4] Inspecting Hardware Acceleration...
%PYTHON_EXE% -c "import torch; dev = 'CUDA GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (CUDA not detected - running on CPU)'; print(f'      Active Compute Device: {dev}')"
echo.

:: 3. Launching Web Dashboard
echo [3/4] Initializing RayXplain Web Application...
echo       Host URL : http://127.0.0.1:7860
echo       Dashboard: Dual-Column Clinical Intake ^& Grad-CAM Explainability
echo.

:: 4. Background browser open timer (waits 3 seconds then opens default browser)
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:7860"

echo [4/4] Starting server process (Press Ctrl+C to stop)...
echo ==============================================================================
echo.

%PYTHON_EXE% app.py

echo.
echo ==============================================================================
echo   RayXplain session ended.
echo ==============================================================================
pause

