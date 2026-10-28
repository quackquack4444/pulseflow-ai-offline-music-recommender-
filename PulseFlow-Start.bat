
@echo off
cd /d "%~dp0"

if not exist "pulseflow-local.bat" (
    echo ERROR: pulseflow-local.bat is missing.
    echo Copy pulseflow-local.example.bat to pulseflow-local.bat
    echo and configure your local Conda path.
    pause
    exit /b 1
)

call "pulseflow-local.bat"

if not defined CONDA_ACTIVATE (
    echo ERROR: CONDA_ACTIVATE is not configured.
    pause
    exit /b 1
)

echo Starting MySQL...
net start MySQL57 >nul 2>&1

echo Activating musicrec...
call "%CONDA_ACTIVATE%" musicrec
if errorlevel 1 (
    echo ERROR: Could not activate musicrec.
    pause
    exit /b 1
)

echo Starting PulseFlow web server...
start "PulseFlow Web" cmd /k "cd /d "%~dp0" && call "%CONDA_ACTIVATE%" musicrec && python main.py web"

echo.
echo PulseFlow web server: http://127.0.0.1:7860
echo.
pause
