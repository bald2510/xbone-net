@echo off
setlocal EnableExtensions

rem Run from the repository root regardless of the caller's working directory.
cd /d "%~dp0\.."

rem Set XBONE_PYTHON to override the Python executable used by this launcher.
set "PYTHON_EXEC=%XBONE_PYTHON%"
if not defined PYTHON_EXEC set "PYTHON_EXEC=python"

echo ============================================================
echo   XBone-Net Experiment Training
echo ============================================================

"%PYTHON_EXEC%" tools\training.py --train-only %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Training stopped with exit code %EXIT_CODE%.
) else (
    echo [DONE] Checkpoints were written under checkpoints\.
)

exit /b %EXIT_CODE%
