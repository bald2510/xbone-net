@echo off
rem ===============================================================================
rem XBone-Net Full Experiment Suite — Training Launcher
rem ===============================================================================

set PYTHON_EXEC=C:\Users\lebat\miniconda3\envs\Thesis\python.exe
if not exist "%PYTHON_EXEC%" set PYTHON_EXEC=python

echo ============================================================
echo   XBone-Net Suite — Training Launcher
echo ============================================================

"%PYTHON_EXEC%" tools/run_all.py %*

echo.
echo ============================================================
echo   Training Complete! All checkpoints saved in checkpoints/
echo   Run 'eval.bat' to evaluate and aggregate metrics.
echo ============================================================
echo.
echo [System] Shutting down the computer in 60 seconds...
echo [System] To cancel the shutdown, run: shutdown /a
shutdown /s /f /t 60
