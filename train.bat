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
