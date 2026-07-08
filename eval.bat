@echo off
rem ===============================================================================
rem XBone-Net Full Experiment Suite — Evaluation Launcher
rem ===============================================================================

set PYTHON_EXEC=C:\Users\lebat\miniconda3\envs\Thesis\python.exe
if not exist "%PYTHON_EXEC%" set PYTHON_EXEC=python

echo ============================================================
echo   XBone-Net Suite — Step 2: Evaluating All Experiments
echo ============================================================

"%PYTHON_EXEC%" tools/run_all.py --eval-only %*

echo.
echo ============================================================
echo   Evaluation Complete! Results exported to results/
echo ============================================================
