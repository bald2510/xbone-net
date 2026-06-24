@echo off
:: Batch script to run the entire XBone-Net experiment suite with Seed 42.
:: Sets up UTF-8 encoding and executes the runs.

set PYTHONUTF8=1
set PythonExe=C:\Users\lebat\miniconda3\envs\Thesis\python.exe

echo ==========================================================
echo   Starting XBone-Net Experiment Suite (Seed 42)
echo ==========================================================
echo Using Python: %PythonExe%
echo.

%PythonExe% run_all.py --seeds 42

echo.
echo ==========================================================
echo   Experiment Suite Execution Completed Successfully!
echo ==========================================================
pause
