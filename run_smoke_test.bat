@echo off
rem ===============================================================================
rem XBone-Net Full Experiment Suite — Training Launcher
rem ===============================================================================

set PYTHON_EXEC=C:\Users\lebat\miniconda3\envs\Thesis\python.exe
if not exist "%PYTHON_EXEC%" set PYTHON_EXEC=python

%PYTHON_EXEC% tools\smoke_test_all.py --config-root configs --manifest tools\smoke_manifest.yaml --device cuda --batch-size 2
pause
