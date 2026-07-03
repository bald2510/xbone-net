@echo off
rem ===============================================================================
rem XBone-Net Full Experiment Suite — Training Launcher
rem ===============================================================================
rem Executes train.py sequentially across all experiments and seeds in configs.
rem Saves trained checkpoints to checkpoints/{experiment}/seed_{seed}/.
rem
rem Usage:
rem   train.bat                  (Train ALL experiments)
rem   train.bat -g btxrd         (Train BTXRD experiments only)
rem   train.bat -g btxrd_proposed (Train proposed model only)
rem ===============================================================================

echo ============================================================
echo   XBone-Net Suite — Step 1: Training All Experiments
echo ============================================================

python tools/run_all.py --train-only %*

echo.
echo ============================================================
echo   Training Complete! All checkpoints saved in checkpoints/
echo   Run 'eval.bat' to evaluate and aggregate metrics.
echo ============================================================
