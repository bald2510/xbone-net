@echo off
rem ===============================================================================
rem XBone-Net Full Experiment Suite — Evaluation Launcher
rem ===============================================================================
rem Executes evaluate.py across all trained checkpoints and aggregates metrics.
rem Generates metrics.json and aggregated_results.json in results/.
rem
rem Usage:
rem   eval.bat                   (Evaluate ALL experiments)
rem   eval.bat -g btxrd          (Evaluate BTXRD experiments only)
rem   eval.bat --table           (Print summary table from existing results)
rem ===============================================================================

echo ============================================================
echo   XBone-Net Suite — Step 2: Evaluating All Experiments
echo ============================================================

python tools/run_all.py --eval-only %*

echo.
echo ============================================================
echo   Evaluation Complete! Results exported to results/
echo ============================================================
