# Script to run the entire XBone-Net experiment suite with Seed 42.
# Sets up UTF-8 encoding and runs training + evaluation for all configs.

$env:PYTHONUTF8="1"
$PythonExe = "C:\Users\lebat\miniconda3\envs\Thesis\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting XBone-Net Experiment Suite (Seed 42)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Using Python: $PythonExe"
Write-Host ""

& $PythonExe run_all.py --seeds 42

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Experiment Suite Execution Completed Successfully!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
