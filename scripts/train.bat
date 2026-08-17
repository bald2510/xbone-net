@echo off
setlocal EnableExtensions

rem Run from the repository root regardless of the caller's working directory.
cd /d "%~dp0\.."

rem Set XBONE_PYTHON to override the Python executable used by this launcher.
set "PYTHON_EXEC=%XBONE_PYTHON%"
if not defined PYTHON_EXEC if exist "%USERPROFILE%\miniconda3\envs\Thesis\python.exe" set "PYTHON_EXEC=%USERPROFILE%\miniconda3\envs\Thesis\python.exe"
if not defined PYTHON_EXEC set "PYTHON_EXEC=python"

rem Allow the shutdown countdown to be overridden, for example:
rem set XBONE_SHUTDOWN_DELAY=300
set "SHUTDOWN_DELAY=%XBONE_SHUTDOWN_DELAY%"
if not defined SHUTDOWN_DELAY set "SHUTDOWN_DELAY=60"
set "SHUTDOWN_ENABLED=1"
if /I "%XBONE_NO_SHUTDOWN%"=="1" set "SHUTDOWN_ENABLED=0"
for %%A in (%*) do (
    if /I "%%~A"=="--dry-run" set "SHUTDOWN_ENABLED=0"
    if /I "%%~A"=="--list-configs" set "SHUTDOWN_ENABLED=0"
    if /I "%%~A"=="--table" set "SHUTDOWN_ENABLED=0"
)

echo ============================================================
echo   XBone-Net Experiment Training
echo ============================================================
echo Python: %PYTHON_EXEC%
echo Experiments: tools\experiments.txt
if "%SHUTDOWN_ENABLED%"=="1" (
    echo The computer will shut down %SHUTDOWN_DELAY% seconds after success.
) else (
    echo Automatic shutdown is disabled for this run.
)
echo.

rem Run both training and evaluation so every completed seed has metrics.json.
rem Extra arguments passed to train.bat are forwarded to tools\training.py.
"%PYTHON_EXEC%" tools\training.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Training stopped with exit code %EXIT_CODE%.
    echo [SAFEGUARD] Automatic shutdown was cancelled because the run failed.
    exit /b %EXIT_CODE%
)

echo [DONE] Experiment command completed successfully.
echo [DONE] Checkpoints: checkpoints\
echo [DONE] Metrics: results\

if "%SHUTDOWN_ENABLED%"=="0" (
    echo [SHUTDOWN] Automatic shutdown is disabled for this run.
    exit /b 0
)

echo [SHUTDOWN] Scheduling Windows shutdown in %SHUTDOWN_DELAY% seconds.
echo [SHUTDOWN] Run "shutdown /a" in another terminal to cancel.

shutdown.exe /s /t %SHUTDOWN_DELAY% /c "XBone-Net experiments completed successfully."
set "SHUTDOWN_EXIT_CODE=%ERRORLEVEL%"
if not "%SHUTDOWN_EXIT_CODE%"=="0" (
    echo [ERROR] Windows shutdown could not be scheduled.
    exit /b %SHUTDOWN_EXIT_CODE%
)

exit /b 0
