@echo off
rem HyperCore-Tune\max_tune.bat - Windows 10/11 entry point for the Bayesian (Optuna) sweep.
rem
rem   max_tune.bat --model q2_0 --gen      --cntx 131072
rem   max_tune.bat --model q2_0 --prefill  --cntx 131072 --trials 60
rem   max_tune.bat --model q2_0 --latency  --mode staged --dry-run
rem   max_tune.bat --show-project
rem   max_tune.bat --install
rem
rem Dimensions: vram pcie ple-io kv kvgrow hostcore affinity worker tail draft-floor
rem Freeze any of them with the matching flag (--pcie, --kv, --draft-floor, ...).
rem --trials counts unseen configurations; repeats in history are redrawn.
rem --prompt-cache-tail is never written into the deliverable.
rem
rem This app is standalone. It talks to a Strata checkout found from this folder's
rem location (nested as <Strata>\HyperCore-Tune, or beside one as %USERPROFILE%\HyperCore-CPP).
rem Every path the sweep writes is absolute.
setlocal EnableExtensions
title HyperCore-Tune
cd /d "%~dp0"
set "HERE=%~dp0"

call :findpy
if not defined PY (
    echo max_tune.bat: no python interpreter found.
    echo   Install 64-bit Python 3.10+ from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH", then run: max_tune.bat --install
    pause
    exit /b 1
)

if /i "%~1"=="--install" goto install
if /i "%~1"=="--show-project" goto show

"%PY%" -c "import optuna" >nul 2>nul
if errorlevel 1 (
    echo max_tune.bat: %PY% cannot import optuna. Install it:
    echo     "%PY%" -m pip install -r "%HERE%requirements.txt"
    echo   or run: max_tune.bat --install
    pause
    exit /b 1
)

"%PY%" "%HERE%bayesian_tune.py" %*
exit /b %ERRORLEVEL%

:install
echo [*] installing optuna into %PY%
"%PY%" -m pip install --quiet -r "%HERE%requirements.txt"
if errorlevel 1 (
    echo pip install failed
    pause
    exit /b 1
)
"%PY%" -c "import optuna; print('optuna', optuna.__version__)"
exit /b 0

:show
"%PY%" "%HERE%strata_env.py"
exit /b %ERRORLEVEL%

:findpy
if defined TUNE_PYTHON if exist "%TUNE_PYTHON%" (
    set "PY=%TUNE_PYTHON%"
    goto :eof
)
if exist "%HERE%..\.venv\Scripts\python.exe" (
    set "PY=%HERE%..\.venv\Scripts\python.exe"
    goto :eof
)
if exist "%HERE%.venv\Scripts\python.exe" (
    set "PY=%HERE%.venv\Scripts\python.exe"
    goto :eof
)
if exist "%HERE%..\Strata\.venv\Scripts\python.exe" (
    set "PY=%HERE%..\Strata\.venv\Scripts\python.exe"
    goto :eof
)
if exist "%HERE%..\HyperCore\.venv\Scripts\python.exe" (
    set "PY=%HERE%..\HyperCore\.venv\Scripts\python.exe"
    goto :eof
)
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)"') do set "PY=%%I"
    goto :eof
)
where python >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PY=%%I"
)
goto :eof
