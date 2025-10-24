@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ================== CONFIG ==================
:: Required Python major/minor
set "REQ_MAJOR=3"
set "REQ_MINOR=11"

:: Name of the virtual environment folder
set "VENV_DIR=venv"

:: Log file
set "LOGFILE=%~dp0setup_env.log"
:: ================== CONFIG ==================

:: Always run from the script's own folder (safe even if launched from elsewhere)
cd /d "%~dp0"

call :init_log
call :log "=== Microcap Pipeline - Setup ==="
call :log "Project dir: %CD%"
call :log ""

echo.
echo [STEP] Checking Python...
call :check_python
if errorlevel 1 goto end_fail

echo.
echo [STEP] Creating virtual environment "%VENV_DIR%" (if not already present)...
call :create_venv
if errorlevel 1 goto end_fail

echo.
echo [STEP] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 goto end_fail

echo.
echo [STEP] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto end_fail

echo.
echo [STEP] Installing requirements from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 goto end_fail

echo.
echo [STEP] Ensuring data/ and logs/ folders exist...
if not exist "data" mkdir "data"
if not exist "logs" mkdir "logs"

:: Make sure runlog.csv and errorlog.csv headers exist so GUI doesn't choke on first open
if not exist "logs\runlog.csv" (
    echo timestamp,module,rows_added,duration_ms,note> "logs\runlog.csv"
)
if not exist "logs\errorlog.csv" (
    echo timestamp,module,message> "logs\errorlog.csv"
)

echo.
echo [INFO] Setup complete.
echo [INFO] Virtual environment is ACTIVE in this window.
echo [INFO] You can now run:
echo            python gui.py
echo or:
echo            python run_weekly.py
echo            python run_daily.py

call :log ""
call :log "[SUCCESS] Environment ready. venv active."
call :log "You can now run python gui.py"

title Microcap Pipeline (venv ready)

:: Keep this shell open with venv already activated
cmd /k


:: ========== FUNCTIONS ==========

:init_log
    if exist "%LOGFILE%" del "%LOGFILE%" >nul 2>&1
    echo [%date% %time%] microcap-pipeline setup log > "%LOGFILE%"
    exit /b 0

:log
    >> "%LOGFILE%" echo [%date% %time%] %*
    exit /b 0

:check_python
    call :log "Checking python on PATH"

    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found on PATH.
        echo         Install Python 3.11+ and run setup_env.bat again.
        call :log "ERROR: python.exe not found"
        exit /b 1
    )

    for /f "usebackq tokens=*" %%V in (`python -c "import sys;print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))"`) do (
        set "PYVER=%%V"
    )

    for /f "tokens=1,2 delims=." %%A in ("%PYVER%") do (
        set "PYMAJOR=%%A"
        set "PYMINOR=%%B"
    )

    echo [INFO] Found Python version %PYMAJOR%.%PYMINOR%
    call :log "Python version %PYMAJOR%.%PYMINOR%"

    :: Must be >= REQ_MAJOR.REQ_MINOR, i.e. >= 3.11
    if %PYMAJOR% LSS %REQ_MAJOR% (
        echo [ERROR] Python too old. Need %REQ_MAJOR%.%REQ_MINOR% or newer.
        call :log "ERROR: Python major too old"
        exit /b 1
    )

    if %PYMAJOR% GTR %REQ_MAJOR% (
        exit /b 0
    )

    :: same major, compare minor
    if %PYMINOR% LSS %REQ_MINOR% (
        echo [ERROR] Python too old. Need %REQ_MAJOR%.%REQ_MINOR% or newer.
        call :log "ERROR: Python minor too old"
        exit /b 1
    )

    exit /b 0

:create_venv
    if exist "%VENV_DIR%\Scripts\activate" (
        echo [INFO] venv already exists, skipping create.
        call :log "venv exists, reusing %VENV_DIR%"
        exit /b 0
    )

    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv "%VENV_DIR%".
        call :log "ERROR: python -m venv failed"
        exit /b 1
    )

    call :log "venv created at %VENV_DIR%"
    exit /b 0


:end_fail
echo.
echo [ERROR] Setup failed. See log file:
echo         "%LOGFILE%"
echo.
echo The shell will stay open for inspection.
cmd /k
