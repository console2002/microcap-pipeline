@echo off
:: Always work from this script's folder
cd /d "%~dp0"

:: Activate the existing venv
call venv\Scripts\activate

echo.
echo [INFO] Virtual environment is active.
echo You can now run:
echo     python gui.py
echo     python run_weekly.py
echo     python run_daily.py
echo.

:: Keep the window open so you can work in the venv
cmd /k
