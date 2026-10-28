@echo off
REM VisionPath one-time setup for Windows (run from the project folder)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.9 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist config.json copy config.sample.json config.json >nul

echo.
echo Downloading detection model...
python scripts\download_model.py
if errorlevel 1 echo (Model download needs attention - see the messages above and models\README.md)

echo.
echo Running automated tests...
python -m unittest discover -s tests

echo.
echo Setup finished. Start the app with run.bat or:  python main.py
pause
exit /b 0

:fail
echo Setup failed. See the messages above.
pause
exit /b 1
