@echo off
setlocal
set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "BOT=%~dp0src\telegram_bot.py"
if not exist "%PYTHON%" (
    echo The project's Python environment is missing. See README.md for setup.
    pause
    exit /b 1
)
if /i "%~1"=="--check" (
    "%PYTHON%" -X utf8 "%BOT%" --check
    exit /b
)
"%PYTHON%" -X utf8 "%BOT%"
set "BOT_EXIT=%ERRORLEVEL%"
if not "%BOT_EXIT%"=="0" pause
exit /b %BOT_EXIT%
