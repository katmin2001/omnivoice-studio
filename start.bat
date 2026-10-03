@echo off
title OmniVoice Studio
cd /d "%~dp0"
start "" /b cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8765"
".venv\Scripts\python.exe" server.py %*
pause
