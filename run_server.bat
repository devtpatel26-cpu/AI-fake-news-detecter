@echo off
REM Run the AI server. Activates .venv if present.
setlocal
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)
python "ai-engine\ai_server.py"
endlocal
