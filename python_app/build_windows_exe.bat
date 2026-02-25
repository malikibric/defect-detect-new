@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment...
  py -m venv .venv
)

call ".venv\Scripts\activate"
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

echo [INFO] Building DefectDetect.exe ...
pyinstaller --noconfirm --clean DefectDetect.spec

if exist "dist\DefectDetect.exe" (
  echo [OK] Build completed: %cd%\dist\DefectDetect.exe
) else (
  echo [ERROR] Build failed.
  exit /b 1
)

endlocal
