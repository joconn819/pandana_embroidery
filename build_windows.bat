@echo off
setlocal

REM Run this script from a Windows machine; PyInstaller does not reliably
REM cross-compile a Windows executable from Linux.
py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name PandanaEmbroidery ^
  --collect-all rembg ^
  --collect-all skimage ^
  app.py

echo.
echo Built dist\PandanaEmbroidery.exe
echo The first run downloads the rembg model weights.
endlocal
