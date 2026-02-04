# DefectDetect (Python/PySide6)

## Run locally
1. Create a virtual env.
2. Install deps.
3. Run the app.

```bash
python -m venv .venv
. .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Build Windows .exe (PyInstaller)
```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed \
  --name DefectDetect \
  --add-data "resources;resources" \
  --icon resources/logo-mini.png \
  app.py
```

The .exe will appear in the `dist/` folder.

Alternative (spec file):

```bash
pyinstaller DefectDetect.spec
```
