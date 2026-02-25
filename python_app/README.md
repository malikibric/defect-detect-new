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

### Recommended local Windows build

On Windows, run:

```bat
build_windows_exe.bat
```

It creates `dist/DefectDetect.exe`.

### Building `.exe` from macOS/Linux (recommended)

Windows `.exe` should be built on a Windows environment.
This repository includes GitHub Actions workflow:

- `.github/workflows/build-windows-exe.yml`

Run workflow **Build Windows EXE** from Actions tab (`workflow_dispatch`), then download artifact `DefectDetect-windows.zip`.

If you push a tag like `v4.2.2`, the workflow also uploads `DefectDetect-windows.zip` directly to GitHub Release for that tag.
