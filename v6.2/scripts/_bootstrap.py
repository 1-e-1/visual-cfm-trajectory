"""Make the isolated source package importable without installation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT
SOURCE_ROOT = PROJECT_ROOT / "src"

for path in (str(SOURCE_ROOT), str(WORKSPACE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

