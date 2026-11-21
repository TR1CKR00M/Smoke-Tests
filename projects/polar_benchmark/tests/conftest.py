"""Make the polar_benchmark package and scripts importable from repository-root pytest."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "script"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
