from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
