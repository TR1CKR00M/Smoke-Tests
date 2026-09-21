from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def require_unique(frame: pd.DataFrame, column: str, path: Path) -> None:
    if frame[column].duplicated().any():
        raise ValueError(f"Column {column!r} contains duplicate values in {path}")


def require_non_null(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    counts = {column: int(frame[column].isna().sum()) for column in columns}
    bad = {column: count for column, count in counts.items() if count}
    if bad:
        raise ValueError(f"Null values found in {path}: {bad}")


def validate_numeric(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Column {column!r} in {path} is not numeric")


def require_keys(config: dict[str, Any], required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required).difference(config))
    if missing:
        raise ValueError(f"Missing config keys in {path}: {missing}")
