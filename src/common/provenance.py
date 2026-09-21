from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def runtime_metadata() -> dict[str, str]:
    import platform

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def file_manifest(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
