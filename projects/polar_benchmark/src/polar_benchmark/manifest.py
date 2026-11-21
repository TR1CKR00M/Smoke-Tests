"""Build the data manifest and model registry for ``projects/polar_benchmark``."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from .vasp_io import CHARGE_LABELS, parse_outcar

IGNORED_NAMES = {".DS_Store", "DATA_MANIFEST.csv"}
DISTORTION_RE = re.compile(r"^D_-?\d+$")
SPECIAL_DISTORTIONS = {"Rattled", "Unperturbed"}
DEFECT_RE = re.compile(r"V_?(Sb|Se)(\d+)")
HASH_CHUNK_BYTES = 1 << 20

MANIFEST_COLUMNS = [
    "path",
    "stage",
    "defect",
    "charge_label",
    "charge_from_label",
    "distortion",
    "file_type",
    "size_bytes",
    "sha256",
    "nelect",
    "charge_from_nelect",
    "charge_consistent",
    "calc_type",
    "converged",
    "unconverged_reason",
    "n_ionic_steps",
    "final_energy_eV",
    "final_magnetization",
    "functional",
    "dispersion",
    "ivdw",
    "kpoint_grid",
    "encut_eV",
    "ispin",
    "vasp_version",
]

REGISTRY_HASH_COLUMNS = ["path", "size_bytes", "sha256"]
REGISTRY_TODO_COLUMNS = [
    "role",
    "source_url",
    "version",
    "license",
    "training_config",
    "training_log",
    "notes",
]
REGISTRY_COLUMNS = REGISTRY_HASH_COLUMNS + REGISTRY_TODO_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_parts(parts: Iterable[str]) -> dict[str, str]:
    """Infer stage/defect/charge/distortion from the path components below ``data/``."""
    parts = list(parts)
    info = {"stage": parts[0] if parts else "", "defect": "", "charge_label": "", "distortion": ""}
    for part in parts[:-1]:
        defect = DEFECT_RE.search(part)
        if defect and not info["defect"]:
            info["defect"] = f"V_{defect.group(1)}{defect.group(2)}"
        for token in part.split("_"):
            if token in CHARGE_LABELS and not info["charge_label"]:
                info["charge_label"] = token
        if DISTORTION_RE.match(part) or part in SPECIAL_DISTORTIONS:
            info["distortion"] = part
    return info


def file_type(name: str) -> str:
    for prefix in ("OUTCAR", "CONTCAR", "POSCAR", "INCAR", "KPOINTS"):
        if name.startswith(prefix):
            return prefix
    return Path(name).suffix.lstrip(".") or name


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def manifest_row(path: Path, data_root: Path, relative_to: Path) -> dict[str, str]:
    """One manifest row for ``path``; OUTCAR files get the parsed DFT metadata."""
    info = classify_parts(path.relative_to(data_root).parts)
    kind = file_type(path.name)
    label_charge = CHARGE_LABELS.get(info["charge_label"])
    row: dict[str, object] = {
        "path": path.relative_to(relative_to).as_posix(),
        **info,
        "charge_from_label": label_charge,
        "file_type": kind,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if kind == "OUTCAR":
        summary = parse_outcar(path)
        q_nelect = summary.charge_from_nelect
        row.update(
            nelect=summary.nelect,
            charge_from_nelect=q_nelect,
            charge_consistent=(
                None
                if label_charge is None or q_nelect is None
                else abs(q_nelect - label_charge) < 1e-3
            ),
            calc_type="relaxation" if summary.is_relaxation else "single_point",
            converged=summary.is_converged,
            unconverged_reason=summary.unconverged_reason,
            n_ionic_steps=summary.n_ionic_steps,
            final_energy_eV=summary.final_energy_eV,
            final_magnetization=summary.final_magnetization,
            functional=summary.functional,
            dispersion=summary.dispersion,
            ivdw=summary.ivdw,
            kpoint_grid=summary.kpoint_grid,
            encut_eV=summary.encut_eV,
            ispin=summary.ispin,
            vasp_version=summary.vasp_version,
        )
    return {column: _fmt(row.get(column)) for column in MANIFEST_COLUMNS}


def iter_data_files(data_root: Path) -> list[Path]:
    return sorted(
        path for path in data_root.rglob("*") if path.is_file() and path.name not in IGNORED_NAMES
    )


def build_manifest_rows(data_root: Path, relative_to: Path) -> list[dict[str, str]]:
    return [manifest_row(path, data_root, relative_to) for path in iter_data_files(data_root)]


def write_csv(rows: Iterable[Mapping[str, str]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def build_model_registry(
    project_root: Path, existing: Mapping[str, Mapping[str, str]] | None = None
) -> list[dict[str, str]]:
    """Hash every ``*.model`` under ``project_root``; keep hand-filled metadata columns."""
    existing = existing or {}
    rows = []
    for path in sorted(project_root.rglob("*.model")):
        relative = path.relative_to(project_root).as_posix()
        previous = existing.get(relative, {})
        row = {
            "path": relative,
            "size_bytes": str(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for column in REGISTRY_TODO_COLUMNS:
            row[column] = previous.get(column, "") or "TODO"
        rows.append(row)
    return rows
