"""Write data/DATA_MANIFEST.csv and MODEL_REGISTRY.csv for the polar benchmark.

Usage (from the repository root):

    uv run python projects/polar_benchmark/script/build_manifest.py

The data manifest lists every file under ``data/`` with its sha256; OUTCARs
additionally get NELECT, the charge implied by NELECT, convergence, final
magnetisation and the DFT settings (functional, IVDW, k-point grid).

The model registry hashes every ``*.model`` file. Its metadata columns
(source URL, version, license, ...) are filled in by hand and are preserved
when the script is re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_benchmark.manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    REGISTRY_COLUMNS,
    build_manifest_rows,
    build_model_registry,
    read_existing_registry,
    write_csv,
)

LOGGER = logging.getLogger("build_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "DATA_MANIFEST.csv"
    )
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "MODEL_REGISTRY.csv")
    parser.add_argument(
        "--skip-registry", action="store_true", help="Only write the data manifest."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_root}")

    rows = build_manifest_rows(args.data_root, relative_to=PROJECT_ROOT)
    write_csv(rows, args.manifest, MANIFEST_COLUMNS)
    outcars = [row for row in rows if row["file_type"] == "OUTCAR"]
    for row in outcars:
        if row["converged"] != "True":
            LOGGER.warning("Unconverged reference: %s (%s)", row["path"], row["unconverged_reason"])
        if row["charge_consistent"] == "False":
            LOGGER.warning(
                "Charge label %s disagrees with NELECT-derived charge %s: %s",
                row["charge_from_label"],
                row["charge_from_nelect"],
                row["path"],
            )
    LOGGER.info("Wrote %d rows (%d OUTCARs) to %s", len(rows), len(outcars), args.manifest)

    if not args.skip_registry:
        registry = build_model_registry(PROJECT_ROOT, read_existing_registry(args.registry))
        write_csv(registry, args.registry, REGISTRY_COLUMNS)
        LOGGER.info("Wrote %d models to %s", len(registry), args.registry)


if __name__ == "__main__":
    main()
