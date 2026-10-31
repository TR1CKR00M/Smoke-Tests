#!/usr/bin/env python3
"""Compatibility entrypoint for the Plot Atlas trajectory panel.

Use ``reproduce_plots.py`` for the complete figure set. This historical
entrypoint remains so old commands continue to work, but delegates all plot
logic to the single validated implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    from src.reproduce_plots import configure_style, plot_trajectory, read_logs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("txt_name", help="metadata file under data/metadata")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="custom plot labels in the same order as the CSV paths in the metadata file",
    )
    args = parser.parse_args()
    metadata = ROOT / "data" / "metadata" / args.txt_name
    if metadata.suffix != ".txt":
        metadata = metadata.with_suffix(".txt")
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    paths = [ROOT / line.strip() for line in metadata.read_text().splitlines() if line.strip()]
    if args.labels is not None and len(args.labels) != len(paths):
        parser.error(
            f"--labels needs one label per CSV path ({len(paths)} expected, {len(args.labels)} given)"
        )
    if args.labels is not None and len(set(args.labels)) != len(args.labels):
        parser.error("--labels must be unique")
    configure_style()
    data = read_logs(paths)
    if args.labels is not None:
        data = dict(zip(args.labels, data.values(), strict=True))
    plot_trajectory(
        data,
        ROOT / "results" / "figures" / f"atlas_{metadata.stem}",
        raw_alpha=0.12,
    )


if __name__ == "__main__":
    main()
