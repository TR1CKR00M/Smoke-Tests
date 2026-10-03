#!/usr/bin/env python3
"""Compatibility entrypoint for the Plot Atlas trajectory panel.

Use ``reproduce_plots.py`` for the complete figure set. This historical
entrypoint remains so old commands continue to work, but delegates all plot
logic to the single validated implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reproduce_plots import configure_style, plot_trajectory, read_logs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("txt_name", help="metadata file under data/metadata")
    args = parser.parse_args()
    metadata = ROOT / "data" / "metadata" / args.txt_name
    if metadata.suffix != ".txt":
        metadata = metadata.with_suffix(".txt")
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    paths = [ROOT / line.strip() for line in metadata.read_text().splitlines() if line.strip()]
    configure_style()
    plot_trajectory(
        read_logs(paths),
        ROOT / "results" / "figures" / f"atlas_{metadata.stem}",
        raw_alpha=0.12,
    )


if __name__ == "__main__":
    main()
