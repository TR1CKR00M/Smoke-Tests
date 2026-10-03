#!/usr/bin/env python3
"""Compatibility entrypoint for the Plot Atlas pairwise time-course panel."""

from __future__ import annotations

from reproduce_plots import DEFAULT_RAW, configure_style, plot_pairwise, read_logs


def main() -> None:
    configure_style()
    data = read_logs(sorted(DEFAULT_RAW.glob("*_log.csv")))
    plot_pairwise(data, DEFAULT_RAW.parent / "figures" / "atlas_time_sequence_analysis")


if __name__ == "__main__":
    main()
