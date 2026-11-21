"""The CLI modules must import and parse arguments without mace/torch installed."""

import argparse
import csv

import pytest


def test_relax_cli_imports_without_mace_and_parses_defaults():
    import mace_relax_comparison as cli

    args = cli.parse_args([])

    assert args.dispersion == "none"
    assert args.head == "omat_pbe"
    assert args.set_total_charge is True
    assert args.include_unconverged_refs is False
    assert args.charges == ["mi3", "mi2", "mi1", "neu", "pl1"]


def test_relax_cli_d3_shorthand_and_model_specs():
    import mace_relax_comparison as cli

    args = cli.parse_args(["--d3", "--model", "Polar-S=polar:polar-1-s"])

    assert args.dispersion == "d3-zero"
    assert cli.parse_model_specs(args.model) == [("Polar-S", "polar:polar-1-s")]
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_model_specs(["no-separator"])
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_model_specs(["bad/name=x.model"])


def test_plot_from_csv_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from plot_relax_comparison import plot_from_csv

    csv_path = tmp_path / "relax_per_charge.csv"
    rows = [
        {
            "model": "M",
            "charge": q,
            "rmsd_gs_all_A": 0.1 * (i + 1),
            "ground_state_hit": hit,
            "spearman_rho": 0.5,
        }
        for i, (q, hit) in enumerate([(-1, "True"), (0, "False"), (1, "")])
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = plot_from_csv(csv_path, tmp_path / "fig.png")

    assert output.is_file() and output.stat().st_size > 0
