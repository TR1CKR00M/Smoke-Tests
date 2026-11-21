import math

import pytest
from polar_benchmark.benchmark import group_rows, summarize_group


def row(distortion, mlip, dft, **overrides):
    base = {
        "model": "M",
        "charge_label": "pl1",
        "charge": 1,
        "distortion": distortion,
        "mlip_energy_eV": mlip,
        "bfgs_converged": True,
        "dft_energy_eV": dft,
        "dft_converged": True,
        "dft_excluded_reason": "",
        "rmsd_all_A": 0.2,
        "error": "",
    }
    base.update(overrides)
    return base


def test_summary_excludes_unconverged_dft_and_failed_relaxations():
    rows = [
        # Lowest DFT energy, but the reference is unconverged -> must be excluded.
        row("D_-60", -9.0, -99.0, dft_converged=False, dft_excluded_reason="unconverged"),
        row("D_-10", -3.0, -5.0),
        row("D_0", -2.0, -4.0, rmsd_all_A=0.05),
        row("D_10", -1.0, -3.0),
        row("D_20", math.nan, -2.0, error="RuntimeError: boom"),
        row("D_30", -50.0, -1.0, bfgs_converged=False),
    ]

    summary = summarize_group(rows)

    assert summary["n_structures"] == 6
    assert summary["n_failed"] == 1
    assert summary["n_bfgs_unconverged"] == 1
    assert summary["n_dft_excluded"] == 1
    assert summary["n_candidates"] == 3
    assert summary["dft_ground_state"] == "D_-10"
    assert summary["mlip_ground_state"] == "D_-10"
    assert summary["ground_state_hit"] is True
    assert summary["mlip_gs_minus_dft_gs_relative_energy_eV"] == pytest.approx(0.0)
    assert summary["spearman_rho"] == pytest.approx(1.0)
    assert summary["diag_min_rmsd_same_distortion_A"] == pytest.approx(0.05)
    assert summary["diag_min_rmsd_distortion"] == "D_0"


def test_summary_reports_energy_cost_of_a_miss():
    rows = [row("D_-10", -1.0, -5.0), row("D_0", -3.0, -4.5), row("D_10", -2.0, -4.0)]

    summary = summarize_group(rows)

    assert summary["ground_state_hit"] is False
    assert summary["mlip_ground_state"] == "D_0"
    assert summary["mlip_gs_minus_dft_gs_relative_energy_eV"] == pytest.approx(0.5)


def test_summary_accepts_csv_string_values():
    rows = [
        row("D_0", "-2.0", "-4.0", bfgs_converged="True", dft_converged="True"),
        row("D_10", "-1.0", "-3.0", bfgs_converged="True", dft_converged="True"),
    ]

    summary = summarize_group(rows)

    assert summary["mlip_ground_state"] == "D_0"
    assert summary["ground_state_hit"] is True


def test_group_rows_by_model_and_charge():
    rows = [row("D_0", -1, -1), row("D_0", -1, -1, charge_label="neu"), row("D_10", -1, -1)]

    groups = group_rows(rows)

    assert set(groups) == {("M", "pl1"), ("M", "neu")}
    assert len(groups[("M", "pl1")]) == 2
