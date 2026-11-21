from pathlib import Path

import pytest
from polar_benchmark.vasp_io import (
    find_structure,
    parse_defect_site_comment,
    parse_outcar,
    parse_outcar_text,
    read_poscar_comment,
)

STAGE1 = Path(__file__).resolve().parents[1] / "data/1_FM_benchmark/V_Sb1/DFT_PBE"
CONVERGED_OUTCAR = STAGE1 / "neu/D_0/OUTCAR_0"
UNCONVERGED_OUTCAR = STAGE1 / "pl1/D_-60/OUTCAR_-60"

requires_data = pytest.mark.skipif(
    not CONVERGED_OUTCAR.is_file(), reason="committed stage-1 OUTCARs not available"
)


@requires_data
def test_converged_neutral_outcar():
    summary = parse_outcar(CONVERGED_OUTCAR)

    assert summary.is_converged is True
    assert summary.reached_required_accuracy is True
    assert summary.unconverged_reason is None
    assert summary.nelect == pytest.approx(331.0)
    assert summary.neutral_valence == pytest.approx(23 * 5 + 36 * 6)
    assert summary.charge_from_nelect == pytest.approx(0.0)
    assert summary.final_magnetization == pytest.approx(1.0, abs=1e-3)
    assert summary.n_ions == 59
    assert summary.functional == "PBE"
    assert summary.ivdw == 11
    assert summary.dispersion == "D3(zero)"
    assert summary.kpoint_grid == "1x1x1"
    assert summary.final_energy_eV is not None
    assert summary.n_ionic_steps > 1


@requires_data
def test_unconverged_pl1_d_minus60_is_flagged():
    summary = parse_outcar(UNCONVERGED_OUTCAR)

    assert summary.is_converged is False
    assert summary.reached_required_accuracy is False
    assert "did not reach required accuracy" in summary.unconverged_reason
    assert summary.nelect == pytest.approx(330.0)
    assert summary.charge_from_nelect == pytest.approx(1.0)
    assert summary.final_magnetization == pytest.approx(0.0, abs=1e-3)
    assert summary.final_energy_eV == pytest.approx(-248.07679349)


def test_single_point_without_relaxation_counts_as_converged():
    text = (
        "   NSW    =      0    number of steps for IOM\n"
        "   NELECT =     330.0000    total number of electrons\n"
        "  free  energy   TOTEN  =      -10.0 eV\n"
        "  energy  without entropy=  -10.1  energy(sigma->0) =      -10.05\n"
    )

    summary = parse_outcar_text(text)

    assert summary.is_relaxation is False
    assert summary.is_converged is True
    assert summary.final_energy_eV == pytest.approx(-10.05)
    assert summary.n_ionic_steps == 1


def test_empty_outcar_is_not_converged():
    summary = parse_outcar_text("")

    assert summary.is_converged is False
    assert summary.unconverged_reason == "no completed ionic step (no energy found)"


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("0.0% N(Distort)=3 ~[0.4,0.5,0.4]", (0.4, 0.5, 0.4)),
        (
            "-0.0%_BDM__vac_3_Se[0.58333333 0.37211005 0.44742291]_-dNELECT=0__num_neighbours=2",
            (0.58333333, 0.37211005, 0.44742291),
        ),
        ("no site here", None),
    ],
)
def test_parse_defect_site_comment(comment, expected):
    assert parse_defect_site_comment(comment) == (pytest.approx(expected) if expected else None)


@requires_data
def test_find_structure_handles_suffixed_names():
    directory = STAGE1 / "mi3/D_-30"

    poscar = find_structure(directory, "POSCAR")

    assert poscar.name == "POSCAR_-30"
    assert parse_defect_site_comment(read_poscar_comment(poscar)) == pytest.approx((0.4, 0.5, 0.4))
