import math

import numpy as np
import pytest
from ase import Atoms
from polar_benchmark.metrics import (
    atoms_near_site,
    compare_ground_states,
    mic_rmsd,
    select_ground_state,
    spearman_rank,
)

CELL = np.diag([10.0, 10.0, 10.0])


def make_atoms(positions, symbols="SbSe"):
    return Atoms(symbols, positions=positions, cell=CELL, pbc=True)


def test_mic_rmsd_ignores_periodic_wrapping():
    # Arrange: atom 0 sits at x=0.05 in the reference and x=9.95 (wrapped) in the
    # prediction, i.e. a true displacement of 0.1 Å, not 9.9 Å.
    reference = make_atoms([[0.05, 5.0, 5.0], [5.0, 5.0, 5.0]])
    predicted = make_atoms([[9.95, 5.0, 5.0], [5.0, 5.0, 5.0]])

    # Act
    rmsd = mic_rmsd(predicted, reference)

    # Assert: sqrt((0.1^2 + 0) / 2)
    assert rmsd == pytest.approx(0.1 / math.sqrt(2))


def test_mic_rmsd_over_subset_of_indices():
    reference = make_atoms([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    predicted = make_atoms([[1.0, 1.0, 1.3], [5.0, 5.0, 5.0]])

    assert mic_rmsd(predicted, reference, indices=[0]) == pytest.approx(0.3)
    assert mic_rmsd(predicted, reference, indices=[1]) == pytest.approx(0.0)
    assert math.isnan(mic_rmsd(predicted, reference, indices=[]))


def test_mic_rmsd_rejects_different_species_order():
    reference = make_atoms([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]], symbols="SbSe")
    predicted = make_atoms([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]], symbols="SeSb")

    with pytest.raises(ValueError, match="Species order differs"):
        mic_rmsd(predicted, reference)


def test_mic_rmsd_rejects_different_atom_counts():
    reference = make_atoms([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    predicted = Atoms("Sb", positions=[[1.0, 1.0, 1.0]], cell=CELL, pbc=True)

    with pytest.raises(ValueError, match="Atom count mismatch"):
        mic_rmsd(predicted, reference)


def test_atoms_near_site_uses_minimum_image():
    # Site at the cell corner; atom 0 is 0.5 Å away through the boundary.
    atoms = make_atoms([[9.5, 0.0, 0.0], [5.0, 5.0, 5.0]])

    assert atoms_near_site(atoms, [0.0, 0.0, 0.0], cutoff=1.0) == [0]


def test_select_ground_state_skips_non_finite():
    energies = {"D_0": -1.0, "D_10": -2.0, "D_20": math.nan}

    assert select_ground_state(energies) == "D_10"
    assert select_ground_state({}) is None


def test_compare_ground_states_hit_and_rank():
    dft = {"D_-10": -3.0, "D_0": -2.0, "D_10": -1.0, "D_20": 0.0}
    mlip = {"D_-10": -30.0, "D_0": -20.0, "D_10": -10.0, "D_20": 5.0}

    summary = compare_ground_states(mlip, dft)

    assert summary.mlip_ground_state == "D_-10"
    assert summary.dft_ground_state == "D_-10"
    assert summary.ground_state_hit is True
    assert summary.spearman_rho == pytest.approx(1.0)
    assert summary.n_candidates == 4


def test_compare_ground_states_miss_and_excluded_reference():
    # The MLIP minimum (D_20) has no usable DFT reference, so it cannot be selected.
    dft = {"D_-10": -3.0, "D_0": -2.0, "D_10": -1.0}
    mlip = {"D_-10": -1.0, "D_0": -2.0, "D_10": -3.0, "D_20": -9.0}

    summary = compare_ground_states(mlip, dft)

    assert summary.mlip_ground_state == "D_10"
    assert summary.dft_ground_state == "D_-10"
    assert summary.ground_state_hit is False
    assert summary.spearman_rho == pytest.approx(-1.0)


def test_spearman_needs_three_points_and_variation():
    assert all(math.isnan(v) for v in spearman_rank({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}))
    flat = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert all(math.isnan(v) for v in spearman_rank(flat, {"a": 1.0, "b": 2.0, "c": 3.0}))
