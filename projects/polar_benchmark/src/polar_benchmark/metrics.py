"""Structure and energy-ranking metrics for the charged-defect benchmark.

All functions are pure (no I/O, no MACE) so they can be unit tested in CI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.stats import spearmanr

DEFAULT_LOCAL_CUTOFF_ANGSTROM = 4.0


def _check_compatible(atoms_a: Atoms, atoms_b: Atoms) -> None:
    if len(atoms_a) != len(atoms_b):
        raise ValueError(f"Atom count mismatch: {len(atoms_a)} != {len(atoms_b)}")
    symbols_a = atoms_a.get_chemical_symbols()
    symbols_b = atoms_b.get_chemical_symbols()
    if symbols_a != symbols_b:
        first = next(i for i, (a, b) in enumerate(zip(symbols_a, symbols_b, strict=True)) if a != b)
        raise ValueError(
            f"Species order differs at index {first}: {symbols_a[first]} != {symbols_b[first]}"
        )


def mic_displacements(atoms_a: Atoms, atoms_b: Atoms) -> np.ndarray:
    """Per-atom minimum-image displacement vectors from ``atoms_b`` to ``atoms_a``.

    The cell and PBC of ``atoms_b`` (the reference) are used. Species order must match.
    """
    _check_compatible(atoms_a, atoms_b)
    raw = atoms_a.get_positions() - atoms_b.get_positions()
    displacements, _ = find_mic(raw, atoms_b.get_cell(), pbc=atoms_b.get_pbc())
    return displacements


def mic_rmsd(atoms_a: Atoms, atoms_b: Atoms, indices: Sequence[int] | None = None) -> float:
    """Root-mean-square minimum-image displacement (Å), optionally over ``indices`` only.

    No rigid-body alignment is applied: both structures share the same cell and
    the vacancy pins the frame, so a global translation is itself meaningful.
    """
    displacements = mic_displacements(atoms_a, atoms_b)
    if indices is not None:
        if len(indices) == 0:
            return math.nan
        displacements = displacements[np.asarray(indices, dtype=int)]
    return float(np.sqrt(np.mean(np.sum(displacements**2, axis=1))))


def atoms_near_site(
    atoms: Atoms,
    site_fractional: Sequence[float],
    cutoff: float = DEFAULT_LOCAL_CUTOFF_ANGSTROM,
) -> list[int]:
    """Indices of atoms within ``cutoff`` Å (minimum image) of a fractional site."""
    site_cartesian = np.asarray(site_fractional, dtype=float) @ atoms.get_cell().array
    vectors = atoms.get_positions() - site_cartesian
    _, distances = find_mic(vectors, atoms.get_cell(), pbc=atoms.get_pbc())
    return [int(i) for i in np.flatnonzero(distances <= cutoff)]


@dataclass(frozen=True)
class GroundStateSummary:
    """Ground-state comparison for one (model, charge) group."""

    n_candidates: int
    mlip_ground_state: str | None
    dft_ground_state: str | None
    ground_state_hit: bool | None
    spearman_rho: float
    spearman_p: float


def select_ground_state(energies: Mapping[str, float]) -> str | None:
    """Label with the lowest finite energy (ties resolved by label order)."""
    finite = {k: v for k, v in energies.items() if v is not None and math.isfinite(v)}
    if not finite:
        return None
    return min(sorted(finite), key=lambda key: finite[key])


def spearman_rank(
    mlip_energies: Mapping[str, float], dft_energies: Mapping[str, float]
) -> tuple[float, float]:
    """Spearman correlation of energies over labels finite in both mappings."""
    common = sorted(
        k
        for k in set(mlip_energies) & set(dft_energies)
        if math.isfinite(mlip_energies[k]) and math.isfinite(dft_energies[k])
    )
    if len(common) < 3:
        return math.nan, math.nan
    mlip = [mlip_energies[k] for k in common]
    dft = [dft_energies[k] for k in common]
    if len(set(mlip)) == 1 or len(set(dft)) == 1:
        return math.nan, math.nan
    result = spearmanr(mlip, dft)
    return float(result.statistic), float(result.pvalue)


def compare_ground_states(
    mlip_energies: Mapping[str, float], dft_energies: Mapping[str, float]
) -> GroundStateSummary:
    """Compare MLIP and DFT energy landscapes over the same set of distortions.

    Only labels with a finite energy in *both* mappings are candidates, so an
    excluded (e.g. unconverged) DFT reference cannot win on either side.
    """
    common = {
        k
        for k in set(mlip_energies) & set(dft_energies)
        if math.isfinite(mlip_energies[k]) and math.isfinite(dft_energies[k])
    }
    mlip_gs = select_ground_state({k: mlip_energies[k] for k in common})
    dft_gs = select_ground_state({k: dft_energies[k] for k in common})
    hit = None if mlip_gs is None or dft_gs is None else mlip_gs == dft_gs
    rho, p_value = spearman_rank(mlip_energies, dft_energies)
    return GroundStateSummary(
        n_candidates=len(common),
        mlip_ground_state=mlip_gs,
        dft_ground_state=dft_gs,
        ground_state_hit=hit,
        spearman_rho=rho,
        spearman_p=p_value,
    )
