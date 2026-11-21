"""Per-charge aggregation of per-structure relaxation results (MACE-free)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .metrics import compare_ground_states

PER_STRUCTURE_COLUMNS = [
    "model",
    "charge_label",
    "charge",
    "distortion",
    "poscar",
    "dft_contcar",
    "dft_outcar",
    "dft_energy_eV",
    "dft_converged",
    "dft_excluded_reason",
    "dft_nelect",
    "dft_magnetization",
    "defect_site_frac",
    "n_local_atoms",
    "mlip_energy_eV",
    "bfgs_converged",
    "bfgs_steps",
    "rmsd_all_A",
    "rmsd_local_A",
    "rmsd_initial_vs_dft_A",
    "relaxed_structure",
    "error",
]

PER_CHARGE_COLUMNS = [
    "model",
    "charge_label",
    "charge",
    "n_structures",
    "n_failed",
    "n_bfgs_unconverged",
    "n_dft_excluded",
    "n_candidates",
    "mlip_ground_state",
    "dft_ground_state",
    "ground_state_hit",
    "rmsd_gs_all_A",
    "rmsd_gs_local_A",
    "spearman_rho",
    "spearman_p",
    "mlip_gs_minus_dft_gs_relative_energy_eV",
    "diag_min_rmsd_same_distortion_A",
    "diag_min_rmsd_distortion",
]


def _is_true(value: Any) -> bool:
    return value is True or str(value) == "True"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def usable_mlip_energy(row: Mapping[str, Any]) -> float | None:
    """MLIP energy if the relaxation ran without error and BFGS converged."""
    if row.get("error") or not _is_true(row.get("bfgs_converged")):
        return None
    return _finite(row.get("mlip_energy_eV"))


def usable_dft_energy(row: Mapping[str, Any]) -> float | None:
    """DFT energy if the reference is converged and not excluded."""
    if row.get("dft_excluded_reason") or not _is_true(row.get("dft_converged")):
        return None
    return _finite(row.get("dft_energy_eV"))


def summarize_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Ground-state / ranking summary for the rows of one (model, charge) group.

    Structural cross-RMSDs (MLIP ground state vs. DFT ground state) need the
    relaxed structures and are filled in by the caller.
    """
    mlip = {r["distortion"]: e for r in rows if (e := usable_mlip_energy(r)) is not None}
    dft = {r["distortion"]: e for r in rows if (e := usable_dft_energy(r)) is not None}
    comparison = compare_ground_states(mlip, dft)

    relative_gap = math.nan
    if comparison.mlip_ground_state and comparison.dft_ground_state:
        # DFT energy of the MLIP-selected structure above the DFT ground state:
        # 0 for a hit, >0 measures how costly the miss is on the DFT surface.
        relative_gap = dft[comparison.mlip_ground_state] - dft[comparison.dft_ground_state]

    diagnostic = [
        (float(r["rmsd_all_A"]), r["distortion"])
        for r in rows
        if not r.get("error") and _finite(r.get("rmsd_all_A")) is not None
    ]
    diag_rmsd, diag_label = min(diagnostic) if diagnostic else (math.nan, "")

    first = rows[0]
    return {
        "model": first["model"],
        "charge_label": first["charge_label"],
        "charge": first["charge"],
        "n_structures": len(rows),
        "n_failed": sum(1 for r in rows if r.get("error")),
        "n_bfgs_unconverged": sum(
            1 for r in rows if not r.get("error") and not _is_true(r.get("bfgs_converged"))
        ),
        "n_dft_excluded": sum(1 for r in rows if usable_dft_energy(r) is None),
        "n_candidates": comparison.n_candidates,
        "mlip_ground_state": comparison.mlip_ground_state or "",
        "dft_ground_state": comparison.dft_ground_state or "",
        "ground_state_hit": comparison.ground_state_hit,
        "rmsd_gs_all_A": math.nan,
        "rmsd_gs_local_A": math.nan,
        "spearman_rho": comparison.spearman_rho,
        "spearman_p": comparison.spearman_p,
        "mlip_gs_minus_dft_gs_relative_energy_eV": relative_gap,
        "diag_min_rmsd_same_distortion_A": diag_rmsd,
        "diag_min_rmsd_distortion": diag_label,
    }


def group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["charge_label"]), []).append(row)
    return groups
