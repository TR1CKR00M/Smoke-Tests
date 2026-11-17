#!/usr/bin/env python3
"""Run a constrained DFT torsion scan through QCEngine/TorsionDrive.

The starting geometry must be supplied as an XYZ file, preferably the
published optimized geometry in the exact atom order used for the dihedral.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import qcengine as qcng
from qcelemental.models import Molecule
from qcelemental.models.common_models import DriverEnum, Model
from qcelemental.models.procedures import (
    OptimizationSpecification,
    QCInputSpecification,
    TDKeywords,
    TorsionDriveInput,
)


def parse_dihedral(value: str) -> tuple[int, int, int, int]:
    try:
        atoms = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        message = "dihedral must be four comma-separated zero-based atom indices"
        raise argparse.ArgumentTypeError(message) from exc
    if len(atoms) != 4 or len(set(atoms)) != 4 or min(atoms) < 0:
        message = "dihedral must contain four distinct nonnegative atom indices"
        raise argparse.ArgumentTypeError(message)
    return atoms  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path,
                        help="published optimized starting geometry in XYZ format")
    parser.add_argument("--dihedral", required=True, type=parse_dihedral,
                        help="four zero-based atom indices, e.g. 2,5,8,11")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("torsiondrive-result.json"))
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--memory", type=float, default=8.0, help="QCEngine memory in GiB")
    parser.add_argument("--maxiter", type=int, default=300,
                        help="geometry optimizer iteration limit")
    args = parser.parse_args()

    if not args.xyz.is_file():
        parser.error(f"XYZ file not found: {args.xyz}")
    if args.ncores < 1 or args.memory <= 0 or args.maxiter < 1:
        parser.error("ncores, memory, and maxiter must be positive")

    molecule = Molecule.from_data(
        args.xyz.read_text(encoding="utf-8"),
        dtype="xyz",
        molecular_charge=args.charge,
        molecular_multiplicity=args.multiplicity,
        fix_com=True,
        fix_orientation=True,
    )
    if max(args.dihedral) >= len(molecule.symbols):
        parser.error(f"dihedral index out of range for {len(molecule.symbols)} atoms")

    # Grid includes -180, -165, ..., 165 degrees (24 unique points).
    td_keywords = TDKeywords(dihedrals=[list(args.dihedral)], grid_spacing=[15])
    qc_spec = QCInputSpecification(
        driver=DriverEnum.gradient,
        model=Model(method="wb97m-d3bj", basis="def2-tzvppd"),
    )
    opt_spec = OptimizationSpecification(
        procedure="geometric",
        keywords={
            "coordsys": "tric",
            "maxiter": args.maxiter,
            "program": "psi4",
            "reset": True,
            "qccnv": True,
        },
    )
    td_input = TorsionDriveInput(
        keywords=td_keywords,
        input_specification=qc_spec,
        # QCSchema accepts one or more seed geometries as a list.
        initial_molecule=[molecule],
        optimization_spec=opt_spec,
    )

    result = qcng.compute_procedure(
        input_data=td_input,
        procedure="torsiondrive",
        raise_error=True,
        local_options={"ncores": args.ncores, "memory": args.memory},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.json(indent=2), encoding="utf-8")
    print(f"TorsionDrive complete: {args.output}")
    print(f"Successful grid points: {len(result.final_energies)} / 24")


if __name__ == "__main__":
    main()
