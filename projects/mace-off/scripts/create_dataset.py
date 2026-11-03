#!/usr/bin/env python3
"""Convert TorsionDrive JSON results into a SPICE-style HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from qcelemental.models.procedures import TorsionDriveResult


def convert(input_dir: Path, output_path: Path) -> int:
    files = sorted(input_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No TorsionDrive JSON files found in {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with h5py.File(output_path, "w") as output:
        for path in files:
            result = TorsionDriveResult.parse_file(path)
            if not result.success:
                print(f"Skipping failed TorsionDrive result: {path}")
                continue
            molecule = result.initial_molecule[0]
            smiles = molecule.extras["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
            group_name = smiles.replace("/", "")
            if group_name in output:
                raise ValueError(f"Duplicate mapped SMILES group {group_name!r} in {path}")
            group = output.create_group(group_name)
            group.create_dataset("smiles", data=[smiles], dtype=h5py.string_dtype())
            group.create_dataset("atomic_numbers", data=molecule.atomic_numbers, dtype=np.int16)

            angles = sorted(result.final_molecules, key=lambda value: float(value.split(",")[0]))
            energies = [result.final_energies[angle] for angle in angles]
            conformations = [result.final_molecules[angle].geometry for angle in angles]
            gradients = [result.optimization_history[angle][0].trajectory[0]
                         .properties.return_gradient for angle in angles]
            conformations_ds = group.create_dataset("conformations", data=np.asarray(conformations),
                                                      dtype=np.float32)
            conformations_ds.attrs["units"] = "bohr"
            energy_ds = group.create_dataset("dft total energy", data=np.asarray(energies),
                                             dtype=np.float64)
            energy_ds.attrs["units"] = "hartree"
            gradient_ds = group.create_dataset("dft total gradient", data=np.asarray(gradients),
                                               dtype=np.float32)
            gradient_ds.attrs["units"] = "hartree/bohr"
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="folder containing TorsionDrive JSON files")
    parser.add_argument("output_hdf5", type=Path, help="destination SPICE-style HDF5 file")
    args = parser.parse_args()
    count = convert(args.input_dir, args.output_hdf5)
    print(f"Wrote {count} molecules to {args.output_hdf5}")


if __name__ == "__main__":
    main()
