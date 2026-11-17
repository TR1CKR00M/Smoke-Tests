"""Train SOAP-based surrogate regressors for the four MACE test-set errors.

For every structure in the configured test XYZ, this script computes a global
inner-averaged SOAP power spectrum and each MACE model's absolute total-energy
error. One GradientBoostingRegressor is fitted per MACE model. Features,
targets, fitted estimators, and a short fit report are saved to an output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from ase.data import atomic_numbers
from ase.io import read
from dscribe.descriptors import SOAP
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric
from sklearn.ensemble import GradientBoostingRegressor


DEFAULT_MODELS = [
    "mace_k128_L0",
    "mace_k32_L0",
    "mace_k128_L2",
    "mace_k32_L2",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "surrogate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=str(DATA_DIR / "defect_test_1200K.xyz"), help="Test structures with REF_energy labels")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="MACE model prefixes; expects <prefix>_stagetwo.model",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for features, targets, and models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="MACE inference device")
    return parser.parse_args()


def read_structures(path: str) -> list:
    xyz_path = Path(path)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Test XYZ file not found: {xyz_path}")
    frames = read(xyz_path, index=":")
    if not frames:
        raise ValueError(f"Test XYZ file contains no structures: {xyz_path}")
    for frame_index, atoms in enumerate(frames):
        if "REF_energy" not in atoms.info:
            raise KeyError(f"{xyz_path}, frame {frame_index}: missing REF_energy")
    return frames


def make_soap_feature_names(species: list[str], n_max: int, l_max: int) -> list[str]:
    """Return names in DScribe's uncompressed SOAP power-spectrum order.

    DScribe orders species by atomic number, then angular degree, followed by
    radial-index pairs. For same-species blocks only the unique symmetric
    radial pairs are included. Radial indices in these human-readable names
    are one-based, as in labels such as n13.
    """
    ordered_species = sorted(species, key=lambda symbol: atomic_numbers[symbol])
    names = []
    for first_index, first_species in enumerate(ordered_species):
        for second_species in ordered_species[first_index:]:
            for angular_degree in range(l_max + 1):
                for n1 in range(1, n_max + 1):
                    n2_start = n1 if first_species == second_species else 1
                    for n2 in range(n2_start, n_max + 1):
                        names.append(
                            f"{first_species}-{second_species}_n{n1}{n2}_l{angular_degree}"
                        )
    return names


def build_soap(frames) -> tuple[np.ndarray, list[str]]:
    species = sorted(
        {symbol for atoms in frames for symbol in atoms.get_chemical_symbols()},
        key=lambda symbol: atomic_numbers[symbol],
    )
    descriptor = SOAP(
        species=species,
        r_cut=5.0,
        n_max=4,
        l_max=4,
        average="inner",
        periodic=True,
        sparse=False,
    )
    features = np.asarray(descriptor.create(frames), dtype=np.float64)
    feature_names = make_soap_feature_names(species, n_max=4, l_max=4)
    if features.ndim != 2 or features.shape[0] != len(frames):
        raise RuntimeError(f"Unexpected SOAP matrix shape {features.shape}; expected one vector per structure")
    if features.shape[1] != len(feature_names):
        raise RuntimeError(
            "SOAP feature-name count does not match DScribe output: "
            f"{len(feature_names)} names for {features.shape[1]} columns. "
            "Check the installed DScribe version and feature ordering."
        )
    if not np.isfinite(features).all():
        raise ValueError("SOAP feature matrix contains NaN or infinite values")
    return features, feature_names


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    original_jit_load = torch.jit.load

    def cpu_jit_load(file, map_location=None, **kwargs):
        return original_jit_load(file, map_location=torch.device("cpu"), **kwargs)

    torch.jit.load = cpu_jit_load
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    finally:
        torch.jit.load = original_jit_load
    model = model.to(device)
    model.eval()
    return model


def predict_absolute_energy_errors(model: torch.nn.Module, frames: list, device: torch.device) -> np.ndarray:
    model_dtype = next(model.parameters()).dtype
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    errors = []

    for frame_index, atoms in enumerate(frames):
        data = AtomicData.from_config(config_from_atoms(atoms), z_table=z_table, cutoff=cutoff)
        batch = torch_geometric.Batch.from_data_list([data]).to(device)
        batch_dict = batch.to_dict()
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                batch_dict[key] = value.to(dtype=model_dtype)

        # MACE computes forces through autograd even though only energy is used.
        output = model(batch_dict, compute_force=True)
        predicted_energy = float(output["energy"].detach().cpu().reshape(-1)[0])
        reference_energy = float(atoms.info["REF_energy"])
        errors.append(abs(predicted_energy - reference_energy))

    return np.asarray(errors, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is not available")
    device = torch.device(args.device)
    frames = read_structures(args.test)

    print(f"Computing global SOAP features for {len(frames)} structures...")
    features, feature_names = build_soap(frames)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "soap_features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", *feature_names])
        for frame_index, row in enumerate(features):
            writer.writerow([frame_index, *row.tolist()])

    targets: dict[str, np.ndarray] = {}
    report_rows = []

    for model_name in args.models:
        model_path = MODEL_DIR / f"{model_name}_stagetwo.model"
        print(f"Predicting energy errors with {model_path}...")
        mace_model = load_model(model_path, device)
        delta_energy = predict_absolute_energy_errors(mace_model, frames, device)
        del mace_model
        targets[model_name] = delta_energy

        surrogate = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
        )
        surrogate.fit(features, delta_energy)
        fitted = surrogate.predict(features)
        train_rmse = float(np.sqrt(np.mean(np.square(fitted - delta_energy))))
        train_mae = float(np.mean(np.abs(fitted - delta_energy)))

        model_file = output_dir / f"{model_name}_soap_surrogate.joblib"
        joblib.dump(surrogate, model_file)
        report_rows.append(
            {
                "model": model_name,
                "frames": len(frames),
                "mean_abs_energy_error_eV": float(delta_energy.mean()),
                "max_abs_energy_error_eV": float(delta_energy.max()),
                "surrogate_fit_rmse_eV": train_rmse,
                "surrogate_fit_mae_eV": train_mae,
                "surrogate_file": model_file.name,
            }
        )

    targets_path = output_dir / "mace_absolute_energy_errors.csv"
    with targets_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", *[f"deltaE_{name}_eV" for name in args.models]])
        for frame_index in range(len(frames)):
            writer.writerow([frame_index, *[targets[name][frame_index] for name in args.models]])

    # Save the mapping and fit diagnostics for reproducibility. The reported
    # fit metrics are in-sample scores, not held-out surrogate validation.
    manifest = {
        "test_xyz": str(Path(args.test)),
        "n_structures": len(frames),
        "soap": {
            "species": sorted(
                {symbol for atoms in frames for symbol in atoms.get_chemical_symbols()},
                key=lambda symbol: atomic_numbers[symbol],
            ),
            "r_cut_angstrom": 5.0,
            "n_max": 4,
            "l_max": 4,
            "average": "inner",
            "periodic": True,
            "n_features": len(feature_names),
            "feature_name_format": "ElementA-ElementB_n{n1}{n2}_l{l}; radial indices are one-based",
        },
        "surrogate": {
            "estimator": "sklearn.ensemble.GradientBoostingRegressor",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 8,
            "random_state": 42,
            "target": "absolute total-energy error |E_MACE - E_REF_energy| in eV",
            "fit_metrics_note": "In-sample diagnostics; no surrogate holdout split is used.",
        },
        "models": report_rows,
    }
    manifest_path = output_dir / "surrogate_training_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSurrogate training summary (in-sample fit):")
    print(f"{'MACE model':<16} | {'Mean ΔE (eV)':>13} | {'Max ΔE (eV)':>12} | {'Fit RMSE (eV)':>13} | {'Fit MAE (eV)':>12}")
    print("-" * 82)
    for row in report_rows:
        print(
            f"{row['model']:<16} | {row['mean_abs_energy_error_eV']:>13.6f} | "
            f"{row['max_abs_energy_error_eV']:>12.6f} | {row['surrogate_fit_rmse_eV']:>13.6f} | "
            f"{row['surrogate_fit_mae_eV']:>12.6f}"
        )
    print(f"\nSaved SOAP features, error targets, fitted surrogates, and report under: {output_dir}")


if __name__ == "__main__":
    main()
