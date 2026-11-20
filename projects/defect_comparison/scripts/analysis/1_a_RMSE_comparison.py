"""Compare four trained MACE models with energy and force RMSE tables.

The default datasets match train_grid.sh: train, validation, and the 1200 K
out-of-distribution test set. Energy errors are normalized per atom before
computing RMSE; force RMSE is pooled over all atoms and Cartesian components.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric

DEFAULT_MODELS = [
    "mace_k128_L0",
    "mace_k32_L0",
    "mace_k128_L2",
    "mace_k32_L2",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", default=str(DATA_DIR / "defect_train_300K.xyz"), help="Training XYZ"
    )
    parser.add_argument(
        "--val", default=str(DATA_DIR / "defect_val_300K.xyz"), help="Validation XYZ"
    )
    parser.add_argument(
        "--test",
        default=str(DATA_DIR / "defect_test_1200K.xyz"),
        help="Test XYZ (default: 1200 K OOD set)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model prefixes; each file is expected as <prefix>_stagetwo.model",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Some e3nn/MACE versions load TorchScript submodules without specifying
    # map_location. Keep those submodules on CPU for this comparison script.
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


def read_dataset(path: str) -> list:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    frames = read(file_path, index=":")
    if not frames:
        raise ValueError(f"Dataset contains no frames: {file_path}")
    for frame_index, atoms in enumerate(frames):
        if "REF_energy" not in atoms.info:
            raise KeyError(f"{file_path}, frame {frame_index}: missing REF_energy")
        if "REF_forces" not in atoms.arrays:
            raise KeyError(f"{file_path}, frame {frame_index}: missing REF_forces")
    return frames


def evaluate_model(
    model: torch.nn.Module, frames: list, device: torch.device
) -> tuple[float, float]:
    model_dtype = next(model.parameters()).dtype
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    energy_squared_errors = []
    force_squared_error_sum = 0.0
    force_component_count = 0

    # Force prediction requires autograd internally, so do not use no_grad or
    # inference_mode around the model call.
    for atoms in frames:
        data = AtomicData.from_config(config_from_atoms(atoms), z_table=z_table, cutoff=cutoff)
        batch = torch_geometric.Batch.from_data_list([data]).to(device)
        batch_dict = batch.to_dict()
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                batch_dict[key] = value.to(dtype=model_dtype)

        output = model(batch_dict, compute_force=True)
        predicted_energy = float(output["energy"].detach().cpu().reshape(-1)[0])
        predicted_forces = output["forces"].detach().cpu().numpy()
        reference_energy = float(atoms.info["REF_energy"])
        reference_forces = np.asarray(atoms.arrays["REF_forces"])

        if predicted_forces.shape != reference_forces.shape:
            raise ValueError(
                f"Force shape mismatch: model returned {predicted_forces.shape}, "
                f"reference has {reference_forces.shape}"
            )
        energy_squared_errors.append(((predicted_energy - reference_energy) / len(atoms)) ** 2)
        force_squared_error_sum += float(np.square(predicted_forces - reference_forces).sum())
        force_component_count += int(reference_forces.size)

    energy_rmse_mev_atom = float(np.sqrt(np.mean(energy_squared_errors)) * 1000.0)
    force_rmse_mev_angstrom = float(
        np.sqrt(force_squared_error_sum / force_component_count) * 1000.0
    )
    return energy_rmse_mev_atom, force_rmse_mev_angstrom


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is not available")
    device = torch.device(args.device)

    datasets = [
        ("train (300 K)", read_dataset(args.train)),
        ("validation (300 K)", read_dataset(args.val)),
        ("test (1200 K)", read_dataset(args.test)),
    ]

    print(
        f"{'Model':<16} | {'Dataset':<18} | {'Frames':>6} | "
        f"{'Energy RMSE (meV/atom)':>23} | {'Force RMSE (meV/Å)':>20}"
    )
    print("-" * 97)
    for model_name in args.models:
        model_path = MODEL_DIR / f"{model_name}_stagetwo.model"
        model = load_model(model_path, device)
        for dataset_name, frames in datasets:
            energy_rmse, force_rmse = evaluate_model(model, frames, device)
            print(
                f"{model_name:<16} | {dataset_name:<18} | {len(frames):>6d} | "
                f"{energy_rmse:>23.3f} | {force_rmse:>20.3f}"
            )
        del model


if __name__ == "__main__":
    main()
