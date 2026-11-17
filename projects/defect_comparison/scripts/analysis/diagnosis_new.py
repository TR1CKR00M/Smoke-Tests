"""Model error and defect-localization analysis for the MoS2 defect datasets.

This script evaluates MACE models on the 300 K training subset and 1200 K
out-of-distribution test set, then estimates how much high-angular-momentum
Cr SOAP features contribute to the test-set energy-error model.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric
from sklearn.ensemble import GradientBoostingRegressor
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(DATA_DIR / "defect_train_300K.xyz"), help="300 K training XYZ")
    parser.add_argument("--test", default=str(DATA_DIR / "defect_test_1200K.xyz"), help="1200 K test XYZ")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mace_k128_L0", "mace_k32_L0", "mace_k128_L2", "mace_k32_L2"],
        help="Model name prefixes (expects <name>_stagetwo.model)",
    )
    parser.add_argument("--train-frames", type=int, default=30, help="Number of initial training frames")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    # e3nn may call torch.jit.load without map_location; force its modules to CPU.
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


def evaluate_and_gradcam(model_path: Path, frames, device: torch.device):
    model = load_model(model_path, device)
    model_dtype = next(model.parameters()).dtype
    torch.set_default_dtype(model_dtype)

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)
    e_errors, f_errors, selectivity_ratios = [], [], []
    activations = []

    def forward_hook(_module, _inputs, output):
        feat = output[0] if isinstance(output, tuple) else output
        activations.append(feat)
        feat.retain_grad()

    handle = model.interactions[-1].register_forward_hook(forward_hook)
    try:
        for atoms in frames:
            activations.clear()
            config = config_from_atoms(atoms)
            data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            batch = torch_geometric.Batch.from_data_list([data]).to(device)
            batch_dict = batch.to_dict()
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    batch_dict[key] = value.to(dtype=model_dtype)

            # Compute energy and forces with gradients enabled for Grad-CAM.
            model.zero_grad(set_to_none=True)
            output = model(batch_dict, compute_force=True, training=True)
            pred_e = output["energy"]
            pred_f = output["forces"].detach().cpu().numpy()

            ref_e = atoms.info["REF_energy"]
            ref_f = atoms.arrays["REF_forces"]
            e_errors.append(abs(pred_e.item() - ref_e) / len(atoms) * 1000.0)
            f_errors.append(np.sqrt(np.mean((pred_f - ref_f) ** 2)) * 1000.0)

            pred_e.backward()
            activation = activations[0]
            if activation.grad is None:
                raise RuntimeError("Could not capture interaction activations for Grad-CAM")

            # The final interaction features can contain scalar and higher-order
            # equivariant channels. Use per-atom L2 norms so the score is
            # invariant to orthogonal SO(3) rotations of the feature components.
            grad_norm = torch.linalg.vector_norm(activation.grad, ord=2, dim=-1)
            activation_norm = torch.linalg.vector_norm(activation, ord=2, dim=-1)
            cam = (grad_norm * activation_norm).detach().cpu().numpy()

            cr_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
            i_list, j_list = neighbor_list("ij", atoms, cutoff=3.5)
            defect_zone = set(cr_indices)
            for i_atom, j_atom in zip(i_list, j_list):
                if i_atom in cr_indices:
                    defect_zone.add(j_atom)
            bulk_zone = [i for i in range(len(atoms)) if i not in defect_zone]
            if not defect_zone or not bulk_zone:
                raise ValueError("Grad-CAM selectivity requires nonempty defect and bulk regions")
            ratio = (np.mean(cam[list(defect_zone)]) + 1e-8) / (np.mean(cam[bulk_zone]) + 1e-8)
            selectivity_ratios.append(ratio)
    finally:
        handle.remove()

    return np.asarray(e_errors), np.asarray(f_errors), float(np.mean(selectivity_ratios))


def build_soap_features(frames):
    species = ["Cr", "Mo", "S"]
    descriptor = SOAP(species=species, r_cut=5.5, n_max=6, l_max=3, average="inner", periodic=True)
    features = np.asarray(descriptor.create(frames))

    feature_names = []
    for first in range(len(species)):
        for second in range(first, len(species)):
            pair_count = 21 if first == second else 36
            for pair_index in range(pair_count):
                for angular_momentum in range(4):
                    feature_names.append(
                        f"{species[first]}-{species[second]}_n{pair_index}_l{angular_momentum}"
                    )
    return features, feature_names[: features.shape[1]]


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    train_atoms = read(args.train, index=":")
    test_atoms = read(args.test, index=":")
    if not train_atoms or not test_atoms:
        raise ValueError("Training and test XYZ files must both contain frames")
    train_subset = train_atoms[: min(args.train_frames, len(train_atoms))]
    X_soap, feature_names = build_soap_features(test_atoms)

    print(
        f"{'Model Name':<15} | {'300K F-RMSE':<12} | {'1200K F-RMSE':<12} | "
        f"{'Overfit Ratio':<13} | {'Grad-CAM S_def':<14} | {'SHAP Cr(l>=2) Share'}"
    )
    print("-" * 95)

    high_l_cr_mask = np.asarray(
        ["Cr" in name and ("_l2" in name or "_l3" in name) for name in feature_names], dtype=bool
    )
    for model_name in args.models:
        model_path = MODEL_DIR / f"{model_name}_stagetwo.model"
        _, train_force_errors, _ = evaluate_and_gradcam(model_path, train_subset, device)
        test_energy_errors, test_force_errors, defect_selectivity = evaluate_and_gradcam(
            model_path, test_atoms, device
        )
        train_force_mean = float(np.mean(train_force_errors))
        test_force_mean = float(np.mean(test_force_errors))
        overfit_ratio = test_force_mean / train_force_mean if train_force_mean else float("inf")

        regressor = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        regressor.fit(X_soap, test_energy_errors)
        shap_values = np.abs(shap.TreeExplainer(regressor).shap_values(X_soap)).mean(axis=0)
        high_l_share = float(shap_values[high_l_cr_mask].sum() / (shap_values.sum() + 1e-8) * 100.0)

        print(
            f"{model_name:<15} | {train_force_mean:>8.1f} meV/A | {test_force_mean:>8.1f} meV/A | "
            f"{overfit_ratio:>11.2f}x | {defect_selectivity:>12.2f}x | {high_l_share:>10.1f}%"
        )


if __name__ == "__main__":
    main()
