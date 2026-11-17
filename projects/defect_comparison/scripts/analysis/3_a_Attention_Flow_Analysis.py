"""Compare representation effective rank at MACE embedding and product layers.

The script uses the same deterministic test configurations for all four
models, hooks the node embedding and final EquivariantProductBasisBlock,
collects per-atom features, centers each channel over atoms, and calculates
singular-value spectral effective rank and normalized channel utilization.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric


DEFAULT_MODELS = ["mace_k32_L0", "mace_k32_L2", "mace_k128_L0", "mace_k128_L2"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "attention_flow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=str(DATA_DIR / "defect_test_1200K.xyz"), help="Test structures used for all models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model name prefixes")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--min-atoms", type=int, default=1000, help="Minimum pooled atom rows for each feature matrix")
    parser.add_argument("--min-configs", type=int, default=10, help="Minimum number of test configurations")
    parser.add_argument("--preferred-max-configs", type=int, default=20, help="Preferred maximum; may be exceeded to meet --min-atoms")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="MACE inference device")
    return parser.parse_args()


def select_configurations(frames: list, min_atoms: int, min_configs: int, preferred_max_configs: int) -> tuple[list, list[int]]:
    if len(frames) < min_configs:
        raise ValueError(f"Test set has {len(frames)} frames; at least {min_configs} are required")
    if min_atoms < 1 or min_configs < 1 or preferred_max_configs < min_configs:
        raise ValueError("Require min_atoms > 0 and preferred_max_configs >= min_configs > 0")

    # Pick evenly spaced configurations to cover the full dataset deterministically.
    selected_count = min_configs
    while selected_count <= len(frames):
        indices = np.linspace(0, len(frames) - 1, selected_count, dtype=int).tolist()
        atom_count = sum(len(frames[index]) for index in indices)
        if atom_count >= min_atoms:
            break
        selected_count += 1
    else:
        raise ValueError(
            f"The full test set has only {sum(len(atoms) for atoms in frames)} atoms, "
            f"below the requested minimum {min_atoms}"
        )

    if selected_count > preferred_max_configs:
        warnings.warn(
            f"Selected {selected_count} configurations to meet the {min_atoms}-atom minimum; "
            f"this exceeds the preferred maximum of {preferred_max_configs}.",
            stacklevel=2,
        )
    return [frames[index] for index in indices], indices


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


def infer_width(model_name: str) -> int:
    match = re.search(r"k(\d+)_L\d+", model_name)
    if not match:
        raise ValueError(f"Cannot infer hidden channel width k from model name {model_name!r}")
    return int(match.group(1))


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"Could not find tensor in module output of type {type(value).__name__}")


def locate_node_embedding(model):
    module = getattr(model, "node_embedding", None)
    if isinstance(module, torch.nn.Module):
        return module
    matches = [(name, submodule) for name, submodule in model.named_modules() if "node_embedding" in name.lower()]
    if matches:
        return matches[0][1]
    raise AttributeError("Could not locate MACE node_embedding module")


def locate_last_product_block(model):
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if "equivariantproductbasisblock" in module.__class__.__name__.lower()
    ]
    if matches:
        return matches[-1][1]
    products = getattr(model, "products", None)
    if products is not None and len(products):
        return products[-1]
    raise AttributeError("Could not locate the final EquivariantProductBasisBlock")


def capture_feature_matrices(model: torch.nn.Module, frames: list, width: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    node_outputs = []
    product_outputs = []

    def save_node(_module, _inputs, output):
        node_outputs.append(first_tensor(output).detach().cpu())

    def save_product(_module, _inputs, output):
        product_outputs.append(first_tensor(output).detach().cpu())

    node_handle = locate_node_embedding(model).register_forward_hook(save_node)
    product_handle = locate_last_product_block(model).register_forward_hook(save_product)
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    node_rows, product_rows = [], []

    try:
        for frame_index, atoms in enumerate(frames):
            node_outputs.clear()
            product_outputs.clear()
            data = AtomicData.from_config(config_from_atoms(atoms), z_table=z_table, cutoff=cutoff)
            batch = torch_geometric.Batch.from_data_list([data]).to(device)
            batch_dict = batch.to_dict()
            model_dtype = next(model.parameters()).dtype
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    batch_dict[key] = value.to(dtype=model_dtype)

            # Force computation is unnecessary; the target modules are on the
            # energy forward path and are captured during a regular inference.
            with torch.no_grad():
                model(batch_dict, compute_force=False)

            if len(node_outputs) != 1 or len(product_outputs) != 1:
                raise RuntimeError(
                    f"Frame {frame_index}: expected one output from each hook, "
                    f"got node={len(node_outputs)}, product={len(product_outputs)}"
                )
            node = node_outputs[0]
            product = product_outputs[0]
            if node.ndim != 2 or node.shape[1] != width:
                raise ValueError(
                    f"Node embedding width mismatch: expected [N, {width}], got {tuple(node.shape)}"
                )
            if product.ndim != 2 or product.shape[1] < width:
                raise ValueError(
                    f"Product block output must have at least {width} columns; got {tuple(product.shape)}"
                )
            if node.shape[0] != len(atoms) or product.shape[0] != len(atoms):
                raise ValueError("Hook output atom count does not match input structure")

            # The configured hidden irreps list scalar 0e channels first;
            # retain those first k components for the like-width comparison.
            node_rows.append(node.numpy())
            product_rows.append(product[:, :width].numpy())
    finally:
        node_handle.remove()
        product_handle.remove()

    return np.concatenate(node_rows, axis=0), np.concatenate(product_rows, axis=0)


def spectral_effective_rank(matrix: np.ndarray) -> tuple[float, float, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"Expected a nonempty 2D feature matrix, got {matrix.shape}")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    if singular_values.size != matrix.shape[1]:
        raise ValueError(
            f"Feature matrix has {matrix.shape[0]} rows and {matrix.shape[1]} channels, "
            f"so SVD returned only {singular_values.size} singular values; need M >= k"
        )
    if singular_values[0] == 0:
        return 0.0, 0.0, singular_values

    tolerance = np.finfo(singular_values.dtype).eps * max(matrix.shape) * singular_values[0]
    spectral = singular_values[singular_values > tolerance]
    probabilities = spectral / spectral.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    effective_rank = float(np.exp(entropy))
    utilization = effective_rank / matrix.shape[1] * 100.0
    return effective_rank, float(utilization), singular_values


def main() -> None:
    args = parse_args()
    if len(args.models) != 4:
        raise ValueError("Expected exactly four model prefixes")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is not available")

    test_path = Path(args.test)
    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    all_frames = read(test_path, index=":")
    if not all_frames:
        raise ValueError(f"Test dataset contains no frames: {test_path}")
    selected_frames, selected_indices = select_configurations(
        all_frames, args.min_atoms, args.min_configs, args.preferred_max_configs
    )
    total_atoms = sum(len(atoms) for atoms in selected_frames)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Using {len(selected_frames)} shared configurations (indices {selected_indices}); "
        f"pooled atom count M={total_atoms}."
    )
    results = []
    for model_name in args.models:
        width = infer_width(model_name)
        model_path = MODEL_DIR / f"{model_name}_stagetwo.model"
        print(f"Analyzing {model_name} (k={width})...")
        model = load_model(model_path, device)
        node_matrix, product_matrix = capture_feature_matrices(model, selected_frames, width, device)
        del model

        for layer_name, matrix in [("node_embedding", node_matrix), ("final_product_scalar", product_matrix)]:
            effective_rank, utilization, singular_values = spectral_effective_rank(matrix)
            np.save(output_dir / f"{model_name}_{layer_name}_singular_values.npy", singular_values)
            results.append(
                {
                    "model": model_name,
                    "k": width,
                    "layer": layer_name,
                    "n_atoms": int(matrix.shape[0]),
                    "effective_rank": effective_rank,
                    "channel_utilization_percent": utilization,
                }
            )
            np.save(output_dir / f"{model_name}_{layer_name}_features.npy", matrix)

    results_path = output_dir / "effective_rank_results.csv"
    with results_path.open("w", encoding="utf-8") as file:
        file.write("model,k,layer,n_atoms,effective_rank,channel_utilization_percent\n")
        for row in results:
            file.write(
                f"{row['model']},{row['k']},{row['layer']},{row['n_atoms']},"
                f"{row['effective_rank']:.10g},{row['channel_utilization_percent']:.10g}\n"
            )

    print("\nAbsolute effective rank and normalized channel utilization:")
    print(f"{'Model':<16} | {'Layer':<22} | {'k':>4} | {'ERank':>10} | {'Utilization':>12}")
    print("-" * 75)
    for row in results:
        print(
            f"{row['model']:<16} | {row['layer']:<22} | {row['k']:>4d} | "
            f"{row['effective_rank']:>10.3f} | {row['channel_utilization_percent']:>11.2f}%"
        )

    deep_results = [row for row in results if row["layer"] == "final_product_scalar"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [row["model"].replace("mace_", "") for row in deep_results]
    axes[0].bar(labels, [row["effective_rank"] for row in deep_results], color="#4C78A8")
    axes[0].set_title("Final Product Block: Absolute Effective Rank")
    axes[0].set_ylabel("Effective rank")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [row["channel_utilization_percent"] for row in deep_results], color="#F58518")
    axes[1].set_title("Final Product Block: Channel Utilization")
    axes[1].set_ylabel("ERank / k (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "final_product_effective_rank.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved feature matrices, singular values, results, and plot under: {output_dir}")


if __name__ == "__main__":
    main()
