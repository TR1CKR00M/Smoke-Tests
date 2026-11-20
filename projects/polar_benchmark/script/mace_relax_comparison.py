"""Relax ShakeNBreak defect structures with MACE models and compare with DFT.

For every (model, charge state, distortion) the initial POSCAR is relaxed with
BFGS and compared with the DFT CONTCAR/OUTCAR in the same directory. Results:

* ``results/processed/relax_per_structure.csv``: one row per relaxation
  (MLIP energy, BFGS convergence/steps, DFT final energy and convergence,
  minimum-image RMSD over all atoms and over atoms near the defect, errors).
* ``results/processed/relax_per_charge.csv``: per (model, charge) the MLIP and
  DFT ground states (lowest energy, chosen independently), the ground-state hit,
  the RMSD between the two ground-state structures and the Spearman rank
  correlation of the distortion energies.
* ``results/raw/<model>/<charge>_<distortion>.extxyz``: relaxed structures.
* ``results/processed/relax_run_metadata.json``: command, versions, model sha256.

Figures are produced from the CSV by ``script/plot_relax_comparison.py``.

Example (MACE-MH-1 baseline, CPU, all charges and distortions):

    python projects/polar_benchmark/script/mace_relax_comparison.py --device cpu

A MACE-Polar checkpoint is plugged in with ``--model NAME=polar:<key-or-path>``,
e.g. ``--model MACE-Polar-1-S=polar:polar-1-s``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import re
import shlex
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_benchmark.benchmark import (  # noqa: E402
    PER_CHARGE_COLUMNS,
    PER_STRUCTURE_COLUMNS,
    group_rows,
    summarize_group,
)
from polar_benchmark.manifest import sha256_file  # noqa: E402
from polar_benchmark.vasp_io import (  # noqa: E402
    find_structure,
    parse_defect_site_comment,
    parse_outcar,
    read_poscar_comment,
)

LOGGER = logging.getLogger("mace_relax_comparison")

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/1_FM_benchmark/V_Sb1/DFT_PBE"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model/mace-mh-1.model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
CHARGE_VALUES = {"mi3": -3, "mi2": -2, "mi1": -1, "neu": 0, "pl1": 1}
DEFAULT_DISTORTIONS = tuple(
    f"D_{d}" for d in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
)
DEFAULT_LOCAL_CUTOFF = 4.0
DISPERSION_DAMPING = {"d3-zero": "zero", "d3-bj": "bj"}
POLAR_PREFIX = "polar:"


def parse_model_specs(values: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ``NAME=MODEL`` arguments."""
    specs = []
    for value in values:
        name, separator, model = value.partition("=")
        if not separator or not name or not model:
            raise argparse.ArgumentTypeError(f"Model must be NAME=MODEL, got: {value!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
            raise argparse.ArgumentTypeError(
                f"Model NAME may only contain letters, digits and _.+- (used as a folder): {name!r}"
            )
        specs.append((name, model))
    return specs


# --------------------------------------------------------------------------- MACE


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but this PyTorch build has no CUDA.")
    return device


def build_calculator(spec: str, device: str, dtype: str, head: str | None, dispersion: str):
    """Create a MACE calculator; ``head`` is only applied to multi-head models."""
    try:
        import torch
        from ase.calculators.mixing import SumCalculator
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise RuntimeError(
            "This script needs torch, mace-torch, ase, numpy, scipy and pandas "
            "(see projects/polar_benchmark/requirements.txt)."
        ) from exc

    if spec.startswith(POLAR_PREFIX):
        try:
            from mace.calculators import mace_polar
        except ImportError as exc:
            raise RuntimeError("This mace-torch version has no mace_polar().") from exc
        calc = mace_polar(model=spec[len(POLAR_PREFIX) :], device=device, default_dtype=dtype)
    else:
        calc = mace_mp(model=spec, device=device, default_dtype=dtype, head=head)

    available = list(getattr(calc, "available_heads", []) or [])
    if head is not None and spec.startswith(POLAR_PREFIX):
        LOGGER.info("--head is ignored for MACE-Polar models (heads: %s)", available)
    elif head is not None and len(available) > 1 and head not in available:
        # MACECalculator silently falls back to the last head; refuse instead.
        raise RuntimeError(f"Head {head!r} not in model heads {available}")
    elif head is not None and len(available) <= 1:
        LOGGER.info("Model has a single head %s; --head %s is not used", available, head)
    LOGGER.info("Using head %s (available: %s)", getattr(calc, "head", None), available)

    if dispersion == "none":
        return calc
    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:
        raise RuntimeError("--dispersion needs the torch-dftd package.") from exc
    d3 = TorchDFTD3Calculator(
        device=device,
        damping=DISPERSION_DAMPING[dispersion],
        xc="pbe",
        dtype=torch.float64 if dtype == "float64" else torch.float32,
    )
    LOGGER.info("Adding DFT-D3 (%s damping, PBE parameters)", DISPERSION_DAMPING[dispersion])
    return SumCalculator([calc, d3])


# ------------------------------------------------------------------- one structure


def resolve_defect_site(
    poscar: Path, override: list[float] | None
) -> tuple[list[float] | None, str]:
    if override is not None:
        return list(override), "cli"
    site = parse_defect_site_comment(read_poscar_comment(poscar))
    return (list(site), "poscar_comment") if site else (None, "")


def dft_reference(distortion_dir: Path, include_unconverged: bool) -> dict[str, Any]:
    outcar = find_structure(distortion_dir, "OUTCAR")
    if outcar is None:
        return {"dft_outcar": "", "dft_converged": False, "dft_excluded_reason": "no OUTCAR"}
    summary = parse_outcar(outcar)
    reason = summary.unconverged_reason
    if reason and include_unconverged:
        LOGGER.warning("Using unconverged DFT reference %s: %s", outcar, reason)
        reason = None
    elif reason:
        LOGGER.warning("Excluding DFT reference %s: %s", outcar, reason)
    return {
        "dft_outcar": outcar,
        "dft_energy_eV": summary.final_energy_eV,
        "dft_converged": summary.is_converged,
        "dft_excluded_reason": reason or "",
        "dft_nelect": summary.nelect,
        "dft_magnetization": summary.final_magnetization,
    }


def relax_one(
    calc, poscar: Path, charge: int, args: argparse.Namespace
) -> tuple[Any, dict[str, Any]]:
    from ase.io import read
    from ase.optimize import BFGS

    atoms = read(poscar, format="vasp")
    if args.set_total_charge:
        atoms.info["total_charge"] = charge
    atoms.calc = calc
    optimizer = BFGS(atoms, logfile=None)
    converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
    energy = float(atoms.get_potential_energy())
    return atoms, {
        "mlip_energy_eV": energy,
        "bfgs_converged": converged,
        "bfgs_steps": optimizer.nsteps,
    }


def save_relaxed(atoms, path: Path, row: dict[str, Any]) -> None:
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = atoms.copy()
    snapshot.calc = None
    snapshot.info.update(
        {
            "model": row["model"],
            "charge": row["charge"],
            "distortion": row["distortion"],
            "mlip_energy_eV": row["mlip_energy_eV"],
            "bfgs_converged": row["bfgs_converged"],
        }
    )
    write(path, snapshot, format="extxyz")


def compare_with_dft(relaxed, poscar: Path, contcar: Path, row: dict[str, Any], cutoff: float):
    from ase.io import read
    from polar_benchmark.metrics import atoms_near_site, mic_rmsd

    reference = read(contcar, format="vasp")
    initial = read(poscar, format="vasp")
    site = row["defect_site_frac"]
    local = atoms_near_site(reference, site, cutoff) if site is not None else None
    row["n_local_atoms"] = len(local) if local is not None else None
    row["rmsd_all_A"] = mic_rmsd(relaxed, reference)
    row["rmsd_local_A"] = mic_rmsd(relaxed, reference, local) if local is not None else math.nan
    row["rmsd_initial_vs_dft_A"] = mic_rmsd(initial, reference)


def evaluate_structure(calc, model_name, charge_label, distortion_dir, args, relaxed_store):
    distortion = distortion_dir.name
    charge = CHARGE_VALUES[charge_label]
    row: dict[str, Any] = {
        "model": model_name,
        "charge_label": charge_label,
        "charge": charge,
        "distortion": distortion,
        "error": "",
    }
    poscar = find_structure(distortion_dir, "POSCAR")
    contcar = find_structure(distortion_dir, "CONTCAR")
    row["poscar"] = poscar
    row["dft_contcar"] = contcar
    row.update(dft_reference(distortion_dir, args.include_unconverged_refs))
    if poscar is None or contcar is None:
        row["error"] = "missing POSCAR or CONTCAR"
        return row
    site, site_source = resolve_defect_site(poscar, args.defect_site)
    row["defect_site_frac"] = site
    if site is None:
        LOGGER.warning("No defect site for %s; local RMSD will be NaN", poscar)
    try:
        relaxed, result = relax_one(calc, poscar, charge, args)
        row.update(result)
        compare_with_dft(relaxed, poscar, contcar, row, args.local_cutoff)
        out = args.output_dir / "raw" / model_name / f"{charge_label}_{distortion}.extxyz"
        save_relaxed(relaxed, out, row)
        row["relaxed_structure"] = out
        relaxed_store[(model_name, charge_label, distortion)] = relaxed
    except Exception as exc:  # recorded per structure, never silently dropped
        row["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.error(
            "Relaxation failed for %s/%s/%s: %s", model_name, charge_label, distortion, exc
        )
    LOGGER.info(
        "%s %s %s: E=%s conv=%s steps=%s rmsd=%s (site from %s)",
        model_name,
        charge_label,
        distortion,
        row.get("mlip_energy_eV"),
        row.get("bfgs_converged"),
        row.get("bfgs_steps"),
        row.get("rmsd_all_A"),
        site_source or "none",
    )
    return row


# ---------------------------------------------------------------------- driver


def run_model(model_name, spec, args, relaxed_store) -> list[dict[str, Any]]:
    device = resolve_device(args.device)
    LOGGER.info("Evaluating %s (%s) on %s", model_name, spec, device)
    calc = build_calculator(spec, device, args.dtype, args.head, args.dispersion)
    rows = []
    for charge_label in args.charges:
        charge_dir = args.data_root / charge_label
        if not charge_dir.is_dir():
            LOGGER.warning("Skipping %s: directory %s does not exist", charge_label, charge_dir)
            continue
        for distortion in args.distortions:
            if args.max_structures is not None and len(rows) >= args.max_structures:
                return rows
            distortion_dir = charge_dir / distortion
            if not distortion_dir.is_dir():
                LOGGER.warning("Skipping missing %s", distortion_dir)
                continue
            rows.append(
                evaluate_structure(
                    calc, model_name, charge_label, distortion_dir, args, relaxed_store
                )
            )
    return rows


def add_ground_state_rmsd(summary: dict[str, Any], relaxed_store, args) -> None:
    """RMSD between the MLIP ground-state structure and the DFT ground-state CONTCAR."""
    from ase.io import read
    from polar_benchmark.metrics import atoms_near_site, mic_rmsd

    key = (summary["model"], summary["charge_label"], summary["mlip_ground_state"])
    if not summary["dft_ground_state"] or key not in relaxed_store:
        return
    dft_dir = args.data_root / summary["charge_label"] / summary["dft_ground_state"]
    contcar = find_structure(dft_dir, "CONTCAR")
    poscar = find_structure(dft_dir, "POSCAR")
    if contcar is None or poscar is None:
        return
    reference = read(contcar, format="vasp")
    relaxed = relaxed_store[key]
    summary["rmsd_gs_all_A"] = mic_rmsd(relaxed, reference)
    site, _ = resolve_defect_site(poscar, args.defect_site)
    if site is not None:
        local = atoms_near_site(reference, site, args.local_cutoff)
        summary["rmsd_gs_local_A"] = mic_rmsd(relaxed, reference, local)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        try:
            return value.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(value)
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.10g}"
    if isinstance(value, list):
        return " ".join(f"{v:.6g}" for v in value)
    return str(value)


def write_rows(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})


def write_metadata(args, model_specs, path: Path) -> None:
    import importlib.metadata

    versions = {}
    for package in ("mace-torch", "torch", "ase", "numpy", "scipy", "torch-dftd"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    models = []
    for name, spec in model_specs:
        local = Path(spec.removeprefix(POLAR_PREFIX))
        models.append(
            {
                "name": name,
                "spec": spec,
                "sha256": sha256_file(local) if local.is_file() else None,
            }
        )
    metadata = {
        "command": shlex.join([Path(sys.executable).name, *sys.argv]),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "models": models,
        "settings": {
            key: _cell(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME=MODEL",
        help=(
            "Model to evaluate (repeatable). MODEL is a path or mace_mp name; prefix with "
            "'polar:' for MACE-Polar (key such as polar-1-s, or a path). "
            "Default: MACE-MH-1=model/mace-mh-1.model"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--head",
        default="omat_pbe",
        help="Head for multi-head models such as MH-1 (default omat_pbe, plain PBE without "
        "D3). Ignored for single-head and MACE-Polar models; an unknown head is an error.",
    )
    parser.add_argument(
        "--dispersion",
        choices=("none", "d3-zero", "d3-bj"),
        default="none",
        help="Add DFT-D3 on top of the MLIP. The reference DFT used PBE+D3 with zero "
        "damping (IVDW=11), so d3-zero is the like-for-like choice. Default: none.",
    )
    parser.add_argument(
        "--d3",
        dest="dispersion",
        action="store_const",
        const="d3-zero",
        help="Shorthand for --dispersion d3-zero (matches IVDW=11).",
    )
    parser.add_argument(
        "--charges",
        nargs="+",
        choices=tuple(CHARGE_VALUES),
        default=list(CHARGE_VALUES),
    )
    parser.add_argument(
        "--distortions",
        nargs="+",
        default=list(DEFAULT_DISTORTIONS),
        help="Distortion folders, e.g. D_0 D_10 (also Unperturbed / Rattled where present).",
    )
    parser.add_argument(
        "--max-structures", type=int, default=None, help="Per model, for smoke tests."
    )
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--local-cutoff",
        type=float,
        default=DEFAULT_LOCAL_CUTOFF,
        help="Radius (Å) around the vacancy site for the defect-local RMSD.",
    )
    parser.add_argument(
        "--defect-site",
        type=float,
        nargs=3,
        metavar=("FX", "FY", "FZ"),
        default=None,
        help="Fractional vacancy site. Default: parsed from the ShakeNBreak POSCAR comment "
        "(e.g. '~[0.4,0.5,0.4]', rounded to 0.1, i.e. approximate).",
    )
    parser.add_argument(
        "--no-total-charge",
        dest="set_total_charge",
        action="store_false",
        help="Do not set atoms.info['total_charge'] (read by charge-aware models).",
    )
    parser.add_argument(
        "--include-unconverged-refs",
        action="store_true",
        help="Use DFT references that did not reach the required accuracy (logged).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional figure path; if given, the figure is plotted from the CSV at the end.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if not args.data_root.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_root}")
    if args.model is None and not DEFAULT_MODEL_PATH.is_file():
        raise SystemExit(f"Default MH-1 model not found: {DEFAULT_MODEL_PATH}")
    model_specs = parse_model_specs(args.model or [f"MACE-MH-1={DEFAULT_MODEL_PATH}"])

    relaxed_store: dict = {}
    rows: list[dict[str, Any]] = []
    for name, spec in model_specs:
        rows.extend(run_model(name, spec, args, relaxed_store))

    summaries = []
    for group in group_rows(rows).values():
        summary = summarize_group(group)
        add_ground_state_rmsd(summary, relaxed_store, args)
        summaries.append(summary)

    processed = args.output_dir / "processed"
    write_rows(rows, processed / "relax_per_structure.csv", PER_STRUCTURE_COLUMNS)
    write_rows(summaries, processed / "relax_per_charge.csv", PER_CHARGE_COLUMNS)
    write_metadata(args, model_specs, processed / "relax_run_metadata.json")
    LOGGER.info(
        "Wrote %d structures and %d charge summaries to %s", len(rows), len(summaries), processed
    )

    if args.output is not None:
        from plot_relax_comparison import plot_from_csv

        plot_from_csv(processed / "relax_per_charge.csv", args.output, show=args.show)


if __name__ == "__main__":
    main()
