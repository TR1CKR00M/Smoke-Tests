#!/usr/bin/env python
"""Validate paths/config and, optionally, a completed output directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ase.io import read


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--check-output", action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    inp = Path(cfg["input_xyz"]).expanduser()
    model = Path(cfg["model_path"]).expanduser()
    out = Path(cfg["output_dir"]).expanduser()
    for label, path in (("input_xyz", inp), ("model_path", model)):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    atoms = read(inp, index=-1)
    if cfg.get("expected_atoms") and len(atoms) != int(cfg["expected_atoms"]):
        raise SystemExit(f"atom count mismatch: {len(atoms)}")
    if not cfg.get("initialize_momenta", False) and atoms.get_momenta() is None:
        raise SystemExit("input final frame has no momenta")
    print(f"input_sha256={digest(inp)}")
    print(f"model_sha256={digest(model)}")
    print(f"input_atoms={len(atoms)}")
    if a.check_output:
        status = out / "status.json"
        restart = out / "latest_restart.extxyz"
        if not status.is_file() or not restart.is_file():
            raise SystemExit("output is incomplete: status.json/latest_restart.extxyz missing")
        data = json.loads(status.read_text())
        if int(data["absolute_step"]) < int(cfg["start_step"]) + int(cfg["steps"]):
            raise SystemExit("output final step is below requested target")
        final = read(restart, index=-1)
        if final.get_momenta() is None:
            raise SystemExit("restart frame has no momenta")
        if len(final) != len(atoms):
            raise SystemExit("restart atom count differs from input")
        print(json.dumps(data, indent=2))
        print(f"segments={len(list(out.glob('segment_*.extxyz')))}")


if __name__ == "__main__":
    main()
