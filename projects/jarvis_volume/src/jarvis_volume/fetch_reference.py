from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.db.figshare import data as jarvis_data


def extract_jids(raw_predictions: Path) -> list[str]:
    frame = pd.read_csv(raw_predictions)
    jids = (
        frame["ID"]
        .dropna()
        .astype(str)
        .map(lambda value: value.split("_")[0])
        .drop_duplicates()
        .tolist()
    )
    return jids


def fetch_reference(
    raw_predictions: Path,
    reference_csv: Path,
) -> list[str]:
    jids = extract_jids(raw_predictions)
    dft_3d = jarvis_data("dft_3d")
    by_jid = {entry["jid"]: entry for entry in dft_3d}
    records: list[dict[str, object]] = []
    for jid in jids:
        entry = by_jid.get(jid)
        if entry is None:
            continue
        atoms = JarvisAtoms.from_dict(entry["atoms"])
        records.append(
            {
                "jid": jid,
                "dft_volume": atoms.volume,
            }
        )
    frame = pd.DataFrame(records).sort_values("jid")
    reference_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reference_csv, index=False)
    manifest = {
        "source": "JARVIS dft_3d",
        "source_tool": "jarvis-tools",
        "reference_rows": int(len(frame)),
        "sha256": hashlib.sha256(reference_csv.read_bytes()).hexdigest(),
    }
    manifest_path = reference_csv.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return [record["jid"] for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    args = parser.parse_args()
    fetch_reference(args.raw_predictions, args.reference_csv)


if __name__ == "__main__":
    main()
