from pathlib import Path

import pandas as pd
from src.reproduce_plots import read_logs
from src.run_md import load_previous_csv_logs

ROOT = Path(__file__).resolve().parents[1]


def test_committed_logs_have_manifest_compatible_schema():
    logs = sorted((ROOT / "results" / "raw").glob("*_log.csv"))
    data = read_logs(logs)
    assert len(data) == 4
    assert all(len(frame) > 10 for frame in data.values())


def test_group_metadata_resolves_to_distinct_models():
    metadata = ROOT / "data" / "metadata" / "group_1.txt"
    entries = [line.strip() for line in metadata.read_text().splitlines() if line.strip()]
    assert entries
    names = [Path(entry).parent.name for entry in entries]
    assert len(set(names)) == len(names)
    assert all(Path(ROOT / entry).is_file() for entry in entries)


def test_log_schema_is_explicit():
    frame = pd.read_csv(sorted((ROOT / "results" / "raw").glob("*_log.csv"))[0])
    assert list(frame.columns) == [
        "time_fs",
        "potential_energy_eV",
        "temperature_K",
        "pressure_GPa",
    ]


def test_restart_loader_skips_csv_header(tmp_path):
    path = tmp_path / "thermo_log.csv"
    path.write_text(
        "time_fs,potential_energy_eV,temperature_K,pressure_GPa\n"
        "0.0,-10.0,300.0,0.1\n",
        encoding="utf-8",
    )
    rows = load_previous_csv_logs(path)
    assert rows == [(0.0, -10.0, 300.0, 0.1)]
