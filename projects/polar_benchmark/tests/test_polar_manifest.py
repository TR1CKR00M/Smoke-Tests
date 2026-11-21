import hashlib

import pytest
from polar_benchmark.manifest import (
    build_manifest_rows,
    build_model_registry,
    classify_parts,
)

FAKE_OUTCAR = """ vasp.6.4.3 19Mar24
   NELECT = 330
   ions per type =              23  36
   NSW    =    500    number of steps for IOM
   ZVAL   =   5.00  6.00
   GGA     = PE    functional components
   LHFCALC =     F    Hartree Fock is set to
   IVDW    =    11    specifies the selected vdW correction
 generate k-points for:    1    1    1
 number of electron     330.0000000 magnetization       2.0000000
  free  energy   TOTEN  =      -252.0 eV
  energy  without entropy=  -252.0  energy(sigma->0) =      -252.05
 reached required accuracy - stopping structural energy minimisation
"""


@pytest.fixture
def project(tmp_path):
    data = tmp_path / "data"
    run = data / "1_FM_benchmark/V_Sb1/DFT_PBE/pl1/D_-30"
    run.mkdir(parents=True)
    (run / "OUTCAR_-30").write_text(FAKE_OUTCAR)
    (run / "POSCAR_-30").write_text("-30.0% N(Distort)=3 ~[0.4,0.5,0.4]\n")
    (run / ".DS_Store").write_text("junk")
    (tmp_path / "model").mkdir()
    (tmp_path / "model/a.model").write_bytes(b"weights")
    return tmp_path


def test_classify_parts_for_each_stage():
    stage1 = classify_parts(["1_FM_benchmark", "V_Sb1", "DFT_PBE", "mi3", "Rattled", "OUTCAR_Ra"])
    stage3 = classify_parts(["3_MF", "MF", "MF_MACE", "VSe1_neu", "Prediction", "OUTCAR_x"])

    assert stage1 == {
        "stage": "1_FM_benchmark",
        "defect": "V_Sb1",
        "charge_label": "mi3",
        "distortion": "Rattled",
    }
    assert stage3["defect"] == "V_Se1"
    assert stage3["charge_label"] == "neu"
    assert stage3["distortion"] == ""


def test_manifest_rows_parse_outcar_and_hash_files(project):
    rows = build_manifest_rows(project / "data", relative_to=project)

    by_type = {r["file_type"]: r for r in rows}
    assert set(by_type) == {"OUTCAR", "POSCAR"}  # .DS_Store is ignored
    outcar = by_type["OUTCAR"]
    assert outcar["path"] == "data/1_FM_benchmark/V_Sb1/DFT_PBE/pl1/D_-30/OUTCAR_-30"
    assert outcar["stage"] == "1_FM_benchmark"
    assert outcar["charge_label"] == "pl1"
    assert outcar["charge_from_label"] == "1"
    assert outcar["charge_from_nelect"] == "1"
    assert outcar["charge_consistent"] == "True"
    assert outcar["distortion"] == "D_-30"
    assert outcar["converged"] == "True"
    assert outcar["final_magnetization"] == "2"
    assert outcar["functional"] == "PBE"
    assert outcar["dispersion"] == "D3(zero)"
    assert outcar["kpoint_grid"] == "1x1x1"
    assert outcar["sha256"] == hashlib.sha256(FAKE_OUTCAR.encode()).hexdigest()
    assert by_type["POSCAR"]["nelect"] == ""


def test_model_registry_preserves_hand_filled_columns(project):
    existing = {"model/a.model": {"source_url": "https://example.org/a", "license": "MIT"}}

    rows = build_model_registry(project, existing)

    assert len(rows) == 1
    assert rows[0]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert rows[0]["source_url"] == "https://example.org/a"
    assert rows[0]["license"] == "MIT"
    assert rows[0]["version"] == "TODO"
