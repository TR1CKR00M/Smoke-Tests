from common.provenance import file_manifest


def test_file_manifest_records_hashes(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("hello", encoding="utf-8")
    second.write_text("world", encoding="utf-8")
    manifest = file_manifest([first, second], tmp_path)
    assert [entry["path"] for entry in manifest] == ["a.txt", "b.txt"]
    assert all(entry["bytes"] for entry in manifest)
