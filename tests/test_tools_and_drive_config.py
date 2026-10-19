import pytest

from deepfake_detection.data.drive_archives import load_drive_archives
from deepfake_detection.utils.tools import find_tool


def test_find_tool_prefers_explicit_path_then_environment_then_path(monkeypatch, tmp_path):
    exe = tmp_path / "sometool.exe"
    exe.write_text("")
    monkeypatch.setenv("SOMETOOL_PATH", str(exe))
    assert find_tool("sometool", "given.exe").name == "given.exe"
    assert find_tool("sometool") == exe
    monkeypatch.delenv("SOMETOOL_PATH")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_tool("sometool").name.lower().startswith("sometool")


def test_find_tool_names_what_is_missing(monkeypatch):
    monkeypatch.delenv("NOSUCHTOOL_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="NOSUCHTOOL_PATH"):
        find_tool("nosuchtool")


def test_drive_archives_read_ids_names_and_sizes(tmp_path):
    path = tmp_path / "drive.yaml"
    path.write_text("celebdfv2:\n  - {id: abc, name: a.zip, bytes: 5}\nhidf:\n  - {id: d, name: r.zip}\n  - {id: e, name: f.zip}\n", encoding="utf-8")
    table = load_drive_archives(path)
    assert table["celebdfv2"] == [("abc", "a.zip", 5)]
    assert [name for _, name, _ in table["hidf"]] == ["r.zip", "f.zip"]


def test_drive_archives_missing_file_is_an_error_unless_optional(tmp_path):
    missing = tmp_path / "none.yaml"
    with pytest.raises(FileNotFoundError, match="drive_archives.example.yaml"):
        load_drive_archives(missing)
    assert load_drive_archives(missing, required=False) == {}
