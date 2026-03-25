from pathlib import Path

BAT_PATH = Path(__file__).resolve().parent.parent / "start.bat"


def test_start_bat_sets_utf8_codepage():
    content = BAT_PATH.read_text(encoding="utf-8")
    assert "chcp 65001" in content, "start.bat must set codepage to UTF-8 (chcp 65001) for correct Chinese display"


def test_start_bat_uses_crlf_line_endings():
    raw = BAT_PATH.read_bytes()
    assert b"\r\n" in raw, "start.bat must use CRLF line endings for Windows compatibility"
    # Ensure there are no bare LF (without preceding CR)
    normalized = raw.replace(b"\r\n", b"")
    assert b"\n" not in normalized, "start.bat contains bare LF line endings mixed with CRLF"
