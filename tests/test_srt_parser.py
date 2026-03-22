import pytest

from translate_gemma_ui.srt_parser import SrtEntry, parse_srt, serialize_srt


class TestSrtEntry:
    def test_frozen(self):
        entry = SrtEntry(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="hello")
        with pytest.raises(AttributeError):
            entry.text = "changed"


class TestParseSrt:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="空"):
            parse_srt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="空"):
            parse_srt("   \n\n  ")

    def test_single_entry(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\nHello world\n"
        entries = parse_srt(content)
        assert len(entries) == 1
        assert entries[0] == SrtEntry(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Hello world")

    def test_multiple_entries(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond\n"
        entries = parse_srt(content)
        assert len(entries) == 2
        assert entries[0].text == "First"
        assert entries[1].text == "Second"
        assert entries[1].index == 2

    def test_multiline_subtitle(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\nLine one\nLine two\n"
        entries = parse_srt(content)
        assert entries[0].text == "Line one\nLine two"

    def test_empty_subtitle_text(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\n\n"
        entries = parse_srt(content)
        assert entries[0].text == ""

    def test_invalid_index_raises(self):
        content = "abc\n00:00:01,000 --> 00:00:02,000\nHello\n"
        with pytest.raises(ValueError, match="序號"):
            parse_srt(content)

    def test_invalid_timestamp_raises(self):
        content = "1\n00:00:01 --> 00:00:02\nHello\n"
        with pytest.raises(ValueError, match="時間軸"):
            parse_srt(content)

    def test_missing_timestamp_raises(self):
        content = "1\nHello\n"
        with pytest.raises(ValueError, match="時間軸"):
            parse_srt(content)

    def test_windows_line_endings(self):
        content = "1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n"
        entries = parse_srt(content)
        assert entries[0].text == "Hello"

    def test_trailing_newlines(self):
        content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n\n\n"
        entries = parse_srt(content)
        assert len(entries) == 1


class TestSerializeSrt:
    def test_empty_list(self):
        assert serialize_srt([]) == ""

    def test_single_entry(self):
        entry = SrtEntry(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Hello")
        result = serialize_srt([entry])
        assert result == "1\n00:00:01,000 --> 00:00:02,000\nHello\n"

    def test_multiple_entries(self):
        entries = [
            SrtEntry(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="First"),
            SrtEntry(index=2, start_time="00:00:03,000", end_time="00:00:04,000", text="Second"),
        ]
        result = serialize_srt(entries)
        assert result == "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond\n"

    def test_multiline_text(self):
        entry = SrtEntry(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="Line one\nLine two")
        result = serialize_srt([entry])
        assert "Line one\nLine two" in result

    def test_empty_text(self):
        entry = SrtEntry(index=1, start_time="00:00:01,000", end_time="00:00:02,000", text="")
        result = serialize_srt([entry])
        assert result == "1\n00:00:01,000 --> 00:00:02,000\n\n"

    def test_roundtrip(self):
        original = "1\n00:00:01,000 --> 00:00:02,000\nFirst line\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond line\n"
        entries = parse_srt(original)
        result = serialize_srt(entries)
        reparsed = parse_srt(result)
        assert entries == reparsed
