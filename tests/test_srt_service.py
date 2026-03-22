from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.srt_service import (
    SrtTranslationChunk,
    build_context_text,
    extract_target_translation,
    translate_srt,
)
from translate_gemma_ui.translator import FakeTranslator


def _entry(index: int, text: str) -> SrtEntry:
    return SrtEntry(index=index, start_time=f"00:00:{index:02d},000", end_time=f"00:00:{index + 1:02d},000", text=text)


class TestBuildContextText:
    def test_no_context(self):
        entries = [_entry(1, "Hello")]
        text, idx = build_context_text(entries, 0, context_size=0)
        assert text == "Hello"
        assert idx == 0

    def test_context_at_start(self):
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C"), _entry(4, "D")]
        text, idx = build_context_text(entries, 0, context_size=3)
        lines = text.split("\n")
        assert lines[0] == "A"
        assert idx == 0

    def test_context_at_end(self):
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        text, idx = build_context_text(entries, 2, context_size=3)
        lines = text.split("\n")
        assert lines[idx] == "C"

    def test_context_in_middle(self):
        entries = [_entry(i, f"Line{i}") for i in range(1, 8)]
        text, idx = build_context_text(entries, 3, context_size=2)
        lines = text.split("\n")
        assert len(lines) == 5  # 2 before + target + 2 after
        assert lines[idx] == "Line4"

    def test_skips_empty_text_in_context(self):
        entries = [_entry(1, "A"), _entry(2, ""), _entry(3, "C")]
        text, idx = build_context_text(entries, 2, context_size=2)
        lines = text.split("\n")
        assert lines[idx] == "C"


class TestExtractTargetTranslation:
    def test_matching_line_count(self):
        translated = "AA\nBB\nCC"
        result = extract_target_translation(translated, target_line_idx=1, expected_lines=3)
        assert result == "BB"

    def test_mismatched_line_count_returns_full(self):
        translated = "Only one line"
        result = extract_target_translation(translated, target_line_idx=1, expected_lines=3)
        assert result == "Only one line"


class TestTranslateSrt:
    def test_single_entry(self):
        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja"))
        assert len(chunks) > 0
        last = chunks[-1]
        assert isinstance(last, SrtTranslationChunk)
        assert last.entries[0].text != "Hello"  # translated

    def test_empty_text_preserved(self):
        entries = [_entry(1, ""), _entry(2, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja", context_size=0))
        last = chunks[-1]
        assert last.entries[0].text == ""
        assert last.entries[1].text != "Hello"

    def test_progress_tracking(self):
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja"))
        last = chunks[-1]
        assert "3/3" in last.progress

    def test_translation_error_continues(self):
        class FailOnSecondTranslator(FakeTranslator):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            def translate(self, text, source_lang, target_lang):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("Simulated failure")
                yield from super().translate(text, source_lang, target_lang)

        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        chunks = list(translate_srt(FailOnSecondTranslator(), entries, "en", "ja", context_size=0))
        last = chunks[-1]
        # Second entry should keep original text
        assert last.entries[1].text == "B"
        # First and third should be translated
        assert last.entries[0].text != "A"
        assert last.entries[2].text != "C"

    def test_context_size_zero(self):
        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja", context_size=0))
        last = chunks[-1]
        assert last.entries[0].text != "Hello"
