import pytest

from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.srt_service import (
    SrtTranslationChunk,
    translate_srt,
    translate_srt_full_file,
)
from translate_gemma_ui.translator import FakeTranslator, OutOfMemoryError


def _entry(index: int, text: str) -> SrtEntry:
    return SrtEntry(index=index, start_time=f"00:00:{index:02d},000", end_time=f"00:00:{index + 1:02d},000", text=text)


class TestTranslateSrt:
    def test_single_entry(self):
        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja"))
        assert len(chunks) > 0
        last = chunks[-1]
        assert isinstance(last, SrtTranslationChunk)
        assert last.entries[0].text != "Hello"

    def test_empty_text_preserved(self):
        entries = [_entry(1, ""), _entry(2, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja"))
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
        chunks = list(translate_srt(FailOnSecondTranslator(), entries, "en", "ja"))
        last = chunks[-1]
        assert last.entries[1].text == "B"
        assert last.entries[0].text != "A"
        assert last.entries[2].text != "C"

    def test_translates_only_target_text(self):
        class EchoTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                yield f"[translated] {text}"

        entries = [_entry(1, "Hello"), _entry(2, "World"), _entry(3, "Foo")]
        chunks = list(translate_srt(EchoTranslator(), entries, "en", "ja"))
        last = chunks[-1]

        assert last.entries[0].text == "[translated] Hello"
        assert last.entries[1].text == "[translated] World"
        assert last.entries[2].text == "[translated] Foo"

    def test_glossary_pre_mode(self, spy_translator):
        glossary = [("API", "應用程式介面")]
        entries = [_entry(1, "The API is ready")]
        list(translate_srt(spy_translator, entries, "en", "zh-TW", glossary=glossary, glossary_mode="pre"))
        assert "應用程式介面" in spy_translator.recorded_texts[0]
        assert "API" not in spy_translator.recorded_texts[0]

    def test_glossary_post_mode(self, spy_translator):
        glossary = [("API", "應用程式介面")]
        entries = [_entry(1, "The API is ready")]
        list(translate_srt(spy_translator, entries, "en", "zh-TW", glossary=glossary, glossary_mode="post"))
        assert "API" in spy_translator.recorded_texts[0]

    def test_glossary_none_works(self):
        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja", glossary=None))
        assert len(chunks) > 0


class TestTranslateSrtBatch:
    def test_batch_size_groups_entries(self, spy_translator):
        entries = [_entry(i, f"Line {i}") for i in range(1, 5)]
        list(translate_srt(spy_translator, entries, "en", "ja", batch_size=2))
        assert len(spy_translator.recorded_texts) == 2

    def test_batch_size_uses_separator(self, spy_translator):
        entries = [_entry(1, "Hello"), _entry(2, "World")]
        list(translate_srt(spy_translator, entries, "en", "ja", batch_size=2))
        assert spy_translator.recorded_texts[0] == "Hello\n\nWorld"

    def test_batch_size_one_is_default(self, spy_translator):
        entries = [_entry(1, "A"), _entry(2, "B")]
        list(translate_srt(spy_translator, entries, "en", "ja"))
        assert len(spy_translator.recorded_texts) == 2

    def test_batch_split_mismatch_preserves_originals(self):
        class SingleOutputTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                yield "single output without separator"

        entries = [_entry(1, "A"), _entry(2, "B")]
        chunks = list(translate_srt(SingleOutputTranslator(), entries, "en", "ja", batch_size=2))
        last = chunks[-1]
        assert last.entries[0].text == "single output without separator"
        assert last.entries[1].text == "B"

    def test_batch_error_preserves_originals_and_continues(self):
        class FailOnFirstBatchTranslator(FakeTranslator):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            def translate(self, text, source_lang, target_lang):
                self._call_count += 1
                if self._call_count == 1:
                    raise RuntimeError("Simulated failure")
                yield from super().translate(text, source_lang, target_lang)

        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C"), _entry(4, "D")]
        chunks = list(translate_srt(FailOnFirstBatchTranslator(), entries, "en", "ja", batch_size=2))
        last = chunks[-1]
        # First batch failed — originals preserved
        assert last.entries[0].text == "A"
        assert last.entries[1].text == "B"
        # Second batch succeeded — at least first entry in batch is translated
        assert last.entries[2].text != "C"

    def test_batch_progress_shows_batch_count(self):
        entries = [_entry(i, f"Line {i}") for i in range(1, 5)]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja", batch_size=2))
        last = chunks[-1]
        assert "2/2" in last.progress

    def test_batch_handles_uneven_groups(self, spy_translator):
        entries = [_entry(i, f"Line {i}") for i in range(1, 4)]
        list(translate_srt(spy_translator, entries, "en", "ja", batch_size=2))
        assert len(spy_translator.recorded_texts) == 2
        assert "Line 3" in spy_translator.recorded_texts[1]

    def test_batch_skips_empty_entries(self, spy_translator):
        entries = [_entry(1, "Hello"), _entry(2, ""), _entry(3, "World")]
        list(translate_srt(spy_translator, entries, "en", "ja", batch_size=3))
        assert spy_translator.recorded_texts[0] == "Hello\n\nWorld"


class TestTranslateSrtFullFile:
    def test_sends_serialized_srt(self, spy_translator):
        entries = [_entry(1, "Hello"), _entry(2, "World")]
        list(translate_srt_full_file(spy_translator, entries, "en", "ja"))
        assert "00:00:01,000 --> 00:00:02,000" in spy_translator.recorded_texts[0]
        assert "Hello" in spy_translator.recorded_texts[0]

    def test_parses_translated_output(self):
        class SrtEchoTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                yield "1\n00:00:01,000 --> 00:00:02,000\nTranslated\n"

        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt_full_file(SrtEchoTranslator(), entries, "en", "ja"))
        last = chunks[-1]
        assert last.entries[0].text == "Translated"

    def test_parse_failure_returns_originals(self):
        class BadOutputTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                yield "not valid srt at all"

        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt_full_file(BadOutputTranslator(), entries, "en", "ja"))
        last = chunks[-1]
        assert last.entries[0].text == "Hello"
        assert "失敗" in last.progress

    def test_context_length_exceeded_raises(self):
        class TinyTranslator(FakeTranslator):
            @property
            def max_tokens(self) -> int:
                return 5

        entries = [_entry(1, "This is a long subtitle that exceeds the context")]
        with pytest.raises(ValueError, match="批次模式"):
            list(translate_srt_full_file(TinyTranslator(), entries, "en", "ja"))

    def test_progress_shows_status(self):
        class SrtEchoTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                yield "1\n00:00:01,000 --> 00:00:02,000\nDone\n"

        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt_full_file(SrtEchoTranslator(), entries, "en", "ja"))
        last = chunks[-1]
        assert "翻譯完成" in last.progress

    def test_oom_propagates_instead_of_being_swallowed(self):
        class OOMTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                raise OutOfMemoryError("CUDA out of memory")

        entries = [_entry(1, "Hello")]
        with pytest.raises(OutOfMemoryError):
            list(translate_srt_full_file(OOMTranslator(), entries, "en", "ja"))
