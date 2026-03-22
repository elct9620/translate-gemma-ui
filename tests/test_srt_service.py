from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.srt_service import (
    SrtTranslationChunk,
    translate_srt,
)
from translate_gemma_ui.translator import FakeTranslator


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
