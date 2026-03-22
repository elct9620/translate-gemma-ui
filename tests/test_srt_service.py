from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.srt_service import (
    SrtTranslationChunk,
    _build_context,
    translate_srt,
)
from translate_gemma_ui.translator import FakeTranslator, TranslationContext


def _entry(index: int, text: str) -> SrtEntry:
    return SrtEntry(index=index, start_time=f"00:00:{index:02d},000", end_time=f"00:00:{index + 1:02d},000", text=text)


class TestBuildContext:
    def test_returns_none_when_context_size_zero(self):
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        assert _build_context(entries, 1, context_size=0) is None

    def test_returns_none_when_no_surrounding_text(self):
        entries = [_entry(1, "A")]
        assert _build_context(entries, 0, context_size=3) is None

    def test_builds_context_at_start(self):
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        ctx = _build_context(entries, 0, context_size=2)
        assert ctx is not None
        assert ctx.previous == []
        assert ctx.following == ["B", "C"]

    def test_builds_context_at_end(self):
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        ctx = _build_context(entries, 2, context_size=2)
        assert ctx is not None
        assert ctx.previous == ["A", "B"]
        assert ctx.following == []

    def test_builds_context_in_middle(self):
        entries = [_entry(i, f"Line{i}") for i in range(1, 8)]
        ctx = _build_context(entries, 3, context_size=2)
        assert ctx is not None
        assert ctx.previous == ["Line2", "Line3"]
        assert ctx.following == ["Line5", "Line6"]

    def test_skips_empty_text_entries(self):
        entries = [_entry(1, "A"), _entry(2, ""), _entry(3, "C")]
        ctx = _build_context(entries, 2, context_size=2)
        assert ctx is not None
        assert ctx.previous == ["A"]
        assert ctx.following == []


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

            def translate(self, text, source_lang, target_lang, context=None):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("Simulated failure")
                yield from super().translate(text, source_lang, target_lang, context=context)

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

    def test_passes_context_to_translator(self, spy_translator):
        """Verify that translate_srt passes TranslationContext to the translator."""
        entries = [_entry(1, "A"), _entry(2, "B"), _entry(3, "C")]
        list(translate_srt(spy_translator, entries, "en", "ja", context_size=1))

        # Entry A: no previous, following=["B"]
        assert isinstance(spy_translator.recorded_contexts[0], TranslationContext)
        assert spy_translator.recorded_contexts[0].previous == []
        assert spy_translator.recorded_contexts[0].following == ["B"]

        # Entry B: previous=["A"], following=["C"]
        assert spy_translator.recorded_contexts[1].previous == ["A"]
        assert spy_translator.recorded_contexts[1].following == ["C"]

        # Entry C: previous=["B"], no following
        assert spy_translator.recorded_contexts[2].previous == ["B"]
        assert spy_translator.recorded_contexts[2].following == []

    def test_passes_none_context_when_size_zero(self, spy_translator):
        """Verify that context is None when context_size=0."""
        entries = [_entry(1, "A"), _entry(2, "B")]
        list(translate_srt(spy_translator, entries, "en", "ja", context_size=0))

        assert all(ctx is None for ctx in spy_translator.recorded_contexts)

    def test_glossary_entries_passed_to_translator(self, spy_translator):
        """Verify matched glossary entries are included in context."""
        glossary = [("API", "應用程式介面"), ("dog", "狗")]
        entries = [_entry(1, "The API is ready"), _entry(2, "Hello world")]
        list(translate_srt(spy_translator, entries, "en", "zh-TW", context_size=0, glossary=glossary))

        # First entry contains "API" → glossary_prompt should include it
        assert spy_translator.recorded_contexts[0] is not None
        assert "API -> 應用程式介面" in spy_translator.recorded_contexts[0].glossary_prompt
        assert "dog" not in spy_translator.recorded_contexts[0].glossary_prompt

        # Second entry has no glossary match → context is None (context_size=0, no glossary match)
        assert spy_translator.recorded_contexts[1] is None

    def test_glossary_none_works(self):
        """Verify glossary=None doesn't break anything."""
        entries = [_entry(1, "Hello")]
        chunks = list(translate_srt(FakeTranslator(), entries, "en", "ja", context_size=0, glossary=None))
        assert len(chunks) > 0

    def test_translates_only_target_text(self):
        """Verify each subtitle is translated individually, not as a concatenated block."""

        class EchoTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang, context=None):
                yield f"[translated] {text}"

        entries = [_entry(1, "Hello"), _entry(2, "World"), _entry(3, "Foo")]
        chunks = list(translate_srt(EchoTranslator(), entries, "en", "ja", context_size=2))
        last = chunks[-1]

        # Each entry should only contain its own translation, not context lines
        assert last.entries[0].text == "[translated] Hello"
        assert last.entries[1].text == "[translated] World"
        assert last.entries[2].text == "[translated] Foo"
