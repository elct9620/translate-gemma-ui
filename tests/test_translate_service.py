from translate_gemma_ui.translate_service import TranslationChunk, translate_text
from translate_gemma_ui.translator import FakeTranslator


class TestTranslateText:
    def test_short_text_no_progress(self):
        translator = FakeTranslator()
        chunks = list(translate_text(translator, "hello", "en", "ja"))
        assert len(chunks) > 0
        assert all(isinstance(c, TranslationChunk) for c in chunks)
        assert chunks[-1].progress == ""
        assert "hello" in chunks[-1].text

    def test_long_text_shows_progress(self):
        class SmallTokenTranslator(FakeTranslator):
            @property
            def max_tokens(self) -> int:
                return 5

        translator = SmallTokenTranslator()
        chunks = list(translate_text(translator, "First sentence. Second sentence. Third sentence.", "en", "ja"))
        assert len(chunks) > 0
        assert "翻譯完成" in chunks[-1].progress

    def test_segment_failure_preserves_original(self):
        class FailingTranslator(FakeTranslator):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            @property
            def max_tokens(self) -> int:
                return 5

            def translate(self, text, source_lang, target_lang):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("Simulated failure")
                yield from super().translate(text, source_lang, target_lang)

        translator = FailingTranslator()
        chunks = list(translate_text(translator, "First sentence. Second sentence. Third sentence.", "en", "ja"))
        assert len(chunks) > 0
        failed_chunks = [c for c in chunks if "翻譯失敗" in c.progress]
        assert len(failed_chunks) > 0
