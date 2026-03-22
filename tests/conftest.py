import pytest

from translate_gemma_ui.translator import FakeTranslator


class SpyTranslator(FakeTranslator):
    """FakeTranslator that records texts passed to each translate call."""

    def __init__(self):
        super().__init__()
        self.recorded_texts: list[str] = []

    def translate(self, text, source_lang, target_lang):
        self.recorded_texts.append(text)
        yield from super().translate(text, source_lang, target_lang)


@pytest.fixture
def spy_translator():
    return SpyTranslator()
