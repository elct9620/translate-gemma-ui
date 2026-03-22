import pytest

from translate_gemma_ui.translator import FakeTranslator


class SpyTranslator(FakeTranslator):
    """FakeTranslator that records context passed to each translate call."""

    def __init__(self):
        super().__init__()
        self.recorded_contexts: list = []

    def translate(self, text, source_lang, target_lang, context=None):
        self.recorded_contexts.append(context)
        yield from super().translate(text, source_lang, target_lang, context=context)


@pytest.fixture
def spy_translator():
    return SpyTranslator()
