from unittest.mock import MagicMock, patch

from translate_gemma_ui.translator import (
    SUPPORTED_LANGUAGES,
    FakeTranslator,
    Translator,
)


class TestSupportedLanguages:
    def test_contains_exactly_three_languages(self):
        assert len(SUPPORTED_LANGUAGES) == 3

    def test_contains_english(self):
        assert SUPPORTED_LANGUAGES["en"] == "English"

    def test_contains_traditional_chinese(self):
        assert SUPPORTED_LANGUAGES["zh-TW"] == "Chinese (Traditional)"

    def test_contains_japanese(self):
        assert SUPPORTED_LANGUAGES["ja"] == "Japanese"

    def test_no_duplicate_display_names(self):
        names = list(SUPPORTED_LANGUAGES.values())
        assert len(names) == len(set(names))


class TestFakeTranslator:
    def test_fake_translator_uses_supported_languages(self):
        translator = FakeTranslator()
        assert translator.languages is SUPPORTED_LANGUAGES

    def test_fake_translator_is_ready(self):
        translator = FakeTranslator()
        assert translator.is_ready is True

    def test_fake_translator_max_tokens(self):
        translator = FakeTranslator()
        assert translator.max_tokens > 0

    def test_fake_translator_translate_streams(self):
        translator = FakeTranslator()
        results = list(translator.translate("hello", "en", "ja"))
        assert len(results) > 0
        assert "hello" in results[-1]

    def test_fake_translator_translate_accumulates(self):
        translator = FakeTranslator()
        results = list(translator.translate("hello world", "en", "zh-TW"))
        for i in range(1, len(results)):
            assert len(results[i]) > len(results[i - 1])

    def test_fake_translator_model_name(self):
        translator = FakeTranslator()
        assert translator.model_name == "FakeTranslator"

    def test_fake_translator_conforms_to_protocol(self):
        translator = FakeTranslator()
        assert isinstance(translator, Translator)


class TestTranslateGemmaTranslatorInit:
    """Tests for TranslateGemmaTranslator initialization behavior."""

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_uses_supported_languages(self, mock_processor_cls, mock_model_cls):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        assert translator.languages is SUPPORTED_LANGUAGES

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_sets_max_tokens_from_config(self, mock_processor_cls, mock_model_cls):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_model_cls.from_pretrained.return_value = mock_model

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        assert translator.max_tokens == 4096

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_reports_ready(self, mock_processor_cls, mock_model_cls):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        assert translator.is_ready is True
