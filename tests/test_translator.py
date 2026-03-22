from unittest.mock import MagicMock, patch

from translate_gemma_ui.translator import FakeTranslator, Translator, _extract_languages_from_template


class TestExtractLanguages:
    def test_extracts_language_pairs(self):
        template = '"en": "English", "ja": "Japanese", "zh-TW": "Chinese"'
        result = _extract_languages_from_template(template)
        assert result == {"en": "English", "ja": "Japanese", "zh-TW": "Chinese"}

    def test_empty_template(self):
        assert _extract_languages_from_template("") == {}

    def test_no_matching_format(self):
        assert _extract_languages_from_template("random text without language pairs") == {}

    def test_first_occurrence_wins(self):
        template = '"en": "English", "en": "British English"'
        result = _extract_languages_from_template(template)
        assert result["en"] == "English"


class TestFakeTranslator:
    def test_fake_translator_has_languages(self):
        translator = FakeTranslator()
        assert len(translator.languages) > 0
        assert "en" in translator.languages

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
    """Tests for TranslateGemmaTranslator initialization behavior.

    Patches target `transformers.X` because translator.py uses lazy imports
    (from transformers import X) inside __init__, which resolve at the source module.
    """

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_sets_languages_from_processor(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor.chat_template = '"en": "English", "ja": "Japanese"'
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert translator.languages == {"en": "English", "ja": "Japanese"}

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_sets_max_tokens_from_config(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor.chat_template = '"en": "English"'
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_model_cls.from_pretrained.return_value = mock_model

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert translator.max_tokens == 4096

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_reports_ready(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor.chat_template = '"en": "English"'
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert translator.is_ready is True
