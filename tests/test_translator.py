from unittest.mock import MagicMock, patch

from translate_gemma_ui.translator import (
    SUPPORTED_LANGUAGES,
    FakeTranslator,
    TranslationContext,
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


class TestTranslationContext:
    def test_is_frozen_dataclass(self):
        ctx = TranslationContext(previous=["A"], following=["B"])
        assert ctx.previous == ["A"]
        assert ctx.following == ["B"]

    def test_empty_lists(self):
        ctx = TranslationContext(previous=[], following=[])
        assert ctx.previous == []
        assert ctx.following == []

    def test_glossary_prompt_defaults_to_empty(self):
        ctx = TranslationContext(previous=[], following=[])
        assert ctx.glossary_prompt == ""

    def test_glossary_prompt_with_value(self):
        ctx = TranslationContext(previous=[], following=[], glossary_prompt="[Glossary]\nAPI -> 應用程式介面\n")
        assert "API -> 應用程式介面" in ctx.glossary_prompt


class TestFakeTranslatorWithContext:
    def test_ignores_context(self):
        translator = FakeTranslator()
        ctx = TranslationContext(previous=["prev"], following=["next"])
        results = list(translator.translate("hello", "en", "ja", context=ctx))
        # Should produce same output regardless of context
        results_without = list(translator.translate("hello", "en", "ja"))
        assert results[-1] == results_without[-1]


class TestBuildContextPrompt:
    """Tests for TranslateGemmaTranslator._build_context_prompt output format."""

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_contains_control_tokens(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=["Hello"], following=["World"])
        prompt = translator._build_context_prompt("Test text", "en", "zh-TW", ctx)

        assert prompt.startswith("<start_of_turn>user\n")
        assert prompt.endswith("<start_of_turn>model\n")
        assert "<end_of_turn>" in prompt

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_uses_traditional_chinese_for_zh_tw(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=[], following=[])
        prompt = translator._build_context_prompt("Hello", "en", "zh-TW", ctx)

        assert "Chinese (Traditional)" in prompt
        assert "Chinese (Traditional) (zh-TW)" in prompt

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_contains_context_section(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=["prev line"], following=["next line"])
        prompt = translator._build_context_prompt("Target", "en", "zh-TW", ctx)

        assert "[Context]" in prompt
        assert "Previous: prev line" in prompt
        assert "Next: next line" in prompt

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_places_target_text_after_translate_instruction(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=["prev"], following=[])
        prompt = translator._build_context_prompt("My target text", "en", "zh-TW", ctx)

        # Target text should appear after the translate instruction, before end_of_turn
        instruction_idx = prompt.index("Please translate the following")
        target_idx = prompt.index("My target text")
        end_idx = prompt.index("<end_of_turn>")
        assert instruction_idx < target_idx < end_idx

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_omits_context_section_when_both_empty(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=[], following=[])
        prompt = translator._build_context_prompt("Text", "en", "zh-TW", ctx)

        assert "[Context]" not in prompt

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_has_no_extra_blank_lines_between_sections(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=["prev"], following=["next"])
        prompt = translator._build_context_prompt("Text", "en", "zh-TW", ctx)

        # No triple+ newlines except the intentional ones before target text
        lines = prompt.split("\n")
        consecutive_empty = 0
        max_consecutive = 0
        for line in lines:
            if line.strip() == "":
                consecutive_empty += 1
                max_consecutive = max(max_consecutive, consecutive_empty)
            else:
                consecutive_empty = 0
        # The 2 blank lines before target text are the only allowed consecutive empties
        assert max_consecutive <= 2


class TestBuildContextPromptWithGlossary:
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_includes_glossary_section(self, mock_processor_cls, mock_model_cls):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=[], following=[], glossary_prompt="[Glossary]\nAPI -> 應用程式介面\n")
        prompt = translator._build_context_prompt("Use the API", "en", "zh-TW", ctx)

        assert "[Glossary]" in prompt
        assert "API -> 應用程式介面" in prompt

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_prompt_omits_glossary_when_empty(self, mock_processor_cls, mock_model_cls):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        ctx = TranslationContext(previous=[], following=[])
        prompt = translator._build_context_prompt("Hello", "en", "zh-TW", ctx)

        assert "[Glossary]" not in prompt


class TestTranslateGemmaTranslatorInit:
    """Tests for TranslateGemmaTranslator initialization behavior.

    Patches target `transformers.X` because translator.py uses lazy imports
    (from transformers import X) inside __init__, which resolve at the source module.
    """

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_uses_supported_languages(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert translator.languages is SUPPORTED_LANGUAGES

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_sets_max_tokens_from_config(self, mock_processor_cls, mock_model_cls):
        mock_processor = MagicMock()
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
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert translator.is_ready is True
