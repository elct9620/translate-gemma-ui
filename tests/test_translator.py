from unittest.mock import MagicMock, patch

import pytest

from translate_gemma_ui.translator import (
    SUPPORTED_LANGUAGES,
    FakeTranslator,
    ModelLoadError,
    OutOfMemoryError,
    Translator,
    _classify_load_error,
    _is_model_cached,
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


@patch("translate_gemma_ui.translator._is_model_cached", return_value=False)
class TestTranslateGemmaTranslatorInit:
    """Tests for TranslateGemmaTranslator initialization behavior."""

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_uses_supported_languages(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        assert translator.languages is SUPPORTED_LANGUAGES

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_sets_max_tokens_from_config(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_model_cls.from_pretrained.return_value = mock_model

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        assert translator.max_tokens == 4096

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_init_reports_ready(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token")
        assert translator.is_ready is True


@patch("translate_gemma_ui.translator._is_model_cached", return_value=False)
class TestTranslateGemmaQuantization:
    """Tests for automatic 4-bit quantization when VRAM is insufficient."""

    @patch("transformers.BitsAndBytesConfig")
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_reports_quantized_when_vram_below_threshold(
        self, mock_processor_cls, mock_model_cls, mock_bnb_config, _mock_cached
    ):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_bnb_config.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=4 * 1024**3)

        assert translator.is_quantized is True

    @patch("transformers.BitsAndBytesConfig")
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_passes_quantization_config_when_vram_below_threshold(
        self, mock_processor_cls, mock_model_cls, mock_bnb_config, _mock_cached
    ):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_bnb_config.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=4 * 1024**3)

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs
        assert "dtype" not in call_kwargs

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_no_quantization_when_vram_sufficient(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=16 * 1024**3)

        assert translator.is_quantized is False
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" not in call_kwargs
        assert "dtype" in call_kwargs

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_no_quantization_when_vram_equals_threshold(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=8 * 1024**3)

        assert translator.is_quantized is False

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_no_quantization_when_vram_bytes_is_none(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=None)

        assert translator.is_quantized is False

    @patch("transformers.BitsAndBytesConfig", side_effect=ImportError("No module named 'bitsandbytes'"))
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_fallback_when_bitsandbytes_missing(self, mock_processor_cls, mock_model_cls, _mock_bnb, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=4 * 1024**3)

        assert translator.is_quantized is False
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "dtype" in call_kwargs

    @patch("transformers.BitsAndBytesConfig")
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_falls_back_to_cpu_when_quantized_load_fails(
        self, mock_processor_cls, mock_model_cls, mock_bnb_config, _mock_cached
    ):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_bnb_config.return_value = MagicMock()

        # First call (GPU quantized) raises dispatch error, second call (CPU fallback) succeeds
        mock_model_cls.from_pretrained.side_effect = [
            RuntimeError(
                "Some modules are dispatched on the CPU or the disk. "
                "Make sure you have enough GPU RAM to fit the quantized model."
            ),
            MagicMock(),
        ]

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=4 * 1024**3)

        assert translator.is_quantized is False
        assert translator.is_ready is True

        # Verify the retry used CPU mode
        retry_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert retry_kwargs["device_map"] == "cpu"
        assert "quantization_config" not in retry_kwargs

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_falls_back_to_cpu_when_non_quantized_load_fails_with_oom(
        self, mock_processor_cls, mock_model_cls, _mock_cached
    ):
        import torch

        mock_processor_cls.from_pretrained.return_value = MagicMock()

        # VRAM >= 8GB so no quantization is used, but GPU still OOMs
        mock_model_cls.from_pretrained.side_effect = [
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            MagicMock(),
        ]

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(model_id="test-model", token="fake-token", vram_bytes=16 * 1024**3)

        assert translator.is_quantized is False
        assert translator.is_ready is True

        # Verify the retry used CPU mode with float32
        retry_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert retry_kwargs["device_map"] == "cpu"
        assert retry_kwargs["dtype"] == torch.float32
        assert "quantization_config" not in retry_kwargs

    def test_fake_translator_is_not_quantized(self, _mock_cached):
        translator = FakeTranslator()
        assert translator.is_quantized is False


@patch("translate_gemma_ui.translator._is_model_cached", return_value=False)
class TestTranslateGemmaForceCpu:
    """Tests for force_cpu parameter skipping quantization and using CPU device_map."""

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_force_cpu_uses_cpu_device_map(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        TranslateGemmaTranslator(model_id="test-model", token="fake-token", force_cpu=True)

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs["device_map"] == "cpu"

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_force_cpu_skips_quantization_even_with_low_vram(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator(
            model_id="test-model", token="fake-token", vram_bytes=4 * 1024**3, force_cpu=True
        )

        assert translator.is_quantized is False
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" not in call_kwargs

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_force_cpu_uses_float32_dtype(self, mock_processor_cls, mock_model_cls, _mock_cached):
        import torch

        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        TranslateGemmaTranslator(model_id="test-model", token="fake-token", force_cpu=True)

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == torch.float32

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_default_uses_auto_device_map(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs["device_map"] == "auto"


class TestIsModelCached:
    @patch("huggingface_hub.try_to_load_from_cache", return_value="/path/to/config.json")
    def test_returns_true_when_cached(self, _mock):
        assert _is_model_cached("some/model") is True

    @patch("huggingface_hub.try_to_load_from_cache", return_value=None)
    def test_returns_false_when_not_cached(self, _mock):
        assert _is_model_cached("some/model") is False


class TestTranslateGemmaCacheDetection:
    """Tests that cached models load with local_files_only=True and no token."""

    @patch("translate_gemma_ui.translator._is_model_cached", return_value=True)
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_uses_local_files_only_when_cached(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        TranslateGemmaTranslator(model_id="test-model", token="should-be-ignored")

        processor_kwargs = mock_processor_cls.from_pretrained.call_args
        assert processor_kwargs[1]["local_files_only"] is True
        assert processor_kwargs[1]["token"] is None

        model_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert model_kwargs["local_files_only"] is True
        assert model_kwargs["token"] is None

    @patch("translate_gemma_ui.translator._is_model_cached", return_value=False)
    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_uses_token_when_not_cached(self, mock_processor_cls, mock_model_cls, _mock_cached):
        mock_processor_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        TranslateGemmaTranslator(model_id="test-model", token="my-token")

        processor_kwargs = mock_processor_cls.from_pretrained.call_args
        assert processor_kwargs[1]["token"] == "my-token"
        assert processor_kwargs[1]["local_files_only"] is False

        model_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert model_kwargs["token"] == "my-token"
        assert model_kwargs["local_files_only"] is False


class TestOutOfMemoryError:
    def test_is_runtime_error_subclass(self):
        assert issubclass(OutOfMemoryError, RuntimeError)

    def test_carries_friendly_message(self):
        err = OutOfMemoryError("記憶體不足")
        assert "記憶體不足" in str(err)


class TestModelLoadError:
    def test_is_runtime_error_subclass(self):
        assert issubclass(ModelLoadError, RuntimeError)

    def test_carries_error_type(self):
        err = ModelLoadError("fail", error_type="auth")
        assert err.error_type == "auth"
        assert "fail" in str(err)

    def test_original_accessible_via_cause(self):
        original = ValueError("original")
        try:
            raise ModelLoadError("fail", error_type="unknown") from original
        except ModelLoadError as err:
            assert err.__cause__ is original


class TestClassifyLoadError:
    def test_classifies_gated_repo_error_as_auth(self):
        hf_errors = pytest.importorskip("huggingface_hub.errors")
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {}
        exc = hf_errors.GatedRepoError("access denied", response=mock_response)
        result = _classify_load_error(exc)
        assert result.error_type == "auth"

    def test_classifies_repo_not_found_as_auth(self):
        hf_errors = pytest.importorskip("huggingface_hub.errors")
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        exc = hf_errors.RepositoryNotFoundError("not found", response=mock_response)
        result = _classify_load_error(exc)
        assert result.error_type == "auth"

    def test_classifies_connection_error_as_network(self):
        exc = ConnectionError("connection refused")
        result = _classify_load_error(exc)
        assert result.error_type == "network"

    def test_classifies_os_error_with_network_keyword_as_network(self):
        exc = OSError("network unreachable")
        result = _classify_load_error(exc)
        assert result.error_type == "network"

    def test_classifies_os_error_without_keyword_as_unknown(self):
        exc = OSError("disk full")
        result = _classify_load_error(exc)
        assert result.error_type == "unknown"

    def test_classifies_oom_runtime_error(self):
        exc = RuntimeError("CUDA out of memory")
        result = _classify_load_error(exc)
        assert result.error_type == "out_of_memory"

    def test_classifies_cpu_dispatch_error_as_out_of_memory(self):
        exc = RuntimeError(
            "Some modules are dispatched on the CPU or the disk. "
            "Make sure you have enough GPU RAM to fit the quantized model."
        )
        result = _classify_load_error(exc)
        assert result.error_type == "out_of_memory"

    def test_classifies_generic_error_as_unknown(self):
        exc = ValueError("something went wrong")
        result = _classify_load_error(exc)
        assert result.error_type == "unknown"


class TestTranslateGemmaRaisesModelLoadError:
    @patch("translate_gemma_ui.translator._is_model_cached", return_value=False)
    @patch("transformers.AutoProcessor")
    def test_init_wraps_error_as_model_load_error(self, mock_processor_cls, _mock_cached):
        mock_processor_cls.from_pretrained.side_effect = RuntimeError("something broke")

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        with pytest.raises(ModelLoadError) as exc_info:
            TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert exc_info.value.error_type == "unknown"

    @patch("translate_gemma_ui.translator._is_model_cached", return_value=False)
    @patch("transformers.AutoProcessor")
    def test_init_wraps_connection_error(self, mock_processor_cls, _mock_cached):
        mock_processor_cls.from_pretrained.side_effect = ConnectionError("connection refused")

        from translate_gemma_ui.translator import TranslateGemmaTranslator

        with pytest.raises(ModelLoadError) as exc_info:
            TranslateGemmaTranslator(model_id="test-model", token="fake-token")

        assert exc_info.value.error_type == "network"
