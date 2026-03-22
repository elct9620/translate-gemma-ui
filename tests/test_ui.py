from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.translator import FakeTranslator
from translate_gemma_ui.ui import (
    _build_device_display,
    _build_model_status,
    _make_load_model_fn,
    _make_srt_translate_fn,
    _make_translate_fn,
    _parse_glossary_file,
    create_app,
)


def _cpu_device():
    return DeviceInfo(device_name="CPU", memory_info="16.00 GB RAM", is_cpu=True)


def _gpu_device():
    return DeviceInfo(device_name="NVIDIA RTX 4090", memory_info="8.00 GB VRAM", is_cpu=False)


class TestCreateApp:
    def test_returns_blocks(self):
        app = create_app(FakeTranslator(), _cpu_device())
        assert isinstance(app, gr.Blocks)

    def test_returns_blocks_with_model_error(self):
        app = create_app(FakeTranslator(), _cpu_device(), model_error="some error")
        assert isinstance(app, gr.Blocks)


class TestDeviceDisplay:
    def test_cpu_shows_warning(self):
        display = _build_device_display(_cpu_device())
        assert "CPU" in display
        assert "翻譯速度可能較慢" in display

    def test_gpu_no_warning(self):
        display = _build_device_display(_gpu_device())
        assert "NVIDIA RTX 4090" in display
        assert "翻譯速度可能較慢" not in display


class TestModelStatus:
    def test_shows_error_when_load_failed(self):
        status = _build_model_status(FakeTranslator(), error="connection error")
        assert "載入失敗" in status
        assert "connection error" in status

    def test_shows_fake_translator_warning(self):
        status = _build_model_status(FakeTranslator())
        assert "開發模式" in status

    def test_shows_success_for_real_translator(self):
        class StubTranslator:
            @property
            def languages(self):
                return {"en": "English"}

            @property
            def max_tokens(self):
                return 1024

            @property
            def is_ready(self):
                return True

            @property
            def is_quantized(self):
                return False

            @property
            def model_name(self):
                return "google/translategemma-4b-it"

            def translate(self, text, source_lang, target_lang):
                yield text

        status = _build_model_status(StubTranslator())
        assert "已載入" in status
        assert "量化" not in status

    def test_shows_quantized_status_for_quantized_model(self):
        class QuantizedStub:
            @property
            def languages(self):
                return {"en": "English"}

            @property
            def max_tokens(self):
                return 1024

            @property
            def is_ready(self):
                return True

            @property
            def is_quantized(self):
                return True

            @property
            def model_name(self):
                return "google/translategemma-4b-it"

            def translate(self, text, source_lang, target_lang):
                yield text

        status = _build_model_status(QuantizedStub())
        assert "已載入" in status
        assert "量化" in status


class TestLoadModelFn:
    @patch("translate_gemma_ui.translator.TranslateGemmaTranslator", side_effect=RuntimeError("model not found"))
    def test_load_failure_returns_error_message(self, _mock_cls):
        translator_ref = [FakeTranslator()]
        fn = _make_load_model_fn(translator_ref, _gpu_device())
        result = fn("")
        assert "載入失敗" in result
        assert isinstance(translator_ref[0], FakeTranslator)

    @patch("translate_gemma_ui.translator.TranslateGemmaTranslator", side_effect=RuntimeError("model not found"))
    def test_load_with_empty_token_still_attempts(self, _mock_cls):
        translator_ref = [FakeTranslator()]
        fn = _make_load_model_fn(translator_ref, _gpu_device())
        result = fn("   ")
        assert "載入失敗" in result

    @patch("translate_gemma_ui.translator.TranslateGemmaTranslator")
    def test_load_success_updates_translator_ref(self, mock_cls):
        mock_translator = MagicMock()
        mock_cls.return_value = mock_translator

        translator_ref = [FakeTranslator()]
        fn = _make_load_model_fn(translator_ref, _gpu_device())
        result = fn("test-token")

        assert "載入成功" in result
        assert translator_ref[0] is mock_translator

    @patch("translate_gemma_ui.translator.TranslateGemmaTranslator")
    def test_load_passes_vram_bytes_to_translator(self, mock_cls):
        mock_cls.return_value = MagicMock()
        device = DeviceInfo(device_name="GTX 3050", memory_info="4.00 GB VRAM", is_cpu=False, vram_bytes=4 * 1024**3)

        translator_ref = [FakeTranslator()]
        fn = _make_load_model_fn(translator_ref, device)
        fn("test-token")

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["vram_bytes"] == 4 * 1024**3


class TestTranslateFn:
    def test_empty_input_raises_error(self):
        fn = _make_translate_fn([FakeTranslator()])
        with pytest.raises(gr.Error, match="輸入"):
            list(fn("", "en", "ja"))

    def test_whitespace_input_raises_error(self):
        fn = _make_translate_fn([FakeTranslator()])
        with pytest.raises(gr.Error, match="輸入"):
            list(fn("   ", "en", "ja"))

    def test_same_language_raises_error(self):
        fn = _make_translate_fn([FakeTranslator()])
        with pytest.raises(gr.Error, match="相同"):
            list(fn("hello", "en", "en"))

    def test_model_not_ready_raises_error(self):
        class NotReadyTranslator(FakeTranslator):
            @property
            def is_ready(self) -> bool:
                return False

        fn = _make_translate_fn([NotReadyTranslator()])
        with pytest.raises(gr.Error, match="載入"):
            list(fn("hello", "en", "ja"))

    def test_valid_input_streams_result(self):
        fn = _make_translate_fn([FakeTranslator()])
        results = list(fn("hello", "en", "ja"))
        assert len(results) > 0
        text, _progress = results[-1]
        assert "hello" in text

    def test_short_text_returns_empty_progress(self):
        fn = _make_translate_fn([FakeTranslator()])
        results = list(fn("hello", "en", "ja"))
        _text, progress = results[-1]
        assert progress == ""

    def test_long_text_shows_segment_progress(self):
        class SmallTokenTranslator(FakeTranslator):
            @property
            def max_tokens(self) -> int:
                return 5

        fn = _make_translate_fn([SmallTokenTranslator()])
        text = "First sentence. Second sentence. Third sentence."
        results = list(fn(text, "en", "ja"))
        assert len(results) > 0
        _text, progress = results[-1]
        assert "翻譯完成" in progress

    def test_uses_latest_translator_ref(self):
        fake1 = FakeTranslator()
        fake2 = FakeTranslator()
        translator_ref = [fake1]
        fn = _make_translate_fn(translator_ref)
        translator_ref[0] = fake2
        results = list(fn("hello", "en", "ja"))
        assert len(results) > 0


class TestParseGlossaryFile:
    def _write_csv(self, tmp_path, content):
        csv_file = tmp_path / "glossary.csv"
        csv_file.write_text(content, encoding="utf-8")
        return str(csv_file)

    def test_valid_csv_returns_glossary(self, tmp_path):
        path = self._write_csv(tmp_path, "API,應用程式介面\ncloud,雲端")
        result = _parse_glossary_file(path)
        assert result == [("API", "應用程式介面"), ("cloud", "雲端")]

    def test_none_path_returns_none(self):
        assert _parse_glossary_file(None) is None

    def test_invalid_csv_raises_gr_error(self, tmp_path):
        path = self._write_csv(tmp_path, "API")
        with pytest.raises(gr.Error, match="格式"):
            _parse_glossary_file(path)


class TestSrtTranslateFn:
    def _write_srt(self, tmp_path, content):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(content, encoding="utf-8")
        return str(srt_file)

    def test_no_file_raises_error(self):
        fn = _make_srt_translate_fn([FakeTranslator()])
        with pytest.raises(gr.Error, match="上傳"):
            list(fn(None, "en", "ja", "batch", 1))

    def test_invalid_srt_raises_error(self, tmp_path):
        path = self._write_srt(tmp_path, "not a valid srt")
        fn = _make_srt_translate_fn([FakeTranslator()])
        with pytest.raises(gr.Error, match="格式"):
            list(fn(path, "en", "ja", "batch", 1))

    def test_same_language_raises_error(self, tmp_path):
        path = self._write_srt(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        fn = _make_srt_translate_fn([FakeTranslator()])
        with pytest.raises(gr.Error, match="相同"):
            list(fn(path, "en", "en", "batch", 1))

    def test_model_not_ready_raises_error(self, tmp_path):
        class NotReadyTranslator(FakeTranslator):
            @property
            def is_ready(self) -> bool:
                return False

        path = self._write_srt(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        fn = _make_srt_translate_fn([NotReadyTranslator()])
        with pytest.raises(gr.Error, match="載入"):
            list(fn(path, "en", "ja", "batch", 1))

    def test_valid_srt_produces_output(self, tmp_path):
        path = self._write_srt(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        fn = _make_srt_translate_fn([FakeTranslator()])
        results = list(fn(path, "en", "ja", "batch", 1))
        assert len(results) > 0
        progress, preview, output_file = results[-1]
        assert output_file is not None
        assert output_file.endswith(".srt")
        assert "翻譯完成" in progress

    def test_srt_preview_contains_timestamps(self, tmp_path):
        path = self._write_srt(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        fn = _make_srt_translate_fn([FakeTranslator()])
        results = list(fn(path, "en", "ja", "batch", 1))
        _progress, preview, _file = results[-1]
        assert "00:00:01,000 --> 00:00:02,000" in preview

    def test_full_file_mode(self, tmp_path):
        class SrtEchoTranslator(FakeTranslator):
            def translate(self, text, source_lang, target_lang):
                yield "1\n00:00:01,000 --> 00:00:02,000\nTranslated\n"

        path = self._write_srt(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        fn = _make_srt_translate_fn([SrtEchoTranslator()])
        results = list(fn(path, "en", "ja", "full", 1))
        assert len(results) > 0
        _progress, preview, _file = results[-1]
        assert "Translated" in preview
