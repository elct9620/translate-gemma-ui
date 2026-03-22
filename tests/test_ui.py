import gradio as gr
import pytest

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.translator import FakeTranslator
from translate_gemma_ui.ui import _build_device_display, _make_translate_fn, create_app


def _cpu_device():
    return DeviceInfo(device_name="CPU", memory_info="16.00 GB RAM", is_cpu=True)


def _gpu_device():
    return DeviceInfo(device_name="NVIDIA RTX 4090", memory_info="8.00 GB VRAM", is_cpu=False)


class TestCreateApp:
    def test_returns_blocks(self):
        app = create_app(FakeTranslator(), _cpu_device())
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


class TestTranslateFn:
    def test_empty_input_raises_error(self):
        fn = _make_translate_fn(FakeTranslator())
        with pytest.raises(gr.Error, match="輸入"):
            list(fn("", "en", "ja"))

    def test_whitespace_input_raises_error(self):
        fn = _make_translate_fn(FakeTranslator())
        with pytest.raises(gr.Error, match="輸入"):
            list(fn("   ", "en", "ja"))

    def test_same_language_raises_error(self):
        fn = _make_translate_fn(FakeTranslator())
        with pytest.raises(gr.Error, match="相同"):
            list(fn("hello", "en", "en"))

    def test_model_not_ready_raises_error(self):
        class NotReadyTranslator(FakeTranslator):
            @property
            def is_ready(self) -> bool:
                return False

        fn = _make_translate_fn(NotReadyTranslator())
        with pytest.raises(gr.Error, match="載入"):
            list(fn("hello", "en", "ja"))

    def test_valid_input_streams_result(self):
        fn = _make_translate_fn(FakeTranslator())
        results = list(fn("hello", "en", "ja"))
        assert len(results) > 0
        text, _progress = results[-1]
        assert "hello" in text

    def test_short_text_returns_empty_progress(self):
        fn = _make_translate_fn(FakeTranslator())
        results = list(fn("hello", "en", "ja"))
        _text, progress = results[-1]
        assert progress == ""

    def test_long_text_shows_segment_progress(self):
        class SmallTokenTranslator(FakeTranslator):
            @property
            def max_tokens(self) -> int:
                return 5

        fn = _make_translate_fn(SmallTokenTranslator())
        text = "First sentence. Second sentence. Third sentence."
        results = list(fn(text, "en", "ja"))
        assert len(results) > 0
        _text, progress = results[-1]
        assert "翻譯完成" in progress
