import gradio as gr

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.translator import FakeTranslator
from translate_gemma_ui.ui import _make_translate_fn, create_app


def _cpu_device():
    return DeviceInfo(device_name="CPU", memory_info="16.00 GB RAM", is_cpu=True)


def _gpu_device():
    return DeviceInfo(device_name="NVIDIA RTX 4090", memory_info="8.00 GB VRAM", is_cpu=False)


class TestCreateApp:
    def test_returns_blocks(self):
        app = create_app(FakeTranslator(), _cpu_device())
        assert isinstance(app, gr.Blocks)

    def test_cpu_device_shows_warning(self):
        app = create_app(FakeTranslator(), _cpu_device())
        assert app is not None

    def test_gpu_device_no_warning(self):
        app = create_app(FakeTranslator(), _gpu_device())
        assert app is not None


class TestTranslateFn:
    def test_empty_input_raises_error(self):
        translator = FakeTranslator()
        fn = _make_translate_fn(translator)
        try:
            list(fn("", "en", "ja"))
            assert False, "Should have raised gr.Error"
        except gr.Error as e:
            assert "輸入" in str(e)

    def test_whitespace_input_raises_error(self):
        translator = FakeTranslator()
        fn = _make_translate_fn(translator)
        try:
            list(fn("   ", "en", "ja"))
            assert False, "Should have raised gr.Error"
        except gr.Error as e:
            assert "輸入" in str(e)

    def test_same_language_raises_error(self):
        translator = FakeTranslator()
        fn = _make_translate_fn(translator)
        try:
            list(fn("hello", "en", "en"))
            assert False, "Should have raised gr.Error"
        except gr.Error as e:
            assert "相同" in str(e)

    def test_model_not_ready_raises_error(self):
        class NotReadyTranslator(FakeTranslator):
            @property
            def is_ready(self) -> bool:
                return False

        fn = _make_translate_fn(NotReadyTranslator())
        try:
            list(fn("hello", "en", "ja"))
            assert False, "Should have raised gr.Error"
        except gr.Error as e:
            assert "載入" in str(e)

    def test_valid_input_streams_result(self):
        translator = FakeTranslator()
        fn = _make_translate_fn(translator)
        results = list(fn("hello", "en", "ja"))
        assert len(results) > 0
        text, _progress = results[-1]
        assert "hello" in text

    def test_short_text_returns_empty_progress(self):
        translator = FakeTranslator()
        fn = _make_translate_fn(translator)
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
