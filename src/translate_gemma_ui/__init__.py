"""TranslateGemma UI - Local translation tool powered by TranslateGemma 4B."""

import logging

import gradio as gr

logger = logging.getLogger(__name__)


def create_default_app() -> gr.Blocks:
    from translate_gemma_ui.device import get_device_info
    from translate_gemma_ui.translator import FakeTranslator
    from translate_gemma_ui.ui import create_app

    device_info = get_device_info()
    model_error: str | None = None

    try:
        from translate_gemma_ui.translator import TranslateGemmaTranslator

        translator = TranslateGemmaTranslator()
    except Exception as e:
        logger.exception("Failed to load model, falling back to FakeTranslator")
        model_error = str(e)
        translator = FakeTranslator()

    return create_app(translator, device_info, model_error=model_error)


app = create_default_app()
