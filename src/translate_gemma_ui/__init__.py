"""TranslateGemma UI - Local translation tool powered by TranslateGemma 4B."""

import logging
import os

import gradio as gr

logger = logging.getLogger(__name__)


def create_default_app() -> gr.Blocks:
    from translate_gemma_ui.device import get_device_info
    from translate_gemma_ui.translator import FakeTranslator
    from translate_gemma_ui.ui import create_app

    device_info = get_device_info()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from translate_gemma_ui.translator import TranslateGemmaTranslator

            translator = TranslateGemmaTranslator()
        except Exception:
            logger.exception("Failed to load model, falling back to FakeTranslator")
            translator = FakeTranslator()
    else:
        logger.info("HF_TOKEN not set, using FakeTranslator for development")
        translator = FakeTranslator()

    return create_app(translator, device_info)


app = create_default_app()
