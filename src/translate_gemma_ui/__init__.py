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

        translator = TranslateGemmaTranslator(vram_bytes=device_info.vram_bytes, force_cpu=device_info.is_cpu)
    except OSError as e:
        if "getaddrinfo" in str(e).lower() or "name or service not known" in str(e).lower():
            logger.exception("Network error during model loading")
            model_error = "無法連線至 HuggingFace，請確認網路連線或先下載模型至本地快取。"
        else:
            logger.exception("Failed to load model, falling back to FakeTranslator")
            model_error = str(e)
        translator = FakeTranslator()
    except Exception as e:
        logger.exception("Failed to load model, falling back to FakeTranslator")
        model_error = str(e)
        translator = FakeTranslator()

    return create_app(translator, device_info, model_error=model_error)


app = create_default_app()
