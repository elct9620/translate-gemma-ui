from collections.abc import Callable, Iterator

import gradio as gr

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.translate_service import translate_text
from translate_gemma_ui.translator import Translator


def _build_device_display(device_info: DeviceInfo) -> str:
    parts = [f"**裝置：** {device_info.device_name} ({device_info.memory_info})"]
    if device_info.is_cpu:
        parts.append("⚠️ 使用 CPU 運算，翻譯速度可能較慢")
    return "\n\n".join(parts)


def _make_translate_fn(
    translator: Translator,
) -> Callable[..., Iterator[tuple[str, str]]]:
    def translate(text: str, source_lang: str, target_lang: str) -> Iterator[tuple[str, str]]:
        if not translator.is_ready:
            raise gr.Error("模型載入中，請稍候")
        if not text or not text.strip():
            raise gr.Error("請輸入要翻譯的文字")
        if source_lang == target_lang:
            raise gr.Error("來源語言與目標語言不得相同")

        for chunk in translate_text(translator, text, source_lang, target_lang):
            yield chunk.text, chunk.progress

    return translate


def create_app(translator: Translator, device_info: DeviceInfo) -> gr.Blocks:
    lang_choices = [(name, code) for code, name in translator.languages.items()]

    with gr.Blocks(title="TranslateGemma UI") as app:
        gr.Markdown("# TranslateGemma UI")
        gr.Markdown(_build_device_display(device_info))

        with gr.Row():
            source_lang = gr.Dropdown(
                choices=lang_choices,
                value="en",
                label="來源語言",
                filterable=True,
            )
            target_lang = gr.Dropdown(
                choices=lang_choices,
                value="zh-TW",
                label="目標語言",
                filterable=True,
            )

        input_text = gr.Textbox(label="輸入文字", lines=8, placeholder="請輸入要翻譯的文字...")
        translate_btn = gr.Button("翻譯", variant="primary")
        output_text = gr.Textbox(label="翻譯結果", lines=8, interactive=False)
        progress_text = gr.Markdown("")

        translate_btn.click(
            fn=_make_translate_fn(translator),
            inputs=[input_text, source_lang, target_lang],
            outputs=[output_text, progress_text],
            show_progress="full",
        )

    return app
