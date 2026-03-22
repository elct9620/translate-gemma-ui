from collections.abc import Iterator

import gradio as gr

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.text_splitter import create_windows, merge_translations, split_sentences
from translate_gemma_ui.translator import Translator


def _build_device_display(device_info: DeviceInfo) -> str:
    parts = [f"**裝置：** {device_info.device_name} ({device_info.memory_info})"]
    if device_info.is_cpu:
        parts.append("⚠️ 使用 CPU 運算，翻譯速度可能較慢")
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def _make_translate_fn(
    translator: Translator,
) -> callable:
    def translate(text: str, source_lang: str, target_lang: str) -> Iterator[tuple[str, str]]:
        if not translator.is_ready:
            raise gr.Error("模型載入中，請稍候")
        if not text or not text.strip():
            raise gr.Error("請輸入要翻譯的文字")
        if source_lang == target_lang:
            raise gr.Error("來源語言與目標語言不得相同")

        sentences = split_sentences(text)
        windows = create_windows(sentences, translator.max_tokens, _estimate_tokens)

        if len(windows) <= 1:
            for chunk in translator.translate(text, source_lang, target_lang):
                yield chunk, ""
            return

        translations: list[str] = []
        for i, window in enumerate(windows):
            progress = f"翻譯中... ({i + 1}/{len(windows)})"
            last_chunk = ""
            try:
                for chunk in translator.translate(window.text, source_lang, target_lang):
                    last_chunk = chunk
                    partial = translations + [last_chunk]
                    partial_result = merge_translations(windows[: len(partial)], partial, sentences)
                    yield partial_result, progress
            except Exception:
                last_chunk = window.text
                yield (
                    merge_translations(windows[: len(translations) + 1], translations + [last_chunk], sentences),
                    (f"段落 {i + 1} 翻譯失敗，保留原文"),
                )
            translations.append(last_chunk)

        final = merge_translations(windows, translations, sentences)
        yield final, f"翻譯完成 ({len(windows)} 段)"

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
