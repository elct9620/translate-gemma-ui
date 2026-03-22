import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import gradio as gr

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.srt_parser import SrtEntry, parse_srt, serialize_srt
from translate_gemma_ui.srt_service import translate_srt
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


def _make_srt_translate_fn(
    translator: Translator,
) -> Callable[..., Iterator[tuple[str, str | None]]]:
    def translate(
        file_path: str | None, source_lang: str, target_lang: str, context_size: int
    ) -> Iterator[tuple[str, str | None]]:
        if not translator.is_ready:
            raise gr.Error("模型載入中，請稍候")
        if not file_path:
            raise gr.Error("請上傳 SRT 檔案")
        if source_lang == target_lang:
            raise gr.Error("來源語言與目標語言不得相同")

        content = Path(file_path).read_text(encoding="utf-8")
        try:
            entries = parse_srt(content)
        except ValueError as e:
            raise gr.Error(f"SRT 格式錯誤：{e}")

        output_path = None
        for chunk in translate_srt(translator, entries, source_lang, target_lang, context_size=int(context_size)):
            output_path = _write_srt_temp(chunk.entries, file_path)
            yield chunk.progress, output_path

    return translate


def _write_srt_temp(entries: list[SrtEntry], original_path: str) -> str:
    stem = Path(original_path).stem
    output_name = f"{stem}_translated.srt"
    output_path = Path(tempfile.gettempdir()) / output_name
    output_path.write_text(serialize_srt(entries), encoding="utf-8")
    return str(output_path)


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

        with gr.Tabs():
            with gr.TabItem("文字翻譯"):
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

            with gr.TabItem("SRT 字幕翻譯"):
                srt_file = gr.File(file_types=[".srt"], type="filepath", label="上傳 SRT 檔案")
                context_slider = gr.Slider(minimum=0, maximum=10, value=3, step=1, label="上下文窗口大小 (N)")
                srt_translate_btn = gr.Button("翻譯字幕", variant="primary")
                srt_progress = gr.Markdown("")
                srt_output_file = gr.File(label="下載翻譯結果", interactive=False)

                srt_translate_btn.click(
                    fn=_make_srt_translate_fn(translator),
                    inputs=[srt_file, source_lang, target_lang, context_slider],
                    outputs=[srt_progress, srt_output_file],
                    show_progress="full",
                )

    return app
