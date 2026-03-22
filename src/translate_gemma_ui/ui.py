import logging
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import gradio as gr

from translate_gemma_ui.device import DeviceInfo
from translate_gemma_ui.glossary import parse_glossary
from translate_gemma_ui.srt_parser import SrtEntry, parse_srt, serialize_srt
from translate_gemma_ui.srt_service import translate_srt, translate_srt_full_file
from translate_gemma_ui.translate_service import translate_text
from translate_gemma_ui.translator import OutOfMemoryError, Translator

logger = logging.getLogger(__name__)

GLOSSARY_MODE_CHOICES = [("翻譯前替換", "pre"), ("翻譯後替換", "post")]


def _parse_glossary_file(file_path: str | None) -> list[tuple[str, str]] | None:
    """Parse glossary CSV file, return None if no file provided."""
    if not file_path:
        return None
    content = Path(file_path).read_text(encoding="utf-8")
    try:
        return parse_glossary(content)
    except ValueError as e:
        raise gr.Error(f"詞彙表格式錯誤：{e}。正確格式為 CSV（每行：來源詞,目標詞）")


def _build_device_display(device_info: DeviceInfo) -> str:
    parts = [f"**裝置：** {device_info.device_name} ({device_info.memory_info})"]
    if device_info.is_cpu:
        parts.append("⚠️ 使用 CPU 運算，翻譯速度可能較慢")
    return "\n\n".join(parts)


def _build_model_status(translator: Translator, error: str | None = None) -> str:
    if error:
        return f"⚠️ 模型載入失敗：{error}\n\n請輸入 HF Token 後點擊「載入模型」重試。"
    if translator.model_name == "FakeTranslator":
        return "⚠️ 目前使用開發模式（FakeTranslator），翻譯結果僅為模擬。請載入模型以使用真正的翻譯功能。"
    if translator.is_quantized:
        return "✅ 模型已載入（4-bit 量化模式）"
    return "✅ 模型已載入"


def _make_translate_fn(
    translator_ref: list[Translator],
) -> Callable[..., Iterator[tuple[str, str]]]:
    def translate(
        text: str,
        source_lang: str,
        target_lang: str,
        glossary_path: str | None = None,
        glossary_mode: str = "pre",
    ) -> Iterator[tuple[str, str]]:
        translator = translator_ref[0]
        if not translator.is_ready:
            raise gr.Error("模型載入中，請稍候")
        if not text or not text.strip():
            raise gr.Error("請輸入要翻譯的文字")
        if source_lang == target_lang:
            raise gr.Error("來源語言與目標語言不得相同")

        glossary = _parse_glossary_file(glossary_path)
        try:
            for chunk in translate_text(
                translator, text, source_lang, target_lang, glossary=glossary, glossary_mode=glossary_mode
            ):
                yield chunk.text, chunk.progress
        except OutOfMemoryError:
            raise gr.Error("記憶體不足，建議關閉其他應用程式或改用 CPU 模式")

    return translate


def _make_srt_translate_fn(
    translator_ref: list[Translator],
) -> Callable[..., Iterator[tuple[str, str, str | None]]]:
    def translate(
        file_path: str | None,
        source_lang: str,
        target_lang: str,
        mode: str = "batch",
        batch_size: int = 1,
        glossary_path: str | None = None,
        glossary_mode: str = "pre",
    ) -> Iterator[tuple[str, str, str | None]]:
        translator = translator_ref[0]
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

        glossary = _parse_glossary_file(glossary_path)

        try:
            if mode == "full":
                try:
                    chunks = translate_srt_full_file(
                        translator, entries, source_lang, target_lang, glossary=glossary, glossary_mode=glossary_mode
                    )
                except ValueError as e:
                    raise gr.Error(str(e))
                for chunk in chunks:
                    output_path = _write_srt_temp(chunk.entries, file_path)
                    yield chunk.progress, serialize_srt(chunk.entries), output_path
            else:
                for chunk in translate_srt(
                    translator,
                    entries,
                    source_lang,
                    target_lang,
                    batch_size=int(batch_size),
                    glossary=glossary,
                    glossary_mode=glossary_mode,
                ):
                    output_path = _write_srt_temp(chunk.entries, file_path)
                    yield chunk.progress, serialize_srt(chunk.entries), output_path
        except OutOfMemoryError:
            raise gr.Error("記憶體不足，建議關閉其他應用程式或改用 CPU 模式")

    return translate


def _make_load_model_fn(
    translator_ref: list[Translator],
    device_info: DeviceInfo,
) -> Callable[[str], str]:
    def load_model(token: str) -> str:
        from translate_gemma_ui.translator import TranslateGemmaTranslator

        token_value = token.strip() if token and token.strip() else None
        try:
            translator_ref[0] = TranslateGemmaTranslator(token=token_value, vram_bytes=device_info.vram_bytes)
            quantized_note = "（4-bit 量化模式）" if translator_ref[0].is_quantized else ""
            return f"✅ 模型載入成功{quantized_note}"
        except Exception as e:
            logger.exception("Failed to load model with provided token")
            return f"⚠️ 模型載入失敗：{e}"

    return load_model


def _write_srt_temp(entries: list[SrtEntry], original_path: str) -> str:
    stem = Path(original_path).stem
    output_name = f"{stem}_translated.srt"
    output_path = Path(tempfile.gettempdir()) / output_name
    output_path.write_text(serialize_srt(entries), encoding="utf-8")
    return str(output_path)


def create_app(translator: Translator, device_info: DeviceInfo, *, model_error: str | None = None) -> gr.Blocks:
    translator_ref: list[Translator] = [translator]
    lang_choices = [(name, code) for code, name in translator.languages.items()]

    with gr.Blocks(title="TranslateGemma UI") as app:
        gr.Markdown("# TranslateGemma UI")
        gr.Markdown(_build_device_display(device_info))

        with gr.Accordion("模型設定", open=model_error is not None):
            model_status = gr.Markdown(_build_model_status(translator, model_error))
            hf_token_input = gr.Textbox(
                label="HF Token",
                type="password",
                placeholder="hf_...",
                info="Hugging Face 存取權杖，可從 https://huggingface.co/settings/tokens 取得",
            )
            load_model_btn = gr.Button("載入模型", variant="secondary")

            load_model_btn.click(
                fn=_make_load_model_fn(translator_ref, device_info),
                inputs=[hf_token_input],
                outputs=[model_status],
            )

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

        srt_mode_choices = [("整檔模式", "full"), ("批次模式", "batch")]

        with gr.Tabs():
            with gr.TabItem("文字翻譯"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(label="輸入文字", lines=8, placeholder="請輸入要翻譯的文字...")
                        with gr.Row():
                            text_glossary = gr.File(file_types=[".csv"], type="filepath", label="詞彙表（選填）")
                            text_glossary_mode = gr.Radio(
                                choices=GLOSSARY_MODE_CHOICES, value="pre", label="詞彙替換方式"
                            )
                        translate_btn = gr.Button("翻譯", variant="primary")
                    with gr.Column():
                        output_text = gr.Textbox(label="翻譯結果", lines=8, interactive=False)
                        progress_text = gr.Markdown("")

                translate_btn.click(
                    fn=_make_translate_fn(translator_ref),
                    inputs=[input_text, source_lang, target_lang, text_glossary, text_glossary_mode],
                    outputs=[output_text, progress_text],
                    show_progress="full",
                    concurrency_limit=1,
                    concurrency_id="translate",
                )

            with gr.TabItem("SRT 字幕翻譯"):
                with gr.Row():
                    with gr.Column():
                        srt_file = gr.File(file_types=[".srt"], type="filepath", label="上傳 SRT 檔案")
                        srt_mode = gr.Radio(choices=srt_mode_choices, value="batch", label="翻譯模式")
                        srt_batch_size = gr.Number(value=1, minimum=1, label="每批字幕數 (N)", precision=0)
                        with gr.Row():
                            srt_glossary = gr.File(file_types=[".csv"], type="filepath", label="詞彙表（選填）")
                            srt_glossary_mode = gr.Radio(
                                choices=GLOSSARY_MODE_CHOICES, value="pre", label="詞彙替換方式"
                            )
                        srt_translate_btn = gr.Button("翻譯字幕", variant="primary")
                    with gr.Column():
                        srt_preview = gr.Textbox(label="翻譯預覽", lines=20, interactive=False)
                        srt_progress = gr.Markdown("")
                        srt_output_file = gr.File(label="下載翻譯結果", interactive=False)

                srt_translate_btn.click(
                    fn=_make_srt_translate_fn(translator_ref),
                    inputs=[
                        srt_file,
                        source_lang,
                        target_lang,
                        srt_mode,
                        srt_batch_size,
                        srt_glossary,
                        srt_glossary_mode,
                    ],
                    outputs=[srt_progress, srt_preview, srt_output_file],
                    show_progress="full",
                    concurrency_limit=1,
                    concurrency_id="translate",
                )

    return app
