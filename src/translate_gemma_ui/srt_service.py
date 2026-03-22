import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from translate_gemma_ui.glossary import apply_glossary_post, apply_glossary_pre
from translate_gemma_ui.srt_parser import SrtEntry, parse_srt, serialize_srt
from translate_gemma_ui.translator import Translator

logger = logging.getLogger(__name__)

GlossaryMode = Literal["pre", "post"]


@dataclass(frozen=True)
class SrtTranslationChunk:
    entries: list[SrtEntry]
    progress: str


BATCH_SEPARATOR = "\n\n"


def translate_srt(
    translator: Translator,
    entries: list[SrtEntry],
    source_lang: str,
    target_lang: str,
    batch_size: int = 1,
    glossary: list[tuple[str, str]] | None = None,
    glossary_mode: GlossaryMode = "pre",
) -> Iterator[SrtTranslationChunk]:
    results = list(entries)

    if batch_size <= 1:
        yield from _translate_srt_single(translator, results, source_lang, target_lang, glossary, glossary_mode)
    else:
        yield from _translate_srt_batch(
            translator, results, source_lang, target_lang, batch_size, glossary, glossary_mode
        )


def _translate_srt_single(
    translator: Translator,
    results: list[SrtEntry],
    source_lang: str,
    target_lang: str,
    glossary: list[tuple[str, str]] | None,
    glossary_mode: GlossaryMode,
) -> Iterator[SrtTranslationChunk]:
    total = len(results)

    for i, entry in enumerate(results):
        if not entry.text:
            yield SrtTranslationChunk(
                entries=list(results),
                progress=f"翻譯中... ({i + 1}/{total})",
            )
            continue

        text = entry.text
        if glossary_mode == "pre":
            text = apply_glossary_pre(text, glossary)

        try:
            last_chunk = ""
            for chunk in translator.translate(text, source_lang, target_lang):
                last_chunk = chunk

            translated = last_chunk.strip()
            if glossary_mode == "post":
                translated = apply_glossary_post(translated, glossary)

            results[i] = SrtEntry(
                index=entry.index,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=translated,
            )
        except Exception:
            logger.exception("字幕 %d/%d 翻譯失敗", i + 1, total)

        yield SrtTranslationChunk(
            entries=list(results),
            progress=f"翻譯中... ({i + 1}/{total})",
        )

    yield SrtTranslationChunk(
        entries=list(results),
        progress=f"翻譯完成 ({total}/{total})",
    )


def _translate_srt_batch(
    translator: Translator,
    results: list[SrtEntry],
    source_lang: str,
    target_lang: str,
    batch_size: int,
    glossary: list[tuple[str, str]] | None,
    glossary_mode: GlossaryMode,
) -> Iterator[SrtTranslationChunk]:
    non_empty = [(i, entry) for i, entry in enumerate(results) if entry.text]
    batches = [non_empty[j : j + batch_size] for j in range(0, len(non_empty), batch_size)]
    total_batches = len(batches)

    for batch_idx, batch in enumerate(batches):
        texts = []
        for _, entry in batch:
            text = entry.text
            if glossary_mode == "pre":
                text = apply_glossary_pre(text, glossary)
            texts.append(text)

        combined = BATCH_SEPARATOR.join(texts)

        try:
            last_chunk = ""
            for chunk in translator.translate(combined, source_lang, target_lang):
                last_chunk = chunk

            translated_parts = last_chunk.strip().split(BATCH_SEPARATOR)

            for part_idx, (entry_idx, entry) in enumerate(batch):
                if part_idx < len(translated_parts):
                    translated = translated_parts[part_idx].strip()
                    if glossary_mode == "post":
                        translated = apply_glossary_post(translated, glossary)
                    results[entry_idx] = SrtEntry(
                        index=entry.index,
                        start_time=entry.start_time,
                        end_time=entry.end_time,
                        text=translated,
                    )
        except Exception:
            logger.exception("批次 %d/%d 翻譯失敗", batch_idx + 1, total_batches)

        yield SrtTranslationChunk(
            entries=list(results),
            progress=f"翻譯中... (批次 {batch_idx + 1}/{total_batches})",
        )

    yield SrtTranslationChunk(
        entries=list(results),
        progress=f"翻譯完成 (批次 {total_batches}/{total_batches})",
    )


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def translate_srt_full_file(
    translator: Translator,
    entries: list[SrtEntry],
    source_lang: str,
    target_lang: str,
    glossary: list[tuple[str, str]] | None = None,
    glossary_mode: GlossaryMode = "pre",
) -> Iterator[SrtTranslationChunk]:
    srt_text = serialize_srt(entries)

    if _estimate_tokens(srt_text) > translator.max_tokens:
        raise ValueError("SRT 內容超過模型上下文長度，建議改用批次模式")

    if glossary_mode == "pre":
        srt_text = apply_glossary_pre(srt_text, glossary)

    yield SrtTranslationChunk(entries=list(entries), progress="翻譯中...")

    last_chunk = ""
    try:
        for chunk in translator.translate(srt_text, source_lang, target_lang):
            last_chunk = chunk
    except Exception:
        logger.exception("整檔翻譯失敗")
        yield SrtTranslationChunk(entries=list(entries), progress="整檔翻譯失敗，建議改用批次模式")
        return

    try:
        translated_entries = parse_srt(last_chunk.strip())
    except ValueError:
        logger.warning("整檔翻譯輸出格式無法解析")
        yield SrtTranslationChunk(
            entries=list(entries), progress="整檔翻譯失敗：模型輸出格式無法解析，建議改用批次模式"
        )
        return

    results = list(entries)
    for i, entry in enumerate(results):
        if i < len(translated_entries):
            translated_text = translated_entries[i].text
            if glossary_mode == "post":
                translated_text = apply_glossary_post(translated_text, glossary)
            results[i] = SrtEntry(
                index=entry.index,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=translated_text,
            )

    yield SrtTranslationChunk(entries=results, progress="翻譯完成")
