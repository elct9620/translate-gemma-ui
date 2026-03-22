import logging
from collections.abc import Iterator
from dataclasses import dataclass

from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.translator import Translator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SrtTranslationChunk:
    entries: list[SrtEntry]
    progress: str


def build_context_text(entries: list[SrtEntry], target_idx: int, context_size: int) -> tuple[str, int]:
    start = max(0, target_idx - context_size)
    end = min(len(entries), target_idx + context_size + 1)
    context_entries = entries[start:end]
    lines = [e.text for e in context_entries]
    line_idx = target_idx - start
    return "\n".join(lines), line_idx


def extract_target_translation(translated: str, target_line_idx: int, expected_lines: int) -> str:
    lines = translated.split("\n")
    if len(lines) == expected_lines:
        return lines[target_line_idx]
    return translated


def translate_srt(
    translator: Translator,
    entries: list[SrtEntry],
    source_lang: str,
    target_lang: str,
    context_size: int = 3,
) -> Iterator[SrtTranslationChunk]:
    results = list(entries)
    total = len(entries)

    for i, entry in enumerate(entries):
        if not entry.text:
            yield SrtTranslationChunk(
                entries=list(results),
                progress=f"翻譯中... ({i + 1}/{total})",
            )
            continue

        context_text, line_idx = build_context_text(entries, i, context_size)
        expected_lines = min(i + context_size + 1, total) - max(0, i - context_size)

        try:
            last_chunk = ""
            for chunk in translator.translate(context_text, source_lang, target_lang):
                last_chunk = chunk

            translated = extract_target_translation(last_chunk, line_idx, expected_lines)
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
