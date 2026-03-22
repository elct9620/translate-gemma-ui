import logging
from collections.abc import Iterator
from dataclasses import dataclass

from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.translator import TranslationContext, Translator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SrtTranslationChunk:
    entries: list[SrtEntry]
    progress: str


def _build_context(entries: list[SrtEntry], target_idx: int, context_size: int) -> TranslationContext | None:
    if context_size <= 0:
        return None

    start = max(0, target_idx - context_size)
    end = min(len(entries), target_idx + context_size + 1)
    previous = [entries[j].text for j in range(start, target_idx) if entries[j].text]
    following = [entries[j].text for j in range(target_idx + 1, end) if entries[j].text]

    if not previous and not following:
        return None

    return TranslationContext(previous=previous, following=following)


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

        context = _build_context(entries, i, context_size)

        try:
            last_chunk = ""
            for chunk in translator.translate(entry.text, source_lang, target_lang, context=context):
                last_chunk = chunk

            results[i] = SrtEntry(
                index=entry.index,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=last_chunk.strip(),
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
