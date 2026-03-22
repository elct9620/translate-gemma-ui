import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from translate_gemma_ui.glossary import apply_glossary_post, apply_glossary_pre
from translate_gemma_ui.srt_parser import SrtEntry
from translate_gemma_ui.translator import Translator

logger = logging.getLogger(__name__)

GlossaryMode = Literal["pre", "post"]


@dataclass(frozen=True)
class SrtTranslationChunk:
    entries: list[SrtEntry]
    progress: str


def translate_srt(
    translator: Translator,
    entries: list[SrtEntry],
    source_lang: str,
    target_lang: str,
    glossary: list[tuple[str, str]] | None = None,
    glossary_mode: GlossaryMode = "pre",
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
