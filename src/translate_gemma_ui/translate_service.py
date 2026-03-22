import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from translate_gemma_ui.glossary import match_glossary
from translate_gemma_ui.text_splitter import create_windows, merge_translations, split_sentences
from translate_gemma_ui.translator import TranslationContext, Translator

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


@dataclass(frozen=True)
class TranslationChunk:
    text: str
    progress: str


def translate_text(
    translator: Translator,
    text: str,
    source_lang: str,
    target_lang: str,
    token_count_fn: Callable[[str], int] = _estimate_tokens,
    glossary: list[tuple[str, str]] | None = None,
) -> Iterator[TranslationChunk]:
    sentences = split_sentences(text)
    windows = create_windows(sentences, translator.max_tokens, token_count_fn)

    if len(windows) <= 1:
        matched = match_glossary(text, glossary) if glossary else []
        context = TranslationContext(previous=[], following=[], glossary=matched) if matched else None
        for chunk in translator.translate(text, source_lang, target_lang, context=context):
            yield TranslationChunk(text=chunk, progress="")
        return

    translations: list[str] = []
    for i, window in enumerate(windows):
        progress = f"翻譯中... ({i + 1}/{len(windows)})"
        last_chunk = ""
        matched = match_glossary(window.text, glossary) if glossary else []
        context = TranslationContext(previous=[], following=[], glossary=matched) if matched else None
        try:
            for chunk in translator.translate(window.text, source_lang, target_lang, context=context):
                last_chunk = chunk
                partial = translations + [last_chunk]
                partial_result = merge_translations(windows[: len(partial)], partial, sentences)
                yield TranslationChunk(text=partial_result, progress=progress)
        except Exception:
            logger.exception("Segment %d/%d translation failed", i + 1, len(windows))
            last_chunk = window.text
            yield TranslationChunk(
                text=merge_translations(windows[: len(translations) + 1], translations + [last_chunk], sentences),
                progress=f"段落 {i + 1} 翻譯失敗，保留原文",
            )
        translations.append(last_chunk)

    final = merge_translations(windows, translations, sentences)
    yield TranslationChunk(text=final, progress=f"翻譯完成 ({len(windows)} 段)")
