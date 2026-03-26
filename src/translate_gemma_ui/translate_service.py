import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from translate_gemma_ui.glossary import GlossaryMode, apply_glossary_post, apply_glossary_pre
from translate_gemma_ui.text_splitter import create_windows, estimate_tokens, merge_translations, split_sentences
from translate_gemma_ui.translator import OutOfMemoryError, Translator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationChunk:
    text: str
    progress: str


def translate_text(
    translator: Translator,
    text: str,
    source_lang: str,
    target_lang: str,
    token_count_fn: Callable[[str], int] = estimate_tokens,
    glossary: list[tuple[str, str]] | None = None,
    glossary_mode: GlossaryMode = "pre",
) -> Iterator[TranslationChunk]:
    if glossary_mode == "pre":
        text = apply_glossary_pre(text, glossary)

    sentences = split_sentences(text)
    windows = create_windows(sentences, translator.max_tokens, token_count_fn)

    if len(windows) <= 1:
        for chunk in translator.translate(text, source_lang, target_lang):
            result = apply_glossary_post(chunk, glossary) if glossary_mode == "post" else chunk
            yield TranslationChunk(text=result, progress="")
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
                if glossary_mode == "post":
                    partial_result = apply_glossary_post(partial_result, glossary)
                yield TranslationChunk(text=partial_result, progress=progress)
        except OutOfMemoryError:
            raise
        except Exception:
            logger.exception("Segment %d/%d translation failed", i + 1, len(windows))
            last_chunk = window.text
            yield TranslationChunk(
                text=merge_translations(windows[: len(translations) + 1], translations + [last_chunk], sentences),
                progress=f"段落 {i + 1} 翻譯失敗，保留原文",
            )
        translations.append(last_chunk)

    final = merge_translations(windows, translations, sentences)
    if glossary_mode == "post":
        final = apply_glossary_post(final, glossary)
    yield TranslationChunk(text=final, progress=f"翻譯完成 ({len(windows)} 段)")
