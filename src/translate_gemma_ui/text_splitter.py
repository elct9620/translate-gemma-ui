import re
from collections.abc import Callable
from dataclasses import dataclass

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.。?？!！])\s*")


@dataclass(frozen=True)
class Window:
    text: str
    start_idx: int
    end_idx: int
    overlap_count: int


def split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)
    return [s for s in parts if s.strip()]


def create_windows(
    sentences: list[str],
    max_tokens: int,
    token_count_fn: Callable[[str], int],
    overlap: int = 2,
) -> list[Window]:
    if not sentences:
        return []

    windows: list[Window] = []
    start = 0

    while start < len(sentences):
        end = start
        current_text = sentences[start]

        while end + 1 < len(sentences):
            candidate = current_text + " " + sentences[end + 1]
            if token_count_fn(candidate) > max_tokens:
                break
            end += 1
            current_text = candidate

        overlap_count = min(overlap, start) if windows else 0
        window_start = start - overlap_count
        window_text = " ".join(sentences[window_start : end + 1])

        windows.append(Window(text=window_text, start_idx=window_start, end_idx=end, overlap_count=overlap_count))

        end += 1
        if end <= start:
            end = start + 1
        start = end

    return windows


def merge_translations(
    windows: list[Window],
    translations: list[str],
    original_sentences: list[str],
) -> str:
    if not windows:
        return ""

    total = original_sentences.__len__()
    result_sentences: list[str] = [""] * total

    for window, translation in zip(windows, translations):
        window_sentences = split_sentences(translation)
        expected_count = window.end_idx - window.start_idx + 1

        if len(window_sentences) == expected_count:
            for i, sent in enumerate(window_sentences):
                result_sentences[window.start_idx + i] = sent
        else:
            for i in range(window.start_idx, window.end_idx + 1):
                result_sentences[i] = ""
            if window_sentences:
                result_sentences[window.start_idx] = translation

    for i in range(total):
        if not result_sentences[i]:
            result_sentences[i] = original_sentences[i]

    return " ".join(result_sentences)
