import csv
import io
import re
import unicodedata


def _is_non_latin(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith("Lo")


def _has_non_latin(text: str) -> bool:
    return any(_is_non_latin(c) for c in text)


def parse_glossary(content: str) -> list[tuple[str, str]]:
    """Parse CSV glossary content into (source, target) pairs."""
    content = content.lstrip("\ufeff")
    entries: list[tuple[str, str]] = []

    reader = csv.reader(io.StringIO(content))
    for line_num, row in enumerate(reader, start=1):
        if not row or all(cell.strip() == "" for cell in row):
            continue
        if len(row) != 2:
            raise ValueError(f"第 {line_num} 行欄位數錯誤：預期 2 欄，實際 {len(row)} 欄")
        source, target = row[0].strip(), row[1].strip()
        if not source or not target:
            raise ValueError(f"第 {line_num} 行詞彙不得為空白")
        entries.append((source, target))

    return entries


def match_glossary(text: str, glossary: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return glossary entries whose source term appears in text (case-insensitive, whole-word)."""
    matched: list[tuple[str, str]] = []
    for source, target in glossary:
        if _has_non_latin(source):
            if source.lower() in text.lower():
                matched.append((source, target))
        else:
            pattern = r"\b" + re.escape(source) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                matched.append((source, target))
    return matched


def _replace_term(text: str, source: str, target: str) -> str:
    """Replace a single glossary term in text, preserving word boundaries for Latin terms."""
    if _has_non_latin(source):
        return re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    pattern = r"\b" + re.escape(source) + r"\b"
    return re.sub(pattern, target, text, flags=re.IGNORECASE)


def apply_glossary_pre(text: str, glossary: list[tuple[str, str]] | None) -> str:
    """Pre-processing: replace source terms with target terms before translation."""
    if not glossary:
        return text
    matched = match_glossary(text, glossary)
    for source, target in matched:
        text = _replace_term(text, source, target)
    return text


def apply_glossary_post(text: str, glossary: list[tuple[str, str]] | None) -> str:
    """Post-processing: replace source terms remaining in translated text with target terms.

    Uses substring matching (no word boundary) since translated text may mix scripts.
    """
    if not glossary:
        return text
    for source, target in glossary:
        pattern = re.compile(re.escape(source), re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(target, text)
    return text
