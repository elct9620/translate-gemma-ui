import csv
import io
import re
import unicodedata


def _is_cjk(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith("Lo")


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(c) for c in text)


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
        if _has_cjk(source):
            if source.lower() in text.lower():
                matched.append((source, target))
        else:
            pattern = r"\b" + re.escape(source) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                matched.append((source, target))
    return matched


def format_glossary_prompt(entries: list[tuple[str, str]]) -> str:
    """Format matched glossary entries as a prompt section."""
    if not entries:
        return ""
    lines = [f"{source} -> {target}" for source, target in entries]
    return "[Glossary]\n" + "\n".join(lines) + "\n"
