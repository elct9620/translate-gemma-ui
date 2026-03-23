import re
from dataclasses import dataclass

TIMESTAMP_PATTERN = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")


@dataclass(frozen=True)
class SrtEntry:
    index: int
    start_time: str
    end_time: str
    text: str


def parse_srt(content: str) -> list[SrtEntry]:
    content = content.removeprefix("\ufeff")
    content = content.replace("\r\n", "\n")
    if not content.strip():
        raise ValueError("SRT 檔案內容為空")

    blocks = re.split(r"\n\n+", content.strip())
    entries: list[SrtEntry] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or not lines[0].strip():
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"無效的序號：{lines[0].strip()}") from None

        if len(lines) < 2:
            raise ValueError(f"區塊 {index} 缺少時間軸")

        match = TIMESTAMP_PATTERN.match(lines[1].strip())
        if not match:
            raise ValueError(f"區塊 {index} 的時間軸格式無效：{lines[1].strip()}")

        start_time = match.group(1)
        end_time = match.group(2)
        text = "\n".join(lines[2:]) if len(lines) > 2 else ""

        entries.append(SrtEntry(index=index, start_time=start_time, end_time=end_time, text=text))

    if not entries:
        raise ValueError("SRT 檔案內容為空")

    return entries


def serialize_srt(entries: list[SrtEntry]) -> str:
    if not entries:
        return ""

    blocks: list[str] = []
    for entry in entries:
        block = f"{entry.index}\n{entry.start_time} --> {entry.end_time}\n{entry.text}\n"
        blocks.append(block)

    return "\n".join(blocks)
