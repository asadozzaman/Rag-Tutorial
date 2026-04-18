"""
Lightweight chunking helpers for Day 2.

These utilities avoid depending on optional splitter packages while still
showing the main chunking ideas: fixed-size, recursive, and markdown-aware.
"""

from __future__ import annotations


def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step

    return chunks


def _recursive_units(text: str, separators: list[str]) -> list[str]:
    if len(text) <= 1:
        return [text]
    if not separators:
        return list(text)

    separator = separators[0]
    if separator and separator in text:
        pieces = text.split(separator)
        units = []
        for piece in pieces:
            piece = piece.strip()
            if piece:
                units.append(piece)
        if units:
            return units

    return _recursive_units(text, separators[1:])


def recursive_text_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    separators = separators or ["\n\n", "\n", ". ", " ", ""]
    units = _recursive_units(text, separators)

    if not units:
        return []

    chunks = []
    current = []
    current_len = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(fixed_size_chunks(unit, chunk_size, chunk_overlap))
            continue

        projected = current_len + len(unit) + (1 if current else 0)
        if projected <= chunk_size:
            current.append(unit)
            current_len = projected
            continue

        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

        if chunk_overlap > 0 and chunk:
            overlap_text = chunk[-chunk_overlap:].strip()
            current = [overlap_text, unit] if overlap_text else [unit]
            current_len = len(" ".join(current))
        else:
            current = [unit]
            current_len = len(unit)

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def markdown_sections(markdown_text: str) -> list[dict]:
    sections = []
    headers = {"h1": None, "h2": None, "h3": None}
    buffer = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            metadata = {key: value for key, value in headers.items() if value}
            sections.append({"content": content, "metadata": metadata})

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("### "):
            flush()
            buffer.clear()
            headers["h3"] = stripped[4:].strip()
            continue
        if stripped.startswith("## "):
            flush()
            buffer.clear()
            headers["h2"] = stripped[3:].strip()
            headers["h3"] = None
            continue
        if stripped.startswith("# "):
            flush()
            buffer.clear()
            headers["h1"] = stripped[2:].strip()
            headers["h2"] = None
            headers["h3"] = None
            continue

        buffer.append(line)

    flush()
    return sections


def markdown_aware_chunks(
    markdown_text: str,
    chunk_size: int = 450,
    chunk_overlap: int = 60,
) -> list[dict]:
    chunks = []
    for section in markdown_sections(markdown_text):
        text_chunks = recursive_text_chunks(
            section["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk in text_chunks:
            chunks.append({"content": chunk, "metadata": dict(section["metadata"])})
    return chunks
