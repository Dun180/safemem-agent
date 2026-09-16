from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative "
            "and smaller than chunk_size"
        )

    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []

    for chunk_id, start in enumerate(
        range(0, len(text), step)
    ):
        value = text[
            start:start + chunk_size
        ]

        if not value:
            break

        chunks.append(
            DocumentChunk(
                text=value,
                source=source,
                chunk_id=chunk_id,
            )
        )

        if start + chunk_size >= len(text):
            break

    return chunks

def load_documents(
    directory: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    directory = Path(directory)

    chunks: list[DocumentChunk] = []

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(
            encoding="utf-8",
        )

        chunks.extend(
            chunk_text(
                text=text,
                source=path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks