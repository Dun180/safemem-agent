from dataclasses import dataclass
from pathlib import Path
import numpy as np
from embedding_client import (
    embed_text,
    embed_texts,
)
from ollama_client import chat_with_model

@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int

@dataclass
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: list[float]

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float

RAG_SYSTEM_PROMPT = """
You are a retrieval-grounded assistant.

Answer the user's question using only the retrieved context provided to you.
Do not use outside knowledge.

If the retrieved context does not contain enough information to answer,
say exactly:

I don't have enough information in the retrieved documents.
""".strip()

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

def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    a_array = np.asarray(
        a,
        dtype=float,
    )

    b_array = np.asarray(
        b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a_array)
        * np.linalg.norm(b_array)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a_array, b_array)
        / denominator
    )

def build_index(
    chunks: list[DocumentChunk],
    embed_fn=embed_texts,
) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    embeddings = embed_fn(
        [chunk.text for chunk in chunks]
    )

    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
        )
        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        )
    ]

def retrieve(
    query: str,
    index: list[EmbeddedChunk],
    top_k: int = 3,
    embed_fn=embed_text,
) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero"
        )

    if not index:
        return []

    query_embedding = embed_fn(query)

    results = [
        RetrievalResult(
            chunk=item.chunk,
            score=cosine_similarity(
                query_embedding,
                item.embedding,
            ),
        )
        for item in index
    ]

    results.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return results[:top_k]

def answer_from_context(
    question: str,
    results: list[RetrievalResult],
    chat_fn=chat_with_model,
) -> str:

    if not results:
        return (
            "I don't have enough information "
            "in the retrieved documents."
        )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[Source: {result.chunk.source}, "
            f"chunk: {result.chunk.chunk_id}]\n"
            f"{result.chunk.text}"
        )

    context = "\n\n".join(
        context_parts
    )

    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n\n"
                f"{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    response = chat_fn(
        messages=messages,
        tools=[],
    )

    return response.message.content